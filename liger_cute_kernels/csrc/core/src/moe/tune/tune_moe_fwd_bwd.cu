// Combined autotuner for the fused MoE forward + backward kernels.
//
// For each shape (T, D, I, E, K), tunes fwd and bwd INDEPENDENTLY within
// each TileM bucket (64, 128). The bwd template inherits the fwd's sort
// layout (tile_expert_ids granularity / x_sorted slot stride /
// expert_offsets alignment), so the two kernels MUST agree on TileM —
// but once TileM is fixed they are separate sequential kernel launches
// whose timings don't depend on the other direction's config. So instead
// of the O(F·B) fwd×bwd cross product, we run O(F+B) per bucket:
//   - FWD sweep: vary the fwd config, hold bwd at a fixed default; keep
//     the fwd config with the lowest fwd_ms.
//   - BWD sweep: vary the bwd config, hold fwd at a fixed default; keep
//     the bwd config with the lowest bwd_ms.
// Each measurement runs a full fwd→bwd sequence (the bwd needs the fwd's
// sort outputs) but reads only the half being tuned.
//
// The bucket's combined = best_fwd_ms + best_bwd_ms. The winning TileM is
// the bucket with the lowest combined; we dump its (fwd template, bwd
// template, shared TileM) into moe_fwd_bwd_tuning_configs.cuh so the
// runtime "auto" dispatcher can pick both templates together by shape.
//
// Run with: srun --mpi=pmi2 --ntasks=N ./tune_moe_fwd_bwd
// Output:   ../src/liger_comm_kernels/moe/moe_fwd_bwd_tuning_configs.cuh
//           (override with LIGER_MOE_FWDBWD_TUNED_OUTPUT=/path/to/*_sm90.cuh
//            or /path/to/*_sm100.cuh matching the detected GPU)

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <tuple>

#include <cutlass/numeric_types.h>

#include "moe_fwd_bwd_tune_configs.hpp"   // X-macro config menus (../, on the include path)
#include "moe_dispatch_configs_sm90.cuh"
#include "moe_dispatch_configs_sm100.cuh"
#include "moe_utils.cuh"
#include "moe_launch.h"                    // liger::MoeFwdArgs / MoeBwdArgs (../)
#include "liger_cute/nvshmem.h"            // flat ABI: init_pmi / finalize / pool clear

// 0 = build every dispatch family (developer fallback). CMake sets this to 90
// for Hopper builds and 100 for Blackwell builds.
#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

// ── Minimal NVSHMEM host-API surface ────────────────────────────────
//
// Avoid <nvshmem.h> in THIS TU: pulling in inline device helpers forces RDC on
// a host-only TU that takes function pointers to RDC kernels, tripping nvlink's
// kernel-symbol double-link bug. nvshmem_team_t is int32 (== int); the raw
// team-query symbols resolve against libnvshmem at link time.
using nvshmem_team_t = int;
static constexpr nvshmem_team_t NVSHMEM_TEAM_WORLD = 0;
extern "C" {
int   nvshmem_team_my_pe(nvshmem_team_t team);
int   nvshmem_team_n_pes(nvshmem_team_t team);
int   nvshmem_team_sync(nvshmem_team_t team);
}

static inline void tuner_team_sync(nvshmem_team_t team, int n_pes) {
	if (n_pes > 1) nvshmem_team_sync(team);
}

// ── Forward declarations of the internal launchers (defined in moe.cu /
// moe_bwd.cu, compiled into this binary — the core .so hides them, so the
// tuner builds the kernel sources itself with default visibility). The
// templated-launcher signatures must match moe.cu / moe_bwd.cu exactly so the
// registry function pointers below bind to the library's instantiations. ──
namespace liger {

void moe_configure_symmetric(
	int max_tokens, int hidden_dim, int max_num_experts,
	int max_top_k, int num_pes, int num_hosts, int gpus_per_host);

void moe_pop_fwd();

template <
	typename Element_,
	int TileN1, int TileK1, int Stages1,
	int TileN2, int TileK2, int Stages2,
	int ZBufferSlots, int CommNumStages,
	int EpiChunkN1 = 64, int EpiChunkN2 = 64,
	int TileM = 128, int GemmTileM = TileM, int Compute = 90>
void moe_fused_fwd_bf16(const MoeFwdArgs& a, int static_nsplit);

template <
	int NSplit, int NSplit2,
	int TileN1, int TileK1, int Stages1,
	int TileM3, int TileN3, int TileK3, int Stages3,
	int EpiChunkN1, int EpiChunkN25, int EpiChunkN34,
	int CommNumStages,
	int TileM, int GemmTileM = TileM, int Compute = 90>
void moe_bwd_fwd_bf16_tuned(const MoeBwdArgs& a);

} // namespace liger

// ── Fwd registry — generated from LIGER_MOE_TUNE_CONFIGS ─────────────

using MoeFwdFn = void (*)(const liger::MoeFwdArgs&, int static_nsplit);

struct TunerEntryFwd {
	const char* name;
	int Compute;
	int NSplit;
	int TileN1, TileK1, Stages1, EpiChunkN1;
	int TileN2, TileK2, Stages2, EpiChunkN2;
	int ZBufferSlots, CommNumStages;
	int TileM;
	MoeFwdFn fn;
};

#define LIGER_MOE_FWD_REGISTRY_ENTRY_C(Compute, NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM) \
	{                                                                                                \
		"NS" #NSplit "_TM" #TM "_TN1-" #TN1 "/" #TK1 "/" #S1 "/EC" #EC1                              \
		"_TN2-" #TN2 "/" #TK2 "/" #S2 "/EC" #EC2                                                     \
		"_ZB" #ZBuf "_CS" #CStages,                                                                  \
		Compute, NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM,                    \
		&liger::moe_fused_fwd_bf16<                                                                  \
			cutlass::bfloat16_t, TN1, TK1, S1, TN2, TK2, S2, ZBuf, CStages, EC1, EC2, TM, TM, Compute> \
	},
#define LIGER_MOE_FWD_REGISTRY_ENTRY_SM90(NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM) \
	LIGER_MOE_FWD_REGISTRY_ENTRY_C(90, NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM)
#define LIGER_MOE_FWD_REGISTRY_ENTRY_SM100(NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM) \
	LIGER_MOE_FWD_REGISTRY_ENTRY_C(100, NSplit, TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZBuf, CStages, TM)

static const TunerEntryFwd kRegistryFwd[] = {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
	LIGER_MOE_TUNE_CONFIGS(LIGER_MOE_FWD_REGISTRY_ENTRY_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
	LIGER_MOE_FWD_TUNE_CONFIGS_TM128(LIGER_MOE_FWD_REGISTRY_ENTRY_SM100)
#endif
};
#undef LIGER_MOE_FWD_REGISTRY_ENTRY_SM100
#undef LIGER_MOE_FWD_REGISTRY_ENTRY_SM90
#undef LIGER_MOE_FWD_REGISTRY_ENTRY_C

static constexpr int kNumFwdConfigs =
	sizeof(kRegistryFwd) / sizeof(kRegistryFwd[0]);

// ── Bwd registry — generated from LIGER_MOE_BWD_TUNE_CONFIGS ─────────

using MoeBwdFwdFn = void (*)(const liger::MoeBwdArgs&);

struct TunerEntryBwd {
	const char* name;
	int Compute;
	int NSplit, NSplit2;
	int TileN1, TileK1, Stages1;
	int TileM3, TileN3, TileK3, Stages3;
	int EpiChunkN1, EpiChunkN25, EpiChunkN34;
	int CommNumStages;
	int TileM;
	MoeBwdFwdFn fn;
};

#define LIGER_MOE_BWD_REGISTRY_ENTRY_C(Compute, NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	{                                                                                                \
		"NS" #NS "_NS2-" #NS2                                                                        \
		"_TN1-" #TN1 "/" #TK1 "/" #S1                                                                \
		"_TM3-" #TM3 "_TN3-" #TN3 "/" #TK3 "/" #S3                                                   \
		"_EN1-" #EN1 "_EN25-" #EN25 "_EN34-" #EN34 "_CS" #CS "_TM" #TM,                              \
		Compute, NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM,                  \
		&liger::moe_bwd_fwd_bf16_tuned<                                                              \
			NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM, TM, Compute>          \
	},
#define LIGER_MOE_BWD_REGISTRY_ENTRY_SM90(NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	LIGER_MOE_BWD_REGISTRY_ENTRY_C(90, NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)
#define LIGER_MOE_BWD_REGISTRY_ENTRY_SM100(NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM) \
	LIGER_MOE_BWD_REGISTRY_ENTRY_C(100, NS, NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)

static const TunerEntryBwd kRegistryBwd[] = {
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 90
	LIGER_MOE_BWD_TUNE_CONFIGS(LIGER_MOE_BWD_REGISTRY_ENTRY_SM90)
#endif
#if LIGER_CUTE_DISPATCH_COMPUTE == 0 || LIGER_CUTE_DISPATCH_COMPUTE == 100
	LIGER_MOE_BWD_TUNE_CONFIGS_SM100(LIGER_MOE_BWD_REGISTRY_ENTRY_SM100)
#endif
};
#undef LIGER_MOE_BWD_REGISTRY_ENTRY_SM100
#undef LIGER_MOE_BWD_REGISTRY_ENTRY_SM90
#undef LIGER_MOE_BWD_REGISTRY_ENTRY_C

static constexpr int kNumBwdConfigs =
	sizeof(kRegistryBwd) / sizeof(kRegistryBwd[0]);

// ── Shape validity ──────────────────────────────────────────────────

// Fwd validity (mirrors tune_moe.cu / tuned_config_valid in moe.cu).
static bool fwd_shape_valid(int D, int I, const TunerEntryFwd& e) {
	if (e.TileM != 64 && e.TileM != 128) return false;
	if (D % e.TileK1 != 0) return false;
	if (I % e.TileN1 != 0) return false;
	if (I % e.TileK2 != 0) return false;
	if (D % e.TileN2 != 0) return false;
	if (D % 8 != 0 || I % 8 != 0) return false;
	const int wg_tile_n1 = (e.TileM == 128) ? e.TileN1 : e.TileN1 / 2;
	const int wg_tile_n2 = (e.TileM == 128) ? e.TileN2 : e.TileN2 / 2;
	if (wg_tile_n1 <= 0 || wg_tile_n1 % e.EpiChunkN1 != 0) return false;
	if (wg_tile_n2 <= 0 || wg_tile_n2 % e.EpiChunkN2 != 0) return false;
	if (e.TileM == 64 && (e.TileN1 % 2 != 0 || e.TileN2 % 2 != 0)) return false;

	// SMEM cap mirrors tune_moe.cu.
	int mlp1_bytes = 2 * (e.TileM * e.TileK1 * e.Stages1
	                    + 2       * e.TileN1 * e.TileK1 * e.Stages1
	                    + 2       * 64       * e.EpiChunkN1);
	int mlp2_bytes = 2 * (e.TileM * e.TileK2 * e.Stages2
	                    +           e.TileN2 * e.TileK2 * e.Stages2
	                    + 2       * 64       * e.EpiChunkN2);
	int smem = mlp1_bytes > mlp2_bytes ? mlp1_bytes : mlp2_bytes;
	constexpr int kSmemBudget = 224 * 1024;
	if (smem > kSmemBudget) return false;
	return true;
}

// Bwd validity (mirrors tune_moe_bwd.cu / tuned_config_valid_bwd).
static bool bwd_shape_valid(int D, int I, const TunerEntryBwd& e) {
	if (D % e.TileK1 != 0)        return false;
	if (D % (2 * e.TileN1) != 0)  return false;
	if (D % e.TileM3 != 0)        return false;
	if (I % (2 * e.TileN1) != 0)  return false;
	if (I % e.TileN3 != 0)        return false;
	if (I % e.TileK1 != 0)        return false;
	if (I % e.TileK3 != 0)        return false;
	if (D % 8 != 0 || I % 8 != 0) return false;
	int num_n_tiles_1 = I / e.TileN1;
	if (num_n_tiles_1 < e.NSplit) return false;

	// Phase-2 cooperative layout (#102) supports only (256,128)/(128,256).
	if (!((e.TileM3 == 256 && e.TileN3 == 128) ||
	      (e.TileM3 == 128 && e.TileN3 == 256))) return false;

	// SMEM-fit guard (mirrors the per-phase estimate in
	// moe_fwd_bwd_tune_configs.hpp). The MoeBwdSmem union takes the max across
	// phases; budget = 228 KiB minus ~4 KiB for the 5 out-of-union pipe
	// storages. bf16 = 2 bytes. Pinned: TileM1=TileM (phase-1 M), TileK1
	// shared by 1a/1b'/1d, EpiChunkN1 for mlp1_act.
	{
		const int TM = e.TileM, TN1 = e.TileN1, TK1 = e.TileK1, S1 = e.Stages1;
		const int TM3 = e.TileM3, TN3 = e.TileN3, TK3 = e.TileK3, S3 = e.Stages3;
		const int EN1 = e.EpiChunkN1, EN25 = e.EpiChunkN25, EN34 = e.EpiChunkN34;
		int mlp1 = 2 * ((TM + 2 * TN1) * TK1 * S1) + 6 * 128 * EN1;
		int mlp25 = 2 * ((128 + 256) * TK1 * S1) + 2 * 128 * EN25;   // mlp2_t / mlp5
		int mlp3 = 2 * ((TM3 + 2 * TN3) * TK3 * S3) + 4 * TM3 * EN34;
		int mlp4 = 2 * ((2 * TN3 + TM3) * TK3 * S3) + 4 * TN3 * EN34;
		int smem = mlp1;
		smem = mlp25 > smem ? mlp25 : smem;
		smem = mlp3  > smem ? mlp3  : smem;
		smem = mlp4  > smem ? mlp4  : smem;
		if (smem > (228 - 4) * 1024) return false;
	}
	return true;
}

// Memory estimate — keep the bwd model (it's strictly more demanding
// than fwd-only because of the dA/dB/dC grad outputs).
static size_t estimate_memory_bytes(int T, int D, int I, int E, int K, int epp) {
	constexpr size_t bf16 = 2;
	size_t weight = (size_t)epp * I * D * bf16;
	size_t weight_like = 6 * weight;
	size_t total_slots = (size_t)T * K;
	size_t bwd_pool = 4 * total_slots * I * bf16
	                + 2 * total_slots * D * bf16
	                + (1ULL << 30);
	size_t fwd_pool = 2 * total_slots * D * bf16
	                + (1ULL << 30);
	size_t per_shape = (size_t)T * D * bf16 * 4;
	size_t base = weight_like + bwd_pool + fwd_pool + per_shape;
	return (size_t)(base * 1.3);
}

// ── Pair-level run: one fwd→bwd sequence, three timings ──────────────

struct PairResult {
	float fwd_ms;       // < 0 on failure
	float bwd_ms;       // < 0 on failure
	float combined_ms;  // fwd_ms + bwd_ms; < 0 on failure
	bool  oom;
};

static bool check_after(const char* fwd_name, const char* bwd_name,
                        const char* step, int pe) {
	cudaDeviceSynchronize();
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		if (pe == 0)
			fprintf(stderr, "  [FWD=%s × BWD=%s @ %s] CUDA error: %s\n",
				fwd_name, bwd_name, step, cudaGetErrorString(err));
		return true;
	}
	return false;
}

// Run one fwd→bwd sequence, recording cudaEvents around each kernel.
// fwd_start_ev / fwd_stop_ev / bwd_stop_ev are caller-owned, optional.
// When all three are non-null they bracket: [fwd_start_ev .. fwd_stop_ev]
// is the fwd; [fwd_stop_ev .. bwd_stop_ev] is the bwd. Always pops the
// symm-stack entries the fwd pushed so the next iteration starts clean.
static bool run_pair_once(
		const TunerEntryFwd& fwd_e, const TunerEntryBwd& bwd_e,
		const torch::Tensor& X, const torch::Tensor& dY,
		const torch::Tensor& expert_indices,
		const torch::Tensor& expert_weights,
		const torch::Tensor& all_B, const torch::Tensor& all_C,
		const torch::Tensor& all_A,
		int E, int K, nvshmem_team_t team, int pe, int n_pes,
		bool* saw_oom,
		const char* step,
		cudaEvent_t* fwd_start_ev = nullptr,
		cudaEvent_t* fwd_stop_ev  = nullptr,
		cudaEvent_t* bwd_stop_ev  = nullptr) {

	// Dims (the templated launchers take raw pointers + dims via the arg structs).
	const int T    = (int)X.size(0);
	const int D    = (int)X.size(1);
	const int Imid = (int)all_B.size(1);
	const int epp  = (int)all_B.size(0);
	const int max_total_slots = T * K + E * 128;   // 128 = fixed comm TileM
	const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	int device = 0; cudaGetDevice(&device);
	auto i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

	// Symmetric buffers the fwd launcher allocates + threads into the bwd.
	void* x_sorted = nullptr;
	void* y_buf    = nullptr;
	void* all_off  = nullptr;
	// Caller-owned sort outputs the bwd consumes (kept alive across both calls).
	torch::Tensor tok_slots, tile_ids;

	if (fwd_start_ev) cudaEventRecord(*fwd_start_ev);
	try {
		torch::Tensor Y = torch::empty({T, D}, bf16);
		tok_slots = torch::empty({max_total_slots}, i32);
		tile_ids  = torch::empty({max_total_slots / 128}, i32);
		liger::MoeFwdArgs fa{};
		fa.X = X.data_ptr();
		fa.expert_indices = expert_indices.data_ptr<int>();
		fa.expert_weights = expert_weights.data_ptr();
		fa.all_B = all_B.data_ptr(); fa.all_C = all_C.data_ptr(); fa.all_A = all_A.data_ptr();
		fa.num_tokens = T; fa.hidden_dim = D; fa.intermediate_dim = Imid; fa.experts_per_pe = epp;
		fa.num_experts = E; fa.top_k = K; fa.team = team; fa.stream = stream; fa.device = device;
		fa.Y = Y.data_ptr();
		fa.token_expert_slots = tok_slots.data_ptr<int>();
		fa.tile_expert_ids    = tile_ids.data_ptr<int>();
		fa.x_sorted_out = &x_sorted; fa.y_buf_out = &y_buf; fa.all_expert_offsets_out = &all_off;
		fwd_e.fn(fa, fwd_e.NSplit);
	} catch (const std::exception& ex) {
		const char* msg = ex.what();
		if (msg && (std::strstr(msg, "out of memory") ||
		            std::strstr(msg, "CUDA out of memory") ||
		            std::strstr(msg, "OutOfMemoryError"))) {
			*saw_oom = true;
		}
		if (pe == 0) {
			fprintf(stderr, "  [FWD=%s × BWD=%s @ %s FWD] exception: %s\n",
				fwd_e.name, bwd_e.name, step, msg ? msg : "(no message)");
			fflush(stderr);
		}
		(void)cudaGetLastError();
		return true;
	}
	// Cross-PE fwd→bwd handoff barrier. The BWD's remote gets read x_sorted /
	// expert_offsets that the FWD wrote into PEER-PE symmetric memory; the
	// FWD's tail NVSHMEM traffic + its end-of-kernel team sync must be globally
	// visible before the BWD launches, or the BWD's comm reads stale data and
	// deadlocks. A device sync drains the FWD (incl. its device-side team_sync)
	// on every PE; team_sync makes the cross-PE completion collective. Recording
	// fwd_stop AFTER the drain keeps timing attribution correct: fwd cost
	// includes its comm tail, and bwd = bwd_stop - fwd_stop measures BWD alone.
	cudaDeviceSynchronize();
	tuner_team_sync(team, n_pes);
	if (fwd_stop_ev) cudaEventRecord(*fwd_stop_ev);

	bool bwd_failed = false;
	try {
		torch::Tensor dX = torch::empty({T, D}, bf16);
		torch::Tensor dB = torch::empty({epp, Imid, D}, bf16);
		torch::Tensor dC = torch::empty({epp, Imid, D}, bf16);
		torch::Tensor dA = torch::empty({epp, D, Imid}, bf16);
		torch::Tensor dW = torch::empty({T, K}, bf16);
		liger::MoeBwdArgs ba{};
		ba.dY = dY.data_ptr();
		ba.Y_fwd = y_buf;            // fwd's expert-sorted output buffer
		ba.x_sorted = x_sorted;
		ba.token_expert_slots = tok_slots.data_ptr<int>();
		ba.tile_expert_ids    = tile_ids.data_ptr<int>();
		ba.expert_offsets     = static_cast<int*>(all_off);
		ba.expert_indices = expert_indices.data_ptr<int>();
		ba.expert_weights = expert_weights.data_ptr();
		ba.all_B = all_B.data_ptr(); ba.all_C = all_C.data_ptr(); ba.all_A = all_A.data_ptr();
		ba.num_tokens = T; ba.hidden_dim = D; ba.intermediate_dim = Imid; ba.experts_per_pe = epp;
		ba.num_experts = E; ba.top_k = K; ba.team = team; ba.stream = stream; ba.device = device;
		ba.dX = dX.data_ptr(); ba.dB = dB.data_ptr(); ba.dC = dC.data_ptr();
		ba.dA = dA.data_ptr(); ba.dW = dW.data_ptr();
		bwd_e.fn(ba);
	} catch (const std::exception& ex) {
		const char* msg = ex.what();
		if (msg && (std::strstr(msg, "out of memory") ||
		            std::strstr(msg, "CUDA out of memory") ||
		            std::strstr(msg, "OutOfMemoryError"))) {
			*saw_oom = true;
		}
		if (pe == 0) {
			fprintf(stderr, "  [FWD=%s × BWD=%s @ %s BWD] exception: %s\n",
				fwd_e.name, bwd_e.name, step, msg ? msg : "(no message)");
			fflush(stderr);
		}
		(void)cudaGetLastError();
		bwd_failed = true;
	}
	if (bwd_stop_ev) cudaEventRecord(*bwd_stop_ev);

	// Drain fwd's symm-stack entries so the next sequence starts clean. MUST
	// come after bwd has finished reading them (x_sorted / y_buf / all_off are
	// stack-owned symmetric memory).
	liger::moe_pop_fwd();
	return bwd_failed;
}

static PairResult run_pair(
		const TunerEntryFwd& fwd_e, const TunerEntryBwd& bwd_e,
		const torch::Tensor& X, const torch::Tensor& dY,
		const torch::Tensor& expert_indices,
		const torch::Tensor& expert_weights,
		const torch::Tensor& all_B, const torch::Tensor& all_C,
		const torch::Tensor& all_A,
		int E, int K, nvshmem_team_t team, int pe, int n_pes) {

	if (pe == 0 && getenv("MOE_FWDBWD_TUNE_VERBOSE")) {
		fprintf(stderr, "  trying FWD=%s × BWD=%s\n", fwd_e.name, bwd_e.name);
		fflush(stderr);
	}

	// Settle outstanding async traffic from the previous pair before
	// starting (mirrors tune_moe_bwd.cu's per-pair sync).
	cudaDeviceSynchronize();
	tuner_team_sync(team, n_pes);

	bool saw_oom = false;

	// Sanity launch — surfaces template-specific runtime errors before
	// we commit to the timing loop.
	if (run_pair_once(fwd_e, bwd_e, X, dY, expert_indices, expert_weights,
	                  all_B, all_C, all_A, E, K, team, pe, n_pes, &saw_oom, "sanity"))
		return {-1.0f, -1.0f, -1.0f, saw_oom};
	if (check_after(fwd_e.name, bwd_e.name, "sanity", pe))
		return {-1.0f, -1.0f, -1.0f, false};
	tuner_team_sync(team, n_pes);

	constexpr int kWarmup = 3;
	constexpr int kIters  = 5;
	for (int i = 0; i < kWarmup; ++i) {
		if (run_pair_once(fwd_e, bwd_e, X, dY, expert_indices, expert_weights,
		                  all_B, all_C, all_A, E, K, team, pe, n_pes, &saw_oom, "warmup"))
			return {-1.0f, -1.0f, -1.0f, saw_oom};
		if (check_after(fwd_e.name, bwd_e.name, "warmup", pe))
			return {-1.0f, -1.0f, -1.0f, false};
		// Per-iter sync (same reason as tune_moe_bwd.cu — back-to-back
		// async launches race torch's allocator activity against the
		// kernel's in-flight NVSHMEM puts).
		cudaDeviceSynchronize();
		tuner_team_sync(team, n_pes);
	}

	cudaEvent_t fwd_start, fwd_stop, bwd_stop;
	cudaEventCreate(&fwd_start);
	cudaEventCreate(&fwd_stop);
	cudaEventCreate(&bwd_stop);
	float total_fwd_ms = 0, total_bwd_ms = 0;
	for (int i = 0; i < kIters; ++i) {
		cudaDeviceSynchronize();
		tuner_team_sync(team, n_pes);
		if (run_pair_once(fwd_e, bwd_e, X, dY, expert_indices, expert_weights,
		                  all_B, all_C, all_A, E, K, team, pe, n_pes, &saw_oom, "timing",
		                  &fwd_start, &fwd_stop, &bwd_stop)) {
			cudaEventDestroy(fwd_start);
			cudaEventDestroy(fwd_stop);
			cudaEventDestroy(bwd_stop);
			return {-1.0f, -1.0f, -1.0f, saw_oom};
		}
		cudaEventSynchronize(bwd_stop);
		if (check_after(fwd_e.name, bwd_e.name, "timing", pe)) {
			cudaEventDestroy(fwd_start);
			cudaEventDestroy(fwd_stop);
			cudaEventDestroy(bwd_stop);
			return {-1.0f, -1.0f, -1.0f, false};
		}
		float iter_fwd_ms = 0, iter_bwd_ms = 0;
		cudaEventElapsedTime(&iter_fwd_ms, fwd_start, fwd_stop);
		cudaEventElapsedTime(&iter_bwd_ms, fwd_stop, bwd_stop);
		total_fwd_ms += iter_fwd_ms;
		total_bwd_ms += iter_bwd_ms;
	}
	cudaEventDestroy(fwd_start);
	cudaEventDestroy(fwd_stop);
	cudaEventDestroy(bwd_stop);

	float fwd_ms = total_fwd_ms / kIters;
	float bwd_ms = total_bwd_ms / kIters;
	return {fwd_ms, bwd_ms, fwd_ms + bwd_ms, false};
}

// ── Collector for winning (fwd, bwd) pair per shape ──────────────────

struct TunedRow {
	int Compute;
	int TK, TKE, D, I;  // TK = T * top_k (M-axis); TKE = TK / E_local where
	                    // E_local = E / n_pes (per-LOCAL-expert avg K-range
	                    // — each PE only walks its own experts in mlp3/mlp4
	                    // bwd, so the local count is what drives walk-
	                    // scheduler balance and L2 reuse).
	int fwd_ci, bwd_ci;
	float fwd_ms, bwd_ms, combined_ms;
};
static std::vector<TunedRow> g_tuned_rows;

// Tuning-time PE count and compute capability, set in main(). Together they
// select the generated subtable target: single-vs-multi wrapper, then sm90-vs-
// sm100 subtable. The umbrella moe_fwd_bwd_tuning_configs.cuh includes all
// world-size/compute subtables.
static int g_tuner_n_pes = 0;
static int g_tuner_compute = 0;

static const char* tuned_output_compute_suffix() {
	return (g_tuner_compute == 100) ? "_sm100.cuh" : "_sm90.cuh";
}

static bool ends_with(const char* s, const char* suffix) {
	if (!s || !suffix) return false;
	const size_t n = std::strlen(s);
	const size_t m = std::strlen(suffix);
	return n >= m && std::strcmp(s + n - m, suffix) == 0;
}

static void dump_tuned_configs() {
	const bool single = (g_tuner_n_pes <= 1);
	const char* cls = single ? "Single" : "Multi";  // array/count name suffix
	const char* compute_suffix = (g_tuner_compute == 100) ? "Sm100" : "Sm90";
	const char* default_path = single
		? ((g_tuner_compute == 100)
			? "moe_fwd_bwd_tuning_configs_single_sm100.cuh"
			: "moe_fwd_bwd_tuning_configs_single_sm90.cuh")
		: ((g_tuner_compute == 100)
			? "moe_fwd_bwd_tuning_configs_multi_sm100.cuh"
			: "moe_fwd_bwd_tuning_configs_multi_sm90.cuh");
	const char* env = getenv("LIGER_MOE_FWDBWD_TUNED_OUTPUT");
	const char* out_path = env ? env : default_path;
	const char* required_suffix = tuned_output_compute_suffix();
	if (!ends_with(out_path, required_suffix)) {
		fprintf(stderr,
		        "Refusing to write tuned config for sm_%d to '%s'. Output path "
		        "must end with '%s' so each compute capability only updates its "
		        "matching subtable header.\n",
		        g_tuner_compute, out_path, required_suffix);
		std::exit(1);
	}
	std::ofstream f(out_path);
	if (!f) {
		fprintf(stderr, "Failed to open tuned config header for write: %s\n", out_path);
		return;
	}
	const std::string arr   = std::string("kTunedConfigs") + cls + compute_suffix;
	const std::string count = std::string("kNumTunedConfigs") + cls + compute_suffix;
	f << "#pragma once\n\n"
	  << "// Auto-generated by benchmarks/tune_moe_fwd_bwd — do not edit by hand.\n"
	  << "// World-size CLASS: " << (single ? "single-GPU (n_pes == 1)" : "multi-GPU (n_pes > 1)")
	  << "; compute capability: sm_" << g_tuner_compute
	  << "; tuned at n_pes=" << g_tuner_n_pes << ".\n"
	  << "// Key (TK, TKE, D, I): TK = T*top_k (total routed tokens),\n"
	  << "// TKE = TK / E_local where E_local = E / n_pes (per-LOCAL-expert\n"
	  << "// avg K-range — affects mlp3/mlp4 bwd perf since each PE only\n"
	  << "// walks its own experts). Deployment lookup selects this subtable by Compute;\n"
	  << "// n_pes is the tuning-time PE count;\n"
	  << "// deployment lookup should compute TKE the same way.\n"
	  << "// Each row pairs the best fwd template with the best bwd template for\n"
	  << "// that shape. The COMM tile is fixed at 128 for both directions;\n"
	  << "// Fwd_TileM / Bwd_TileM are the per-direction GEMM tiles and may differ.\n"
	  << "// Written incrementally; a partial file means the tuner crashed.\n\n"
	  << "#include \"moe_fwd_bwd_tuning_config_types.cuh\"\n\n"
	  << "namespace liger {\n\n";
	if (g_tuned_rows.empty()) {
		f << "static const TunedConfigFwdBwd " << arr << "[1] = {{\n"
		  << "\t/*TK=*/0, /*TKE=*/0, /*D=*/0, /*I=*/0,\n"
		  << "\t/*Fwd_NSplit=*/0,\n"
		  << "\t/*Fwd_TileN1=*/0, /*Fwd_TileK1=*/0, /*Fwd_Stages1=*/0, /*Fwd_EpiChunkN1=*/0,\n"
		  << "\t/*Fwd_TileN2=*/0, /*Fwd_TileK2=*/0, /*Fwd_Stages2=*/0, /*Fwd_EpiChunkN2=*/0,\n"
		  << "\t/*Fwd_ZBufferSlots=*/0, /*Fwd_CommNumStages=*/0,\n"
		  << "\t/*Bwd_NSplit=*/0, /*Bwd_NSplit2=*/0,\n"
		  << "\t/*Bwd_TileN1=*/0, /*Bwd_TileK1=*/0, /*Bwd_Stages1=*/0,\n"
		  << "\t/*Bwd_TileM3=*/0, /*Bwd_TileN3=*/0, /*Bwd_TileK3=*/0, /*Bwd_Stages3=*/0,\n"
		  << "\t/*Bwd_EpiChunkN1=*/0, /*Bwd_EpiChunkN25=*/0, /*Bwd_EpiChunkN34=*/0,\n"
		  << "\t/*Bwd_CommNumStages=*/0,\n"
		  << "\t/*Fwd_TileM=*/0, /*Bwd_TileM=*/0,\n"
		  << "\t/*fwd_ms=*/0.0f, /*bwd_ms=*/0.0f, /*combined_ms=*/0.0f,\n"
		  << "}};\n\n"
		  << "static constexpr int " << count << " = 0;\n\n";
	} else {
		f << "static const TunedConfigFwdBwd " << arr << "[] = {\n";
		for (const auto& r : g_tuned_rows) {
			const auto& fe = kRegistryFwd[r.fwd_ci];
			const auto& be = kRegistryBwd[r.bwd_ci];
			f << "\t{" << r.TK << ", " << r.TKE << ", " << r.D << ", " << r.I << ", "
			  << fe.NSplit << ", "
			  << fe.TileN1 << ", " << fe.TileK1 << ", " << fe.Stages1 << ", " << fe.EpiChunkN1 << ", "
			  << fe.TileN2 << ", " << fe.TileK2 << ", " << fe.Stages2 << ", " << fe.EpiChunkN2 << ", "
			  << fe.ZBufferSlots << ", " << fe.CommNumStages << ", "
			  << be.NSplit << ", " << be.NSplit2 << ", "
			  << be.TileN1 << ", " << be.TileK1 << ", " << be.Stages1 << ", "
			  << be.TileM3 << ", " << be.TileN3 << ", " << be.TileK3 << ", " << be.Stages3 << ", "
			  << be.EpiChunkN1 << ", " << be.EpiChunkN25 << ", " << be.EpiChunkN34 << ", "
			  << be.CommNumStages << ", "
			  << fe.TileM << ", " << be.TileM << ", "
			  << r.fwd_ms << "f, " << r.bwd_ms << "f, " << r.combined_ms << "f"
			  << "},  // FWD=" << fe.name << " | BWD=" << be.name << "\n";
		}
		f << "};\n\n"
		  << "static constexpr int " << count << " =\n"
		  << "\tsizeof(" << arr << ") / sizeof(" << arr << "[0]);\n\n";
	}
	f << "} // namespace liger\n";
}

// ── Tune one shape: decouple fwd/bwd within each TileM bucket ────────

static void tune_shape(
		int T, int D, int I, int E, int K,
		int pe, int n_pes, nvshmem_team_t team) {

	int epp = E / n_pes;

	// Pre-filter: every fwd / bwd config that passes its own shape-validity
	// check. The TileM grouping below partitions these into the two buckets
	// the bwd is allowed to pair with (bwd inherits fwd's sort layout, so a
	// fwd and bwd timed together must agree on TileM).
	std::vector<int> fwd_candidates, bwd_candidates;
	for (int ci = 0; ci < kNumFwdConfigs; ++ci)
		if (kRegistryFwd[ci].Compute == g_tuner_compute &&
		    fwd_shape_valid(D, I, kRegistryFwd[ci])) fwd_candidates.push_back(ci);
	for (int ci = 0; ci < kNumBwdConfigs; ++ci)
		if (kRegistryBwd[ci].Compute == g_tuner_compute &&
		    bwd_shape_valid(D, I, kRegistryBwd[ci])) bwd_candidates.push_back(ci);

	// Optional: force a specific TileM (e.g. MOE_FWDBWD_TUNE_TILEM=128) to
	// inspect a single bucket. Filters both fwd and bwd candidates so only
	// that TileM's bucket is viable below.
	if (const char* tilem_env = getenv("MOE_FWDBWD_TUNE_TILEM")) {
		int want_tm = atoi(tilem_env);
		fwd_candidates.erase(std::remove_if(fwd_candidates.begin(), fwd_candidates.end(),
			[&](int ci) { return kRegistryFwd[ci].TileM != want_tm; }),
			fwd_candidates.end());
		bwd_candidates.erase(std::remove_if(bwd_candidates.begin(), bwd_candidates.end(),
			[&](int ci) { return kRegistryBwd[ci].TileM != want_tm; }),
			bwd_candidates.end());
	}

	// Group candidates into TileM buckets. The fwd and bwd directions are
	// tuned INDEPENDENTLY within a bucket (see header), so a bucket is
	// viable only if it has at least one fwd AND one bwd candidate.
	static constexpr int kTileMBuckets[2] = {64, 128};
	std::vector<int> fwd_by_tm[2], bwd_by_tm[2];
	for (int fi : fwd_candidates)
		for (int b = 0; b < 2; ++b)
			if (kRegistryFwd[fi].TileM == kTileMBuckets[b]) fwd_by_tm[b].push_back(fi);
	for (int bi : bwd_candidates)
		for (int b = 0; b < 2; ++b)
			if (kRegistryBwd[bi].TileM == kTileMBuckets[b]) bwd_by_tm[b].push_back(bi);

	bool any_bucket = false;
	for (int b = 0; b < 2; ++b)
		if (!fwd_by_tm[b].empty() && !bwd_by_tm[b].empty()) any_bucket = true;
	if (!any_bucket) {
		if (pe == 0) {
			printf("%-7d %-6d %-6d  (no viable fwd+bwd bucket)\n", T, D, I);
			fflush(stdout);
		}
		return;
	}

	// Per-shape memory gate (same logic as tune_moe_bwd.cu: NVSHMEM's
	// lazy heap commits mean a static budget can't see prior-shape
	// reservations — query cudaMemGetInfo for the live free count).
	c10::cuda::CUDACachingAllocator::emptyCache();
	size_t free_bytes = 0, total_bytes = 0;
	cudaMemGetInfo(&free_bytes, &total_bytes);
	size_t need_bytes = estimate_memory_bytes(T, D, I, E, K, epp);
	if (need_bytes > free_bytes) {
		if (pe == 0) {
			printf("%-7d %-6d %-6d  SKIP (insufficient memory: need %.1f GiB, free %.1f GiB)\n",
				T, D, I,
				need_bytes / 1024.0 / 1024.0 / 1024.0,
				free_bytes / 1024.0 / 1024.0 / 1024.0);
			fflush(stdout);
		}
		return;
	}

	// Inputs live in REGULAR (non-symmetric) device memory. The fused MoE
	// kernels read every user input with LOCAL ops — there is no NVSHMEM
	// access to X / dY / router outputs / weights:
	//   - X            → local dispatch_tokens, scattered into x_sorted
	//                    (moe.cu: p.tokens, dispatch.cuh int4 loads)
	//   - expert_indices → local sort_tokens (moe.cu sort)
	//   - expert_weights → local combine_tokens / combine_tokens_bwd
	//   - all_B/all_C/all_A → local TMA loads
	//   - dY / Y_fwd   → local combine_tokens_bwd (moe_bwd.cu Phase 3)
	// All cross-PE NVSHMEM traffic rides on kernel-INTERNAL symmetric
	// buffers (x_sorted / y_buf / all_expert_offsets on the symm-stack,
	// dy_sorted / dx_sorted / staging / signals from the buffer pool),
	// which the kernel wrappers allocate themselves. So the inputs need
	// not be symmetric, and keeping them out of the NVSHMEM heap leaves
	// more symmetric room for those internal buffers.
	//
	// (An older version pre-allocated these in symm memory to dodge a
	// top_k=1 hang attributed to a since-removed "auto-copy" wrapper path;
	// that path no longer exists — today's templated wrappers read inputs
	// directly with local ops.)
	auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

	torch::Tensor X, dY, gate_weight, all_B, all_C, all_A;
	torch::Tensor expert_indices, expert_weights;
	try {
		X     = torch::rand({T, D}, opts_bf16) - 0.5f;
		dY    = torch::rand({T, D}, opts_bf16) - 0.5f;
		all_B = torch::rand({epp, I, D}, opts_bf16) - 0.5f;
		all_C = torch::rand({epp, I, D}, opts_bf16) - 0.5f;
		all_A = torch::rand({epp, D, I}, opts_bf16) - 0.5f;

		// Standard top-K gate in plain torch (the router kernel is out of scope
		// for this package — only the routing OUTPUTS feed the fused kernels, and
		// the tuner cares about kernel timing, not the routing values). int32
		// indices + bf16 softmax weights match what the kernels expect.
		gate_weight = torch::rand({E, D}, opts_bf16) - 0.5f;
		auto logits = torch::matmul(X.to(torch::kFloat32), gate_weight.to(torch::kFloat32).t());
		auto topk = torch::topk(logits, K, /*dim=*/1);
		expert_indices = std::get<1>(topk).to(torch::kInt32).contiguous();
		expert_weights = torch::softmax(std::get<0>(topk), /*dim=*/1).to(torch::kBFloat16).contiguous();
		cudaDeviceSynchronize();

		tuner_team_sync(team, n_pes);
	} catch (const std::exception& ex) {
		if (pe == 0) {
			printf("%-7d %-6d %-6d  SKIP (input alloc failed: %s)\n",
				T, D, I, ex.what());
			fflush(stdout);
		}
		(void)cudaGetLastError();
		c10::cuda::CUDACachingAllocator::emptyCache();
		return;
	}

	// Router load-balance stats — bin expert_indices [T, K] into a per-
	// expert histogram and report min/max/mean tokens per expert. This is
	// a property of the (random) inputs + the trained router weights; for
	// a fully balanced router every expert would get exactly T*K/E tokens.
	// Computed once per shape; merged into the BEST-COMBINED row below so
	// each shape produces a single line (rather than a separate LB row).
	int64_t lb_min = 0, lb_max = 0;
	double  lb_mean = 0.0;
	double  lb_imbalance = 1.0;  // max / mean (1.0 = perfectly balanced)
	if (pe == 0) {
		// bincount returns int64; cap minlength at E so empty experts
		// still appear in the histogram (otherwise the tail is dropped).
		// Filter to valid [0, E) ids first — the router can emit -1 padding
		// slots, which bincount rejects ("1-d non-negative integral inputs").
		// These stats are report-only; a bad histogram must not abort tuning.
		auto idx_flat = expert_indices.flatten().to(torch::kInt64);
		idx_flat = idx_flat.index({(idx_flat >= 0) & (idx_flat < E)});
		auto counts = torch::bincount(
			idx_flat,
			/*weights=*/torch::Tensor(),
			/*minlength=*/E
		).cpu();
		int64_t* cnt = counts.data_ptr<int64_t>();
		lb_min = cnt[0];
		lb_max = cnt[0];
		int64_t sum = 0;
		for (int e = 0; e < E; ++e) {
			if (cnt[e] < lb_min) lb_min = cnt[e];
			if (cnt[e] > lb_max) lb_max = cnt[e];
			sum += cnt[e];
		}
		lb_mean = (double)sum / E;
		lb_imbalance = (lb_mean > 0) ? (double)lb_max / lb_mean : 1.0;
	}

	// ── Decoupled per-TileM sweep ────────────────────────────────────
	// Within each TileM bucket, tune fwd and bwd INDEPENDENTLY. Every
	// measurement still runs a full fwd→bwd sequence via run_pair (the bwd
	// needs fwd-produced sort inputs), but we read only the direction being
	// tuned and hold the OTHER direction at a fixed default from the same
	// bucket. fwd/bwd timings are invariant to the other direction's config
	// once TileM is fixed (separate sequential launches), so the bucket's
	// combined-best is best_fwd_ms + best_bwd_ms — found in O(F+B) runs
	// instead of the O(F·B) cross product.
	struct BucketBest {
		int   fwd_ci = -1, bwd_ci = -1;
		float fwd_ms = 1e30f, bwd_ms = 1e30f;
	};
	BucketBest bucket[2];
	bool shape_oom = false;

	for (int b = 0; b < 2 && !shape_oom; ++b) {
		if (fwd_by_tm[b].empty() || bwd_by_tm[b].empty()) continue;
		const int default_bwd = bwd_by_tm[b][0];

		// FWD sweep: vary fwd, hold bwd at default_bwd, keep fwd_ms.
		for (size_t i = 0; i < fwd_by_tm[b].size() && !shape_oom; ++i) {
			int fi = fwd_by_tm[b][i];
			auto r = run_pair(kRegistryFwd[fi], kRegistryBwd[default_bwd],
				X, dY, expert_indices, expert_weights,
				all_B, all_C, all_A, E, K, team, pe, n_pes);
			cudaDeviceSynchronize();
			tuner_team_sync(team, n_pes);
			if (r.oom) { shape_oom = true; break; }
			if (r.fwd_ms > 0 && r.fwd_ms < bucket[b].fwd_ms) {
				bucket[b].fwd_ms = r.fwd_ms;
				bucket[b].fwd_ci = fi;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		// Seed the BWD sweep with the best fwd that ACTUALLY ran (it produced
		// fwd_ms > 0, so its sort outputs are valid), not just the first
		// candidate — a statically-valid fwd can still fault at runtime, and
		// a broken seed would sink every bwd measurement in the bucket. If no
		// fwd ran, the bucket can't form a combined time, so skip its bwd
		// sweep entirely.
		if (bucket[b].fwd_ci < 0) continue;
		const int seed_fwd = bucket[b].fwd_ci;

		// BWD sweep: vary bwd, hold fwd at seed_fwd, keep bwd_ms.
		for (size_t i = 0; i < bwd_by_tm[b].size() && !shape_oom; ++i) {
			int bi = bwd_by_tm[b][i];
			auto r = run_pair(kRegistryFwd[seed_fwd], kRegistryBwd[bi],
				X, dY, expert_indices, expert_weights,
				all_B, all_C, all_A, E, K, team, pe, n_pes);
			cudaDeviceSynchronize();
			tuner_team_sync(team, n_pes);
			if (r.oom) { shape_oom = true; break; }
			if (r.bwd_ms > 0 && r.bwd_ms < bucket[b].bwd_ms) {
				bucket[b].bwd_ms = r.bwd_ms;
				bucket[b].bwd_ci = bi;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	// Pick the best fwd and best bwd INDEPENDENTLY across the two GEMM-tile
	// buckets. The comm tile is fixed at 128 for both directions, so the bwd
	// always inherits a 128-aligned sort layout regardless of either side's
	// GEMM tile — fwd and bwd no longer need to share TileM. (bwd_ms in each
	// bucket was measured paired with that bucket's seed fwd, but the bwd
	// timing is invariant to the fwd config, so cross-bucket comparison is
	// valid.) best_fwd_b / best_bwd_b may select different buckets.
	int   best_fwd_b = -1, best_bwd_b = -1;
	float best_fwd_ms = 1e30f, best_bwd_ms = 1e30f;
	for (int b = 0; b < 2; ++b) {
		if (bucket[b].fwd_ci >= 0 && bucket[b].fwd_ms < best_fwd_ms) {
			best_fwd_ms = bucket[b].fwd_ms; best_fwd_b = b;
		}
		if (bucket[b].bwd_ci >= 0 && bucket[b].bwd_ms < best_bwd_ms) {
			best_bwd_ms = bucket[b].bwd_ms; best_bwd_b = b;
		}
	}
	// A complete row needs both a fwd and a bwd winner.
	int best_b = (best_fwd_b >= 0 && best_bwd_b >= 0) ? best_fwd_b : -1;

	// FLOPs accounting (per direction so the printed columns match the
	// user's mental model of "fwd flops" vs "bwd flops"; combined TFLOPS
	// is the sum / combined_ms).
	double Td = T, Dd = D, Id = I, Kd = K;
	double fwd_flops = 2.0 * 2.0 * Td * Kd * Dd * Id    // MLP1 (B,C)
	                 + 9.0 * Td * Kd * Id               // SiLU
	                 + 2.0 * Td * Kd * Id * Dd          // MLP2
	                 + 2.0 * Td * Kd * Dd;              // combine
	double bwd_flops = 16.0 * Td * Kd * Dd * Id;        // matches bench_moe_bwd

	if (pe == 0) {
		// One line per TileM bucket. Full template names land in the .cuh
		// output file (trailing `// FWD=... | BWD=...` comments); the
		// terminal row carries TileM + the two NSplit knobs that distinguish
		// most winners. LB stats appear only on the chosen (BEST-COMBINED)
		// row — they're a shape property, identical across buckets.
		auto print_bucket = [&](const char* tag, int b, bool with_lb) {
			if (b < 0 || bucket[b].fwd_ci < 0 || bucket[b].bwd_ci < 0) return;
			const auto& fe = kRegistryFwd[bucket[b].fwd_ci];
			const auto& be = kRegistryBwd[bucket[b].bwd_ci];
			float fwd_ms = bucket[b].fwd_ms;
			float bwd_ms = bucket[b].bwd_ms;
			float combined_ms = fwd_ms + bwd_ms;
			double fwd_tf = fwd_flops / (fwd_ms * 1e-3) / 1e12;
			double bwd_tf = bwd_flops / (bwd_ms * 1e-3) / 1e12;
			double com_tf = (fwd_flops + bwd_flops) / (combined_ms * 1e-3) / 1e12;
			if (with_lb) {
				printf("%-7d %-6d %-6d  [%s] "
				       "LB[min=%lld/%.1f%% max=%lld/%.1f%% imb=%.2fx]  "
				       "fwd=%6.2fms/%5.0fTF  bwd=%6.2fms/%5.0fTF  "
				       "combined=%6.2fms/%5.0fTF  [TM%d FWD=NS%d BWD=NS%d/NS2-%d]\n",
					T, D, I, tag,
					(long long)lb_min, 100.0 * lb_min / ((double)T * K),
					(long long)lb_max, 100.0 * lb_max / ((double)T * K),
					lb_imbalance,
					fwd_ms, fwd_tf, bwd_ms, bwd_tf, combined_ms, com_tf,
					fe.TileM, fe.NSplit, be.NSplit, be.NSplit2);
			} else {
				printf("%-7d %-6d %-6d  [%s]                                          "
				       "fwd=%6.2fms/%5.0fTF  bwd=%6.2fms/%5.0fTF  "
				       "combined=%6.2fms/%5.0fTF  [TM%d FWD=NS%d BWD=NS%d/NS2-%d]\n",
					T, D, I, tag,
					fwd_ms, fwd_tf, bwd_ms, bwd_tf, combined_ms, com_tf,
					fe.TileM, fe.NSplit, be.NSplit, be.NSplit2);
			}
		};
		print_bucket("BEST-COMBINED", best_b, /*with_lb=*/true);
		// Also surface the non-winning bucket as a diagnostic.
		for (int b = 0; b < 2; ++b)
			if (b != best_b) print_bucket("ALT-BUCKET   ", b, /*with_lb=*/false);
		if (shape_oom) {
			printf("%-7d %-6d %-6d  PARTIAL (OOM mid-shape — remaining configs dropped)\n",
				T, D, I);
		} else if (best_b < 0) {
			printf("%-7d %-6d %-6d  (no bucket produced a fwd+bwd winner)\n",
				T, D, I);
		}
		fflush(stdout);
	}

	// Persist the winning bucket's (fwd, bwd) pair after every successful
	// shape so later crashes don't lose data we've already collected.
	if (pe == 0 && best_b >= 0) {
		// TKE = TK / E_local; E_local = epp = E / n_pes (already computed
		// at the top of tune_shape). Guard against epp=0 in case some
		// future caller passes a malformed shape. fwd and bwd are taken from
		// their INDEPENDENT winning buckets (may differ in GEMM tile).
		const int E_local = epp > 0 ? epp : 1;
		g_tuned_rows.push_back({
			kRegistryFwd[bucket[best_fwd_b].fwd_ci].Compute,
			T * K, (T * K) / E_local, D, I,
			bucket[best_fwd_b].fwd_ci, bucket[best_bwd_b].bwd_ci,
			bucket[best_fwd_b].fwd_ms, bucket[best_bwd_b].bwd_ms,
			bucket[best_fwd_b].fwd_ms + bucket[best_bwd_b].bwd_ms,
		});
		dump_tuned_configs();
	}

	// Drop torch allocator's cached blocks so the next shape sees the
	// actual free memory (mirrors tune_moe_bwd.cu). Inputs are plain torch
	// tensors now, so freeing the cache reclaims them — no symm frees.
	c10::cuda::CUDACachingAllocator::emptyCache();
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
	setvbuf(stdout, nullptr, _IONBF, 0);

	const char* procid = getenv("SLURM_PROCID");
	if (procid) cudaSetDevice(atoi(procid));
	const bool has_pmi =
		procid || getenv("PMI_RANK") || getenv("PMIX_RANK") ||
		getenv("OMPI_COMM_WORLD_RANK");
	auto nvshmem_status = LIGER_CUTE_OK;
	if (has_pmi) {
		nvshmem_status = liger_cute_nvshmem_init_pmi();
	}
	if (!has_pmi || nvshmem_status != LIGER_CUTE_OK) {
		// Standalone single-GPU tuning pods often do not run under PMI/Slurm.
		// Fall back to the unique-id bootstrap for that case; multi-rank tuning
		// should still be launched with PMI so ranks receive a shared ID.
		size_t uid_nbytes = 0;
		nvshmem_status = liger_cute_nvshmem_uniqueid_nbytes(&uid_nbytes);
		if (nvshmem_status != LIGER_CUTE_OK) return 1;
		std::vector<unsigned char> uid(uid_nbytes);
		nvshmem_status = liger_cute_nvshmem_get_uniqueid(uid.data());
		if (nvshmem_status != LIGER_CUTE_OK) return 1;
		nvshmem_status = liger_cute_nvshmem_init_with_uniqueid(0, 1, uid.data());
		if (nvshmem_status != LIGER_CUTE_OK) return 1;
	}
	nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
	int pe    = nvshmem_team_my_pe(team);
	int n_pes = nvshmem_team_n_pes(team);
	cudaSetDevice(pe);
	g_tuner_n_pes = n_pes;  // selects single- vs multi-GPU config class on dump
	int dev = 0;
	cudaGetDevice(&dev);
	g_tuner_compute = liger::moe_detect_compute_dispatch_key_noexcept(dev);
	if (g_tuner_compute <= 0) {
		if (pe == 0) {
			int unsupported = -g_tuner_compute;
			printf("Error: unsupported CUDA compute capability sm_%d. "
			       "Supported compute capabilities are sm_90 (Hopper) and sm_100 (Blackwell).\n",
			       unsupported);
		}
		(void)liger_cute_nvshmem_finalize();
		return 1;
	}

	// Sweep matches tune_moe_bwd.cu (the more restrictive of the two
	// existing tuners — T=16384 trips an unspecified launch failure on
	// the bwd path; capped at 8192 until investigated).
	const std::vector<int>   token_sweep  = {256, 512, 1024, 2048, 4096, 8192};
	const std::vector<int>   hidden_sweep = {512, 1024, 2048, 4096, 8192};
	const std::vector<float> ratio_sweep  = {0.5f, 1.0f, 2.0f};
	constexpr int kMinIntermediate = 256;

	// Sweep on E_local (experts per PE) — matches TKE = TK / E_local in
	// the tuning table. Each sweep value v produces a synthetic-shape run
	// at E_global = v * n_pes, which is what gets passed to the kernel.
	// Sweeping in local terms means realistic deployment sizes (e.g.
	// E_local=8 means 8 experts per PE — matches Qwen3-style configs at
	// 16 PEs), rather than implying impractical global counts.
	//
	// MOE_FWDBWD_TUNE_E=N overrides to a single E_local value (also in
	// local terms — set N=2 to pin every synthetic shape at E_local=2).
	// Generic-sweep top_k (fixed at 2 for the synthetic shapes). Per-model
	// top_k for named shapes is pinned in the NamedShape table below.
	const int K = 2;
	std::vector<int> expert_sweep_local = {1, 2, 4, 8};
	const char* e_env = getenv("MOE_FWDBWD_TUNE_E");
	if (e_env) expert_sweep_local = {atoi(e_env)};
	expert_sweep_local.erase(
		std::remove_if(expert_sweep_local.begin(), expert_sweep_local.end(),
			[K](int e) { return e <= 0 || e < K; }),
		expert_sweep_local.end());
	if (expert_sweep_local.empty()) {
		if (pe == 0)
			printf("Error: empty expert_sweep_local (env override was non-positive?)\n");
		(void)liger_cute_nvshmem_finalize();
		return 1;
	}

	// Named model shapes — pinned to each model's real (E, top_k) so the
	// tuner sees the exact deployment shape rather than a synthetic E. Every
	// row uses the model's EXACT config.json values: D = hidden_size, I =
	// the per-expert MoE FFN width (moe_intermediate_size, or intermediate_size
	// for the Mixtral/Scout routed experts), E = total experts, K = experts
	// per token. All values verified against the published config.json:
	//
	//   Llama-4 Scout 17B-16E:  D=5120,  I=8192,  E=16,  K=1
	//     (num_local_experts=16, num_experts_per_tok=1, hidden_size=5120,
	//      intermediate_size=8192 — the routed-expert FFN; the dense
	//      intermediate_size_mlp=16384 is a separate non-MoE path).
	//   Mixtral 8x7B:           D=4096,  I=14336, E=8,   K=2
	//   Mixtral 8x22B:          D=6144,  I=16384, E=8,   K=2
	//     (num_local_experts=8, num_experts_per_tok=2).
	//   Qwen3-30B-A3B:          D=2048,  I=768,   E=128, K=8
	//   Qwen3-235B-A22B:        D=4096,  I=1536,  E=128, K=8
	//     (num_experts=128, num_experts_per_tok=8, moe_intermediate_size).
	//   Qwen3.5-35B-A3B:        D=2048,  I=512,   E=256, K=8
	//   Qwen3.5-122B-A10B:      D=3072,  I=1024,  E=256, K=8
	//   Qwen3.5-397B-A17B:      D=4096,  I=1024,  E=512, K=10
	//     (Qwen3_5MoeForConditionalGeneration; num_experts, num_experts_per_tok,
	//      moe_intermediate_size).
	//
	// Large-E families (Qwen3 E=128, Qwen3.5 E=256/512) are pinned in
	// family_sweeps below so their real per-GPU expert density is exercised
	// (see that table). E must be divisible by n_pes for the real-config pass
	// to run; the natural deployment is 8 PEs.
	struct NamedShape { const char* name; int T, D, I, E, K; };
	const NamedShape named_shapes[] = {
		{"scout-17B-16E-1024",        1024, 5120,  8192,  16,  1},
		{"scout-17B-16E-2048",        2048, 5120,  8192,  16,  1},
		{"scout-17B-16E-4096",        4096, 5120,  8192,  16,  1},
		{"scout-17B-16E-8192",        8192, 5120,  8192,  16,  1},
		{"mixtral-8x7B-1024",         1024, 4096, 14336,   8,  2},
		{"mixtral-8x7B-2048",         2048, 4096, 14336,   8,  2},
		{"mixtral-8x7B-4096",         4096, 4096, 14336,   8,  2},
		{"mixtral-8x7B-8192",         8192, 4096, 14336,   8,  2},
		{"mixtral-8x22B-1024",        1024, 6144, 16384,   8,  2},
		{"mixtral-8x22B-2048",        2048, 6144, 16384,   8,  2},
		{"mixtral-8x22B-4096",        4096, 6144, 16384,   8,  2},
		{"mixtral-8x22B-8192",        8192, 6144, 16384,   8,  2},
		{"qwen3-30B-A3B-1024",        1024, 2048,   768, 128,  8},
		{"qwen3-30B-A3B-2048",        2048, 2048,   768, 128,  8},
		{"qwen3-30B-A3B-4096",        4096, 2048,   768, 128,  8},
		{"qwen3-30B-A3B-8192",        8192, 2048,   768, 128,  8},
		{"qwen3-235B-A22B-1024",      1024, 4096,  1536, 128,  8},
		{"qwen3-235B-A22B-2048",      2048, 4096,  1536, 128,  8},
		{"qwen3-235B-A22B-4096",      4096, 4096,  1536, 128,  8},
		{"qwen3-235B-A22B-8192",      8192, 4096,  1536, 128,  8},
		{"qwen3.5-35B-A3B-1024",      1024, 2048,   512, 256,  8},
		{"qwen3.5-35B-A3B-2048",      2048, 2048,   512, 256,  8},
		{"qwen3.5-35B-A3B-4096",      4096, 2048,   512, 256,  8},
		{"qwen3.5-35B-A3B-8192",      8192, 2048,   512, 256,  8},
		{"qwen3.5-122B-A10B-1024",    1024, 3072,  1024, 256,  8},
		{"qwen3.5-122B-A10B-2048",    2048, 3072,  1024, 256,  8},
		{"qwen3.5-122B-A10B-4096",    4096, 3072,  1024, 256,  8},
		{"qwen3.5-122B-A10B-8192",    8192, 3072,  1024, 256,  8},
		{"qwen3.5-397B-A17B-1024",    1024, 4096,  1024, 512, 10},
		{"qwen3.5-397B-A17B-2048",    2048, 4096,  1024, 512, 10},
		{"qwen3.5-397B-A17B-4096",    4096, 4096,  1024, 512, 10},
		{"qwen3.5-397B-A17B-8192",    8192, 4096,  1024, 512, 10},
	};

	// Per-family overrides for named-shape (E, K). Set
	//   MOE_FWDBWD_TUNE_SCOUT_E=N / _K=N
	//   MOE_FWDBWD_TUNE_MIXTRAL_E=N / _K=N
	//   MOE_FWDBWD_TUNE_QWEN3_E=N / _K=N
	//   MOE_FWDBWD_TUNE_QWEN35_E=N / _K=N
	// to tune a named model at a non-default expert count or top_k (useful
	// for measuring how perf scales with E without editing source). When
	// unset, falls back to the model's real config above.
	//
	// NOTE: "qwen3.5" contains "qwen3" as a substring, so the qwen3.5 check
	// MUST precede the qwen3 check or qwen3.5 shapes would be misclassified.
	auto family_of = [](const char* name) -> const char* {
		if (std::strstr(name, "scout"))    return "SCOUT";
		if (std::strstr(name, "mixtral"))  return "MIXTRAL";
		if (std::strstr(name, "qwen3.5"))  return "QWEN35";
		if (std::strstr(name, "qwen3"))    return "QWEN3";
		return nullptr;
	};
	auto env_int_or = [](const char* var, int fallback) {
		const char* v = std::getenv(var);
		return v ? std::atoi(v) : fallback;
	};
	auto effective_E = [&](const NamedShape& s) {
		const char* fam = family_of(s.name);
		if (!fam) return s.E;
		char var[64];
		std::snprintf(var, sizeof(var), "MOE_FWDBWD_TUNE_%s_E", fam);
		return env_int_or(var, s.E);
	};
	auto effective_K = [&](const NamedShape& s) {
		const char* fam = family_of(s.name);
		if (!fam) return s.K;
		char var[64];
		std::snprintf(var, sizeof(var), "MOE_FWDBWD_TUNE_%s_K", fam);
		return env_int_or(var, s.K);
	};

	// Per-family E_local sweep — each named shape is tuned at all of
	// these E_local values at K=2 (synthetic, for cross-E perf curves)
	// plus once at its real (E_global, K) above. Values are LOCAL
	// (per-PE) counts; the loop converts to E_global = v * n_pes before
	// calling tune_shape. Defaults to {1, 2, 4, 8} for unlisted families.
	//
	// Large-E families are pinned to the E_local that reproduces their real
	// per-GPU expert density at the natural 8-PE deployment, so the synthetic
	// K=2 sweep exercises the actual expert count (the default {1,2,4,8} tops
	// out at E_local=8 = E_global 64, far below these models). Pinning here
	// also keeps the symm pool sized for the real E (pool sizing reads these).
	//   Qwen3   E=128         -> E_local 16  (128 / 8)
	//   Qwen3.5 E=256 and 512 -> E_local 32 and 64  (256/8, 512/8)
	struct FamilySweep { const char* family; std::vector<int> E_local_values; };
	const std::vector<FamilySweep> family_sweeps = {
		// Qwen3 pinned to 16 experts/GPU (= real 128-expert model on 8 PEs).
		{"QWEN3", {16}},
		// Qwen3.5 spans two expert counts; sweep both real densities so the
		// 256- and 512-expert variants each get a real-density data point.
		{"QWEN35", {32, 64}},
		// SCOUT / MIXTRAL (and any unlisted family) default to {1, 2, 4, 8}.
	};
	auto sweep_local_Es_for_family = [&](const char* fam) -> std::vector<int> {
		if (fam) {
			for (const auto& fs : family_sweeps)
				if (std::strcmp(fs.family, fam) == 0) return fs.E_local_values;
		}
		return {1, 2, 4, 8};
	};

	// Env-driven sweep filters — hoisted above pool sizing because both
	// skip_generic and the family filter affect max_E / max_K.
	const char* t_env = getenv("MOE_FWDBWD_TUNE_T");
	const char* d_env = getenv("MOE_FWDBWD_TUNE_D");
	const char* i_env = getenv("MOE_FWDBWD_TUNE_I");
	const bool skip_named   = getenv("MOE_FWDBWD_TUNE_SKIP_NAMED")   != nullptr;
	const bool skip_generic = getenv("MOE_FWDBWD_TUNE_SKIP_GENERIC") != nullptr;
	if (pe == 0 && skip_generic && !getenv("LIGER_MOE_FWDBWD_TUNED_OUTPUT")) {
		fprintf(stderr,
			"WARNING: MOE_FWDBWD_TUNE_SKIP_GENERIC=1 will overwrite the default\n"
			"  output .cuh with named-shape rows only — generic-sweep rows from\n"
			"  the previous run will be LOST. Set LIGER_MOE_FWDBWD_TUNED_OUTPUT\n"
			"  to a path ending in %s to redirect, then merge manually. Continuing in 3 seconds...\n",
			tuned_output_compute_suffix());
		std::this_thread::sleep_for(std::chrono::seconds(3));
	}

	// MOE_FWDBWD_TUNE_FAMILY=scout|mixtral|qwen3|qwen3.5 — restrict the named-shape
	// pass to one family AND collapse the symm-pool sizing to just that
	// family. Useful when the combined max_E / max_K across all families bloats
	// the pool beyond what NVSHMEM can allocate (or causes a hang at the
	// first kernel call). Affects pool sizing AND which named shapes run.
	const char* fam_filter = std::getenv("MOE_FWDBWD_TUNE_FAMILY");
	auto family_in_filter = [&](const NamedShape& s) {
		if (!fam_filter) return true;
		const char* fam = family_of(s.name);
		if (!fam) return false;
		// Compare case-insensitively (filter "scout" matches family "SCOUT").
		// Also strip '.' so the natural "qwen3.5" filter matches tag "QWEN35".
		std::string f;
		for (char c : std::string(fam_filter))
			if (c != '.') f += std::toupper((unsigned char)c);
		return f == fam;
	};

	// Pool sizing: must cover the generic sweep AND the worst-case named
	// shape that will actually run (after family filter + override),
	// including the per-family E_local sweep contribution. Sweep values
	// are E_local; multiply by n_pes to get the E_global the pool sees.
	int max_E = skip_generic
		? 0  // No generic sweep — pool sized by named shapes alone.
		: *std::max_element(expert_sweep_local.begin(), expert_sweep_local.end()) * n_pes;
	int max_K = skip_generic ? 0 : K;
	for (const auto& s : named_shapes) {
		if (!family_in_filter(s)) continue;
		const char* fam = family_of(s.name);
		// Family E_local-sweep contribution (synthetic runs at K=2).
		for (int E_local_s : sweep_local_Es_for_family(fam))
			max_E = std::max(max_E, E_local_s * n_pes);
		if (fam) max_K = std::max(max_K, 2);
		// Real-model contribution (effective_E returns E_global).
		max_E = std::max(max_E, effective_E(s));
		max_K = std::max(max_K, effective_K(s));
	}
	if (max_E == 0 || max_K == 0) {
		if (pe == 0) printf("Error: no shapes to tune (filters skip everything)\n");
		(void)liger_cute_nvshmem_finalize();
		return 1;
	}

	int max_tokens = t_env ? std::atoi(t_env) : *std::max_element(token_sweep.begin(), token_sweep.end());
	int max_hidden = d_env ? std::atoi(d_env) : *std::max_element(hidden_sweep.begin(), hidden_sweep.end());

	// Single-host tuner: N=1 hosts, M=n_pes GPUs.
	liger::moe_configure_symmetric(max_tokens, max_hidden, max_E, max_K, n_pes,
		/*num_hosts=*/1, /*gpus_per_host=*/n_pes);

	if (pe == 0) {
		printf("=== MoE Fused Forward+Backward Combined Autotuner ===\n");
		printf("PEs=%d K=%d  num_fwd_configs=%d  num_bwd_configs=%d\n",
			n_pes, K, kNumFwdConfigs, kNumBwdConfigs);
		printf("E_local sweep: [");
		for (size_t i = 0; i < expert_sweep_local.size(); ++i)
			printf("%s%d", i ? ", " : "", expert_sweep_local[i]);
		printf("]  (E_global = E_local * n_pes; TKE = TK / E_local)\n");
		printf("Token sweep: %d..%d  Hidden sweep: %d..%d  I min=%d  I/D ratios: 0.5..2\n",
			token_sweep.front(), token_sweep.back(),
			hidden_sweep.front(), hidden_sweep.back(),
			kMinIntermediate);
		printf("Class: %s (tuned at n_pes=%d)\n",
			n_pes <= 1 ? "SINGLE-GPU" : "MULTI-GPU", n_pes);
		printf("Output: ../src/liger_comm_kernels/moe/moe_fwd_bwd_tuning_configs_%s.cuh\n\n",
			n_pes <= 1 ? "single" : "multi");
		printf("%-7s %-6s %-6s  result\n", "T", "D", "I");
		printf("─────────────────────────────────────────────────────────────────────────────────────────\n");
	}

	// MOE_FWDBWD_TUNE_T/D/I env filters consumed below pin individual axes
	// to a single value for smoke testing (declared up top — see hoist).

	// E_local outermost so output is grouped by E (easier to read
	// incrementally, and partial crashes preserve complete-E slices of
	// the table). Sweep values are LOCAL; multiply by n_pes for the
	// E_global the kernel sees.
	if (!skip_generic) {
		for (int E_local : expert_sweep_local) {
			int E = E_local * n_pes;
			if (pe == 0)
				printf("\n========= E_local=%d (E_global=%d) =========\n",
					E_local, E);
			for (int D : hidden_sweep) {
				if (d_env && D != atoi(d_env)) continue;
				for (float r : ratio_sweep) {
					int I_raw = std::max(kMinIntermediate, (int)(D * r));
					int I = (I_raw + 127) & ~127;
					if (i_env && I != atoi(i_env)) continue;
					for (int T : token_sweep) {
						if (t_env && T != atoi(t_env)) continue;
						tune_shape(T, D, I, E, K, pe, n_pes, team);
						cudaDeviceSynchronize();
						tuner_team_sync(team, n_pes);
					}
					if (pe == 0) printf("\n");
				}
			}
		}
	}

	// Named-shape pass: for each shape, run
	//   (a) the per-family E sweep at K=2 (synthetic — covers the perf
	//       curve across expert counts), then
	//   (b) one run at the model's real (E, K) (after env override).
	// Both contribute rows to the .cuh; the append-mode merge on output
	// deduplicates by (TK, TKE, D, I) shape key, so the real (E, K) row
	// always adds at least one unique entry.
	if (!skip_named) {
		if (pe == 0) printf("\n=== Named model shapes%s ===\n",
			fam_filter ? " [family filter active]" : "");
		for (const auto& s : named_shapes) {
			if (!family_in_filter(s)) continue;
			const char* fam = family_of(s.name);

			// (a) Family E_local sweep at K=2. Values are LOCAL; convert
			// to E_global = E_local * n_pes before tuning.
			for (int E_local_s : sweep_local_Es_for_family(fam)) {
				int E_s = E_local_s * n_pes;
				if (pe == 0)
					printf("%s [sweep E_local=%d (E=%d), K=2]:\n",
						s.name, E_local_s, E_s);
				tune_shape(s.T, s.D, s.I, E_s, 2, pe, n_pes, team);
				cudaDeviceSynchronize();
				tuner_team_sync(team, n_pes);
			}

			// (b) Real model config (after env override).
			int E_eff = effective_E(s);
			int K_eff = effective_K(s);
			if (E_eff % n_pes != 0) {
				if (pe == 0)
					printf("%s [real]:  SKIP (E=%d not divisible by num_pes=%d)\n",
						s.name, E_eff, n_pes);
				continue;
			}
			// The router trait fixes MaxTopK=8 (router.cu RouterTraits); K>8
			// (e.g. qwen3.5-397B's K=10) would throw "top_k exceeds MaxTopK" in
			// moe_router_fwd. Skip cleanly instead of catching a stack trace.
			if (K_eff > 8) {
				if (pe == 0)
					printf("%s [real]:  SKIP (top_k=%d > MaxTopK=8 unsupported)\n",
						s.name, K_eff);
				continue;
			}
			if (pe == 0) {
				bool E_ovr = E_eff != s.E;
				bool K_ovr = K_eff != s.K;
				printf("%s [real E=%d%s, K=%d%s]:\n",
					s.name,
					E_eff, E_ovr ? " [override]" : "",
					K_eff, K_ovr ? " [override]" : "");
			}
			tune_shape(s.T, s.D, s.I, E_eff, K_eff, pe, n_pes, team);
			cudaDeviceSynchronize();
			tuner_team_sync(team, n_pes);
		}
	}

	if (pe == 0)
		printf("─────────────────────────────────────────────────────────────────────────────────────────\n");

	(void)liger_cute_pool_clear_all();

	if (pe == 0) {
		dump_tuned_configs();
		printf("\nWrote %zu tuned configs\n", g_tuned_rows.size());
	}

	(void)liger_cute_nvshmem_finalize();
	return 0;
}
