// ═══════════════════════════════════════════════════════════════════
// Numerical-correctness + TFLOPS tests for the MLP4 device functions:
//   * mlp4_{producer,consumer}  →  dB = dU^T·X,  dC = dV^T·X  (weight grads)
//
// mlp4 = mlp3 + a two-phase (dB / dC) loop. Per cell the cooperative 2-WG
// consumer runs TWO sequential phases with INDEPENDENT accumulators (cleared
// fresh each phase — NOT a cross-phase sum): phase 0 computes dB = dU^T·X,
// phase 1 computes dC = dV^T·X. Both operands are MN-major; X is shared by both
// phases/WGs. Both outputs are written via SM90_TMA_REDUCE_ADD, so dB and dC
// MUST be zero-initialized by the caller (and re-zeroed between launches).
//
// Self-contained (no torch, no nvshmem): each TEST builds its own inputs on the
// host, drives a persistent chunk-fixed launcher kernel (grid.x = cells, one
// cell = (chunk, walk-lane), both phases run internally), and compares dB / dC
// against fp32 CPU references computed from the *same bf16-rounded* inputs.
//
// Exercises the mlp4 consumers on BOTH architectures, AUTO-GATED to the running
// GPU so the output stays clean:
//   * sm_100 (Blackwell) → Compute=100 — always the paired-CTA 2SM path:
//     Mlp4Traits2Sm + mlp4_fwd<Traits,100> (cudaLaunchKernelEx, even grid,
//     clusterDim=(2,1,1), UMMA + make_tma_copy_{A,B}_sm100 operand loads).
//   * sm_90  (Hopper)    → Compute=90  — the original 1SM path: Mlp4Traits +
//     mlp4_fwd<Traits,90> (ordinary <<<>>> launch, WGMMA, plain make_tma_copy).
// Both paths share cpu_reference and tolerances (run4 is templated on Compute).
// The test kernel itself is a thin wrapper that forwards straight to the
// unified liger::mlp4_fwd<Traits,Compute> device function, so one source
// builds cleanly for sm_90a and sm_100a — mlp4_fwd internally gates its
// Compute=100 body on __CUDA_ARCH__>=1000 (and traps otherwise).
//
// Two tiny single-tile DIAGNOSTIC tests isolate the two-phase routing:
//   * PhaseDB (dV=0 → dB = dU^T·X, dC = 0)  — isolates phase 0 + independent clear.
//   * PhaseDC (dU=0 → dC = dV^T·X, dB = 0)  — isolates phase 1 + independent clear.
// If an isolated case passes but the combined case fails → phase output routing /
// acc-carry-across-phases / re-zero bug. If an isolated case fails structurally →
// the MN-major operand or the store-buf mapping is wrong.
// ═══════════════════════════════════════════════════════════════════

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
// SM100 2SM (paired-CTA cluster) TMA descriptor factories (make_tma_copy_A_sm100
// / make_tma_copy_B_sm100) and the SM100_TMA_2SM_LOAD copy atom, mirroring
// moe_bwd.cu's Phase-2 TMA construction for the Compute=100 X/dU^T/dV^T operands.
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp4.cuh"

using namespace cute;
using liger::Mlp4Traits;
using liger::Mlp4Traits2Sm;
using liger::Mlp4Smem;
using Element = cutlass::bfloat16_t;

// Compute=90 (Hopper/WGMMA, 1SM): mlp4's default M-split (TileM=256,
// TileN=128), TileK=64, Stages=4, EpiChunkN=64. Also test the N-split (128,
// 256) config: it exercises the other store-buf mapping (byte-for-byte
// atom-row formula) + X-split UMMA.
using TraitsM = Mlp4Traits<Element, /*TileM=*/256, /*TileN=*/128,
                           /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;
using TraitsN = Mlp4Traits<Element, /*TileM=*/128, /*TileN=*/256,
                           /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;

// Compute=100 (Blackwell/UMMA, paired-CTA 2SM): the production refactor makes
// this the ONLY Compute=100 path (no more 1SM UMMA). Mlp4Traits2Sm requires
// BOTH TileM==256 and TileN==256 (no N-split variant, unlike mlp3); default
// TileK=64, Stages=3, EpiChunkN=128, ClusterM=2 — the same defaults
// moe_bwd.cu's Phase-2 Traits4 config uses.
using TraitsBw = Mlp4Traits2Sm<Element, /*TileM=*/256, /*TileN=*/256,
                               /*TileK=*/64, /*Stages=*/3, /*EpiChunkN=*/128>;

#define CUDA_OK(expr)                                                       \
	do {                                                                    \
		cudaError_t _e = (expr);                                            \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)

// ═══════════════════════════════════════════════════════════════════
// Persistent chunk-fixed launcher kernel (1D grid for Compute=90; paired-CTA
// cluster grid for Compute=100 — blockIdx.x/gridDim.x are raw CTA
// coordinates, mlp4_fwd itself divides by Traits::ClusterM internally for the
// 2SM path). Each CTA grid-strides over cells, running both dB/dC phases
// internally via the unified liger::mlp4_fwd<Traits,Compute> (no hand-rolled
// pipe/TMEM/producer-consumer code). Compute=100 uses the UMMA-aware paired
// pipe; Compute=90 the Hopper pipe. Both share SharedStorage — Mlp4Smem<Traits>
// drives either (Mlp4Smem2Sm<Traits> is literally an alias for Mlp4Smem<Traits>,
// unlike mlp3 which has two distinct smem structs).
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, int Compute, typename TmaLoadX, typename TmaLoadA,
          typename TmaReduceAdd>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp4_test_kernel(
		__grid_constant__ TmaLoadX    const tma_load_x,
		__grid_constant__ TmaLoadA    const tma_load_dut,
		__grid_constant__ TmaLoadA    const tma_load_dvt,
		__grid_constant__ TmaReduceAdd const tma_reduce_db,
		__grid_constant__ TmaReduceAdd const tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim, int hidden_dim,
		int num_tokens, int total_m_rows,
		int num_m_tiles, int num_n_tiles,
		int outer_split) {

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Mlp4Smem<Traits>*>(raw_smem);

	liger::mlp4_fwd<Traits, Compute, Compute == 90>(
		smem, tma_load_x, tma_load_dut, tma_load_dvt,
		tma_reduce_db, tma_reduce_dc,
		expert_k_starts, expert_k_ends, num_experts,
		intermediate_dim, hidden_dim, num_tokens, total_m_rows,
		num_m_tiles, num_n_tiles, outer_split);
}

// ═══════════════════════════════════════════════════════════════════
// Host helpers
// ═══════════════════════════════════════════════════════════════════

struct Mlp4Shape {
	int num_tokens;        // T — contraction axis (multiple of E·TileK)
	int hidden_dim;        // H — N axis (output width), multiple of TileN
	int intermediate_dim;  // I — M axis (output height), multiple of TileM
	int num_experts;       // E
};

enum ZeroMode { ZM_NONE = 0, ZM_DU0 = 1, ZM_DV0 = 2 };   // phase diagnostics

struct DevBf16 {
	Element* ptr = nullptr;
	size_t   n   = 0;
	~DevBf16() { if (ptr) cudaFree(ptr); }
};

static void upload_bf16(DevBf16& d, const std::vector<float>& host_f) {
	std::vector<Element> host_b(host_f.size());
	for (size_t i = 0; i < host_f.size(); ++i) host_b[i] = Element(host_f[i]);
	d.n = host_f.size();
	cudaMalloc(&d.ptr, d.n * sizeof(Element));
	cudaMemcpy(d.ptr, host_b.data(), d.n * sizeof(Element), cudaMemcpyHostToDevice);
}

static void zero_dev(DevBf16& d) {
	if (d.ptr) cudaMemset(d.ptr, 0, d.n * sizeof(Element));
}

static inline float bf16_round(float x) { return float(Element(x)); }

struct ErrStats { float max_abs; float mean_rel; float max_rel; };

static ErrStats compare(const std::vector<float>& got,
                        const std::vector<float>& ref) {
	const float atol = 1e-3f;
	float max_abs = 0.f, sum_rel = 0.f, max_rel = 0.f;
	for (size_t i = 0; i < ref.size(); ++i) {
		float d = std::fabs(got[i] - ref[i]);
		float r = d / std::max(std::fabs(ref[i]), atol);
		max_abs = std::max(max_abs, d);
		max_rel = std::max(max_rel, r);
		sum_rel += r;
	}
	return {max_abs, sum_rel / ref.size(), max_rel};
}

// CPU reference. Expert e owns the contiguous token range [e·T/E, (e+1)·T/E).
//   dB[e·I+i, h] = Σ_{t∈e} dU[t,i]·X[t,h]
//   dC[e·I+i, h] = Σ_{t∈e} dV[t,i]·X[t,h]
// Inputs are bf16-rounded; accumulation is fp32. Outputs are [E·I, H] row-major.
static void cpu_reference(
		const std::vector<float>& dU,    // [T, I] bf16-rounded
		const std::vector<float>& dV,    // [T, I]
		const std::vector<float>& X,     // [T, H]
		const Mlp4Shape& s,
		std::vector<float>& refB,        // out [E·I, H]
		std::vector<float>& refC) {
	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim, E = s.num_experts;
	refB.assign((size_t)E * I * H, 0.f);
	refC.assign((size_t)E * I * H, 0.f);
	int t_per = T / E;
	for (int e = 0; e < E; ++e) {
		int t0 = e * t_per, t1 = t0 + t_per;
		for (int i = 0; i < I; ++i) {
			for (int h = 0; h < H; ++h) {
				float accB = 0.f, accC = 0.f;
				for (int t = t0; t < t1; ++t) {
					float x = X[(size_t)t * H + h];
					accB += dU[(size_t)t * I + i] * x;
					accC += dV[(size_t)t * I + i] * x;
				}
				refB[((size_t)e * I + i) * H + h] = accB;
				refC[((size_t)e * I + i) * H + h] = accC;
			}
		}
	}
}

struct Inputs {
	std::vector<float> dU, dV, X;              // bf16-rounded host copies
	std::vector<int>   k_starts, k_ends;       // per-expert K-block range
	DevBf16 dDU, dDV, dX, ddB, ddC;
	int* d_kstart = nullptr;
	int* d_kend   = nullptr;
	int num_m_tiles, num_n_tiles, total_m_rows;
	~Inputs() { if (d_kstart) cudaFree(d_kstart); if (d_kend) cudaFree(d_kend); }
};

template <typename Traits>
static void make_inputs(const Mlp4Shape& s, Inputs& in, unsigned seed, ZeroMode zm) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> nd(0.f, 1.f);
	auto fill = [&](std::vector<float>& v, size_t n, bool zero) {
		v.resize(n);
		for (size_t i = 0; i < n; ++i) v[i] = zero ? 0.f : bf16_round(nd(rng));
	};
	fill(in.dU, (size_t)s.num_tokens * s.intermediate_dim, zm == ZM_DU0);
	fill(in.dV, (size_t)s.num_tokens * s.intermediate_dim, zm == ZM_DV0);
	fill(in.X,  (size_t)s.num_tokens * s.hidden_dim,        false);

	in.num_m_tiles  = s.intermediate_dim / Traits::TileM;
	in.num_n_tiles  = s.hidden_dim / Traits::TileN;
	in.total_m_rows = s.num_experts * s.intermediate_dim;

	// Expert e owns tokens [e·T/E, (e+1)·T/E) → K-blocks [·/TileK).
	int t_per = s.num_tokens / s.num_experts;
	in.k_starts.resize(s.num_experts);
	in.k_ends.resize(s.num_experts);
	for (int e = 0; e < s.num_experts; ++e) {
		in.k_starts[e] = (e * t_per) / Traits::TileK;
		in.k_ends[e]   = ((e + 1) * t_per) / Traits::TileK;
	}

	upload_bf16(in.dDU, in.dU);
	upload_bf16(in.dDV, in.dV);
	upload_bf16(in.dX,  in.X);
	// Outputs zero-initialized (REDUCE_ADD accumulates).
	in.ddB.n = (size_t)in.total_m_rows * s.hidden_dim;
	in.ddC.n = in.ddB.n;
	cudaMalloc(&in.ddB.ptr, in.ddB.n * sizeof(Element));
	cudaMalloc(&in.ddC.ptr, in.ddC.n * sizeof(Element));
	zero_dev(in.ddB);
	zero_dev(in.ddC);

	cudaMalloc(&in.d_kstart, s.num_experts * sizeof(int));
	cudaMalloc(&in.d_kend,   s.num_experts * sizeof(int));
	cudaMemcpy(in.d_kstart, in.k_starts.data(), s.num_experts * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(in.d_kend,   in.k_ends.data(),   s.num_experts * sizeof(int), cudaMemcpyHostToDevice);
}

static std::vector<float> download_bf16(const DevBf16& d) {
	std::vector<Element> hb(d.n);
	cudaMemcpy(hb.data(), d.ptr, d.n * sizeof(Element), cudaMemcpyDeviceToHost);
	std::vector<float> out(d.n);
	for (size_t i = 0; i < out.size(); ++i) out[i] = float(hb[i]);
	return out;
}

// Divisors of n → balanced walk-axis (outer_split) partitions.
static std::vector<int> divisors_of(int n) {
	std::vector<int> ds;
	for (int d = 1; d <= n; ++d)
		if (n % d == 0) ds.push_back(d);
	if (ds.empty()) ds.push_back(1);
	return ds;
}

// Build the five TMA descriptors for one shape (X/dU/dV loads, dB/dC
// reduces). Compute=90 uses ordinary 1SM make_tma_copy; Compute=100 uses the
// pair-aware make_tma_copy_{A,B}_sm100 factories (SM100_TMA_2SM_LOAD copy op,
// keyed off Traits::TileShape + Traits::TiledMma2Sm) — exactly moe_bwd.cu's
// Phase-2 X/dU^T/dV^T descriptor construction for Config::kUsesTwoSm. The
// dB/dC reduce-add outputs stay ordinary (non-paired) TMA_REDUCE_ADD either
// way. The field types differ between Compute values (different Copy_Atom
// specializations), so the aggregate returned by make_tmas is a LOCAL struct
// with types deduced via decltype of already-constructed local variables
// (not a pre-declared template `Tmas<Traits>`, which could only hold one
// fixed set of field types).
template <typename Traits, int Compute>
static auto make_tmas(const Mlp4Shape& s, Inputs& in) {
	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim;
	// X: [T, H] row-major → (H, T) MN-major (H contiguous) — operand B.
	auto tX = make_tensor(make_gmem_ptr((const Element*)in.dX.ptr),
		make_shape(H, T), make_stride(Int<1>{}, H));
	// dU/dV: [T, I] row-major → (I, T) MN-major (I contiguous) — operand A.
	auto tDU = make_tensor(make_gmem_ptr((const Element*)in.dDU.ptr),
		make_shape(I, T), make_stride(Int<1>{}, I));
	auto tDV = make_tensor(make_gmem_ptr((const Element*)in.dDV.ptr),
		make_shape(I, T), make_stride(Int<1>{}, I));
	auto x_tma = [&] {
		if constexpr (Compute == 100) {
			return make_tma_copy_B_sm100(SM100_TMA_2SM_LOAD{}, tX,
				typename Traits::SmemLayoutX_1{},
				typename Traits::TileShape{},
				typename Traits::TiledMma2Sm{});
		} else {
			return make_tma_copy(SM90_TMA_LOAD{}, tX, typename Traits::SmemLayoutX_1{});
		}
	}();
	auto dut_tma = [&] {
		if constexpr (Compute == 100) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tDU,
				typename Traits::SmemLayoutA_1{},
				typename Traits::TileShape{},
				typename Traits::TiledMma2Sm{});
		} else {
			return make_tma_copy(SM90_TMA_LOAD{}, tDU, typename Traits::SmemLayoutA_1{});
		}
	}();
	auto dvt_tma = [&] {
		if constexpr (Compute == 100) {
			return make_tma_copy_A_sm100(SM100_TMA_2SM_LOAD{}, tDV,
				typename Traits::SmemLayoutA_1{},
				typename Traits::TileShape{},
				typename Traits::TiledMma2Sm{});
		} else {
			return make_tma_copy(SM90_TMA_LOAD{}, tDV, typename Traits::SmemLayoutA_1{});
		}
	}();
	auto db_tma = [&] {
		if constexpr (Compute == 90) {
			auto tDB = make_tensor(make_gmem_ptr((Element*)in.ddB.ptr),
				make_shape(I, H, s.num_experts),
				make_stride(H, Int<1>{}, I * H));
			return make_tma_copy(
				SM90_TMA_REDUCE_ADD{}, tDB, typename Traits::SmemLayoutStore{});
		} else {
			auto tDB = make_tensor(make_gmem_ptr((Element*)in.ddB.ptr),
				make_shape(in.total_m_rows, H), make_stride(H, Int<1>{}));
			return make_tma_copy(
				SM90_TMA_REDUCE_ADD{}, tDB, typename Traits::SmemLayoutStore{});
		}
	}();
	auto dc_tma = [&] {
		if constexpr (Compute == 90) {
			auto tDC = make_tensor(make_gmem_ptr((Element*)in.ddC.ptr),
				make_shape(I, H, s.num_experts),
				make_stride(H, Int<1>{}, I * H));
			return make_tma_copy(
				SM90_TMA_REDUCE_ADD{}, tDC, typename Traits::SmemLayoutStore{});
		} else {
			auto tDC = make_tensor(make_gmem_ptr((Element*)in.ddC.ptr),
				make_shape(in.total_m_rows, H), make_stride(H, Int<1>{}));
			return make_tma_copy(
				SM90_TMA_REDUCE_ADD{}, tDC, typename Traits::SmemLayoutStore{});
		}
	}();

	struct Result {
		decltype(x_tma) x; decltype(dut_tma) dut; decltype(dvt_tma) dvt;
		decltype(db_tma) db; decltype(dc_tma) dc;
	};
	return Result{x_tma, dut_tma, dvt_tma, db_tma, dc_tma};
}

// Compute=100 is always the paired-CTA 2SM path (chunk=(e,n_tile), walks
// m_tile — the same convention as the 1SM kMSplit=true case); Compute=90 walks
// whichever axis Traits::kMSplit selects. `Traits::kMSplit` is looked up only
// inside the untaken (discarded) branch when Compute==100, so this compiles
// even though Mlp4Traits2Sm has no kMSplit member.
template <typename Traits, int Compute>
static constexpr bool mlp4_walks_m() {
	if constexpr (Compute == 100) return true;
	else return Traits::kMSplit;
}

// ═══════════════════════════════════════════════════════════════════
// Variant runner — one launch at a given outer_split, compare dB & dC.
// ═══════════════════════════════════════════════════════════════════
template <typename Traits, int Compute>
static void run4_once(const Mlp4Shape& s, Inputs& in, int outer_split,
                      bool verbose, const char* tag,
                      ErrStats* outB, ErrStats* outC) {
	auto tmas = make_tmas<Traits, Compute>(s, in);

	size_t smem_size = sizeof(Mlp4Smem<Traits>);
	auto kernel = mlp4_test_kernel<Traits, Compute,
		decltype(tmas.x), decltype(tmas.dut), decltype(tmas.db)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	constexpr bool walks_m = mlp4_walks_m<Traits, Compute>();
	int total_chunks = s.num_experts * (walks_m ? in.num_n_tiles : in.num_m_tiles);
	int total_cells = total_chunks * outer_split;   // k_split = 1

	// Re-zero outputs (REDUCE_ADD accumulates; caller may reuse buffers).
	zero_dev(in.ddB);
	zero_dev(in.ddC);
	CUDA_OK(cudaDeviceSynchronize());

	if constexpr (Compute == 100) {
		// Paired-CTA cluster launch: opt in to non-portable cluster sizes,
		// then cudaLaunchKernelEx with an EVEN grid.x (a multiple of
		// Traits::ClusterM — mlp4_fwd __traps otherwise) and
		// clusterDim=(ClusterM,1,1). One CTA-pair per cell (matching this
		// function's original uncapped-grid philosophy for Compute=90 below):
		// the grid-stride cell loop is a no-op for any pairs beyond
		// total_cells, so rounding up to a ClusterM multiple is harmless.
		CUDA_OK(cudaFuncSetAttribute(kernel,
			cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
		int pairs  = std::max(1, total_cells);
		int grid_x = pairs * Traits::ClusterM;

		cudaLaunchConfig_t launch_config = {};
		launch_config.gridDim  = dim3(grid_x);
		launch_config.blockDim = dim3(Traits::NumThreads);
		launch_config.dynamicSmemBytes = smem_size;
		launch_config.stream = nullptr;
		cudaLaunchAttribute cluster_attr = {};
		cluster_attr.id = cudaLaunchAttributeClusterDimension;
		cluster_attr.val.clusterDim.x = Traits::ClusterM;
		cluster_attr.val.clusterDim.y = 1;
		cluster_attr.val.clusterDim.z = 1;
		launch_config.attrs = &cluster_attr;
		launch_config.numAttrs = 1;
		CUDA_OK(cudaLaunchKernelEx(&launch_config, kernel,
			tmas.x, tmas.dut, tmas.dvt, tmas.db, tmas.dc,
			in.d_kstart, in.d_kend, s.num_experts,
			s.intermediate_dim, s.hidden_dim, s.num_tokens, in.total_m_rows,
			in.num_m_tiles, in.num_n_tiles, outer_split));
	} else {
		dim3 grid(total_cells, 1, 1);
		kernel<<<grid, Traits::NumThreads, smem_size>>>(
			tmas.x, tmas.dut, tmas.dvt, tmas.db, tmas.dc,
			in.d_kstart, in.d_kend, s.num_experts,
			s.intermediate_dim, s.hidden_dim, s.num_tokens, in.total_m_rows,
			in.num_m_tiles, in.num_n_tiles, outer_split);
	}
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto gotB = download_bf16(in.ddB);
	auto gotC = download_bf16(in.ddC);
	std::vector<float> refB, refC;
	cpu_reference(in.dU, in.dV, in.X, s, refB, refC);
	*outB = compare(gotB, refB);
	*outC = compare(gotC, refC);
	if (verbose)
		printf("[mlp4 C=%-3d %-10s T=%d H=%d I=%d E=%d osplit=%d] "
		       "dB{mean=%.3f%% max=%.3f%% abs=%.3g}  dC{mean=%.3f%% max=%.3f%% abs=%.3g}\n",
			Compute, tag, s.num_tokens, s.hidden_dim, s.intermediate_dim,
			s.num_experts, outer_split,
			outB->mean_rel * 100, outB->max_rel * 100, outB->max_abs,
			outC->mean_rel * 100, outC->max_rel * 100, outC->max_abs);
}

// Full-shape correctness: outer_split = 1 AND, when the walk axis has ≥2
// tiles, a 2-way split, exercising the walk-axis partition. Both dB and dC
// must match. Compute=100's 2SM producer/consumer use a balanced
// multiply-before-divide split that correctly covers a NON-divisible walk axis
// (see mlp4.cuh's "Balanced split" comment), so it is exercised here even when
// the split doesn't divide evenly; Compute=90's naive floor-division split
// would silently drop the tail, so it is only exercised when it divides evenly
// (matching the ORIGINAL, pre-refactor test coverage for that path).
template <typename Traits, int Compute>
static void run4(const Mlp4Shape& s) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234, ZM_NONE);
	constexpr bool walks_m = mlp4_walks_m<Traits, Compute>();
	int walk = walks_m ? in.num_m_tiles : in.num_n_tiles;

	ErrStats b1{}, c1{};
	run4_once<Traits, Compute>(s, in, /*outer_split=*/1, /*verbose=*/true, "", &b1, &c1);
	EXPECT_LT(b1.mean_rel, 0.01f); EXPECT_LT(b1.max_rel, 0.05f);
	EXPECT_LT(c1.mean_rel, 0.01f); EXPECT_LT(c1.max_rel, 0.05f);

	if (walk >= 2 && (Compute == 100 || walk % 2 == 0)) {
		const char* tag = (walk % 2 == 0) ? "osplit2" : "osplit2_tail";
		ErrStats b2{}, c2{};
		run4_once<Traits, Compute>(s, in, /*outer_split=*/2, /*verbose=*/true, tag, &b2, &c2);
		EXPECT_LT(b2.mean_rel, 0.01f); EXPECT_LT(b2.max_rel, 0.05f);
		EXPECT_LT(c2.mean_rel, 0.01f); EXPECT_LT(c2.max_rel, 0.05f);
	}
}

// Diagnostic: tiny single-tile phase isolation (dV=0 → dB only, dU=0 → dC only).
template <typename Traits, int Compute>
static void run4_isolate(const Mlp4Shape& s, ZeroMode zm, const char* tag) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/777, zm);
	ErrStats b{}, c{};
	run4_once<Traits, Compute>(s, in, /*outer_split=*/1, /*verbose=*/true, tag, &b, &c);
	// Both the active and the zeroed output must be correct (zeroed → ref is 0,
	// so any acc-carry / mis-route leaks a large value and fails here).
	EXPECT_LT(b.mean_rel, 0.01f) << tag << " dB mean_rel too high";
	EXPECT_LT(b.max_rel,  0.05f) << tag << " dB max_rel too high";
	EXPECT_LT(c.mean_rel, 0.01f) << tag << " dC mean_rel too high";
	EXPECT_LT(c.max_rel,  0.05f) << tag << " dC max_rel too high";
}

// ═══════════════════════════════════════════════════════════════════
// TFLOPS benchmark (opt-in via MLP4_BENCH env; timing-only, no CPU ref)
// ═══════════════════════════════════════════════════════════════════
//
//     TFLOPS = 4·T·H·I / median_kernel_seconds / 1e12   (two GEMMs)
//
// Persistent grid: grid.x = num_sms, CTAs grid-stride over cells. Sweep
// outer_split over divisors of the walk axis → finds the split that best fills
// the SMs. Outputs are zeroed ONCE before each split's timing loop (NOT inside
// the timed region): with REDUCE_ADD the outputs accumulate across launches,
// but the GEMM work — hence the timing — is identical regardless of the
// accumulated magnitude, and values stay finite in bf16 over ~60 launches. This
// matches the mlp5 methodology (time only the kernel); folding a ~0.5 GB
// double-memset into every sample would measure memset bandwidth, not GEMM
// throughput. (run4/run4_isolate DO re-zero per launch — that path checks
// numerics.)

struct BenchCfg { int warmup = 10; int iters = 50; };

static int sm_count() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return 0;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
	return p.multiProcessorCount;
}

static double median_ms(std::vector<float>& v) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	size_t n = v.size();
	return (n & 1) ? (double)v[n / 2] : 0.5 * ((double)v[n / 2 - 1] + (double)v[n / 2]);
}

static bool mlp4_bench_enabled() { return std::getenv("MLP4_BENCH") != nullptr; }

static double tflops_of(const Mlp4Shape& s, double ms) {
	double flops = 4.0 * (double)s.num_tokens * (double)s.hidden_dim
	             * (double)s.intermediate_dim;
	return flops / (ms * 1e-3) / 1e12;
}

template <typename Traits, int Compute>
static void run4_bench(const Mlp4Shape& s, const BenchCfg& cfg) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234, ZM_NONE);
	auto tmas = make_tmas<Traits, Compute>(s, in);

	size_t smem_size = sizeof(Mlp4Smem<Traits>);
	auto kernel = mlp4_test_kernel<Traits, Compute,
		decltype(tmas.x), decltype(tmas.dut), decltype(tmas.db)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
	if constexpr (Compute == 100) {
		CUDA_OK(cudaFuncSetAttribute(kernel,
			cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
	}

	int nsm = sm_count();
	constexpr bool walks_m = mlp4_walks_m<Traits, Compute>();
	int walk = walks_m ? in.num_m_tiles : in.num_n_tiles;
	int total_chunks = s.num_experts * (walks_m ? in.num_n_tiles : in.num_m_tiles);

	double best_tf = 0.0, best_ms = 0.0; int best_split = 1, best_gx = 1;
	for (int osplit : divisors_of(walk)) {
		int total_cells = total_chunks * osplit;
		int gx;
		if constexpr (Compute == 100) {
			int pairs = std::max(1, std::min(nsm / Traits::ClusterM, total_cells));
			gx = pairs * Traits::ClusterM;
		} else {
			gx = std::max(1, std::min(nsm, total_cells));
		}
		auto launch = [&]() {
			if constexpr (Compute == 100) {
				cudaLaunchConfig_t launch_config = {};
				launch_config.gridDim  = dim3(gx);
				launch_config.blockDim = dim3(Traits::NumThreads);
				launch_config.dynamicSmemBytes = smem_size;
				launch_config.stream = nullptr;
				cudaLaunchAttribute cluster_attr = {};
				cluster_attr.id = cudaLaunchAttributeClusterDimension;
				cluster_attr.val.clusterDim.x = Traits::ClusterM;
				cluster_attr.val.clusterDim.y = 1;
				cluster_attr.val.clusterDim.z = 1;
				launch_config.attrs = &cluster_attr;
				launch_config.numAttrs = 1;
				cudaLaunchKernelEx(&launch_config, kernel,
					tmas.x, tmas.dut, tmas.dvt, tmas.db, tmas.dc,
					in.d_kstart, in.d_kend, s.num_experts,
					s.intermediate_dim, s.hidden_dim, s.num_tokens, in.total_m_rows,
					in.num_m_tiles, in.num_n_tiles, osplit);
			} else {
				dim3 grid(gx, 1, 1);
				kernel<<<grid, Traits::NumThreads, smem_size>>>(
					tmas.x, tmas.dut, tmas.dvt, tmas.db, tmas.dc,
					in.d_kstart, in.d_kend, s.num_experts,
					s.intermediate_dim, s.hidden_dim, s.num_tokens, in.total_m_rows,
					in.num_m_tiles, in.num_n_tiles, osplit);
			}
		};
		// Zero once (REDUCE_ADD): keeps values finite; NOT in the timed region.
		zero_dev(in.ddB); zero_dev(in.ddC);
		launch(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());

		// Median over iters — pure kernel timing (memset excluded).
		cudaEvent_t start, stop;
		cudaEventCreate(&start); cudaEventCreate(&stop);
		for (int i = 0; i < cfg.warmup; ++i) launch();
		CUDA_OK(cudaDeviceSynchronize());
		std::vector<float> samples; samples.reserve(cfg.iters);
		for (int i = 0; i < cfg.iters; ++i) {
			cudaEventRecord(start);
			launch();
			cudaEventRecord(stop);
			CUDA_OK(cudaEventSynchronize(stop));
			float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
			samples.push_back(ms);
		}
		cudaEventDestroy(start); cudaEventDestroy(stop);
		double ms = median_ms(samples);
		double tf = tflops_of(s, ms);
		if (tf > best_tf) { best_tf = tf; best_ms = ms; best_split = osplit; best_gx = gx; }
	}
	printf("[mlp4-bench C=%-3d T=%-5d H=%d I=%d E=%d] "
	       "peak %8.2f TFLOPS @ %8.4f ms (osplit=%-2d, grid.x=%d, %d SMs)\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		best_tf, best_ms, best_split, best_gx, nsm);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

static bool blackwell_available() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return false;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
	return p.major == 10;
}

static bool hopper_available() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return false;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
	return p.major == 9;
}

// Tiny single-tile shape for phase isolation (Compute=100 / TraitsBw, both
// TileM and TileN fixed at 256): one M-tile, one N-tile, one K-tile
// (TileK=64), one expert. total_m_rows=256.
static const Mlp4Shape kTinyShape = {/*T=*/64, /*H=*/256, /*I=*/256, /*E=*/1};

// Small correctness shapes. I multiple of TileM(256), H of TileN(128), and
// T/E a multiple of TileK(64) → exact FLOP count, clean expert token ranges.
static const std::vector<Mlp4Shape> kShapes = {
	{  64, 128, 256, 1},   // single tile, single k-block
	{ 128, 256, 256, 1},   // two N-tiles, two K-blocks
	{ 256, 256, 512, 2},   // two M-tiles / two experts
	{ 512, 384, 768, 2},   // three M-tiles, three N-tiles, deeper K
};

// N-split (TraitsN, TileM=128 TileN=256) correctness shapes: I multiple of 128,
// H of 256, T/E multiple of 64.
static const std::vector<Mlp4Shape> kShapesN = {
	{  64, 256, 128, 1},   // single tile
	{ 128, 512, 256, 1},   // two M-tiles, two N-tiles
	{ 256, 512, 256, 2},   // two experts
};

// Compute=100 (paired-CTA 2SM, TraitsBw) correctness shapes: Mlp4Traits2Sm
// requires TileM==256 AND TileN==256 fixed (no N-split variant, unlike mlp3)
// — I multiple of 256, H multiple of 256, T/E multiple of 64. The last shape
// gives num_m_tiles=3 (768/256) — an odd, non-divisible walk-axis tile count
// that explicitly exercises the 2SM producer/consumer's balanced
// multiply-before-divide outer_split=2 tail handling (run4() always tries a
// 2-way split for Compute=100, divisible or not).
static const std::vector<Mlp4Shape> kShapes2Sm = {
	{  64, 256, 256, 1},   // single tile, single k-block
	{ 128, 256, 512, 1},   // two M-tiles, one N-tile, two K-blocks
	{ 256, 512, 512, 2},   // two experts, two M-tiles, two N-tiles
	{ 512, 512, 768, 2},   // two experts, three M-tiles (odd!), two N-tiles
};

// Large, GPU-saturating shapes for the TFLOPS benchmark (H=I=4096, E=8).
static const std::vector<Mlp4Shape> kBenchShapes = {
	{ 2048, 4096, 4096, 8},
	{ 4096, 4096, 4096, 8},
	{ 8192, 4096, 4096, 8},
	{16384, 4096, 4096, 8},
};

// ── Diagnostics (run FIRST): tiny single-tile phase isolation. ──
TEST(Mlp4, PhaseDB) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	run4_isolate<TraitsBw, 100>(kTinyShape, ZM_DV0, "DB(dU^T*X)");
}

TEST(Mlp4, PhaseDC) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	run4_isolate<TraitsBw, 100>(kTinyShape, ZM_DU0, "DC(dV^T*X)");
}

// ── Blackwell (Compute=100) — always the paired-CTA 2SM path; requires an
//    sm_100 GPU at runtime ──
TEST(Mlp4, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapes2Sm) run4<TraitsBw, 100>(s);
}

// ── Hopper (Compute=90 / WGMMA, 1SM) — requires an sm_90 GPU at runtime ──
TEST(Mlp4Sm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes)  run4<TraitsM, 90>(s);
	for (const auto& s : kShapesN) run4<TraitsN, 90>(s);
}

// ── TFLOPS benchmarks — opt-in via MLP4_BENCH=1. ──
TEST(Mlp4, TFLOPs_Blackwell) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (!mlp4_bench_enabled())  GTEST_SKIP() << "set MLP4_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run4_bench<TraitsBw, 100>(s, cfg);
}

TEST(Mlp4, TFLOPs_Hopper) {
	if (!hopper_available())   GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	if (!mlp4_bench_enabled()) GTEST_SKIP() << "set MLP4_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run4_bench<TraitsM, 90>(s, cfg);
}

// ═══════════════════════════════════════════════════════════════════
// Entry point — arch-aware default filter (clean output). Diagnostics run
// before Correctness (registration order) so a phase-isolation failure shows
// first. An explicit --gtest_filter or --gtest_list_tests takes precedence.
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

	const bool user_filtered = GTEST_FLAG_GET(filter) != "*";
	const bool listing       = GTEST_FLAG_GET(list_tests);
	if (!user_filtered && !listing) {
		std::string f;
		if (blackwell_available()) {
			f = "Mlp4.PhaseDB:Mlp4.PhaseDC:Mlp4.Correctness";
			if (mlp4_bench_enabled())
				f += ":Mlp4.TFLOPs_Blackwell";
		} else if (hopper_available()) {
			f = "Mlp4Sm90.Correctness";
			if (mlp4_bench_enabled())
				f += ":Mlp4.TFLOPs_Hopper";
		}
		if (!f.empty()) GTEST_FLAG_SET(filter, f);
	}
	return RUN_ALL_TESTS();
}
