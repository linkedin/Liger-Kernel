// ═══════════════════════════════════════════════════════════════════
// Numerical-correctness tests for the single-tile MLP1 device functions:
//   * mlp1_fused_{producer,consumer}        → Z = SiLU(B@X) · (C@X)
//   * mlp1_fused_act_{producer,consumer}    → U' = V·silu'(U),
//                                              V' = silu(U),
//                                              Z  = silu(U)·V
//
// Self-contained (no torch, no nvshmem): each TEST builds its own inputs
// on the host, drives a stand-alone launcher kernel modelled on
// src/.../moe/mlp1.cu, and compares the device output against an fp32 CPU
// reference computed from the *same bf16-rounded* inputs. The only error
// source is bf16 input/output rounding (fast_silu is exact), so a tight
// relative tolerance holds.
//
// Exercises the mlp1 consumers on BOTH architectures, one TEST per kernel,
// AUTO-GATED to the running GPU so the output stays clean (only the matching
// path's results are printed):
//   * sm_100 (Blackwell) → Compute=100 / UMMA  (Traits::MainloopPipelineUmma)
//   * sm_90  (Hopper)    → Compute=90  / WGMMA (Traits::MainloopPipeline)
// Both paths share the same shapes, cpu_reference and tolerances (run_fused /
// run_act are templated only on Compute), so neither arch is held to a looser
// bar. The non-matching path is still compiled — the Compute=100 body is gated
// on __CUDA_ARCH__>=1000 and the Compute=90 launcher call on __CUDA_ARCH__<1000
// (both trap otherwise) — so one source builds cleanly for sm_90a and sm_100a.
// ═══════════════════════════════════════════════════════════════════

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp1_fused.cuh"
#include "mlp1_fused_act.cuh"

using namespace cute;
using liger::Mlp1Traits;
using liger::Mlp1FusedSmem;
using liger::Mlp1FusedActSmem;
using Element = cutlass::bfloat16_t;

// Shape/pipeline config. TileM=128 exercises the cooperative M-split. The act
// variant uses fewer stages to keep its three extra store buffers under the
// 228 KiB smem cap.
using TraitsFused = Mlp1Traits<Element, /*TileM=*/128, /*TileN=*/128,
                               /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;
using TraitsAct   = Mlp1Traits<Element, /*TileM=*/128, /*TileN=*/128,
                               /*TileK=*/64, /*Stages=*/3, /*EpiChunkN=*/32>;

#define CUDA_OK(expr)                                                       \
	do {                                                                    \
		cudaError_t _e = (expr);                                            \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)

// ═══════════════════════════════════════════════════════════════════
// Stand-alone launcher kernels (host-driven outer M-tile loop, single
// fused X+W1+W2 TMA pipe), mirroring src/.../moe/mlp1.cu.
// ═══════════════════════════════════════════════════════════════════

// Mainloop pipeline type for a Compute path: Hopper (90) uses the plain TMA
// pipeline; Blackwell (100) uses the UMMA-aware TMA pipeline. Params /
// PipelineState / SharedStorage alias across the two, so the launcher, producer
// and consumer drive either one unchanged.
template <typename Traits, int Compute>
using MainloopPipelineFor = std::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

template <typename Traits, int Compute>
struct Mlp1FusedKernelSmem {
	Mlp1FusedSmem<Traits> tile;
	typename MainloopPipelineFor<Traits, Compute>::SharedStorage pipe_storage;
};

template <typename Traits, int Compute>
struct Mlp1ActKernelSmem {
	Mlp1FusedActSmem<Traits> tile;
	typename MainloopPipelineFor<Traits, Compute>::SharedStorage pipe_storage;
};

template <typename Traits, int Compute, typename TmaLoadX, typename TmaLoadW, typename TmaStoreZ>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp1_fused_test_kernel(
		__grid_constant__ TmaLoadX const tma_load_x,
		__grid_constant__ TmaLoadW const tma_load_b,
		__grid_constant__ TmaLoadW const tma_load_c,
		__grid_constant__ TmaStoreZ const tma_store_z,
		const int* expert_ids,
		int num_tokens, int hidden_dim, int total_n_rows,
		int num_m_tiles, int num_n_tiles) {

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Mlp1FusedKernelSmem<Traits, Compute>*>(raw_smem);

	using Pipeline  = MainloopPipelineFor<Traits, Compute>;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int num_k_tiles = hidden_dim / Traits::TileK;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return liger::mlp1_make_pipe_umma<Traits>(smem.pipe_storage);
		else
			return liger::mlp1_make_pipe<Traits>(smem.pipe_storage);
	}();
	__syncthreads();

	PipeState prod_state = cutlass::make_producer_start_state<Pipeline>();
	PipeState cons_state;

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
		int expert = expert_ids[m];
		int expert_n_offset = expert * num_n_tiles;
		if (is_producer) {
			liger::mlp1_fused_producer<Traits>(
				pipe, prod_state, smem.tile,
				tma_load_x, tma_load_b, tma_load_c,
				m, expert_n_offset, num_tokens, hidden_dim, total_n_rows,
				num_n_tiles, num_k_tiles);
		} else if (is_consumer) {
			if constexpr (Compute == 100) {
				liger::mlp1_fused_consumer<Traits, 100>(
					pipe, cons_state, smem.tile, tma_store_z,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles);
			} else {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
				liger::mlp1_fused_consumer<Traits, 90>(
					pipe, cons_state, smem.tile, tma_store_z,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles);
#else
				__trap();  // Compute=90 WGMMA body is not compiled for sm_100a
#endif
			}
		}
	}
	__syncthreads();
}

template <typename Traits, int Compute, typename TmaLoadX, typename TmaLoadW, typename TmaStore>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp1_act_test_kernel(
		__grid_constant__ TmaLoadX const tma_load_x,
		__grid_constant__ TmaLoadW const tma_load_b,
		__grid_constant__ TmaLoadW const tma_load_c,
		__grid_constant__ TmaStore const tma_store_du,
		__grid_constant__ TmaStore const tma_store_dv,
		__grid_constant__ TmaStore const tma_store_z,
		const int* expert_ids,
		int num_tokens, int hidden_dim, int total_n_rows,
		int num_m_tiles, int num_n_tiles) {

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Mlp1ActKernelSmem<Traits, Compute>*>(raw_smem);

	using Pipeline  = MainloopPipelineFor<Traits, Compute>;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int num_k_tiles = hidden_dim / Traits::TileK;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return liger::mlp1_make_pipe_umma<Traits>(smem.pipe_storage);
		else
			return liger::mlp1_make_pipe<Traits>(smem.pipe_storage);
	}();
	__syncthreads();

	PipeState prod_state = cutlass::make_producer_start_state<Pipeline>();
	PipeState cons_state;

	// Single block per M-tile here (gridDim.y == 1), so split_idx/num_splits
	// degenerate to "this block owns every n-tile".
	int split_idx  = blockIdx.y;
	int num_splits = gridDim.y;

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
		int expert = expert_ids[m];
		int expert_n_offset = expert * num_n_tiles;
		if (is_producer) {
			liger::mlp1_fused_act_producer<Traits>(
				pipe, prod_state, smem.tile,
				tma_load_x, tma_load_b, tma_load_c,
				m, expert_n_offset, num_tokens, hidden_dim, total_n_rows,
				num_n_tiles, num_k_tiles, split_idx, num_splits);
		} else if (is_consumer) {
			if constexpr (Compute == 100) {
				liger::mlp1_fused_act_consumer<Traits, 100>(
					pipe, cons_state, smem.tile,
					tma_store_du, tma_store_dv, tma_store_z,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
			} else {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
				liger::mlp1_fused_act_consumer<Traits, 90>(
					pipe, cons_state, smem.tile,
					tma_store_du, tma_store_dv, tma_store_z,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
#else
				__trap();  // Compute=90 WGMMA body is not compiled for sm_100a
#endif
			}
		}
	}
	__syncthreads();
}

// ═══════════════════════════════════════════════════════════════════
// Host helpers
// ═══════════════════════════════════════════════════════════════════

struct Mlp1Shape {
	int num_tokens;        // multiple of TileM (no token padding in the test)
	int hidden_dim;        // multiple of TileK
	int intermediate_dim;  // multiple of TileN
	int num_experts;
};

// A device buffer of bf16 filled from host floats (rounded to bf16).
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

// Round a float through bf16 (so the CPU reference sees the exact same
// operands the kernel does).
static inline float bf16_round(float x) { return float(Element(x)); }

// Aggregate error metrics between a device output and an fp32 reference.
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

// CPU reference: U = X·B[e]^T, V = X·C[e]^T per M-tile, then the requested
// outputs. Inputs are bf16-rounded; accumulation is fp32.
struct RefOutputs {
	std::vector<float> Z;   // silu(U)·V
	std::vector<float> Up;  // V·silu'(U)   (act only)
	std::vector<float> Vp;  // silu(U)      (act only)
};

static RefOutputs cpu_reference(
		const std::vector<float>& X,      // [tokens, hidden] bf16-rounded
		const std::vector<float>& B,      // [E, inter, hidden]
		const std::vector<float>& C,
		const std::vector<int>&   expert_ids,
		const Mlp1Shape& s, int TileM, bool with_act) {

	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim;
	RefOutputs out;
	out.Z.assign((size_t)T * I, 0.f);
	if (with_act) { out.Up.assign((size_t)T * I, 0.f); out.Vp.assign((size_t)T * I, 0.f); }

	int num_m_tiles = (T + TileM - 1) / TileM;
	for (int m = 0; m < num_m_tiles; ++m) {
		int e = expert_ids[m];
		int r0 = m * TileM, r1 = std::min(r0 + TileM, T);
		for (int r = r0; r < r1; ++r) {
			for (int j = 0; j < I; ++j) {
				float u = 0.f, v = 0.f;
				const float* xr = &X[(size_t)r * H];
				const float* bj = &B[((size_t)e * I + j) * H];
				const float* cj = &C[((size_t)e * I + j) * H];
				for (int k = 0; k < H; ++k) { u += xr[k] * bj[k]; v += xr[k] * cj[k]; }
				float sig    = 1.0f / (1.0f + std::exp(-u));
				float silu   = u * sig;
				out.Z[(size_t)r * I + j] = silu * v;
				if (with_act) {
					float silu_d = sig + silu * (1.0f - sig);
					out.Up[(size_t)r * I + j] = v * silu_d;
					out.Vp[(size_t)r * I + j] = silu;
				}
			}
		}
	}
	return out;
}

// Build host inputs (bf16-rounded floats) + device buffers shared by both
// variants. expert_ids[m] = m % num_experts.
struct Inputs {
	std::vector<float> X, B, C;           // bf16-rounded host copies
	std::vector<int>   expert_ids;
	DevBf16 dX, dB, dC;
	int* d_expert_ids = nullptr;
	int num_m_tiles, num_n_tiles, total_n_rows;
	~Inputs() { if (d_expert_ids) cudaFree(d_expert_ids); }
};

template <typename Traits>
static void make_inputs(const Mlp1Shape& s, Inputs& in, unsigned seed) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> nd(0.f, 1.f);
	auto fill = [&](std::vector<float>& v, size_t n) {
		v.resize(n);
		for (size_t i = 0; i < n; ++i) v[i] = bf16_round(nd(rng));
	};
	fill(in.X, (size_t)s.num_tokens * s.hidden_dim);
	fill(in.B, (size_t)s.num_experts * s.intermediate_dim * s.hidden_dim);
	fill(in.C, (size_t)s.num_experts * s.intermediate_dim * s.hidden_dim);

	in.num_m_tiles  = (s.num_tokens + Traits::TileM - 1) / Traits::TileM;
	in.num_n_tiles  = s.intermediate_dim / Traits::TileN;
	in.total_n_rows = s.num_experts * s.intermediate_dim;

	in.expert_ids.resize(in.num_m_tiles);
	for (int m = 0; m < in.num_m_tiles; ++m) in.expert_ids[m] = m % s.num_experts;

	upload_bf16(in.dX, in.X);
	upload_bf16(in.dB, in.B);
	upload_bf16(in.dC, in.C);
	cudaMalloc(&in.d_expert_ids, in.num_m_tiles * sizeof(int));
	cudaMemcpy(in.d_expert_ids, in.expert_ids.data(),
		in.num_m_tiles * sizeof(int), cudaMemcpyHostToDevice);
}

// Download a bf16 device output (padded to num_m_tiles*TileM rows) and return
// the first num_tokens rows as floats.
static std::vector<float> download_rows(const Element* d, int padded_tokens,
                                        int num_tokens, int inter) {
	std::vector<Element> hb((size_t)padded_tokens * inter);
	cudaMemcpy(hb.data(), d, hb.size() * sizeof(Element), cudaMemcpyDeviceToHost);
	std::vector<float> out((size_t)num_tokens * inter);
	for (size_t i = 0; i < out.size(); ++i) out[i] = float(hb[i]);
	return out;
}

// ═══════════════════════════════════════════════════════════════════
// Variant runners
// ═══════════════════════════════════════════════════════════════════

template <int Compute>
static void run_fused(const Mlp1Shape& s) {
	using Traits = TraitsFused;
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234);

	int padded = in.num_m_tiles * Traits::TileM;
	Element* dZ = nullptr;
	cudaMalloc(&dZ, (size_t)padded * s.intermediate_dim * sizeof(Element));
	cudaMemset(dZ, 0, (size_t)padded * s.intermediate_dim * sizeof(Element));

	// ── TMA descriptors ──
	auto tX = make_tensor(make_gmem_ptr(in.dX.ptr),
		make_shape(s.num_tokens, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
		make_shape(in.total_n_rows, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
		make_shape(in.total_n_rows, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto tZ = make_tensor(make_gmem_ptr(dZ),
		make_shape(padded, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));

	auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, tX, typename Traits::SmemLayoutX_1{});
	auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, tB, typename Traits::SmemLayoutW_1{});
	auto tma_c = make_tma_copy(SM90_TMA_LOAD{}, tC, typename Traits::SmemLayoutW_1{});
	auto tma_z = make_tma_copy(SM90_TMA_STORE{}, tZ, typename Traits::SmemLayoutStoreSlot{});

	size_t smem_size = sizeof(Mlp1FusedKernelSmem<Traits, Compute>);
	auto kernel = mlp1_fused_test_kernel<Traits, Compute,
		decltype(tma_x), decltype(tma_b), decltype(tma_z)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	dim3 grid(in.num_m_tiles, 1);
	kernel<<<grid, Traits::NumThreads, smem_size>>>(
		tma_x, tma_b, tma_c, tma_z, in.d_expert_ids,
		s.num_tokens, s.hidden_dim, in.total_n_rows, in.num_m_tiles, in.num_n_tiles);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto Z = download_rows(dZ, padded, s.num_tokens, s.intermediate_dim);
	cudaFree(dZ);

	auto ref = cpu_reference(in.X, in.B, in.C, in.expert_ids, s,
		Traits::TileM, /*with_act=*/false);
	auto e = compare(Z, ref.Z);
	printf("[fused C=%-3d T=%d H=%d I=%d E=%d] mean_rel=%.3f%% max_rel=%.3f%% max_abs=%.3g\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		e.mean_rel * 100, e.max_rel * 100, e.max_abs);

	EXPECT_LT(e.mean_rel, 0.01f);   // mean within 1%
	EXPECT_LT(e.max_rel,  0.05f);   // every element within 5% (bf16 output rounding)
}

template <int Compute>
static void run_act(const Mlp1Shape& s) {
	using Traits = TraitsAct;
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/5678);

	int padded = in.num_m_tiles * Traits::TileM;
	size_t obytes = (size_t)padded * s.intermediate_dim * sizeof(Element);
	Element *dU = nullptr, *dV = nullptr, *dZ = nullptr;
	cudaMalloc(&dU, obytes); cudaMalloc(&dV, obytes); cudaMalloc(&dZ, obytes);
	cudaMemset(dU, 0, obytes); cudaMemset(dV, 0, obytes); cudaMemset(dZ, 0, obytes);

	auto tX = make_tensor(make_gmem_ptr(in.dX.ptr),
		make_shape(s.num_tokens, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
		make_shape(in.total_n_rows, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
		make_shape(in.total_n_rows, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));
	auto mkZ = [&](Element* p) {
		return make_tensor(make_gmem_ptr(p),
			make_shape(padded, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	};
	auto tma_x  = make_tma_copy(SM90_TMA_LOAD{},  tX, typename Traits::SmemLayoutX_1{});
	auto tma_b  = make_tma_copy(SM90_TMA_LOAD{},  tB, typename Traits::SmemLayoutW_1{});
	auto tma_c  = make_tma_copy(SM90_TMA_LOAD{},  tC, typename Traits::SmemLayoutW_1{});
	auto tma_du = make_tma_copy(SM90_TMA_STORE{}, mkZ(dU), typename Traits::SmemLayoutStoreSlot{});
	auto tma_dv = make_tma_copy(SM90_TMA_STORE{}, mkZ(dV), typename Traits::SmemLayoutStoreSlot{});
	auto tma_z  = make_tma_copy(SM90_TMA_STORE{}, mkZ(dZ), typename Traits::SmemLayoutStoreSlot{});

	size_t smem_size = sizeof(Mlp1ActKernelSmem<Traits, Compute>);
	auto kernel = mlp1_act_test_kernel<Traits, Compute,
		decltype(tma_x), decltype(tma_b), decltype(tma_z)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	dim3 grid(in.num_m_tiles, 1);
	kernel<<<grid, Traits::NumThreads, smem_size>>>(
		tma_x, tma_b, tma_c, tma_du, tma_dv, tma_z, in.d_expert_ids,
		s.num_tokens, s.hidden_dim, in.total_n_rows, in.num_m_tiles, in.num_n_tiles);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto U = download_rows(dU, padded, s.num_tokens, s.intermediate_dim);
	auto V = download_rows(dV, padded, s.num_tokens, s.intermediate_dim);
	auto Z = download_rows(dZ, padded, s.num_tokens, s.intermediate_dim);
	cudaFree(dU); cudaFree(dV); cudaFree(dZ);

	auto ref = cpu_reference(in.X, in.B, in.C, in.expert_ids, s,
		Traits::TileM, /*with_act=*/true);
	auto eU = compare(U, ref.Up);
	auto eV = compare(V, ref.Vp);
	auto eZ = compare(Z, ref.Z);
	printf("[act   C=%-3d T=%d H=%d I=%d E=%d] "
		"U' mean_rel=%.3f%% / V' %.3f%% / Z %.3f%%  (max_rel %.2f%%/%.2f%%/%.2f%%)\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		eU.mean_rel * 100, eV.mean_rel * 100, eZ.mean_rel * 100,
		eU.max_rel * 100, eV.max_rel * 100, eZ.max_rel * 100);

	for (auto* e : {&eU, &eV, &eZ}) {
		EXPECT_LT(e->mean_rel, 0.01f);
		EXPECT_LT(e->max_rel,  0.05f);
	}
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

// Skip on non-Blackwell devices: the Compute=100 kernels need sm_100 (UMMA/
// tcgen05). Built for sm_100a, the Compute=100 body is only instantiated here.
static bool blackwell_available() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return false;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
	return p.major == 10;
}

// Skip on non-Hopper devices: the Compute=90 kernels need sm_90 (WGMMA). Built
// for sm_90a, the Compute=90 body is instantiated with real WGMMA; the
// Compute=100 body traps (never launched — its test is blackwell_available()-
// guarded). This is the standard "detect capability at runtime, else SKIP"
// pattern for GPU-arch-specific unit tests.
static bool hopper_available() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return false;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
	return p.major == 9;
}

static const std::vector<Mlp1Shape> kShapes = {
	{128, 256,  128, 1},   // single M-tile, single N-tile, single expert
	{128, 512,  256, 1},   // deeper K, two N-tiles
	{256, 256,  256, 2},   // two M-tiles across two experts
	{384, 256,  128, 3},   // three M-tiles, one expert each
};

// ── Blackwell (Compute=100 / UMMA) — requires an sm_100 GPU at runtime ──
TEST(Mlp1Fused, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapes) run_fused<100>(s);
}

TEST(Mlp1FusedAct, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapes) run_act<100>(s);
}

// ── Hopper (Compute=90 / WGMMA) — requires an sm_90 GPU at runtime ──
// Same shapes, same cpu_reference, same tolerances as the Blackwell tests
// (run_fused/run_act are shared, templated only on Compute): the Hopper path is
// held to the identical bar — no relaxed thresholds, no bias.
TEST(Mlp1FusedSm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes) run_fused<90>(s);
}

TEST(Mlp1FusedActSm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes) run_act<90>(s);
}
