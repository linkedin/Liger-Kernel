// ═══════════════════════════════════════════════════════════════════
// Numerical-correctness tests for the single-tile MLP5 device functions:
//   * mlp5_fused_{producer,consumer}  →  dX = dU·B + dV·C  (backward input grad)
//
// Two GEMMs fused into one continuous 2·num_k_tiles k-loop, accumulated into
// ONE accumulator: phase 1 (k<K) does acc += dU·B, phase 2 (k≥K) does
// acc += dV·C. Operands B/C are consumed MN-major (column-major weight view,
// like mlp2_t). The output dX is fully written (no zero-init).
//
// Self-contained (no torch, no nvshmem): each TEST builds its own inputs on
// the host, drives a stand-alone 2D-grid launcher kernel (grid.x = m-tiles,
// grid.y = N-split), and compares against an fp32 CPU reference computed from
// the *same bf16-rounded* inputs. The only error source is bf16 input/output
// rounding, so a tight relative tolerance holds.
//
// Exercises the mlp5 consumers on BOTH architectures, AUTO-GATED to the running
// GPU so the output stays clean:
//   * sm_100 (Blackwell) → Compute=100 / UMMA  (Traits::MainloopPipelineUmma)
//   * sm_90  (Hopper)    → Compute=90  / WGMMA (Traits::MainloopPipeline)
// Both paths share shapes, cpu_reference and tolerances (run5 is templated only
// on Compute). The non-matching path is still compiled — the Compute=100 body
// is gated on __CUDA_ARCH__>=1000 and the Compute=90 launcher call on
// __CUDA_ARCH__<1000 (both trap otherwise) — so one source builds cleanly for
// sm_90a and sm_100a.
//
// Two tiny single-tile DIAGNOSTIC tests isolate the cross-phase accumulate:
//   * Phase1_C0 (C=0 → dX = dU·B only)  — exercises phase 1 + accumulate clear.
//   * Phase2_B0 (B=0 → dX = dV·C only)  — exercises phase 2 accumulate.
// If an isolated case passes but the combined case fails → the accumulate bit
// at the phase boundary is wrong. If an isolated case fails structurally → the
// MN-major operand is wrong.
// ═══════════════════════════════════════════════════════════════════

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp5_fused.cuh"

using namespace cute;
using liger::Mlp5Traits;
using liger::Mlp5Smem;
using Element = cutlass::bfloat16_t;

// Shape/pipeline config — mlp5's default: TileM=128 (cooperative M-split),
// TileN=256, EpiChunkN=64 (the widest ported epilogue → TmemLoadOp<64> =
// SM100_TMEM_LOAD_32dp32b64x on the Blackwell epilogue).
using TraitsFused = Mlp5Traits<Element, /*TileM=*/128, /*TileN=*/256,
                               /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;

#define CUDA_OK(expr)                                                       \
	do {                                                                    \
		cudaError_t _e = (expr);                                            \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)

// ═══════════════════════════════════════════════════════════════════
// Stand-alone 2D-grid launcher kernel (host-driven outer M-tile loop over
// grid.x; N-split over grid.y). Single fused dU/dV + B/C TMA pipe per k-step.
// CTAs sharing blockIdx.x (same m, different n-split) reload the identical
// dU/dV tiles → cross-CTA L2 multicast between (m,0) and (m,1).
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, int Compute>
using MainloopPipelineFor = std::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

template <typename Traits, int Compute>
struct Mlp5FusedKernelSmem {
	Mlp5Smem<Traits> tile;
	typename MainloopPipelineFor<Traits, Compute>::SharedStorage pipe_storage;
};

template <typename Traits, int Compute, typename TmaLoadZ, typename TmaLoadW, typename TmaStoreDX>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp5_fused_test_kernel(
		__grid_constant__ TmaLoadZ const tma_load_du,
		__grid_constant__ TmaLoadZ const tma_load_dv,
		__grid_constant__ TmaLoadW const tma_load_b,
		__grid_constant__ TmaLoadW const tma_load_c,
		__grid_constant__ TmaStoreDX const tma_store_dx,
		const int* expert_for_m_block,
		int num_tokens, int hidden_dim, int intermediate_dim, int total_k_cols,
		int num_m_tiles, int num_n_tiles) {

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Mlp5FusedKernelSmem<Traits, Compute>*>(raw_smem);

	using Pipeline  = MainloopPipelineFor<Traits, Compute>;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int num_k_tiles = intermediate_dim / Traits::TileK;
	bool is_producer = (warp_id == 0);
	constexpr int kFirstConsumerWarp = (Compute == 100) ? 3 : 4;
	bool is_consumer = (warp_id >= kFirstConsumerWarp && warp_id <= 11);

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return liger::mlp5_make_pipe_umma<Traits>(smem.pipe_storage);
		else
			return liger::mlp5_make_pipe<Traits>(smem.pipe_storage);
	}();

	// TMEM allocation is once-per-CTA, NOT per m-tile. The mlp5 2D grid makes
	// each CTA rasterize several m-tiles (grid.x = num_sms/NSplit < num_m_tiles
	// for large shapes), so a per-tile tcgen05.alloc/relinquish would allocate
	// after the permit was relinquished → "phase invalid during alloc" trap.
	// Warp 3 allocs all accumulator stages here; the __syncthreads below
	// publishes smem.tile.tmem_base to every consumer warp. Freed after the loop.
	// Arch-guarded: tcgen05 PTX only exists on sm_100a (the Compute=100 kernel
	// is still *instantiated* on sm_90a — where its consumer traps — so the
	// alloc must be compiled out there).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	cute::TMEM::Allocator1Sm tmem_alloc{};
	if constexpr (Compute == 100) {
		constexpr int kTmemColumns = Traits::AccStages * Traits::TileN;
		if (warp_id == 3) {
			tmem_alloc.allocate(kTmemColumns, &smem.tile.tmem_base);
			__syncwarp();
		}
	}
#endif
	__syncthreads();

	PipeState prod_state = cutlass::make_producer_start_state<Pipeline>();
	PipeState cons_state;

	const int split_idx  = (int)blockIdx.y;
	const int num_splits = (int)gridDim.y;

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
		int expert = expert_for_m_block[m];
		int expert_k_offset = expert * num_k_tiles;
		if (is_producer) {
			liger::mlp5_fused_producer<Traits, Compute == 90>(
				pipe, prod_state, smem.tile,
				tma_load_du, tma_load_dv, tma_load_b, tma_load_c,
				m, (Compute == 90) ? expert : expert_k_offset,
				num_tokens, hidden_dim, intermediate_dim,
				total_k_cols / intermediate_dim, total_k_cols,
				num_n_tiles, num_k_tiles, split_idx, num_splits);
		} else if (is_consumer) {
			if constexpr (Compute == 100) {
				liger::mlp5_fused_consumer<Traits, 100>(
					pipe, cons_state, smem.tile, tma_store_dx,
					m, hidden_dim, num_m_tiles, num_n_tiles, num_k_tiles,
					split_idx, num_splits);
			} else {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
				liger::mlp5_fused_consumer<Traits, 90>(
					pipe, cons_state, smem.tile, tma_store_dx,
					m, hidden_dim, num_m_tiles, num_n_tiles, num_k_tiles,
					split_idx, num_splits);
#else
				__trap();  // Compute=90 WGMMA body is not compiled for sm_100a
#endif
			}
		}
	}
	__syncthreads();

	// Free the CTA's TMEM allocation once, after every m-tile is drained.
	// release_allocation_lock (relinquish permit) + dealloc are warp-synchronous;
	// issue from the same warp that allocated, not one elected thread.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	if constexpr (Compute == 100) {
		constexpr int kTmemColumns = Traits::AccStages * Traits::TileN;
		if (warp_id == 3) {
			tmem_alloc.release_allocation_lock();
			tmem_alloc.free(smem.tile.tmem_base, kTmemColumns);
		}
	}
#endif
}

// ═══════════════════════════════════════════════════════════════════
// Host helpers
// ═══════════════════════════════════════════════════════════════════

struct Mlp5Shape {
	int num_tokens;        // T — multiple of TileM
	int hidden_dim;        // H — N axis (output width), multiple of TileN
	int intermediate_dim;  // I — K axis (contraction), multiple of TileK
	int num_experts;       // E
};

enum ZeroMode { ZM_NONE = 0, ZM_C0 = 1, ZM_B0 = 2 };  // diagnostics

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

// CPU reference: dX[t,h] = Σ_i dU[t,i]·B[e,i,h] + Σ_i dV[t,i]·C[e,i,h] per
// M-tile (expert e = expert_for_m_block[m]). B and C stored [E, I, H]
// row-major (matching the MN-major TMA view (H, E·I) stride (1, H)). Inputs
// are bf16-rounded; accumulation is fp32.
static std::vector<float> cpu_reference(
		const std::vector<float>& dU,     // [T, I] bf16-rounded
		const std::vector<float>& dV,     // [T, I]
		const std::vector<float>& B,      // [E, I, H]
		const std::vector<float>& C,      // [E, I, H]
		const std::vector<int>&   expert_for_m_block,
		const Mlp5Shape& s, int TileM) {

	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim;
	std::vector<float> dX((size_t)T * H, 0.f);

	int num_m_tiles = (T + TileM - 1) / TileM;
	for (int m = 0; m < num_m_tiles; ++m) {
		int e = expert_for_m_block[m];
		int r0 = m * TileM, r1 = std::min(r0 + TileM, T);
		for (int r = r0; r < r1; ++r) {
			const float* dur = &dU[(size_t)r * I];
			const float* dvr = &dV[(size_t)r * I];
			for (int h = 0; h < H; ++h) {
				float acc = 0.f;
				for (int i = 0; i < I; ++i) {
					float be = B[((size_t)e * I + i) * H + h];
					float ce = C[((size_t)e * I + i) * H + h];
					acc += dur[i] * be + dvr[i] * ce;
				}
				dX[(size_t)r * H + h] = acc;
			}
		}
	}
	return dX;
}

struct Inputs {
	std::vector<float> dU, dV, B, C;      // bf16-rounded host copies
	std::vector<int>   expert_for_m_block;
	DevBf16 dDU, dDV, dB, dC;
	int* d_expert = nullptr;
	int num_m_tiles, num_n_tiles, num_k_tiles, total_k_cols;
	~Inputs() { if (d_expert) cudaFree(d_expert); }
};

template <typename Traits>
static void make_inputs(const Mlp5Shape& s, Inputs& in, unsigned seed, ZeroMode zm) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> nd(0.f, 1.f);
	auto fill = [&](std::vector<float>& v, size_t n, bool zero) {
		v.resize(n);
		for (size_t i = 0; i < n; ++i) v[i] = zero ? 0.f : bf16_round(nd(rng));
	};
	fill(in.dU, (size_t)s.num_tokens * s.intermediate_dim, false);
	fill(in.dV, (size_t)s.num_tokens * s.intermediate_dim, false);
	fill(in.B,  (size_t)s.num_experts * s.intermediate_dim * s.hidden_dim, zm == ZM_B0);
	fill(in.C,  (size_t)s.num_experts * s.intermediate_dim * s.hidden_dim, zm == ZM_C0);

	in.num_m_tiles  = (s.num_tokens + Traits::TileM - 1) / Traits::TileM;
	in.num_n_tiles  = s.hidden_dim / Traits::TileN;
	in.num_k_tiles  = s.intermediate_dim / Traits::TileK;
	in.total_k_cols = s.num_experts * s.intermediate_dim;

	in.expert_for_m_block.resize(in.num_m_tiles);
	for (int m = 0; m < in.num_m_tiles; ++m) in.expert_for_m_block[m] = m % s.num_experts;

	upload_bf16(in.dDU, in.dU);
	upload_bf16(in.dDV, in.dV);
	upload_bf16(in.dB,  in.B);
	upload_bf16(in.dC,  in.C);
	cudaMalloc(&in.d_expert, in.num_m_tiles * sizeof(int));
	cudaMemcpy(in.d_expert, in.expert_for_m_block.data(),
		in.num_m_tiles * sizeof(int), cudaMemcpyHostToDevice);
}

static std::vector<float> download_rows(const Element* d, int padded_tokens,
                                        int num_tokens, int width) {
	std::vector<Element> hb((size_t)padded_tokens * width);
	cudaMemcpy(hb.data(), d, hb.size() * sizeof(Element), cudaMemcpyDeviceToHost);
	std::vector<float> out((size_t)num_tokens * width);
	for (size_t i = 0; i < out.size(); ++i) out[i] = float(hb[i]);
	return out;
}

// ═══════════════════════════════════════════════════════════════════
// Variant runner
// ═══════════════════════════════════════════════════════════════════
//
// dU/dV are row-major [T, I] (K-major operand A). B/C use the MN-major
// (column-major) view: logical shape (H, E·I) with stride (1, H) so the H (=N)
// axis is contiguous, matching Mlp5Traits::SmemLayoutW_1. dX is row-major
// [padded_T, H].

template <int Compute>
static void run5_once(const Mlp5Shape& s, Inputs& in, int num_splits,
                      bool verbose, const char* tag, ErrStats* out) {
	using Traits = TraitsFused;

	int padded = in.num_m_tiles * Traits::TileM;
	Element* dX = nullptr;
	cudaMalloc(&dX, (size_t)padded * s.hidden_dim * sizeof(Element));
	cudaMemset(dX, 0, (size_t)padded * s.hidden_dim * sizeof(Element));

	// dU/dV: row-major [T, I] (K-major operand A).
	auto tDU = make_tensor(make_gmem_ptr(in.dDU.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	auto tDV = make_tensor(make_gmem_ptr(in.dDV.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	// dX: row-major [padded_T, H].
	auto tDX = make_tensor(make_gmem_ptr(dX),
		make_shape(padded, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));

	auto tma_du = make_tma_copy(SM90_TMA_LOAD{},  tDU, typename Traits::SmemLayoutZ_1{});
	auto tma_dv = make_tma_copy(SM90_TMA_LOAD{},  tDV, typename Traits::SmemLayoutZ_1{});
	auto tma_b = [&]() {
		if constexpr (Compute == 90) {
			auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(Int<1>{}, s.hidden_dim,
					s.intermediate_dim * s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tB, typename Traits::SmemLayoutW_1{});
		} else {
			auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
				make_shape(s.hidden_dim, in.total_k_cols),
				make_stride(Int<1>{}, s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tB, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_c = [&]() {
		if constexpr (Compute == 90) {
			auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(Int<1>{}, s.hidden_dim,
					s.intermediate_dim * s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tC, typename Traits::SmemLayoutW_1{});
		} else {
			auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
				make_shape(s.hidden_dim, in.total_k_cols),
				make_stride(Int<1>{}, s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tC, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_dx = make_tma_copy(SM90_TMA_STORE{}, tDX, typename Traits::SmemLayoutStore_1{});

	size_t smem_size = sizeof(Mlp5FusedKernelSmem<Traits, Compute>);
	auto kernel = mlp5_fused_test_kernel<Traits, Compute,
		decltype(tma_du), decltype(tma_b), decltype(tma_dx)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	dim3 grid(in.num_m_tiles, num_splits);
	kernel<<<grid, Traits::NumThreads, smem_size>>>(
		tma_du, tma_dv, tma_b, tma_c, tma_dx, in.d_expert,
		s.num_tokens, s.hidden_dim, s.intermediate_dim, in.total_k_cols,
		in.num_m_tiles, in.num_n_tiles);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto got = download_rows(dX, padded, s.num_tokens, s.hidden_dim);
	cudaFree(dX);

	auto ref = cpu_reference(in.dU, in.dV, in.B, in.C, in.expert_for_m_block, s, Traits::TileM);
	auto e = compare(got, ref);
	if (verbose)
		printf("[mlp5 C=%-3d %s T=%d H=%d I=%d E=%d splits=%d] "
		       "mean_rel=%.3f%% max_rel=%.3f%% max_abs=%.3g\n",
			Compute, tag, s.num_tokens, s.hidden_dim, s.intermediate_dim,
			s.num_experts, num_splits, e.mean_rel * 100, e.max_rel * 100, e.max_abs);
	*out = e;
}

// Full-shape correctness: run at num_splits=1 (single CTA per m) AND a 2-split
// (exercises the 2D grid + cross-CTA dU/dV multicast). Both must match ref.
template <int Compute>
static void run5(const Mlp5Shape& s) {
	Inputs in; make_inputs<TraitsFused>(s, in, /*seed=*/1234, ZM_NONE);

	ErrStats e1{};
	run5_once<Compute>(s, in, /*num_splits=*/1, /*verbose=*/true, "", &e1);
	EXPECT_LT(e1.mean_rel, 0.01f);
	EXPECT_LT(e1.max_rel,  0.05f);

	if (in.num_n_tiles >= 2) {
		ErrStats e2{};
		run5_once<Compute>(s, in, /*num_splits=*/2, /*verbose=*/true, "2Dgrid", &e2);
		EXPECT_LT(e2.mean_rel, 0.01f);
		EXPECT_LT(e2.max_rel,  0.05f);
	}
}

// Diagnostic: single-tile isolate of one phase (C=0 → dU·B, B=0 → dV·C).
template <int Compute>
static void run5_isolate(const Mlp5Shape& s, ZeroMode zm, const char* tag) {
	Inputs in; make_inputs<TraitsFused>(s, in, /*seed=*/777, zm);
	ErrStats e{};
	run5_once<Compute>(s, in, /*num_splits=*/1, /*verbose=*/true, tag, &e);
	EXPECT_LT(e.mean_rel, 0.01f) << tag << " mean_rel too high";
	EXPECT_LT(e.max_rel,  0.05f) << tag << " max_rel too high";
}

// ═══════════════════════════════════════════════════════════════════
// TFLOPS benchmark (opt-in via MLP5_BENCH env; timing-only, no CPU ref)
// ═══════════════════════════════════════════════════════════════════
//
//     TFLOPS = 4·T·H·I / median_kernel_seconds / 1e12   (two GEMMs)
//
// N-split sweep over grid.y (== the N-split identity), grid = (num_sms/NSplit,
// NSplit) to fill the SMs; the fused producer/consumer take split_idx/num_splits
// from blockIdx.y / gridDim.y, so no kernel change is needed. Reports peak.

struct BenchCfg { int warmup = 10; int iters = 50; };

static int sm_count() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return 0;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
	return p.multiProcessorCount;
}

// Divisors of num_n_tiles → balanced N-splits (each CTA gets equal n-tiles).
static std::vector<int> candidate_splits(int num_n_tiles) {
	std::vector<int> ds;
	for (int d = 1; d <= num_n_tiles; ++d)
		if (num_n_tiles % d == 0) ds.push_back(d);
	if (ds.empty()) ds.push_back(1);
	return ds;
}

static double median_ms(std::vector<float>& v) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	size_t n = v.size();
	return (n & 1) ? (double)v[n / 2] : 0.5 * ((double)v[n / 2 - 1] + (double)v[n / 2]);
}

static bool mlp5_bench_enabled() { return std::getenv("MLP5_BENCH") != nullptr; }

template <typename LaunchFn>
static double time_kernel_ms(const BenchCfg& cfg, LaunchFn&& launch) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	for (int i = 0; i < cfg.warmup; ++i) launch();
	if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
		ADD_FAILURE() << "bench warmup sync: " << cudaGetErrorString(e);
	std::vector<float> samples; samples.reserve(cfg.iters);
	for (int i = 0; i < cfg.iters; ++i) {
		cudaEventRecord(start);
		launch();
		cudaEventRecord(stop);
		if (cudaError_t e = cudaEventSynchronize(stop); e != cudaSuccess)
			ADD_FAILURE() << "bench event sync: " << cudaGetErrorString(e);
		float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
		samples.push_back(ms);
	}
	cudaEventDestroy(start); cudaEventDestroy(stop);
	return median_ms(samples);
}

// FLOPs = two GEMMs dX = dU·B + dV·C, 4·T·H·I; cast epilogue ignored.
static double tflops_of(const Mlp5Shape& s, double ms) {
	double flops = 4.0 * (double)s.num_tokens * (double)s.hidden_dim
	             * (double)s.intermediate_dim;
	return flops / (ms * 1e-3) / 1e12;
}

template <int Compute>
static void run5_bench(const Mlp5Shape& s, const BenchCfg& cfg) {
	using Traits = TraitsFused;
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234, ZM_NONE);

	int padded = in.num_m_tiles * Traits::TileM;
	Element* dX = nullptr;
	cudaMalloc(&dX, (size_t)padded * s.hidden_dim * sizeof(Element));
	cudaMemset(dX, 0, (size_t)padded * s.hidden_dim * sizeof(Element));

	auto tDU = make_tensor(make_gmem_ptr(in.dDU.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	auto tDV = make_tensor(make_gmem_ptr(in.dDV.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	auto tDX = make_tensor(make_gmem_ptr(dX),
		make_shape(padded, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));

	auto tma_du = make_tma_copy(SM90_TMA_LOAD{},  tDU, typename Traits::SmemLayoutZ_1{});
	auto tma_dv = make_tma_copy(SM90_TMA_LOAD{},  tDV, typename Traits::SmemLayoutZ_1{});
	auto tma_b = [&]() {
		if constexpr (Compute == 90) {
			auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(Int<1>{}, s.hidden_dim,
					s.intermediate_dim * s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tB, typename Traits::SmemLayoutW_1{});
		} else {
			auto tB = make_tensor(make_gmem_ptr(in.dB.ptr),
				make_shape(s.hidden_dim, in.total_k_cols),
				make_stride(Int<1>{}, s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tB, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_c = [&]() {
		if constexpr (Compute == 90) {
			auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(Int<1>{}, s.hidden_dim,
					s.intermediate_dim * s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tC, typename Traits::SmemLayoutW_1{});
		} else {
			auto tC = make_tensor(make_gmem_ptr(in.dC.ptr),
				make_shape(s.hidden_dim, in.total_k_cols),
				make_stride(Int<1>{}, s.hidden_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tC, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_dx = make_tma_copy(SM90_TMA_STORE{}, tDX, typename Traits::SmemLayoutStore_1{});

	size_t smem_size = sizeof(Mlp5FusedKernelSmem<Traits, Compute>);
	auto kernel = mlp5_fused_test_kernel<Traits, Compute,
		decltype(tma_du), decltype(tma_b), decltype(tma_dx)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	int nsm = sm_count();
	double best_tf = 0.0, best_ms = 0.0; int best_split = 1;
	for (int split : candidate_splits(in.num_n_tiles)) {
		int gx = std::max(1, nsm / split);
		dim3 grid(gx, split);
		auto launch = [&]() {
			kernel<<<grid, Traits::NumThreads, smem_size>>>(
				tma_du, tma_dv, tma_b, tma_c, tma_dx, in.d_expert,
				s.num_tokens, s.hidden_dim, s.intermediate_dim, in.total_k_cols,
				in.num_m_tiles, in.num_n_tiles);
		};
		launch(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
		double ms = time_kernel_ms(cfg, launch);
		double tf = tflops_of(s, ms);
		if (tf > best_tf) { best_tf = tf; best_ms = ms; best_split = split; }
	}
	printf("[mlp5-bench C=%-3d T=%-5d H=%d I=%d E=%d] "
	       "peak %8.2f TFLOPS @ %8.4f ms (splits=%-2d, grid=(%d,%d), %d SMs)\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		best_tf, best_ms, best_split, std::max(1, nsm / best_split), best_split, nsm);

	cudaFree(dX);
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

// Tiny single-tile shape for the phase-isolation diagnostics: one M-tile
// (128), one N-tile (256), one K-tile (64), one expert.
static const Mlp5Shape kTinyShape = {128, 256, 64, 1};

// Small correctness shapes. T multiple of TileM (128), H of TileN (256), I of
// TileK (64) → exact FLOP count, no padding.
static const std::vector<Mlp5Shape> kShapes = {
	{128, 256,  64, 1},   // single tile, single k
	{128, 512, 128, 1},   // two N-tiles, two K-tiles
	{256, 512, 256, 2},   // two M-tiles / two experts, deeper K
	{384, 256, 256, 3},   // three M-tiles, one expert each
};

// Large, GPU-saturating shapes for the TFLOPS benchmark. Realistic MoE dims
// (H=I=4096, E=8); T a multiple of TileM → no token padding, so 4·T·H·I exact.
static const std::vector<Mlp5Shape> kBenchShapes = {
	{ 2048, 4096, 4096, 8},
	{ 4096, 4096, 4096, 8},
	{ 8192, 4096, 4096, 8},
	{16384, 4096, 4096, 8},
};

// ── Diagnostics (run FIRST): tiny single-tile phase isolation. ──
TEST(Mlp5, Phase1_C0) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	run5_isolate<100>(kTinyShape, ZM_C0, "C0(dU*B)");
}

TEST(Mlp5, Phase2_B0) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	run5_isolate<100>(kTinyShape, ZM_B0, "B0(dV*C)");
}

// ── Blackwell (Compute=100 / UMMA) — requires an sm_100 GPU at runtime ──
TEST(Mlp5, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapes) run5<100>(s);
}

// ── Hopper (Compute=90 / WGMMA) — requires an sm_90 GPU at runtime ──
TEST(Mlp5Sm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes) run5<90>(s);
}

// ── TFLOPS benchmarks — opt-in via MLP5_BENCH=1. ──
TEST(Mlp5, TFLOPs_Blackwell) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (!mlp5_bench_enabled())  GTEST_SKIP() << "set MLP5_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run5_bench<100>(s, cfg);
}

TEST(Mlp5, TFLOPs_Hopper) {
	if (!hopper_available())   GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	if (!mlp5_bench_enabled()) GTEST_SKIP() << "set MLP5_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run5_bench<90>(s, cfg);
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
			f = "Mlp5.Phase1_C0:Mlp5.Phase2_B0:Mlp5.Correctness";
			if (mlp5_bench_enabled())
				f += ":Mlp5.TFLOPs_Blackwell";
		} else if (hopper_available()) {
			f = "Mlp5Sm90.Correctness";
			if (mlp5_bench_enabled())
				f += ":Mlp5.TFLOPs_Hopper";
		}
		if (!f.empty()) GTEST_FLAG_SET(filter, f);
	}
	return RUN_ALL_TESTS();
}
