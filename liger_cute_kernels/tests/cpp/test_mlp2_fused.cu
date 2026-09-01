// ═══════════════════════════════════════════════════════════════════
// Numerical-correctness tests for the single-tile MLP2 device functions:
//   * mlp2_fused_{producer,consumer}   →  Y = Z · Aᵀ  (down-projection)
//
// Self-contained (no torch, no nvshmem): each TEST builds its own inputs
// on the host, drives a stand-alone launcher kernel modelled on
// src/.../moe/mlp2.cu, and compares the device output against an fp32 CPU
// reference computed from the *same bf16-rounded* inputs. The only error
// source is bf16 input/output rounding, so a tight relative tolerance holds.
//
// Exercises the mlp2 consumers on BOTH architectures, AUTO-GATED to the
// running GPU so the output stays clean (only the matching path's results
// are printed):
//   * sm_100 (Blackwell) → Compute=100 / UMMA  (Traits::MainloopPipelineUmma)
//   * sm_90  (Hopper)    → Compute=90  / WGMMA (Traits::MainloopPipeline)
// Both paths share the same shapes, cpu_reference and tolerances, so neither
// arch is held to a looser bar. The
// non-matching path is still compiled — the Compute=100 body is gated on
// __CUDA_ARCH__>=1000 and the Compute=90 launcher call on __CUDA_ARCH__<1000
// (both trap otherwise) — so one source builds cleanly for sm_90a and sm_100a.
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
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp2_fused.cuh"

using namespace cute;
using liger::Mlp2Traits;
using liger::Mlp2FusedSmem;
using Element = cutlass::bfloat16_t;

// Shape/pipeline config. TileM=128 exercises the cooperative M-split.
// EpiChunkN=32 → TmemLoadOp<32> = SM100_TMEM_LOAD_32dp32b32x on the Blackwell
// epilogue (mlp2_fused's default epilogue chunk width).
using TraitsFused = Mlp2Traits<Element, /*TileM=*/128, /*TileN=*/128,
                               /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/32>;
using TraitsFused192 = Mlp2Traits<Element, /*TileM=*/128, /*TileN=*/192,
                                  /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;

#define CUDA_OK(expr)                                                       \
	do {                                                                    \
		cudaError_t _e = (expr);                                            \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)

// ═══════════════════════════════════════════════════════════════════
// Stand-alone launcher kernel (host-driven outer M-tile loop, single
// fused Z+A TMA pipe), mirroring src/.../moe/mlp2.cu.
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
struct Mlp2FusedKernelSmem {
	Mlp2FusedSmem<Traits> tile;
	typename MainloopPipelineFor<Traits, Compute>::SharedStorage pipe_storage;
};

template <typename Traits, int Compute, typename TmaLoadZ, typename TmaLoadA, typename TmaStoreY>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp2_fused_test_kernel(
		__grid_constant__ TmaLoadZ const tma_load_z,
		__grid_constant__ TmaLoadA const tma_load_a,
		__grid_constant__ TmaStoreY const tma_store_y,
		const int* expert_ids,
		int num_tokens, int intermediate_dim, int total_n_rows,
		int num_m_tiles, int num_n_tiles) {

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Mlp2FusedKernelSmem<Traits, Compute>*>(raw_smem);

	using Pipeline  = MainloopPipelineFor<Traits, Compute>;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int num_k_tiles = intermediate_dim / Traits::TileK;
	bool is_producer = (warp_id == 0);
	constexpr int kFirstConsumerWarp = (Compute == 100) ? 3 : 4;
	bool is_consumer = (warp_id >= kFirstConsumerWarp && warp_id <= 11);

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return liger::mlp2_make_pipe_umma<Traits>(smem.pipe_storage);
		else
			return liger::mlp2_make_pipe<Traits>(smem.pipe_storage);
	}();
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

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
		int expert = expert_ids[m];
		int expert_n_offset = expert * num_n_tiles;
		if (is_producer) {
			liger::mlp2_fused_producer<Traits, Compute == 90>(
				pipe, prod_state, smem.tile,
				tma_load_z, tma_load_a,
				m, (Compute == 90) ? expert : expert_n_offset, num_tokens,
				num_n_tiles * Traits::TileN, intermediate_dim,
				total_n_rows / (num_n_tiles * Traits::TileN), total_n_rows,
				num_n_tiles, num_k_tiles);
		} else if (is_consumer) {
			if constexpr (Compute == 100) {
				liger::mlp2_fused_consumer<Traits, 100>(
					pipe, cons_state, smem.tile, tma_store_y,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles);
			} else {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
				liger::mlp2_fused_consumer<Traits, 90>(
					pipe, cons_state, smem.tile, tma_store_y,
					m, num_n_tiles * Traits::TileN,
					num_m_tiles, num_n_tiles, num_k_tiles);
#else
				__trap();  // Compute=90 WGMMA body is not compiled for sm_100a
#endif
			}
		}
	}
	__syncthreads();
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

struct Mlp2Shape {
	int num_tokens;        // T — multiple of TileM (no token padding in the test)
	int hidden_dim;        // H — N axis (output width), multiple of TileN
	int intermediate_dim;  // I — K axis (contraction), multiple of TileK
	int num_experts;       // E
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

// CPU reference: Y = Z · A[e]ᵀ per M-tile (contract over I). Y[t,h] =
// Σ_i Z[t,i]·A[e,h,i]. Inputs are bf16-rounded; accumulation is fp32.
static std::vector<float> cpu_reference(
		const std::vector<float>& Z,      // [T, I] bf16-rounded
		const std::vector<float>& A,      // [E, H, I] bf16-rounded
		const std::vector<int>&   expert_ids,
		const Mlp2Shape& s, int TileM) {

	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim;
	std::vector<float> Y((size_t)T * H, 0.f);

	int num_m_tiles = (T + TileM - 1) / TileM;
	for (int m = 0; m < num_m_tiles; ++m) {
		int e = expert_ids[m];
		int r0 = m * TileM, r1 = std::min(r0 + TileM, T);
		for (int r = r0; r < r1; ++r) {
			const float* zr = &Z[(size_t)r * I];
			for (int h = 0; h < H; ++h) {
				const float* ah = &A[((size_t)e * H + h) * I];
				float acc = 0.f;
				for (int i = 0; i < I; ++i) acc += zr[i] * ah[i];
				Y[(size_t)r * H + h] = acc;
			}
		}
	}
	return Y;
}

// Build host inputs (bf16-rounded floats) + device buffers.
// expert_ids[m] = m % num_experts.
struct Inputs {
	std::vector<float> Z, A;              // bf16-rounded host copies
	std::vector<int>   expert_ids;
	DevBf16 dZ, dA;
	int* d_expert_ids = nullptr;
	int num_m_tiles, num_n_tiles, total_n_rows;
	~Inputs() { if (d_expert_ids) cudaFree(d_expert_ids); }
};

template <typename Traits>
static void make_inputs(const Mlp2Shape& s, Inputs& in, unsigned seed) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> nd(0.f, 1.f);
	auto fill = [&](std::vector<float>& v, size_t n) {
		v.resize(n);
		for (size_t i = 0; i < n; ++i) v[i] = bf16_round(nd(rng));
	};
	fill(in.Z, (size_t)s.num_tokens * s.intermediate_dim);
	fill(in.A, (size_t)s.num_experts * s.hidden_dim * s.intermediate_dim);

	in.num_m_tiles  = (s.num_tokens + Traits::TileM - 1) / Traits::TileM;
	in.num_n_tiles  = s.hidden_dim / Traits::TileN;
	in.total_n_rows = s.num_experts * s.hidden_dim;

	in.expert_ids.resize(in.num_m_tiles);
	for (int m = 0; m < in.num_m_tiles; ++m) in.expert_ids[m] = m % s.num_experts;

	upload_bf16(in.dZ, in.Z);
	upload_bf16(in.dA, in.A);
	cudaMalloc(&in.d_expert_ids, in.num_m_tiles * sizeof(int));
	cudaMemcpy(in.d_expert_ids, in.expert_ids.data(),
		in.num_m_tiles * sizeof(int), cudaMemcpyHostToDevice);
}

// Download a bf16 device output (padded to num_m_tiles*TileM rows) and return
// the first num_tokens rows as floats.
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

template <typename Traits, int Compute>
static void run_fused(const Mlp2Shape& s) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234);

	int padded = in.num_m_tiles * Traits::TileM;
	Element* dY = nullptr;
	cudaMalloc(&dY, (size_t)padded * s.hidden_dim * sizeof(Element));
	cudaMemset(dY, 0, (size_t)padded * s.hidden_dim * sizeof(Element));

	// ── TMA descriptors ──
	auto tZ = make_tensor(make_gmem_ptr(in.dZ.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	auto tY = make_tensor(make_gmem_ptr(dY),
		make_shape(padded, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));

	auto tma_z = make_tma_copy(SM90_TMA_LOAD{},  tZ, typename Traits::SmemLayoutZ_1{});
	auto tma_a = [&]() {
		if constexpr (Compute == 90) {
			auto tA = make_tensor(make_gmem_ptr(in.dA.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(s.intermediate_dim, Int<1>{},
					s.hidden_dim * s.intermediate_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tA, typename Traits::SmemLayoutW_1{});
		} else {
			auto tA = make_tensor(make_gmem_ptr(in.dA.ptr),
				make_shape(in.total_n_rows, s.intermediate_dim),
				make_stride(s.intermediate_dim, Int<1>{}));
			return make_tma_copy(SM90_TMA_LOAD{}, tA, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_y = make_tma_copy(SM90_TMA_STORE{}, tY, typename Traits::SmemLayoutStoreSlot{});

	size_t smem_size = sizeof(Mlp2FusedKernelSmem<Traits, Compute>);
	auto kernel = mlp2_fused_test_kernel<Traits, Compute,
		decltype(tma_z), decltype(tma_a), decltype(tma_y)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	dim3 grid(in.num_m_tiles, 1);
	kernel<<<grid, Traits::NumThreads, smem_size>>>(
		tma_z, tma_a, tma_y, in.d_expert_ids,
		s.num_tokens, s.intermediate_dim, in.total_n_rows, in.num_m_tiles, in.num_n_tiles);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto Y = download_rows(dY, padded, s.num_tokens, s.hidden_dim);
	cudaFree(dY);

	auto ref = cpu_reference(in.Z, in.A, in.expert_ids, s, Traits::TileM);
	auto e = compare(Y, ref);
	printf("[mlp2 C=%-3d T=%d H=%d I=%d E=%d] mean_rel=%.3f%% max_rel=%.3f%% max_abs=%.3g\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		e.mean_rel * 100, e.max_rel * 100, e.max_abs);

	EXPECT_LT(e.mean_rel, 0.01f);   // mean within 1%
	EXPECT_LT(e.max_rel,  0.05f);   // every element within 5% (bf16 output rounding)
}

// ═══════════════════════════════════════════════════════════════════
// TFLOPS benchmark (opt-in via MLP2_BENCH env; timing-only, no CPU ref)
// ═══════════════════════════════════════════════════════════════════
//
// Measures achieved throughput of the MLP2 fused consumer at large,
// GPU-saturating shapes. FLOPs are counted manually as the single GEMM
// Y = Z·Aᵀ (2·T·H·I, contracting over I); the cast epilogue is ignored
// (negligible). E-independent:
//
//     TFLOPS = 2·T·H·I / median_kernel_seconds / 1e12
//
// Timing uses CUDA events around each kernel launch (warm-up + repeat,
// median). No CPU reference is computed here — correctness is covered by
// the small-shape tests above. The grid is N-split (grid.y = num_splits)
// so small M-tile counts still fill the SMs; the fused producer/consumer
// fall back to blockIdx.y / gridDim.y for the split identity, so no kernel
// change is needed.

struct BenchCfg { int warmup = 10; int iters = 50; };

static int sm_count() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return 0;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
	return p.multiProcessorCount;
}

// Candidate N-splits to sweep: the divisors of num_n_tiles, so each CTA gets an
// equal number of n-tiles (balanced). The benchmark times every candidate and
// reports the peak — this removes any guesswork about the best launch shape and
// shows the effect of occupancy (small splits under-fill the SMs).
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

static bool mlp2_bench_enabled() { return std::getenv("MLP2_BENCH") != nullptr; }

// Time a launch closure: warm-up, then `iters` event-timed launches → median ms.
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

// FLOPs = one GEMM Y = Z·Aᵀ, 2·T·H·I; cast epilogue ignored.
static double tflops_of(const Mlp2Shape& s, double ms) {
	double flops = 2.0 * (double)s.num_tokens * (double)s.hidden_dim
	             * (double)s.intermediate_dim;
	return flops / (ms * 1e-3) / 1e12;
}

template <int Compute>
static void run_fused_bench(const Mlp2Shape& s, const BenchCfg& cfg) {
	using Traits = TraitsFused;
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234);

	int padded = in.num_m_tiles * Traits::TileM;
	Element* dY = nullptr;
	cudaMalloc(&dY, (size_t)padded * s.hidden_dim * sizeof(Element));
	cudaMemset(dY, 0, (size_t)padded * s.hidden_dim * sizeof(Element));

	auto tZ = make_tensor(make_gmem_ptr(in.dZ.ptr),
		make_shape(s.num_tokens, s.intermediate_dim), make_stride(s.intermediate_dim, Int<1>{}));
	auto tY = make_tensor(make_gmem_ptr(dY),
		make_shape(padded, s.hidden_dim), make_stride(s.hidden_dim, Int<1>{}));

	auto tma_z = make_tma_copy(SM90_TMA_LOAD{},  tZ, typename Traits::SmemLayoutZ_1{});
	auto tma_a = [&]() {
		if constexpr (Compute == 90) {
			auto tA = make_tensor(make_gmem_ptr(in.dA.ptr),
				make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
				make_stride(s.intermediate_dim, Int<1>{},
					s.hidden_dim * s.intermediate_dim));
			return make_tma_copy(SM90_TMA_LOAD{}, tA, typename Traits::SmemLayoutW_1{});
		} else {
			auto tA = make_tensor(make_gmem_ptr(in.dA.ptr),
				make_shape(in.total_n_rows, s.intermediate_dim),
				make_stride(s.intermediate_dim, Int<1>{}));
			return make_tma_copy(SM90_TMA_LOAD{}, tA, typename Traits::SmemLayoutW_1{});
		}
	}();
	auto tma_y = make_tma_copy(SM90_TMA_STORE{}, tY, typename Traits::SmemLayoutStoreSlot{});

	size_t smem_size = sizeof(Mlp2FusedKernelSmem<Traits, Compute>);
	auto kernel = mlp2_fused_test_kernel<Traits, Compute,
		decltype(tma_z), decltype(tma_a), decltype(tma_y)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	// Sweep candidate N-splits (only grid.y changes); report the peak.
	dim3 grid(in.num_m_tiles, 1);
	auto launch = [&]() {
		kernel<<<grid, Traits::NumThreads, smem_size>>>(
			tma_z, tma_a, tma_y, in.d_expert_ids,
			s.num_tokens, s.intermediate_dim, in.total_n_rows,
			in.num_m_tiles, in.num_n_tiles);
	};
	double best_tf = 0.0, best_ms = 0.0; int best_split = 1;
	for (int split : candidate_splits(in.num_n_tiles)) {
		grid.y = split;
		launch(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
		double ms = time_kernel_ms(cfg, launch);
		double tf = tflops_of(s, ms);
		if (tf > best_tf) { best_tf = tf; best_ms = ms; best_split = split; }
	}
	printf("[mlp2-bench C=%-3d T=%-5d H=%d I=%d E=%d] "
	       "peak %7.2f TFLOPS @ %7.4f ms (splits=%-2d, %4d CTAs / %d SMs)\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		best_tf, best_ms, best_split, in.num_m_tiles * best_split, sm_count());

	cudaFree(dY);
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
// guarded). Standard "detect capability at runtime, else SKIP" pattern.
static bool hopper_available() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return false;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
	return p.major == 9;
}

// Small correctness shapes. T is a multiple of TileM (128), H of TileN (128),
// I of TileK (64) → exact FLOP count, no padding.
static const std::vector<Mlp2Shape> kShapes = {
	{128, 128, 128, 1},   // single M-tile, single N-tile, single expert
	{128, 256, 256, 1},   // two N-tiles, deeper K
	{256, 256, 256, 2},   // two M-tiles across two experts
	{384, 128, 256, 3},   // three M-tiles, one expert each
};

static const std::vector<Mlp2Shape> kShapes192 = {
	{128, 192, 128, 1},
	{128, 384, 256, 1},
	{256, 384, 256, 2},
	{384, 192, 256, 3},
};

// Large, GPU-saturating shapes for the TFLOPS benchmark. T is a multiple of
// TileM (128) → no token padding, so 2·T·H·I is exact; H is a multiple of
// TileN (128). Realistic MoE dims (H=I=4096, E=8).
static const std::vector<Mlp2Shape> kBenchShapes = {
	{ 2048, 4096, 4096, 8},
	{ 4096, 4096, 4096, 8},
	{ 8192, 4096, 4096, 8},
	{16384, 4096, 4096, 8},
};

// ── Blackwell (Compute=100 / UMMA) — requires an sm_100 GPU at runtime ──
TEST(Mlp2Fused, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapes) run_fused<TraitsFused, 100>(s);
}

// ── Hopper (Compute=90 / WGMMA) — requires an sm_90 GPU at runtime ──
// Same shapes, same cpu_reference, same tolerances as the Blackwell test
// (run_fused is shared, templated only on Compute): the Hopper path is held to
// the identical bar — no relaxed thresholds, no bias.
TEST(Mlp2FusedSm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes) run_fused<TraitsFused, 90>(s);
}

TEST(Mlp2FusedSm90, TileN192Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes192) run_fused<TraitsFused192, 90>(s);
}

// ═══════════════════════════════════════════════════════════════════
// TFLOPS benchmarks — opt-in via MLP2_BENCH=1 (skipped by default so the
// correctness run stays fast). Arch-gated like the tests above: run the
// binary on a B200 for the Blackwell numbers, on an H100 for Hopper.
// Filter with: --gtest_filter='*TFLOPs*'
// ═══════════════════════════════════════════════════════════════════

TEST(Mlp2Fused, TFLOPs_Blackwell) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (!mlp2_bench_enabled())  GTEST_SKIP() << "set MLP2_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run_fused_bench<100>(s, cfg);
}

TEST(Mlp2Fused, TFLOPs_Hopper) {
	if (!hopper_available())   GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	if (!mlp2_bench_enabled()) GTEST_SKIP() << "set MLP2_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run_fused_bench<90>(s, cfg);
}

// ═══════════════════════════════════════════════════════════════════
// Entry point — arch-aware default filter (clean output)
// ═══════════════════════════════════════════════════════════════════
// By default, run only the tests that match the GPU actually present, so a
// Blackwell box shows just the Compute=100 results with no Hopper/skip noise
// (and vice-versa on Hopper). The TFLOPS benchmarks are added to the default
// selection only when MLP2_BENCH is set. An explicit --gtest_filter, or
// --gtest_list_tests (used by ctest discovery), always takes precedence.
int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

	const bool user_filtered = GTEST_FLAG_GET(filter) != "*";
	const bool listing       = GTEST_FLAG_GET(list_tests);
	if (!user_filtered && !listing) {
		std::string f;
		if (blackwell_available()) {
			f = "Mlp2Fused.Correctness";
			if (mlp2_bench_enabled())
				f += ":Mlp2Fused.TFLOPs_Blackwell";
		} else if (hopper_available()) {
			f = "Mlp2FusedSm90.Correctness";
			if (mlp2_bench_enabled())
				f += ":Mlp2Fused.TFLOPs_Hopper";
		}
		if (!f.empty()) GTEST_FLAG_SET(filter, f);
	}
	return RUN_ALL_TESTS();
}
