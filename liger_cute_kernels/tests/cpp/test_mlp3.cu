
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp3_sm100.cuh"

using namespace cute;
using liger::Mlp3Traits;
using liger::Mlp3TraitsSm100;
using liger::Mlp3Smem;
using liger::Mlp3SmemSm100;
using liger::mlp3_fwd;
using Element = cutlass::bfloat16_t;

// Compute=90 (Hopper/WGMMA, 1SM): mlp3's N-split default — TileM=128,
// TileN=256, EpiChunkN=64 (the widest ported epilogue → TmemLoadOp<64> =
// SM100_TMEM_LOAD_32dp32b64x on the Blackwell epilogue).
using Traits3 = Mlp3Traits<Element, /*TileM=*/128, /*TileN=*/256,
                           /*TileK=*/64, /*Stages=*/4, /*EpiChunkN=*/64>;

// Match the production single-CTA Blackwell weight-gradient tile.
using Traits3Blackwell = Mlp3TraitsSm100<Element, /*TileM=*/128, /*TileN=*/128,
                                 /*TileK=*/64, /*Stages=*/5, /*EpiChunkN=*/64>;
static_assert(Traits3Blackwell::ClusterM == 1);
static_assert(size(typename Traits3Blackwell::TiledMmaSm100::AtomThrID{}) == 1);
static_assert(Traits3Blackwell::TmaTransBytes ==
              Traits3Blackwell::TmaTransBytesDYT + Traits3Blackwell::TmaTransBytesZ);

#define CUDA_OK(expr)                                                       \
	do {                                                                    \
		cudaError_t _e = (expr);                                            \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)


template <typename Traits, int Compute,
          typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceDA>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp3_test_kernel(
		__grid_constant__ TmaLoadDYT const tma_load_dyt,
		__grid_constant__ TmaLoadZ   const tma_load_z,
		__grid_constant__ TmaReduceDA const tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts, int hidden_dim, int intermediate_dim, int num_tokens,
		int total_n_rows, int num_m_tiles, int num_n_tiles, int outer_split) {

	extern __shared__ char raw_smem[];
	using Smem = cute::conditional_t<Compute == 100, Mlp3SmemSm100<Traits>, Mlp3Smem<Traits>>;
	auto& smem = *reinterpret_cast<Smem*>(raw_smem);

	mlp3_fwd<Traits, Compute, Compute == 90>(
		smem, tma_load_dyt, tma_load_z, tma_reduce_da,
		expert_k_starts, expert_k_ends, num_experts,
		hidden_dim, intermediate_dim, num_tokens, total_n_rows,
		num_m_tiles, num_n_tiles, outer_split);
}

// ═══════════════════════════════════════════════════════════════════
// Host helpers
// ═══════════════════════════════════════════════════════════════════

struct Mlp3Shape {
	int num_tokens;        // T — contraction axis, multiple of TileK; (T/TileK)%E==0
	int hidden_dim;        // H — M axis (dA rows), multiple of TileM
	int intermediate_dim;  // I — N axis (dA cols), multiple of TileN
	int num_experts;       // E
};

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

// CPU reference: dA[e, h, i] = Σ_{t ∈ tokens(e)} dY[t, h] · Z[t, i], where
// expert e owns the contiguous token-block range [e·bpe, (e+1)·bpe) (bpe =
// (T/TileK)/E blocks of TileK tokens each). Inputs bf16-rounded; accum fp32.
// dA laid out [E·H, I] row-major to match the SM90_TMA_REDUCE_ADD gmem view.
static std::vector<float> cpu_reference(
		const std::vector<float>& dY,   // [T, H] bf16-rounded
		const std::vector<float>& Z,    // [T, I] bf16-rounded
		const Mlp3Shape& s, int TileK) {
	int T = s.num_tokens, H = s.hidden_dim, I = s.intermediate_dim, E = s.num_experts;
	int num_k_blocks    = T / TileK;
	int blocks_per_exp  = num_k_blocks / E;
	int toks_per_exp    = blocks_per_exp * TileK;

	std::vector<float> dA((size_t)E * H * I, 0.f);
	for (int e = 0; e < E; ++e) {
		int t0 = e * toks_per_exp, t1 = t0 + toks_per_exp;
		float* dAe = &dA[(size_t)e * H * I];
		for (int t = t0; t < t1; ++t) {
			const float* dyr = &dY[(size_t)t * H];
			const float* zr  = &Z[(size_t)t * I];
			for (int h = 0; h < H; ++h) {
				float dyh = dyr[h];
				float* dArow = &dAe[(size_t)h * I];
				for (int i = 0; i < I; ++i)
					dArow[i] += dyh * zr[i];
			}
		}
	}
	return dA;
}

struct Inputs {
	std::vector<float> dY, Z;             // bf16-rounded host copies
	std::vector<int>   k_starts, k_ends;  // per-expert K-block ranges (TileK units)
	DevBf16 dDY, dZ;
	int* d_k_starts = nullptr;
	int* d_k_ends   = nullptr;
	int  num_m_tiles = 0, num_n_tiles = 0;
	~Inputs() {
		if (d_k_starts) cudaFree(d_k_starts);
		if (d_k_ends)   cudaFree(d_k_ends);
	}
};

template <typename Traits>
static void make_inputs(const Mlp3Shape& s, Inputs& in, unsigned seed) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> nd(0.f, 1.f);
	auto fill = [&](std::vector<float>& v, size_t n) {
		v.resize(n);
		for (size_t i = 0; i < n; ++i) v[i] = bf16_round(nd(rng));
	};
	fill(in.dY, (size_t)s.num_tokens * s.hidden_dim);
	fill(in.Z,  (size_t)s.num_tokens * s.intermediate_dim);

	in.num_m_tiles = s.hidden_dim       / Traits::TileM;
	in.num_n_tiles = s.intermediate_dim / Traits::TileN;

	int num_k_blocks   = s.num_tokens / Traits::TileK;
	int blocks_per_exp = num_k_blocks / s.num_experts;
	in.k_starts.resize(s.num_experts);
	in.k_ends.resize(s.num_experts);
	for (int e = 0; e < s.num_experts; ++e) {
		in.k_starts[e] = e * blocks_per_exp;
		in.k_ends[e]   = (e + 1) * blocks_per_exp;
	}

	upload_bf16(in.dDY, in.dY);
	upload_bf16(in.dZ,  in.Z);
	cudaMalloc(&in.d_k_starts, s.num_experts * sizeof(int));
	cudaMalloc(&in.d_k_ends,   s.num_experts * sizeof(int));
	cudaMemcpy(in.d_k_starts, in.k_starts.data(), s.num_experts * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(in.d_k_ends,   in.k_ends.data(),   s.num_experts * sizeof(int), cudaMemcpyHostToDevice);
}

static std::vector<float> download_bf16(const Element* d, size_t n) {
	std::vector<Element> hb(n);
	cudaMemcpy(hb.data(), d, n * sizeof(Element), cudaMemcpyDeviceToHost);
	std::vector<float> out(n);
	for (size_t i = 0; i < n; ++i) out[i] = float(hb[i]);
	return out;
}

template <typename Traits, int Compute>
static auto make_dyt_tma(const Inputs& in, const Mlp3Shape& s) {
	auto t = make_tensor(make_gmem_ptr(in.dDY.ptr),
		make_shape(s.hidden_dim, s.num_tokens), make_stride(Int<1>{}, s.hidden_dim));
	if constexpr (Compute == 100) {
		return make_tma_copy_A_sm100(SM90_TMA_LOAD{}, t,
			typename Traits::SmemLayoutDYT_1{},
			typename Traits::TileShape{},
			typename Traits::TiledMmaSm100{});
	} else {
		return make_tma_copy(SM90_TMA_LOAD{}, t, typename Traits::SmemLayoutDYT_1{});
	}
}
template <typename Traits, int Compute>
static auto make_z_tma(const Inputs& in, const Mlp3Shape& s) {
	auto t = make_tensor(make_gmem_ptr(in.dZ.ptr),
		make_shape(s.intermediate_dim, s.num_tokens), make_stride(Int<1>{}, s.intermediate_dim));
	if constexpr (Compute == 100) {
		return make_tma_copy_B_sm100(SM90_TMA_LOAD{}, t,
			typename Traits::SmemLayoutZ_1{},
			typename Traits::TileShape{},
			typename Traits::TiledMmaSm100{});
	} else {
		return make_tma_copy(SM90_TMA_LOAD{}, t, typename Traits::SmemLayoutZ_1{});
	}
}
template <typename Traits, int Compute>
static auto make_da_tma(Element* dA, const Mlp3Shape& s) {
	if constexpr (Compute == 90) {
		auto t = make_tensor(make_gmem_ptr(dA),
			make_shape(s.hidden_dim, s.intermediate_dim, s.num_experts),
			make_stride(s.intermediate_dim, Int<1>{},
				s.hidden_dim * s.intermediate_dim));
		return make_tma_copy(SM90_TMA_REDUCE_ADD{}, t, typename Traits::SmemLayoutStore{});
	} else {
		int total_n_rows = s.num_experts * s.hidden_dim;
		auto t = make_tensor(make_gmem_ptr(dA),
			make_shape(total_n_rows, s.intermediate_dim),
			make_stride(s.intermediate_dim, Int<1>{}));
		return make_tma_copy(SM90_TMA_REDUCE_ADD{}, t, typename Traits::SmemLayoutStore{});
	}
}

static int sm_count() {
	int dev = 0; cudaDeviceProp p{};
	if (cudaGetDevice(&dev) != cudaSuccess) return 0;
	if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
	return p.multiProcessorCount;
}

template <typename Traits, int Compute>
static constexpr bool mlp3_walks_m() {
	if constexpr (Compute == 100) return true;
	else return Traits::kMSplit;
}

// ═══════════════════════════════════════════════════════════════════
// Variant runner (correctness)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, int Compute>
static void run3_once(const Mlp3Shape& s, Inputs& in, int outer_split,
                      bool verbose, const char* tag, ErrStats* out,
                      std::vector<float>* got_opt = nullptr) {
	int total_n_rows = s.num_experts * s.hidden_dim;   // E·H
	size_t dA_elems  = (size_t)total_n_rows * s.intermediate_dim;
	Element* dA = nullptr;
	cudaMalloc(&dA, dA_elems * sizeof(Element));
	cudaMemset(dA, 0, dA_elems * sizeof(Element));   // REDUCE_ADD ⇒ zero-init

	auto tma_dyt = make_dyt_tma<Traits, Compute>(in, s);
	auto tma_z   = make_z_tma<Traits, Compute>(in, s);
	auto tma_da  = make_da_tma<Traits, Compute>(dA, s);

	using Smem = cute::conditional_t<Compute == 100, Mlp3SmemSm100<Traits>, Mlp3Smem<Traits>>;
	size_t smem_size = sizeof(Smem);
	auto kernel = mlp3_test_kernel<Traits, Compute,
		decltype(tma_dyt), decltype(tma_z), decltype(tma_da)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	constexpr bool walks_m = mlp3_walks_m<Traits, Compute>();
	int total_chunks = s.num_experts * (walks_m ? in.num_n_tiles : in.num_m_tiles);
	int total_cells  = total_chunks * outer_split;
	int nsm = sm_count();

	if constexpr (Compute == 100) {
		CUDA_OK(cudaFuncSetAttribute(kernel,
			cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
		int pairs  = std::max(1, std::min(nsm / Traits::ClusterM, total_cells));
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
			tma_dyt, tma_z, tma_da, in.d_k_starts, in.d_k_ends,
			s.num_experts, s.hidden_dim, s.intermediate_dim, s.num_tokens,
			total_n_rows, in.num_m_tiles, in.num_n_tiles, outer_split));
	} else {
		int grid_x = std::max(1, std::min(nsm, total_cells));
		kernel<<<dim3(grid_x), Traits::NumThreads, smem_size>>>(
			tma_dyt, tma_z, tma_da, in.d_k_starts, in.d_k_ends,
			s.num_experts, s.hidden_dim, s.intermediate_dim, s.num_tokens,
			total_n_rows, in.num_m_tiles, in.num_n_tiles, outer_split);
	}
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	auto got = download_bf16(dA, dA_elems);
	cudaFree(dA);

	auto ref = cpu_reference(in.dY, in.Z, s, Traits::TileK);
	auto e = compare(got, ref);
	if (verbose)
		printf("[mlp3 C=%-3d %s T=%d H=%d I=%d E=%d osplit=%d] "
		       "mean_rel=%.3f%% max_rel=%.3f%% max_abs=%.3g\n",
			Compute, tag, s.num_tokens, s.hidden_dim, s.intermediate_dim,
			s.num_experts, outer_split, e.mean_rel * 100, e.max_rel * 100, e.max_abs);
	*out = e;
	if (got_opt) *got_opt = std::move(got);
}

template <typename Traits, int Compute>
static void run3(const Mlp3Shape& s) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234);

	ErrStats e1{};
	run3_once<Traits, Compute>(s, in, /*outer_split=*/1, /*verbose=*/true, "", &e1);
	EXPECT_LT(e1.mean_rel, 0.01f);
	EXPECT_LT(e1.max_rel,  0.05f);

	constexpr bool walks_m = mlp3_walks_m<Traits, Compute>();
	int walk = walks_m ? in.num_m_tiles : in.num_n_tiles;
	if (walk >= 2 && (Compute == 100 || walk % 2 == 0)) {
		const char* tag = (walk % 2 == 0) ? "split2" : "split2_tail";
		ErrStats e2{};
		run3_once<Traits, Compute>(s, in, /*outer_split=*/2, /*verbose=*/true, tag, &e2);
		EXPECT_LT(e2.mean_rel, 0.01f);
		EXPECT_LT(e2.max_rel,  0.05f);
	}
}

// ═══════════════════════════════════════════════════════════════════
// TFLOPS benchmark (opt-in via MLP3_BENCH env; timing-only, no CPU ref)
// ═══════════════════════════════════════════════════════════════════
//
//     TFLOPS = 2·T·H·I / median_kernel_seconds / 1e12   (one GEMM)
//
// Sweep over `outer_split` (every divisor of the walk-axis tile count); the
// grid is 1D chunk-fixed with grid.x = min(num_sms, total_cells) for Compute=90
// or the nearest SM-pair-capped even grid for Compute=100. Reports the peak and
// the winning split. dA is RE-ZEROED before every timed launch (REDUCE_ADD
// accumulates — skipping the re-zero would grow dA without bound and change the
// memory-traffic profile).

struct BenchCfg { int warmup = 10; int iters = 50; };

// Divisors of the walk-axis tile count → balanced splits (each cell gets an
// equal share of the walk axis).
static std::vector<int> candidate_splits(int num_walk_tiles) {
	std::vector<int> ds;
	for (int d = 1; d <= num_walk_tiles; ++d)
		if (num_walk_tiles % d == 0) ds.push_back(d);
	if (ds.empty()) ds.push_back(1);
	return ds;
}

static double median_ms(std::vector<float>& v) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	size_t n = v.size();
	return (n & 1) ? (double)v[n / 2] : 0.5 * ((double)v[n / 2 - 1] + (double)v[n / 2]);
}

static bool mlp3_bench_enabled() { return std::getenv("MLP3_BENCH") != nullptr; }

// FLOPs = one GEMM dA = dY^T·Z, 2·T·H·I.
static double tflops_of(const Mlp3Shape& s, double ms) {
	double flops = 2.0 * (double)s.num_tokens * (double)s.hidden_dim
	             * (double)s.intermediate_dim;
	return flops / (ms * 1e-3) / 1e12;
}

template <typename Traits, int Compute>
static void run3_bench(const Mlp3Shape& s, const BenchCfg& cfg) {
	Inputs in; make_inputs<Traits>(s, in, /*seed=*/1234);

	int total_n_rows = s.num_experts * s.hidden_dim;
	size_t dA_elems  = (size_t)total_n_rows * s.intermediate_dim;
	Element* dA = nullptr;
	cudaMalloc(&dA, dA_elems * sizeof(Element));

	auto tma_dyt = make_dyt_tma<Traits, Compute>(in, s);
	auto tma_z   = make_z_tma<Traits, Compute>(in, s);
	auto tma_da  = make_da_tma<Traits, Compute>(dA, s);

	using Smem = cute::conditional_t<Compute == 100, Mlp3SmemSm100<Traits>, Mlp3Smem<Traits>>;
	size_t smem_size = sizeof(Smem);
	auto kernel = mlp3_test_kernel<Traits, Compute,
		decltype(tma_dyt), decltype(tma_z), decltype(tma_da)>;
	CUDA_OK(cudaFuncSetAttribute(kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
	if constexpr (Compute == 100) {
		CUDA_OK(cudaFuncSetAttribute(kernel,
			cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
	}

	int nsm = sm_count();
	constexpr bool walks_m = mlp3_walks_m<Traits, Compute>();
	int total_chunks = s.num_experts * (walks_m ? in.num_n_tiles : in.num_m_tiles);
	int walk = walks_m ? in.num_m_tiles : in.num_n_tiles;

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	double best_tf = 0.0, best_ms = 0.0; int best_split = 1, best_gx = 1;
	for (int osplit : candidate_splits(walk)) {
		int total_cells = total_chunks * osplit;
		int grid_x;
		if constexpr (Compute == 100) {
			int pairs = std::max(1, std::min(nsm / Traits::ClusterM, total_cells));
			grid_x = pairs * Traits::ClusterM;
		} else {
			grid_x = std::max(1, std::min(nsm, total_cells));
		}
		// Re-zero (REDUCE_ADD accumulates) each launch, but stream-ordered
		// BEFORE the timed start→stop window so only the GEMM kernel is timed —
		// the memset is caller-side output prep, not part of the kernel FLOPs.
		auto rezero = [&]() { cudaMemsetAsync(dA, 0, dA_elems * sizeof(Element)); };
		auto launch = [&]() {
			if constexpr (Compute == 100) {
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
				cudaLaunchKernelEx(&launch_config, kernel,
					tma_dyt, tma_z, tma_da, in.d_k_starts, in.d_k_ends,
					s.num_experts, s.hidden_dim, s.intermediate_dim, s.num_tokens,
					total_n_rows, in.num_m_tiles, in.num_n_tiles, osplit);
			} else {
				kernel<<<dim3(grid_x), Traits::NumThreads, smem_size>>>(
					tma_dyt, tma_z, tma_da, in.d_k_starts, in.d_k_ends,
					s.num_experts, s.hidden_dim, s.intermediate_dim, s.num_tokens,
					total_n_rows, in.num_m_tiles, in.num_n_tiles, osplit);
			}
		};
		rezero(); launch(); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());

		for (int i = 0; i < cfg.warmup; ++i) { rezero(); launch(); }
		CUDA_OK(cudaDeviceSynchronize());
		std::vector<float> samples; samples.reserve(cfg.iters);
		for (int i = 0; i < cfg.iters; ++i) {
			rezero();                       // not timed (stream-ordered pre-start)
			cudaEventRecord(start);
			launch();
			cudaEventRecord(stop);
			if (cudaError_t e = cudaEventSynchronize(stop); e != cudaSuccess)
				ADD_FAILURE() << "bench event sync: " << cudaGetErrorString(e);
			float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
			samples.push_back(ms);
		}
		double ms = median_ms(samples);
		double tf = tflops_of(s, ms);
		if (tf > best_tf) { best_tf = tf; best_ms = ms; best_split = osplit; best_gx = grid_x; }
	}
	cudaEventDestroy(start); cudaEventDestroy(stop);

	printf("[mlp3-bench C=%-3d T=%-5d H=%d I=%d E=%d] "
	       "peak %8.2f TFLOPS @ %8.4f ms (osplit=%-2d, grid.x=%-3d, %d SMs)\n",
		Compute, s.num_tokens, s.hidden_dim, s.intermediate_dim, s.num_experts,
		best_tf, best_ms, best_split, best_gx, nsm);

	cudaFree(dA);
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

static const Mlp3Shape kTinyShape = {64, 256, 256, 1};

// Small correctness shapes for Compute=90 (1SM, Traits3). H multiple of TileM
// (128), I of TileN (256), T of TileK (64) with (T/TileK)%E==0 → exact FLOP
// count, no padding.
static const std::vector<Mlp3Shape> kShapes = {
	{  64, 128, 256, 1},   // single tile, single k-block
	{ 128, 256, 512, 1},   // 2 m-tiles, 2 n-tiles, 2 k-blocks
	{ 256, 256, 512, 2},   // 2 experts (2 k-blocks each), 2×2 tiles
	{ 512, 384, 256, 4},   // 4 experts, 3 m-tiles, 1 n-tile, 2 k-blocks/expert
};

static const std::vector<Mlp3Shape> kShapesSm100 = {
	{  64, 256, 256, 1},   // single tile, single k-block
	{ 128, 512, 256, 1},   // 2 m-tiles, 1 n-tile, 2 k-blocks
	{ 256, 512, 512, 2},   // 2 experts, 2×2 tiles
	{ 512, 768, 256, 4},   // 4 experts, 3 m-tiles (odd!), 1 n-tile, 2 k-blocks/expert
};

static const std::vector<Mlp3Shape> kBenchShapes = {
	{ 2048, 4096, 4096, 8},
	{ 4096, 4096, 4096, 8},
	{ 8192, 4096, 4096, 8},
	{16384, 4096, 4096, 8},
};

// ── Diagnostic (runs FIRST): tiny single-tile element-by-element compare. A
//    structured mismatch here localizes a TMEM→store_buf mapping bug or a
//    swapped MN-major operand before the larger shapes muddy the signal. ──
TEST(Mlp3, SingleTile) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	Inputs in; make_inputs<Traits3Blackwell>(kTinyShape, in, /*seed=*/7);
	ErrStats e{}; std::vector<float> got;
	run3_once<Traits3Blackwell, 100>(kTinyShape, in, /*outer_split=*/1, /*verbose=*/true, "tiny", &e, &got);
	auto ref = cpu_reference(in.dY, in.Z, kTinyShape, Traits3Blackwell::TileK);

	int I = kTinyShape.intermediate_dim, H = kTinyShape.hidden_dim;
	int mismatches = 0;
	for (int h = 0; h < H && mismatches < 8; ++h)
		for (int i = 0; i < I && mismatches < 8; ++i) {
			float g = got[(size_t)h * I + i], r = ref[(size_t)h * I + i];
			if (std::fabs(g - r) > std::max(1e-2f, 0.05f * std::fabs(r))) {
				printf("  mismatch dA[h=%d,i=%d]: got=%.4f ref=%.4f\n", h, i, g, r);
				++mismatches;
			}
		}
	EXPECT_LT(e.mean_rel, 0.01f) << "single-tile mean_rel too high (mapping pin?)";
	EXPECT_LT(e.max_rel,  0.05f) << "single-tile max_rel too high (mapping pin?)";
}

TEST(Mlp3, Correctness) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& s : kShapesSm100) run3<Traits3Blackwell, 100>(s);
}

// ── Hopper (Compute=90 / WGMMA, 1SM) — requires an sm_90 GPU at runtime ──
TEST(Mlp3Sm90, Correctness) {
	if (!hopper_available()) GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	for (const auto& s : kShapes) run3<Traits3, 90>(s);
}

// ── TFLOPS benchmarks — opt-in via MLP3_BENCH=1. ──
TEST(Mlp3, TFLOPs_Blackwell) {
	if (!blackwell_available()) GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (!mlp3_bench_enabled())  GTEST_SKIP() << "set MLP3_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run3_bench<Traits3Blackwell, 100>(s, cfg);
}

TEST(Mlp3, TFLOPs_Hopper) {
	if (!hopper_available())   GTEST_SKIP() << "requires an sm_90 (Hopper) GPU";
	if (!mlp3_bench_enabled()) GTEST_SKIP() << "set MLP3_BENCH=1 to run the TFLOPS benchmark";
	BenchCfg cfg;
	for (const auto& s : kBenchShapes) run3_bench<Traits3, 90>(s, cfg);
}

// ═══════════════════════════════════════════════════════════════════
// Entry point — arch-aware default filter (clean output). SingleTile runs
// before Correctness (registration order) so a mapping-pin failure shows first.
// An explicit --gtest_filter or --gtest_list_tests takes precedence.
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

	const bool user_filtered = GTEST_FLAG_GET(filter) != "*";
	const bool listing       = GTEST_FLAG_GET(list_tests);
	if (!user_filtered && !listing) {
		std::string f;
		if (blackwell_available()) {
			f = "Mlp3.SingleTile:Mlp3.Correctness";
			if (mlp3_bench_enabled())
				f += ":Mlp3.TFLOPs_Blackwell";
		} else if (hopper_available()) {
			f = "Mlp3Sm90.Correctness";
			if (mlp3_bench_enabled())
				f += ":Mlp3.TFLOPs_Hopper";
		}
		if (!f.empty()) GTEST_FLAG_SET(filter, f);
	}
	return RUN_ALL_TESTS();
}
