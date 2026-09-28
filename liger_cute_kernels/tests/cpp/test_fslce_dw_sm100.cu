// Standalone SM100 TP-FSLCE dW correctness and throughput probe.
//
// This intentionally exercises only dW = dZ^T @ X.  It reuses the proven
// MoE MLP3 paired-CTA mainloop as the initial baseline: M256xN256xK64 joined
// tile, 2x1 cluster, FP32 TMEM accumulation, BF16 shared-memory epilogue, and
// TMA store/reduce-add.  The production dW pipeline is developed separately;
// this probe keeps a stable cuBLAS-backed correctness/performance oracle.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "backward_dw_gemm_sm100.cuh"
#include "backward_dw_store_sm100.cuh"
#include "mlp3_sm100.cuh"

namespace fslce_dw_test {

using namespace cute;
using Element = cutlass::bfloat16_t;
using Traits = liger::Mlp3Traits2Sm<
	Element,
	256,
	256,
	64,
	6,
	64,
	2,
	2>;
using Smem = liger::Mlp3Smem2Sm<Traits>;

#define CUDA_CHECK(expr)                                                     \
	do {                                                                     \
		cudaError_t error_ = (expr);                                          \
		if (error_ != cudaSuccess) {                                          \
			std::fprintf(                                                     \
				stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
				cudaGetErrorString(error_));                                  \
			std::exit(1);                                                     \
		}                                                                    \
	} while (0)

#define CUBLAS_CHECK(expr)                                                   \
	do {                                                                     \
		cublasStatus_t status_ = (expr);                                      \
		if (status_ != CUBLAS_STATUS_SUCCESS) {                               \
			std::fprintf(                                                     \
				stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #expr, \
				static_cast<int>(status_));                                   \
			std::exit(1);                                                     \
		}                                                                    \
	} while (0)

#define CUTLASS_CHECK(expr)                                                  \
	do {                                                                     \
		cutlass::Status status_ = (expr);                                     \
		if (status_ != cutlass::Status::kSuccess) {                           \
			std::fprintf(                                                     \
				stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
				cutlassGetStatusString(status_));                              \
			std::exit(1);                                                     \
		}                                                                    \
	} while (0)

template <class TmaA, class TmaB, class TmaOut>
__global__ __launch_bounds__(Traits::NumThreads, 1) __cluster_dims__(2, 1, 1)
void dw_moe_baseline_kernel(
		__grid_constant__ const TmaA tma_a,
		__grid_constant__ const TmaB tma_b,
		__grid_constant__ const TmaOut tma_out,
		const int* k_starts,
		const int* k_ends,
		int m,
		int n,
		int k,
		int m_tiles,
		int n_tiles) {
	extern __shared__ char storage[];
	auto& smem = *reinterpret_cast<Smem*>(storage);
	liger::mlp3_fwd<Traits, 100, false>(
		smem,
		tma_a,
		tma_b,
		tma_out,
		k_starts,
		k_ends,
		1,
		m,
		n,
		k,
		m,
		m_tiles,
		n_tiles,
		m_tiles,
		0,
		(k + Traits::TileK - 1) / Traits::TileK);
}

__global__ void fill_bf16(Element* data, std::size_t count, std::uint32_t seed) {
	std::size_t index =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	std::uint32_t value =
		static_cast<std::uint32_t>(index) * 747796405u + seed * 2891336453u;
	value = ((value >> ((value >> 28) + 4)) ^ value) * 277803737u;
	value = (value >> 22) ^ value;
	float unit = static_cast<float>(value & 0xffffu) * (1.0f / 65535.0f);
	data[index] = Element((unit - 0.5f) * 0.0625f);
}

__global__ void fill_constant(
		Element* data, std::size_t count, float value) {
	std::size_t index =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < count) data[index] = Element(value);
}

struct ErrorStats {
	unsigned long long mismatches;
	unsigned int max_abs_bits;
	unsigned int max_rel_bits;
	double sum_abs;
	double sum_sq;
};

__global__ void compare_bf16(
		const Element* got,
		const Element* reference,
		std::size_t count,
		ErrorStats* stats) {
	std::size_t index =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	float lhs = static_cast<float>(got[index]);
	float rhs = static_cast<float>(reference[index]);
	float absolute = std::fabs(lhs - rhs);
	float relative = absolute / fmaxf(std::fabs(rhs), 1.0e-3f);
	if (reinterpret_cast<const std::uint16_t*>(got)[index] !=
		reinterpret_cast<const std::uint16_t*>(reference)[index]) {
		atomicAdd(&stats->mismatches, 1ull);
	}
	atomicMax(&stats->max_abs_bits, __float_as_uint(absolute));
	atomicMax(&stats->max_rel_bits, __float_as_uint(relative));
	atomicAdd(&stats->sum_abs, static_cast<double>(absolute));
	atomicAdd(
		&stats->sum_sq,
		static_cast<double>(absolute) * static_cast<double>(absolute));
}

struct Problem {
	int m;
	int n;
	int k;
};

struct DeviceBuffers {
	Element* a = nullptr;
	Element* b = nullptr;
	Element* output = nullptr;
	Element* reference = nullptr;
	int* k_starts = nullptr;
	int* k_ends = nullptr;

	DeviceBuffers() = default;
	DeviceBuffers(const DeviceBuffers&) = delete;
	DeviceBuffers& operator=(const DeviceBuffers&) = delete;
	DeviceBuffers(DeviceBuffers&& other) noexcept
		: a(std::exchange(other.a, nullptr)),
		  b(std::exchange(other.b, nullptr)),
		  output(std::exchange(other.output, nullptr)),
		  reference(std::exchange(other.reference, nullptr)),
		  k_starts(std::exchange(other.k_starts, nullptr)),
		  k_ends(std::exchange(other.k_ends, nullptr)) {}

	~DeviceBuffers() {
		cudaFree(a);
		cudaFree(b);
		cudaFree(output);
		cudaFree(reference);
		cudaFree(k_starts);
		cudaFree(k_ends);
	}
};

DeviceBuffers allocate_problem(const Problem& problem) {
	DeviceBuffers buffers;
	std::size_t a_elements =
		static_cast<std::size_t>(problem.k) * problem.m;
	std::size_t b_elements =
		static_cast<std::size_t>(problem.k) * problem.n;
	std::size_t c_elements =
		static_cast<std::size_t>(problem.m) * problem.n;
	CUDA_CHECK(cudaMalloc(&buffers.a, a_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&buffers.b, b_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&buffers.output, c_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&buffers.reference, c_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&buffers.k_starts, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&buffers.k_ends, sizeof(int)));
	int k_start = 0;
	int k_end = (problem.k + Traits::TileK - 1) / Traits::TileK;
	CUDA_CHECK(cudaMemcpy(
		buffers.k_starts, &k_start, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		buffers.k_ends, &k_end, sizeof(int), cudaMemcpyHostToDevice));
	int threads = 256;
	fill_bf16<<<
		static_cast<unsigned>((a_elements + threads - 1) / threads),
		threads>>>(buffers.a, a_elements, 17u);
	fill_bf16<<<
		static_cast<unsigned>((b_elements + threads - 1) / threads),
		threads>>>(buffers.b, b_elements, 29u);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	return buffers;
}

auto make_tma_a(const Problem& problem, const Element* a) {
	auto tensor = make_tensor(
		make_gmem_ptr(a),
		make_shape(
			static_cast<std::int64_t>(problem.m),
			static_cast<std::int64_t>(problem.k)),
		make_stride(Int<1>{}, static_cast<std::int64_t>(problem.m)));
	return make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor,
		typename Traits::SmemLayoutDYT_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
}

auto make_tma_b(const Problem& problem, const Element* b) {
	auto tensor = make_tensor(
		make_gmem_ptr(b),
		make_shape(
			static_cast<std::int64_t>(problem.n),
			static_cast<std::int64_t>(problem.k)),
		make_stride(Int<1>{}, static_cast<std::int64_t>(problem.n)));
	return make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor,
		typename Traits::SmemLayoutZ_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
}

template <bool Add>
auto make_tma_output(const Problem& problem, Element* output) {
	auto tensor = make_tensor(
		make_gmem_ptr(output),
		make_shape(
			static_cast<std::int64_t>(problem.m),
			static_cast<std::int64_t>(problem.n)),
		make_stride(static_cast<std::int64_t>(problem.n), Int<1>{}));
	if constexpr (Add) {
		return make_tma_copy(
			SM90_TMA_REDUCE_ADD{},
			tensor,
			typename Traits::SmemLayoutStore{});
	} else {
		return make_tma_copy(
			SM90_TMA_STORE{},
			tensor,
			typename Traits::SmemLayoutStore{});
	}
}

template <bool Add>
struct Launcher {
	Problem problem;
	DeviceBuffers* buffers;
	decltype(make_tma_a(
		std::declval<const Problem&>(),
		static_cast<const Element*>(nullptr))) tma_a;
	decltype(make_tma_b(
		std::declval<const Problem&>(),
		static_cast<const Element*>(nullptr))) tma_b;
	decltype(make_tma_output<Add>(
		std::declval<const Problem&>(),
		static_cast<Element*>(nullptr))) tma_output;
	int m_tiles;
	int n_tiles;
	int grid_ctas;

	Launcher(
			const Problem& problem_,
			DeviceBuffers& buffers_,
			int cluster_pairs = 0)
		: problem(problem_),
		  buffers(&buffers_),
		  tma_a(make_tma_a(problem, buffers->a)),
		  tma_b(make_tma_b(problem, buffers->b)),
		  tma_output(make_tma_output<Add>(problem, buffers->output)),
		  m_tiles((problem.m + Traits::TileM - 1) / Traits::TileM),
		  n_tiles((problem.n + Traits::TileN - 1) / Traits::TileN),
		  grid_ctas(
			  2 *
			  (cluster_pairs > 0
					  ? std::min(cluster_pairs, m_tiles * n_tiles)
					  : m_tiles * n_tiles)) {}

	void prepare() const {
		auto kernel = &dw_moe_baseline_kernel<
			decltype(tma_a), decltype(tma_b), decltype(tma_output)>;
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			static_cast<int>(sizeof(Smem))));
	}

	void operator()(cudaStream_t stream = nullptr) const {
		auto kernel = &dw_moe_baseline_kernel<
			decltype(tma_a), decltype(tma_b), decltype(tma_output)>;
		cudaLaunchAttribute cluster_attribute = {};
		cluster_attribute.id = cudaLaunchAttributeClusterDimension;
		cluster_attribute.val.clusterDim.x = 2;
		cluster_attribute.val.clusterDim.y = 1;
		cluster_attribute.val.clusterDim.z = 1;
		cudaLaunchConfig_t config = {};
		config.gridDim = dim3(static_cast<unsigned>(grid_ctas), 1, 1);
		config.blockDim = dim3(Traits::NumThreads, 1, 1);
		config.dynamicSmemBytes = sizeof(Smem);
		config.stream = stream;
		config.attrs = &cluster_attribute;
		config.numAttrs = 1;
		CUDA_CHECK(cudaLaunchKernelEx(
			&config,
			kernel,
			tma_a,
			tma_b,
			tma_output,
			buffers->k_starts,
			buffers->k_ends,
			problem.m,
			problem.n,
			problem.k,
			m_tiles,
			n_tiles));
	}
};

void cublas_reference(
		cublasHandle_t handle,
		const Problem& problem,
		const Element* a,
		const Element* b,
		Element* output,
		float beta) {
	float alpha = 1.0f;
	CUBLAS_CHECK(cublasGemmEx(
		handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		problem.n,
		problem.m,
		problem.k,
		&alpha,
		b,
		CUDA_R_16BF,
		problem.n,
		a,
		CUDA_R_16BF,
		problem.m,
		&beta,
		output,
		CUDA_R_16BF,
		problem.n,
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

ErrorStats compare_outputs(
		const Element* got,
		const Element* reference,
		std::size_t count) {
	ErrorStats* device_stats = nullptr;
	CUDA_CHECK(cudaMalloc(&device_stats, sizeof(ErrorStats)));
	CUDA_CHECK(cudaMemset(device_stats, 0, sizeof(ErrorStats)));
	int threads = 256;
	compare_bf16<<<
		static_cast<unsigned>((count + threads - 1) / threads),
		threads>>>(got, reference, count, device_stats);
	CUDA_CHECK(cudaGetLastError());
	ErrorStats stats{};
	CUDA_CHECK(cudaMemcpy(
		&stats, device_stats, sizeof(stats), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(device_stats));
	return stats;
}

float float_from_bits(unsigned int bits) {
	float value;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

template <class Launch>
double benchmark(const Launch& launch, int warmups, int iterations) {
	for (int i = 0; i < warmups; ++i) launch();
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> samples;
	samples.reserve(iterations);
	for (int i = 0; i < iterations; ++i) {
		CUDA_CHECK(cudaEventRecord(start));
		launch();
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float elapsed = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
		samples.push_back(elapsed);
	}
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	std::sort(samples.begin(), samples.end());
	return samples[samples.size() / 2];
}

template <bool Add>
void run_case(
		const Problem& problem,
		int warmups,
		int iterations,
		cublasHandle_t cublas,
		int cluster_pairs) {
	DeviceBuffers buffers = allocate_problem(problem);
	std::size_t output_elements =
		static_cast<std::size_t>(problem.m) * problem.n;
	CUDA_CHECK(cudaMemset(
		buffers.output, 0, output_elements * sizeof(Element)));
	CUDA_CHECK(cudaMemset(
		buffers.reference, 0, output_elements * sizeof(Element)));

	Launcher<Add> launch(problem, buffers, cluster_pairs);
	launch.prepare();
	launch();
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	cublas_reference(
		cublas,
		problem,
		buffers.a,
		buffers.b,
		buffers.reference,
		Add ? 1.0f : 0.0f);
	CUDA_CHECK(cudaDeviceSynchronize());

	ErrorStats stats =
		compare_outputs(buffers.output, buffers.reference, output_elements);
	float max_abs = float_from_bits(stats.max_abs_bits);
	float max_rel = float_from_bits(stats.max_rel_bits);
	double mean_abs = stats.sum_abs / static_cast<double>(output_elements);
	double rmse =
		std::sqrt(stats.sum_sq / static_cast<double>(output_elements));

	double milliseconds = benchmark(launch, warmups, iterations);
	double flops = 2.0 * static_cast<double>(problem.m) * problem.n * problem.k;
	double tflops = flops / (milliseconds * 1.0e9);
	std::printf(
		"DW_BASELINE mode=%s schedule=%s pairs=%d M=%d N=%d K=%d "
		"ms=%.4f tflops=%.2f "
		"smem=%zu mismatch=%.6f max_abs=%.6g mean_abs=%.6g rmse=%.6g "
		"max_rel=%.6g\n",
		Add ? "add" : "store",
		cluster_pairs > 0 ? "persistent" : "full",
		launch.grid_ctas / 2,
		problem.m,
		problem.n,
		problem.k,
		milliseconds,
		tflops,
		sizeof(Smem),
		static_cast<double>(stats.mismatches) /
			static_cast<double>(output_elements),
		max_abs,
		mean_abs,
		rmse,
		max_rel);
}

template <bool Add, int KTiles>
struct DedicatedLauncher {
	using Bundle = decltype(
		liger::fused_scaled_linear_cross_entropy::
			make_backward_dw_tma_bundle_sm100<Add>(
				static_cast<const Element*>(nullptr),
				static_cast<const Element*>(nullptr),
				static_cast<Element*>(nullptr),
				1,
				1,
				KTiles * 64));
	using TmaA = typename Bundle::TmaAType;
	using TmaB = typename Bundle::TmaBType;
	using TmaOutput = typename Bundle::TmaOutputType;
	using DedicatedTraits =
		liger::fused_scaled_linear_cross_entropy::DwGemmTraitsSm100<
			TmaA,
			TmaB,
			TmaOutput>;
	using DedicatedSmem =
		liger::fused_scaled_linear_cross_entropy::DwGemmSharedStorageSm100<
			DedicatedTraits>;
	Problem problem;
	Bundle tma;
	int cluster_pairs;
	cudaClusterSchedulingPolicy scheduling_policy;

	DedicatedLauncher(
			const Problem& problem_,
			DeviceBuffers& buffers,
			int cluster_pairs_,
			cudaClusterSchedulingPolicy scheduling_policy_)
		: problem(problem_),
		  tma(
			  liger::fused_scaled_linear_cross_entropy::
				  make_backward_dw_tma_bundle_sm100<Add>(
					  buffers.a,
					  buffers.b,
					  buffers.output,
					  problem.m,
					  problem.n,
					  problem.k)),
		  cluster_pairs(cluster_pairs_),
		  scheduling_policy(scheduling_policy_) {
		if (problem.k != KTiles *
				liger::fused_scaled_linear_cross_entropy::
					DwGemmConfigSm100::kTileK) {
			std::fprintf(stderr, "invalid specialized K\n");
			std::exit(1);
		}
	}

	void prepare() const {
		CUDA_CHECK((
			liger::fused_scaled_linear_cross_entropy::
				prepare_backward_dw_gemm_sm100<
				KTiles,
				TmaA,
				TmaB,
				TmaOutput>()));
	}

	void operator()(cudaStream_t stream = nullptr) const {
		(void)scheduling_policy;
		CUDA_CHECK(
			liger::fused_scaled_linear_cross_entropy::
				launch_backward_dw_gemm_sm100<KTiles>(
					tma,
					problem.m,
					problem.n,
					cluster_pairs,
					stream));
	}
};

template <bool Add, int KTiles>
void run_dedicated_case(
		const Problem& problem,
		int warmups,
		int iterations,
		cublasHandle_t cublas,
		int cluster_pairs,
		cudaClusterSchedulingPolicy scheduling_policy) {
	DeviceBuffers buffers = allocate_problem(problem);
	std::size_t output_elements =
		static_cast<std::size_t>(problem.m) * problem.n;
	CUDA_CHECK(cudaMemset(
		buffers.output, 0, output_elements * sizeof(Element)));
	CUDA_CHECK(cudaMemset(
		buffers.reference, 0, output_elements * sizeof(Element)));

	DedicatedLauncher<Add, KTiles> launch(
		problem,
		buffers,
		cluster_pairs,
		scheduling_policy);
	launch.prepare();
	launch();
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	cublas_reference(
		cublas,
		problem,
		buffers.a,
		buffers.b,
		buffers.reference,
		Add ? 1.0f : 0.0f);
	CUDA_CHECK(cudaDeviceSynchronize());

	ErrorStats stats =
		compare_outputs(buffers.output, buffers.reference, output_elements);
	if constexpr (Add) {
		fill_constant<<<
			static_cast<unsigned>((output_elements + 255) / 256),
			256>>>(buffers.output, output_elements, 0.5f);
	}
	CUDA_CHECK(cudaDeviceSynchronize());
	double milliseconds = benchmark(launch, warmups, iterations);
	double flops = 2.0 * static_cast<double>(problem.m) * problem.n * problem.k;
	double tflops = flops / (milliseconds * 1.0e9);
	std::printf(
		"DW_DEDICATED mode=%s policy=%s pairs=%d M=%d N=%d K=%d "
		"perf_input=random_bf16 ms=%.4f tflops=%.2f "
		"smem=%zu mismatch=%.6f max_abs=%.6g "
		"mean_abs=%.6g rmse=%.6g\n",
		Add ? "add" : "store",
		scheduling_policy == cudaClusterSchedulingPolicySpread
			? "spread"
			: "balanced",
		cluster_pairs,
		problem.m,
		problem.n,
		problem.k,
		milliseconds,
		tflops,
		sizeof(typename DedicatedLauncher<Add, KTiles>::DedicatedSmem),
		static_cast<double>(stats.mismatches) /
			static_cast<double>(output_elements),
		float_from_bits(stats.max_abs_bits),
		stats.sum_abs / static_cast<double>(output_elements),
		std::sqrt(stats.sum_sq / static_cast<double>(output_elements)));
}

void run_multiwave_rounding(
		int waves,
		int warmups,
		int iterations,
		cublasHandle_t cublas) {
	Problem wave_problem{65536, 4096, 1024};
	Problem full_problem{
		65536,
		4096,
		waves * wave_problem.k};
	DeviceBuffers buffers = allocate_problem(full_problem);
	std::size_t output_elements =
		static_cast<std::size_t>(full_problem.m) * full_problem.n;
	CUDA_CHECK(cudaMemset(
		buffers.output, 0, output_elements * sizeof(Element)));
	CUDA_CHECK(cudaMemset(
		buffers.reference, 0, output_elements * sizeof(Element)));

	std::vector<std::unique_ptr<DedicatedLauncher<true, 16>>>
		add_launchers;
	add_launchers.reserve(waves);
	for (int wave = 1; wave < waves; ++wave) {
		DeviceBuffers wave_buffers;
		wave_buffers.a =
			buffers.a +
			static_cast<std::size_t>(wave) * wave_problem.k *
				wave_problem.m;
		wave_buffers.b =
			buffers.b +
			static_cast<std::size_t>(wave) * wave_problem.k *
				wave_problem.n;
		wave_buffers.output = buffers.output;
		add_launchers.emplace_back(
			std::make_unique<DedicatedLauncher<true, 16>>(
				wave_problem,
				wave_buffers,
				74,
				cudaClusterSchedulingPolicySpread));
		wave_buffers.a = nullptr;
		wave_buffers.b = nullptr;
		wave_buffers.output = nullptr;
	}
	using StoreConfig =
		liger::fused_scaled_linear_cross_entropy::BackwardDwStoreSm100;
	StoreConfig::Gemm store_gemm;
	auto store_arguments = StoreConfig::arguments(
		buffers.a,
		buffers.b,
		buffers.output,
		wave_problem.m,
		wave_problem.n,
		wave_problem.k,
		148);
	CUDA_CHECK(StoreConfig::prepare_kernel());
	std::size_t store_workspace_bytes =
		StoreConfig::Gemm::get_workspace_size(store_arguments);
	void* store_workspace = nullptr;
	if (store_workspace_bytes != 0) {
		CUDA_CHECK(cudaMalloc(
			&store_workspace,
			store_workspace_bytes));
	}
	CUTLASS_CHECK(store_gemm.can_implement(store_arguments));
	CUTLASS_CHECK(store_gemm.initialize(
		store_arguments,
		store_workspace));
	if (!add_launchers.empty()) add_launchers.front()->prepare();

	auto launch_waves = [&]() {
		CUTLASS_CHECK(store_gemm.run());
		for (auto& add_launch : add_launchers) {
			(*add_launch)();
		}
	};
	launch_waves();
	CUDA_CHECK(cudaDeviceSynchronize());
	cublas_reference(
		cublas,
		full_problem,
		buffers.a,
		buffers.b,
		buffers.reference,
		0.0f);
	CUDA_CHECK(cudaDeviceSynchronize());
	ErrorStats stats =
		compare_outputs(buffers.output, buffers.reference, output_elements);
	std::size_t a_elements =
		static_cast<std::size_t>(full_problem.k) * full_problem.m;
	std::size_t b_elements =
		static_cast<std::size_t>(full_problem.k) * full_problem.n;
	fill_constant<<<
		static_cast<unsigned>((a_elements + 255) / 256),
		256>>>(buffers.a, a_elements, 1.0f / 256.0f);
	fill_constant<<<
		static_cast<unsigned>((b_elements + 255) / 256),
		256>>>(buffers.b, b_elements, 1.0f / 256.0f);
	CUDA_CHECK(cudaDeviceSynchronize());
	double milliseconds = benchmark(launch_waves, warmups, iterations);
	double flops =
		2.0 * static_cast<double>(full_problem.m) *
		full_problem.n * full_problem.k;
	std::printf(
		"DW_MULTIWAVE waves=%d M=%d N=%d K=%d "
		"perf_input=constant_1_over_256 ms=%.4f tflops=%.2f "
		"mismatch=%.6f max_abs=%.6g mean_abs=%.6g rmse=%.6g\n",
		waves,
		full_problem.m,
		full_problem.n,
		full_problem.k,
		milliseconds,
		flops / (milliseconds * 1.0e9),
		static_cast<double>(stats.mismatches) /
			static_cast<double>(output_elements),
		float_from_bits(stats.max_abs_bits),
		stats.sum_abs / static_cast<double>(output_elements),
		std::sqrt(stats.sum_sq / static_cast<double>(output_elements)));
	CUDA_CHECK(cudaFree(store_workspace));
}

}  // namespace fslce_dw_test

int main(int argc, char** argv) {
	using namespace fslce_dw_test;
	int warmups = argc > 1 ? std::atoi(argv[1]) : 5;
	int iterations = argc > 2 ? std::atoi(argv[2]) : 21;
	bool dedicated_only =
		argc > 3 && std::strcmp(argv[3], "dedicated") == 0;
	int device = 0;
	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	if (properties.major != 10) {
		std::fprintf(stderr, "SM100-family GPU required\n");
		return 2;
	}
	std::printf(
		"DW_ENV gpu=\"%s\" sm=%d%d sms=%d stages=%d threads=%d\n",
		properties.name,
		properties.major,
		properties.minor,
		properties.multiProcessorCount,
		liger::fused_scaled_linear_cross_entropy::
			DwGemmConfigSm100::kMainloopStages,
		liger::fused_scaled_linear_cross_entropy::
			DwGemmConfigSm100::kNumThreads);
	cublasHandle_t cublas = nullptr;
	CUBLAS_CHECK(cublasCreate(&cublas));
	if (!dedicated_only) {
		run_case<false>(
			{65536, 4096, 4096}, warmups, iterations, cublas, 0);
		run_case<false>(
			{65536, 4096, 4096}, warmups, iterations, cublas, 74);
		run_case<false>(
			{65536, 4096, 1024}, warmups, iterations, cublas, 74);
		run_case<true>(
			{65536, 4096, 1024}, warmups, iterations, cublas, 74);
	}
	run_dedicated_case<false, 64>(
		{65536, 4096, 4096},
		warmups,
		iterations,
		cublas,
		74,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<false, 64>(
		{65536, 4096, 4096},
		warmups,
		iterations,
		cublas,
		148,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<false, 16>(
		{65536, 4096, 1024},
		warmups,
		iterations,
		cublas,
		74,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<true, 16>(
		{65536, 4096, 1024},
		warmups,
		iterations,
		cublas,
		74,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<false, 16>(
		{776, 520, 1024},
		0,
		1,
		cublas,
		8,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<true, 16>(
		{776, 520, 1024},
		0,
		1,
		cublas,
		8,
		cudaClusterSchedulingPolicySpread);
	run_dedicated_case<false, 64>(
		{520, 264, 4096},
		0,
		1,
		cublas,
		3,
		cudaClusterSchedulingPolicySpread);
	run_multiwave_rounding(4, warmups, iterations, cublas);
	CUBLAS_CHECK(cublasDestroy(cublas));
	return 0;
}
