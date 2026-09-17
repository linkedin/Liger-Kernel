// Standalone correctness and raster benchmark for the SM100 TP-FSLCE dX GEMM.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>

#include "backward_dx_gemm_sm100.cuh"
#include "backward_dx_gemm_pair_sm100.cuh"

namespace fslce = liger::fused_scaled_linear_cross_entropy;
using namespace cute;

using Traits = fslce::DxGemmTraitsSm100<4, 32>;
using NPairTraits = fslce::DxGemmNPairTraitsSm100<4, 32>;
using MPairTraits = fslce::DxGemmMPairTraitsSm100<4, 32>;
using Element = typename Traits::Element;

namespace dx_test {

void check_cuda(cudaError_t error, const char* expression) {
	if (error != cudaSuccess) {
		throw std::runtime_error(
			std::string(expression) + ": " + cudaGetErrorString(error));
	}
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

template <class T>
struct DeviceBuffer {
	T* data = nullptr;
	std::size_t elements = 0;

	DeviceBuffer() = default;
	explicit DeviceBuffer(std::size_t count) : elements(count) {
		CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
	}
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
	DeviceBuffer(DeviceBuffer&& other) noexcept
		: data(other.data), elements(other.elements) {
		other.data = nullptr;
		other.elements = 0;
	}
	~DeviceBuffer() {
		if (data != nullptr) cudaFree(data);
	}
};

struct DxProblem {
	int m;
	int n;
	int k;
};

constexpr int ceil_div(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

constexpr int round_up(int value, int alignment) {
	return ceil_div(value, alignment) * alignment;
}

constexpr int a_stride(const DxProblem& shape) {
	return round_up(shape.k, 8);
}

constexpr int b_stride(const DxProblem& shape) {
	return round_up(shape.n, 8);
}

std::size_t stage_elements(const DxProblem& shape) {
	int m_tiles = ceil_div(shape.m, Traits::kTileM);
	int n_tiles = ceil_div(shape.n, Traits::kTileN);
	return static_cast<std::size_t>(m_tiles) *
		static_cast<std::size_t>(n_tiles) *
		Traits::kClusterM *
		Traits::kCtaTileM *
		Traits::kTileN;
}

std::size_t stage_index(
		const DxProblem& shape, int row, int column) {
	int n_tiles = ceil_div(shape.n, Traits::kTileN);
	int m_tile = row / Traits::kTileM;
	int cta_rank =
		(row % Traits::kTileM) / Traits::kCtaTileM;
	int row_in_cta = row % Traits::kCtaTileM;
	int n_tile = column / Traits::kTileN;
	int column_in_tile = column % Traits::kTileN;
	int tile_linear = m_tile * n_tiles + n_tile;
	return (
		(static_cast<std::size_t>(tile_linear) *
				Traits::kClusterM +
			cta_rank) *
				Traits::kCtaTileM +
			row_in_cta) *
			Traits::kTileN +
		column_in_tile;
}

template <fslce::DxRasterOrderSm100 Raster>
auto make_kernel_descriptors(
		const DxProblem& shape,
		Element* a,
		Element* b,
		float* staging) {
	auto tensor_a = make_tensor(
		make_gmem_ptr(a),
		make_shape(
			static_cast<std::int64_t>(shape.m),
			static_cast<std::int64_t>(shape.k)),
		make_stride(
			static_cast<std::int64_t>(a_stride(shape)),
			Int<1>{}));
	auto tensor_b = make_tensor(
		make_gmem_ptr(b),
		make_shape(
			static_cast<std::int64_t>(shape.n),
			static_cast<std::int64_t>(shape.k)),
		make_stride(
			Int<1>{},
			static_cast<std::int64_t>(b_stride(shape))));
	auto tma_a = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_a,
		typename Traits::SmemLayoutA1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});
	auto tma_b = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_b,
		typename Traits::SmemLayoutB1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});

	int m_tiles = ceil_div(shape.m, Traits::kTileM);
	int n_tiles = ceil_div(shape.n, Traits::kTileN);
	std::int64_t staging_rows =
		static_cast<std::int64_t>(m_tiles) *
		n_tiles *
		Traits::kClusterM *
		Traits::kCtaTileM;
	auto tensor_store = make_tensor(
		make_gmem_ptr(staging),
		make_shape(
			staging_rows,
			static_cast<std::int64_t>(Traits::kTileN)),
		make_stride(
			static_cast<std::int64_t>(Traits::kTileN),
			Int<1>{}));
	auto tma_store = make_tma_copy(
		SM90_TMA_STORE{},
		tensor_store,
		typename Traits::SmemLayoutStore{});
	return cute::make_tuple(tma_a, tma_b, tma_store);
}

struct LaunchInfo {
	int sm_count = 0;
	int max_active_clusters = 0;
	int launched_clusters = 0;
	int launched_ctas = 0;
	int registers = 0;
	int static_smem_bytes = 0;
	int dynamic_smem_bytes = 0;
};

template <
	fslce::DxRasterOrderSm100 Raster,
	class TmaA,
	class TmaB,
	class TmaStore>
LaunchInfo prepare_launch(
		const DxProblem& shape,
		const TmaA&,
		const TmaB&,
		const TmaStore&) {
	auto kernel = fslce::dx_gemm_kernel_sm100<
		Raster, Traits, TmaA, TmaB, TmaStore>;
	constexpr int smem_bytes =
		static_cast<int>(sizeof(fslce::DxGemmSmemSm100<Traits>));

	int device = 0;
	cudaDeviceProp properties = {};
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	int optin = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(
		&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (smem_bytes > optin) {
		throw std::runtime_error(
			"dX dynamic shared memory exceeds the device opt-in limit");
	}
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		smem_bytes));
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeNonPortableClusterSizeAllowed,
		1));
	if (const char* policy = std::getenv("FSLCE_DX_CLUSTER_POLICY")) {
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel,
			cudaFuncAttributeClusterSchedulingPolicyPreference,
			std::atoi(policy)));
	}

	cudaLaunchAttribute cluster_attribute = {};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = Traits::kClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	cudaLaunchConfig_t occupancy_config = {};
	occupancy_config.gridDim =
		dim3(static_cast<unsigned>(Traits::kClusterM), 1, 1);
	occupancy_config.blockDim =
		dim3(static_cast<unsigned>(Traits::kNumThreads), 1, 1);
	occupancy_config.dynamicSmemBytes = smem_bytes;
	occupancy_config.attrs = &cluster_attribute;
	occupancy_config.numAttrs = 1;

	int max_active_clusters = 0;
	CUDA_CHECK(cudaOccupancyMaxActiveClusters(
		&max_active_clusters, kernel, &occupancy_config));
	if (max_active_clusters <= 0) {
		throw std::runtime_error(
			"cudaOccupancyMaxActiveClusters returned no resident clusters");
	}

	int total_tiles =
		ceil_div(shape.m, Traits::kTileM) *
		ceil_div(shape.n, Traits::kTileN);
	// One cluster owns one logical tile.  Besides preserving a clean GEMM-only
	// launch (no persistent software wave loop), this keeps the accumulator
	// and TMA pipeline lifetime identical for every tile.
	int launched_clusters = total_tiles;
	if (const char* cap = std::getenv("FSLCE_DX_CLUSTERS")) {
		launched_clusters = std::min(
			launched_clusters, std::max(1, std::atoi(cap)));
	}
	cudaFuncAttributes attributes = {};
	CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));

	LaunchInfo info;
	info.sm_count = properties.multiProcessorCount;
	info.max_active_clusters = max_active_clusters;
	info.launched_clusters = launched_clusters;
	info.launched_ctas = launched_clusters * Traits::kClusterM;
	info.registers = attributes.numRegs;
	info.static_smem_bytes =
		static_cast<int>(attributes.sharedSizeBytes);
	info.dynamic_smem_bytes = smem_bytes;
	return info;
}

template <
	fslce::DxRasterOrderSm100 Raster,
	class TmaA,
	class TmaB,
	class TmaStore>
void launch_dx(
		const DxProblem& shape,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const TmaStore& tma_store,
		const LaunchInfo& info,
		cudaStream_t stream = nullptr) {
	auto kernel = fslce::dx_gemm_kernel_sm100<
		Raster, Traits, TmaA, TmaB, TmaStore>;
	cudaLaunchAttribute cluster_attribute = {};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = Traits::kClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	cudaLaunchConfig_t launch = {};
	launch.gridDim = dim3(
		static_cast<unsigned>(info.launched_ctas), 1, 1);
	launch.blockDim = dim3(
		static_cast<unsigned>(Traits::kNumThreads), 1, 1);
	launch.dynamicSmemBytes = info.dynamic_smem_bytes;
	launch.stream = stream;
	launch.attrs = &cluster_attribute;
	launch.numAttrs = 1;
	CUDA_CHECK(cudaLaunchKernelEx(
		&launch,
		kernel,
		tma_a,
		tma_b,
		tma_store,
		shape.m,
		shape.n,
		shape.k,
		ceil_div(shape.m, Traits::kTileM),
		ceil_div(shape.n, Traits::kTileN)));
}

template <
	fslce::DxPairAxisSm100 PairAxis,
	bool FixedRepresentative = false,
	class TmaA,
	class TmaB,
	class TmaStore>
LaunchInfo prepare_pair_launch(
		const DxProblem& shape,
		const TmaA&,
		const TmaB&,
		const TmaStore&) {
	using PairTraits =
		fslce::DxGemmPairTraitsSm100<PairAxis, 4, 32>;
	auto kernel = fslce::dx_gemm_pair_kernel_sm100<
		FixedRepresentative, PairTraits, TmaA, TmaB, TmaStore>;
	constexpr int smem_bytes =
		static_cast<int>(
			sizeof(fslce::DxGemmPairSmemSm100<PairTraits>));

	int device = 0;
	cudaDeviceProp properties = {};
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	int optin = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(
		&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (smem_bytes > optin) {
		throw std::runtime_error(
			"horizontal dX shared memory exceeds the device opt-in limit");
	}
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		smem_bytes));
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeNonPortableClusterSizeAllowed,
		1));
	if (const char* policy = std::getenv("FSLCE_DX_CLUSTER_POLICY")) {
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel,
			cudaFuncAttributeClusterSchedulingPolicyPreference,
			std::atoi(policy)));
	}

	cudaLaunchAttribute cluster_attribute = {};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = PairTraits::kClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	cudaLaunchConfig_t occupancy_config = {};
	occupancy_config.gridDim =
		dim3(static_cast<unsigned>(PairTraits::kClusterM), 1, 1);
	occupancy_config.blockDim =
		dim3(static_cast<unsigned>(PairTraits::kNumThreads), 1, 1);
	occupancy_config.dynamicSmemBytes = smem_bytes;
	occupancy_config.attrs = &cluster_attribute;
	occupancy_config.numAttrs = 1;

	int max_active_clusters = 0;
	CUDA_CHECK(cudaOccupancyMaxActiveClusters(
		&max_active_clusters, kernel, &occupancy_config));
	if (max_active_clusters <= 0) {
		throw std::runtime_error(
			"horizontal dX has no resident cluster placement");
	}

	int m_tiles = ceil_div(shape.m, PairTraits::kTileM);
	int n_tiles = ceil_div(shape.n, PairTraits::kTileN);
	int total_groups = PairAxis == fslce::DxPairAxisSm100::kN
		? m_tiles * ceil_div(n_tiles, 2)
		: ceil_div(m_tiles, 2) * n_tiles;
	int launched_clusters =
		FixedRepresentative ? 64 : total_groups;
	const char* cap = std::getenv(
		PairAxis == fslce::DxPairAxisSm100::kN
			? "FSLCE_DX_NPAIR_CLUSTERS"
			: "FSLCE_DX_MPAIR_CLUSTERS");
	if (cap != nullptr) {
		launched_clusters = std::min(
			launched_clusters, std::max(1, std::atoi(cap)));
	}
	cudaFuncAttributes attributes = {};
	CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));

	LaunchInfo info;
	info.sm_count = properties.multiProcessorCount;
	info.max_active_clusters = max_active_clusters;
	info.launched_clusters = launched_clusters;
	info.launched_ctas =
		launched_clusters * PairTraits::kClusterM;
	info.registers = attributes.numRegs;
	info.static_smem_bytes =
		static_cast<int>(attributes.sharedSizeBytes);
	info.dynamic_smem_bytes = smem_bytes;
	return info;
}

template <
	fslce::DxPairAxisSm100 PairAxis,
	bool FixedRepresentative = false,
	class TmaA,
	class TmaB,
	class TmaStore>
void launch_pair(
		const DxProblem& shape,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const TmaStore& tma_store,
		const LaunchInfo& info,
		cudaStream_t stream = nullptr) {
	using PairTraits =
		fslce::DxGemmPairTraitsSm100<PairAxis, 4, 32>;
	auto kernel = fslce::dx_gemm_pair_kernel_sm100<
		FixedRepresentative, PairTraits, TmaA, TmaB, TmaStore>;
	cudaLaunchAttribute cluster_attribute = {};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = PairTraits::kClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	cudaLaunchConfig_t launch = {};
	launch.gridDim = dim3(
		static_cast<unsigned>(info.launched_ctas), 1, 1);
	launch.blockDim = dim3(
		static_cast<unsigned>(PairTraits::kNumThreads), 1, 1);
	launch.dynamicSmemBytes = info.dynamic_smem_bytes;
	launch.stream = stream;
	launch.attrs = &cluster_attribute;
	launch.numAttrs = 1;
	CUDA_CHECK(cudaLaunchKernelEx(
		&launch,
		kernel,
		tma_a,
		tma_b,
		tma_store,
		shape.m,
		shape.n,
		shape.k,
		ceil_div(shape.m, PairTraits::kTileM),
		ceil_div(shape.n, PairTraits::kTileN)));
}

std::vector<float> cpu_reference(
		const DxProblem& shape,
		const std::vector<Element>& a,
		const std::vector<Element>& b) {
	std::vector<float> reference(
		static_cast<std::size_t>(shape.m) * shape.n, 0.0f);
	for (int row = 0; row < shape.m; ++row) {
		for (int column = 0; column < shape.n; ++column) {
			float sum = 0.0f;
			for (int kk = 0; kk < shape.k; ++kk) {
				sum = std::fma(
					static_cast<float>(
						a[static_cast<std::size_t>(row) *
							a_stride(shape) +
							kk]),
					static_cast<float>(
						b[static_cast<std::size_t>(kk) *
							b_stride(shape) +
							column]),
					sum);
			}
			reference[
				static_cast<std::size_t>(row) * shape.n +
				column] = sum;
		}
	}
	return reference;
}

struct ErrorStats {
	float max_abs = 0.0f;
	float max_rel = 0.0f;
	double mean_rel = 0.0;
	float max_padding_abs = 0.0f;
};

ErrorStats compare_stage(
		const DxProblem& shape,
		const std::vector<float>& staging,
		const std::vector<float>& reference) {
	ErrorStats stats;
	double relative_sum = 0.0;
	for (int row = 0; row < shape.m; ++row) {
		for (int column = 0; column < shape.n; ++column) {
			float got = staging[stage_index(shape, row, column)];
			float expected =
				reference[
					static_cast<std::size_t>(row) * shape.n +
					column];
			float absolute = std::fabs(got - expected);
			float relative =
				absolute / std::max(std::fabs(expected), 1.0e-3f);
			stats.max_abs = std::max(stats.max_abs, absolute);
			stats.max_rel = std::max(stats.max_rel, relative);
			relative_sum += relative;
		}
	}
	stats.mean_rel =
		relative_sum /
		(static_cast<double>(shape.m) * shape.n);

	int m_tiles = ceil_div(shape.m, Traits::kTileM);
	int n_tiles = ceil_div(shape.n, Traits::kTileN);
	for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
		for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
			int tile_linear = m_tile * n_tiles + n_tile;
			for (int rank = 0; rank < Traits::kClusterM; ++rank) {
				for (int row = 0; row < Traits::kCtaTileM; ++row) {
					int global_row =
						m_tile * Traits::kTileM +
						rank * Traits::kCtaTileM +
						row;
					for (int column = 0;
							column < Traits::kTileN;
							++column) {
						int global_column =
							n_tile * Traits::kTileN +
							column;
						if (
							global_row < shape.m &&
							global_column < shape.n) {
							continue;
						}
						std::size_t index =
							((static_cast<std::size_t>(tile_linear) *
										Traits::kClusterM +
									rank) *
										Traits::kCtaTileM +
									row) *
									Traits::kTileN +
								column;
						float value = staging[index];
						if (!std::isfinite(value)) {
							stats.max_padding_abs = INFINITY;
						} else {
							stats.max_padding_abs = std::max(
								stats.max_padding_abs,
								std::fabs(value));
						}
					}
				}
			}
		}
	}
	return stats;
}

std::vector<Element> random_bf16(
		std::size_t elements, std::mt19937& generator) {
	std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
	std::vector<Element> values(elements);
	for (Element& value : values) {
		value = Element(distribution(generator));
	}
	return values;
}

template <fslce::DxRasterOrderSm100 Raster>
std::vector<float> run_correctness_raster(
		const DxProblem& shape,
		const std::vector<Element>& host_a,
		const std::vector<Element>& host_b,
		LaunchInfo* launch_info) {
	DeviceBuffer<Element> device_a(host_a.size());
	DeviceBuffer<Element> device_b(host_b.size());
	DeviceBuffer<float> staging(stage_elements(shape));
	CUDA_CHECK(cudaMemcpy(
		device_a.data,
		host_a.data(),
		host_a.size() * sizeof(Element),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		device_b.data,
		host_b.data(),
		host_b.size() * sizeof(Element),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(
		staging.data, 0xff, staging.elements * sizeof(float)));

	auto descriptors = make_kernel_descriptors<Raster>(
		shape, device_a.data, device_b.data, staging.data);
	auto& tma_a = get<0>(descriptors);
	auto& tma_b = get<1>(descriptors);
	auto& tma_store = get<2>(descriptors);
	LaunchInfo info =
		prepare_launch<Raster>(shape, tma_a, tma_b, tma_store);
	launch_dx<Raster>(
		shape, tma_a, tma_b, tma_store, info);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<float> host_stage(staging.elements);
	CUDA_CHECK(cudaMemcpy(
		host_stage.data(),
		staging.data,
		staging.elements * sizeof(float),
		cudaMemcpyDeviceToHost));
	if (launch_info != nullptr) *launch_info = info;
	return host_stage;
}

template <fslce::DxPairAxisSm100 PairAxis>
std::vector<float> run_correctness_pair(
		const DxProblem& shape,
		const std::vector<Element>& host_a,
		const std::vector<Element>& host_b,
		LaunchInfo* launch_info) {
	DeviceBuffer<Element> device_a(host_a.size());
	DeviceBuffer<Element> device_b(host_b.size());
	DeviceBuffer<float> staging(stage_elements(shape));
	CUDA_CHECK(cudaMemcpy(
		device_a.data,
		host_a.data(),
		host_a.size() * sizeof(Element),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		device_b.data,
		host_b.data(),
		host_b.size() * sizeof(Element),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(
		staging.data, 0xff, staging.elements * sizeof(float)));

	auto descriptors =
		make_kernel_descriptors<
			fslce::DxRasterOrderSm100::kNFast>(
				shape, device_a.data, device_b.data, staging.data);
	auto& tma_a = get<0>(descriptors);
	auto& tma_b = get<1>(descriptors);
	auto& tma_store = get<2>(descriptors);
	LaunchInfo info =
		prepare_pair_launch<PairAxis>(
			shape, tma_a, tma_b, tma_store);
	launch_pair<PairAxis>(
		shape, tma_a, tma_b, tma_store, info);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<float> host_stage(staging.elements);
	CUDA_CHECK(cudaMemcpy(
		host_stage.data(),
		staging.data,
		staging.elements * sizeof(float),
		cudaMemcpyDeviceToHost));
	if (launch_info != nullptr) *launch_info = info;
	return host_stage;
}

__global__ void fill_bf16(Element* data, std::size_t elements, std::uint32_t seed) {
	std::size_t index =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x +
		threadIdx.x;
	if (index >= elements) return;
	std::uint32_t value =
		static_cast<std::uint32_t>(index) ^ seed;
	value ^= value >> 16;
	value *= 0x7feb352du;
	value ^= value >> 15;
	value *= 0x846ca68bu;
	value ^= value >> 16;
	float result =
		(static_cast<int>(value & 0xffffu) - 32768) *
		(1.0f / 131072.0f);
	data[index] = Element(result);
}

void initialize_benchmark_input(
		DeviceBuffer<Element>& buffer, std::uint32_t seed) {
	constexpr int threads = 256;
	int blocks = static_cast<int>(
		(buffer.elements + threads - 1) / threads);
	fill_bf16<<<blocks, threads>>>(
		buffer.data, buffer.elements, seed);
	CUDA_CHECK(cudaGetLastError());
}

double median(std::vector<float> values) {
	std::sort(values.begin(), values.end());
	std::size_t middle = values.size() / 2;
	return values.size() & 1
		? static_cast<double>(values[middle])
		: 0.5 *
			(static_cast<double>(values[middle - 1]) +
				static_cast<double>(values[middle]));
}

double tflops(const DxProblem& shape, double milliseconds) {
	double flops =
		2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
	return flops / (milliseconds * 1.0e9);
}

template <
	fslce::DxRasterOrderSm100 Raster,
	class TmaA,
	class TmaB,
	class TmaStore>
float time_launch(
		const DxProblem& shape,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const TmaStore& tma_store,
		const LaunchInfo& info,
		cudaEvent_t start,
		cudaEvent_t stop) {
	CUDA_CHECK(cudaEventRecord(start));
	launch_dx<Raster>(
		shape, tma_a, tma_b, tma_store, info);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(
		&milliseconds, start, stop));
	return milliseconds;
}

template <
	fslce::DxPairAxisSm100 PairAxis,
	bool FixedRepresentative = false,
	class TmaA,
	class TmaB,
	class TmaStore>
float time_pair_launch(
		const DxProblem& shape,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const TmaStore& tma_store,
		const LaunchInfo& info,
		cudaEvent_t start,
		cudaEvent_t stop) {
	CUDA_CHECK(cudaEventRecord(start));
	launch_pair<PairAxis, FixedRepresentative>(
		shape, tma_a, tma_b, tma_store, info);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(
		&milliseconds, start, stop));
	return milliseconds;
}

int environment_int(const char* name, int fallback) {
	const char* value = std::getenv(name);
	return value == nullptr ? fallback : std::max(1, std::atoi(value));
}

bool benchmark_enabled() {
	return std::getenv("FSLCE_DX_BENCH") != nullptr;
}

bool blackwell_available() {
	int device = 0;
	cudaDeviceProp properties = {};
	return cudaGetDevice(&device) == cudaSuccess &&
		cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
		properties.major == 10;
}

}  // namespace dx_test

using namespace dx_test;

TEST(FslceDxGemmSm100, Geometry) {
	EXPECT_EQ(Traits::kClusterM, 2);
	EXPECT_EQ(Traits::kTileM, 256);
	EXPECT_EQ(Traits::kTileN, 256);
	EXPECT_EQ(Traits::kTileK, 64);
	EXPECT_EQ(Traits::kStages, 4);
	EXPECT_TRUE((std::is_same_v<typename Traits::ElementAccum, float>));

	auto n_fast =
		fslce::dx_tile_coord_sm100<
			fslce::DxRasterOrderSm100::kNFast>(5, 3, 4);
	EXPECT_EQ(n_fast.m_tile, 1);
	EXPECT_EQ(n_fast.n_tile, 1);
	auto m_fast =
		fslce::dx_tile_coord_sm100<
			fslce::DxRasterOrderSm100::kMFast>(5, 3, 4);
	EXPECT_EQ(m_fast.m_tile, 2);
	EXPECT_EQ(m_fast.n_tile, 1);
}

TEST(FslceDxGemmSm100, AlignedAndRagged) {
	if (!blackwell_available()) {
		GTEST_SKIP() << "requires an SM100-family GPU";
	}
	const std::vector<DxProblem> shapes = {
		{256, 256, 64},
		{512, 512, 128},
		{273, 529, 131},
	};
	for (const DxProblem& shape : shapes) {
		std::mt19937 generator(
			static_cast<std::uint32_t>(
				shape.m * 1000003 + shape.n * 101 + shape.k));
		std::vector<Element> a = random_bf16(
			static_cast<std::size_t>(shape.m) * a_stride(shape),
			generator);
		std::vector<Element> b = random_bf16(
			static_cast<std::size_t>(shape.k) * b_stride(shape),
			generator);
		std::vector<float> reference =
			cpu_reference(shape, a, b);

		LaunchInfo n_info;
		std::vector<float> n_stage =
			run_correctness_raster<
				fslce::DxRasterOrderSm100::kNFast>(
					shape, a, b, &n_info);
		LaunchInfo m_info;
		std::vector<float> m_stage =
			run_correctness_raster<
				fslce::DxRasterOrderSm100::kMFast>(
					shape, a, b, &m_info);
		LaunchInfo npair_info;
		std::vector<float> npair_stage =
			run_correctness_pair<
				fslce::DxPairAxisSm100::kN>(
					shape, a, b, &npair_info);
		LaunchInfo mpair_info;
		std::vector<float> mpair_stage =
			run_correctness_pair<
				fslce::DxPairAxisSm100::kM>(
					shape, a, b, &mpair_info);

		ErrorStats n_error =
			compare_stage(shape, n_stage, reference);
		ErrorStats m_error =
			compare_stage(shape, m_stage, reference);
		ErrorStats npair_error =
			compare_stage(shape, npair_stage, reference);
		ErrorStats mpair_error =
			compare_stage(shape, mpair_stage, reference);
		float raster_delta = 0.0f;
		for (std::size_t i = 0; i < n_stage.size(); ++i) {
			raster_delta = std::max(
				raster_delta,
				std::fabs(n_stage[i] - m_stage[i]));
		}
		std::printf(
			"DX_CORRECTNESS M=%d N=%d K=%d "
			"N_FAST(max_abs=%.6g mean_rel=%.6g max_rel=%.6g pad=%.6g) "
			"M_FAST(max_abs=%.6g mean_rel=%.6g max_rel=%.6g pad=%.6g) "
			"N_PAIR(max_abs=%.6g mean_rel=%.6g max_rel=%.6g pad=%.6g) "
			"M_PAIR(max_abs=%.6g mean_rel=%.6g max_rel=%.6g pad=%.6g) "
			"raster_delta=%.6g smem=%d pair_smem=%d regs=%d "
			"npair_regs=%d mpair_regs=%d\n",
			shape.m,
			shape.n,
			shape.k,
			n_error.max_abs,
			n_error.mean_rel,
			n_error.max_rel,
			n_error.max_padding_abs,
			m_error.max_abs,
			m_error.mean_rel,
			m_error.max_rel,
			m_error.max_padding_abs,
			npair_error.max_abs,
			npair_error.mean_rel,
			npair_error.max_rel,
			npair_error.max_padding_abs,
			mpair_error.max_abs,
			mpair_error.mean_rel,
			mpair_error.max_rel,
			mpair_error.max_padding_abs,
			raster_delta,
			n_info.dynamic_smem_bytes,
			npair_info.dynamic_smem_bytes,
			n_info.registers,
			npair_info.registers,
			mpair_info.registers);
		EXPECT_LT(n_error.max_abs, 0.03f);
		EXPECT_LT(m_error.max_abs, 0.03f);
		EXPECT_LT(npair_error.max_abs, 0.03f);
		EXPECT_LT(mpair_error.max_abs, 0.03f);
		EXPECT_LT(n_error.mean_rel, 0.01);
		EXPECT_LT(m_error.mean_rel, 0.01);
		EXPECT_LT(npair_error.mean_rel, 0.01);
		EXPECT_LT(mpair_error.mean_rel, 0.01);
		EXPECT_LT(n_error.max_padding_abs, 1.0e-6f);
		EXPECT_LT(m_error.max_padding_abs, 1.0e-6f);
		EXPECT_LT(npair_error.max_padding_abs, 1.0e-6f);
		EXPECT_LT(mpair_error.max_padding_abs, 1.0e-6f);
		EXPECT_LT(raster_delta, 1.0e-6f);
	}
}

TEST(FslceDxGemmSm100, RasterComparison) {
	if (!blackwell_available()) {
		GTEST_SKIP() << "requires an SM100-family GPU";
	}
	if (!benchmark_enabled()) {
		GTEST_SKIP() << "set FSLCE_DX_BENCH=1 to run the raster benchmark";
	}

	const DxProblem shape = {
		environment_int("FSLCE_DX_M", 4096),
		environment_int("FSLCE_DX_N", 4096),
		environment_int("FSLCE_DX_K", 65536),
	};
	DeviceBuffer<Element> a(
		static_cast<std::size_t>(shape.m) * a_stride(shape));
	DeviceBuffer<Element> b(
		static_cast<std::size_t>(shape.k) * b_stride(shape));
	DeviceBuffer<float> staging(stage_elements(shape));
	initialize_benchmark_input(a, 0x12345678u);
	initialize_benchmark_input(b, 0x9abcdef0u);
	CUDA_CHECK(cudaDeviceSynchronize());

	auto n_descriptors =
		make_kernel_descriptors<
			fslce::DxRasterOrderSm100::kNFast>(
				shape, a.data, b.data, staging.data);
	auto& n_tma_a = get<0>(n_descriptors);
	auto& n_tma_b = get<1>(n_descriptors);
	auto& n_tma_store = get<2>(n_descriptors);
	LaunchInfo n_info =
		prepare_launch<
			fslce::DxRasterOrderSm100::kNFast>(
				shape, n_tma_a, n_tma_b, n_tma_store);

	auto m_descriptors =
		make_kernel_descriptors<
			fslce::DxRasterOrderSm100::kMFast>(
				shape, a.data, b.data, staging.data);
	auto& m_tma_a = get<0>(m_descriptors);
	auto& m_tma_b = get<1>(m_descriptors);
	auto& m_tma_store = get<2>(m_descriptors);
	LaunchInfo m_info =
		prepare_launch<
			fslce::DxRasterOrderSm100::kMFast>(
				shape, m_tma_a, m_tma_b, m_tma_store);
	std::printf(
		"DX_LAUNCH_INFO M=%d N=%d K=%d clusters=%d max_clusters=%d "
		"ctas=%d sms=%d regs_n=%d regs_m=%d smem=%d\n",
		shape.m,
		shape.n,
		shape.k,
		n_info.launched_clusters,
		n_info.max_active_clusters,
		n_info.launched_ctas,
		n_info.sm_count,
		n_info.registers,
		m_info.registers,
		n_info.dynamic_smem_bytes);
	std::fflush(stdout);

	int warmups = environment_int("FSLCE_DX_WARMUPS", 6);
	int iterations = environment_int("FSLCE_DX_ITERS", 20);
	for (int index = 0; index < warmups; ++index) {
		launch_dx<fslce::DxRasterOrderSm100::kNFast>(
			shape, n_tma_a, n_tma_b, n_tma_store, n_info);
		CUDA_CHECK(cudaDeviceSynchronize());
		launch_dx<fslce::DxRasterOrderSm100::kMFast>(
			shape, m_tma_a, m_tma_b, m_tma_store, m_info);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> n_samples;
	std::vector<float> m_samples;
	n_samples.reserve(iterations);
	m_samples.reserve(iterations);
	for (int index = 0; index < iterations; ++index) {
		if ((index & 1) == 0) {
			n_samples.push_back(
				time_launch<
					fslce::DxRasterOrderSm100::kNFast>(
						shape,
						n_tma_a,
						n_tma_b,
						n_tma_store,
						n_info,
						start,
						stop));
			m_samples.push_back(
				time_launch<
					fslce::DxRasterOrderSm100::kMFast>(
						shape,
						m_tma_a,
						m_tma_b,
						m_tma_store,
						m_info,
						start,
						stop));
		} else {
			m_samples.push_back(
				time_launch<
					fslce::DxRasterOrderSm100::kMFast>(
						shape,
						m_tma_a,
						m_tma_b,
						m_tma_store,
						m_info,
						start,
						stop));
			n_samples.push_back(
				time_launch<
					fslce::DxRasterOrderSm100::kNFast>(
						shape,
						n_tma_a,
						n_tma_b,
						n_tma_store,
						n_info,
						start,
						stop));
		}
	}
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	double n_ms = median(n_samples);
	double m_ms = median(m_samples);
	double n_tflops = tflops(shape, n_ms);
	double m_tflops = tflops(shape, m_ms);
	const char* winner = n_ms <= m_ms ? "N_FAST" : "M_FAST";
	double coverage =
		100.0 *
		std::min(
			n_info.launched_clusters,
			n_info.max_active_clusters) *
		Traits::kClusterM /
		static_cast<double>(n_info.sm_count);
	std::printf(
		"DX_RASTER_COMPARISON "
		"M=%d N=%d K=%d tile=256x256x64 stages=4 cluster=2x1 "
		"N_FAST_ms=%.6f N_FAST_TFLOPS=%.2f "
		"M_FAST_ms=%.6f M_FAST_TFLOPS=%.2f "
		"winner=%s regs=%d dynamic_smem=%d static_smem=%d "
		"max_active_clusters=%d launched_clusters=%d "
		"launched_ctas=%d sms=%d placement_coverage=%.1f%% "
		"stage_bytes=%zu warmups=%d iterations=%d\n",
		shape.m,
		shape.n,
		shape.k,
		n_ms,
		n_tflops,
		m_ms,
		m_tflops,
		winner,
		n_info.registers,
		n_info.dynamic_smem_bytes,
		n_info.static_smem_bytes,
		n_info.max_active_clusters,
		n_info.launched_clusters,
		n_info.launched_ctas,
		n_info.sm_count,
		coverage,
		staging.elements * sizeof(float),
		warmups,
		iterations);
	EXPECT_GT(n_ms, 0.0);
	EXPECT_GT(m_ms, 0.0);
}

TEST(FslceDxGemmSm100, HorizontalPair) {
	if (!blackwell_available()) {
		GTEST_SKIP() << "requires an SM100-family GPU";
	}
	if (!benchmark_enabled()) {
		GTEST_SKIP() << "set FSLCE_DX_BENCH=1 to run the dX benchmark";
	}

	const DxProblem shape = {
		environment_int("FSLCE_DX_M", 4096),
		environment_int("FSLCE_DX_N", 4096),
		environment_int("FSLCE_DX_K", 65536),
	};
	DeviceBuffer<Element> a(
		static_cast<std::size_t>(shape.m) * a_stride(shape));
	DeviceBuffer<Element> b(
		static_cast<std::size_t>(shape.k) * b_stride(shape));
	DeviceBuffer<float> staging(stage_elements(shape));
	initialize_benchmark_input(a, 0x12345678u);
	initialize_benchmark_input(b, 0x9abcdef0u);
	CUDA_CHECK(cudaDeviceSynchronize());

	auto descriptors =
		make_kernel_descriptors<
			fslce::DxRasterOrderSm100::kNFast>(
				shape, a.data, b.data, staging.data);
	auto& tma_a = get<0>(descriptors);
	auto& tma_b = get<1>(descriptors);
	auto& tma_store = get<2>(descriptors);
	LaunchInfo n_info =
		prepare_pair_launch<
			fslce::DxPairAxisSm100::kN, true>(
			shape, tma_a, tma_b, tma_store);
	LaunchInfo m_info =
		prepare_pair_launch<
			fslce::DxPairAxisSm100::kM, true>(
			shape, tma_a, tma_b, tma_store);
	std::printf(
		"DX_PAIR_LAUNCH_INFO M=%d N=%d K=%d "
		"n_clusters=%d m_clusters=%d max_clusters=%d "
		"n_ctas=%d m_ctas=%d sms=%d n_regs=%d m_regs=%d smem=%d\n",
		shape.m,
		shape.n,
		shape.k,
		n_info.launched_clusters,
		m_info.launched_clusters,
		n_info.max_active_clusters,
		n_info.launched_ctas,
		m_info.launched_ctas,
		n_info.sm_count,
		n_info.registers,
		m_info.registers,
		n_info.dynamic_smem_bytes);
	std::fflush(stdout);

	int warmups = environment_int("FSLCE_DX_WARMUPS", 6);
	int iterations = environment_int("FSLCE_DX_ITERS", 20);
	for (int index = 0; index < warmups; ++index) {
		launch_pair<fslce::DxPairAxisSm100::kN, true>(
			shape, tma_a, tma_b, tma_store, n_info);
		CUDA_CHECK(cudaDeviceSynchronize());
		launch_pair<fslce::DxPairAxisSm100::kM, true>(
			shape, tma_a, tma_b, tma_store, m_info);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> n_samples;
	std::vector<float> m_samples;
	n_samples.reserve(iterations);
	m_samples.reserve(iterations);
	for (int index = 0; index < iterations; ++index) {
		if ((index & 1) == 0) {
			n_samples.push_back(
				time_pair_launch<
					fslce::DxPairAxisSm100::kN, true>(
						shape,
						tma_a,
						tma_b,
						tma_store,
						n_info,
						start,
						stop));
			m_samples.push_back(
				time_pair_launch<
					fslce::DxPairAxisSm100::kM, true>(
						shape,
						tma_a,
						tma_b,
						tma_store,
						m_info,
						start,
						stop));
		} else {
			m_samples.push_back(
				time_pair_launch<
					fslce::DxPairAxisSm100::kM, true>(
						shape,
						tma_a,
						tma_b,
						tma_store,
						m_info,
						start,
						stop));
			n_samples.push_back(
				time_pair_launch<
					fslce::DxPairAxisSm100::kN, true>(
						shape,
						tma_a,
						tma_b,
						tma_store,
						n_info,
						start,
						stop));
		}
	}
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	double n_ms = median(n_samples);
	double m_ms = median(m_samples);
	double n_tflops = tflops(shape, n_ms);
	double m_tflops = tflops(shape, m_ms);
	const char* winner = n_ms <= m_ms ? "N_PAIR" : "M_PAIR";
	double n_coverage =
		100.0 *
		std::min(
			n_info.launched_clusters,
			n_info.max_active_clusters) *
		NPairTraits::kClusterM /
		static_cast<double>(n_info.sm_count);
	double m_coverage =
		100.0 *
		std::min(
			m_info.launched_clusters,
			m_info.max_active_clusters) *
		MPairTraits::kClusterM /
		static_cast<double>(m_info.sm_count);
	std::printf(
		"DX_PAIR_COMPARISON "
		"M=%d N=%d K=%d logical_tile=256x256x64 stages=4 cluster=2x1 "
		"N_PAIR_ms=%.6f N_PAIR_TFLOPS=%.2f "
		"M_PAIR_ms=%.6f M_PAIR_TFLOPS=%.2f winner=%s "
		"N_regs=%d M_regs=%d dynamic_smem=%d static_smem=%d "
		"max_active_clusters=%d N_clusters=%d M_clusters=%d "
		"N_ctas=%d M_ctas=%d sms=%d "
		"N_placement=%.1f%% M_placement=%.1f%% "
		"stage_bytes=%zu warmups=%d iterations=%d\n",
		shape.m,
		shape.n,
		shape.k,
		n_ms,
		n_tflops,
		m_ms,
		m_tflops,
		winner,
		n_info.registers,
		m_info.registers,
		n_info.dynamic_smem_bytes,
		n_info.static_smem_bytes,
		n_info.max_active_clusters,
		n_info.launched_clusters,
		m_info.launched_clusters,
		n_info.launched_ctas,
		m_info.launched_ctas,
		n_info.sm_count,
		n_coverage,
		m_coverage,
		staging.elements * sizeof(float),
		warmups,
		iterations);
	EXPECT_GT(n_ms, 0.0);
	EXPECT_GT(m_ms, 0.0);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	if (
		GTEST_FLAG_GET(filter) == "*" &&
		!GTEST_FLAG_GET(list_tests)) {
		std::string filter =
			"FslceDxGemmSm100.Geometry:"
			"FslceDxGemmSm100.AlignedAndRagged";
		if (benchmark_enabled()) {
			filter +=
				":FslceDxGemmSm100.RasterComparison"
				":FslceDxGemmSm100.HorizontalPair";
		}
		GTEST_FLAG_SET(filter, filter);
	}
	return RUN_ALL_TESTS();
}
