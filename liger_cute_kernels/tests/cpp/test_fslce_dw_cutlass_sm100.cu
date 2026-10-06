// TP=1 correctness and throughput benchmark for the SM100 dW store path.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "backward_dw_store_sm100.cuh"

namespace fslce_dw_cutlass_test {

using StoreConfig =
	liger::fused_scaled_linear_cross_entropy::BackwardDwStoreSm100;
using Element = StoreConfig::Element;
using GemmKernel = StoreConfig::Kernel;
using Gemm = StoreConfig::Gemm;
using RasterOrder = StoreConfig::RasterOrder;

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

__global__ void fill_random(
		Element* data, std::size_t count, std::uint32_t seed) {
	std::size_t index =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	std::uint32_t value =
		static_cast<std::uint32_t>(index) * 747796405u +
		seed * 2891336453u;
	value = ((value >> ((value >> 28) + 4)) ^ value) * 277803737u;
	value = (value >> 22) ^ value;
	float unit =
		static_cast<float>(value & 0xffffu) * (1.0f / 65535.0f);
	data[index] = Element((unit - 0.5f) * 0.0625f);
}

struct ErrorStats {
	unsigned long long mismatches;
	unsigned int max_abs_bits;
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
	float absolute = fabsf(lhs - rhs);
	if (reinterpret_cast<const std::uint16_t*>(got)[index] !=
		reinterpret_cast<const std::uint16_t*>(reference)[index]) {
		atomicAdd(&stats->mismatches, 1ull);
	}
	atomicMax(&stats->max_abs_bits, __float_as_uint(absolute));
	atomicAdd(&stats->sum_abs, static_cast<double>(absolute));
	atomicAdd(
		&stats->sum_sq,
		static_cast<double>(absolute) * static_cast<double>(absolute));
}

float float_from_bits(unsigned int bits) {
	float value;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

double median(std::vector<float>& values) {
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

void run(
		int warmups,
		int iterations,
		int m,
		int n,
		int k,
		int sm_count,
		int swizzle,
		RasterOrder raster) {
	std::size_t a_elements = static_cast<std::size_t>(m) * k;
	std::size_t b_elements = static_cast<std::size_t>(k) * n;
	std::size_t d_elements = static_cast<std::size_t>(m) * n;

	Element* a = nullptr;
	Element* b = nullptr;
	Element* output = nullptr;
	Element* reference = nullptr;
	cudaStream_t stream = nullptr;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMalloc(&a, a_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&b, b_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&output, d_elements * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&reference, d_elements * sizeof(Element)));

	fill_random<<<
		static_cast<unsigned>((a_elements + 255) / 256),
		256>>>(a, a_elements, 17u);
	fill_random<<<
		static_cast<unsigned>((b_elements + 255) / 256),
		256>>>(b, b_elements, 29u);
	CUDA_CHECK(cudaMemset(output, 0, d_elements * sizeof(Element)));
	CUDA_CHECK(cudaMemset(reference, 0, d_elements * sizeof(Element)));
	CUDA_CHECK(cudaDeviceSynchronize());

	auto arguments =
		StoreConfig::arguments(a, b, output, m, n, k, sm_count);
	arguments.scheduler.max_swizzle_size = swizzle;
	arguments.scheduler.raster_order = raster;

	Gemm gemm;
	CUDA_CHECK(StoreConfig::prepare_kernel());
	std::size_t workspace_size = Gemm::get_workspace_size(arguments);
	void* workspace = nullptr;
	if (workspace_size != 0) {
		CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
	}
	CUTLASS_CHECK(gemm.can_implement(arguments));
	CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
	CUTLASS_CHECK(StoreConfig::run(
		a, b, output, m, n, k, sm_count, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	cublasHandle_t cublas = nullptr;
	CUBLAS_CHECK(cublasCreate(&cublas));
	CUBLAS_CHECK(cublasSetStream(cublas, stream));
	float alpha = 1.0f;
	float beta = 0.0f;
	auto run_cublas = [&]() {
		CUBLAS_CHECK(cublasGemmEx(
			cublas,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			n,
			m,
			k,
			&alpha,
			b,
			CUDA_R_16BF,
			n,
			a,
			CUDA_R_16BF,
			m,
			&beta,
			reference,
			CUDA_R_16BF,
			n,
			CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	};
	run_cublas();
	CUDA_CHECK(cudaDeviceSynchronize());

	ErrorStats* device_stats = nullptr;
	CUDA_CHECK(cudaMalloc(&device_stats, sizeof(ErrorStats)));
	CUDA_CHECK(cudaMemset(device_stats, 0, sizeof(ErrorStats)));
	compare_bf16<<<
		static_cast<unsigned>((d_elements + 255) / 256),
		256>>>(output, reference, d_elements, device_stats);
	ErrorStats stats{};
	CUDA_CHECK(cudaMemcpy(
		&stats, device_stats, sizeof(stats), cudaMemcpyDeviceToHost));

	for (int iteration = 0; iteration < warmups; ++iteration) {
		CUTLASS_CHECK(gemm.run(stream));
		run_cublas();
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> cutlass_samples;
	std::vector<float> cublas_samples;
	cutlass_samples.reserve(iterations);
	cublas_samples.reserve(iterations);
	auto time_cutlass = [&]() {
		CUDA_CHECK(cudaEventRecord(start, stream));
		CUTLASS_CHECK(gemm.run(stream));
		CUDA_CHECK(cudaEventRecord(stop, stream));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float elapsed = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
		return elapsed;
	};
	auto time_cublas = [&]() {
		CUDA_CHECK(cudaEventRecord(start, stream));
		run_cublas();
		CUDA_CHECK(cudaEventRecord(stop, stream));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float elapsed = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
		return elapsed;
	};
	for (int iteration = 0; iteration < iterations; ++iteration) {
		if ((iteration & 1) == 0) {
			cutlass_samples.push_back(time_cutlass());
			cublas_samples.push_back(time_cublas());
		} else {
			cublas_samples.push_back(time_cublas());
			cutlass_samples.push_back(time_cutlass());
		}
	}
	double cutlass_ms = median(cutlass_samples);
	double cublas_ms = median(cublas_samples);
	double cutlass_tflops =
		2.0 * static_cast<double>(m) * n * k /
		(cutlass_ms * 1.0e9);
	double cublas_tflops =
		2.0 * static_cast<double>(m) * n * k /
		(cublas_ms * 1.0e9);
	std::printf(
		"DW_CUTLASS mode=store raster=%s swizzle=%d sm_count=%d "
		"cluster_policy=spread perf_input=random_bf16 "
		"M=%d N=%d K=%d cutlass_ms=%.4f cutlass_tflops=%.2f "
		"cublas_ms=%.4f cublas_tflops=%.2f "
		"smem=%zu workspace=%zu mismatch=%.6f max_abs=%.6g "
		"mean_abs=%.6g rmse=%.6g\n",
		raster == RasterOrder::AlongN ? "N" : "M",
		swizzle,
		sm_count,
		m,
		n,
		k,
		cutlass_ms,
		cutlass_tflops,
		cublas_ms,
		cublas_tflops,
		sizeof(typename GemmKernel::SharedStorage),
		workspace_size,
		static_cast<double>(stats.mismatches) /
			static_cast<double>(d_elements),
		float_from_bits(stats.max_abs_bits),
		stats.sum_abs / static_cast<double>(d_elements),
		std::sqrt(stats.sum_sq / static_cast<double>(d_elements)));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaStreamDestroy(stream);
	cublasDestroy(cublas);
	cudaFree(device_stats);
	cudaFree(workspace);
	cudaFree(reference);
	cudaFree(output);
	cudaFree(b);
	cudaFree(a);
}

}  // namespace fslce_dw_cutlass_test

int main(int argc, char** argv) {
	using namespace fslce_dw_cutlass_test;
	int warmups = argc > 1 ? std::atoi(argv[1]) : 5;
	int iterations = argc > 2 ? std::atoi(argv[2]) : 21;
	int swizzle = argc > 3 ? std::atoi(argv[3]) : 1;
	int sm_count = argc > 4 ? std::atoi(argv[4]) : 148;
	RasterOrder raster =
		argc > 5 && argv[5][0] == 'M'
		? RasterOrder::AlongM
		: RasterOrder::AlongN;
	bool k4096_only = argc > 6 &&
		std::strcmp(argv[6], "k4096-only") == 0;
	run(
		warmups,
		iterations,
		65536,
		4096,
		4096,
		sm_count,
		swizzle,
		raster);
	if (k4096_only) return 0;
	run(
		warmups,
		iterations,
		65536,
		4096,
		1024,
		sm_count,
		swizzle,
		raster);
	run(0, 1, 520, 264, 4096, sm_count, swizzle, raster);
	return 0;
}
