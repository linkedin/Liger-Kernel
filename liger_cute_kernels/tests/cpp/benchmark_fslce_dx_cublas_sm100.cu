#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutlass/numeric_types.h>

namespace {

using Element = cutlass::bfloat16_t;

void check_cuda(cudaError_t status, const char* expression) {
	if (status != cudaSuccess) {
		throw std::runtime_error(
			std::string(expression) + ": " + cudaGetErrorString(status));
	}
}

void check_cublas(cublasStatus_t status, const char* expression) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error(
			std::string(expression) + ": cuBLAS status " +
			std::to_string(static_cast<int>(status)));
	}
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr)

template <class T>
struct DeviceBuffer {
	T* data = nullptr;
	std::size_t count = 0;

	explicit DeviceBuffer(std::size_t elements) : count(elements) {
		CUDA_CHECK(cudaMalloc(&data, elements * sizeof(T)));
	}
	~DeviceBuffer() {
		if (data != nullptr) cudaFree(data);
	}
};

__global__ void fill_bf16(
		Element* data, std::size_t elements, std::uint32_t seed) {
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
	data[index] = Element(
		(static_cast<int>(value & 0xffffu) - 32768) *
		(1.0f / 131072.0f));
}

double median(std::vector<float> samples) {
	std::sort(samples.begin(), samples.end());
	std::size_t middle = samples.size() / 2;
	return samples.size() & 1
		? samples[middle]
		: 0.5 * (samples[middle - 1] + samples[middle]);
}

int environment_int(const char* name, int fallback) {
	const char* value = std::getenv(name);
	return value == nullptr ? fallback : std::max(1, std::atoi(value));
}

int algo_attribute(
		const cublasLtMatmulAlgo_t& algorithm,
		cublasLtMatmulAlgoConfigAttributes_t attribute) {
	int value = -1;
	std::size_t written = 0;
	CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
		&algorithm, attribute, &value, sizeof(value), &written));
	return value;
}

}  // namespace

int main() {
	try {
		const int m = environment_int("FSLCE_DX_M", 4096);
		const int n = environment_int("FSLCE_DX_N", 4096);
		const int k = environment_int("FSLCE_DX_K", 65536);
		const int warmups =
			environment_int("FSLCE_DX_WARMUPS", 20);
		const int iterations =
			environment_int("FSLCE_DX_ITERS", 50);
		constexpr std::size_t kWorkspaceBytes =
			1024ull * 1024ull * 1024ull;

		DeviceBuffer<Element> a(
			static_cast<std::size_t>(m) * k);
		DeviceBuffer<Element> b(
			static_cast<std::size_t>(k) * n);
		DeviceBuffer<float> d(
			static_cast<std::size_t>(m) * n);
		DeviceBuffer<std::uint8_t> workspace(kWorkspaceBytes);
		constexpr int threads = 256;
		fill_bf16<<<
			static_cast<int>((a.count + threads - 1) / threads),
			threads>>>(a.data, a.count, 0x12345678u);
		fill_bf16<<<
			static_cast<int>((b.count + threads - 1) / threads),
			threads>>>(b.data, b.count, 0x9abcdef0u);
		CUDA_CHECK(cudaDeviceSynchronize());

		cublasLtHandle_t handle = nullptr;
		cublasLtMatmulDesc_t operation = nullptr;
		cublasLtMatrixLayout_t a_layout = nullptr;
		cublasLtMatrixLayout_t b_layout = nullptr;
		cublasLtMatrixLayout_t d_layout = nullptr;
		cublasLtMatmulPreference_t preference = nullptr;
		CUBLAS_CHECK(cublasLtCreate(&handle));
		CUBLAS_CHECK(cublasLtMatmulDescCreate(
			&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
		cublasOperation_t no_transpose = CUBLAS_OP_N;
		CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
			operation,
			CUBLASLT_MATMUL_DESC_TRANSA,
			&no_transpose,
			sizeof(no_transpose)));
		CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
			operation,
			CUBLASLT_MATMUL_DESC_TRANSB,
			&no_transpose,
			sizeof(no_transpose)));
		CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
			&a_layout, CUDA_R_16BF, m, k, k));
		CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
			&b_layout, CUDA_R_16BF, k, n, n));
		CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
			&d_layout, CUDA_R_32F, m, n, n));
		cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
		CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
			a_layout,
			CUBLASLT_MATRIX_LAYOUT_ORDER,
			&row_order,
			sizeof(row_order)));
		CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
			b_layout,
			CUBLASLT_MATRIX_LAYOUT_ORDER,
			&row_order,
			sizeof(row_order)));
		CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
			d_layout,
			CUBLASLT_MATRIX_LAYOUT_ORDER,
			&row_order,
			sizeof(row_order)));
		CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
		CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
			preference,
			CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
			&kWorkspaceBytes,
			sizeof(kWorkspaceBytes)));

		constexpr int kRequestedAlgorithms = 32;
		cublasLtMatmulHeuristicResult_t heuristics[
			kRequestedAlgorithms] = {};
		int returned = 0;
		CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
			handle,
			operation,
			a_layout,
			b_layout,
			d_layout,
			d_layout,
			preference,
			kRequestedAlgorithms,
			heuristics,
			&returned));
		if (returned == 0) {
			throw std::runtime_error("cuBLASLt returned no algorithms");
		}

		float alpha = 1.0f;
		float beta = 0.0f;
		cudaEvent_t start = nullptr;
		cudaEvent_t stop = nullptr;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));

		int best = -1;
		double best_ms = 1.0e30;
		int first_candidate = 0;
		int candidate_end = returned;
		if (std::getenv("FSLCE_DX_CUBLAS_SKIP_TUNE") != nullptr) {
			candidate_end = std::min(returned, 1);
		}
		for (int index = first_candidate; index < candidate_end; ++index) {
			if (
				heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
				heuristics[index].workspaceSize > kWorkspaceBytes) {
				continue;
			}
			bool valid = true;
			for (int repeat = 0; repeat < 3; ++repeat) {
				cublasStatus_t status = cublasLtMatmul(
					handle,
					operation,
					&alpha,
					a.data,
					a_layout,
					b.data,
					b_layout,
					&beta,
					d.data,
					d_layout,
					d.data,
					d_layout,
					&heuristics[index].algo,
					workspace.data,
					heuristics[index].workspaceSize,
					nullptr);
				if (status != CUBLAS_STATUS_SUCCESS) {
					valid = false;
					break;
				}
			}
			if (!valid || cudaDeviceSynchronize() != cudaSuccess) {
				cudaGetLastError();
				continue;
			}
			std::vector<float> samples;
			for (int repeat = 0; repeat < 5; ++repeat) {
				CUDA_CHECK(cudaEventRecord(start));
				cublasStatus_t status = cublasLtMatmul(
					handle,
					operation,
					&alpha,
					a.data,
					a_layout,
					b.data,
					b_layout,
					&beta,
					d.data,
					d_layout,
					d.data,
					d_layout,
					&heuristics[index].algo,
					workspace.data,
					heuristics[index].workspaceSize,
					nullptr);
				if (status != CUBLAS_STATUS_SUCCESS) {
					valid = false;
					break;
				}
				CUDA_CHECK(cudaEventRecord(stop));
				CUDA_CHECK(cudaEventSynchronize(stop));
				float milliseconds = 0.0f;
				CUDA_CHECK(cudaEventElapsedTime(
					&milliseconds, start, stop));
				samples.push_back(milliseconds);
			}
			if (!valid) continue;
			double milliseconds = median(samples);
			if (milliseconds < best_ms) {
				best_ms = milliseconds;
				best = index;
			}
		}
		if (best < 0) {
			throw std::runtime_error(
				"no cuBLASLt algorithm executed successfully");
		}

		for (int index = 0; index < warmups; ++index) {
			CUBLAS_CHECK(cublasLtMatmul(
				handle,
				operation,
				&alpha,
				a.data,
				a_layout,
				b.data,
				b_layout,
				&beta,
				d.data,
				d_layout,
				d.data,
				d_layout,
				&heuristics[best].algo,
				workspace.data,
				heuristics[best].workspaceSize,
				nullptr));
		}
		CUDA_CHECK(cudaDeviceSynchronize());

		std::vector<float> samples;
		samples.reserve(iterations);
		for (int index = 0; index < iterations; ++index) {
			CUDA_CHECK(cudaEventRecord(start));
			CUBLAS_CHECK(cublasLtMatmul(
				handle,
				operation,
				&alpha,
				a.data,
				a_layout,
				b.data,
				b_layout,
				&beta,
				d.data,
				d_layout,
				d.data,
				d_layout,
				&heuristics[best].algo,
				workspace.data,
				heuristics[best].workspaceSize,
				nullptr));
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			float milliseconds = 0.0f;
			CUDA_CHECK(cudaEventElapsedTime(
				&milliseconds, start, stop));
			samples.push_back(milliseconds);
		}
		double milliseconds = median(samples);
		double flops =
			2.0 * static_cast<double>(m) * n * k;
		double achieved_tflops =
			flops / (milliseconds * 1.0e9);
		int algorithm_id = algo_attribute(
			heuristics[best].algo,
			CUBLASLT_ALGO_CONFIG_ID);
		int tile_id = algo_attribute(
			heuristics[best].algo,
			CUBLASLT_ALGO_CONFIG_TILE_ID);
		int stages_id = algo_attribute(
			heuristics[best].algo,
			CUBLASLT_ALGO_CONFIG_STAGES_ID);
		int split_k = algo_attribute(
			heuristics[best].algo,
			CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
		std::printf(
			"DX_CUBLASLT M=%d N=%d K=%d ms=%.6f TFLOPS=%.2f "
			"algo=%d tile=%d stages=%d split_k=%d workspace=%zu "
			"warmups=%d iterations=%d\n",
			m,
			n,
			k,
			milliseconds,
			achieved_tflops,
			algorithm_id,
			tile_id,
			stages_id,
			split_k,
			heuristics[best].workspaceSize,
			warmups,
			iterations);

		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));
		CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
		CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_layout));
		CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
		CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
		CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
		CUBLAS_CHECK(cublasLtDestroy(handle));
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "DX_CUBLASLT_ERROR %s\n", error.what());
		return 1;
	}
}
