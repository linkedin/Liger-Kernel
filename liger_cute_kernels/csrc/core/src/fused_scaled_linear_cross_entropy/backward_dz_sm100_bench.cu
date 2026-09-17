// Standalone correctness and performance harness for backward_dz_gemm_sm100.
//
// Build on an SM100/SM103 host:
//   nvcc -std=c++17 -O3 -arch=sm_100f --use_fast_math \
//     --extra-device-vectorization --fmad=true --prec-div=false \
//     --prec-sqrt=false -Xptxas=-O3,-v \
//     -I${CUTLASS_HOME}/include -I${CUTLASS_HOME}/tools/util/include \
//     -I../moe -I. backward_dz_sm100_bench.cu \
//     -lcublasLt -lcublas -o backward_dz_sm100_bench

#include "backward_dz_gemm_sm100.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace dz =
	liger::fused_scaled_linear_cross_entropy::dz_sm100;
using Element = dz::Traits::Element;

namespace {

[[noreturn]] void fail(const char* what, const char* detail) {
	std::fprintf(stderr, "FAIL: %s: %s\n", what, detail);
	std::exit(1);
}

void check_cuda(cudaError_t status, const char* what) {
	if (status != cudaSuccess) {
		fail(what, cudaGetErrorString(status));
	}
}

const char* cublas_status_string(cublasStatus_t status) {
	switch (status) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
		default:
			return "CUBLAS_STATUS_UNKNOWN";
	}
}

void check_cublas(cublasStatus_t status, const char* what) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		fail(what, cublas_status_string(status));
	}
}

template <class T>
class DeviceBuffer {
public:
	DeviceBuffer() = default;
	explicit DeviceBuffer(std::size_t count) : count_(count) {
		check_cuda(
			cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
			"cudaMalloc");
	}
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
	DeviceBuffer(DeviceBuffer&& other) noexcept
		: pointer_(other.pointer_), count_(other.count_) {
		other.pointer_ = nullptr;
		other.count_ = 0;
	}
	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
		if (this != &other) {
			if (pointer_ != nullptr) cudaFree(pointer_);
			pointer_ = other.pointer_;
			count_ = other.count_;
			other.pointer_ = nullptr;
			other.count_ = 0;
		}
		return *this;
	}
	~DeviceBuffer() {
		if (pointer_ != nullptr) cudaFree(pointer_);
	}

	T* get() { return pointer_; }
	const T* get() const { return pointer_; }
	std::size_t size() const { return count_; }
	std::size_t bytes() const { return count_ * sizeof(T); }

private:
	T* pointer_ = nullptr;
	std::size_t count_ = 0;
};

template <class T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
	if (destination.size() != source.size()) {
		fail("copy_to_device", "size mismatch");
	}
	check_cuda(
		cudaMemcpy(
			destination.get(),
			source.data(),
			destination.bytes(),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(H2D)");
}

template <class T>
std::vector<T> copy_to_host(const DeviceBuffer<T>& source) {
	std::vector<T> result(source.size());
	check_cuda(
		cudaMemcpy(
			result.data(),
			source.get(),
			source.bytes(),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(D2H)");
	return result;
}

__global__ void fill_random_bf16(
		Element* data,
		std::size_t count,
		std::uint32_t seed) {
	for (std::size_t index =
				static_cast<std::size_t>(blockIdx.x) * blockDim.x +
				threadIdx.x;
			index < count;
			index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
		std::uint32_t value =
			seed ^ static_cast<std::uint32_t>(index * 0x9e3779b9u);
		value ^= value << 13;
		value ^= value >> 17;
		value ^= value << 5;
		float unit =
			static_cast<float>(value & 0x00ffffffu) /
			static_cast<float>(0x01000000u);
		data[index] = Element((unit - 0.5f) * 0.06f);
	}
}

double median(std::vector<float> values) {
	std::sort(values.begin(), values.end());
	std::size_t middle = values.size() / 2;
	if ((values.size() & 1u) != 0u) return values[middle];
	return 0.5 * (
		static_cast<double>(values[middle - 1]) +
		static_cast<double>(values[middle]));
}

struct Shape {
	int m;
	int n;
	int k;
	const char* name;
};

struct HostProblem {
	Shape shape;
	int padded_m;
	int padded_n;
	std::int64_t vocab_start;
	std::int64_t ignore_index;
	float inverse_temperature;
	std::vector<Element> x;
	std::vector<Element> weight;
	std::vector<std::int64_t> target;
	std::vector<float> grad_output;
	std::vector<float> lse;
	std::vector<float> entropy;
	std::vector<float> entropy_grad;
	std::vector<float> logits;
};

float input_value(std::size_t index, float scale, float phase) {
	return scale * (
		std::sin(static_cast<float>(index % 104729u) * 0.0137f + phase) +
		0.5f * std::cos(static_cast<float>(index % 65521u) * 0.0071f));
}

HostProblem make_host_problem(Shape shape) {
	HostProblem problem;
	problem.shape = shape;
	problem.padded_m =
		dz::ceil_div_int(shape.m, dz::Config::kTileM) *
		dz::Config::kTileM;
	problem.padded_n =
		dz::ceil_div_int(shape.n, dz::Config::kTileN) *
		dz::Config::kTileN;
	problem.vocab_start = 1000;
	problem.ignore_index = -100;
	problem.inverse_temperature = 1.25f;
	problem.x.resize(
		static_cast<std::size_t>(shape.m) *
		static_cast<std::size_t>(shape.k));
	problem.weight.resize(
		static_cast<std::size_t>(shape.n) *
		static_cast<std::size_t>(shape.k));
	problem.target.resize(shape.m);
	problem.grad_output.resize(shape.m);
	problem.lse.resize(shape.m);
	problem.entropy.resize(shape.m);
	problem.entropy_grad.resize(shape.m);
	problem.logits.resize(
		static_cast<std::size_t>(shape.m) *
		static_cast<std::size_t>(shape.n));

	for (std::size_t i = 0; i < problem.x.size(); ++i) {
		problem.x[i] = Element(input_value(i, 0.035f, 0.3f));
	}
	for (std::size_t i = 0; i < problem.weight.size(); ++i) {
		problem.weight[i] = Element(input_value(i, 0.031f, 1.7f));
	}
	for (int row = 0; row < shape.m; ++row) {
		if (row % 11 == 0) {
			problem.target[row] = problem.ignore_index;
		} else if (row % 13 == 0) {
			problem.target[row] =
				problem.vocab_start + shape.n + 17;
		} else {
			problem.target[row] =
				problem.vocab_start +
				((row * 37 + 5) % shape.n);
		}
		problem.grad_output[row] =
			0.5f + 0.01f * static_cast<float>(row % 17);
		problem.entropy_grad[row] =
			-0.15f + 0.02f * static_cast<float>(row % 19);
	}

	for (int row = 0; row < shape.m; ++row) {
		float maximum = -std::numeric_limits<float>::infinity();
		for (int column = 0; column < shape.n; ++column) {
			float accumulator = 0.0f;
			for (int inner = 0; inner < shape.k; ++inner) {
				accumulator = std::fma(
					static_cast<float>(
						problem.x[
							static_cast<std::size_t>(row) *
								shape.k +
							inner]),
					static_cast<float>(
						problem.weight[
							static_cast<std::size_t>(column) *
								shape.k +
							inner]),
					accumulator);
			}
			problem.logits[
				static_cast<std::size_t>(row) * shape.n + column] =
				accumulator;
			maximum = std::max(
				maximum,
				accumulator * problem.inverse_temperature);
		}
		float sum = 0.0f;
		float weighted = 0.0f;
		for (int column = 0; column < shape.n; ++column) {
			float scaled =
				problem.logits[
					static_cast<std::size_t>(row) * shape.n + column] *
				problem.inverse_temperature;
			float probability_numerator = std::exp(scaled - maximum);
			sum += probability_numerator;
			weighted += probability_numerator * scaled;
		}
		problem.lse[row] = maximum + std::log(sum);
		problem.entropy[row] =
			problem.lse[row] - weighted / sum;
	}
	return problem;
}

std::vector<float> make_reference(
		const HostProblem& problem,
		dz::EpilogueMode mode) {
	const Shape& shape = problem.shape;
	std::vector<float> reference(
		static_cast<std::size_t>(problem.padded_m) *
			static_cast<std::size_t>(problem.padded_n),
		0.0f);
	for (int row = 0; row < shape.m; ++row) {
		bool ignored =
			problem.target[row] == problem.ignore_index;
		float scale = ignored ? 0.0f : problem.grad_output[row];
		std::int64_t local_target =
			problem.target[row] - problem.vocab_start;
		if (
			local_target < 0 ||
			local_target >= static_cast<std::int64_t>(shape.n)) {
			local_target = -1;
		}
		for (int column = 0; column < shape.n; ++column) {
			float logit =
				problem.logits[
					static_cast<std::size_t>(row) * shape.n + column];
			float value = logit;
			if (mode != dz::EpilogueMode::kRawGemm) {
				float probability = ignored
					? 0.0f
					: std::exp(
						logit * problem.inverse_temperature -
						problem.lse[row]);
				if (
					mode ==
					dz::EpilogueMode::kSoftmaxGradientEntropy) {
					float entropy_scale =
						ignored ? 0.0f : problem.entropy_grad[row];
					float bias =
						(problem.lse[row] - problem.entropy[row]) *
							entropy_scale +
						scale;
					float slope =
						-problem.inverse_temperature * entropy_scale;
					value = probability * std::fma(logit, slope, bias);
				} else {
					value = probability * scale;
				}
				if (column == local_target) value -= scale;
				value *= problem.inverse_temperature;
			}
			reference[
				static_cast<std::size_t>(row) * problem.padded_n + column] =
				static_cast<float>(Element(value));
		}
	}
	return reference;
}

template <dz::EpilogueMode Mode>
void run_correctness_case(const Shape& shape) {
	HostProblem host = make_host_problem(shape);
	DeviceBuffer<Element> x(host.x.size());
	DeviceBuffer<Element> weight(host.weight.size());
	DeviceBuffer<std::int64_t> target(host.target.size());
	DeviceBuffer<float> grad_output(host.grad_output.size());
	DeviceBuffer<float> lse(host.lse.size());
	DeviceBuffer<float> entropy(host.entropy.size());
	DeviceBuffer<float> entropy_grad(host.entropy_grad.size());
	DeviceBuffer<Element> output(
		static_cast<std::size_t>(host.padded_m) *
		static_cast<std::size_t>(host.padded_n));

	copy_to_device(x, host.x);
	copy_to_device(weight, host.weight);
	copy_to_device(target, host.target);
	copy_to_device(grad_output, host.grad_output);
	copy_to_device(lse, host.lse);
	copy_to_device(entropy, host.entropy);
	copy_to_device(entropy_grad, host.entropy_grad);
	check_cuda(
		cudaMemset(output.get(), 0x7f, output.bytes()),
		"cudaMemset(output)");

	dz::Params params;
	params.x = x.get();
	params.weight = weight.get();
	params.target = target.get();
	params.grad_output = grad_output.get();
	params.lse = lse.get();
	params.entropy = entropy.get();
	params.entropy_grad = entropy_grad.get();
	params.output = output.get();
	params.tokens = shape.m;
	params.hidden = shape.k;
	params.local_vocab = shape.n;
	params.padded_tokens = host.padded_m;
	params.padded_vocab = host.padded_n;
	params.vocab_start = host.vocab_start;
	params.ignore_index = host.ignore_index;
	params.inverse_temperature = host.inverse_temperature;

	check_cuda(dz::launch<Mode>(params, nullptr), "dZ correctness launch");
	check_cuda(cudaDeviceSynchronize(), "dZ correctness synchronize");
	std::vector<Element> actual_bf16 = copy_to_host(output);
	std::vector<float> reference = make_reference(host, Mode);

	double max_abs = 0.0;
	double max_rel = 0.0;
	double mean_abs = 0.0;
	double ignored_max = 0.0;
	double padding_max = 0.0;
	std::size_t compared = 0;
	std::size_t bad = 0;
	for (std::size_t i = 0; i < actual_bf16.size(); ++i) {
		float actual = static_cast<float>(actual_bf16[i]);
		float expected = reference[i];
		int row = static_cast<int>(i / host.padded_n);
		int column = static_cast<int>(i % host.padded_n);
		double absolute =
			std::abs(static_cast<double>(actual) - expected);
		double relative =
			absolute / std::max(1.0e-5, std::abs(static_cast<double>(expected)));
		max_abs = std::max(max_abs, absolute);
		max_rel = std::max(max_rel, relative);
		mean_abs += absolute;
		++compared;
		if (
			Mode != dz::EpilogueMode::kRawGemm &&
			row < shape.m &&
			host.target[row] == host.ignore_index) {
			ignored_max = std::max(
				ignored_max,
				std::abs(static_cast<double>(actual)));
		}
		if (row >= shape.m || column >= shape.n) {
			padding_max = std::max(
				padding_max,
				std::abs(static_cast<double>(actual)));
		}
		double absolute_tolerance =
			Mode == dz::EpilogueMode::kRawGemm ? 0.02 : 0.004;
		double relative_tolerance =
			Mode == dz::EpilogueMode::kRawGemm ? 0.06 : 0.10;
		if (
			absolute > absolute_tolerance &&
			relative > relative_tolerance) {
			if (bad < 5) {
				std::fprintf(
					stderr,
					"mismatch %s mode=%d (%d,%d): got %.8g expected %.8g "
					"abs %.5g rel %.5g\n",
					shape.name,
					static_cast<int>(Mode),
					row,
					column,
					actual,
					expected,
					absolute,
					relative);
			}
			++bad;
		}
	}
	mean_abs /= static_cast<double>(compared);
	std::printf(
		"DZ_CORRECTNESS shape=%s mode=%d M=%d N=%d K=%d "
		"padded_M=%d padded_N=%d bad=%zu max_abs=%.8g max_rel=%.8g "
		"mean_abs=%.8g ignored_max=%.8g padding_max=%.8g\n",
		shape.name,
		static_cast<int>(Mode),
		shape.m,
		shape.n,
		shape.k,
		host.padded_m,
		host.padded_n,
		bad,
		max_abs,
		max_rel,
		mean_abs,
		ignored_max,
		padding_max);
	if (bad != 0) fail("dZ correctness", "comparison failed");
}

struct LtDescriptors {
	cublasLtHandle_t handle = nullptr;
	cublasLtMatmulDesc_t operation = nullptr;
	cublasLtMatrixLayout_t a = nullptr;
	cublasLtMatrixLayout_t b = nullptr;
	cublasLtMatrixLayout_t c = nullptr;
	cublasLtMatmulPreference_t preference = nullptr;

	LtDescriptors() = default;
	LtDescriptors(const LtDescriptors&) = delete;
	LtDescriptors& operator=(const LtDescriptors&) = delete;
	LtDescriptors(LtDescriptors&& other) noexcept
		: handle(other.handle),
		  operation(other.operation),
		  a(other.a),
		  b(other.b),
		  c(other.c),
		  preference(other.preference) {
		other.handle = nullptr;
		other.operation = nullptr;
		other.a = nullptr;
		other.b = nullptr;
		other.c = nullptr;
		other.preference = nullptr;
	}

	~LtDescriptors() {
		if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
		if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
		if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
		if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
		if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
		if (handle != nullptr) cublasLtDestroy(handle);
	}
};

LtDescriptors make_lt_descriptors(
		int m,
		int n,
		int k,
		std::size_t workspace_bytes) {
	LtDescriptors descriptors;
	check_cublas(cublasLtCreate(&descriptors.handle), "cublasLtCreate");
	check_cublas(
		cublasLtMatmulDescCreate(
			&descriptors.operation,
			CUBLAS_COMPUTE_32F,
			CUDA_R_32F),
		"cublasLtMatmulDescCreate");
	cublasOperation_t trans_b = CUBLAS_OP_T;
	check_cublas(
		cublasLtMatmulDescSetAttribute(
			descriptors.operation,
			CUBLASLT_MATMUL_DESC_TRANSB,
			&trans_b,
			sizeof(trans_b)),
		"cublasLtMatmulDescSetAttribute(TRANSB)");

	check_cublas(
		cublasLtMatrixLayoutCreate(
			&descriptors.a,
			CUDA_R_16BF,
			m,
			k,
			k),
		"cublasLtMatrixLayoutCreate(A)");
	check_cublas(
		cublasLtMatrixLayoutCreate(
			&descriptors.b,
			CUDA_R_16BF,
			n,
			k,
			k),
		"cublasLtMatrixLayoutCreate(B)");
	check_cublas(
		cublasLtMatrixLayoutCreate(
			&descriptors.c,
			CUDA_R_16BF,
			m,
			n,
			n),
		"cublasLtMatrixLayoutCreate(C)");
	cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
	for (cublasLtMatrixLayout_t layout : {
				descriptors.a, descriptors.b, descriptors.c}) {
		check_cublas(
			cublasLtMatrixLayoutSetAttribute(
				layout,
				CUBLASLT_MATRIX_LAYOUT_ORDER,
				&order,
				sizeof(order)),
			"cublasLtMatrixLayoutSetAttribute(ORDER)");
	}
	check_cublas(
		cublasLtMatmulPreferenceCreate(&descriptors.preference),
		"cublasLtMatmulPreferenceCreate");
	check_cublas(
		cublasLtMatmulPreferenceSetAttribute(
			descriptors.preference,
			CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
			&workspace_bytes,
			sizeof(workspace_bytes)),
		"cublasLtMatmulPreferenceSetAttribute(WORKSPACE)");
	return descriptors;
}

float time_lt_algo(
		const LtDescriptors& descriptors,
		const cublasLtMatmulAlgo_t& algorithm,
		const Element* x,
		const Element* weight,
		Element* output,
		void* workspace,
		std::size_t workspace_bytes,
		int warmups,
		int iterations,
		cudaStream_t stream) {
	const float alpha = 1.0f;
	const float beta = 0.0f;
	auto launch = [&]() {
		return cublasLtMatmul(
			descriptors.handle,
			descriptors.operation,
			&alpha,
			x,
			descriptors.a,
			weight,
			descriptors.b,
			&beta,
			output,
			descriptors.c,
			output,
			descriptors.c,
			&algorithm,
			workspace,
			workspace_bytes,
			stream);
	};
	for (int i = 0; i < warmups; ++i) {
		if (launch() != CUBLAS_STATUS_SUCCESS) {
			cudaGetLastError();
			return std::numeric_limits<float>::infinity();
		}
	}
	if (cudaStreamSynchronize(stream) != cudaSuccess) {
		cudaGetLastError();
		return std::numeric_limits<float>::infinity();
	}

	cudaEvent_t start = nullptr;
	cudaEvent_t end = nullptr;
	check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
	check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
	std::vector<float> samples;
	samples.reserve(iterations);
	for (int i = 0; i < iterations; ++i) {
		check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
		cublasStatus_t status = launch();
		if (status != CUBLAS_STATUS_SUCCESS) {
			cudaEventDestroy(start);
			cudaEventDestroy(end);
			cudaGetLastError();
			return std::numeric_limits<float>::infinity();
		}
		check_cuda(cudaEventRecord(end, stream), "cudaEventRecord(end)");
		check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
		float elapsed = 0.0f;
		check_cuda(
			cudaEventElapsedTime(&elapsed, start, end),
			"cudaEventElapsedTime");
		samples.push_back(elapsed);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return static_cast<float>(median(std::move(samples)));
}

struct LtResult {
	float milliseconds = std::numeric_limits<float>::infinity();
	int algorithm_id = -1;
	int heuristic_index = -1;
	cublasLtMatmulAlgo_t algorithm = {};
};

LtResult benchmark_cublaslt(
		int m,
		int n,
		int k,
		const Element* x,
		const Element* weight,
		Element* output,
		void* workspace,
		std::size_t workspace_bytes,
		int warmups,
		int iterations,
		cudaStream_t stream) {
	LtDescriptors descriptors =
		make_lt_descriptors(m, n, k, workspace_bytes);
	constexpr int kRequested = 32;
	std::vector<cublasLtMatmulHeuristicResult_t> candidates(kRequested);
	int returned = 0;
	check_cublas(
		cublasLtMatmulAlgoGetHeuristic(
			descriptors.handle,
			descriptors.operation,
			descriptors.a,
			descriptors.b,
			descriptors.c,
			descriptors.c,
			descriptors.preference,
			kRequested,
			candidates.data(),
			&returned),
		"cublasLtMatmulAlgoGetHeuristic");
	if (returned == 0) fail("cuBLASLt", "no heuristic algorithms returned");

	LtResult best;
	for (int index = 0; index < returned; ++index) {
		if (candidates[index].state != CUBLAS_STATUS_SUCCESS) continue;
		float elapsed = time_lt_algo(
			descriptors,
			candidates[index].algo,
			x,
			weight,
			output,
			workspace,
			workspace_bytes,
			2,
			5,
			stream);
		if (elapsed < best.milliseconds) {
			best.milliseconds = elapsed;
			best.heuristic_index = index;
			best.algorithm = candidates[index].algo;
		}
	}
	if (!std::isfinite(best.milliseconds)) {
		fail("cuBLASLt", "all heuristic algorithms failed");
	}
	std::size_t written = 0;
	check_cublas(
		cublasLtMatmulAlgoConfigGetAttribute(
			&best.algorithm,
			CUBLASLT_ALGO_CONFIG_ID,
			&best.algorithm_id,
			sizeof(best.algorithm_id),
			&written),
		"cublasLtMatmulAlgoConfigGetAttribute(ID)");
	best.milliseconds = time_lt_algo(
		descriptors,
		best.algorithm,
		x,
		weight,
		output,
		workspace,
		workspace_bytes,
		warmups,
		iterations,
		stream);
	return best;
}

template <dz::EpilogueMode Mode>
float benchmark_kernel(
		const dz::Params& params,
		int warmups,
		int iterations,
		cudaStream_t stream,
		dz::KernelResources* resources) {
	for (int i = 0; i < warmups; ++i) {
		check_cuda(
			dz::launch<Mode>(
				params,
				stream,
				i == 0 ? resources : nullptr),
			"dZ benchmark warmup launch");
	}
	check_cuda(cudaStreamSynchronize(stream), "dZ benchmark warmup sync");

	cudaEvent_t start = nullptr;
	cudaEvent_t end = nullptr;
	check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
	check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
	std::vector<float> samples;
	samples.reserve(iterations);
	for (int i = 0; i < iterations; ++i) {
		check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
		check_cuda(
			dz::launch<Mode>(params, stream),
			"dZ benchmark launch");
		check_cuda(cudaEventRecord(end, stream), "cudaEventRecord(end)");
		check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
		float elapsed = 0.0f;
		check_cuda(
			cudaEventElapsedTime(&elapsed, start, end),
			"cudaEventElapsedTime");
		samples.push_back(elapsed);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return static_cast<float>(median(std::move(samples)));
}

void fill_benchmark_metadata(
		int m,
		int n,
		std::vector<std::int64_t>& target,
		std::vector<float>& grad_output,
		std::vector<float>& lse,
		std::vector<float>& entropy,
		std::vector<float>& entropy_grad) {
	target.resize(m);
	grad_output.resize(m);
	lse.resize(m);
	entropy.resize(m);
	entropy_grad.resize(m);
	float uniform_lse = std::log(static_cast<float>(n));
	for (int row = 0; row < m; ++row) {
		target[row] = row % 17 == 0 ? -100 : row % n;
		grad_output[row] = 1.0f;
		lse[row] = uniform_lse;
		entropy[row] = uniform_lse;
		entropy_grad[row] = 0.125f;
	}
}

void run_benchmark(
		int m,
		int n,
		int k,
		int warmups,
		int iterations,
		std::size_t workspace_bytes,
		int cluster_pairs) {
	int padded_m =
		dz::ceil_div_int(m, dz::Config::kTileM) * dz::Config::kTileM;
	int padded_n =
		dz::ceil_div_int(n, dz::Config::kTileN) * dz::Config::kTileN;
	DeviceBuffer<Element> x(
		static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
	DeviceBuffer<Element> weight(
		static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
	DeviceBuffer<Element> output(
		static_cast<std::size_t>(padded_m) *
		static_cast<std::size_t>(padded_n));
	DeviceBuffer<unsigned char> workspace(workspace_bytes);
	constexpr int kFillThreads = 256;
	int x_blocks = static_cast<int>(
		(x.size() + kFillThreads - 1) / kFillThreads);
	int weight_blocks = static_cast<int>(
		(weight.size() + kFillThreads - 1) / kFillThreads);
	fill_random_bf16<<<x_blocks, kFillThreads>>>(
		x.get(), x.size(), 0x12345678u);
	fill_random_bf16<<<weight_blocks, kFillThreads>>>(
		weight.get(), weight.size(), 0x9abcdef0u);
	check_cuda(cudaGetLastError(), "fill_random_bf16");
	check_cuda(cudaMemset(output.get(), 0, output.bytes()), "cudaMemset(output)");

	std::vector<std::int64_t> host_target;
	std::vector<float> host_grad_output;
	std::vector<float> host_lse;
	std::vector<float> host_entropy;
	std::vector<float> host_entropy_grad;
	fill_benchmark_metadata(
		m,
		n,
		host_target,
		host_grad_output,
		host_lse,
		host_entropy,
		host_entropy_grad);
	DeviceBuffer<std::int64_t> target(host_target.size());
	DeviceBuffer<float> grad_output(host_grad_output.size());
	DeviceBuffer<float> lse(host_lse.size());
	DeviceBuffer<float> entropy(host_entropy.size());
	DeviceBuffer<float> entropy_grad(host_entropy_grad.size());
	copy_to_device(target, host_target);
	copy_to_device(grad_output, host_grad_output);
	copy_to_device(lse, host_lse);
	copy_to_device(entropy, host_entropy);
	copy_to_device(entropy_grad, host_entropy_grad);

	cudaStream_t stream = nullptr;
	check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

	LtResult cublas = benchmark_cublaslt(
		m,
		n,
		k,
		x.get(),
		weight.get(),
		output.get(),
		workspace.get(),
		workspace.bytes(),
		warmups,
		iterations,
		stream);

	dz::Params params;
	params.x = x.get();
	params.weight = weight.get();
	params.target = target.get();
	params.grad_output = grad_output.get();
	params.lse = lse.get();
	params.entropy = entropy.get();
	params.entropy_grad = entropy_grad.get();
	params.output = output.get();
	params.tokens = m;
	params.hidden = k;
	params.local_vocab = n;
	params.padded_tokens = padded_m;
	params.padded_vocab = padded_n;
	params.vocab_start = 0;
	params.ignore_index = -100;
	params.inverse_temperature = 1.0f;
	params.cluster_pairs = cluster_pairs;

	dz::KernelResources raw_resources;
	dz::KernelResources dz_resources;
	dz::KernelResources entropy_resources;
	float raw_ms = benchmark_kernel<dz::EpilogueMode::kRawGemm>(
		params,
		warmups,
		iterations,
		stream,
		&raw_resources);
	float dz_ms = benchmark_kernel<dz::EpilogueMode::kSoftmaxGradient>(
		params,
		warmups,
		iterations,
		stream,
		&dz_resources);
	float entropy_ms =
		benchmark_kernel<
			dz::EpilogueMode::kSoftmaxGradientEntropy>(
				params,
				warmups,
				iterations,
				stream,
				&entropy_resources);

	check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

	double flops =
		2.0 * static_cast<double>(m) * n * k;
	auto tflops = [&](double milliseconds) {
		return flops / (milliseconds * 1.0e9);
	};
	std::printf(
		"DZ_BENCHMARK M=%d N=%d K=%d dtype=BF16 accumulate=FP32 "
		"cluster_pairs=%d "
		"cublaslt_ms=%.6f cublaslt_tflops=%.3f cublaslt_algo=%d "
		"cublaslt_heuristic=%d raw_ms=%.6f raw_tflops=%.3f "
		"raw_ratio=%.5f dz_ms=%.6f dz_effective_tflops=%.3f "
		"dz_inclusive_ratio=%.5f entropy_ms=%.6f "
		"entropy_effective_tflops=%.3f entropy_inclusive_ratio=%.5f\n",
		m,
		n,
		k,
		raw_resources.launched_cluster_pairs,
		cublas.milliseconds,
		tflops(cublas.milliseconds),
		cublas.algorithm_id,
		cublas.heuristic_index,
		raw_ms,
		tflops(raw_ms),
		cublas.milliseconds / raw_ms,
		dz_ms,
		tflops(dz_ms),
		cublas.milliseconds / dz_ms,
		entropy_ms,
		tflops(entropy_ms),
		cublas.milliseconds / entropy_ms);
	auto print_resources = [](const char* mode, const dz::KernelResources& r) {
		std::printf(
			"DZ_RESOURCES mode=%s tile=%dx%dx%d cluster=%dx1 stages=%d "
			"threads=%d tmem_columns=%d dynamic_smem_bytes=%d "
			"registers_per_thread=%d local_bytes_per_thread=%zu "
			"max_active_cluster_pairs=%d launched_cluster_pairs=%d\n",
			mode,
			r.tile_m,
			r.tile_n,
			r.tile_k,
			r.cluster_m,
			r.mainloop_stages,
			r.threads,
			r.tmem_columns,
			r.dynamic_smem_bytes,
			r.registers_per_thread,
			r.local_bytes_per_thread,
			r.max_active_cluster_pairs,
			r.launched_cluster_pairs);
	};
	print_resources("raw", raw_resources);
	print_resources("dz", dz_resources);
	print_resources("entropy", entropy_resources);
}

struct Options {
	bool correctness = true;
	bool benchmark = true;
	int m = 4096;
	int n = 65536;
	int k = 4096;
	int warmups = 8;
	int iterations = 20;
	std::size_t workspace_bytes = 256ull << 20;
	int cluster_pairs = 0;
};

int parse_positive(const char* value, const char* name) {
	char* end = nullptr;
	long parsed = std::strtol(value, &end, 10);
	if (
		end == value ||
		*end != '\0' ||
		parsed <= 0 ||
		parsed > std::numeric_limits<int>::max()) {
		fail(name, "expected a positive integer");
	}
	return static_cast<int>(parsed);
}

int parse_nonnegative(const char* value, const char* name) {
	char* end = nullptr;
	long parsed = std::strtol(value, &end, 10);
	if (
		end == value ||
		*end != '\0' ||
		parsed < 0 ||
		parsed > std::numeric_limits<int>::max()) {
		fail(name, "expected a non-negative integer");
	}
	return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		std::string argument = argv[index];
		auto require_value = [&]() {
			if (index + 1 >= argc) fail(argument.c_str(), "missing value");
			return argv[++index];
		};
		if (argument == "--correctness-only") {
			options.benchmark = false;
		} else if (argument == "--benchmark-only") {
			options.correctness = false;
		} else if (argument == "--m") {
			options.m = parse_positive(require_value(), "--m");
		} else if (argument == "--n") {
			options.n = parse_positive(require_value(), "--n");
		} else if (argument == "--k") {
			options.k = parse_positive(require_value(), "--k");
		} else if (argument == "--warmups") {
			options.warmups = parse_positive(require_value(), "--warmups");
		} else if (argument == "--iterations") {
			options.iterations =
				parse_positive(require_value(), "--iterations");
		} else if (argument == "--workspace-mib") {
			options.workspace_bytes =
				static_cast<std::size_t>(
					parse_positive(require_value(), "--workspace-mib"))
				<< 20;
		} else if (argument == "--clusters") {
			options.cluster_pairs =
				parse_nonnegative(require_value(), "--clusters");
		} else {
			fail("arguments", argument.c_str());
		}
	}
	return options;
}

}  // namespace

int main(int argc, char** argv) {
	Options options = parse_options(argc, argv);
	int device = 0;
	check_cuda(cudaSetDevice(device), "cudaSetDevice");
	cudaDeviceProp properties = {};
	check_cuda(
		cudaGetDeviceProperties(&properties, device),
		"cudaGetDeviceProperties");
	if (properties.major < 10) {
		std::printf("SKIP: requires SM100 or newer, got SM %d.%d\n",
			properties.major,
			properties.minor);
		return 0;
	}
	std::printf(
		"DZ_DEVICE name=\"%s\" sm=%d.%d sms=%d\n",
		properties.name,
		properties.major,
		properties.minor,
		properties.multiProcessorCount);

	if (options.correctness) {
		const Shape aligned{256, 512, 256, "aligned"};
		const Shape ragged{259, 517, 264, "ragged"};
		const Shape persistent{512, 9728, 32, "persistent"};
		run_correctness_case<dz::EpilogueMode::kRawGemm>(aligned);
		run_correctness_case<dz::EpilogueMode::kSoftmaxGradient>(aligned);
		run_correctness_case<
			dz::EpilogueMode::kSoftmaxGradientEntropy>(aligned);
		run_correctness_case<dz::EpilogueMode::kRawGemm>(ragged);
		run_correctness_case<dz::EpilogueMode::kSoftmaxGradient>(ragged);
		run_correctness_case<
			dz::EpilogueMode::kSoftmaxGradientEntropy>(ragged);
		run_correctness_case<dz::EpilogueMode::kRawGemm>(persistent);
		run_correctness_case<
			dz::EpilogueMode::kSoftmaxGradient>(persistent);
		run_correctness_case<
			dz::EpilogueMode::kSoftmaxGradientEntropy>(persistent);
	}
	if (options.benchmark) {
		run_benchmark(
			options.m,
			options.n,
			options.k,
			options.warmups,
			options.iterations,
			options.workspace_bytes,
			options.cluster_pairs);
	}
	return 0;
}
