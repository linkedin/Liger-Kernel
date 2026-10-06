#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_detail.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/reference/device/tensor_fill.h>

#include "backward_gemm_isolated_sm100.cuh"
#include "backward_dx_wide_sm100.cuh"
#include "tma_copy_atom_sm100.cuh"

namespace sm100_bench {

using namespace cute;

// Isolated B300 gate for the three physical backward GEMMs. The layouts name
// the caller's actual row-major storage, including transposed views:
//   dZ: X(row) @ W^T(column view) -> BF16 row
//   dX: dZ(row) @ W(row)          -> FP32 row
//   dW: dZ^T(column view) @ X(row)-> BF16 row
//
// cuBLASLt uses K64 software tiles and 2-CTA UMMA for all three. Its dX winner
// uses a logical 2x2 cluster and four stages, while dW uses 2x1 and six stages.
// The custom architecture intentionally uses 2x1 for dX as well, preserving
// K64/four-stage/2SM UMMA while avoiding the 2x2 placement ceiling. dW uses
// the confirmed six-stage 2x1 topology directly.

#define CUDA_CHECK(call)                                                     \
	do {                                                                       \
		cudaError_t error_ = (call);                                             \
		if (error_ != cudaSuccess) {                                             \
			std::cerr << #call << " failed: " << cudaGetErrorString(error_)       \
					  << '\n';                                                      \
			std::exit(1);                                                         \
		}                                                                        \
	} while (false)

#define CUTLASS_CHECK(call)                                                  \
	do {                                                                       \
		cutlass::Status status_ = (call);                                        \
		if (status_ != cutlass::Status::kSuccess) {                              \
			std::cerr << #call << " failed with CUTLASS status "                  \
					  << static_cast<int>(status_) << '\n';                          \
			std::exit(1);                                                         \
		}                                                                        \
	} while (false)

template <class T>
__global__ void fill_constant(T* values, std::size_t count, float value) {
	for (std::size_t index =
			 static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		 index < count;
		 index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
		values[index] = T(value);
	}
}

__global__ void fill_random_bf16(
		cutlass::bfloat16_t* data,
		std::size_t elements,
		std::uint32_t seed) {
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
	data[index] = cutlass::bfloat16_t(
		(static_cast<int>(value & 0xffffu) - 32768) *
		(1.0f / 131072.0f));
}

template <class T>
float as_float(T value) {
	return static_cast<float>(value);
}

using DzDefinition =
	liger::fused_scaled_linear_cross_entropy::BackwardDzIsolatedGemmSm100;
using DwDefinition =
	liger::fused_scaled_linear_cross_entropy::BackwardDwIsolatedGemmSm100;

struct Options {
	std::string operation = "dx";
	int m = 0;
	int n = 0;
	int k = 0;
	int warmups = 5;
	int iterations = 20;
	int swizzle = 0;
	std::string raster = "heuristic";
	bool verify = false;
	double reference_tflops = 0.0;
};

int parse_int(const char* value, const char* name) {
	char* end = nullptr;
	long parsed = std::strtol(value, &end, 10);
	if (end == value || *end != '\0' || parsed <= 0) {
		std::cerr << "invalid " << name << ": " << value << '\n';
		std::exit(2);
	}
	return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		std::string argument = argv[index];
		auto require_value = [&](const char* name) {
			if (++index >= argc) {
				std::cerr << "missing value for " << name << '\n';
				std::exit(2);
			}
			return argv[index];
		};
		if (argument == "--case") {
			options.operation = require_value("--case");
		} else if (argument == "--m") {
			options.m = parse_int(require_value("--m"), "--m");
		} else if (argument == "--n") {
			options.n = parse_int(require_value("--n"), "--n");
		} else if (argument == "--k") {
			options.k = parse_int(require_value("--k"), "--k");
		} else if (argument == "--warmups") {
			options.warmups =
				parse_int(require_value("--warmups"), "--warmups");
		} else if (argument == "--iterations") {
			options.iterations =
				parse_int(require_value("--iterations"), "--iterations");
		} else if (argument == "--swizzle") {
			options.swizzle =
				parse_int(require_value("--swizzle"), "--swizzle");
		} else if (argument == "--raster") {
			options.raster = require_value("--raster");
		} else if (argument == "--verify") {
			options.verify = true;
		} else if (argument == "--reference-tflops") {
			options.reference_tflops =
				std::stod(require_value("--reference-tflops"));
		} else {
			std::cerr << "unknown argument: " << argument << '\n';
			std::exit(2);
		}
	}

	if (options.operation == "dz") {
		if (options.m == 0) options.m = 4096;
		if (options.n == 0) options.n = 65536;
		if (options.k == 0) options.k = 4096;
	} else if (options.operation == "dx") {
		if (options.m == 0) options.m = 4096;
		if (options.n == 0) options.n = 4096;
		if (options.k == 0) options.k = 65536;
	} else if (options.operation == "dw") {
		if (options.m == 0) options.m = 65536;
		if (options.n == 0) options.n = 4096;
		if (options.k == 0) options.k = 4096;
	} else {
		std::cerr << "--case must be dz, dx, or dw\n";
		std::exit(2);
	}
	return options;
}

cutlass::gemm::kernel::detail::RasterOrderOptions raster_option(
		const std::string& value) {
	using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
	if (value == "m") return Raster::AlongM;
	if (value == "n") return Raster::AlongN;
	if (value == "heuristic") return Raster::Heuristic;
	std::cerr << "--raster must be heuristic, m, or n\n";
	std::exit(2);
}

template <class Definition>
void run(const Options& options) {
	using Gemm = typename Definition::Gemm;
	using ElementA = typename Definition::ElementA;
	using ElementB = typename Definition::ElementB;
	using ElementC = typename Definition::ElementC;
	using ElementD = typename Definition::ElementD;
	using ElementCStorage = std::conditional_t<
		std::is_void_v<ElementC>, ElementD, ElementC>;
	using LayoutA = typename Definition::GmemLayoutA;
	using LayoutB = typename Definition::GmemLayoutB;
	using LayoutC = typename Definition::LayoutC;
	using LayoutD = typename Definition::LayoutD;
	using StrideA = typename Gemm::GemmKernel::StrideA;
	using StrideB = typename Gemm::GemmKernel::StrideB;
	using StrideC = typename Gemm::GemmKernel::StrideC;
	using StrideD = typename Gemm::GemmKernel::StrideD;

	std::size_t a_count =
		static_cast<std::size_t>(options.m) * options.k;
	std::size_t b_count =
		static_cast<std::size_t>(options.k) * options.n;
	std::size_t d_count =
		static_cast<std::size_t>(options.m) * options.n;

	ElementA* a = nullptr;
	ElementB* b = nullptr;
	ElementCStorage* c = nullptr;
	ElementD* d = nullptr;
	ElementD* reference_d = nullptr;
	CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(ElementA)));
	CUDA_CHECK(cudaMalloc(&b, b_count * sizeof(ElementB)));
	CUDA_CHECK(cudaMalloc(&c, d_count * sizeof(ElementCStorage)));
	CUDA_CHECK(cudaMalloc(&d, d_count * sizeof(ElementD)));
	if (options.verify) {
		CUDA_CHECK(cudaMalloc(&reference_d, d_count * sizeof(ElementD)));
	}

	int blocks_d = static_cast<int>((d_count + 255) / 256);
	if (options.verify) {
		cutlass::reference::device::BlockFillRandomUniform(
			a, a_count, 20260907, ElementA(1.0f), ElementA(-1.0f), 3);
		cutlass::reference::device::BlockFillRandomUniform(
			b, b_count, 20260908, ElementB(1.0f), ElementB(-1.0f), 3);
	} else {
		int blocks_a = static_cast<int>((a_count + 255) / 256);
		int blocks_b = static_cast<int>((b_count + 255) / 256);
		fill_constant<<<blocks_a, 256>>>(a, a_count, 0.5f);
		fill_constant<<<blocks_b, 256>>>(b, b_count, 0.25f);
	}
	fill_constant<<<blocks_d, 256>>>(c, d_count, 0.0f);
	fill_constant<<<blocks_d, 256>>>(d, d_count, 0.0f);
	CUDA_CHECK(cudaGetLastError());

	auto stride_a = cutlass::make_cute_packed_stride(
		StrideA{}, make_shape(options.m, options.k, 1));
	auto stride_b = cutlass::make_cute_packed_stride(
		StrideB{}, make_shape(options.n, options.k, 1));
	auto stride_c = cutlass::make_cute_packed_stride(
		StrideC{}, make_shape(options.m, options.n, 1));
	auto stride_d = cutlass::make_cute_packed_stride(
		StrideD{}, make_shape(options.m, options.n, 1));
	int device = 0;
	CUDA_CHECK(cudaGetDevice(&device));
	auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
		typename Gemm::GemmKernel>(device);

	typename Gemm::Arguments arguments{
		cutlass::gemm::GemmUniversalMode::kGemm,
		{options.m, options.n, options.k, 1},
		{a, stride_a, b, stride_b},
		{{1.0f, 0.0f},
		 std::is_void_v<ElementC> ? nullptr : c,
		 stride_c,
		 d,
		 stride_d},
		hardware};
	arguments.scheduler.max_swizzle_size = options.swizzle;
	arguments.scheduler.raster_order = raster_option(options.raster);

	Gemm gemm;
	cudaFuncAttributes attributes{};
	CUDA_CHECK(cudaFuncSetCacheConfig(
		cutlass::device_kernel<typename Gemm::GemmKernel>,
		cudaFuncCachePreferShared));
	CUDA_CHECK(cudaFuncSetAttribute(
		cutlass::device_kernel<typename Gemm::GemmKernel>,
		cudaFuncAttributePreferredSharedMemoryCarveout,
		cudaSharedmemCarveoutMaxShared));
	CUDA_CHECK(cudaFuncSetAttribute(
		cutlass::device_kernel<typename Gemm::GemmKernel>,
		cudaFuncAttributeClusterSchedulingPolicyPreference,
		cudaClusterSchedulingPolicySpread));
	CUDA_CHECK(cudaFuncGetAttributes(
		&attributes,
		cutlass::device_kernel<typename Gemm::GemmKernel>));
	std::size_t workspace_bytes = Gemm::get_workspace_size(arguments);
	void* workspace = nullptr;
	if (workspace_bytes != 0) {
		CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
	}
	CUTLASS_CHECK(gemm.can_implement(arguments));
	CUTLASS_CHECK(gemm.initialize(arguments, workspace));
	for (int iteration = 0; iteration < options.warmups; ++iteration) {
		CUTLASS_CHECK(gemm.run());
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	bool correct = true;
	if (options.verify) {
		using ReferenceGemm = cutlass::reference::device::Gemm<
			ElementA,
			LayoutA,
			ElementB,
			LayoutB,
			ElementD,
			LayoutD,
			float,
			float>;
		cutlass::TensorRef<ElementA, LayoutA> reference_a(
			a, LayoutA::packed({options.m, options.k}));
		cutlass::TensorRef<ElementB, LayoutB> reference_b(
			b, LayoutB::packed({options.k, options.n}));
		cutlass::TensorRef<ElementD, LayoutD> reference_c(
			c, LayoutD::packed({options.m, options.n}));
		cutlass::TensorRef<ElementD, LayoutD> reference_output(
			reference_d, LayoutD::packed({options.m, options.n}));
		ReferenceGemm reference_gemm;
		reference_gemm(
			{options.m, options.n, options.k},
			1.0f,
			reference_a,
			reference_b,
			0.0f,
			reference_c,
			reference_output);
		CUDA_CHECK(cudaDeviceSynchronize());
		ElementD epsilon = std::is_same_v<ElementD, float>
			? ElementD(5.0e-3f)
			: ElementD(2.0e-2f);
		ElementD floor = std::is_same_v<ElementD, float>
			? ElementD(1.0e-3f)
			: ElementD(2.0e-2f);
		correct = cutlass::reference::device::BlockCompareRelativelyEqual(
			d, reference_d, d_count, epsilon, floor);
	}

	ElementD samples[3];
	std::size_t sample_indices[3] = {
		0,
		d_count / 2,
		d_count - 1,
	};
	for (int index = 0; index < 3; ++index) {
		CUDA_CHECK(cudaMemcpy(
			&samples[index],
			d + sample_indices[index],
			sizeof(ElementD),
			cudaMemcpyDeviceToHost));
	}
	float expected = static_cast<float>(options.k) * 0.125f;
	if (!options.verify) {
		float tolerance = std::is_same_v<ElementD, float>
			? std::max(1.0e-3f, std::abs(expected) * 2.0e-5f)
			: std::max(0.03125f, std::abs(expected) * 8.0e-3f);
		for (ElementD sample : samples) {
			correct &= std::abs(as_float(sample) - expected) <= tolerance;
		}
	}

	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));
	for (int iteration = 0; iteration < options.iterations; ++iteration) {
		CUTLASS_CHECK(gemm.run());
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
	double milliseconds =
		static_cast<double>(total_ms) / options.iterations;
	double tflops =
		2.0 * static_cast<double>(options.m) * options.n * options.k /
		(milliseconds * 1.0e9);
	double ratio = options.reference_tflops > 0.0
		? tflops / options.reference_tflops
		: 0.0;
	if (options.reference_tflops > 0.0) {
		correct &= ratio >= 0.95;
	}

	std::cout << "RESULT"
			  << " case=" << options.operation
			  << " m=" << options.m
			  << " n=" << options.n
			  << " k=" << options.k
			  << " stages=" << Definition::CollectiveMainloop::DispatchPolicy::Stages
			  << " swizzle=" << options.swizzle
			  << " raster=" << options.raster
			  << " verify=" << options.verify
			  << " regs=" << attributes.numRegs
			  << " static_smem=" << attributes.sharedSizeBytes
			  << " max_dynamic_smem=" << attributes.maxDynamicSharedSizeBytes
			  << " shared_storage="
			  << sizeof(typename Gemm::GemmKernel::SharedStorage)
			  << " workspace_bytes=" << workspace_bytes
			  << " ms=" << milliseconds
			  << " tflops=" << tflops
			  << " reference_tflops=" << options.reference_tflops
			  << " ratio=" << ratio
			  << " expected=" << expected
			  << " samples=" << as_float(samples[0]) << ","
			  << as_float(samples[1]) << "," << as_float(samples[2])
			  << " status=" << (correct ? "PASS" : "FAIL")
			  << '\n';

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	if (workspace != nullptr) CUDA_CHECK(cudaFree(workspace));
	if (reference_d != nullptr) CUDA_CHECK(cudaFree(reference_d));
	CUDA_CHECK(cudaFree(d));
	CUDA_CHECK(cudaFree(c));
	CUDA_CHECK(cudaFree(b));
	CUDA_CHECK(cudaFree(a));
	if (!correct) std::exit(3);
}

void run_dx_wide(const Options& options) {
	using Traits =
		liger::fused_scaled_linear_cross_entropy::
			BackwardDxWideTraitsSm100;
	using Element = Traits::Element;
	std::size_t a_count =
		static_cast<std::size_t>(options.m) * options.k;
	std::size_t b_count =
		static_cast<std::size_t>(options.k) * options.n;
	std::size_t d_count =
		static_cast<std::size_t>(options.m) * options.n;

	Element* a = nullptr;
	Element* b = nullptr;
	float* d = nullptr;
	float* reference_d = nullptr;
	CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&b, b_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&d, d_count * sizeof(float)));
	if (options.verify) {
		CUDA_CHECK(cudaMalloc(&reference_d, d_count * sizeof(float)));
	}
	fill_random_bf16<<<
		static_cast<int>((a_count + 255) / 256), 256>>>(
			a, a_count, 0x12345678u);
	fill_random_bf16<<<
		static_cast<int>((b_count + 255) / 256), 256>>>(
			b, b_count, 0x9abcdef0u);
	CUDA_CHECK(cudaMemset(d, 0, d_count * sizeof(float)));

	auto tensor_a = make_tensor(
		make_gmem_ptr(a),
		make_shape(
			static_cast<std::int64_t>(options.m),
			static_cast<std::int64_t>(options.k)),
		make_stride(static_cast<std::int64_t>(options.k), _1{}));
	auto tensor_b = make_tensor(
		make_gmem_ptr(b),
		make_shape(
			static_cast<std::int64_t>(options.n),
			static_cast<std::int64_t>(options.k)),
		make_stride(_1{}, static_cast<std::int64_t>(options.n)));
	auto tensor_d = make_tensor(
		make_gmem_ptr(d),
		make_shape(
			static_cast<std::int64_t>(options.m),
			static_cast<std::int64_t>(options.n)),
		make_stride(static_cast<std::int64_t>(options.n), _1{}));
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
	auto tma_d = make_tma_copy(
		::liger::TmaStoreAtomForCompute<100>{},
		tensor_d,
		typename Traits::SmemLayoutStoreTile{});
	liger::fused_scaled_linear_cross_entropy::
		BackwardDxWideTmaBundleSm100<
			decltype(tma_a),
			decltype(tma_b),
			decltype(tma_d)> bundle{tma_a, tma_b, tma_d};
	liger::fused_scaled_linear_cross_entropy::
		BackwardDxWideParamsSm100 params{
			d, options.m, options.n, options.k};
	auto launch = [&] {
		return liger::fused_scaled_linear_cross_entropy::
			launch_backward_dx_wide_sm100(bundle, params);
	};

	for (int iteration = 0; iteration < options.warmups; ++iteration) {
		CUDA_CHECK(launch());
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	bool correct = true;
	if (options.verify) {
		using ReferenceGemm = cutlass::reference::device::Gemm<
			Element,
			cutlass::layout::RowMajor,
			Element,
			cutlass::layout::RowMajor,
			float,
			cutlass::layout::RowMajor,
			float,
			float>;
		cutlass::TensorRef<Element, cutlass::layout::RowMajor> reference_a(
			a,
			cutlass::layout::RowMajor::packed(
				{options.m, options.k}));
		cutlass::TensorRef<Element, cutlass::layout::RowMajor> reference_b(
			b,
			cutlass::layout::RowMajor::packed(
				{options.k, options.n}));
		cutlass::TensorRef<float, cutlass::layout::RowMajor> reference_c(
			d,
			cutlass::layout::RowMajor::packed(
				{options.m, options.n}));
		cutlass::TensorRef<float, cutlass::layout::RowMajor>
			reference_output(
				reference_d,
				cutlass::layout::RowMajor::packed(
					{options.m, options.n}));
		ReferenceGemm reference_gemm;
		reference_gemm(
			{options.m, options.n, options.k},
			1.0f,
			reference_a,
			reference_b,
			0.0f,
			reference_c,
			reference_output);
		CUDA_CHECK(cudaDeviceSynchronize());
		correct =
			cutlass::reference::device::BlockCompareRelativelyEqual(
				d, reference_d, d_count, 5.0e-3f, 1.0e-3f);
	}

	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));
	for (int iteration = 0; iteration < options.iterations; ++iteration) {
		CUDA_CHECK(launch());
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
	double milliseconds =
		static_cast<double>(total_ms) / options.iterations;
	double tflops =
		2.0 * static_cast<double>(options.m) * options.n * options.k /
		(milliseconds * 1.0e9);
	double ratio = options.reference_tflops > 0.0
		? tflops / options.reference_tflops
		: 0.0;
	if (options.reference_tflops > 0.0) correct &= ratio >= 0.95;
	std::cout << "RESULT case=dx"
			  << " m=" << options.m
			  << " n=" << options.n
			  << " k=" << options.k
			  << " stages=" << Traits::kStages
			  << " cluster=2x1"
			  << " tile=256x512"
			  << " ms=" << milliseconds
			  << " tflops=" << tflops
			  << " reference_tflops=" << options.reference_tflops
			  << " ratio=" << ratio
			  << " status=" << (correct ? "PASS" : "FAIL")
			  << '\n';

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	if (reference_d != nullptr) CUDA_CHECK(cudaFree(reference_d));
	CUDA_CHECK(cudaFree(d));
	CUDA_CHECK(cudaFree(b));
	CUDA_CHECK(cudaFree(a));
	if (!correct) std::exit(3);
}

}  // namespace sm100_bench

int main(int argc, char** argv) {
	sm100_bench::Options options = sm100_bench::parse_options(argc, argv);
	if (options.operation == "dz") {
		sm100_bench::run<sm100_bench::DzDefinition>(options);
	} else if (options.operation == "dx") {
		sm100_bench::run_dx_wide(options);
	} else {
		sm100_bench::run<sm100_bench::DwDefinition>(options);
	}
	return 0;
}
