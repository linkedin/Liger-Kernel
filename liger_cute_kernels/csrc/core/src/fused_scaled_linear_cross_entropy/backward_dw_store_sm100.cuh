#pragma once

// SM100 single-pass TP-FSLCE dW store kernel configuration.
//
// The logical GEMM is [M,K] @ [K,N] -> [M,N], while the physical inputs are
// row-major dZ[K,M] and X[K,N]. CUTLASS therefore sees A as column-major and
// B as row-major. The explicit epilogue tile is one CTA's M128 x N32 BF16
// output, exactly 8 KiB per TMA store.

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

struct BackwardDwKernelScheduleSm100
	: cutlass::gemm::KernelSchedule2Sm,
	  cutlass::gemm::KernelScheduleSm100DenseGemm {
	static constexpr int SchedulerPipelineStageCount = 2;
	static constexpr int AccumulatorPipelineStageCount = 2;
};

struct BackwardDwStoreSm100 {
	using Element = cutlass::bfloat16_t;
	using LayoutA = cutlass::layout::ColumnMajor;
	using LayoutB = cutlass::layout::RowMajor;
	using LayoutD = cutlass::layout::RowMajor;
	using TileShape = cute::Shape<cute::_256, cute::_256, cute::_64>;
	using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
	using EpilogueTile = cute::Shape<cute::_128, cute::_32>;

	static constexpr int kMainloopStages = 6;
	static constexpr int kTileM = 256;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kClusterM = 2;
	static constexpr int kCtaTileM = 128;
	static constexpr int kEpilogueN = 32;
	static constexpr int kTmaStoreBytes =
		kCtaTileM * kEpilogueN * sizeof(Element);
	static_assert(kTileM / kClusterM == kCtaTileM);
	static_assert(kTmaStoreBytes == 8 * 1024);

	using FusionOperation =
		cutlass::epilogue::fusion::ScaledAcc<Element, float>;
	using BuiltCollectiveEpilogue =
		typename cutlass::epilogue::collective::
		CollectiveBuilder<
			cutlass::arch::Sm100,
			cutlass::arch::OpClassTensorOp,
			TileShape,
			ClusterShape,
			EpilogueTile,
			float,
			float,
			void,
			LayoutD,
			8,
			Element,
			LayoutD,
			8,
			cutlass::epilogue::TmaWarpSpecialized2Sm,
			FusionOperation>::CollectiveOp;
	// dW has no C operand. A single output stage with immediate TMA issue
	// removes an unused pipeline stage and was the fastest B300 configuration.
	using CollectiveEpilogue =
		cutlass::epilogue::collective::CollectiveEpilogue<
			cutlass::epilogue::Sm100TmaWarpSpecialized<
				1,
				1,
				32,
				false,
				false>,
			typename BuiltCollectiveEpilogue::CtaTileShape,
			typename BuiltCollectiveEpilogue::EpilogueTile,
			typename BuiltCollectiveEpilogue::ElementC,
			typename BuiltCollectiveEpilogue::StrideC,
			typename BuiltCollectiveEpilogue::ElementD,
			typename BuiltCollectiveEpilogue::StrideD,
			typename BuiltCollectiveEpilogue::FusionCallbacks,
			typename BuiltCollectiveEpilogue::CopyOpT2R,
			typename BuiltCollectiveEpilogue::CopyOpG2S,
			typename BuiltCollectiveEpilogue::SmemLayoutAtomC,
			typename BuiltCollectiveEpilogue::CopyOpS2R,
			typename BuiltCollectiveEpilogue::CopyOpS2G,
			typename BuiltCollectiveEpilogue::SmemLayoutAtomD,
			typename BuiltCollectiveEpilogue::CopyOpR2S,
			typename BuiltCollectiveEpilogue::CopyOpR2R>;

	using CollectiveMainloop = typename cutlass::gemm::collective::
		CollectiveBuilder<
			cutlass::arch::Sm100,
			cutlass::arch::OpClassTensorOp,
			Element,
			LayoutA,
			8,
			Element,
			LayoutB,
			8,
			float,
			TileShape,
			ClusterShape,
			cutlass::gemm::collective::StageCount<kMainloopStages>,
			BackwardDwKernelScheduleSm100>::CollectiveOp;

	using Kernel = cutlass::gemm::kernel::GemmUniversal<
		cute::Shape<int, int, int, int>,
		CollectiveMainloop,
		CollectiveEpilogue>;
	using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
	using StrideA = typename Kernel::StrideA;
	using StrideB = typename Kernel::StrideB;
	using StrideC = typename Kernel::StrideC;
	using StrideD = typename Kernel::StrideD;
	using RasterOrder =
		cutlass::gemm::kernel::detail::RasterOrderOptions;

	static typename Gemm::Arguments arguments(
			const Element* dz_km,
			const Element* x_kn,
			Element* dw_mn,
			int m,
			int n,
			int k,
			int sm_count) {
		StrideA stride_a =
			cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
		StrideB stride_b =
			cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
		StrideC stride_c =
			cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
		StrideD stride_d =
			cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

		typename Gemm::Arguments args{
			cutlass::gemm::GemmUniversalMode::kGemm,
			{m, n, k, 1},
			{dz_km, stride_a, x_kn, stride_b},
			{{1.0f}, nullptr, stride_c, dw_mn, stride_d}};
		args.hw_info.sm_count = sm_count;
		// AlongN is the measured vertical/N-pair raster for this row-major
		// dW problem.
		args.scheduler.raster_order = RasterOrder::AlongN;
		args.scheduler.max_swizzle_size = 1;
		return args;
	}

	static cudaError_t prepare_kernel() {
		cudaError_t error = cudaFuncSetAttribute(
			cutlass::device_kernel<Kernel>,
			cudaFuncAttributeClusterSchedulingPolicyPreference,
			static_cast<int>(cudaClusterSchedulingPolicySpread));
		if (error != cudaSuccess) return error;
		return cudaFuncSetAttribute(
			cutlass::device_kernel<Kernel>,
			cudaFuncAttributePreferredSharedMemoryCarveout,
			cudaSharedmemCarveoutMaxShared);
	}

	static cutlass::Status run(
			const Element* dz_km,
			const Element* x_kn,
			Element* dw_mn,
			int m,
			int n,
			int k,
			int sm_count,
			cudaStream_t stream = nullptr) {
		auto args = arguments(
			dz_km, x_kn, dw_mn, m, n, k, sm_count);
		if (Gemm::get_workspace_size(args) != 0) {
			return cutlass::Status::kErrorWorkspaceNull;
		}
		cudaError_t error = prepare_kernel();
		if (error != cudaSuccess) {
			return cutlass::Status::kErrorInternal;
		}
		Gemm gemm;
		return gemm.run(args, nullptr, stream);
	}
};

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
