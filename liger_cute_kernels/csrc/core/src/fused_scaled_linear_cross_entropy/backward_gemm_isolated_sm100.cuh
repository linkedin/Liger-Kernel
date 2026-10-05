#pragma once

// Standalone SM100 GEMM definitions used to gate the fused backward mainloops.
// They operate on the same physical row-major tensors as the production
// backward and deliberately keep its cluster-2 constraint.

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include <cstdint>
#include <type_traits>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

// Confirmed cuBLASLt/CUPTI/SASS targets from B300 (CUDA 12.9.2.10).
// Both kernels use UTCHMMA.2CTA. The runtime descriptor is one
// M256xN256xK16 BF16xBF16->FP32 instruction; four instructions form each K64
// software iteration. Accumulators stay in TMEM with no local-memory spills.
// dX has eight UTMALDG.3D.MULTICAST.2CTA sites plus two paired-CTA loads; dW
// has eight paired-CTA loads and no multicast site. Both use SYNCS.EXCH.64,
// transaction-counted SYNCS.ARRIVE.TRANS64, UCGABAR, UTCBAR.2CTA.MULTICAST,
// UGETNEXTWORKID.BROADCAST and UTCATOMSWS.2CTA.FIND_AND_SET.
//
// dX FP32 stores use LDTM.x8 -> scalar STS -> 64 four-KiB UTMASTG.3D stores.
// dW uses LDTM -> alpha FFMA2 -> F2FP.BF16.PACK_AB -> STSM -> eight eight-KiB
// stores. The later-wave dW cubin differs only by replacing those eight
// UTMASTG.3D instructions with UTMAREDG.3D.ADD.
struct BackwardCublasWarpPlanSm100 {
	static constexpr int kThreads = 256;
	static constexpr int kWarpgroups = 2;
	static constexpr int kThreadsPerWarpgroup = 128;
	static constexpr int kEpilogueWarpgroup = 0;
	static constexpr int kProducerMmaWarpgroup = 1;
	static constexpr int kEpilogueBarrierThreads = 128;
	static constexpr int kUniformRegistersPerThread = 168;
	static constexpr int kRegistersPerCta =
		kThreads * kUniformRegistersPerThread;
	static constexpr int kCacheConfigRequest = 1;
	static constexpr int kClusterSchedulingPolicy = 1;
	static constexpr int kExecutedSmemBytes = 233472;
	static constexpr bool kUsesOptinDynamicSmem = true;
	static constexpr bool kPrefersSharedCarveout = true;
	static constexpr bool kUsesSpreadClusterScheduling = true;
	static constexpr bool kUsesFenceViewAsyncShared = true;
	static constexpr bool kUsesDeferredEpilogueBarrier = true;
	static constexpr bool kUsesTmaCommandFlush = true;
	static constexpr bool kUsesSetmaxnreg = false;
	static constexpr bool kHasStackSpill = false;
	static constexpr bool kHasLocalMemorySpill = false;
};

struct BackwardDxCublasSassTargetSm100 {
	static constexpr int kAlgorithm = 66;
	static constexpr int kTileM = 256;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kCtaOutputM = 256;
	static constexpr int kCtaOutputN = 256;
	static constexpr int kStages = 4;
	static constexpr int kLogicalClusterM = 2;
	static constexpr int kLogicalClusterN = 2;
	static constexpr int kLaunchClusterX = 4;
	static constexpr int kThreads = 256;
	static constexpr int kRegisters = 168;
	static constexpr int kDynamicSmemBytes = 205032;
	static constexpr int kExecutedSmemBytes = 233472;
	static constexpr int kGridX = 256;
	static constexpr int kClusterCount = 64;
	static constexpr int kMaxActiveClusters = 33;
	static constexpr int kClusterOutputM = 512;
	static constexpr int kClusterOutputN = 512;
	static constexpr int kOutputTilesPerCluster = 4;
	static constexpr int kTwoCtaGroupsPerCluster = 2;
	static constexpr int kTwoCtaGroupOutputM = 256;
	static constexpr int kTwoCtaGroupOutputN = 512;
	static constexpr int kUmmaSubtilesPerTwoCtaGroup = 2;
	static constexpr int kK64Iterations = 1024;
	static constexpr int kUmmaPerK64PerTwoCtaGroup = 8;
	static constexpr int kATileBytesPerK64PerTwoCtaGroup = 32768;
	static constexpr int kBTileBytesPerK64PerTwoCtaGroup = 65536;
	static constexpr int kOperandBytesPerK64PerTwoCtaGroup =
		kATileBytesPerK64PerTwoCtaGroup +
		kBTileBytesPerK64PerTwoCtaGroup;
	static constexpr int kMulticastChunksPerCtaPerStage = 4;
	static constexpr int kMulticastChunkBytes = 8192;
	static constexpr int kMulticastFirstSmemOffset = 0;
	static constexpr int kMulticastSmemOffsetStride = 0x2000;
	static constexpr int kMulticastLastSmemOffset = 0x6000;
	static constexpr int kDirectChunksPerCtaPerStage = 1;
	static constexpr int kDirectChunkBytes = 16384;
	static constexpr int kBTileBytesPerCtaPerStage =
		kMulticastChunksPerCtaPerStage * kMulticastChunkBytes;
	static constexpr int kATileBytesPerCtaPerStage =
		kDirectChunksPerCtaPerStage * kDirectChunkBytes;
	static constexpr int kOperandBytesPerStage = 49152;
	static constexpr int kStageRingSlots = 4;
	static constexpr int kStageAdvanceBytes = 0xc000;
	static constexpr int kStageWrapBackSlots = 3;
	static constexpr int kStageWrapBytes = 0x24000;
	static constexpr int kOperandSmemBytes =
		kStages * kOperandBytesPerStage;
	static constexpr int kEpilogueStoreBuffers = 2;
	static constexpr int kFp32StoreBufferBytes = 4096;
	static constexpr int kFp32StoreRows = 4;
	static constexpr int kFp32StoreColumns = 256;
	static constexpr int kFp32StoreElementBytes = 4;
	static constexpr int kFp32StoresPerCta = 64;
	static constexpr int kFp32StoreAdvanceBytes = 0x1000;
	static constexpr int kFp32StoreWrapBytes = 0x1000;
	static constexpr int kFp32StateBytes = 232;
	static constexpr int kSchedulerEpilogueSmemBytes =
		kEpilogueStoreBuffers * kFp32StoreBufferBytes +
		kFp32StateBytes;
	static constexpr int kBf16DynamicSmemBytes = 213192;
	static constexpr int kBf16StoreBufferBytes = 8192;
	static constexpr int kBf16StoreRows = 16;
	static constexpr int kBf16StoreColumns = 256;
	static constexpr int kBf16StoreElementBytes = 2;
	static constexpr int kBf16StoresPerCta = 16;
	static constexpr int kBf16StoreAdvanceBytes = 0x2000;
	static constexpr int kBf16StoreWrapBytes = 0x2000;
	static constexpr int kBf16StateBytes = 200;
	static constexpr int kBf16SchedulerEpilogueSmemBytes =
		kEpilogueStoreBuffers * kBf16StoreBufferBytes +
		kBf16StateBytes;
	static constexpr std::uint32_t kUmmaDescriptor = 0x10408490u;
	static constexpr std::uint64_t kFlopsPerUmma = 2097152ull;
	static constexpr std::uint64_t kDynamicUmmaOperations = 1048576ull;
	static constexpr std::uint64_t kProblemFlops = 2199023255552ull;
	static constexpr int kStaticInstructionCount = 2536;
	static constexpr int kStaticUmmaSites = 16;
	static constexpr int kStaticTmaLoadSites = 10;
	static constexpr int kStaticMulticastTmaLoadSites = 8;
	static constexpr int kStaticPairedTmaLoadSites = 2;
	static constexpr int kStaticLdtmSites = 64;
	static constexpr int kStaticSharedStoreSites = 512;
	static constexpr int kStaticTmaStoreSites = 64;
	static constexpr int kStaticDeferredBarriers = 128;
	static constexpr float kMilliseconds = 1.1200f;
	static constexpr float kTflops = 1963.36f;
	static constexpr float kMaximumTargetMilliseconds = 1.1790f;
	static constexpr float kSmActivePercent = 86.24f;
	static constexpr float kTensorActivePercent = 85.27f;
	// The common N512 B tile is the multicast operand. Its
	// global fetch is shared across the two M groups, while each group's SMEM
	// destination remains separately allocated.
	static constexpr bool kBTileCommonAcrossMGroups = true;
	static constexpr bool kMulticastOperandIsB = true;
	static constexpr bool kDirectOperandIsA = true;
	static constexpr const char* kKernelName =
		"nvjet_tss_256x256_64x4_2x2_2cta_h_bz_NNT";
};

struct BackwardDwCublasSassTargetSm100 {
	static constexpr int kAlgorithm = 66;
	// One CTA owns M128xN256; the 2-CTA UMMA macro tile is M256xN256.
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileM = 256;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kStages = 6;
	static constexpr int kClusterM = 2;
	static constexpr int kClusterN = 1;
	static constexpr int kThreads = 256;
	static constexpr int kRegisters = 168;
	static constexpr int kDynamicSmemBytes = 213280;
	static constexpr int kExecutedSmemBytes = 233472;
	static constexpr int kGridX = 8192;
	static constexpr int kClusterCount = 4096;
	static constexpr int kMaxActiveClusters = 74;
	static constexpr int kClusterOutputM = 256;
	static constexpr int kClusterOutputN = 256;
	static constexpr int kOutputTilesPerCluster = 1;
	static constexpr int kK64Iterations = 64;
	static constexpr int kUmmaPerK64 = 4;
	static constexpr int kAChunksPerCtaPerStage = 2;
	static constexpr int kBChunksPerCtaPerStage = 2;
	static constexpr int kOperandChunkBytes = 8192;
	static constexpr int kATileBytesPerCtaPerStage =
		kAChunksPerCtaPerStage * kOperandChunkBytes;
	static constexpr int kBTileBytesPerCtaPerStage =
		kBChunksPerCtaPerStage * kOperandChunkBytes;
	static constexpr int kOperandBytesPerStage = 32768;
	static constexpr int kStageRingSlots = 6;
	static constexpr int kStageAdvanceBytes = 0x8000;
	static constexpr int kStageWrapBackSlots = 5;
	static constexpr int kStageWrapBytes = 0x28000;
	static constexpr int kOperandSmemBytes =
		kStages * kOperandBytesPerStage;
	static constexpr int kEpilogueStoreBuffers = 2;
	static constexpr int kStoreBufferBytes = 8192;
	static constexpr int kStoreRows = 16;
	static constexpr int kStoreColumns = 256;
	static constexpr int kStoreElementBytes = 2;
	static constexpr int kStoresPerCta = 8;
	static constexpr int kStoreAdvanceBytes = 0x2000;
	static constexpr int kStoreWrapBytes = 0x2000;
	static constexpr int kStateBytes = 288;
	static constexpr int kSchedulerEpilogueSmemBytes =
		kEpilogueStoreBuffers * kStoreBufferBytes + kStateBytes;
	static constexpr std::uint32_t kUmmaDescriptor = 0x10418490u;
	static constexpr std::uint64_t kFlopsPerUmma = 2097152ull;
	static constexpr std::uint64_t kDynamicUmmaOperations = 1048576ull;
	static constexpr std::uint64_t kProblemFlops = 2199023255552ull;
	static constexpr int kStaticInstructionCount = 1112;
	static constexpr int kStaticUmmaSites = 8;
	static constexpr int kStaticTmaLoadSites = 8;
	static constexpr int kStaticMulticastTmaLoadSites = 0;
	static constexpr int kStaticPairedTmaLoadSites = 8;
	static constexpr int kStaticLdtmSites = 16;
	static constexpr int kStaticBf16PackSites = 128;
	static constexpr int kStaticSharedStoreSites = 32;
	static constexpr int kStaticTmaStoreSites = 8;
	static constexpr int kStaticDeferredBarriers = 16;
	static constexpr float kMilliseconds = 0.9935f;
	static constexpr float kTflops = 2213.33f;
	static constexpr float kMaximumTargetMilliseconds = 1.0458f;
	static constexpr float kSmActivePercent = 98.52f;
	static constexpr float kTensorActivePercent = 96.18f;
	static constexpr float kWave1024StoreMilliseconds = 0.2619f;
	static constexpr float kWave1024AddMilliseconds = 0.2927f;
	static constexpr float kFourWaveAddMilliseconds = 1.1400f;
	static constexpr const char* kKernelName =
		"nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT";
	static constexpr const char* kAddKernelName =
		"nvjet_tst_128x256_64x6_2x1_2cta_v_badd_NTT";
};

static_assert(
	BackwardDxCublasSassTargetSm100::kMaximumTargetMilliseconds >=
		BackwardDxCublasSassTargetSm100::kMilliseconds / 0.95f);
static_assert(
	BackwardDwCublasSassTargetSm100::kMaximumTargetMilliseconds >
		BackwardDwCublasSassTargetSm100::kMilliseconds);
static_assert(
	BackwardCublasWarpPlanSm100::kRegistersPerCta == 43008);
static_assert(
	BackwardCublasWarpPlanSm100::kExecutedSmemBytes ==
		BackwardDxCublasSassTargetSm100::kExecutedSmemBytes);
static_assert(
	BackwardCublasWarpPlanSm100::kExecutedSmemBytes ==
		BackwardDwCublasSassTargetSm100::kExecutedSmemBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kThreads ==
		BackwardCublasWarpPlanSm100::kThreads);
static_assert(
	BackwardDwCublasSassTargetSm100::kThreads ==
		BackwardCublasWarpPlanSm100::kThreads);
static_assert(
	BackwardDxCublasSassTargetSm100::kRegisters ==
		BackwardCublasWarpPlanSm100::kUniformRegistersPerThread);
static_assert(
	BackwardDwCublasSassTargetSm100::kRegisters ==
		BackwardCublasWarpPlanSm100::kUniformRegistersPerThread);
static_assert(
	BackwardDxCublasSassTargetSm100::kClusterCount *
			BackwardDxCublasSassTargetSm100::kTwoCtaGroupsPerCluster *
			BackwardDxCublasSassTargetSm100::kK64Iterations *
			BackwardDxCublasSassTargetSm100::
				kUmmaPerK64PerTwoCtaGroup ==
		BackwardDxCublasSassTargetSm100::kDynamicUmmaOperations);
static_assert(
	BackwardDxCublasSassTargetSm100::kTwoCtaGroupOutputN ==
		2 * BackwardDxCublasSassTargetSm100::kTileN);
static_assert(
	BackwardDxCublasSassTargetSm100::kClusterOutputM ==
		BackwardDxCublasSassTargetSm100::kTwoCtaGroupsPerCluster *
			BackwardDxCublasSassTargetSm100::kTwoCtaGroupOutputM);
static_assert(
	BackwardDxCublasSassTargetSm100::kClusterOutputN ==
		BackwardDxCublasSassTargetSm100::kTwoCtaGroupOutputN);
static_assert(
	BackwardDxCublasSassTargetSm100::kUmmaPerK64PerTwoCtaGroup ==
		4 * BackwardDxCublasSassTargetSm100::
			kUmmaSubtilesPerTwoCtaGroup);
static_assert(
	BackwardDxCublasSassTargetSm100::
			kOperandBytesPerK64PerTwoCtaGroup /
			2 ==
		BackwardDxCublasSassTargetSm100::kOperandBytesPerStage);
static_assert(
	BackwardDxCublasSassTargetSm100::kATileBytesPerCtaPerStage +
			BackwardDxCublasSassTargetSm100::kBTileBytesPerCtaPerStage ==
		BackwardDxCublasSassTargetSm100::kOperandBytesPerStage);
static_assert(
	BackwardDxCublasSassTargetSm100::kStageRingSlots ==
		BackwardDxCublasSassTargetSm100::kStages);
static_assert(
	BackwardDxCublasSassTargetSm100::kStageAdvanceBytes ==
		BackwardDxCublasSassTargetSm100::kOperandBytesPerStage);
static_assert(
	BackwardDxCublasSassTargetSm100::kStageWrapBytes ==
		BackwardDxCublasSassTargetSm100::kStageWrapBackSlots *
			BackwardDxCublasSassTargetSm100::kStageAdvanceBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kMulticastLastSmemOffset ==
		(BackwardDxCublasSassTargetSm100::
			 kMulticastChunksPerCtaPerStage -
			1) *
			BackwardDxCublasSassTargetSm100::
				kMulticastSmemOffsetStride);
static_assert(
	BackwardDwCublasSassTargetSm100::kClusterCount *
			BackwardDwCublasSassTargetSm100::kK64Iterations *
			BackwardDwCublasSassTargetSm100::kUmmaPerK64 ==
		BackwardDwCublasSassTargetSm100::kDynamicUmmaOperations);
static_assert(
	BackwardDwCublasSassTargetSm100::kATileBytesPerCtaPerStage +
			BackwardDwCublasSassTargetSm100::kBTileBytesPerCtaPerStage ==
		BackwardDwCublasSassTargetSm100::kOperandBytesPerStage);
static_assert(
	BackwardDwCublasSassTargetSm100::kStageRingSlots ==
		BackwardDwCublasSassTargetSm100::kStages);
static_assert(
	BackwardDwCublasSassTargetSm100::kStageAdvanceBytes ==
		BackwardDwCublasSassTargetSm100::kOperandBytesPerStage);
static_assert(
	BackwardDwCublasSassTargetSm100::kStageWrapBytes ==
		BackwardDwCublasSassTargetSm100::kStageWrapBackSlots *
			BackwardDwCublasSassTargetSm100::kStageAdvanceBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kDynamicUmmaOperations *
			BackwardDxCublasSassTargetSm100::kFlopsPerUmma ==
		BackwardDxCublasSassTargetSm100::kProblemFlops);
static_assert(
	BackwardDwCublasSassTargetSm100::kDynamicUmmaOperations *
			BackwardDwCublasSassTargetSm100::kFlopsPerUmma ==
		BackwardDwCublasSassTargetSm100::kProblemFlops);
static_assert(
	BackwardDxCublasSassTargetSm100::kOperandSmemBytes +
			BackwardDxCublasSassTargetSm100::
				kSchedulerEpilogueSmemBytes ==
		BackwardDxCublasSassTargetSm100::kDynamicSmemBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kOperandSmemBytes +
			BackwardDxCublasSassTargetSm100::
				kBf16SchedulerEpilogueSmemBytes ==
		BackwardDxCublasSassTargetSm100::kBf16DynamicSmemBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kFp32StoreAdvanceBytes ==
		BackwardDxCublasSassTargetSm100::kFp32StoreBufferBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kFp32StoreRows *
			BackwardDxCublasSassTargetSm100::kFp32StoreColumns *
			BackwardDxCublasSassTargetSm100::kFp32StoreElementBytes ==
		BackwardDxCublasSassTargetSm100::kFp32StoreBufferBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kFp32StoresPerCta *
			BackwardDxCublasSassTargetSm100::kFp32StoreRows ==
		BackwardDxCublasSassTargetSm100::kCtaOutputM);
static_assert(
	BackwardDxCublasSassTargetSm100::kFp32StoreWrapBytes ==
		BackwardDxCublasSassTargetSm100::kFp32StoreBufferBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kBf16StoreAdvanceBytes ==
		BackwardDxCublasSassTargetSm100::kBf16StoreBufferBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kBf16StoreRows *
			BackwardDxCublasSassTargetSm100::kBf16StoreColumns *
			BackwardDxCublasSassTargetSm100::kBf16StoreElementBytes ==
		BackwardDxCublasSassTargetSm100::kBf16StoreBufferBytes);
static_assert(
	BackwardDxCublasSassTargetSm100::kBf16StoresPerCta *
			BackwardDxCublasSassTargetSm100::kBf16StoreRows ==
		BackwardDxCublasSassTargetSm100::kCtaOutputM);
static_assert(
	BackwardDxCublasSassTargetSm100::kBf16StoreWrapBytes ==
		BackwardDxCublasSassTargetSm100::kBf16StoreBufferBytes);
static_assert(
	BackwardDwCublasSassTargetSm100::kOperandSmemBytes +
			BackwardDwCublasSassTargetSm100::
				kSchedulerEpilogueSmemBytes ==
		BackwardDwCublasSassTargetSm100::kDynamicSmemBytes);
static_assert(
	BackwardDwCublasSassTargetSm100::kStoreAdvanceBytes ==
		BackwardDwCublasSassTargetSm100::kStoreBufferBytes);
static_assert(
	BackwardDwCublasSassTargetSm100::kStoreRows *
			BackwardDwCublasSassTargetSm100::kStoreColumns *
			BackwardDwCublasSassTargetSm100::kStoreElementBytes ==
		BackwardDwCublasSassTargetSm100::kStoreBufferBytes);
static_assert(
	BackwardDwCublasSassTargetSm100::kStoresPerCta *
			BackwardDwCublasSassTargetSm100::kStoreRows ==
		BackwardDwCublasSassTargetSm100::kCtaTileM);
static_assert(
	BackwardDwCublasSassTargetSm100::kStoreWrapBytes ==
		BackwardDwCublasSassTargetSm100::kStoreBufferBytes);

template <
	class LayoutA,
	class LayoutB,
	class SourceElement,
	class OutputElement,
	class OutputLayout,
	class ClusterShape,
	int Stages,
	class TileSchedulerTag = void>
struct BackwardIsolatedGemmDefinitionSm100 {
	using ElementA = cutlass::bfloat16_t;
	using ElementB = cutlass::bfloat16_t;
	using ElementC = SourceElement;
	using ElementD = OutputElement;
	using ElementAccumulator = float;
	using ElementCompute = float;
	using GmemLayoutA = LayoutA;
	using GmemLayoutB = LayoutB;
	using LayoutC = OutputLayout;
	using LayoutD = OutputLayout;

	static constexpr int kAlignmentA =
		128 / cutlass::sizeof_bits<ElementA>::value;
	static constexpr int kAlignmentB =
		128 / cutlass::sizeof_bits<ElementB>::value;
	static constexpr int kAlignmentD =
		128 / cutlass::sizeof_bits<ElementD>::value;
	static constexpr int kAlignmentC = [] {
		if constexpr (std::is_void_v<ElementC>) {
			return kAlignmentD;
		} else {
			return 128 / cutlass::sizeof_bits<ElementC>::value;
		}
	}();
	static constexpr int kStages = Stages;

	using MmaTileShape = Shape<_256, _256, _64>;
	using EpilogueTile =
		cutlass::epilogue::collective::EpilogueTileAuto;

	using CollectiveEpilogue =
		typename cutlass::epilogue::collective::CollectiveBuilder<
			cutlass::arch::Sm100,
			cutlass::arch::OpClassTensorOp,
			MmaTileShape,
			ClusterShape,
			EpilogueTile,
			ElementAccumulator,
			ElementCompute,
			ElementC,
			LayoutC,
			kAlignmentC,
			ElementD,
			LayoutD,
			kAlignmentD,
			cutlass::epilogue::TmaWarpSpecialized2Sm>::CollectiveOp;

	using CollectiveMainloop =
		typename cutlass::gemm::collective::CollectiveBuilder<
			cutlass::arch::Sm100,
			cutlass::arch::OpClassTensorOp,
			ElementA,
			LayoutA,
			kAlignmentA,
			ElementB,
			LayoutB,
			kAlignmentB,
			ElementAccumulator,
			MmaTileShape,
			ClusterShape,
			cutlass::gemm::collective::StageCount<kStages>,
			cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;

	using Kernel = cutlass::gemm::kernel::GemmUniversal<
		Shape<int, int, int, int>,
		CollectiveMainloop,
		CollectiveEpilogue,
		TileSchedulerTag>;
	using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Physical row-major contracts:
// dZ = X @ W^T and dW = dZ^T @ X. The custom 2x1 N512 dX kernel is defined
// separately in backward_dx_wide_sm100.cuh.
using BackwardDzIsolatedGemmSm100 =
	BackwardIsolatedGemmDefinitionSm100<
		cutlass::layout::RowMajor,
		cutlass::layout::ColumnMajor,
		cutlass::bfloat16_t,
		cutlass::bfloat16_t,
		cutlass::layout::RowMajor,
		Shape<_2, _1, _1>,
		5>;

using BackwardDwIsolatedGemmSm100 =
	BackwardIsolatedGemmDefinitionSm100<
		cutlass::layout::ColumnMajor,
		cutlass::layout::RowMajor,
		void,
		cutlass::bfloat16_t,
		cutlass::layout::RowMajor,
		Shape<_2, _1, _1>,
		6>;

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
