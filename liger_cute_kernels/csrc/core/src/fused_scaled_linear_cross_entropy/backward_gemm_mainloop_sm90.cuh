#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy backward kernels.
//
// dZ uses the cluster-2 M-fast handoff kernel: warp 3 produces TMA panels,
// warps 4..11 run N256 WGMMA, and warps 0..2 transform/store the staged logits.
// dX+dW uses one cluster-2 four-stage mainloop and one R24 accumulator reused
// across phases. dX publishes FP32 N32 TMA tiles to the communication ring;
// dW uses N64/S2 TMA epilogues with hidden-N pairs as the fastest raster.
// ═══════════════════════════════════════════════════════════════════════════

// gemm_sm90.cuh carries the reusable SM90 producer/consumer mechanics and
// enables CuTe's extended MMA shapes, so it must be the first CUTLASS/CuTe
// include of any TU using this header.
#include "gemm_sm90.cuh"

#include "backward.cuh"
#include "backward_gemm_sm90.cuh"
#include "dx_reduce.cuh"
#include "tma_store_sm90.cuh"
#include "liger_cute/detail/local_reduce.cuh"

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

// ───────────────────────────────────────────────────────────────────────────
// SMEM layouts that gemm_sm90.cuh does not carry
// ───────────────────────────────────────────────────────────────────────────

// MN-major operand staging with the 128 B swizzle atom. dX's B operand (W seen
// as [hidden, vocab]) and both dW operands are MN-major, which the K-major
// helper in gemm_sm90.cuh cannot express.
template <class Element, int Rows, int Cols, int Stages, int Compute = 90>
struct SmemLayoutMnMajorSm90 {
	static_assert(Compute == 90, "SmemLayoutMnMajorSm90 requires Compute=90");
	static_assert(Rows * static_cast<int>(sizeof(Element)) % 128 == 0,
		"the SW128 MN-major atom needs 128 B columns");

	static constexpr int kCompute = Compute;

	using Atom = GMMA::Layout_MN_SW128_Atom<Element>;
	using Single = decltype(tile_to_shape(
		Atom{}, Shape<Int<Rows>, Int<Cols>>{}));
	using Staged = decltype(tile_to_shape(
		Atom{}, Shape<Int<Rows>, Int<Cols>, Int<Stages>>{}));

	static constexpr int kStageBytes = static_cast<int>(
		cute::size(Single{}) * sizeof(Element));
	static constexpr int kStagedElements = cosize_v<Staged>;
};

// ───────────────────────────────────────────────────────────────────────────
// Global tensors for the TMA descriptors
// ───────────────────────────────────────────────────────────────────────────

// int64 extents throughout: tokens * hidden and vocab * hidden both exceed
// INT_MAX at production shapes and would wrap CuTe's layout algebra.
template <class Element>
CUTE_HOST_DEVICE auto backward_row_major_tensor(
		const Element* ptr, int64_t rows, int64_t cols) {
	return make_tensor(
		make_gmem_ptr(ptr),
		make_layout(make_shape(rows, cols), make_stride(cols, _1{})));
}

// A transposed *view* of a row-major [cols, rows] matrix: shape (rows, cols)
// with the first mode contiguous. This is how the MN-major operands are
// addressed without ever materialising a transpose.
template <class Element>
CUTE_HOST_DEVICE auto backward_col_major_tensor(
		const Element* ptr, int64_t rows, int64_t cols, int64_t leading) {
	return make_tensor(
		make_gmem_ptr(ptr),
		make_layout(make_shape(rows, cols), make_stride(_1{}, leading)));
}

// ───────────────────────────────────────────────────────────────────────────
// Traits
// ───────────────────────────────────────────────────────────────────────────

template <
	int Compute = 90,
	int MainloopStages = BackwardGemmConfigSm90<Compute>::kMainloopStages,
	int DwStoreTileN = BackwardGemmConfigSm90<Compute>::kDwStoreTileN,
	int DwStoreStages = BackwardGemmConfigSm90<Compute>::kDwStoreStages>
struct BackwardGemmTraitsSm90 {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kTileN = Config::kDzTileN;
	static constexpr int kStages = MainloopStages;
	static constexpr int kNumThreads = Config::kNumThreads;
	static constexpr int kWarpSize = fused_scaled_linear_cross_entropy::kWarpSize;
	static constexpr int kWarpGroupSize = Config::kWarpGroupSize;
	static constexpr int kNumMmaWarpGroups = Config::kNumMmaWarpGroups;
	static constexpr int kConsumerThreads = Config::kConsumerThreads;
	static constexpr int kDwStoreTileN = DwStoreTileN;
	static constexpr int kDwStoreStages = DwStoreStages;

	static_assert(Config::kDzTileN == Config::kDxTileN);
	static_assert(Config::kDzTileN == Config::kDwTileN);
	static_assert(Config::kClusterM == 1,
		"nvshmemx_collective_launch() takes no cluster launch attribute, so "
		"the backward kernel cannot run in a cluster");

	using ClusterShape = Shape<_1, _1, _1>;

	// Operand staging. A is M128xK64, B is N256xK64; both are three stages
	// deep and every phase re-uses the same two SMEM arenas.
	using SmemAKMajor =
		sm90::SmemLayoutKMajorSm90<Element, kTileM, kTileK, kStages, Compute>;
	using SmemBKMajor =
		sm90::SmemLayoutKMajorSm90<Element, kTileN, kTileK, kStages, Compute>;
	using SmemAMnMajor =
		SmemLayoutMnMajorSm90<Element, kTileM, kTileK, kStages, Compute>;
	using SmemBMnMajor =
		SmemLayoutMnMajorSm90<Element, kTileN, kTileK, kStages, Compute>;

	static constexpr int kSmemAElements = SmemAKMajor::kStagedElements;
	static constexpr int kSmemBElements = SmemBKMajor::kStagedElements;
	static_assert(kSmemAElements == SmemAMnMajor::kStagedElements,
		"the K-major and MN-major A arenas must be interchangeable");
	static_assert(kSmemBElements == SmemBMnMajor::kStagedElements,
		"the K-major and MN-major B arenas must be interchangeable");

	// dX stores accumulator fragments directly to FP32 global staging; dW
	// uses this shared-memory TMA epilogue.
	using DwSmemStore1 = sm90::SmemLayoutKMajorSm90<
		Element, kTileM, kDwStoreTileN, 1, Compute>;
	template <int Stages>
	using DwSmemStore = sm90::SmemLayoutKMajorSm90<
		Element, kTileM, kDwStoreTileN, Stages, Compute>;

	static constexpr int kDwStoreElements =
		DwSmemStore<kDwStoreStages>::kStagedElements;
	static constexpr int kStoreElements = kDwStoreElements;

	// WGMMA. Two warp groups split the M128 tile; one N256 atom per group.
	template <GMMA::Major MajorA, GMMA::Major MajorB>
	using TiledMmaTraits = sm90::TiledMmaMSplitSm90<
		Element, Element, ElementAccum, kTileN, kNumMmaWarpGroups,
		MajorA, MajorB, Compute>;

	using DxMmaTraits = TiledMmaTraits<GMMA::Major::K, GMMA::Major::MN>;
	using DwMmaTraits = TiledMmaTraits<GMMA::Major::MN, GMMA::Major::MN>;
	using DwStoreMmaTraits = sm90::TiledMmaMSplitSm90<
		Element,
		Element,
		ElementAccum,
		kDwStoreTileN,
		kNumMmaWarpGroups,
		GMMA::Major::MN,
		GMMA::Major::MN,
		Compute>;
	using AccumShape = Shape<Int<kTileM>, Int<kTileN>>;

	using PipelineTraits = sm90::MainloopPipelineSm90<kStages, Compute>;
	using MainloopPipeline = typename PipelineTraits::Type;
	using PipelineState = typename PipelineTraits::State;

	// One mbarrier per stage covers A and B together.
	static constexpr int kTmaTransBytes =
		SmemAKMajor::kStageBytes + SmemBKMajor::kStageBytes;
	static_assert(
		SmemAKMajor::kStageBytes == SmemAMnMajor::kStageBytes &&
			SmemBKMajor::kStageBytes == SmemBMnMajor::kStageBytes,
		"all three phases must move the same number of bytes per stage");

};


template <int Compute = 90>
struct BackwardDxCluster2TraitsSm90 {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	using Config = BackwardGemmConfigSm90<Compute>;
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = 128;
	static constexpr int kTileK = 64;
	static constexpr int kTileN = 256;
	static constexpr int kFragmentTileN = 256;
	static constexpr int kNumFragments = 1;
	static constexpr int kStoreTileN = 32;
	static constexpr int kStoreStages = 1;
	static constexpr int kStages = 4;
	static constexpr int kClusterM = 2;
	static constexpr int kNumThreads = 384;
	static constexpr int kConsumerThreads = 256;
	static constexpr int kWarpSize = 32;

	using ClusterShape = Shape<Int<kClusterM>, _1, _1>;
	using SmemA =
		sm90::SmemLayoutKMajorSm90<
			Element, kTileM, kTileK, kStages, Compute>;
	using SmemB = SmemLayoutMnMajorSm90<
		Element, kFragmentTileN, kTileK, kStages, Compute>;
	using SmemStore = sm90::SmemLayoutKMajorSm90<
		float, kTileM, kStoreTileN, 1, Compute>;
	using TiledMmaTraits = sm90::TiledMmaMSplitSm90<
		Element,
		Element,
		ElementAccum,
		kFragmentTileN,
		2,
		GMMA::Major::K,
		GMMA::Major::MN,
		Compute>;
	using TiledMma = typename TiledMmaTraits::Type;
	static_assert(cute::is_same_v<
			typename TiledMmaTraits::Atom,
			SM90_64x256x16_F32BF16BF16_SS<
				GMMA::Major::K, GMMA::Major::MN>>,
		"production clustered dX must use native N256 WGMMA");
	using AccumShape = Shape<Int<kTileM>, Int<kFragmentTileN>>;
	using StoreMmaTraits = sm90::TiledMmaMSplitSm90<
		Element,
		Element,
		float,
		kStoreTileN,
		2,
		GMMA::Major::K,
		GMMA::Major::MN,
		Compute>;

	using PipelineTraits = sm90::MainloopPipelineSm90<kStages, Compute>;
	using MainloopPipeline = typename PipelineTraits::Type;
	using PipelineState = typename PipelineTraits::State;

	static constexpr int kSmemAElements = SmemA::kStagedElements;
	static constexpr int kSmemBElements = SmemB::kStagedElements;
	static constexpr int kTmaTransBytes =
		SmemA::kStageBytes + kNumFragments * SmemB::kStageBytes;

	static_assert(kTileN == kNumFragments * kFragmentTileN);
};

template <int Compute = 90>
struct BackwardDzHandoffTraitsSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	using Element = cutlass::bfloat16_t;
	using ElementLogit = cutlass::half_t;
	using ElementAccum = float;

	static constexpr int kTileM = 128;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kStages = 3;
	static constexpr int kStoreTileN = 256;
	static constexpr int kNumSubTiles = 1;
	static constexpr int kLogitStride = kTileN;
	static constexpr int kClusterM = 2;
	static constexpr int kWarpSize =
		fused_scaled_linear_cross_entropy::kWarpSize;
	static constexpr int kConsumerThreads = 256;
	static constexpr int kEpilogueThreads = 96;
	static constexpr int kProducerRegisters = 88;
	static constexpr int kConsumerRegisters = 208;
	static constexpr int kMmaBarrierId = 9;
	static constexpr int kEpilogueBarrierId = 10;

	using ClusterShape = Shape<Int<kClusterM>, _1, _1>;
	using TiledMmaTraits = sm90::TiledMmaMSplitSm90<
		Element,
		Element,
		ElementAccum,
		kTileN,
		2,
		GMMA::Major::K,
		GMMA::Major::K,
		Compute>;
	using TiledMma = typename TiledMmaTraits::Type;
	using AccumShape = Shape<Int<kTileM>, Int<kTileN>>;

	using SmemTraitsA =
		sm90::SmemLayoutKMajorSm90<Element, kTileM, kTileK, kStages, Compute>;
	using SmemTraitsB =
		sm90::SmemLayoutKMajorSm90<Element, kTileN, kTileK, kStages, Compute>;
	using SmemLayoutA = typename SmemTraitsA::Staged;
	using SmemLayoutB = typename SmemTraitsB::Staged;
	using SmemStoreSingle = Layout<
		Shape<Int<kTileM>, Int<kStoreTileN>>,
		Stride<Int<kLogitStride>, _1>>;
	using SmemStore = Layout<
		Shape<Int<kTileM>, Int<kStoreTileN>, Int<kNumSubTiles>>,
		Stride<Int<kLogitStride>, _1, Int<kStoreTileN>>>;

	using PipelineTraits = sm90::MainloopPipelineSm90<kStages, Compute>;
	using MainloopPipeline = typename PipelineTraits::Type;
	using PipelineState = typename PipelineTraits::State;

	static constexpr int kSmemAElements = SmemTraitsA::kStagedElements;
	static constexpr int kSmemBElements = SmemTraitsB::kStagedElements;
	static constexpr int kLogitElements = kTileM * kLogitStride;
	static constexpr int kTmaTransBytes =
		SmemTraitsA::kStageBytes + SmemTraitsB::kStageBytes;

	static_assert(
		kProducerRegisters * 128 + kConsumerRegisters * 256 == 64512);
};



// ───────────────────────────────────────────────────────────────────────────
// Shared memory
// ───────────────────────────────────────────────────────────────────────────

template <int Compute>
struct BackwardDxDwSmemSm90 {
	using Traits = BackwardGemmTraitsSm90<Compute>;
	using Element = typename Traits::Element;

	alignas(16) typename Traits::MainloopPipeline::SharedStorage mainloop_pipeline;
	alignas(8) std::uint64_t
		dx_ready_epoch[kDxCommWarpsPerChannel][kDxRingStages];
	alignas(8) std::uint64_t
		dx_consumed_barrier[kDxCommWarpsPerChannel][kDxRingStages];

	alignas(1024) Element smem_a[Traits::kSmemAElements];
	alignas(1024) Element smem_b[Traits::kSmemBElements];
	alignas(1024) Element smem_store[Traits::kStoreElements];

	CUTE_DEVICE Element* a_data() { return &smem_a[0]; }
	CUTE_DEVICE Element* b_data() { return &smem_b[0]; }
	CUTE_DEVICE Element* store_data() { return &smem_store[0]; }
};


template <int Compute>
struct BackwardDxCluster2SmemSm90 {
	using Traits = BackwardDxCluster2TraitsSm90<Compute>;
	using DwTraits = BackwardGemmTraitsSm90<Compute, 4, 64, 2>;
	using Element = typename Traits::Element;
	static constexpr int kDxStoreBytes =
		Traits::SmemStore::kStagedElements * sizeof(float);
	static constexpr int kDwStoreBytes =
		DwTraits::kDwStoreElements * sizeof(Element);
	static constexpr int kStoreBytes =
		kDxStoreBytes > kDwStoreBytes ? kDxStoreBytes : kDwStoreBytes;
	static constexpr int kStoreElements = kStoreBytes / sizeof(Element);

	alignas(16) typename Traits::MainloopPipeline::SharedStorage mainloop_pipeline;
	alignas(8) std::uint64_t
		dx_ready_epoch[kDxCommWarpsPerChannel][kDxRingStages];
	alignas(8) std::uint64_t
		dx_consumed_barrier[kDxCommWarpsPerChannel][kDxRingStages];
	alignas(1024) Element smem_a[Traits::kSmemAElements];
	alignas(1024) Element smem_b0[Traits::kSmemBElements];
	alignas(1024) Element smem_store[kStoreElements];

	CUTE_DEVICE Element* a_data() { return &smem_a[0]; }
	CUTE_DEVICE Element* b_data() { return &smem_b0[0]; }
	CUTE_DEVICE Element* b0_data() { return &smem_b0[0]; }
	CUTE_DEVICE Element* store_data() { return &smem_store[0]; }
	CUTE_DEVICE float* dx_store_data() {
		return reinterpret_cast<float*>(&smem_store[0]);
	}
};

template <int Compute, bool ReturnEntropy>
struct BackwardDzHandoffSmemSm90 {
	using Traits = BackwardDzHandoffTraitsSm90<Compute>;
	using Element = typename Traits::Element;
	using ElementLogit = typename Traits::ElementLogit;

	alignas(16) typename Traits::MainloopPipeline::SharedStorage mainloop_pipeline;
	alignas(8) std::uint64_t logits_ready;
	alignas(8) std::uint64_t logits_free;
	alignas(16) float row_scale[Traits::kTileM];
	alignas(16) float row_lse[Traits::kTileM];
	alignas(16) float row_entropy[ReturnEntropy ? Traits::kTileM : 1];
	alignas(16) float row_entropy_scale[ReturnEntropy ? Traits::kTileM : 1];
	alignas(16) int row_target[Traits::kTileM];

	alignas(1024) Element smem_a[Traits::kSmemAElements];
	alignas(1024) Element smem_b[Traits::kSmemBElements];
	alignas(1024) ElementLogit smem_logits[Traits::kLogitElements];

	CUTE_DEVICE Element* a_data() { return &smem_a[0]; }
	CUTE_DEVICE Element* b_data() { return &smem_b[0]; }
	CUTE_DEVICE ElementLogit* logit_row(int row) {
		return &smem_logits[row * Traits::kLogitStride];
	}
	CUTE_DEVICE Element* store_data() {
		return reinterpret_cast<Element*>(&smem_logits[0]);
	}
};



// ───────────────────────────────────────────────────────────────────────────
// Named barriers
// ───────────────────────────────────────────────────────────────────────────

template <int BarrierId, int Compute>
CUTE_DEVICE void backward_consumer_barrier_sm90() {
	using Config = BackwardGemmConfigSm90<Compute>;
	sm90::named_barrier_sync_sm90<
		BarrierId, Config::kConsumerThreads, Compute>();
}

// Hands the aliased A/B and store-staging arenas from the dX phase to the dW
// phase across warp 0 and warps 4..11 only. __syncthreads() would drag the
// communication warps in and serialise the dX all-reduce tail against the dW
// GEMM, which is the whole point of keeping them separate.
template <int Compute>
CUTE_DEVICE void backward_compute_barrier_sm90() {
	using Config = BackwardGemmConfigSm90<Compute>;
	sm90::named_barrier_sync_sm90<
		Config::kComputeBarrierId, Config::kComputeBarrierThreads, Compute>();
}

CUTE_DEVICE void dx_ready_store_release(
		std::uint64_t* ready, std::uint64_t epoch) {
	cuda::atomic_ref<std::uint64_t, cuda::thread_scope_block> flag(*ready);
	flag.store(epoch, cuda::memory_order_release);
}

CUTE_DEVICE void dx_ready_wait_acquire(
		std::uint64_t* ready, std::uint64_t epoch) {
	cuda::atomic_ref<std::uint64_t, cuda::thread_scope_block> flag(*ready);
	while (flag.load(cuda::memory_order_acquire) != epoch) {
		__nanosleep(64);
	}
}

// ───────────────────────────────────────────────────────────────────────────
// dZ phase
// ───────────────────────────────────────────────────────────────────────────

struct BackwardTileCoord {
	int m_tile;
	int n_tile;
};

// Flat n-fastest raster; CTA `cta` takes every `num_ctas`-th tile.
__host__ __device__ inline BackwardTileCoord backward_raster_tile(
		int index, int num_n_tiles) {
	BackwardTileCoord coord;
	coord.m_tile = index / num_n_tiles;
	coord.n_tile = index - coord.m_tile * num_n_tiles;
	return coord;
}

// Cluster peers own adjacent token tiles and therefore reuse one multicast W
// panel. Token-pair tiles advance fastest to match the winning dX schedule.
__host__ __device__ inline BackwardTileCoord backward_cluster_m_m_fast_tile(
		int index, int num_m_pairs, int cluster_rank) {
	BackwardTileCoord coord;
	coord.m_tile = 2 * (index % num_m_pairs) + cluster_rank;
	coord.n_tile = index / num_m_pairs;
	return coord;
}

// Cluster peers keep the same vocabulary tile while rank selects adjacent
// hidden tiles. Hidden-N pairs are the fastest-moving work dimension.
__host__ __device__ inline BackwardTileCoord backward_cluster_n_tile(
		int index, int num_n_cluster_tiles, int cluster_rank) {
	BackwardTileCoord coord =
		backward_raster_tile(index, num_n_cluster_tiles);
	coord.n_tile = 2 * coord.n_tile + cluster_rank;
	return coord;
}

__host__ __device__ inline BackwardTileCoord backward_cluster_n_m_fast_tile(
		int index, int num_m_tiles, int cluster_rank) {
	BackwardTileCoord coord;
	coord.m_tile = index % num_m_tiles;
	coord.n_tile = 2 * (index / num_m_tiles) + cluster_rank;
	return coord;
}

template <int Compute = 90>
// ───────────────────────────────────────────────────────────────────────────
// dX phase
// ───────────────────────────────────────────────────────────────────────────

template <typename CommConfig, int Compute = 90>
struct DxPhaseSm90 {
	using Traits = BackwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Element = typename Traits::Element;
	static constexpr int kCompute = Compute;
	static constexpr int kBarrierId = Config::kDxEpilogueBarrierId;
	static_assert(
		CommConfig::kTileElements == Traits::kTileM * Traits::kTileN,
		"dX staging tile stride must match the WGMMA accumulator tile");

	template <
		bool ReturnEntropy,
		class Pipeline,
		class Smem,
		class TmaDz,
		class TmaWt>
	CUTE_DEVICE static void produce(
			Pipeline& pipe,
			typename Traits::PipelineState& state,
			Smem& smem,
			TmaDz const& tma_dz,
			TmaWt const& tma_wt,
			BackwardGemmParamsSm90<Compute> const& params,
			int padded_vocab,
			int num_n_tiles,
			int num_k_tiles,
			int groups_per_wave,
			int first_unit,
			int stride_unit) {
		if (cute::elect_one_sync() == 0) return;

		auto sA = make_tensor(make_smem_ptr(smem.a_data()),
			typename Traits::SmemAKMajor::Staged{});
		auto sB = make_tensor(make_smem_ptr(smem.b_data()),
			typename Traits::SmemBMnMajor::Staged{});

		auto mDz = tma_dz.get_tma_tensor(make_shape(
			static_cast<int64_t>(Config::kWaveRows),
			static_cast<int64_t>(padded_vocab)));
		// W as [hidden, vocab]: the dX B operand is MN-major.
		auto mWt = tma_wt.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.hidden),
			static_cast<int64_t>(params.local_vocab)));

		auto cta_dz = tma_dz.get_slice(Int<0>{});
		auto cta_wt = tma_wt.get_slice(Int<0>{});
		auto tAsA = cta_dz.partition_D(sA);
		auto tBsB = cta_wt.partition_D(sB);

		for (int unit = first_unit; unit < groups_per_wave;
				unit += stride_unit) {
			DxTileGroup group =
				dx_unit_to_group<CommConfig>(unit, num_n_tiles, 0);
			auto gDz = local_tile(
				mDz,
				make_tile(Int<Traits::kTileM>{}, Int<Traits::kTileK>{}),
				make_coord(group.m_tile, _));
			auto tAgA = cta_dz.partition_S(gDz);

			for (int tile = 0; tile < group.num_tiles; ++tile) {
				auto gWt = local_tile(
					mWt,
					make_tile(Int<Traits::kTileN>{}, Int<Traits::kTileK>{}),
					make_coord(group.first_n_tile + tile, _));
				auto tBgB = cta_wt.partition_S(gWt);

				sm90::MainloopKLoopSm90<Compute>::produce(
					pipe, state, num_k_tiles,
					[&](auto* barrier, int stage, int k_tile) {
						copy(tma_dz.with(*barrier, 0),
							tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
						copy(tma_wt.with(*barrier, 0),
							tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
					});
			}
		}
	}

	// Consumer: GEMM, then store the complete FP32 group into this CTA's
	// compact symmetric staging and publish one CTA-local ready arrival.
	template <bool ReturnEntropy, class Pipeline, class Smem>
	CUTE_DEVICE static int consume(
			Pipeline& pipe,
			typename Traits::PipelineState& read_state,
			typename Traits::PipelineState& release_state,
			Smem& smem,
			DxReduceWorkspace<float> const& comm,
			int num_n_tiles,
			int num_k_tiles,
			int wave,
			int groups_per_wave,
			int first_unit,
			int stride_unit,
			int next_index) {
		typename Traits::DxMmaTraits::Type tiled_mma;
		int tid_in_mma =
			static_cast<int>(threadIdx.x) - Config::kFirstConsumerThread;
		auto thr_mma = tiled_mma.get_slice(tid_in_mma);

		auto sA = make_tensor(make_smem_ptr(smem.a_data()),
			typename Traits::SmemAKMajor::Staged{});
		auto sB = make_tensor(make_smem_ptr(smem.b_data()),
			typename Traits::SmemBMnMajor::Staged{});
		auto tCsA = thr_mma.partition_A(sA);
		auto tCsB = thr_mma.partition_B(sB);

		auto accum = partition_fragment_C(
			tiled_mma, typename Traits::AccumShape{});
		auto coord_c = thr_mma.partition_C(
			make_identity_tensor(typename Traits::AccumShape{}));

		for (int unit = first_unit; unit < groups_per_wave;
				unit += stride_unit) {
			int group_id = wave * groups_per_wave + unit;
			DxTileGroup group =
				dx_unit_to_group<CommConfig>(unit, num_n_tiles, group_id);
			DxCtaGroupSlot slot =
				dx_cta_group_slot<CommConfig>(next_index);

			// Only this CTA's comm warps can release this slot. The elected
			// consumer waits for the prior pass, then releases all consumers.
			if (slot.pass > 0) {
				if (tid_in_mma == 0) {
					CUTE_UNROLL
					for (int comm_warp = 0;
							comm_warp < kDxCommWarpsPerChannel;
							++comm_warp) {
						cute::wait_barrier(
							smem.dx_consumed_barrier[
								comm_warp][slot.stage],
							(slot.pass - 1) & 1);
					}
				}
				backward_consumer_barrier_sm90<kBarrierId, Compute>();
			}

			for (int tile = 0; tile < group.num_tiles; ++tile) {
				CUTE_UNROLL
				for (int i = 0;
						i < decltype(cute::size(accum))::value;
						++i) {
					asm volatile(
						"mov.f32 %0, 0f00000000;" : "=f"(accum(i)));
				}
				sm90::MainloopKLoopSm90<Compute>::consume(
					pipe, read_state, release_state, num_k_tiles,
					[&](int stage) {
						warpgroup_fence_operand(accum);
						warpgroup_arrive();
						gemm(tiled_mma, tCsA(_, _, _, stage),
							tCsB(_, _, _, stage), accum);
						warpgroup_commit_batch();
					},
					[&]() {
						warpgroup_fence_operand(accum);
					});

				// Materialize every FP32 accumulator element in the staging
				// arena before communication consumes the tile.
				std::size_t tile_base = dx_slot_offset<CommConfig>(
					static_cast<int>(blockIdx.x),
					slot.comm_warp,
					slot.stage) +
					static_cast<std::size_t>(tile) *
						CommConfig::kTileElements;
				CUTE_UNROLL
				for (int i = 0; i < decltype(cute::size(accum))::value; ++i) {
					int row = get<0>(coord_c(i));
					int column = get<1>(coord_c(i));
					comm.partial[tile_base +
						static_cast<std::size_t>(row) * Traits::kTileN +
						column] = accum(i);
				}
			}

			// The named barrier orders every scalar writer before the elected
			// writer's CTA-release epoch. The comm warp publishes system-wide
			// only after its matching acquire.
			backward_consumer_barrier_sm90<kBarrierId, Compute>();
			if (tid_in_mma == 0) {
				std::uint64_t ready_epoch =
					static_cast<std::uint64_t>(group_id) + 1;
				CUTE_UNROLL
				for (int comm_warp = 0;
						comm_warp < kDxCommWarpsPerChannel;
						++comm_warp) {
					dx_ready_store_release(
						&smem.dx_ready_epoch[comm_warp][slot.stage],
						ready_epoch);
				}
			}
			++next_index;
		}
		return next_index;
	}

	struct alignas(8) Bfloat16x4 {
		__nv_bfloat162 low;
		__nv_bfloat162 high;
	};

};

// ───────────────────────────────────────────────────────────────────────────
// dW phase — rank local, unchanged from the single-GPU reference.
// ───────────────────────────────────────────────────────────────────────────

template <
	int Compute = 90,
	class Traits_ = BackwardGemmTraitsSm90<Compute>>
struct DwPhaseSm90 {
	using Traits = Traits_;
	using Config = typename Traits::Config;
	using Launch = BackwardGemmLaunchSm90<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kStoreTileN = Traits::kDwStoreTileN;
	static constexpr int kStoreStages = Traits::kDwStoreStages;
	static constexpr int kNumSubTiles = Traits::kTileN / kStoreTileN;
	static constexpr int kBarrierId = Config::kDwStoreBarrierId;

	template <
		bool ReturnEntropy,
		class Pipeline,
		class Smem,
		class TmaDzt,
		class TmaXt>
	CUTE_DEVICE static void produce(
			Pipeline& pipe,
			typename Traits::PipelineState& state,
			Smem& smem,
			TmaDzt const& tma_dzt,
			TmaXt const& tma_xt,
			BackwardGemmParamsSm90<Compute> const& params,
			int padded_vocab,
			int wave,
			int num_n_tiles,
			int num_k_tiles,
			int num_tiles,
			int first_index,
			int stride_index,
			int cluster_n_rank = -1,
			std::uint16_t multicast_mask = 0) {
		if (cute::elect_one_sync() == 0) return;

		auto sA = make_tensor(make_smem_ptr(smem.a_data()),
			typename Traits::SmemAMnMajor::Staged{});
		auto sB = make_tensor(make_smem_ptr(smem.b_data()),
			typename Traits::SmemBMnMajor::Staged{});

		// dZ^T as [vocab, wave_rows] and X^T as [hidden, tokens].
		auto mDzt = tma_dzt.get_tma_tensor(make_shape(
			static_cast<int64_t>(padded_vocab),
			static_cast<int64_t>(Config::kWaveRows)));
		auto mXt = tma_xt.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.hidden),
			static_cast<int64_t>(params.tokens)));

		auto cta_dzt =
			tma_dzt.get_slice(cluster_n_rank < 0 ? 0 : cluster_n_rank);
		auto cta_xt = tma_xt.get_slice(Int<0>{});
		auto tAsA = cta_dzt.partition_D(sA);
		auto tBsB = cta_xt.partition_D(sB);

		// The wave's token rows are the K extent of this GEMM.
		int k_base = wave * (Config::kWaveRows / Traits::kTileK);
		for (int t = 0; t < num_tiles; ++t) {
			int work = first_index + t * stride_index;
			BackwardTileCoord tile = cluster_n_rank < 0
				? backward_raster_tile(work, num_n_tiles)
				: backward_cluster_n_tile(
					work, num_n_tiles, cluster_n_rank);
			auto gDzt = local_tile(
				mDzt,
				make_tile(Int<Traits::kTileM>{}, Int<Traits::kTileK>{}),
				make_coord(tile.m_tile, _));
			auto gXt = local_tile(
				mXt,
				make_tile(Int<Traits::kTileN>{}, Int<Traits::kTileK>{}),
				make_coord(tile.n_tile, _));
			auto tAgA = cta_dzt.partition_S(gDzt);
			auto tBgB = cta_xt.partition_S(gXt);

			sm90::MainloopKLoopSm90<Compute>::produce(
				pipe, state, num_k_tiles,
				[&](auto* barrier, int stage, int k_tile) {
					copy(tma_dzt.with(*barrier, multicast_mask),
						tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
					copy(tma_xt.with(*barrier, 0),
						tBgB(_, _, _, k_base + k_tile), tBsB(_, _, _, stage));
				});
		}
	}

	template <
		bool ReturnEntropy,
		class Pipeline,
		class Accumulator,
		class Smem,
		class TmaDwStore,
		class TmaDwAdd>
	CUTE_DEVICE static void consume_with_accumulator(
			Pipeline& pipe,
			typename Traits::PipelineState& read_state,
			typename Traits::PipelineState& release_state,
			Accumulator& accum,
			Smem& smem,
			TmaDwStore const& tma_store,
			TmaDwAdd const& tma_add,
			BackwardGemmParamsSm90<Compute> const& params,
			int wave,
			int num_n_tiles,
			int num_k_tiles,
			int num_tiles,
			int first_index,
			int stride_index,
			int cluster_n_rank = -1) {
		typename Traits::DwMmaTraits::Type tiled_mma;
		int tid_in_mma =
			static_cast<int>(threadIdx.x) - Config::kFirstConsumerThread;
		auto thr_mma = tiled_mma.get_slice(tid_in_mma);

		auto sA = make_tensor(make_smem_ptr(smem.a_data()),
			typename Traits::SmemAMnMajor::Staged{});
		auto sB = make_tensor(make_smem_ptr(smem.b_data()),
			typename Traits::SmemBMnMajor::Staged{});
		auto tCsA = thr_mma.partition_A(sA);
		auto tCsB = thr_mma.partition_B(sB);

		using ExpectedAccumulator = decltype(partition_fragment_C(
			tiled_mma, typename Traits::AccumShape{}));
		static_assert(cute::is_same_v<Accumulator, ExpectedAccumulator>);

		int valid_n_tiles = ceil_div(params.hidden, Traits::kTileN);
		for (int t = 0; t < num_tiles; ++t) {
			CUTE_UNROLL
			for (int i = 0; i < decltype(cute::size(accum))::value; ++i) {
				asm volatile(
					"mov.f32 %0, 0f00000000;" : "=f"(accum(i)));
			}

			sm90::MainloopKLoopSm90<Compute>::consume(
				pipe, read_state, release_state, num_k_tiles,
				[&](int stage) {
					warpgroup_fence_operand(accum);
					warpgroup_arrive();
					gemm(tiled_mma, tCsA(_, _, _, stage),
						tCsB(_, _, _, stage), accum);
					warpgroup_commit_batch();
				},
				[&]() { warpgroup_fence_operand(accum); });


			int work = first_index + t * stride_index;
			BackwardTileCoord tile = cluster_n_rank < 0
				? backward_raster_tile(work, num_n_tiles)
				: backward_cluster_n_tile(
					work, num_n_tiles, cluster_n_rank);
			bool valid_tile = tile.n_tile < valid_n_tiles;

			bool issuer = (tid_in_mma / Traits::kWarpSize) == 0;
			auto sStore = make_tensor(make_smem_ptr(smem.store_data()),
				typename Traits::template DwSmemStore<
					Traits::kDwStoreStages>::Staged{});
			auto shape_dw = make_shape(
				static_cast<int64_t>(params.local_vocab),
				static_cast<int64_t>(params.hidden));
			auto mStore = tma_store.get_tma_tensor(shape_dw);
			auto cta_store = tma_store.get_slice(Int<0>{});
			auto tSsS = cta_store.partition_S(sStore);
			auto tSgS = cta_store.partition_D(local_tile(
				mStore,
				make_tile(Int<Traits::kTileM>{}, Int<kStoreTileN>{}),
				make_coord(_, _)));
			auto mAdd = tma_add.get_tma_tensor(shape_dw);
			auto cta_add = tma_add.get_slice(Int<0>{});
			auto tAsA = cta_add.partition_S(sStore);
			auto tAgA = cta_add.partition_D(local_tile(
				mAdd,
				make_tile(Int<Traits::kTileM>{}, Int<kStoreTileN>{}),
				make_coord(_, _)));

			typename Traits::DwStoreMmaTraits::Type store_mma;
			auto store_thr = store_mma.get_slice(tid_in_mma);
			auto tDsDFrag = store_thr.partition_C(sStore);
			auto d_frag = make_tensor<typename Traits::Element>(
				shape(tDsDFrag(_, _, _, Int<0>{})));
			constexpr int kFragElements =
				decltype(cute::size(d_frag))::value;
			static_assert(
				kFragElements * kNumSubTiles ==
					decltype(cute::size(accum))::value,
				"dW store fragments must exactly cover the accumulator");

			CUTE_UNROLL
			for (int sub = 0; sub < kNumSubTiles; ++sub) {
				int stage = sub % kStoreStages;
				if (sub >= kStoreStages) {
					if (issuer) {
						wait_tma_store<
							kStoreStages - 1>();
					}
					backward_consumer_barrier_sm90<kBarrierId, Compute>();
				}
				CUTE_UNROLL
				for (int j = 0; j < kFragElements; ++j) {
					d_frag(j) = static_cast<typename Traits::Element>(
						accum(sub * kFragElements + j));
				}
				copy(d_frag, tDsDFrag(_, _, _, stage));
				tma_store_fence();
				backward_consumer_barrier_sm90<kBarrierId, Compute>();
				if (valid_tile && issuer && cute::elect_one_sync()) {
					int n = tile.n_tile * kNumSubTiles + sub;
					// Wave 0 initializes dW; later waves accumulate with
					// SM90_TMA_REDUCE_ADD in the same tile order.
					if (wave == 0) {
						copy(tma_store,
							tSsS(_, _, _, stage),
							tSgS(_, _, _, tile.m_tile, n));
					} else {
						copy(tma_add,
							tAsA(_, _, _, stage),
							tAgA(_, _, _, tile.m_tile, n));
					}
					tma_store_arrive();
				}
			}

			if (issuer) {
				wait_tma_store_complete();
				asm volatile("fence.proxy.async.global;" ::: "memory");
			}
			backward_consumer_barrier_sm90<kBarrierId, Compute>();
		}
	}

	template <
		bool ReturnEntropy,
		class Pipeline,
		class Smem,
		class TmaDwStore,
		class TmaDwAdd>
	CUTE_DEVICE static void consume(
			Pipeline& pipe,
			typename Traits::PipelineState& read_state,
			typename Traits::PipelineState& release_state,
			Smem& smem,
			TmaDwStore const& tma_store,
			TmaDwAdd const& tma_add,
			BackwardGemmParamsSm90<Compute> const& params,
			int wave,
			int num_n_tiles,
			int num_k_tiles,
			int num_tiles,
			int first_index,
			int stride_index,
			int cluster_n_rank = -1) {
		typename Traits::DwMmaTraits::Type tiled_mma;
		auto accum = partition_fragment_C(
			tiled_mma, typename Traits::AccumShape{});
		consume_with_accumulator<ReturnEntropy>(
			pipe,
			read_state,
			release_state,
			accum,
			smem,
			tma_store,
			tma_add,
			params,
			wave,
			num_n_tiles,
			num_k_tiles,
			num_tiles,
			first_index,
			stride_index,
			cluster_n_rank);
	}
};

template <
	class TmaDzLoad,
	class TmaWt,
	class TmaDzt,
	class TmaXt,
	class TmaDwStore,
	class TmaDwAdd>
struct BackwardDxDwTmaBundleSm90 {
	TmaDzLoad dz_load;
	TmaWt wt;
	TmaDzt dzt;
	TmaXt xt;
	TmaDwStore dw_store;
	TmaDwAdd dw_add;
};

template <
	bool ReturnEntropy,
	int Compute,
	class TmaX,
	class TmaW,
	class TmaDzStore>
__global__ __launch_bounds__(BackwardGemmConfigSm90<Compute>::kNumThreads, 1)
void backward_dz_handoff_wave_kernel_sm90(
		__grid_constant__ const TmaX tma_x,
		__grid_constant__ const TmaW tma_w,
		__grid_constant__ const TmaDzStore tma_dz_store,
		__grid_constant__ const BackwardGemmParamsSm90<Compute> params,
		int wave) {
	using Traits = BackwardDzHandoffTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Launch = BackwardGemmLaunchSm90<Compute>;
	using Smem = BackwardDzHandoffSmemSm90<Compute, ReturnEntropy>;
	using PipelineTraits = typename Traits::PipelineTraits;
	using Registers = sm90::WarpSpecializedRegistersSm90<
		Traits::kProducerRegisters,
		Traits::kConsumerRegisters,
		Compute>;

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	int warp_group = cutlass::canonical_warp_group_idx();
	int warp = cutlass::canonical_warp_idx_sync();
	int cluster_rank =
		static_cast<int>(cute::block_rank_in_cluster());
	int cluster_idx = static_cast<int>(blockIdx.z);
	int num_clusters = static_cast<int>(gridDim.z);
	int num_n_cluster_tiles = ceil_div(
		Launch::padded_vocab(params.local_vocab),
		Traits::kClusterM * Traits::kTileN);
	int total_tiles =
		Config::kMTilesPerWave * num_n_cluster_tiles;
	int tiles = total_tiles > cluster_idx
		? ceil_div(total_tiles - cluster_idx, num_clusters)
		: 0;

	PipelineTraits::init_barriers(
		smem.mainloop_pipeline,
		Traits::kTmaTransBytes,
		Traits::kConsumerThreads,
		typename Traits::ClusterShape{});
	if (threadIdx.x == 0) {
		cute::initialize_barrier(smem.logits_ready, 1);
		cute::initialize_barrier(smem.logits_free, 1);
	}
	sm90::cluster_pipeline_init_sm90<Compute>();

	if (warp_group == 0) {
		Registers::producer();
		if (warp == 3) {
			auto state = PipelineTraits::producer_start_state();
			auto pipe = PipelineTraits::make_producer(
				smem.mainloop_pipeline,
				Traits::kTmaTransBytes,
				Traits::kConsumerThreads,
				threadIdx.x == 3 * Traits::kWarpSize,
				typename Traits::ClusterShape{});
			cute::prefetch_tma_descriptor(tma_x.get_tma_descriptor());
			cute::prefetch_tma_descriptor(tma_w.get_tma_descriptor());
			if (cute::elect_one_sync()) {
				auto sA = make_tensor(
					make_smem_ptr(smem.a_data()),
					typename Traits::SmemLayoutA{});
				auto sB = make_tensor(
					make_smem_ptr(smem.b_data()),
					typename Traits::SmemLayoutB{});
				auto mX = tma_x.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.tokens),
					static_cast<int64_t>(params.hidden)));
				auto mW = tma_w.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.local_vocab),
					static_cast<int64_t>(params.hidden)));
				auto cta_x = tma_x.get_slice(Int<0>{});
				auto cta_w = tma_w.get_slice(Int<0>{});
				auto tAsA = cta_x.partition_D(sA);
				auto tBsB = cta_w.partition_D(sB);
				int m_base = wave * Config::kMTilesPerWave;

				for (int t = 0; t < tiles; ++t) {
					BackwardTileCoord tile = backward_cluster_n_m_fast_tile(
						cluster_idx + t * num_clusters,
						Config::kMTilesPerWave,
						cluster_rank);
					auto gX = local_tile(
						mX,
						make_tile(
							Int<Traits::kTileM>{},
							Int<Traits::kTileK>{}),
						make_coord(m_base + tile.m_tile, _));
					auto gW = local_tile(
						mW,
						make_tile(
							Int<Traits::kTileN>{},
							Int<Traits::kTileK>{}),
						make_coord(tile.n_tile, _));
					auto tAgA = cta_x.partition_S(gX);
					auto tBgB = cta_w.partition_S(gW);
					sm90::MainloopKLoopSm90<Compute>::produce(
						pipe,
						state,
						Launch::num_dz_k_tiles(params.hidden),
						[&](auto* barrier, int stage, int k_tile) {
							copy(
								tma_x.with(*barrier, 0),
								tAgA(_, _, _, k_tile),
								tAsA(_, _, _, stage));
							copy(
								tma_w.with(*barrier, 0),
								tBgB(_, _, _, k_tile),
								tBsB(_, _, _, stage));
						});
				}
				pipe.producer_tail(state);
			}
		} else if (warp < 3) {
			int epi_tid = static_cast<int>(threadIdx.x);
			auto sStore = make_tensor(
				make_smem_ptr(smem.store_data()),
				typename Traits::SmemStore{});
			auto mDz = tma_dz_store.get_tma_tensor(make_shape(
				static_cast<int64_t>(Config::kWaveRows),
				static_cast<int64_t>(
					Launch::padded_vocab(params.local_vocab))));
			auto cta_dz = tma_dz_store.get_slice(Int<0>{});
			auto tSsS = cta_dz.partition_S(sStore);
			auto tSgS = cta_dz.partition_D(local_tile(
				mDz,
				make_tile(
					Int<Traits::kTileM>{},
					Int<Traits::kStoreTileN>{}),
				make_coord(_, _)));
			float exp_scale =
				params.inverse_temperature * kBackwardLog2E;

			for (int t = 0; t < tiles; ++t) {
				if (epi_tid == 0) {
					cute::wait_barrier(smem.logits_ready, t & 1);
				}
				sm90::named_barrier_sync_sm90<
					Traits::kEpilogueBarrierId,
					Traits::kEpilogueThreads,
					Compute>();
				BackwardTileCoord tile = backward_cluster_n_m_fast_tile(
					cluster_idx + t * num_clusters,
					Config::kMTilesPerWave,
					cluster_rank);

				for (int row = epi_tid;
						row < Traits::kTileM;
						row += Traits::kEpilogueThreads) {
					int token =
						(wave * Config::kMTilesPerWave + tile.m_tile) *
							Traits::kTileM +
						row;
					float scale = 0.0f;
					float exp_bias = 0.0f;
					float correction_bias = 0.0f;
					float correction_slope = 0.0f;
					int target_local = -1;
					if (token < params.tokens) {
						std::int64_t target_id = params.target[token];
						if (target_id != params.ignore_index) {
							float lse = params.lse[token];
							scale = params.grad_output[token];
							exp_bias = -lse * kBackwardLog2E;
							if constexpr (ReturnEntropy) {
								float entropy = params.entropy[token];
								float entropy_scale =
									params.entropy_grad[token];
								correction_bias = fmaf(
									lse - entropy,
									entropy_scale,
									scale);
								correction_slope =
									-params.inverse_temperature *
									entropy_scale;
							}
							std::int64_t local =
								target_id - params.vocab_start;
							if (local >= 0 &&
								local < static_cast<std::int64_t>(
									params.local_vocab)) {
								target_local = static_cast<int>(local);
							}
						}
					}
					smem.row_scale[row] = scale;
					smem.row_lse[row] = exp_bias;
					smem.row_target[row] = target_local;
					if constexpr (ReturnEntropy) {
						smem.row_entropy[row] = correction_bias;
						smem.row_entropy_scale[row] = correction_slope;
					}
				}
				sm90::named_barrier_sync_sm90<
					Traits::kEpilogueBarrierId,
					Traits::kEpilogueThreads,
					Compute>();

				int vocab_base = tile.n_tile * Traits::kTileN;
				int lane = epi_tid & (Traits::kWarpSize - 1);
				CUTE_UNROLL
				for (int sub = 0;
						sub < Traits::kNumSubTiles;
						++sub) {
					for (int row = warp;
							row < Traits::kTileM;
							row += 3) {
						float scale = smem.row_scale[row];
						float exp_bias = smem.row_lse[row];
						int target = smem.row_target[row];
						float correction_bias = 0.0f;
						float correction_slope = 0.0f;
						if constexpr (ReturnEntropy) {
							correction_bias = smem.row_entropy[row];
							correction_slope =
								smem.row_entropy_scale[row];
						}
						CUTE_UNROLL
						for (int j = lane;
								j < Traits::kStoreTileN;
								j += Traits::kWarpSize) {
							int column =
								vocab_base +
								sub * Traits::kStoreTileN + j;
							float value = 0.0f;
							if (column < params.local_vocab) {
								float logit = static_cast<float>(
									smem.logit_row(row)[
										sub * Traits::kStoreTileN + j]);
								if constexpr (ReturnEntropy) {
									if (scale != 0.0f ||
										correction_slope != 0.0f) {
										float probability =
											backward_exp2_sm90(fmaf(
												logit,
												exp_scale,
												exp_bias));
										value = probability * fmaf(
											logit,
											correction_slope,
											correction_bias);
									}
								} else if (scale != 0.0f) {
									value = backward_exp2_sm90(fmaf(
										logit,
										exp_scale,
										exp_bias)) *
										scale;
								}
								if (column == target) value -= scale;
								value *= params.inverse_temperature;
							}
							sStore(row, j, sub) =
								static_cast<typename Traits::Element>(value);
						}
					}
					tma_store_fence();
					sm90::named_barrier_sync_sm90<
						Traits::kEpilogueBarrierId,
						Traits::kEpilogueThreads,
						Compute>();
					if (warp == 0 && cute::elect_one_sync()) {
						copy(
							tma_dz_store,
							tSsS(_, _, _, sub),
							tSgS(
								_, _, _, tile.m_tile,
								tile.n_tile *
									Traits::kNumSubTiles +
									sub));
						tma_store_arrive();
					}
				}
				sm90::named_barrier_sync_sm90<
					Traits::kEpilogueBarrierId,
					Traits::kEpilogueThreads,
					Compute>();
				if (warp == 0) {
					wait_tma_store_complete();
				}
				sm90::named_barrier_sync_sm90<
					Traits::kEpilogueBarrierId,
					Traits::kEpilogueThreads,
					Compute>();
				if (epi_tid == 0) {
					cute::arrive_barrier(smem.logits_free);
				}
			}
		}
	} else {
		Registers::consumer();
		typename Traits::TiledMma tiled_mma;
		int tid_in_mma =
			static_cast<int>(threadIdx.x) - Config::kFirstConsumerThread;
		auto thr_mma = tiled_mma.get_slice(tid_in_mma);
		auto sA = make_tensor(
			make_smem_ptr(smem.a_data()),
			typename Traits::SmemLayoutA{});
		auto sB = make_tensor(
			make_smem_ptr(smem.b_data()),
			typename Traits::SmemLayoutB{});
		auto tCsA = thr_mma.partition_A(sA);
		auto tCsB = thr_mma.partition_B(sB);
		auto accum = partition_fragment_C(
			tiled_mma, typename Traits::AccumShape{});
		auto coord_c = thr_mma.partition_C(
			make_identity_tensor(typename Traits::AccumShape{}));
		typename Traits::PipelineState read_state;
		typename Traits::PipelineState release_state;
		auto pipe = PipelineTraits::make_consumer(
			smem.mainloop_pipeline,
			Traits::kTmaTransBytes,
			Traits::kConsumerThreads,
			typename Traits::ClusterShape{});

		for (int t = 0; t < tiles; ++t) {
			CUTE_UNROLL
			for (int i = 0; i < decltype(cute::size(accum))::value; ++i) {
				asm volatile(
					"mov.f32 %0, 0f00000000;" : "=f"(accum(i)));
			}
			sm90::MainloopKLoopSm90<Compute>::consume(
				pipe,
				read_state,
				release_state,
				Launch::num_dz_k_tiles(params.hidden),
				[&](int stage) {
					warpgroup_fence_operand(accum);
					warpgroup_arrive();
					gemm(
						tiled_mma,
						tCsA(_, _, _, stage),
						tCsB(_, _, _, stage),
						accum);
					warpgroup_commit_batch();
				},
				[&]() { warpgroup_fence_operand(accum); });

			if (t > 0) {
				if (tid_in_mma == 0) {
					cute::wait_barrier(
						smem.logits_free,
						(t - 1) & 1);
				}
				sm90::named_barrier_sync_sm90<
					Traits::kMmaBarrierId,
					Traits::kConsumerThreads,
					Compute>();
			}
			CUTE_UNROLL
			for (int i = 0; i < decltype(cute::size(accum))::value; ++i) {
				int row = get<0>(coord_c(i));
				int column = get<1>(coord_c(i));
				smem.logit_row(row)[column] =
					static_cast<typename Traits::ElementLogit>(accum(i));
			}
			cutlass::arch::fence_view_async_shared();
			asm volatile("fence.acq_rel.cta;" ::: "memory");
			sm90::named_barrier_sync_sm90<
				Traits::kMmaBarrierId,
				Traits::kConsumerThreads,
				Compute>();
			if (tid_in_mma == 0) {
				cute::arrive_barrier(smem.logits_ready);
			}
		}
	}

	__syncthreads();
	sm90::cluster_exit_sm90<Compute>();
}

template <
	class TmaDz,
	class TmaWt,
	class TmaDxStore,
	class TmaDzt,
	class TmaXt,
	class TmaDwStore,
	class TmaDwAdd>
struct BackwardClusterCombinedTmaBundleSm90 {
	TmaDz dz;
	TmaWt wt;
	TmaDxStore dx_store;
	TmaDzt dzt;
	TmaXt xt;
	TmaDwStore dw_store;
	TmaDwAdd dw_add;
};

template <
	int DxSplitK,
	int Compute,
	class CommConfig,
	class Bundle>
__global__ __launch_bounds__(
	BackwardDxCluster2TraitsSm90<Compute>::kNumThreads, 1)
void backward_dx_cluster2_gemm_wave_kernel_sm90(
		__grid_constant__ const Bundle tma,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const liger_cute::detail::NvlsReduceView mapping,
		__grid_constant__ const BackwardGemmParamsSm90<Compute> params,
		int wave) {
	static_assert(DxSplitK == 1 || DxSplitK == 2);
	using Traits = BackwardDxCluster2TraitsSm90<Compute>;
	using DwTraits = BackwardGemmTraitsSm90<Compute, 4, 64, 2>;
	using Config = typename Traits::Config;
	using Launch = BackwardGemmLaunchSm90<Compute>;
	using Smem = BackwardDxCluster2SmemSm90<Compute>;
	using PipelineTraits = typename Traits::PipelineTraits;
	using ClusterShape = typename Traits::ClusterShape;
	using Registers = sm90::WarpSpecializedRegistersSm90<
		DxSplitK == 2 ? 72 : 56,
		DxSplitK == 2 ? 216 : 224,
		Compute>;
	static_assert(Traits::kStages == 4);
	static_assert(DwTraits::kStages == 4);
	static_assert(DwTraits::kDwStoreTileN == 64);
	static_assert(DwTraits::kDwStoreStages == 2);
	static_assert(Traits::kTileM == DwTraits::kTileM);
	static_assert(Traits::kTileN == DwTraits::kTileN);
	static_assert(Traits::kTileK == DwTraits::kTileK);
	static_assert(Traits::kTmaTransBytes == DwTraits::kTmaTransBytes);
	static_assert(cute::is_same_v<
		typename Traits::PipelineState,
		typename DwTraits::PipelineState>);

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	int warp_group = cutlass::canonical_warp_group_idx();
	constexpr int kDwKTiles = Launch::num_dw_k_tiles();

	PipelineTraits::init_barriers(
		smem.mainloop_pipeline,
		Traits::kTmaTransBytes,
		Traits::kConsumerThreads,
		ClusterShape{});
	if (threadIdx.x == 0) {
		CUTE_UNROLL
		for (int comm_warp = 0;
				comm_warp < kDxCommWarpsPerChannel;
				++comm_warp) {
			CUTE_UNROLL
			for (int stage = 0; stage < kDxRingStages; ++stage) {
				cute::initialize_barrier(
					smem.dx_ready_epoch[comm_warp][stage], 1);
				cute::initialize_barrier(
					smem.dx_consumed_barrier[comm_warp][stage], 1);
			}
		}
	}
	sm90::cluster_pipeline_init_sm90<Compute>();

	if (warp_group == 0) {
		Registers::producer();
		int warp = cutlass::canonical_warp_idx_sync();
		if (warp == 0) {
			int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
			int cluster_idx = static_cast<int>(blockIdx.z);
			int num_clusters = static_cast<int>(gridDim.z);
			int num_n_tiles = Launch::num_dx_n_tiles(params.hidden);
			int num_k_tiles =
				Launch::num_dx_k_tiles(params.local_vocab) / DxSplitK;
			constexpr int num_m_pairs =
				Config::kMTilesPerWave / Traits::kClusterM;
			int total_cluster_tiles =
				num_m_pairs * num_n_tiles * DxSplitK;
			auto state = PipelineTraits::producer_start_state();
			auto pipe = PipelineTraits::make_producer(
				smem.mainloop_pipeline,
				Traits::kTmaTransBytes,
				Traits::kConsumerThreads,
				threadIdx.x == 0,
				ClusterShape{});
			cute::prefetch_tma_descriptor(tma.dz.get_tma_descriptor());
			cute::prefetch_tma_descriptor(tma.wt.get_tma_descriptor());
			cute::prefetch_tma_descriptor(tma.dx_store.get_tma_descriptor());
			cute::prefetch_tma_descriptor(tma.dzt.get_tma_descriptor());
			cute::prefetch_tma_descriptor(tma.xt.get_tma_descriptor());
			if (cute::elect_one_sync()) {
				auto sA = make_tensor(
					make_smem_ptr(smem.a_data()),
					typename Traits::SmemA::Staged{});
				auto sB0 = make_tensor(
					make_smem_ptr(smem.b0_data()),
					typename Traits::SmemB::Staged{});
				auto mDz = tma.dz.get_tma_tensor(make_shape(
					static_cast<int64_t>(Config::kWaveRows),
					static_cast<int64_t>(
						Launch::padded_vocab(params.local_vocab))));
				auto mWt = tma.wt.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.hidden),
					static_cast<int64_t>(params.local_vocab)));
				auto cta_dz = tma.dz.get_slice(Int<0>{});
				auto cta_wt = tma.wt.get_slice(cluster_rank);
				auto tAsA = cta_dz.partition_D(sA);
				auto tB0sB = cta_wt.partition_D(sB0);
				std::uint16_t mcast_mask =
					sm90::cluster_multicast_mask_m_sm90<
						ClusterShape, Compute>();

				for (int index = cluster_idx;
						index < total_cluster_tiles;
						index += num_clusters) {
					int logical_index = index / DxSplitK;
					int split_k = index % DxSplitK;
					int first_k_tile = split_k * num_k_tiles;
					BackwardTileCoord tile =
						backward_cluster_m_m_fast_tile(
							logical_index, num_m_pairs, cluster_rank);
					auto gDz = local_tile(
						mDz,
						make_tile(
							Int<Traits::kTileM>{},
							Int<Traits::kTileK>{}),
						make_coord(tile.m_tile, _));
					auto gWt0 = local_tile(
						mWt,
						make_tile(
							Int<Traits::kFragmentTileN>{},
							Int<Traits::kTileK>{}),
						make_coord(
							Traits::kNumFragments * tile.n_tile, _));
					auto tAgA = cta_dz.partition_S(gDz);
					auto tB0gB = cta_wt.partition_S(gWt0);
					// dZ is loaded per CTA; only the shared W^T panel is
					// multicast to the adjacent-token cluster peer.
					sm90::MainloopKLoopSm90<Compute>::produce(
						pipe, state, num_k_tiles,
						[&](auto* barrier, int stage, int k_tile) {
							copy(tma.dz.with(*barrier, 0),
								tAgA(_, _, _, first_k_tile + k_tile),
								tAsA(_, _, _, stage));
							copy(tma.wt.with(*barrier, mcast_mask),
								tB0gB(_, _, _, first_k_tile + k_tile),
								tB0sB(_, _, _, stage));
						});
				}
			}
			int dw_n_tiles = Launch::num_dw_n_tiles(params.hidden);
			int dw_n_pairs = ceil_div(dw_n_tiles, Traits::kClusterM);
			int dw_total_units =
				Launch::num_dw_m_tiles(params.local_vocab) * dw_n_pairs;
			int dw_units = dw_total_units > cluster_idx
				? ceil_div(dw_total_units - cluster_idx, num_clusters)
				: 0;
			std::uint16_t dzt_mcast_mask =
				sm90::cluster_multicast_mask_m_sm90<
					ClusterShape, Compute>();
			DwPhaseSm90<Compute, DwTraits>::template produce<false>(
				pipe,
				state,
				smem,
				tma.dzt,
				tma.xt,
				params,
				Launch::padded_vocab(params.local_vocab),
				wave,
				dw_n_pairs,
				kDwKTiles,
				dw_units,
				cluster_idx,
				num_clusters,
				cluster_rank,
				dzt_mcast_mask);
			if (cute::elect_one_sync()) pipe.producer_tail(state);
		} else if (
				warp >= CommConfig::kFirstCommWarp &&
				warp <= CommConfig::kLastCommWarp) {
			int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
			int cluster_idx = static_cast<int>(blockIdx.z);
			int num_clusters = static_cast<int>(gridDim.z);
			int num_n_tiles = Launch::num_dx_n_tiles(params.hidden);
			constexpr int num_m_pairs =
				Config::kMTilesPerWave / Traits::kClusterM;
			int total_cluster_tiles =
				num_m_pairs * num_n_tiles * DxSplitK;
			int cta = cluster_idx * Traits::kClusterM + cluster_rank;
			int comm_warp = warp - CommConfig::kFirstCommWarp;
			constexpr std::size_t kMessageElements = CommConfig::kTileElements;
			constexpr std::size_t kSegmentElements =
				kMessageElements / kDxCommWarpsPerChannel;
			int local_index = 0;
			for (int index = cluster_idx;
					index < total_cluster_tiles;
					index += num_clusters, ++local_index) {
				int stage = local_index % CommConfig::kNumStages;
				int pass = local_index / CommConfig::kNumStages;
				if (static_cast<int>(threadIdx.x) % Traits::kWarpSize == 0) {
					cute::wait_barrier(
						smem.dx_ready_epoch[comm_warp][stage],
						pass & 1);
				}
				__syncwarp();

				std::size_t base =
					dx_slot_offset<CommConfig>(cta, 0, stage);
				std::size_t segment_begin =
					static_cast<std::size_t>(comm_warp) *
					kSegmentElements;
				BackwardTileCoord tile = backward_cluster_m_m_fast_tile(
					index / DxSplitK, num_m_pairs, cluster_rank);
				std::size_t durable_tile =
					static_cast<std::size_t>(tile.m_tile) * num_n_tiles +
					tile.n_tile;
				std::size_t tiles_per_wave =
					static_cast<std::size_t>(Config::kMTilesPerWave) *
					num_n_tiles;
				std::size_t packed_tile_elements =
					kMessageElements / mapping.size;
				std::size_t packed_segment_elements =
					kSegmentElements / mapping.size;
				float* durable = mapping.reduced_shard +
					(static_cast<std::size_t>(wave) * tiles_per_wave +
					 durable_tile) * packed_tile_elements +
					static_cast<std::size_t>(comm_warp) *
						packed_segment_elements;

				if (mapping.size == 1) {
					constexpr std::size_t kSegmentVectors =
						kSegmentElements / 4;
					for (std::size_t vector =
							 static_cast<std::size_t>(threadIdx.x) %
							 Traits::kWarpSize;
							vector < kSegmentVectors;
							vector += Traits::kWarpSize) {
						float4 value =
							reinterpret_cast<const float4*>(
								comm.partial + base + segment_begin)[vector];
						if constexpr (DxSplitK == 1) {
							reinterpret_cast<float4*>(durable)[vector] =
								value;
						} else {
							float* output = durable + 4 * vector;
							atomicAdd(output + 0, value.x);
							atomicAdd(output + 1, value.y);
							atomicAdd(output + 2, value.z);
							atomicAdd(output + 3, value.w);
						}
					}
				} else {
					std::size_t ready_offset =
						dx_sync_offset<CommConfig>(
							cta,
							comm_warp,
							stage,
							kDxReadyPhase,
							mapping.size);
					std::size_t complete_offset =
						dx_sync_offset<CommConfig>(
							cta,
							comm_warp,
							stage,
							kDxCompletePhase,
							mapping.size);
					std::uint64_t epoch =
						dx_epoch_base(comm) |
						(static_cast<std::uint64_t>(wave + 1) << 16) |
						static_cast<std::uint64_t>(pass + 1);
					liger_cute::detail::nvls_barrier_warp(
						comm.sync + ready_offset,
						mapping.multicast_sync + ready_offset,
						mapping.rank,
						mapping.size,
						epoch);
					if constexpr (DxSplitK == 1) {
						liger_cute::detail::nvls_sum_reduce_scatter_warp(
							durable,
							mapping.multicast_partial +
								base + segment_begin,
							kSegmentElements,
							mapping.rank,
							mapping.size);
					} else {
						liger_cute::detail::nvls_sum_reduce_scatter_warp(
							mapping.multicast_reduced +
								base + segment_begin,
							mapping.multicast_partial +
								base + segment_begin,
							kSegmentElements,
							mapping.rank,
							mapping.size);
						// Re-read the lane and accumulate one scalar at a time
						// so no vector or lane value lives across the atomics.
						for (std::size_t element =
								 static_cast<std::size_t>(threadIdx.x) %
								 Traits::kWarpSize;
								element < packed_segment_elements;
								element += Traits::kWarpSize) {
							atomicAdd(
								durable + element,
								mapping.multicast_reduced[
									base + segment_begin + element]);
						}
						__syncwarp();
					}
					liger_cute::detail::nvls_barrier_warp(
						comm.sync + complete_offset,
						mapping.multicast_sync + complete_offset,
						mapping.rank,
						mapping.size,
						epoch);
				}
				__syncwarp();
				if (static_cast<int>(threadIdx.x) % Traits::kWarpSize == 0) {
					cute::arrive_barrier(
						smem.dx_consumed_barrier[
							comm_warp][stage]);
				}
			}
		}
	} else {
		Registers::consumer();
		int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
		int cluster_idx = static_cast<int>(blockIdx.z);
		int num_clusters = static_cast<int>(gridDim.z);
		int num_n_tiles = Launch::num_dx_n_tiles(params.hidden);
		int num_k_tiles =
			Launch::num_dx_k_tiles(params.local_vocab) / DxSplitK;
		constexpr int num_m_pairs =
			Config::kMTilesPerWave / Traits::kClusterM;
		int total_cluster_tiles =
			num_m_pairs * num_n_tiles * DxSplitK;
		int cta = cluster_idx * Traits::kClusterM + cluster_rank;
		typename Traits::PipelineState read_state;
		typename Traits::PipelineState release_state;
		auto pipe = PipelineTraits::make_consumer(
			smem.mainloop_pipeline,
			Traits::kTmaTransBytes,
			Traits::kConsumerThreads,
			ClusterShape{});
		auto accum0 = partition_fragment_C(
			typename Traits::TiledMma{}, typename Traits::AccumShape{});
		using DwMma = typename DwTraits::DwMmaTraits::Type;
		using DwAccumulator = decltype(partition_fragment_C(
			*static_cast<DwMma*>(nullptr),
			typename DwTraits::AccumShape{}));
		static_assert(cute::is_same_v<
			typename Traits::AccumShape,
			typename DwTraits::AccumShape>);
		static_assert(cute::is_same_v<decltype(accum0), DwAccumulator>);
		{
		typename Traits::TiledMma tiled_mma;
		int tid_in_mma =
			static_cast<int>(threadIdx.x) - Config::kFirstConsumerThread;
		auto thr_mma = tiled_mma.get_slice(tid_in_mma);
		auto sA = make_tensor(
			make_smem_ptr(smem.a_data()),
			typename Traits::SmemA::Staged{});
		auto sB0 = make_tensor(
			make_smem_ptr(smem.b0_data()),
			typename Traits::SmemB::Staged{});
		auto tCsA = thr_mma.partition_A(sA);
		auto tCsB0 = thr_mma.partition_B(sB0);
		auto sStore = make_tensor(
			make_smem_ptr(smem.dx_store_data()),
			typename Traits::SmemStore::Staged{});
		auto mStore = tma.dx_store.get_tma_tensor(make_shape(
			static_cast<int64_t>(
				kMaxDxResidentCtas * kDxCommWarpsPerChannel *
				CommConfig::kNumStages * CommConfig::kTilesPerReduce *
				Traits::kTileM),
			static_cast<int64_t>(Traits::kTileN)));
		auto cta_store = tma.dx_store.get_slice(Int<0>{});
		auto tSsS = cta_store.partition_S(sStore);
		auto tSgS = cta_store.partition_D(local_tile(
			mStore,
			make_tile(
				Int<Traits::kTileM>{}, Int<Traits::kStoreTileN>{}),
			make_coord(_, _)));
		typename Traits::StoreMmaTraits::Type store_mma;
		auto store_thread = store_mma.get_slice(tid_in_mma);
		auto tDsD = store_thread.partition_C(sStore);
		auto output_fragment =
			make_tensor<float>(shape(tDsD(_, _, _, Int<0>{})));
		constexpr int kOutputElements =
			decltype(cute::size(output_fragment))::value;
		static_assert(
			(Traits::kTileN / Traits::kStoreTileN) * kOutputElements ==
				decltype(cute::size(accum0))::value,
			"N32 store fragments must cover the native N256 accumulator");
		bool issuer = (tid_in_mma / Traits::kWarpSize) == 0;
		int local_index = 0;
		for (int index = cluster_idx;
				index < total_cluster_tiles;
				index += num_clusters, ++local_index) {
			int stage = local_index % CommConfig::kNumStages;
			int pass = local_index / CommConfig::kNumStages;
			if (pass > 0) {
				if (tid_in_mma == 0) {
					CUTE_UNROLL
					for (int comm_warp = 0;
							comm_warp < kDxCommWarpsPerChannel;
							++comm_warp) {
						cute::wait_barrier(
							smem.dx_consumed_barrier[
								comm_warp][stage],
							(pass - 1) & 1);
					}
				}
				backward_consumer_barrier_sm90<
					Config::kDxEpilogueBarrierId, Compute>();
			}
			CUTE_UNROLL
			for (int i = 0; i < decltype(cute::size(accum0))::value; ++i) {
				asm volatile(
					"mov.f32 %0, 0f00000000;" : "=f"(accum0(i)));
			}
			sm90::MainloopKLoopSm90<Compute>::consume(
				pipe, read_state, release_state, num_k_tiles,
				[&](int stage) {
					warpgroup_fence_operand(accum0);
					warpgroup_arrive();
					gemm(tiled_mma, tCsA(_, _, _, stage),
						tCsB0(_, _, _, stage), accum0);
					warpgroup_commit_batch();
				},
				[&]() {
					warpgroup_fence_operand(accum0);
				});

			BackwardTileCoord tile =
				backward_cluster_m_m_fast_tile(
					index / DxSplitK, num_m_pairs, cluster_rank);
			std::size_t slot_tile =
				dx_slot_offset<CommConfig>(cta, 0, stage) /
				CommConfig::kTileElements;

			CUTE_UNROLL
			for (int slice = 0;
					slice < Traits::kTileN / Traits::kStoreTileN;
					++slice) {
				if (slice > 0) {
					if (issuer) {
						wait_tma_store_complete();
					}
					backward_consumer_barrier_sm90<
						Config::kDxEpilogueBarrierId, Compute>();
				}
				CUTE_UNROLL
				for (int i = 0; i < kOutputElements; ++i) {
					output_fragment(i) =
						accum0(slice * kOutputElements + i);
				}
				copy(output_fragment, tDsD(_, _, _, Int<0>{}));
				tma_store_fence();
				backward_consumer_barrier_sm90<
					Config::kDxEpilogueBarrierId, Compute>();
				if (issuer && cute::elect_one_sync()) {
					copy(
						tma.dx_store,
						tSsS(_, _, _, Int<0>{}),
						tSgS(
							_, _, _,
							static_cast<int>(slot_tile),
							slice));
					tma_store_arrive();
				}
			}
			// Publish the completed FP32 tile before waking communication.
			if (issuer) {
				wait_tma_store_complete();
				liger_cute::detail::publish_local_reduce_source();
			}
			backward_consumer_barrier_sm90<
				Config::kDxEpilogueBarrierId, Compute>();
			if (tid_in_mma == 0) {
				CUTE_UNROLL
				for (int comm_warp = 0;
						comm_warp < kDxCommWarpsPerChannel;
						++comm_warp) {
					cute::arrive_barrier(
						smem.dx_ready_epoch[comm_warp][stage]);
				}
			}
		}
		}

		int dw_n_tiles = Launch::num_dw_n_tiles(params.hidden);
		int dw_n_pairs = ceil_div(dw_n_tiles, Traits::kClusterM);
		int dw_total_units =
			Launch::num_dw_m_tiles(params.local_vocab) * dw_n_pairs;
		int dw_units = dw_total_units > cluster_idx
			? ceil_div(dw_total_units - cluster_idx, num_clusters)
			: 0;
		DwPhaseSm90<Compute, DwTraits>::
			template consume_with_accumulator<false>(
				pipe,
				read_state,
				release_state,
				accum0,
				smem,
				tma.dw_store,
				tma.dw_add,
				params,
				wave,
				dw_n_pairs,
				kDwKTiles,
				dw_units,
				cluster_idx,
				num_clusters,
				cluster_rank);
	}

	__syncthreads();
	sm90::cluster_exit_sm90<Compute>();
}


}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
