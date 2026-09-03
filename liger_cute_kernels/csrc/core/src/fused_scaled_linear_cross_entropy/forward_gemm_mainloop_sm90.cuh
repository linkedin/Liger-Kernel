#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy — executable forward mainloop.
//
// Structural port of _fused_scaled_cross_entropy_forward_fragment_sm90.py.
// The CuTe/CUTLASS mechanics (TiledMMA construction, PipelineTmaAsync driving,
// TMA partitioning, warpgroup fences) follow the repository's SM90 idioms in
// csrc/core/src/moe, but the loop nest, work assignment, staging order,
// masking and online-softmax sequencing are a literal transcription of the
// reference kernel — not a generic CUTLASS GEMM.
//
// CTA layout (384 threads = 12 warps, cluster = (2, 1, 1)):
//   warp  0     : TMA producer, 24 registers. Loads the CTA's own M128xK64 X
//                 tile and multicasts two N160xK64 W panels to the cluster
//                 peer through one 3-stage mbarrier pipeline.
//   warp  1     : last-arrival local MAX/SUM communication epilogue.
//   warps 2..3  : idle; they still take PRODUCER_REGISTERS so the warp group's
//                 setmaxnreg is uniform.
//   warps 4..11 : two WGMMA consumer warp groups, 240 registers. Each owns 64
//                 token rows and two FP32 N160 accumulators.
//
// Mainloop: the consumer read state and the release state are deliberately
// separate. Stage zero is issued without a release; every later stage is
// committed, `warpgroup_wait<1>()` retires the previous WGMMA group, and only
// then is that previous AB stage returned to the producer. The tail uses
// `warpgroup_wait<0>()` before the final release. This one-stage lag keeps TMA
// from overwriting operands still feeding a live WGMMA group.
//
// Epilogue: each N160 accumulator is staged as two N80 FP16 SMEM slices, so a
// logical N320 tile produces four N80 online-softmax folds. Both padded
// staging buffers are fully consumed before either is overwritten. Two threads
// cooperate per token row (one N40 chunk each) and merge through SMEM once the
// CTA's vocabulary split has been folded.
// ═══════════════════════════════════════════════════════════════════════════

// gemm_sm90.cuh carries the reusable SM90 producer/consumer mechanics and
// enables CuTe's extended MMA shapes (where the N160 WGMMA atom lives), so it
// must be the first CUTLASS/CuTe include of any TU using this header.
#include "gemm_sm90.cuh"

#include "dx_reduce.cuh"
#include "forward_gemm_sm90.cuh"
#include "liger_cute/detail/local_reduce.cuh"
#include "online_softmax.cuh"

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

// ───────────────────────────────────────────────────────────────────────────
// Traits
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ForwardGemmTraitsSm90 {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;
	// The N80 epilogue staging slices are FP16, mirroring the reference's
	// two padded Float16 SMEM buffers.
	using ElementLogit = cutlass::half_t;

	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kAccumulatorTileN = Config::kAccumulatorTileN;
	static constexpr int kLogicalTileN = Config::kLogicalTileN;
	static constexpr int kEpilogueSliceN = Config::kEpilogueSliceN;
	static constexpr int kSoftmaxChunkN = Config::kSoftmaxChunkN;
	static constexpr int kSegmentN = Config::kSegmentN;
	static constexpr int kChunksPerSegment = Config::kChunksPerSegment;
	static constexpr int kStages = Config::kMainloopStages;
	static constexpr int kClusterM = Config::kClusterM;
	static constexpr int kNumAccumulators = Config::kNumAccumulators;
	static constexpr int kNumThreads = Config::kNumThreads;
	static constexpr int kWarpSize = fused_scaled_linear_cross_entropy::kWarpSize;
	static constexpr int kWarpGroupSize = Config::kWarpGroupSize;
	static constexpr int kNumMmaWarpGroups = Config::kNumMmaWarpGroups;
	static constexpr int kConsumerThreads = Config::kConsumerThreads;
	static constexpr int kLogitBuffers = Config::kLogitBuffers;
	static constexpr int kLogitStride = Config::kLogitStride;
	static constexpr int kRowsPerWarpGroup = Config::kRowsPerWarpGroup;

	using ClusterShape = Shape<Int<kClusterM>, _1, _1>;

	// hopper_helpers.make_trivial_tiled_mma(BF16, BF16, K, K, F32, (2,1,1),
	// (64, ACCUMULATOR_N)): the M64 WGMMA atom, two warp groups split over M.
	using TiledMmaTraits = sm90::TiledMmaMSplitSm90<
		Element,
		Element,
		ElementAccum,
		kAccumulatorTileN,
		kNumMmaWarpGroups,
		GMMA::Major::K,
		GMMA::Major::K,
		Compute>;
	using TiledMma = typename TiledMmaTraits::Type;
	// Pin the atom: the reference's N160 fragment must not silently decay to a
	// stack of narrower WGMMA issues.
	static_assert(
		cute::is_same_v<
			typename TiledMmaTraits::Atom,
			SM90_64x160x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>,
		"the forward accumulator must map onto the single N160 WGMMA atom");
	static_assert(TiledMmaTraits::kTileM == kTileM);
	static_assert(TiledMmaTraits::kTileN == kAccumulatorTileN);

	// make_smem_layout_a/b for K-major BF16 with a K64 tile => 128 B rows,
	// hence the SW128 swizzle atom.
	using SmemTraitsX =
		sm90::SmemLayoutKMajorSm90<Element, kTileM, kTileK, kStages, Compute>;
	using SmemTraitsW = sm90::SmemLayoutKMajorSm90<
		Element, kAccumulatorTileN, kTileK, kStages, Compute>;

	using SmemLayoutX1 = typename SmemTraitsX::Single;
	using SmemLayoutW1 = typename SmemTraitsW::Single;
	using SmemLayoutX = typename SmemTraitsX::Staged;
	using SmemLayoutW = typename SmemTraitsW::Staged;

	using PipelineTraits = sm90::MainloopPipelineSm90<kStages, Compute>;
	using MainloopPipeline = typename PipelineTraits::Type;
	using PipelineState = typename PipelineTraits::State;

	static constexpr int kTmaTransBytesX = SmemTraitsX::kStageBytes;
	static constexpr int kTmaTransBytesW = SmemTraitsW::kStageBytes;
	// tx_count = x_stage_bytes + NUM_ACCUMULATORS * w_stage_bytes: X and both
	// N160 panels share one mbarrier per stage.
	static constexpr int kTmaTransBytes =
		kTmaTransBytesX + kNumAccumulators * kTmaTransBytesW;
};

// ───────────────────────────────────────────────────────────────────────────
// Shared memory — same member order as the reference SharedStorage struct.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute, bool ReturnEntropy>
struct ForwardGemmSmemSm90 {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	using Element = typename Traits::Element;
	using ElementLogit = typename Traits::ElementLogit;

	static constexpr int kSmemX = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int kSmemW = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int kLogitElements =
		Traits::kLogitBuffers * Traits::kTileM * Traits::kLogitStride;
	static constexpr int kWeightedElements =
		ReturnEntropy ? Traits::kTileM : 1;

	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(1024) Element smem_x[kSmemX];
	alignas(1024) Element smem_w0[kSmemW];
	alignas(1024) Element smem_w1[kSmemW];
	alignas(1024) ElementLogit smem_logits[kLogitElements];

	// Cross-segment exchange: segment 1 publishes, segment 0 merges.
	alignas(16) float segment_max[Traits::kTileM];
	alignas(16) float segment_sum[Traits::kTileM];
	alignas(16) float segment_target[Traits::kTileM];
	alignas(16) float segment_weighted[kWeightedElements];
	int finalizer;

	CUTE_DEVICE Element* x_data() { return &smem_x[0]; }
	CUTE_DEVICE Element* w0_data() { return &smem_w0[0]; }
	CUTE_DEVICE Element* w1_data() { return &smem_w1[0]; }

	// s_logits[buffer, row, :] over the (LOGIT_BUFFERS, TILE_M, LOGIT_STRIDE)
	// padded layout.
	CUTE_DEVICE ElementLogit* logit_row(int buffer, int row) {
		return &smem_logits[
			(buffer * Traits::kTileM + row) * Traits::kLogitStride];
	}
};

// ───────────────────────────────────────────────────────────────────────────
// EPILOGUE_BARRIER_ID named barrier — the two consumer warp groups only.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
CUTE_DEVICE void forward_epilogue_barrier_sm90() {
	using Config = ForwardGemmConfigSm90<Compute>;
	sm90::named_barrier_sync_sm90<
		Config::kEpilogueBarrierId, Config::kConsumerThreads, Compute>();
}

// ───────────────────────────────────────────────────────────────────────────
// Work assignment — port of the kernel's cluster_work decomposition.
// ───────────────────────────────────────────────────────────────────────────

struct ForwardGemmWorkSm90 {
	int m_pair;
	int split_id;
	int split_count;
	int raw_pid_m;
	int pid_m;
	int output_work;
	bool store_enabled;
};

template <int Compute = 90>
CUTE_DEVICE ForwardGemmWorkSm90 forward_gemm_assign_work_sm90(
		const ForwardGemmSplitSm90<Compute>& split,
		int cluster_work,
		int cluster_rank) {
	using Config = ForwardGemmConfigSm90<Compute>;

	int m_pair;
	int split_id;
	int split_count;
	if (split.extra_m_pairs == 0) {
		m_pair = cluster_work / split.base_split_n;
		split_id = cluster_work % split.base_split_n;
		split_count = split.base_split_n;
	} else {
		// The first extra_m_pairs M pairs carry one extra vocabulary split.
		int extra_work = split.extra_m_pairs * split.split_n;
		if (cluster_work < extra_work) {
			m_pair = cluster_work / split.split_n;
			split_id = cluster_work % split.split_n;
			split_count = split.split_n;
		} else {
			int normal_work = cluster_work - extra_work;
			m_pair = split.extra_m_pairs + normal_work / split.base_split_n;
			split_id = normal_work % split.base_split_n;
			split_count = split.base_split_n;
		}
	}

	ForwardGemmWorkSm90 work;
	work.m_pair = m_pair;
	work.split_id = split_id;
	work.split_count = split_count;
	work.raw_pid_m = m_pair * Config::kClusterM + cluster_rank;
	work.store_enabled = work.raw_pid_m < split.num_m_tiles;
	// The tail CTA of an odd M-tile count still feeds the cluster pipeline;
	// it re-reads the last valid M tile and drops its stores.
	work.pid_m = work.store_enabled
		? work.raw_pid_m
		: (split.num_m_tiles - 1);
	work.output_work = work.raw_pid_m * split.split_n + split_id;
	return work;
}

// ───────────────────────────────────────────────────────────────────────────
// Producer — warp 0. X is CTA-private, both W panels are cluster-multicast.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ForwardGemmProducerSm90 {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	template <bool ReturnEntropy, class Pipeline, class TmaLoadX, class TmaLoadW>
	CUTE_DEVICE static void run(
			Pipeline& pipe,
			typename Traits::PipelineState& state,
			ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
			TmaLoadX const& tma_load_x,
			TmaLoadW const& tma_load_w,
			ForwardGemmParamsSm90<Compute> const& params,
			ForwardGemmWorkSm90 const& work,
			int num_split_tiles,
			int num_k_tiles) {
		// prefetch_descriptor(tma_x) / prefetch_descriptor(tma_w): warm the
		// descriptor cache from the whole producer warp before gating.
		cute::prefetch_tma_descriptor(tma_load_x.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma_load_w.get_tma_descriptor());

		if (cute::elect_one_sync() == 0) return;

		auto sX = make_tensor(
			make_smem_ptr(smem.x_data()), typename Traits::SmemLayoutX{});
		auto sW0 = make_tensor(
			make_smem_ptr(smem.w0_data()), typename Traits::SmemLayoutW{});
		auto sW1 = make_tensor(
			make_smem_ptr(smem.w1_data()), typename Traits::SmemLayoutW{});

		// int64 extents: tokens * hidden and local_vocab * hidden both exceed
		// INT_MAX at production shapes and would wrap CuTe's layout algebra.
		auto mX = tma_load_x.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.tokens),
			static_cast<int64_t>(params.hidden)));
		auto mW = tma_load_w.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.local_vocab),
			static_cast<int64_t>(params.hidden)));

		// X is not multicast; the W panels are multicast along the cluster's
		// M mode, so this CTA issues its own slice of every W box.
		uint32_t cluster_rank = cute::block_rank_in_cluster();
		auto cta_tma_x = tma_load_x.get_slice(Int<0>{});
		auto cta_tma_w = tma_load_w.get_slice(cluster_rank);

		auto tXsX = cta_tma_x.partition_D(sX);
		auto tW0sW0 = cta_tma_w.partition_D(sW0);
		auto tW1sW1 = cta_tma_w.partition_D(sW1);

		auto gX = local_tile(
			mX,
			make_tile(Int<Traits::kTileM>{}, Int<Traits::kTileK>{}),
			make_coord(work.pid_m, _));
		auto tXgX = cta_tma_x.partition_S(gX);

		// make_layout_image_mask(cta_layout, cluster_coord, mode=0): the
		// cluster is (ClusterM, 1, 1), so the W multicast covers every CTA.
		uint16_t mcast_mask_w =
			sm90::cluster_multicast_mask_m_sm90<
				typename Traits::ClusterShape, Compute>();

		for (int local_n = 0; local_n < num_split_tiles; ++local_n) {
			int logical_n = work.split_id + local_n * work.split_count;
			int fragment_base = logical_n * Traits::kNumAccumulators;
			auto gW0 = local_tile(
				mW,
				make_tile(
					Int<Traits::kAccumulatorTileN>{}, Int<Traits::kTileK>{}),
				make_coord(fragment_base + 0, _));
			auto gW1 = local_tile(
				mW,
				make_tile(
					Int<Traits::kAccumulatorTileN>{}, Int<Traits::kTileK>{}),
				make_coord(fragment_base + 1, _));
			auto tW0gW0 = cta_tma_w.partition_S(gW0);
			auto tW1gW1 = cta_tma_w.partition_S(gW1);

			// One mbarrier per stage covers X and both N160 panels.
			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.producer_acquire(state);
				auto* barrier = pipe.producer_get_barrier(state);
				int stage = state.index();
				copy(tma_load_x.with(*barrier, 0),
					tXgX(_, _, _, k_tile), tXsX(_, _, _, stage));
				copy(tma_load_w.with(*barrier, mcast_mask_w),
					tW0gW0(_, _, _, k_tile), tW0sW0(_, _, _, stage));
				copy(tma_load_w.with(*barrier, mcast_mask_w),
					tW1gW1(_, _, _, k_tile), tW1sW1(_, _, _, stage));
				++state;
			}
		}

		// producer_tail: keeps this CTA from retiring while a cluster peer
		// still holds operands multicast out of it.
		pipe.producer_tail(state);
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Consumer — warps 4..11.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ForwardGemmConsumerSm90 {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Epilogue = ForwardGemmEpilogueSm90<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	// _stage_slice: scatter one N80 slice of an FP32 accumulator into a padded
	// FP16 staging buffer, then publish it to both consumer warp groups.
	// coord_c is thr_mma.partition_C over the identity tensor, which reproduces
	// the reference's explicit row_base / col_base fragment arithmetic.
	template <bool ReturnEntropy, int SliceBase, class Accum, class CoordC>
	CUTE_DEVICE static void stage_slice(
			ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
			Accum const& accum,
			CoordC const& coord_c,
			int buffer) {
		CUTE_UNROLL
		for (int i = 0; i < size(accum); ++i) {
			int row = get<0>(coord_c(i));
			int col = get<1>(coord_c(i));
			if (col >= SliceBase &&
				col < SliceBase + Traits::kEpilogueSliceN) {
				smem.logit_row(buffer, row)[col - SliceBase] =
					static_cast<typename Traits::ElementLogit>(accum(i));
			}
		}
		cutlass::arch::fence_view_async_shared();
		asm volatile("fence.acq_rel.cta;" ::: "memory");
		forward_epilogue_barrier_sm90<Compute>();
	}

	// _fold_slice: fold this thread's N40 segment of a staged N80 slice into
	// the rolling statistics, then pick up the target logit if it lands here.
	// `valid_cols = local_vocab - slice_vocab_base` may be negative, matching
	// the reference; the chunk load keeps its mask-free fast path.
	template <bool ReturnEntropy>
	CUTE_DEVICE static void fold_slice(
			ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
			OnlineSoftmaxState& softmax,
			int buffer,
			int row,
			int segment,
			int slice_vocab_base,
			int valid_cols,
			int target_col,
			float inverse_temperature) {
		constexpr int kChunkN = Traits::kSoftmaxChunkN;
		constexpr int kSegmentN = Traits::kSegmentN;
		int segment_base = segment * kSegmentN;
		const typename Traits::ElementLogit* logits =
			smem.logit_row(buffer, row);

		CUTE_UNROLL
		for (int ci = 0; ci < Traits::kChunksPerSegment; ++ci) {
			int c0 = segment_base + ci * kChunkN;
			float values[kChunkN];
			float chunk_max = kForwardMaskLogit;

			if (c0 + kChunkN <= valid_cols) {
				CUTE_UNROLL
				for (int j = 0; j < kChunkN; ++j) {
					float value = static_cast<float>(logits[c0 + j]) *
						inverse_temperature;
					values[j] = value;
					chunk_max = fmaxf(chunk_max, value);
				}
			} else {
				CUTE_UNROLL
				for (int j = 0; j < kChunkN; ++j) {
					float value = static_cast<float>(logits[c0 + j]) *
						inverse_temperature;
					if (c0 + j >= valid_cols) value = kForwardMaskLogit;
					values[j] = value;
					chunk_max = fmaxf(chunk_max, value);
				}
			}

			Epilogue::template fold_chunk<ReturnEntropy, kChunkN>(
				softmax, values, chunk_max);
		}

		// The reference re-reads the target logit from SMEM rather than
		// indexing the chunk registers, which would demote them to local
		// memory. target_col is already a shard-local column index.
		int target_begin = slice_vocab_base + segment_base;
		int target_end = target_begin + kSegmentN;
		if (target_col >= target_begin && target_col < target_end) {
			softmax.target_logit +=
				static_cast<float>(logits[target_col - slice_vocab_base]) *
				inverse_temperature;
			softmax.has_target = 1;
		}
	}

	template <bool ReturnEntropy, class Pipeline>
	CUTE_DEVICE static void run(
			Pipeline& pipe,
			typename Traits::PipelineState& state,
			ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
			ForwardGemmParamsSm90<Compute> const& params,
			ForwardGemmPartialsSm90<Compute> const& partials,
			ForwardGemmWorkSm90 const& work,
			int num_split_tiles,
			int num_k_tiles) {
		typename Traits::TiledMma tiled_mma;
		int tid_in_mma =
			static_cast<int>(threadIdx.x) - Traits::kWarpGroupSize;  // 0..255
		auto thr_mma = tiled_mma.get_slice(tid_in_mma);

		auto sX = make_tensor(
			make_smem_ptr(smem.x_data()), typename Traits::SmemLayoutX{});
		auto sW0 = make_tensor(
			make_smem_ptr(smem.w0_data()), typename Traits::SmemLayoutW{});
		auto sW1 = make_tensor(
			make_smem_ptr(smem.w1_data()), typename Traits::SmemLayoutW{});

		auto tCsX = thr_mma.partition_A(sX);
		auto tCsW0 = thr_mma.partition_B(sW0);
		auto tCsW1 = thr_mma.partition_B(sW1);

		using AccumShape =
			Shape<Int<Traits::kTileM>, Int<Traits::kAccumulatorTileN>>;
		auto accum0 = partition_fragment_C(tiled_mma, AccumShape{});
		auto accum1 = partition_fragment_C(tiled_mma, AccumShape{});

		OnlineSoftmaxState softmax;
		softmax.max_value = kForwardNegInf;
		auto state_release = state;

		for (int local_n = 0; local_n < num_split_tiles; ++local_n) {
			int logical_n = work.split_id + local_n * work.split_count;
			clear(accum0);
			clear(accum1);

			// Two N160 fragments per K stage with the CuTe-DSL one-stage
			// release lag: stage 0 is not released until stage 1 retires.
			pipe.consumer_wait(state);
			warpgroup_fence_operand(accum0);
			warpgroup_fence_operand(accum1);
			warpgroup_arrive();
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW0(_, _, _, state.index()), accum0);
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW1(_, _, _, state.index()), accum1);
			warpgroup_commit_batch();
			++state;

			for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
				pipe.consumer_wait(state);
				warpgroup_fence_operand(accum0);
				warpgroup_fence_operand(accum1);
				warpgroup_arrive();
				gemm(tiled_mma, tCsX(_, _, _, state.index()),
					tCsW0(_, _, _, state.index()), accum0);
				gemm(tiled_mma, tCsX(_, _, _, state.index()),
					tCsW1(_, _, _, state.index()), accum1);
				warpgroup_commit_batch();

				warpgroup_wait<1>();
				warpgroup_fence_operand(accum0);
				warpgroup_fence_operand(accum1);
				pipe.consumer_release(state_release);
				++state_release;
				++state;
			}

			warpgroup_wait<0>();
			warpgroup_fence_operand(accum0);
			warpgroup_fence_operand(accum1);
			pipe.consumer_release(state_release);
			++state_release;

			auto coord_c =
				thr_mma.partition_C(make_identity_tensor(AccumShape{}));
			int tid_in_wg = tid_in_mma % Traits::kWarpGroupSize;
			int mma_wg = tid_in_mma / Traits::kWarpGroupSize;
			int epi_row = mma_wg * Traits::kRowsPerWarpGroup +
				(tid_in_wg % Traits::kRowsPerWarpGroup);
			int epi_segment = tid_in_wg / Traits::kRowsPerWarpGroup;
			int global_row = work.raw_pid_m * Traits::kTileM + epi_row;

			// Keep target addressing out of the live WGMMA region.
			int target_col = -1;
			if (work.store_enabled && global_row < params.tokens) {
				std::int64_t target_id = params.target[global_row];
				if (target_id != params.ignore_index) {
					std::int64_t local = target_id - params.vocab_start;
					if (local >= 0 &&
						local < static_cast<std::int64_t>(params.local_vocab)) {
						target_col = static_cast<int>(local);
					}
				}
			}

			constexpr int kSliceN = Traits::kEpilogueSliceN;
			constexpr int kAccumN = Traits::kAccumulatorTileN;
			int logical_base = logical_n * Traits::kLogicalTileN;
			int local_vocab = params.local_vocab;

			// Each N160 accumulator is folded as two N80 slices. Both buffers
			// are fully consumed before either is overwritten.
			stage_slice<ReturnEntropy, 0>(smem, accum0, coord_c, 0);
			stage_slice<ReturnEntropy, kSliceN>(smem, accum0, coord_c, 1);
			fold_slice<ReturnEntropy>(
				smem, softmax, 0, epi_row, epi_segment,
				logical_base,
				local_vocab - logical_base,
				target_col, params.inverse_temperature);
			forward_epilogue_barrier_sm90<Compute>();

			stage_slice<ReturnEntropy, 0>(smem, accum1, coord_c, 0);
			fold_slice<ReturnEntropy>(
				smem, softmax, 1, epi_row, epi_segment,
				logical_base + kSliceN,
				local_vocab - (logical_base + kSliceN),
				target_col, params.inverse_temperature);
			forward_epilogue_barrier_sm90<Compute>();

			stage_slice<ReturnEntropy, kSliceN>(smem, accum1, coord_c, 1);
			fold_slice<ReturnEntropy>(
				smem, softmax, 0, epi_row, epi_segment,
				logical_base + kAccumN,
				local_vocab - (logical_base + kAccumN),
				target_col, params.inverse_temperature);
			forward_epilogue_barrier_sm90<Compute>();

			fold_slice<ReturnEntropy>(
				smem, softmax, 1, epi_row, epi_segment,
				logical_base + kAccumN + kSliceN,
				local_vocab - (logical_base + kAccumN + kSliceN),
				target_col, params.inverse_temperature);
			forward_epilogue_barrier_sm90<Compute>();
		}

		// Merge the two N40 segment halves of every token row and emit this
		// CTA's split partials.
		int tid_in_wg = tid_in_mma % Traits::kWarpGroupSize;
		int mma_wg = tid_in_mma / Traits::kWarpGroupSize;
		int epi_row = mma_wg * Traits::kRowsPerWarpGroup +
			(tid_in_wg % Traits::kRowsPerWarpGroup);
		int epi_segment = tid_in_wg / Traits::kRowsPerWarpGroup;
		if (epi_segment == 1) {
			smem.segment_max[epi_row] = softmax.max_value;
			smem.segment_sum[epi_row] = softmax.exp_sum;
			smem.segment_target[epi_row] = softmax.target_logit;
			if constexpr (ReturnEntropy) {
				smem.segment_weighted[epi_row] = softmax.exp_weighted_sum;
			}
		}
		forward_epilogue_barrier_sm90<Compute>();

		if (epi_segment == 0 && work.store_enabled) {
			OnlineSoftmaxState other;
			other.max_value = smem.segment_max[epi_row];
			other.exp_sum = smem.segment_sum[epi_row];
			other.target_logit = smem.segment_target[epi_row];
			other.has_target = 0;
			if constexpr (ReturnEntropy) {
				other.exp_weighted_sum = smem.segment_weighted[epi_row];
			}
			OnlineSoftmaxState combined =
				Epilogue::template combine_scaled<ReturnEntropy>(
					softmax, other);
			Epilogue::template store_partial<ReturnEntropy>(
				combined,
				partials,
				work.output_work * Traits::kTileM + epi_row);
		}
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Fragment kernel
// ───────────────────────────────────────────────────────────────────────────

template <bool ReturnEntropy, int Compute, class TmaLoadX, class TmaLoadW>
__device__ __forceinline__ void forward_gemm_compute_sm90(
		ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
		const TmaLoadX& tma_load_x,
		const TmaLoadW& tma_load_w,
		const ForwardGemmParamsSm90<Compute>& params,
		const ForwardGemmPartialsSm90<Compute>& partials,
		const ForwardGemmSplitSm90<Compute>& split) {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Launch = ForwardGemmLaunchSm90<Compute>;
	using Smem = ForwardGemmSmemSm90<Compute, ReturnEntropy>;
	using PipelineTraits = typename Traits::PipelineTraits;
	using ClusterShape = typename Traits::ClusterShape;
	using Registers = sm90::WarpSpecializedRegistersSm90<
		Config::kProducerRegisters, Config::kMmaRegisters, Compute>;

	int warp_group = cutlass::canonical_warp_group_idx();
	int warp = cutlass::canonical_warp_idx_sync();

	// mbarrier init (stores are warp-0 only, the fence is everyone) followed by
	// the cluster-wide handshake. No pipeline object exists yet, so nothing
	// role-specific is live across this point.
	PipelineTraits::init_barriers(
		smem.pipeline,
		Traits::kTmaTransBytes,
		Traits::kConsumerThreads,
		ClusterShape{});
	sm90::cluster_pipeline_init_sm90<Compute>();

	// Each role builds its own pipeline object, work assignment and tile
	// counts inside a tight scope. The work/tile integers are a handful of
	// scalar ops, so recomputing them per role is cheaper than keeping them
	// live across the branch.
	if (warp_group == 0) {
		Registers::producer();
		if (warp == 0) {
			auto pipe = PipelineTraits::make_producer(
				smem.pipeline,
				Traits::kTmaTransBytes,
				Traits::kConsumerThreads,
				threadIdx.x == 0,
				ClusterShape{});
			typename Traits::PipelineState producer_state =
				PipelineTraits::producer_start_state();

			ForwardGemmWorkSm90 work = forward_gemm_assign_work_sm90<Compute>(
				split,
				static_cast<int>(blockIdx.z),
				static_cast<int>(blockIdx.x));
			int num_k_tiles = Launch::num_k_tiles(params.hidden);
			int num_split_tiles = ceil_div(
				split.num_logical_n_tiles - work.split_id, work.split_count);

			ForwardGemmProducerSm90<Compute>::template run<ReturnEntropy>(
				pipe,
				producer_state,
				smem,
				tma_load_x,
				tma_load_w,
				params,
				work,
				num_split_tiles,
				num_k_tiles);
		}
	} else {
		Registers::consumer();
		auto pipe = PipelineTraits::make_consumer(
			smem.pipeline,
			Traits::kTmaTransBytes,
			Traits::kConsumerThreads,
			ClusterShape{});
		typename Traits::PipelineState consumer_state;

		ForwardGemmWorkSm90 work = forward_gemm_assign_work_sm90<Compute>(
			split,
			static_cast<int>(blockIdx.z),
			static_cast<int>(blockIdx.x));
		int num_k_tiles = Launch::num_k_tiles(params.hidden);
		int num_split_tiles = ceil_div(
			split.num_logical_n_tiles - work.split_id, work.split_count);

		ForwardGemmConsumerSm90<Compute>::template run<ReturnEntropy>(
			pipe,
			consumer_state,
			smem,
			params,
			partials,
			work,
			num_split_tiles,
			num_k_tiles);
	}

}

template <bool ReturnEntropy, int Compute, class TmaLoadX, class TmaLoadW>
__global__ __launch_bounds__(ForwardGemmTraitsSm90<Compute>::kNumThreads, 1)
void forward_gemm_kernel_sm90(
		__grid_constant__ const TmaLoadX tma_load_x,
		__grid_constant__ const TmaLoadW tma_load_w,
		__grid_constant__ const ForwardGemmParamsSm90<Compute> params,
		__grid_constant__ const ForwardGemmPartialsSm90<Compute> partials,
		__grid_constant__ const ForwardGemmSplitSm90<Compute> split) {
	using Smem = ForwardGemmSmemSm90<Compute, ReturnEntropy>;
	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	forward_gemm_compute_sm90<ReturnEntropy, Compute>(
		smem, tma_load_x, tma_load_w, params, partials, split);
	sm90::cluster_exit_sm90<Compute>();
}

template <
	liger_cute::detail::LocalReduceBackend Backend,
	liger_cute::detail::ReduceOp Op,
	class Mapping>
__device__ __forceinline__ void forward_local_reduce_warp(
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		std::size_t data_offset,
		std::size_t count,
		int m_tile,
		int phase) {
	unsigned int lane = liger_cute::detail::nvls_lane_id();
	if (mapping.size == 1) {
		for (std::size_t index = lane; index < count; index += 32) {
			comm.reduced[data_offset + index] =
				comm.partial[data_offset + index];
		}
		__syncwarp();
		return;
	}

	std::size_t ready_offset =
		static_cast<std::size_t>(m_tile * 4 + phase * 2) *
		static_cast<std::size_t>(mapping.size);
	std::size_t complete_offset =
		ready_offset + static_cast<std::size_t>(mapping.size);
	std::uint64_t epoch =
		dx_epoch_base(comm) | 0x10000000u |
		(static_cast<std::uint64_t>(m_tile) << 4) |
		static_cast<std::uint64_t>(phase + 1);

	if constexpr (
		Backend == liger_cute::detail::LocalReduceBackend::kNvls) {
		liger_cute::detail::LocalReduceContext<Backend, float> context{
			comm.partial + data_offset,
			comm.reduced + data_offset,
			comm.sync + ready_offset,
			mapping.multicast_sync + ready_offset,
			comm.sync + complete_offset,
			mapping.multicast_sync + complete_offset,
			mapping.rank,
			mapping.size};
		liger_cute::detail::local_all_reduce<Backend, Op>(
			context,
			mapping.multicast_reduced + data_offset,
			mapping.multicast_partial + data_offset,
			count,
			epoch);
	} else {
		liger_cute::detail::LocalReduceContext<Backend, float> context{
			mapping.peer_partial,
			data_offset,
			comm.sync + ready_offset,
			mapping.peer_sync,
			ready_offset,
			comm.sync + complete_offset,
			complete_offset,
			mapping.rank,
			mapping.size};
		liger_cute::detail::local_all_reduce<Backend, Op>(
			context,
			comm.reduced + data_offset,
			comm.partial + data_offset,
			count,
			epoch);
	}
}

template <
	bool ReturnEntropy,
	int Compute,
	bool RequiresRemote,
	liger_cute::detail::LocalReduceBackend Backend,
	class Mapping>
__device__ __forceinline__ void forward_finalize_splits_and_reduce_sm90(
		ForwardGemmSmemSm90<Compute, ReturnEntropy>& smem,
		const ForwardGemmParamsSm90<Compute>& params,
		const ForwardGemmPartialsSm90<Compute>& partials,
		const ForwardGemmSplitSm90<Compute>& split,
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		int* split_ready,
		float* global_max,
		float* reduced,
		float* remote_source) {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Epilogue = ForwardGemmEpilogueSm90<Compute>;

	ForwardGemmWorkSm90 work = forward_gemm_assign_work_sm90<Compute>(
		split,
		static_cast<int>(blockIdx.z),
		static_cast<int>(blockIdx.x));
	int warp_group = cutlass::canonical_warp_group_idx();
	int consumer_thread =
		static_cast<int>(threadIdx.x) - Traits::kWarpGroupSize;
	bool wrote_partial =
		warp_group != 0 &&
		consumer_thread % Traits::kWarpGroupSize <
			Traits::kRowsPerWarpGroup;
	if (wrote_partial && work.store_enabled) {
		__threadfence();
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		smem.finalizer = 0;
		if (work.store_enabled) {
			int completed = atomicAdd(
				split_ready + work.raw_pid_m, 1) + 1;
			smem.finalizer =
				completed == split.split_count_for_pair(
					work.raw_pid_m / Config::kClusterM);
		}
	}
	__syncthreads();

	if (smem.finalizer != 0) {
		int tid = static_cast<int>(threadIdx.x);
		if (tid < Config::kTileM) {
			int row = work.raw_pid_m * Config::kTileM + tid;
			int split_count = split.split_count_for_pair(
				work.raw_pid_m / Config::kClusterM);
			float row_max = kForwardNegInf;
			float row_target = 0.0f;
			for (int split_id = 0; split_id < split_count; ++split_id) {
				int index =
					(work.raw_pid_m * split.split_n + split_id) *
						Config::kTileM +
					tid;
				row_max = fmaxf(row_max, partials.partial_max[index]);
				row_target += partials.partial_target[index];
			}

			float row_sum = 0.0f;
			float row_weighted = 0.0f;
			for (int split_id = 0; split_id < split_count; ++split_id) {
				int index =
					(work.raw_pid_m * split.split_n + split_id) *
						Config::kTileM +
					tid;
				float split_scale = forward_exp2_sm90(
					(partials.partial_max[index] - row_max) *
						kForwardLog2E);
				row_sum += partials.partial_sum[index] * split_scale;
				if constexpr (ReturnEntropy) {
					row_weighted +=
						partials.partial_weighted[index] * split_scale;
				}
			}

			OnlineSoftmaxState state;
			state.max_value = row_max;
			state.exp_sum = row_sum;
			state.target_logit = row_target;
			if constexpr (ReturnEntropy) {
				state.exp_weighted_sum = row_weighted;
			}
			state.has_target = 0;
			if (row < params.tokens) {
				std::int64_t target_id = params.target[row];
				std::int64_t local = target_id - params.vocab_start;
				state.has_target =
					target_id != params.ignore_index &&
					local >= 0 &&
					local < static_cast<std::int64_t>(
						params.local_vocab);
				Epilogue::template store_row<ReturnEntropy>(
					state, params.output, row);
			}
		}
		__syncthreads();

		int warp = cutlass::canonical_warp_idx_sync();
		int lane = static_cast<int>(threadIdx.x) % Traits::kWarpSize;
		if (warp == 1) {
			constexpr std::size_t kRows = Config::kTileM;
			constexpr std::size_t kStride =
				kForwardReducedFields * kRows;
			std::size_t base =
				static_cast<std::size_t>(work.raw_pid_m) * kStride;
			for (int row_in_tile = lane;
					row_in_tile < Config::kTileM;
					row_in_tile += Traits::kWarpSize) {
				int row =
					work.raw_pid_m * Config::kTileM + row_in_tile;
				comm.partial[base + row_in_tile] =
					row < params.tokens
					? params.output.local_max[row]
					: kForwardNegInf;
			}
			__syncwarp();
			forward_local_reduce_warp<
				Backend,
				liger_cute::detail::ReduceOp::kMax>(
					comm,
					mapping,
					base,
					kRows,
					work.raw_pid_m,
					0);

			constexpr std::size_t kFields =
				ReturnEntropy ? kForwardReducedFields : 2;
			constexpr std::size_t kStateFields = 1 + kFields;
			int padded_tokens = ceil_div(
				params.tokens, mapping.size) * mapping.size;
			int rows_per_rank = padded_tokens / mapping.size;
			int owned_row_begin =
				mapping.rank * rows_per_rank;
			for (int row_in_tile = lane;
					row_in_tile < Config::kTileM;
					row_in_tile += Traits::kWarpSize) {
				int row =
					work.raw_pid_m * Config::kTileM + row_in_tile;
				float local_sum = row < params.tokens
					? params.output.local_sum[row]
					: 0.0f;
				float local_max = row < params.tokens
					? params.output.local_max[row]
					: kForwardNegInf;
				float node_max = comm.reduced[base + row_in_tile];
				if (row < params.tokens) {
					params.output.local_max[row] = node_max;
					global_max[row] = node_max;
				}
				float correction = local_sum == 0.0f
					? 0.0f
					: forward_exp2_sm90(
						(local_max - node_max) * kForwardLog2E);
				float local_target = row < params.tokens
					? params.output.local_target[row]
					: 0.0f;
				float local_weighted = 0.0f;
				if constexpr (ReturnEntropy) {
					local_weighted = row < params.tokens
						? params.output.local_weighted_sum[row]
						: 0.0f;
				}
				comm.partial[
					base +
					kForwardReducedSumField * kRows +
					row_in_tile] =
					local_sum * correction;
				comm.partial[
					base +
					kForwardReducedTargetField * kRows +
					row_in_tile] =
					local_target;
				if constexpr (ReturnEntropy) {
					comm.partial[
						base +
						kForwardReducedWeightedField * kRows +
						row_in_tile] =
						local_weighted * correction;
				}
				if constexpr (RequiresRemote) {
					// Every local GPU has the complete node state. Pack only
					// this rank's contiguous token shard for the IBRC pair.
					if (row >= owned_row_begin &&
						row < owned_row_begin + rows_per_rank) {
						int local_row = row - owned_row_begin;
						remote_source[
							static_cast<std::size_t>(local_row) *
								kStateFields] =
							node_max;
					}
				}
			}
			__syncwarp();
			liger_cute::detail::publish_local_reduce_source();
			forward_local_reduce_warp<
				Backend,
				liger_cute::detail::ReduceOp::kSum>(
					comm,
					mapping,
					base,
					kFields * kRows,
					work.raw_pid_m,
					1);
			for (int row_in_tile = lane;
					row_in_tile < Config::kTileM;
					row_in_tile += Traits::kWarpSize) {
				int row =
					work.raw_pid_m * Config::kTileM + row_in_tile;
				if constexpr (RequiresRemote) {
					if (row >= owned_row_begin &&
						row < owned_row_begin + rows_per_rank) {
						int local_row = row - owned_row_begin;
						for (int field = 0;
								field < static_cast<int>(kFields);
								++field) {
							remote_source[
								static_cast<std::size_t>(local_row) *
									kStateFields +
								1 +
								field] =
								comm.reduced[
									base +
									static_cast<std::size_t>(field) *
										kRows +
									row_in_tile];
						}
					}
				} else if (row < params.tokens) {
					CUTE_UNROLL
					for (int field = 0;
							field < static_cast<int>(kFields);
							++field) {
							reduced[field * params.tokens + row] =
								comm.reduced[
									base +
									static_cast<std::size_t>(field) *
										kRows +
										row_in_tile];
					}
				}
			}
		}
		__syncthreads();
	}
}

template <
	bool ReturnEntropy,
	int Compute,
	bool RequiresRemote,
	liger_cute::detail::LocalReduceBackend Backend,
	class TmaLoadX,
	class TmaLoadW,
	class Mapping>
__global__ __launch_bounds__(ForwardGemmTraitsSm90<Compute>::kNumThreads, 1)
void forward_gemm_tp_kernel_sm90(
		__grid_constant__ const TmaLoadX tma_load_x,
		__grid_constant__ const TmaLoadW tma_load_w,
		__grid_constant__ const ForwardGemmParamsSm90<Compute> params,
		__grid_constant__ const ForwardGemmPartialsSm90<Compute> partials,
		__grid_constant__ const ForwardGemmSplitSm90<Compute> split,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const Mapping mapping,
		int* split_ready,
		float* global_max,
		float* reduced,
		float* remote_source) {
	using Smem = ForwardGemmSmemSm90<Compute, ReturnEntropy>;
	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	forward_gemm_compute_sm90<ReturnEntropy, Compute>(
		smem, tma_load_x, tma_load_w, params, partials, split);
	forward_finalize_splits_and_reduce_sm90<
		ReturnEntropy,
		Compute,
		RequiresRemote,
		Backend>(
			smem,
			params,
			partials,
			split,
			comm,
			mapping,
			split_ready,
			global_max,
			reduced,
			remote_source);
	sm90::cluster_exit_sm90<Compute>();
}

// ───────────────────────────────────────────────────────────────────────────
// Test-only split reducer for the GEMM-only self-test. Production performs this
// merge through the last-arrival epilogue above.
// ───────────────────────────────────────────────────────────────────────────

template <bool ReturnEntropy, int Compute>
__global__ __launch_bounds__(ForwardGemmConfigSm90<Compute>::kTileM, 1)
void forward_split_reduce_kernel_sm90(
		__grid_constant__ const ForwardGemmParamsSm90<Compute> params,
		__grid_constant__ const ForwardGemmPartialsSm90<Compute> partials,
		__grid_constant__ const ForwardGemmSplitSm90<Compute> split) {
	using Config = ForwardGemmConfigSm90<Compute>;
	using Epilogue = ForwardGemmEpilogueSm90<Compute>;

	int tid = static_cast<int>(threadIdx.x);
	int pid_m = static_cast<int>(blockIdx.x);
	int row = pid_m * Config::kTileM + tid;
	if (row >= params.tokens) return;

	int split_count = split.split_count_for_pair(pid_m / Config::kClusterM);

	float row_max = kForwardNegInf;
	float row_target = 0.0f;
	for (int split_id = 0; split_id < split_count; ++split_id) {
		int index = (pid_m * split.split_n + split_id) * Config::kTileM + tid;
		row_max = fmaxf(row_max, partials.partial_max[index]);
		row_target += partials.partial_target[index];
	}

	float row_sum = 0.0f;
	float row_weighted = 0.0f;
	for (int split_id = 0; split_id < split_count; ++split_id) {
		int index = (pid_m * split.split_n + split_id) * Config::kTileM + tid;
		float split_scale = forward_exp2_sm90(
			(partials.partial_max[index] - row_max) * kForwardLog2E);
		row_sum += partials.partial_sum[index] * split_scale;
		if constexpr (ReturnEntropy) {
			row_weighted += partials.partial_weighted[index] * split_scale;
		}
	}

	OnlineSoftmaxState state;
	state.max_value = row_max;
	state.exp_sum = row_sum;
	state.target_logit = row_target;
	if constexpr (ReturnEntropy) {
		state.exp_weighted_sum = row_weighted;
	}
	// has_target is recomputed rather than carried: both segment threads see
	// the same shard-local column, so it is a pure function of the target id.
	std::int64_t target_id = params.target[row];
	std::int64_t local = target_id - params.vocab_start;
	state.has_target = (target_id != params.ignore_index && local >= 0 &&
		local < static_cast<std::int64_t>(params.local_vocab)) ? 1 : 0;

	Epilogue::template store_row<ReturnEntropy>(state, params.output, row);
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
