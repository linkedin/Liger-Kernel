#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy — forward GEMM contract.
//
// Structural port of
//   src/liger_kernel/ops/cutedsl/ops/_fused_scaled_cross_entropy_forward_fragment_sm90.py
// (the CuTe-DSL "two-fragment" fragment kernel). Constant names below map 1:1
// onto the reference module constants:
//
//   THREADS_PER_CTA 384   TILE_M 128        TILE_K 64        STAGES 3
//   CLUSTER_M 2           ACCUMULATOR_N 160 NUM_ACCUMULATORS 2
//   LOGICAL_N 320         EPILOGUE_SLICE_N 80                CHUNK_N 40
//   LOGIT_BUFFERS 2       LOGIT_PAD 8       LOGIT_STRIDE 88
//   EPILOGUE_BARRIER_ID 4 PRODUCER_REGISTERS 24              MMA_REGISTERS 240
//
// This header is deliberately CuTe-free: it carries the launch geometry, the
// vocabulary-split scheduler, the warp plan, the online-softmax epilogue
// contract and the host launcher declaration so torch-free consumers (and the
// existing C++ unit tests) can include the umbrella header without CUTLASS.
//
// The executable producer / consumer / mainloop is in
// forward_gemm_mainloop_sm90.cuh and is instantiated by
// fused_linear_scaled_cross_entropy_forward.cu.
//
// Tensor-parallel deltas from the single-GPU reference — everything else is a
// literal port:
//   * W is the *local* vocabulary shard W_local[V_local, H]; local column c is
//     global vocabulary index vocab_start + c.
//   * `target` holds *global* int64 indices, mapped to the shard on the fly.
//   * The reference's finalizer emits NLL/LSE/entropy directly. Here the split
//     reducer stops at per-token *local* (max, sum, lse, target[, weighted])
//     statistics; fused_linear_scaled_cross_entropy_forward corrects and reduces them
//     across the NVSHMEM team.
//   * The reference host-pads H to a multiple of TILE_K. TMA out-of-bounds
//     zero fill covers the K tail here instead, so no padded copy is made.
//     H only has to keep the BF16 row stride 16 B aligned (H % 8 == 0).
//
// The [M, V] logit matrix is never materialised in HBM.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "config.cuh"
#include "online_softmax.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

struct ForwardLocalStatsBuffers {
	float* local_max;
	float* local_sum;
	float* local_target;
	float* local_weighted_sum;
};

inline constexpr int kForwardReducedSumField = 0;
inline constexpr int kForwardReducedTargetField = 1;
inline constexpr int kForwardReducedWeightedField = 2;
inline constexpr int kForwardReducedFields = 3;

// _fused_scaled_cross_entropy_utils_sm90.py constants.
inline constexpr float kForwardLog2E = 1.4426950408889634f;
inline constexpr float kForwardLn2 = 0.6931471805599453f;
// Rolling-max seed. Strictly greater than kForwardMaskLogit so a fold made
// entirely of masked vocabulary columns leaves the running state untouched.
inline constexpr float kForwardNegInf = -1.0e38f;
inline constexpr float kForwardMaskLogit = -3.0e38f;
inline constexpr int kHopperMaxSmemBytes = 227 * 1024;

// exp2 with the reference's fastmath=True lowering (llvm.nvvm.ex2.approx.ftz).
__host__ __device__ inline float forward_exp2_sm90(float value) {
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return result;
#else
	return exp2f(value);
#endif
}

enum class ForwardWarpRole : std::uint8_t {
	kProducer,
	kIdle,
	kConsumer,
	kInactive,
};

// PRODUCER_WARP_GROUP = 0 owns warp 0 (TMA). Warps 1..3 sit in the producer
// warp group but are idle in forward — they carry the dX comms in backward.
// Warps 4..11 are the two WGMMA consumer warp groups.
__host__ __device__ constexpr ForwardWarpRole forward_warp_role(int warp_id) {
	return warp_id == 0
		? ForwardWarpRole::kProducer
		: (warp_id >= 1 && warp_id <= 3
			? ForwardWarpRole::kIdle
			: (warp_id >= 4 && warp_id < kNumWarps
				? ForwardWarpRole::kConsumer
				: ForwardWarpRole::kInactive));
}

template <int Compute = 90>
struct ForwardGemmConfigSm90 {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	static constexpr int kCompute = Compute;
	static constexpr int kTileM = 128;
	static constexpr int kTileK = 64;
	static constexpr int kLogicalTileN = 320;
	static constexpr int kAccumulatorTileN = 160;
	static constexpr int kEpilogueSliceN = 80;
	static constexpr int kSoftmaxChunkN = 40;
	static constexpr int kMainloopStages = 3;
	static constexpr int kClusterM = 2;
	static constexpr int kNumThreads =
		fused_scaled_linear_cross_entropy::kNumThreads;

	// Two N160 accumulators per consumer warp group form the logical N320.
	static constexpr int kNumAccumulators = kLogicalTileN / kAccumulatorTileN;
	static constexpr int kNumMmaWarpGroups = 2;
	static constexpr int kWarpsPerWarpGroup = 4;
	static constexpr int kWarpGroupSize = 128;
	static constexpr int kConsumerThreads =
		kNumMmaWarpGroups * kWarpGroupSize;
	static constexpr int kMmaAtomTileM = kTileM / kNumMmaWarpGroups;

	// Two ping-pong N80 staging buffers, padded to break SMEM bank conflicts.
	static constexpr int kLogitBuffers = 2;
	static constexpr int kLogitPad = 8;
	static constexpr int kLogitStride = kEpilogueSliceN + kLogitPad;

	// Two threads cooperate on each token row; each owns one N40 chunk of the
	// N80 slice currently being folded.
	static constexpr int kSegmentN = kEpilogueSliceN / 2;
	static constexpr int kChunksPerSegment = kSegmentN / kSoftmaxChunkN;
	static constexpr int kSegmentsPerRow = kEpilogueSliceN / kSegmentN;
	static constexpr int kRowsPerWarpGroup = kTileM / kNumMmaWarpGroups;

	static constexpr int kProducerRegisters = 24;
	static constexpr int kMmaRegisters = 240;
	static constexpr int kUsableRegisterBudget = 64512;

	// Hardware named barrier shared by the two consumer warp groups.
	static constexpr int kEpilogueBarrierId = 4;

	// _resolve_split's ceiling on the vocabulary split (config.max_split_n).
	static constexpr int kDefaultMaxSplitN = 9;

	static_assert(kLogicalTileN == 2 * kAccumulatorTileN);
	static_assert(kAccumulatorTileN == 2 * kEpilogueSliceN);
	static_assert(kEpilogueSliceN == 2 * kSoftmaxChunkN);
	static_assert(kSegmentN == kSoftmaxChunkN);
	static_assert(kNumThreads == (kNumMmaWarpGroups + 1) * kWarpGroupSize);
	static_assert(forward_warp_role(0) == ForwardWarpRole::kProducer);
	static_assert(forward_warp_role(3) == ForwardWarpRole::kIdle);
	static_assert(forward_warp_role(4) == ForwardWarpRole::kConsumer);

	__host__ __device__ static constexpr int register_total() {
		return kWarpGroupSize * kProducerRegisters +
			kNumMmaWarpGroups * kWarpGroupSize * kMmaRegisters;
	}
	static_assert(register_total() <= kUsableRegisterBudget,
		"SM90 forward register budget exceeds the usable per-CTA budget");
};

// ───────────────────────────────────────────────────────────────────────────
// Vocabulary-split scheduler — port of _resolve_split plus the kernel-side
// cluster_work decomposition.
// ───────────────────────────────────────────────────────────────────────────

// ScaledCEForwardFragmentConfig's tuning knobs. Zero means "auto".
template <int Compute = 90>
struct ForwardGemmTuningSm90 {
	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	int split_n = 0;
	int base_split_n = 0;
	int extra_m_pairs = 0;
	int target_cluster_pairs = 0;
	int max_split_n = Config::kDefaultMaxSplitN;
};

template <int Compute = 90>
struct ForwardGemmSplitSm90 {
	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	int split_n = 1;
	int base_split_n = 1;
	int extra_m_pairs = 0;
	int num_m_tiles = 0;
	int num_m_pairs = 0;
	int num_logical_n_tiles = 0;
	int num_cluster_pairs = 0;

	// Number of vocabulary splits that actually cover `m_pair`. M pairs below
	// extra_m_pairs get the extra split; the rest get base_split_n.
	__host__ __device__ int split_count_for_pair(int m_pair) const {
		if (extra_m_pairs == 0) return base_split_n;
		return m_pair < extra_m_pairs ? split_n : base_split_n;
	}
};

// FP32 (max, sum, target[, weighted]) split partials, laid out
// [num_m_tiles * split_n, TILE_M] exactly like the reference's
// partial_* tensors.
template <int Compute = 90>
struct ForwardGemmPartialsSm90 {
	static constexpr int kCompute = Compute;

	float* partial_max = nullptr;
	float* partial_sum = nullptr;
	float* partial_target = nullptr;
	float* partial_weighted = nullptr;
};

// Problem + output description. The vocabulary shard is contiguous: local
// column c is global vocabulary index `vocab_start + c`, and targets arrive as
// *global* int64 indices.
template <int Compute = 90>
struct ForwardGemmParamsSm90 {
	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	const void* x = nullptr;               // BF16 [tokens, hidden], row major
	const void* weight = nullptr;          // BF16 [local_vocab, hidden], row major
	const std::int64_t* target = nullptr;  // int64 [tokens], global indices
	ForwardLocalStatsBuffers output = {};  // FP32 per-token local statistics

	// Split partials scratch; size it with
	// ForwardGemmLaunchSm90<Compute>::workspace_bytes(...).
	void* workspace = nullptr;
	std::size_t workspace_bytes = 0;

	int tokens = 0;
	int hidden = 0;
	int local_vocab = 0;
	std::int64_t vocab_start = 0;
	std::int64_t ignore_index = -100;
	float inverse_temperature = 1.0f;

	ForwardGemmTuningSm90<Compute> tuning = {};
};

// ───────────────────────────────────────────────────────────────────────────
// Online-softmax epilogue contract
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ForwardGemmEpilogueSm90 {
	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	// Scalar reference fold: one online-softmax update per logit. Kept as the
	// readable contract; the device path uses the chunked form below.
	template <bool ReturnEntropy, int FragmentSize>
	__host__ __device__ static void fold_fragment(
			OnlineSoftmaxState& state,
			const float (&scaled_logits)[FragmentSize],
			int fragment_vocab_start,
			int local_vocab_size,
			int local_target) {
		#pragma unroll
		for (int i = 0; i < FragmentSize; ++i) {
			int local_vocab_index = fragment_vocab_start + i;
			state.template fold<ReturnEntropy>(
				scaled_logits[i],
				local_vocab_index < local_vocab_size,
				local_vocab_index == local_target);
		}
	}

	// _fold_slice's rolling update, verbatim: one rescale of the running
	// (sum[, weighted]) per N40 chunk. `values` is the chunk already masked
	// and scaled by the caller and `chunk_max` its maximum, matching the
	// reference's two-branch load.
	template <bool ReturnEntropy, int ChunkN>
	__host__ __device__ static void fold_chunk(
			OnlineSoftmaxState& state,
			const float (&values)[ChunkN],
			float chunk_max) {
		float next_max = fmaxf(state.max_value, chunk_max);
		float chunk_sum = 0.0f;

		if constexpr (ReturnEntropy) {
			float chunk_weighted = 0.0f;
			#pragma unroll
			for (int i = 0; i < ChunkN; ++i) {
				float weight =
					forward_exp2_sm90((values[i] - next_max) * kForwardLog2E);
				chunk_sum += weight;
				chunk_weighted += weight * values[i];
			}
			float previous_scale = forward_exp2_sm90(
				(state.max_value - next_max) * kForwardLog2E);
			state.exp_sum = state.exp_sum * previous_scale + chunk_sum;
			state.exp_weighted_sum =
				state.exp_weighted_sum * previous_scale + chunk_weighted;
		} else {
			#pragma unroll
			for (int i = 0; i < ChunkN; ++i) {
				chunk_sum +=
					forward_exp2_sm90((values[i] - next_max) * kForwardLog2E);
			}
			state.exp_sum = state.exp_sum * forward_exp2_sm90(
				(state.max_value - next_max) * kForwardLog2E) + chunk_sum;
		}

		state.max_value = next_max;
	}

	// Branch-free merge of two rolling states, matching the reference's
	// cross-segment combine and its split finalizer. Deliberately not
	// merge_online_softmax(): that shortcuts on exp_sum == 0 and would put
	// data-dependent branches in the epilogue.
	template <bool ReturnEntropy>
	__host__ __device__ static OnlineSoftmaxState combine_scaled(
			const OnlineSoftmaxState& lhs, const OnlineSoftmaxState& rhs) {
		OnlineSoftmaxState result;
		result.max_value = fmaxf(lhs.max_value, rhs.max_value);
		float lhs_scale = forward_exp2_sm90(
			(lhs.max_value - result.max_value) * kForwardLog2E);
		float rhs_scale = forward_exp2_sm90(
			(rhs.max_value - result.max_value) * kForwardLog2E);
		result.exp_sum = lhs.exp_sum * lhs_scale + rhs.exp_sum * rhs_scale;
		if constexpr (ReturnEntropy) {
			result.exp_weighted_sum =
				lhs.exp_weighted_sum * lhs_scale +
				rhs.exp_weighted_sum * rhs_scale;
		}
		// The reference accumulates the target logit additively: at most one
		// contributor is non-zero because the shard is partitioned.
		result.target_logit = lhs.target_logit + rhs.target_logit;
		result.has_target = lhs.has_target || rhs.has_target;
		return result;
	}

	template <bool ReturnEntropy>
	__host__ __device__ static void store_partial(
			const OnlineSoftmaxState& state,
			const ForwardGemmPartialsSm90<Compute>& partials,
			int index) {
		partials.partial_max[index] = state.max_value;
		partials.partial_sum[index] = state.exp_sum;
		partials.partial_target[index] = state.target_logit;
		if constexpr (ReturnEntropy) {
			partials.partial_weighted[index] = state.exp_weighted_sum;
		}
	}

	template <bool ReturnEntropy>
	__host__ __device__ static void store_row(
			const OnlineSoftmaxState& state,
			const ForwardLocalStatsBuffers& output,
			int row) {
		output.local_max[row] = state.max_value;
		output.local_sum[row] = state.exp_sum;
		output.local_target[row] = state.has_target ? state.target_logit : 0.0f;
		if constexpr (ReturnEntropy) {
			output.local_weighted_sum[row] = state.exp_weighted_sum;
		}
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Host-side launch geometry. CuTe-free so the launcher declaration, the split
// resolution and the shared-memory budget stay visible without CUTLASS.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ForwardGemmLaunchSm90 {
	using Config = ForwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	__host__ __device__ static constexpr int num_m_tiles(int tokens) {
		return ceil_div(tokens, Config::kTileM);
	}

	__host__ __device__ static constexpr int num_m_pairs(int tokens) {
		return ceil_div(num_m_tiles(tokens), Config::kClusterM);
	}

	__host__ __device__ static constexpr int num_k_tiles(int hidden) {
		return ceil_div(hidden, Config::kTileK);
	}

	__host__ __device__ static constexpr int num_logical_n_tiles(
			int local_vocab) {
		return ceil_div(local_vocab, Config::kLogicalTileN);
	}

	// _resolve_split. `max_active_cluster_pairs` is the device's occupancy for
	// a CLUSTER_M cluster (cutlass HardwareInfo.get_max_active_clusters).
	__host__ static ForwardGemmSplitSm90<Compute> resolve_split(
			const ForwardGemmTuningSm90<Compute>& tuning,
			int tokens,
			int local_vocab,
			int max_active_cluster_pairs) {
		ForwardGemmSplitSm90<Compute> split;
		split.num_m_tiles = num_m_tiles(tokens);
		split.num_m_pairs = num_m_pairs(tokens);
		split.num_logical_n_tiles = num_logical_n_tiles(local_vocab);

		if (tuning.base_split_n != 0) {
			split.split_n = tuning.split_n;
			split.base_split_n = tuning.base_split_n;
			split.extra_m_pairs = tuning.extra_m_pairs;
		} else if (tuning.split_n != 0) {
			split.split_n = tuning.split_n;
			split.base_split_n = tuning.split_n;
			split.extra_m_pairs = 0;
		} else {
			int max_split_n = tuning.max_split_n > 0
				? tuning.max_split_n
				: Config::kDefaultMaxSplitN;
			int target_cluster_pairs = tuning.target_cluster_pairs > 0
				? tuning.target_cluster_pairs
				: (max_active_cluster_pairs > 0 ? max_active_cluster_pairs : 1);
			int per_pair_cap = max_split_n < split.num_logical_n_tiles
				? max_split_n
				: split.num_logical_n_tiles;
			int max_cluster_pairs = split.num_m_pairs * per_pair_cap;
			int wanted = target_cluster_pairs < max_cluster_pairs
				? target_cluster_pairs
				: max_cluster_pairs;
			int cluster_pairs =
				split.num_m_pairs > wanted ? split.num_m_pairs : wanted;
			split.base_split_n = cluster_pairs / split.num_m_pairs;
			split.extra_m_pairs = cluster_pairs % split.num_m_pairs;
			split.split_n =
				split.base_split_n + (split.extra_m_pairs != 0 ? 1 : 0);
		}

		split.num_cluster_pairs =
			split.num_m_pairs * split.base_split_n + split.extra_m_pairs;
		return split;
	}

	// Upper bound on the split-partial scratch, independent of the runtime
	// split resolution so callers can size it ahead of the launch.
	__host__ __device__ static std::size_t workspace_bytes(
			int tokens,
			int local_vocab,
			bool return_entropy,
			int max_split_n = Config::kDefaultMaxSplitN) {
		int logical_n = num_logical_n_tiles(local_vocab);
		if (logical_n < 1) logical_n = 1;
		int cap = max_split_n < logical_n ? max_split_n : logical_n;
		if (cap < 1) cap = 1;
		std::size_t rows = static_cast<std::size_t>(num_m_tiles(tokens)) *
			static_cast<std::size_t>(cap) *
			static_cast<std::size_t>(Config::kTileM);
		return rows * sizeof(float) * (return_entropy ? 4u : 3u);
	}

	// Mirrors ForwardGemmSmemSm90<Compute, ReturnEntropy> — duplicated so
	// callers can validate the dynamic allocation without CUTLASS.
	__host__ __device__ static constexpr int smem_bytes(bool return_entropy) {
		return
			// X + two N160 weight panels, BF16, kMainloopStages deep.
			(Config::kTileM + Config::kLogicalTileN) * Config::kTileK * 2 *
				Config::kMainloopStages +
			// Two padded FP16 N80 staging buffers.
			Config::kLogitBuffers * Config::kTileM * Config::kLogitStride * 2 +
			// Cross-segment (max, sum, target) exchange.
			3 * Config::kTileM * 4 +
			// Optional weighted-entropy moment exchange.
			(return_entropy ? Config::kTileM : 1) * 4 +
			// TMA mbarrier pairs.
			16 * Config::kMainloopStages +
			// Struct alignment padding (four 1024 B aligned regions).
			4 * 1024;
	}

	static_assert(smem_bytes(true) <= kHopperMaxSmemBytes,
		"SM90 forward shared-memory footprint exceeds the Hopper limit");
};

// End-to-end tensor-parallel forward: the exact CuTe-DSL local-statistic GEMM,
// in-kernel local MAX and corrected SUM reductions, then the optional remote
// reduction and finalization follow-up.
//
// `gemm` describes this rank's contiguous vocabulary shard. Its `output` and
// `workspace` fields are launcher-owned and overwritten: both the local
// statistics and the vocabulary-split partials are internal scratch that never
// escapes the operation, so they come from liger::global_buffer_pool() rather
// than from the caller. Call configure_forward_tp_workspace() first (see
// forward_reduce.cuh).
template <int Compute = 90>
struct ForwardTpParamsSm90 {
	static constexpr int kCompute = Compute;

	ForwardGemmParamsSm90<Compute> gemm = {};

	float* nll = nullptr;      // [tokens]
	float* lse = nullptr;      // [tokens]
	float* entropy = nullptr;  // [tokens], required when ReturnEntropy=true

	std::int64_t team_handle = 0;
};

template <bool ReturnEntropy, int Compute = 90>
void fused_linear_scaled_cross_entropy_forward(
	const ForwardTpParamsSm90<Compute>& params,
	cudaStream_t stream);

extern template void fused_linear_scaled_cross_entropy_forward<false, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);
extern template void fused_linear_scaled_cross_entropy_forward<true, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
