#pragma once

// Native SM100 tensor-parallel fused scaled linear cross entropy contract.
// The executable paired-CTA 2SM UMMA path is kept separate from SM90 in
// forward_gemm_mainloop_sm100.cuh.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "config.cuh"
#include "forward_reduction.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

#ifndef LIGER_CUTE_FSLCE_SM100_STAGES
#define LIGER_CUTE_FSLCE_SM100_STAGES 5
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES
#define LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES 16
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_DISABLE_REMOTE
#define LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_DISABLE_REMOTE 0
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_TIMESTAMPS
#define LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_TIMESTAMPS 0
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_PIPELINED_RING_WAVES
#define LIGER_CUTE_FSLCE_SM100_PIPELINED_RING_WAVES 1
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_USE_WARP_TEAM_COLLECTIVES
#define LIGER_CUTE_FSLCE_SM100_USE_WARP_TEAM_COLLECTIVES 0
#endif

namespace liger {
namespace fused_scaled_linear_cross_entropy {

enum class ForwardWarpRoleSm100 : std::uint8_t {
	kLocalReduceControl,
	kRemoteCommunication,
	kTmaProducer,
	kUmmaProducer,
	kEpilogue,
	kInactive,
};

__host__ __device__ constexpr ForwardWarpRoleSm100 forward_warp_role_sm100(
		int warp_id) {
	return warp_id == 0
		? ForwardWarpRoleSm100::kLocalReduceControl
		: (warp_id == 1
			? ForwardWarpRoleSm100::kRemoteCommunication
			: (warp_id == 2
				? ForwardWarpRoleSm100::kTmaProducer
				: (warp_id == 3
					? ForwardWarpRoleSm100::kUmmaProducer
					: (warp_id >= 4 && warp_id < kNumWarps
						? ForwardWarpRoleSm100::kEpilogue
						: ForwardWarpRoleSm100::kInactive))));
}

template <int Compute = 100>
struct ForwardGemmConfigSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	static constexpr int kCompute = Compute;

	static constexpr int kCtaTileM = 128;
	static constexpr int kTileM = 256;
	static constexpr int kTileK = 64;
	static constexpr int kClusterM = 2;

	static constexpr int kUmmaTileN = 256;
	static constexpr int kAccumulatorPanels = 1;
	// Two N256 TMEM accumulators double-buffer across all 512 columns.
	static constexpr int kAccumulatorStages = 2;
	static constexpr int kTmemStageColumns = kUmmaTileN;
	static constexpr int kLogicalTileN = kTmemStageColumns;
	static constexpr int kTmemColumns =
		kAccumulatorStages * kTmemStageColumns;

	// A real per-CTA SMEM budget for the paired TMA operand destinations.
	static constexpr int kMainloopStages =
		LIGER_CUTE_FSLCE_SM100_STAGES;
	static constexpr int kEpilogueChunkN = 64;
	static constexpr int kEpilogueWarpgroups = 2;
	static constexpr int kEpilogueWarps = 8;
	static constexpr int kEpilogueThreads = kEpilogueWarps * kWarpSize;
	static constexpr int kWarpgroupSize = 4 * kWarpSize;
	static constexpr int kWarpgroupTileN =
		kUmmaTileN / kEpilogueWarpgroups;
	static constexpr int kChunksPerWarpgroup =
		kWarpgroupTileN / kEpilogueChunkN;
	static constexpr int kNumThreads = kNumWarps * kWarpSize;

	static constexpr int kLocalReduceWarp = 0;
	static constexpr int kRemoteCommunicationWarp = 1;
	static constexpr int kTmaWarp = 2;
	static constexpr int kUmmaWarp = 3;
	static constexpr int kFirstEpilogueWarp = 4;
	static constexpr int kLastEpilogueWarp = 11;

	static constexpr int kWarpgroup0BarrierId = 1;
	static constexpr int kWarpgroup1BarrierId = 2;
	static constexpr int kMmaEpilogueBarrierId = 3;
	static constexpr int kEpilogueBarrierId = 4;
	static constexpr int kComputeDoneBarrierId = 5;
	static constexpr int kDefaultMaxSplitN = 9;

	static_assert(kCtaTileM * kClusterM == kTileM);
	static_assert(
		kMainloopStages >= 3 && kMainloopStages <= 6,
		"SM100 forward mainloop stages must be between 3 and 6");
	static_assert(kLogicalTileN % kUmmaTileN == 0);
	static_assert(kWarpgroupTileN % kEpilogueChunkN == 0);
	static_assert(kTmemColumns == 512);
	static_assert(kNumThreads == 384);
	static_assert(
		forward_warp_role_sm100(kLocalReduceWarp) ==
		ForwardWarpRoleSm100::kLocalReduceControl);
	static_assert(
		forward_warp_role_sm100(kRemoteCommunicationWarp) ==
		ForwardWarpRoleSm100::kRemoteCommunication);
	static_assert(
		forward_warp_role_sm100(kTmaWarp) ==
		ForwardWarpRoleSm100::kTmaProducer);
	static_assert(
		forward_warp_role_sm100(kUmmaWarp) ==
		ForwardWarpRoleSm100::kUmmaProducer);
	static_assert(
		forward_warp_role_sm100(kFirstEpilogueWarp) ==
		ForwardWarpRoleSm100::kEpilogue);
	static_assert(
		forward_warp_role_sm100(kLastEpilogueWarp) ==
		ForwardWarpRoleSm100::kEpilogue);
};

inline constexpr float kForwardLog2ESm100 = 1.4426950408889634f;
inline constexpr float kForwardNegInfSm100 = -1.0e38f;
inline constexpr float kForwardMaskLogitSm100 = -3.0e38f;
inline constexpr int kForwardWaveNTilesSm100 =
	LIGER_CUTE_FSLCE_SM100_WAVE_N_TILES;
static_assert(
	kForwardWaveNTilesSm100 == 16 ||
		kForwardWaveNTilesSm100 == 32 ||
		kForwardWaveNTilesSm100 == 64 ||
		kForwardWaveNTilesSm100 == 128,
	"SM100 communication wave width must be 16, 32, 64, or 128 N256 tiles");
inline constexpr int kForwardWaveSourceSlotsSm100 = 4;
inline constexpr bool kForwardDiagnosticDisableRemoteSm100 =
	LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_DISABLE_REMOTE != 0;
inline constexpr bool kForwardDiagnosticTimestampsSm100 =
	LIGER_CUTE_FSLCE_SM100_DIAGNOSTIC_TIMESTAMPS != 0;
inline constexpr bool kForwardPipelinedRingWavesSm100 =
	LIGER_CUTE_FSLCE_SM100_PIPELINED_RING_WAVES != 0;
inline constexpr bool kForwardUseWarpTeamCollectivesSm100 =
	LIGER_CUTE_FSLCE_SM100_USE_WARP_TEAM_COLLECTIVES != 0;
inline constexpr int kForwardDiagnosticMaxWavesSm100 = 16;
inline constexpr int kForwardDiagnosticWaveStrideSm100 = 2;
inline constexpr int kForwardDiagnosticKernelStartSm100 = 0;
inline constexpr int kForwardDiagnosticFinalPublishSm100 = 1;
inline constexpr int kForwardDiagnosticLocalReduceCompleteSm100 = 2;
inline constexpr int kForwardDiagnosticRingCompleteSm100 = 3;
inline constexpr int kForwardDiagnosticAllgatherStartSm100 = 4;
inline constexpr int kForwardDiagnosticAllgatherCompleteSm100 = 5;
inline constexpr int kForwardDiagnosticOutputCompleteSm100 = 6;
inline constexpr int kForwardDiagnosticProducerWaitSm100 = 7;
inline constexpr int kForwardDiagnosticWarp0WaitSm100 = 8;
inline constexpr int kForwardDiagnosticWarp1WaitSm100 = 9;
inline constexpr int kForwardDiagnosticRingTicksSm100 = 10;
inline constexpr int kForwardDiagnosticWaveBaseSm100 = 16;
inline constexpr int kForwardDiagnosticEntriesSm100 =
	kForwardDiagnosticWaveBaseSm100 +
	kForwardDiagnosticMaxWavesSm100 *
		kForwardDiagnosticWaveStrideSm100;
static_assert(kForwardDiagnosticEntriesSm100 == 48);
inline constexpr int kForwardWaveEpochShiftSm100 = 16;
inline constexpr int kForwardWaveEpochBitsSm100 = 12;
inline constexpr int kForwardMaxWavesSm100 =
	(1 << kForwardWaveEpochBitsSm100) - 1;
inline constexpr std::uint64_t kForwardWaveEpochMaskSm100 =
	static_cast<std::uint64_t>(kForwardMaxWavesSm100) <<
	kForwardWaveEpochShiftSm100;
inline constexpr std::uint64_t kForwardOperationEpochMaskSm100 =
	0xf0000000u;
inline constexpr std::uint64_t kForwardLocalReduceEpochSuffixSm100 =
	0x10000000u;
static_assert(
	(kForwardWaveSourceSlotsSm100 &
		(kForwardWaveSourceSlotsSm100 - 1)) == 0);
static_assert(
	kForwardWaveSourceSlotsSm100 >= 2,
	"remote wave production requires at least double buffering");
static_assert(
	!liger_cute::detail::remote_ring_uses_grid_sync<1>(),
	"the SM100 communication warp must not enter a block/grid barrier");
static_assert(
	(kForwardWaveEpochMaskSm100 &
		liger_cute::detail::kRemoteRingStepMask) == 0);
static_assert(
	(kForwardWaveEpochMaskSm100 &
		kForwardOperationEpochMaskSm100) == 0);
static_assert(
	(kForwardLocalReduceEpochSuffixSm100 &
		kForwardWaveEpochMaskSm100) == 0);
static_assert(
	(liger_cute::detail::kForwardRemoteEpochSuffix &
		kForwardWaveEpochMaskSm100) == 0);

__host__ __device__ constexpr int forward_wave_count_sm100(
		int logical_n_tiles) {
	return ceil_div(logical_n_tiles, kForwardWaveNTilesSm100);
}

__host__ __device__ constexpr int forward_wave_begin_tile_sm100(
		int wave) {
	return wave * kForwardWaveNTilesSm100;
}

__host__ __device__ constexpr int forward_wave_end_tile_sm100(
		int wave, int logical_n_tiles) {
	int end = forward_wave_begin_tile_sm100(wave) +
		kForwardWaveNTilesSm100;
	return end < logical_n_tiles ? end : logical_n_tiles;
}

__host__ __device__ constexpr int forward_wave_slot_sm100(int wave) {
	return wave & (kForwardWaveSourceSlotsSm100 - 1);
}

__host__ __device__ constexpr int forward_wave_reused_wave_sm100(
		int wave) {
	return wave >= kForwardWaveSourceSlotsSm100
		? wave - kForwardWaveSourceSlotsSm100
		: -1;
}

__host__ __device__ constexpr bool forward_wave_count_supported_sm100(
		int waves) {
	return waves >= 1 && waves <= kForwardMaxWavesSm100;
}

__host__ __device__ constexpr std::uint64_t forward_wave_epoch_sm100(
		std::uint64_t launch_epoch, int wave) {
	return launch_epoch |
		(static_cast<std::uint64_t>(wave + 1) <<
			kForwardWaveEpochShiftSm100);
}

__host__ __device__ constexpr std::uint64_t
forward_wave_operation_suffix_sm100(
		std::uint64_t operation_suffix, int wave) {
	return operation_suffix |
		(static_cast<std::uint64_t>(wave + 1) <<
			kForwardWaveEpochShiftSm100);
}

__host__ __device__ constexpr int
forward_wave_tile_valid_columns_sm100(
		int local_vocab, int wave, int tile_in_wave) {
	int column =
		(forward_wave_begin_tile_sm100(wave) + tile_in_wave) *
		ForwardGemmConfigSm100<>::kLogicalTileN;
	int remaining = local_vocab - column;
	if (remaining <= 0) return 0;
	return remaining < ForwardGemmConfigSm100<>::kLogicalTileN
		? remaining
		: ForwardGemmConfigSm100<>::kLogicalTileN;
}

__host__ __device__ constexpr bool
forward_wave_tile_is_neutral_sm100(
		int local_vocab, int wave, int tile_in_wave) {
	return forward_wave_tile_valid_columns_sm100(
		local_vocab, wave, tile_in_wave) == 0;
}

__host__ __device__ inline float forward_exp2_sm100(float value) {
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return result;
#else
	return exp2f(value);
#endif
}

template <int Compute = 100>
struct ForwardGemmTuningSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	int split_n = 0;
	int base_split_n = 0;
	int extra_m_pairs = 0;
	int target_cluster_pairs = 0;
	int max_split_n = 9;
};

template <int Compute = 100>
struct ForwardGemmSplitSm100 {
	using Config = ForwardGemmConfigSm100<Compute>;

	int split_n = 1;
	int base_split_n = 1;
	int extra_m_pairs = 0;
	int num_m_tiles = 0;
	int num_m_pairs = 0;
	int num_logical_n_tiles = 0;
	int num_cluster_pairs = 0;
	int num_waves = 0;

	__host__ __device__ int split_count_for_pair(int m_pair) const {
		if (extra_m_pairs == 0) return base_split_n;
		return m_pair < extra_m_pairs ? split_n : base_split_n;
	}
};

template <int Compute = 100>
struct ForwardGemmPartialsSm100 {
	float* partial_max = nullptr;
	float* partial_sum = nullptr;
	float* partial_target = nullptr;
	float* partial_weighted = nullptr;
	std::uint64_t* ready = nullptr;
};

template <int Compute = 100>
struct ForwardWaveWorkspaceSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	std::uint64_t* tile_ready = nullptr;
	std::uint64_t* slot_released = nullptr;
	const std::uint64_t* launch_epoch = nullptr;
	std::uint64_t* diagnostics = nullptr;
	float* source_slots = nullptr;
	float* running_state = nullptr;
	std::size_t source_slot_elements = 0;
	int rows_per_rank = 0;
	int padded_tokens = 0;
};

struct ForwardFinalOutputsSm100 {
	float* nll = nullptr;
	float* lse = nullptr;
	float* entropy = nullptr;
};

template <int Compute = 100>
struct ForwardGemmParamsSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	const void* x = nullptr;
	const void* weight = nullptr;
	const std::int64_t* target = nullptr;
	ForwardLocalStatsBuffers output = {};
	void* workspace = nullptr;
	std::size_t workspace_bytes = 0;

	int tokens = 0;
	int hidden = 0;
	int local_vocab = 0;
	std::int64_t vocab_start = 0;
	std::int64_t ignore_index = -100;
	float inverse_temperature = 1.0f;
	ForwardGemmTuningSm100<Compute> tuning = {};
};

template <int Compute = 100>
struct ForwardGemmEpilogueSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	template <bool ReturnEntropy, int ChunkN, class Element>
	__device__ static void fold_chunk(
			OnlineSoftmaxState& state,
			const Element* logits,
			int valid_cols,
			int target_offset,
			float inverse_temperature) {
		float values[ChunkN];
		float chunk_max = kForwardMaskLogitSm100;
		#pragma unroll
		for (int column = 0; column < ChunkN; ++column) {
			float value =
				static_cast<float>(logits[column]) * inverse_temperature;
			if (column >= valid_cols) value = kForwardMaskLogitSm100;
			values[column] = value;
			chunk_max = fmaxf(chunk_max, value);
		}

		float next_max = fmaxf(state.max_value, chunk_max);
		float chunk_sum = 0.0f;
		float chunk_weighted = 0.0f;
		#pragma unroll
		for (int column = 0; column < ChunkN; ++column) {
			float weight = forward_exp2_sm100(
				(values[column] - next_max) * kForwardLog2ESm100);
			chunk_sum += weight;
			if constexpr (ReturnEntropy) {
				chunk_weighted += weight * values[column];
			}
		}
		float previous_scale = forward_exp2_sm100(
			(state.max_value - next_max) * kForwardLog2ESm100);
		state.exp_sum = state.exp_sum * previous_scale + chunk_sum;
		if constexpr (ReturnEntropy) {
			state.exp_weighted_sum =
				state.exp_weighted_sum * previous_scale + chunk_weighted;
		}
		state.max_value = next_max;

		if (target_offset >= 0 && target_offset < valid_cols) {
			state.target_logit =
				static_cast<float>(logits[target_offset]) *
				inverse_temperature;
			state.has_target = 1;
		}
	}

	template <bool ReturnEntropy>
	__host__ __device__ static OnlineSoftmaxState combine_scaled(
			const OnlineSoftmaxState& lhs,
			const OnlineSoftmaxState& rhs) {
		OnlineSoftmaxState result;
		result.max_value = fmaxf(lhs.max_value, rhs.max_value);
		float lhs_scale = forward_exp2_sm100(
			(lhs.max_value - result.max_value) * kForwardLog2ESm100);
		float rhs_scale = forward_exp2_sm100(
			(rhs.max_value - result.max_value) * kForwardLog2ESm100);
		result.exp_sum =
			lhs.exp_sum * lhs_scale + rhs.exp_sum * rhs_scale;
		if constexpr (ReturnEntropy) {
			result.exp_weighted_sum =
				lhs.exp_weighted_sum * lhs_scale +
				rhs.exp_weighted_sum * rhs_scale;
		}
		result.target_logit = lhs.target_logit + rhs.target_logit;
		result.has_target = lhs.has_target || rhs.has_target;
		return result;
	}

	template <bool ReturnEntropy>
	__host__ __device__ static void store_partial(
			const OnlineSoftmaxState& state,
			const ForwardGemmPartialsSm100<Compute>& partials,
			int index) {
		partials.partial_max[index] = state.max_value;
		partials.partial_sum[index] = state.exp_sum;
		partials.partial_target[index] =
			state.has_target ? state.target_logit : 0.0f;
		if constexpr (ReturnEntropy) {
			partials.partial_weighted[index] =
				state.exp_weighted_sum;
		}
	}

	template <bool ReturnEntropy>
	__host__ __device__ static void store_row(
			const OnlineSoftmaxState& state,
			const ForwardLocalStatsBuffers& output,
			int row) {
		output.local_max[row] = state.max_value;
		output.local_sum[row] = state.exp_sum;
		output.local_target[row] =
			state.has_target ? state.target_logit : 0.0f;
		if constexpr (ReturnEntropy) {
			output.local_weighted_sum[row] =
				state.exp_weighted_sum;
		}
	}
};

template <int Compute = 100>
struct ForwardGemmLaunchSm100 {
	using Config = ForwardGemmConfigSm100<Compute>;
	using Split = ForwardGemmSplitSm100<Compute>;

	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	// Split/finalizer rows stay CTA-local M128 tiles in both MMA modes.
	__host__ __device__ static constexpr int num_m_tiles(int tokens) {
		return ceil_div(tokens, Config::kCtaTileM);
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

	__host__ static Split resolve_split(
			const ForwardGemmTuningSm100<Compute>& tuning,
			int tokens,
			int local_vocab,
			int max_active_clusters) {
		Split split;
		split.num_m_tiles = num_m_tiles(tokens);
		split.num_m_pairs = num_m_pairs(tokens);
		split.num_logical_n_tiles =
			num_logical_n_tiles(local_vocab);
		split.num_waves =
			forward_wave_count_sm100(
				split.num_logical_n_tiles);

		if (tuning.base_split_n != 0) {
			split.split_n = tuning.split_n;
			split.base_split_n = tuning.base_split_n;
			split.extra_m_pairs = tuning.extra_m_pairs;
		} else if (tuning.split_n != 0) {
			split.split_n = tuning.split_n;
			split.base_split_n = tuning.split_n;
		} else {
			int max_split_n = tuning.max_split_n > 0
				? tuning.max_split_n
				: Config::kDefaultMaxSplitN;
			int target_clusters = tuning.target_cluster_pairs > 0
				? tuning.target_cluster_pairs
				: (max_active_clusters > 0
					? max_active_clusters
					: 1);
			int per_pair_cap = max_split_n < split.num_logical_n_tiles
				? max_split_n
				: split.num_logical_n_tiles;
			int max_cluster_pairs =
				split.num_m_pairs * per_pair_cap;
			int wanted = target_clusters < max_cluster_pairs
				? target_clusters
				: max_cluster_pairs;
			int cluster_pairs =
				split.num_m_pairs > wanted
				? split.num_m_pairs
				: wanted;
			split.base_split_n =
				cluster_pairs / split.num_m_pairs;
			split.extra_m_pairs =
				cluster_pairs % split.num_m_pairs;
			split.split_n =
				split.base_split_n +
				(split.extra_m_pairs != 0 ? 1 : 0);
		}

		split.num_cluster_pairs =
			split.num_m_pairs * split.base_split_n +
			split.extra_m_pairs;
		return split;
	}

	__host__ __device__ static std::size_t workspace_bytes(
			int tokens,
			int local_vocab,
			bool return_entropy,
			int max_split_n = Config::kDefaultMaxSplitN) {
		int logical_n = num_logical_n_tiles(local_vocab);
		if (logical_n < 1) logical_n = 1;
		int cap = max_split_n < logical_n
			? max_split_n
			: logical_n;
		if (cap < 1) cap = 1;
		std::size_t rows =
			static_cast<std::size_t>(
				kForwardWaveSourceSlotsSm100) *
			static_cast<std::size_t>(num_m_tiles(tokens)) *
			static_cast<std::size_t>(cap) *
			static_cast<std::size_t>(Config::kCtaTileM);
		return rows * sizeof(float) *
			(return_entropy ? 4u : 3u);
	}

	__host__ __device__ static std::size_t wave_partial_ready_entries(
			int tokens,
			int local_vocab,
			int max_split_n = Config::kDefaultMaxSplitN) {
		int logical_n = num_logical_n_tiles(local_vocab);
		if (logical_n < 1) logical_n = 1;
		int cap = max_split_n < logical_n
			? max_split_n
			: logical_n;
		if (cap < 1) cap = 1;
		return static_cast<std::size_t>(
				kForwardWaveSourceSlotsSm100) *
			static_cast<std::size_t>(num_m_tiles(tokens)) *
			static_cast<std::size_t>(cap);
	}

	__host__ __device__ static std::size_t wave_tile_ready_entries(
			int tokens) {
		return static_cast<std::size_t>(
				kForwardWaveSourceSlotsSm100) *
			static_cast<std::size_t>(num_m_tiles(tokens));
	}
};

template <int Compute = 100>
struct ForwardTpParamsSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	ForwardGemmParamsSm100<Compute> gemm = {};
	float* nll = nullptr;      // FP32 [tokens], zero at ignore_index rows
	float* lse = nullptr;      // FP32 [tokens], globally reduced logsumexp
	float* entropy = nullptr;  // FP32 [tokens], optional globally reduced entropy
	// NVSHMEM team configured by configure_backward_tp_symmetric() and
	// configure_forward_tp_workspace(). Multi-host execution requires a
	// world-covering team with uniform node sizes.
	std::int64_t team_handle = 0;
};

// Collective over params.team_handle. Every PE must launch the same shape and
// ReturnEntropy specialization in the same stream order. `weight` is the local
// contiguous vocabulary shard while targets are global vocabulary indices.
template <bool ReturnEntropy, int Compute = 100>
void fused_linear_scaled_cross_entropy_forward_sm100(
	const ForwardTpParamsSm100<Compute>& params,
	cudaStream_t stream);

extern template void fused_linear_scaled_cross_entropy_forward_sm100<
	false,
	100>(const ForwardTpParamsSm100<100>&, cudaStream_t);
extern template void fused_linear_scaled_cross_entropy_forward_sm100<
	true,
	100>(const ForwardTpParamsSm100<100>&, cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
