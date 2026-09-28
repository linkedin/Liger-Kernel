#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

inline constexpr int kWarpSize = 32;
inline constexpr int kNumThreads = 384;
inline constexpr int kNumWarps = kNumThreads / kWarpSize;

enum class BackwardWarpRole : std::uint8_t {
	kProducer,
	kDxCommunication,
	kReserved,
	kConsumer,
	kInactive,
};

__host__ __device__ constexpr BackwardWarpRole backward_warp_role(int warp_id) {
	return warp_id == 0
		? BackwardWarpRole::kProducer
		: (warp_id >= 1 && warp_id <= 2
			? BackwardWarpRole::kDxCommunication
			: (warp_id == 3
				? BackwardWarpRole::kReserved
				: (warp_id >= 4 && warp_id < kNumWarps
					? BackwardWarpRole::kConsumer
					: BackwardWarpRole::kInactive)));
}

struct DxTileGroup {
	int m_tile;
	int first_n_tile;
	int num_tiles;
	int sequence;
};

__host__ __device__ constexpr int ceil_div(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

template <typename Config>
__host__ __device__ constexpr int dx_groups_per_m_tile(int num_n_tiles) {
	return ceil_div(num_n_tiles, Config::kCoalesceTiles);
}

template <typename Config>
__host__ __device__ constexpr DxTileGroup make_dx_tile_group(
		int m_tile, int group_in_m, int num_n_tiles, int sequence) {
	int first_n_tile = group_in_m * Config::kCoalesceTiles;
	int remaining = num_n_tiles - first_n_tile;
	int num_tiles = remaining < Config::kCoalesceTiles
		? remaining
		: Config::kCoalesceTiles;
	return {m_tile, first_n_tile, num_tiles > 0 ? num_tiles : 0, sequence};
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
