#pragma once

#include <cmath>

#include <cuda_runtime.h>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

__host__ __device__ inline float scaled_cross_entropy_dlogit(
		float scaled_logit,
		float lse,
		bool is_target,
		float nll_scale,
		float temperature,
		float expected_scaled_logit = 0.0f,
		float entropy_scale = 0.0f,
		bool ignored = false) {
	if (ignored) return 0.0f;

	float probability = expf(scaled_logit - lse);
	float gradient = nll_scale * (probability - (is_target ? 1.0f : 0.0f));
	if (entropy_scale != 0.0f) {
		gradient += entropy_scale * probability *
			(expected_scaled_logit - scaled_logit);
	}
	return gradient / temperature;
}

struct BackwardWave {
	int first_m_tile;
	int num_m_tiles;
};

template <typename GemmConfig>
__host__ __device__ constexpr BackwardWave make_backward_wave(
		int wave, int total_m_tiles) {
	int first_m_tile = wave * GemmConfig::kMTilesPerWave;
	int remaining = total_m_tiles - first_m_tile;
	int num_m_tiles = remaining < GemmConfig::kMTilesPerWave
		? remaining
		: GemmConfig::kMTilesPerWave;
	return {first_m_tile, num_m_tiles > 0 ? num_m_tiles : 0};
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
