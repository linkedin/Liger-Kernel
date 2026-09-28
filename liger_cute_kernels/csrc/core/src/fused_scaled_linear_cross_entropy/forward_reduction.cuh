#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "online_softmax.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

struct ForwardLocalStatsBuffers {
	float* local_max = nullptr;
	float* local_sum = nullptr;
	float* local_target = nullptr;
	float* local_weighted_sum = nullptr;
};

inline constexpr int kForwardReducedSumField = 0;
inline constexpr int kForwardReducedTargetField = 1;
inline constexpr int kForwardReducedWeightedField = 2;
inline constexpr int kForwardReducedFields = 3;

template <bool ReturnEntropy>
__host__ __device__ constexpr int forward_reduced_fields() {
	return ReturnEntropy ? kForwardReducedFields : 2;
}

template <bool ReturnEntropy>
__host__ __device__ constexpr int forward_reduced_state_fields() {
	return 1 + forward_reduced_fields<ReturnEntropy>();
}

template <bool ReturnEntropy>
__host__ __device__ inline ReducedSoftmaxState load_forward_reduced_state(
		const float* state) {
	ReducedSoftmaxState result{
		state[0],
		state[1 + kForwardReducedSumField],
		state[1 + kForwardReducedTargetField],
		0.0f};
	if constexpr (ReturnEntropy) {
		result.exp_weighted_sum =
			state[1 + kForwardReducedWeightedField];
	}
	return result;
}

template <bool ReturnEntropy>
__host__ __device__ inline void store_forward_reduced_state(
		float* state,
		const ReducedSoftmaxState& value) {
	state[0] = value.max_value;
	state[1 + kForwardReducedSumField] = value.exp_sum;
	state[1 + kForwardReducedTargetField] = value.target_logit;
	if constexpr (ReturnEntropy) {
		state[1 + kForwardReducedWeightedField] =
			value.exp_weighted_sum;
	}
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
