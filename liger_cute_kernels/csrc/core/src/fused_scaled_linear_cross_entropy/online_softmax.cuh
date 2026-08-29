#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

struct OnlineSoftmaxState {
	float max_value;
	float exp_sum;
	float exp_weighted_sum;
	float target_logit;
	int has_target;

	__host__ __device__ OnlineSoftmaxState()
		: max_value(-1.0e38f),
		  exp_sum(0.0f),
		  exp_weighted_sum(0.0f),
		  target_logit(0.0f),
		  has_target(0) {}

	template <bool ReturnEntropy>
	__host__ __device__ void fold(float logit, bool valid, bool is_target) {
		if (!valid) return;

		float next_max = fmaxf(max_value, logit);
		float previous_scale = exp_sum == 0.0f
			? 0.0f
			: expf(max_value - next_max);
		float logit_scale = expf(logit - next_max);

		exp_sum = exp_sum * previous_scale + logit_scale;
		if constexpr (ReturnEntropy) {
			exp_weighted_sum =
				exp_weighted_sum * previous_scale + logit_scale * logit;
		}
		max_value = next_max;

		if (is_target) {
			target_logit = logit;
			has_target = 1;
		}
	}

	__host__ __device__ float local_lse() const {
		return max_value + logf(exp_sum);
	}
};

template <bool ReturnEntropy>
__host__ __device__ inline OnlineSoftmaxState merge_online_softmax(
		const OnlineSoftmaxState& lhs, const OnlineSoftmaxState& rhs) {
	if (lhs.exp_sum == 0.0f) return rhs;
	if (rhs.exp_sum == 0.0f) return lhs;

	OnlineSoftmaxState result;
	result.max_value = fmaxf(lhs.max_value, rhs.max_value);
	float lhs_scale = expf(lhs.max_value - result.max_value);
	float rhs_scale = expf(rhs.max_value - result.max_value);
	result.exp_sum = lhs.exp_sum * lhs_scale + rhs.exp_sum * rhs_scale;
	if constexpr (ReturnEntropy) {
		result.exp_weighted_sum =
			lhs.exp_weighted_sum * lhs_scale + rhs.exp_weighted_sum * rhs_scale;
	}
	result.has_target = lhs.has_target || rhs.has_target;
	result.target_logit = lhs.has_target ? lhs.target_logit : rhs.target_logit;
	return result;
}

struct CorrectedSoftmaxStats {
	float exp_sum;
	float exp_weighted_sum;
	float target_logit;
};

template <bool ReturnEntropy>
__host__ __device__ inline CorrectedSoftmaxStats correct_for_global_max(
		const OnlineSoftmaxState& local, float global_max) {
	if (local.exp_sum == 0.0f) {
		return {0.0f, 0.0f, local.has_target ? local.target_logit : 0.0f};
	}
	float scale = expf(local.max_value - global_max);
	float corrected_weighted_sum = 0.0f;
	if constexpr (ReturnEntropy) {
		corrected_weighted_sum = local.exp_weighted_sum * scale;
	}
	return {
		local.exp_sum * scale,
		corrected_weighted_sum,
		local.has_target ? local.target_logit : 0.0f,
	};
}

struct FinalizedSoftmax {
	float lse;
	float nll;
	float entropy;
};

template <bool ReturnEntropy>
__host__ __device__ inline FinalizedSoftmax finalize_softmax(
		float global_max,
		float global_sum,
		float global_target_logit,
		float global_weighted_sum,
		bool ignored) {
	float lse = global_max + logf(global_sum);
	float nll = ignored ? 0.0f : lse - global_target_logit;
	float entropy = 0.0f;
	if constexpr (ReturnEntropy) {
		if (!ignored) {
			entropy = lse - global_weighted_sum / global_sum;
		}
	}
	return {lse, nll, entropy};
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
