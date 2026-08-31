#pragma once

// Launcher-owned workspace for the tensor-parallel forward. The GEMM and
// split reducer produce rank-local FP32 statistics here; the host launcher then
// performs NCCL MAX/SUM all-reduces with small CUDA kernels between/after them.
// No forward buffer is remotely addressed by device code, so every allocation
// is ordinary device memory from the shared BufferPool.

#include <cuda_runtime.h>

#include <cstddef>

#include "forward_gemm_sm90.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

inline constexpr int kForwardReducedSumField = 0;
inline constexpr int kForwardReducedTargetField = 1;
inline constexpr int kForwardReducedWeightedField = 2;
inline constexpr int kForwardReducedFields = 3;

struct ForwardBufferNames {
	static constexpr const char* kLocalMax =
		"fused_scaled_linear_cross_entropy_tp_forward_local_max";
	static constexpr const char* kGlobalMax =
		"fused_scaled_linear_cross_entropy_tp_forward_global_max";
	static constexpr const char* kPacked =
		"fused_scaled_linear_cross_entropy_tp_forward_packed";
	static constexpr const char* kReduced =
		"fused_scaled_linear_cross_entropy_tp_forward_reduced";
	static constexpr const char* kGemmSplitPartials =
		"fused_scaled_linear_cross_entropy_tp_forward_split_partials";
	static constexpr const char* kLocalSum =
		"fused_scaled_linear_cross_entropy_tp_forward_local_sum";
	static constexpr const char* kLocalTarget =
		"fused_scaled_linear_cross_entropy_tp_forward_local_target";
	static constexpr const char* kLocalWeighted =
		"fused_scaled_linear_cross_entropy_tp_forward_local_weighted";
};

__host__ __device__ constexpr int forward_reduced_fields(
		bool return_entropy) {
	return return_entropy
		? kForwardReducedFields
		: kForwardReducedFields - 1;
}

struct ForwardTpWorkspace {
	ForwardLocalStatsBuffers local;
	float* global_max;
	float* packed;
	float* reduced;
	void* gemm_split_partials;
	std::size_t gemm_split_partials_bytes;
	int tokens;
	int fields;
};

// Fixes reusable workspace capacities. Capacities are immutable after the
// first call; callers must configure their maximum shape upfront.
void configure_forward_tp_workspace(int max_tokens, int max_local_vocab);

std::size_t forward_tp_workspace_device_bytes(
	int max_tokens,
	int max_local_vocab);

// Invalidates cached capacities after the shared BufferPool is cleared.
void reset_forward_tp_workspace_configuration();

template <bool ReturnEntropy>
ForwardTpWorkspace reserve_forward_tp_workspace(int tokens);

extern template ForwardTpWorkspace
	reserve_forward_tp_workspace<false>(int);
extern template ForwardTpWorkspace
	reserve_forward_tp_workspace<true>(int);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
