#pragma once

// Launcher-owned device workspace for forward split partials, local statistics,
// split completion counters, and finalized global statistics. Cross-PE staging
// reuses the symmetric TP reduction workspace configured for backward.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "dx_reduce.cuh"
#include "forward_gemm_sm90.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

struct ForwardBufferNames {
	static constexpr const char* kLocalMax =
		"fused_scaled_linear_cross_entropy_tp_forward_local_max";
	static constexpr const char* kGlobalMax =
		"fused_scaled_linear_cross_entropy_tp_forward_global_max";
	static constexpr const char* kReduced =
		"fused_scaled_linear_cross_entropy_tp_forward_reduced";
	static constexpr const char* kGemmSplitPartials =
		"fused_scaled_linear_cross_entropy_tp_forward_split_partials";
	static constexpr const char* kSplitReady =
		"fused_scaled_linear_cross_entropy_tp_forward_split_ready";
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
	float* reduced;
	int* split_ready;
	void* gemm_split_partials;
	std::size_t gemm_split_partials_bytes;
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

void launch_forward_remote_finalize(
	bool return_entropy,
	const liger_cute::detail::RemoteReduceView& remote,
	const liger_cute::detail::NvlsReduceView& local,
	const DxReduceWorkspace<float>& comm,
	const std::uint64_t* launch_epoch,
	const ForwardTpWorkspace& workspace,
	const std::int64_t* target,
	float* nll,
	float* lse,
	float* entropy,
	int tokens,
	std::int64_t ignore_index,
	cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
