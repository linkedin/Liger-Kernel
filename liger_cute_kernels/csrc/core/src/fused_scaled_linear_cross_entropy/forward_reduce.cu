#include "forward_reduce.cuh"

#include <cstddef>
#include <cstdint>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "buffer_pool.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/nvls.cuh"
#include "online_softmax.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

int g_capacity_tokens = 0;
std::size_t g_capacity_split_partials_bytes = 0;

std::size_t token_bytes_at(int tokens) {
	return static_cast<std::size_t>(tokens) * sizeof(float);
}

std::size_t packed_bytes_at(int tokens) {
	return static_cast<std::size_t>(kForwardReducedFields) *
		token_bytes_at(tokens);
}

std::size_t split_partials_bytes_at(int tokens, int local_vocab) {
	return ForwardGemmLaunchSm90<90>::workspace_bytes(
		tokens, local_vocab, /*return_entropy=*/true);
}

std::size_t split_ready_bytes_at(int tokens) {
	return static_cast<std::size_t>(
		ForwardGemmLaunchSm90<90>::num_m_tiles(tokens)) * sizeof(int);
}

void ensure_capacity(int tokens) {
	LIGER_CHECK(tokens > 0, "tokens must be positive");
	LIGER_CHECK(
		g_capacity_tokens != 0,
		"fused_scaled_linear_cross_entropy forward: call "
		"configure_forward_tp_workspace(max_tokens, max_local_vocab) before "
		"the first launch");
	LIGER_CHECK(
		tokens <= g_capacity_tokens,
		"fused_scaled_linear_cross_entropy forward workspace was configured "
		"for ",
		g_capacity_tokens,
		" tokens but ",
		tokens,
		" were requested");
}

template <bool ReturnEntropy>
__device__ __forceinline__ void store_finalized_row(
		const float* global_max,
		const float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index,
		int row) {
	float global_weighted = 0.0f;
	if constexpr (ReturnEntropy) {
		global_weighted =
			reduced[kForwardReducedWeightedField * tokens + row];
	}
	FinalizedSoftmax result = finalize_softmax<ReturnEntropy>(
		global_max[row],
		reduced[kForwardReducedSumField * tokens + row],
		reduced[kForwardReducedTargetField * tokens + row],
		global_weighted,
		target[row] == ignore_index);
	nll[row] = result.nll;
	lse[row] = result.lse;
	if constexpr (ReturnEntropy) {
		entropy[row] = result.entropy;
	}
}

template <bool ReturnEntropy>
__global__ void local_finalize_forward_kernel(
		const float* global_max,
		const float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index) {
	for (int row = static_cast<int>(
				blockIdx.x * blockDim.x + threadIdx.x);
			row < tokens;
			row += static_cast<int>(blockDim.x * gridDim.x)) {
		store_finalized_row<ReturnEntropy>(
			global_max,
			reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			row);
	}
}

__device__ __forceinline__ void remote_handshake_block(
		std::uint64_t* signal,
		const std::uint64_t* launch_epoch,
		std::uint64_t suffix,
		int peer) {
	__syncthreads();
	if (threadIdx.x == 0) {
		std::uint64_t epoch = *launch_epoch | suffix;
		nvshmemx_signal_op(
			signal, epoch, NVSHMEM_SIGNAL_SET, peer);
		nvshmem_quiet();
		nvshmem_signal_wait_until(
			signal, NVSHMEM_CMP_EQ, epoch);
	}
	__syncthreads();
}

template <bool ReturnEntropy>
__global__ void remote_finalize_forward_kernel(
		__grid_constant__ const liger_cute::detail::RemoteReduceView remote,
		__grid_constant__ const liger_cute::detail::NvlsReduceView local,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		const std::uint64_t* launch_epoch,
		float* remote_source,
		float* global_max,
		float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index) {
	constexpr int kFields =
		ReturnEntropy ? kForwardReducedFields : 2;
	constexpr int kStateFields = 1 + kFields;
	int rows_per_rank =
		(tokens + local.size - 1) / local.size;
	int padded_tokens = rows_per_rank * local.size;
	std::size_t rows_owned =
		static_cast<std::size_t>(rows_per_rank);

	// Corresponding local ranks exchange disjoint token shards. The merged
	// shards are gathered once through the node-local NVLS mapping below.
	nvshmemx_float_put_block(
		remote.inbox,
		remote_source,
		rows_owned * static_cast<std::size_t>(kStateFields),
		remote.peer_world);
	remote_handshake_block(
		remote.ready,
		launch_epoch,
		0x40000000u,
		remote.peer_world);

	for (std::size_t owned_row =
				static_cast<std::size_t>(threadIdx.x);
			owned_row < rows_owned;
			owned_row += static_cast<std::size_t>(blockDim.x)) {
		std::size_t state =
			owned_row * static_cast<std::size_t>(kStateFields);
		float node_max = remote_source[state];
		float remote_max = remote.inbox[state];
		float max_value = fmaxf(node_max, remote_max);
		float node_sum =
			remote_source[state + 1 + kForwardReducedSumField];
		float remote_sum =
			remote.inbox[state + 1 + kForwardReducedSumField];
		float node_correction = node_sum == 0.0f
			? 0.0f
			: expf(node_max - max_value);
		float remote_correction = remote_sum == 0.0f
			? 0.0f
			: expf(remote_max - max_value);
		remote_source[state] = max_value;
		remote_source[state + 1 + kForwardReducedSumField] =
			node_sum * node_correction +
			remote_sum * remote_correction;
		remote_source[state + 1 + kForwardReducedTargetField] =
			remote_source[state + 1 + kForwardReducedTargetField] +
			remote.inbox[state + 1 + kForwardReducedTargetField];
		if constexpr (ReturnEntropy) {
			remote_source[state + 1 + kForwardReducedWeightedField] =
				remote_source[
					state + 1 + kForwardReducedWeightedField] *
					node_correction +
				remote.inbox[
					state + 1 + kForwardReducedWeightedField] *
					remote_correction;
		}
	}
	remote_handshake_block(
		remote.consumed,
		launch_epoch,
		0x40000000u,
		remote.peer_world);

	__syncthreads();
	int warp = static_cast<int>(threadIdx.x) / 32;
	if (warp == 0) {
		std::uint64_t epoch = *launch_epoch | 0x60000000u;
		liger_cute::detail::nvls_barrier_warp(
			comm.sync,
			local.multicast_sync,
			local.rank,
			local.size,
			epoch);
		liger_cute::detail::nvls_allgather_warp(
			local.multicast_reduced,
			remote_source,
			static_cast<std::size_t>(padded_tokens) *
				static_cast<std::size_t>(kStateFields),
			local.rank,
			local.size);
		liger_cute::detail::nvls_barrier_warp(
			comm.sync + local.size,
			local.multicast_sync + local.size,
			local.rank,
			local.size,
			epoch);
	}
	__syncthreads();

	for (int row = static_cast<int>(threadIdx.x);
			row < tokens;
			row += static_cast<int>(blockDim.x)) {
		std::size_t state =
			static_cast<std::size_t>(row) *
				static_cast<std::size_t>(kStateFields);
		global_max[row] = comm.reduced[state];
		for (int field = 0; field < kFields; ++field) {
			reduced[field * tokens + row] =
				comm.reduced[state + 1 + field];
		}
		store_finalized_row<ReturnEntropy>(
			global_max,
			reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			row);
	}
}

}  // namespace

void configure_forward_tp_workspace(int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");

	std::size_t split_bytes =
		split_partials_bytes_at(max_tokens, max_local_vocab);
	if (max_tokens <= g_capacity_tokens &&
		split_bytes <= g_capacity_split_partials_bytes) {
		return;
	}
	LIGER_CHECK(
		g_capacity_tokens == 0,
		"fused_scaled_linear_cross_entropy forward workspace is immutable "
		"after its first configuration");

	using Names = ForwardBufferNames;
	std::size_t token_bytes = token_bytes_at(max_tokens);
	std::size_t packed_bytes = packed_bytes_at(max_tokens);
	auto& pool = global_buffer_pool();

	pool.get_device(Names::kLocalMax, token_bytes);
	pool.get_device(Names::kGlobalMax, token_bytes);
	pool.get_device(Names::kReduced, packed_bytes);
	pool.get_device(Names::kGemmSplitPartials, split_bytes);
	pool.get_device(Names::kSplitReady, split_ready_bytes_at(max_tokens));
	pool.get_device(Names::kLocalSum, token_bytes);
	pool.get_device(Names::kLocalTarget, token_bytes);
	pool.get_device(Names::kLocalWeighted, token_bytes);

	g_capacity_tokens = max_tokens;
	g_capacity_split_partials_bytes = split_bytes;
}

std::size_t forward_tp_workspace_device_bytes(
		int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	return 5 * token_bytes_at(max_tokens) +
		packed_bytes_at(max_tokens) +
		split_partials_bytes_at(max_tokens, max_local_vocab) +
		split_ready_bytes_at(max_tokens);
}

void reset_forward_tp_workspace_configuration() {
	g_capacity_tokens = 0;
	g_capacity_split_partials_bytes = 0;
}

template <bool ReturnEntropy>
ForwardTpWorkspace reserve_forward_tp_workspace(int tokens) {
	ensure_capacity(tokens);

	using Names = ForwardBufferNames;
	std::size_t token_bytes = token_bytes_at(g_capacity_tokens);
	std::size_t packed_bytes = packed_bytes_at(g_capacity_tokens);
	auto& pool = global_buffer_pool();

	ForwardTpWorkspace workspace = {};
	workspace.local.local_max = static_cast<float*>(
		pool.get_device(Names::kLocalMax, token_bytes));
	workspace.global_max = static_cast<float*>(
		pool.get_device(Names::kGlobalMax, token_bytes));
	workspace.reduced = static_cast<float*>(
		pool.get_device(Names::kReduced, packed_bytes));
	workspace.split_ready = static_cast<int*>(
		pool.get_device(Names::kSplitReady, split_ready_bytes_at(g_capacity_tokens)));
	workspace.gemm_split_partials = pool.get_device(
		Names::kGemmSplitPartials, g_capacity_split_partials_bytes);
	workspace.gemm_split_partials_bytes =
		g_capacity_split_partials_bytes;
	workspace.local.local_sum = static_cast<float*>(
		pool.get_device(Names::kLocalSum, token_bytes));
	workspace.local.local_target = static_cast<float*>(
		pool.get_device(Names::kLocalTarget, token_bytes));
	workspace.local.local_weighted_sum = static_cast<float*>(
		pool.get_device(Names::kLocalWeighted, token_bytes));
	workspace.fields = forward_reduced_fields(ReturnEntropy);
	return workspace;
}

template ForwardTpWorkspace reserve_forward_tp_workspace<false>(int);
template ForwardTpWorkspace reserve_forward_tp_workspace<true>(int);

template <bool ReturnEntropy>
void launch_forward_remote_finalize_typed(
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
		cudaStream_t stream) {
	if (!remote.enabled()) {
		constexpr int kThreads = 256;
		int blocks = (tokens + kThreads - 1) / kThreads;
		local_finalize_forward_kernel<ReturnEntropy>
			<<<blocks, kThreads, 0, stream>>>(
			workspace.global_max,
			workspace.reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index);
	} else {
		float* remote_source = remote.reduced_shard;
		remote_finalize_forward_kernel<ReturnEntropy>
			<<<1, 256, 0, stream>>>(
			remote,
			local,
			comm,
			launch_epoch,
			remote_source,
			workspace.global_max,
			workspace.reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index);
	}
	cudaError_t error = cudaGetLastError();
	LIGER_CHECK(
		error == cudaSuccess,
		"forward finalize launch failed: ",
		cudaGetErrorString(error));
}

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
		cudaStream_t stream) {
	if (return_entropy) {
		launch_forward_remote_finalize_typed<true>(
			remote,
			local,
			comm,
			launch_epoch,
			workspace,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			stream);
	} else {
		launch_forward_remote_finalize_typed<false>(
			remote,
			local,
			comm,
			launch_epoch,
			workspace,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			stream);
	}
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
