#pragma once

// Architecture-neutral forward reduction primitives. SM100 calls the
// single-warp specialization from its persistent kernel; SM90 retains the
// standalone multi-warp finalizer.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dx_reduce.cuh"
#include "forward_reduction.cuh"
#include "liger_cute/detail/nvls.cuh"
#include "liger_cute/detail/remote_ring.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

inline constexpr std::uint64_t kForwardLocalAllgatherEpochSuffix =
	0x60000000u;
// Preserve the standalone finalizer's original 256-thread local parallelism.
inline constexpr int kForwardStandaloneRemoteWorkerWarpsPerBlock = 8;
static_assert(
	liger_cute::detail::remote_ring_worker_threads_per_block<
		kForwardStandaloneRemoteWorkerWarpsPerBlock>() == 256);

template <bool ReturnEntropy>
__device__ __forceinline__ void merge_forward_reduced_state_strided(
		float* accumulator,
		const float* contribution,
		std::size_t rows,
		std::size_t worker,
		std::size_t workers) {
	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	for (std::size_t row = worker;
			row < rows;
			row += workers) {
		std::size_t offset = row * kStateFields;
		ReducedSoftmaxState merged = merge_reduced_softmax<ReturnEntropy>(
			load_forward_reduced_state<ReturnEntropy>(
				accumulator + offset),
			load_forward_reduced_state<ReturnEntropy>(
				contribution + offset));
		store_forward_reduced_state<ReturnEntropy>(
			accumulator + offset, merged);
	}
}

template <bool ReturnEntropy>
struct ForwardReducedStateMerge {
	float* accumulator;
	std::size_t rows;

	__device__ __forceinline__ void operator()(
			const float* contribution,
			std::size_t,
			int worker,
			int workers) const {
		merge_forward_reduced_state_strided<ReturnEntropy>(
			accumulator,
			contribution,
			rows,
			static_cast<std::size_t>(worker),
			static_cast<std::size_t>(workers));
	}
};

template <bool ReturnEntropy>
__device__ __forceinline__ void reduce_forward_state_team_warp(
		const liger_cute::detail::RemoteReduceView& remote,
		float* state,
		std::size_t rows) {
	constexpr std::size_t kFields =
		forward_reduced_fields<ReturnEntropy>();
	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	int lane = liger_cute::detail::remote_ring_lane();
	float* max_source = remote.inbox;
	float* result = remote.inbox + remote.inbox_slot_elements;

	for (std::size_t row = static_cast<std::size_t>(lane);
			row < rows;
			row += kWarpSize) {
		max_source[row] =
			load_forward_reduced_state<ReturnEntropy>(
				state + row * kStateFields).max_value;
	}
	__syncwarp(liger_cute::detail::kRemoteRingFullWarpMask);
	nvshmemx_float_max_reduce_warp(
		static_cast<nvshmem_team_t>(remote.team_handle),
		result,
		max_source,
		rows);
	__syncwarp(liger_cute::detail::kRemoteRingFullWarpMask);

	for (std::size_t row = static_cast<std::size_t>(lane);
			row < rows;
			row += kWarpSize) {
		ReducedSoftmaxState local =
			load_forward_reduced_state<ReturnEntropy>(
				state + row * kStateFields);
		float global_max = result[row];
		float correction = local.exp_sum == 0.0f
			? 0.0f
			: expf(local.max_value - global_max);
		float* corrected = max_source + row * kFields;
		corrected[kForwardReducedSumField] =
			local.exp_sum * correction;
		corrected[kForwardReducedTargetField] =
			local.target_logit;
		if constexpr (ReturnEntropy) {
			corrected[kForwardReducedWeightedField] =
				local.exp_weighted_sum * correction;
		}
	}
	__syncwarp(liger_cute::detail::kRemoteRingFullWarpMask);
	float* reduced = result + rows;
	nvshmemx_float_sum_reduce_warp(
		static_cast<nvshmem_team_t>(remote.team_handle),
		reduced,
		max_source,
		rows * kFields);
	__syncwarp(liger_cute::detail::kRemoteRingFullWarpMask);

	for (std::size_t row = static_cast<std::size_t>(lane);
			row < rows;
			row += kWarpSize) {
		const float* fields = reduced + row * kFields;
		ReducedSoftmaxState global{
			result[row],
			fields[kForwardReducedSumField],
			fields[kForwardReducedTargetField],
			0.0f};
		if constexpr (ReturnEntropy) {
			global.exp_weighted_sum =
				fields[kForwardReducedWeightedField];
		}
		store_forward_reduced_state<ReturnEntropy>(
			state + row * kStateFields, global);
	}
	__syncwarp(liger_cute::detail::kRemoteRingFullWarpMask);
}

template <bool ReturnEntropy, int NumWorkerWarpsPerBlock>
__device__ __forceinline__ void reduce_forward_state_ring(
		const liger_cute::detail::RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		float* accumulator,
		std::size_t rows,
		std::uint64_t operation_suffix =
			liger_cute::detail::kForwardRemoteEpochSuffix,
		int operation_sequence = 0,
		std::uint64_t previous_operation_suffix = 0,
		bool drain = true) {
	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	std::size_t payload_elements = rows * kStateFields;
	liger_cute::detail::remote_ring_reduce<
		NumWorkerWarpsPerBlock>(
		remote,
		launch_epoch,
		operation_suffix,
		accumulator,
		payload_elements,
		ForwardReducedStateMerge<ReturnEntropy>{
			accumulator, rows},
		operation_sequence,
		previous_operation_suffix,
		drain);
}

template <bool ReturnEntropy>
__device__ __forceinline__ void allgather_forward_reduced_state_warp(
		const liger_cute::detail::NvlsReduceView& local,
		const DxReduceWorkspace<float>& comm,
		const std::uint64_t* launch_epoch,
		const float* local_shard,
		int padded_tokens,
		int warp,
		int communication_warp,
		std::uint64_t operation_suffix =
			kForwardLocalAllgatherEpochSuffix) {
	if (warp != communication_warp) return;

	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	std::uint64_t epoch =
		*launch_epoch | operation_suffix;
	liger_cute::detail::nvls_barrier_warp(
		comm.sync,
		local.multicast_sync,
		local.rank,
		local.size,
		epoch);
	liger_cute::detail::nvls_allgather_warp(
		local.multicast_reduced,
		local_shard,
		static_cast<std::size_t>(padded_tokens) * kStateFields,
		local.rank,
		local.size);
	liger_cute::detail::nvls_barrier_warp(
		comm.sync + local.size,
		local.multicast_sync + local.size,
		local.rank,
		local.size,
		epoch);
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
