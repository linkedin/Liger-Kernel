#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

#include "liger_cute/detail/tp_reduce.cuh"

namespace liger_cute {
namespace detail {

#if defined(__CUDACC__)
__device__ __forceinline__ int remote_ring_lane() {
	return static_cast<int>(threadIdx.x) &
		(kRemoteRingWarpSize - 1);
}

__device__ __forceinline__ void remote_ring_wait_warp(
		std::uint64_t* signal,
		std::uint64_t value,
		int lane) {
	__syncwarp(kRemoteRingFullWarpMask);
	if (lane == 0) {
		nvshmem_signal_wait_until(
			signal, NVSHMEM_CMP_GE, value);
	}
	__syncwarp(kRemoteRingFullWarpMask);
}

__device__ __forceinline__ void remote_ring_signal_warp(
		std::uint64_t* signal,
		std::uint64_t value,
		int peer,
		int qp_handle,
		int lane) {
	__syncwarp(kRemoteRingFullWarpMask);
	if (lane == 0) {
		nvshmemx_qp_signal_op(
			signal,
			value,
			NVSHMEM_SIGNAL_SET,
			peer,
			static_cast<nvshmemx_qp_handle_t>(qp_handle));
	}
	__syncwarp(kRemoteRingFullWarpMask);
}

__device__ __forceinline__ void remote_ring_put_warp(
		float* destination,
		const float* source,
		std::size_t count,
		std::uint64_t* ready,
		std::uint64_t value,
		int peer,
		int qp_handle) {
	// One blocking warp collective publishes the entire payload and ready
	// signal through the same QP. Do not fragment the message or add a
	// separate fence, signal, or completion operation.
	nvshmemx_qp_float_put_signal_warp(
		destination,
		source,
		count,
		ready,
		value,
		NVSHMEM_SIGNAL_SET,
		peer,
		static_cast<nvshmemx_qp_handle_t>(qp_handle));
	__syncwarp(kRemoteRingFullWarpMask);
}

__device__ __forceinline__ float* remote_ring_inbox_slot(
		const RemoteReduceView& remote,
		int step,
		int operation_sequence) {
	return remote.inbox +
		static_cast<std::size_t>(
			remote_ring_transport_slot(
				remote.size, step, operation_sequence)) *
			remote.inbox_slot_elements;
}

__device__ __forceinline__ float* remote_ring_transport_step_warp(
		const RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		std::uint64_t operation_suffix,
		std::uint64_t previous_operation_suffix,
		int operation_sequence,
		int step,
		const float* original,
		std::size_t count,
		int lane) {
	int reused_step = remote_ring_reused_step(step);
	int slot = remote_ring_transport_slot(
		remote.size, step, operation_sequence);
	if (
		remote_ring_uses_consumed(remote.size) &&
		reused_step >= 0) {
		remote_ring_wait_warp(
			remote.consumed + slot,
			remote_ring_signal_value(
				*launch_epoch,
				operation_suffix,
				reused_step),
			lane);
	} else if (
		remote_ring_uses_consumed(remote.size) &&
		previous_operation_suffix != 0) {
		int previous_step =
			remote_ring_last_step_for_slot(remote.size, slot);
		if (previous_step >= 0) {
			remote_ring_wait_warp(
				remote.consumed + slot,
				remote_ring_signal_value(
					*launch_epoch,
					previous_operation_suffix,
					previous_step),
				lane);
		}
	}

	int forwarded_step = remote_ring_forwarded_step(step);
	const float* source = forwarded_step < 0
		? original
		: remote_ring_inbox_slot(
			remote, forwarded_step, operation_sequence);
	float* destination =
		remote_ring_inbox_slot(
			remote, step, operation_sequence);
	std::uint64_t value = remote_ring_signal_value(
		*launch_epoch, operation_suffix, step);
	remote_ring_put_warp(
		destination,
		source,
		count,
		remote.ready + slot,
		value,
		remote.next_world,
		remote.qp_handle);

	if (
		remote_ring_uses_consumed(remote.size) &&
		forwarded_step >= 0) {
		remote_ring_signal_warp(
			remote.consumed +
				remote_ring_slot(forwarded_step),
			remote_ring_signal_value(
				*launch_epoch,
				operation_suffix,
				forwarded_step),
			remote.previous_world,
			remote.qp_handle,
			lane);
	}
	remote_ring_wait_warp(remote.ready + slot, value, lane);
	return destination;
}

__device__ __forceinline__ void remote_ring_finish_warp(
		const RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		std::uint64_t operation_suffix,
		int lane,
		bool drain) {
	if (!remote_ring_uses_consumed(remote.size)) return;
	int steps = remote_ring_step_count(remote.size);
	int final_step = steps - 1;
	remote_ring_signal_warp(
		remote.consumed + remote_ring_slot(final_step),
		remote_ring_signal_value(
			*launch_epoch, operation_suffix, final_step),
		remote.previous_world,
		remote.qp_handle,
		lane);
	if (!drain) return;

	for (int step = remote_ring_drain_begin_step(remote.size);
			step < steps;
			++step) {
		remote_ring_wait_warp(
			remote.consumed + remote_ring_slot(step),
			remote_ring_signal_value(
				*launch_epoch, operation_suffix, step),
			lane);
	}
}

// Circulates original contributions around the matching-rank ring. Step zero
// sends `original`; later steps forward the preceding inbox slot, never the
// accumulator owned by `merge`. Communication warp zero alone executes the
// transport. Additional blocks and warps only merge the received contribution.
template <int NumWorkerWarpsPerBlock, class Merge>
__device__ __forceinline__ void remote_ring_reduce(
		const RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		std::uint64_t operation_suffix,
		const float* original,
		std::size_t count,
		Merge merge,
		int operation_sequence = 0,
		std::uint64_t previous_operation_suffix = 0,
		bool drain = true) {
	static_assert(
		NumWorkerWarpsPerBlock >= 1 &&
			NumWorkerWarpsPerBlock <=
				kMaxRemoteRingWorkerWarpsPerBlock);
	int steps = remote_ring_step_count(remote.size);

	if constexpr (NumWorkerWarpsPerBlock == 1) {
		constexpr int kWorkers = kRemoteRingWarpSize;
		int lane = remote_ring_lane();
		for (int step = 0; step < steps; ++step) {
			const float* contribution =
				remote_ring_transport_step_warp(
					remote,
					launch_epoch,
					operation_suffix,
					previous_operation_suffix,
					operation_sequence,
					step,
					original,
					count,
					lane);
			merge(contribution, count, lane, kWorkers);
			__syncwarp(kRemoteRingFullWarpMask);
		}
		remote_ring_finish_warp(
			remote,
			launch_epoch,
			operation_suffix,
			lane,
			drain);
	} else {
		cooperative_groups::grid_group grid =
			cooperative_groups::this_grid();
		int global_worker = remote_ring_global_worker_index(
			static_cast<int>(blockIdx.x),
			static_cast<int>(blockDim.x),
			static_cast<int>(threadIdx.x));
		int global_workers = remote_ring_global_worker_count(
			static_cast<int>(gridDim.x),
			static_cast<int>(blockDim.x));
		int global_warp = global_worker / kRemoteRingWarpSize;
		int lane = global_worker & (kRemoteRingWarpSize - 1);
		for (int step = 0; step < steps; ++step) {
			if (global_warp == 0) {
				remote_ring_transport_step_warp(
					remote,
					launch_epoch,
					operation_suffix,
					previous_operation_suffix,
					operation_sequence,
					step,
					original,
					count,
					lane);
			}
			// Publish global communication warp zero's ready observation
			// before any grid worker reads the inbox.
			grid.sync();
			const float* contribution =
				remote_ring_inbox_slot(
					remote, step, operation_sequence);
			merge(
				contribution,
				count,
				global_worker,
				global_workers);
			// All grid readers must finish before communication warp zero
			// forwards or acknowledges this slot on the next step.
			grid.sync();
		}
		if (global_warp == 0) {
			remote_ring_finish_warp(
				remote,
				launch_epoch,
				operation_suffix,
				lane,
				drain);
		}
		grid.sync();
	}
}
#endif

}  // namespace detail
}  // namespace liger_cute
