#include "remote_all_reduce.cuh"

#include <nvshmem.h>
#include <nvshmemx.h>

#include "liger_cute/check.h"
#include "liger_cute/detail/remote_ring.cuh"

namespace liger_cute {
namespace detail {
namespace {

void launch_remote_collective(
		const void* kernel,
		dim3 grid,
		dim3 block,
		void** args,
		cudaStream_t stream,
		const char* what) {
	// RemoteReduceView is enabled only for a TP team covering WORLD, so every
	// PE enters these launches with identical kernel geometry and ordering.
	int status = nvshmemx_collective_launch(
		kernel, grid, block, args, 0, stream);
	LIGER_CHECK(
		status == 0,
		what,
		" failed with status ",
		status);
	cudaError_t error = cudaGetLastError();
	LIGER_CHECK(
		error == cudaSuccess,
		what,
		" CUDA launch failed: ",
		cudaGetErrorString(error));
}

struct RemoteSumMerge {
	float* destination;

	__device__ __forceinline__ void operator()(
			const float* contribution,
			std::size_t count,
			int worker,
			int workers) const {
		for (std::size_t index = static_cast<std::size_t>(worker);
				index < count;
				index += static_cast<std::size_t>(workers)) {
			destination[index] += contribution[index];
		}
	}
};

template <int NumWorkerWarpsPerBlock>
__global__ void remote_ring_sum_kernel(
		RemoteReduceView remote,
		float* destination,
		const float* source,
		std::size_t count,
		const std::uint64_t* launch_epoch,
		std::uint64_t epoch_suffix) {
	int lane = remote_ring_lane();
	int worker = lane;
	int workers = kRemoteRingWarpSize;
	if constexpr (NumWorkerWarpsPerBlock > 1) {
		worker = remote_ring_global_worker_index(
			static_cast<int>(blockIdx.x),
			static_cast<int>(blockDim.x),
			static_cast<int>(threadIdx.x));
		workers = remote_ring_global_worker_count(
			static_cast<int>(gridDim.x),
			static_cast<int>(blockDim.x));
	}
	if (destination != source) {
		for (std::size_t index = static_cast<std::size_t>(worker);
				index < count;
				index += static_cast<std::size_t>(workers)) {
			destination[index] = source[index];
		}
	}
	if constexpr (
			remote_ring_uses_grid_sync<
				NumWorkerWarpsPerBlock>()) {
		cooperative_groups::this_grid().sync();
	} else {
		__syncwarp(kRemoteRingFullWarpMask);
	}
	remote_ring_reduce<NumWorkerWarpsPerBlock>(
		remote,
		launch_epoch,
		epoch_suffix,
		source,
		count,
		RemoteSumMerge{destination});
}

}  // namespace

void launch_remote_ring_all_reduce(
		const RemoteReduceView& remote,
		float* destination,
		const float* source,
		std::size_t count,
		const std::uint64_t* launch_epoch,
		std::uint64_t epoch_suffix,
		cudaStream_t stream) {
	LIGER_CHECK(
		remote.enabled() && destination != nullptr && source != nullptr &&
			launch_epoch != nullptr,
		"remote all-reduce buffers must be non-null");
	LIGER_CHECK(count > 0, "remote all-reduce count must be positive");
	LIGER_CHECK(
		count <= remote.inbox_slot_elements,
		"remote all-reduce payload exceeds one inbox slot");
	LIGER_CHECK(
		(epoch_suffix & kRemoteRingStepMask) == 0,
		"remote all-reduce epoch suffix overlaps the ring step bits");

	RemoteReduceView remote_arg = remote;
	float* destination_arg = destination;
	const float* source_arg = source;
	std::size_t count_arg = count;
	const std::uint64_t* launch_epoch_arg = launch_epoch;
	std::uint64_t epoch_suffix_arg = epoch_suffix;
	void* args[] = {
		&remote_arg,
		&destination_arg,
		&source_arg,
		&count_arg,
		&launch_epoch_arg,
		&epoch_suffix_arg};
	auto* kernel =
		&remote_ring_sum_kernel<
			kRemoteSumWorkerWarpsPerBlock>;
	constexpr int kThreadsPerBlock =
		remote_ring_worker_threads_per_block<
			kRemoteSumWorkerWarpsPerBlock>();
	dim3 block_dims(kThreadsPerBlock, 1, 1);
	// TP remote execution already requires a uniform world-covering topology.
	// With the same kernel, block size, and payload on homogeneous GPUs, every
	// PE obtains and deterministically caps the same cooperative grid size.
	int max_grid_size = 0;
	int query_status =
		nvshmemx_collective_launch_query_gridsize(
			reinterpret_cast<const void*>(kernel),
			block_dims,
			args,
			0,
			&max_grid_size);
	LIGER_CHECK(
		query_status == 0,
		"remote ring sum cooperative grid query failed with status ",
		query_status);
	LIGER_CHECK(
		max_grid_size > 0,
		"remote ring sum cooperative grid query returned ",
		max_grid_size,
		" blocks");
	std::size_t count_blocks =
		(count + static_cast<std::size_t>(kThreadsPerBlock) - 1) /
		static_cast<std::size_t>(kThreadsPerBlock);
	int grid_size = count_blocks <
			static_cast<std::size_t>(max_grid_size)
		? static_cast<int>(count_blocks)
		: max_grid_size;
	LIGER_CHECK(
		grid_size > 0 && grid_size <= max_grid_size,
		"invalid remote ring sum cooperative grid size ",
		grid_size,
		" (maximum ",
		max_grid_size,
		")");
	launch_remote_collective(
		reinterpret_cast<const void*>(kernel),
		dim3(static_cast<unsigned int>(grid_size), 1, 1),
		block_dims,
		args,
		stream,
		"remote ring sum collective launch");
	cudaError_t error = cudaGetLastError();
	LIGER_CHECK(
		error == cudaSuccess,
		"remote ring all-reduce launch failed: ",
		cudaGetErrorString(error));
}

}  // namespace detail
}  // namespace liger_cute
