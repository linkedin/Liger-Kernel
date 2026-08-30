#include "liger_cute/detail/remote_all_reduce.cuh"

#include <nvshmem.h>
#include <nvshmemx.h>

#include "liger_cute/check.h"

namespace liger_cute {
namespace detail {
namespace {

__global__ void remote_put(
		float* inbox,
		const float* source,
		std::size_t count,
		int peer) {
	nvshmemx_float_put_block(inbox, source, count, peer);
}

__global__ void remote_handshake(
		std::uint64_t* signal,
		const std::uint64_t* launch_epoch,
		std::uint64_t epoch_suffix,
		int peer) {
	std::uint64_t epoch = *launch_epoch | epoch_suffix;
	nvshmemx_signal_op(
		signal, epoch, NVSHMEM_SIGNAL_SET, peer);
	nvshmem_quiet();
	nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, epoch);
}

__global__ void remote_sum(
		float* destination,
		const float* source,
		const float* inbox,
		std::size_t count) {
	for (std::size_t index =
				blockIdx.x * blockDim.x + threadIdx.x;
			index < count;
			index += blockDim.x * gridDim.x) {
		destination[index] = source[index] + inbox[index];
	}
}

}  // namespace

void launch_remote_pair_all_reduce(
		float* destination,
		float* inbox,
		const float* source,
		std::size_t count,
		std::uint64_t* ready,
		std::uint64_t* consumed,
		const std::uint64_t* launch_epoch,
		std::uint64_t epoch_suffix,
		int peer,
		cudaStream_t stream) {
	LIGER_CHECK(
		destination != nullptr && inbox != nullptr && source != nullptr &&
			ready != nullptr && consumed != nullptr &&
			launch_epoch != nullptr,
		"remote all-reduce buffers must be non-null");
	LIGER_CHECK(count > 0, "remote all-reduce count must be positive");
	LIGER_CHECK(peer >= 0, "remote all-reduce peer must be non-negative");

	remote_put<<<1, 256, 0, stream>>>(inbox, source, count, peer);
	nvshmemx_quiet_on_stream(stream);
	remote_handshake<<<1, 1, 0, stream>>>(
		ready, launch_epoch, epoch_suffix, peer);
	int blocks = static_cast<int>(
		(count + 255) / 256 < 1024 ? (count + 255) / 256 : 1024);
	remote_sum<<<blocks, 256, 0, stream>>>(
		destination, source, inbox, count);
	remote_handshake<<<1, 1, 0, stream>>>(
		consumed, launch_epoch, epoch_suffix, peer);
	cudaError_t error = cudaGetLastError();
	LIGER_CHECK(
		error == cudaSuccess,
		"remote pair all-reduce launch failed: ",
		cudaGetErrorString(error));
}

}  // namespace detail
}  // namespace liger_cute
