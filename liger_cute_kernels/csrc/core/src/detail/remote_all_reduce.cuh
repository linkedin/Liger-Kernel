#pragma once

// Internal implementation detail behind launch_remote_reduce().

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "liger_cute/detail/tp_reduce.cuh"

namespace liger_cute {
namespace detail {

// dX shards can be several MiB; each cooperative-grid CTA uses the established
// 256-thread local reduction geometry.
inline constexpr int kRemoteSumWorkerWarpsPerBlock = 8;
static_assert(
	remote_ring_worker_threads_per_block<
		kRemoteSumWorkerWarpsPerBlock>() == 256);

void launch_remote_ring_all_reduce(
	const RemoteReduceView& remote,
	float* destination,
	const float* source,
	std::size_t count,
	const std::uint64_t* launch_epoch,
	std::uint64_t epoch_suffix,
	cudaStream_t stream);

}  // namespace detail
}  // namespace liger_cute
