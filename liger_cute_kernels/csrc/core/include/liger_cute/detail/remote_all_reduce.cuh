#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace liger_cute {
namespace detail {

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
	cudaStream_t stream);

}  // namespace detail
}  // namespace liger_cute
