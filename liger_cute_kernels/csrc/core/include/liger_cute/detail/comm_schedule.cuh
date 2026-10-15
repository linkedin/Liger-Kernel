#pragma once

namespace liger_cute {
namespace detail {

#ifdef __CUDACC__
// Team-local round robin needs no shared schedule state or HCA layout assumption.
__host__ __device__ __forceinline__ int comm_peer(
    int rank, int offset, int num_pes) {
  return (rank + offset) % num_pes;
}

#endif  // __CUDACC__

}  // namespace detail
}  // namespace liger_cute
