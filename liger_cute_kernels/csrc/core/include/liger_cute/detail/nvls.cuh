// Team-state-free NVLink SHARP primitives.
//
// Callers resolve the multicast pointers once, outside the consuming kernel,
// and pass them as ordinary kernel arguments. The helpers below therefore do
// not include NVSHMEM headers or reference NVSHMEM device state, allowing the
// consuming translation unit to remain non-RDC.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace liger_cute {
namespace detail {

__device__ __forceinline__ unsigned int nvls_lane_id() {
  unsigned int lane = 0;
#if defined(__CUDA_ARCH__)
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
#endif
  return lane;
}

// Barrier for one independent NVLS channel.
//
// Each PE multicast-stores its epoch into a distinct rank slot, then waits on
// the local replicas of every slot. Different CTAs may use the same multicast
// mapping concurrently as long as their signal vectors do not overlap.
__device__ __forceinline__ void nvls_barrier_warp(
    std::uint64_t* local_signals, std::uint64_t* multicast_signals,
    int team_rank, int team_size, std::uint64_t epoch) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900,
                "nvls_barrier_warp requires SM90 or newer");

  const unsigned int lane = nvls_lane_id();
  if (lane == 0) {
    asm volatile("multimem.st.release.sys.global.b64 [%0], %1;"
                 :
                 : "l"(multicast_signals + team_rank), "l"(epoch)
                 : "memory");
  }
  __syncwarp();

  for (int rank = static_cast<int>(lane); rank < team_size; rank += 32) {
    std::uint64_t observed = 0;
    do {
      asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                   : "=l"(observed)
                   : "l"(local_signals + rank)
                   : "memory");
    } while (observed != epoch);
  }
  __syncwarp();
#else
  (void)local_signals;
  (void)multicast_signals;
  (void)team_rank;
  (void)team_size;
  (void)epoch;
#endif
}

// FP32 SUM all-reduce using the NVLS two-shot algorithm.
//
// Each PE reduces one disjoint slice of the region through the multicast
// source pointer, then multicast-stores that slice to every PE's destination.
// The combination of all corresponding warps therefore produces the complete
// result on every PE.
//
// This primitive intentionally owns no collective state. Concurrent CTAs may
// call it for disjoint, identically prefixed regions through the same TP team's
// multicast mapping. The caller must separately guarantee:
//
//   * every PE has published its source region before any PE reads it;
//   * one converged full warp per PE calls with warp-uniform arguments;
//   * source and destination are not reused until every PE has completed;
//   * multicast_source and multicast_dest belong to the same NVLS-capable team.
//   * team_size is positive, team_rank is in [0, team_size), and both match the
//     multicast mapping's membership;
//   * count, rank assignment, and the region prefix are identical across PEs;
//   * nonempty regions are at least four-byte aligned and the source/destination
//     regions are either exactly in-place or non-overlapping.
//
// The pointers passed here must already include any block/channel prefix.
// A cross-PE completion protocol is still required after this function returns.
// Requires an NVLS-capable SM90-or-newer target.
__device__ __forceinline__ void nvls_sum_reduce_warp_twoshot(
    float* multicast_dest, const float* multicast_source, std::size_t count,
    int team_rank, int team_size) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900,
                "nvls_sum_reduce_warp_twoshot requires SM90 or newer");

  if (count == 0) return;

  const unsigned int lane = nvls_lane_id();
  const std::size_t elements_per_pe =
      count / static_cast<std::size_t>(team_size);
  const std::size_t rank_offset =
      elements_per_pe * static_cast<std::size_t>(team_rank);
  const std::size_t local_count =
      elements_per_pe +
      (team_rank == team_size - 1
           ? count % static_cast<std::size_t>(team_size)
           : 0);

  const float* source = multicast_source + rank_offset;
  float* dest = multicast_dest + rank_offset;
  std::size_t remaining = local_count;

  __syncwarp();
  asm volatile("" ::: "memory");

  if ((reinterpret_cast<std::uintptr_t>(source) & 0xF) == 0 &&
      (reinterpret_cast<std::uintptr_t>(dest) & 0xF) == 0) {
    const std::size_t vectors = remaining / 4;
    const auto* source_vectors = reinterpret_cast<const int4*>(source);
    auto* dest_vectors = reinterpret_cast<int4*>(dest);
    for (std::size_t index = static_cast<std::size_t>(lane); index < vectors;
         index += 32) {
      std::uint32_t value[4];
      asm volatile(
          "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 "
          "{%0, %1, %2, %3}, [%4];"
          : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
          : "l"(source_vectors + index)
          : "memory");
      asm volatile(
          "multimem.st.relaxed.sys.global.v4.f32 "
          "[%0], {%1, %2, %3, %4};"
          :
          : "l"(dest_vectors + index), "r"(value[0]), "r"(value[1]),
            "r"(value[2]), "r"(value[3])
          : "memory");
    }
    const std::size_t consumed = vectors * 4;
    source += consumed;
    dest += consumed;
    remaining -= consumed;
  }

  if ((reinterpret_cast<std::uintptr_t>(source) & 0x7) == 0 &&
      (reinterpret_cast<std::uintptr_t>(dest) & 0x7) == 0) {
    const std::size_t vectors = remaining / 2;
    const auto* source_vectors =
        reinterpret_cast<const std::uint64_t*>(source);
    auto* dest_vectors = reinterpret_cast<std::uint64_t*>(dest);
    for (std::size_t index = static_cast<std::size_t>(lane); index < vectors;
         index += 32) {
      std::uint32_t value[2];
      asm volatile(
          "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 "
          "{%0, %1}, [%2];"
          : "=r"(value[0]), "=r"(value[1])
          : "l"(source_vectors + index)
          : "memory");
      asm volatile(
          "multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};"
          :
          : "l"(dest_vectors + index), "r"(value[0]), "r"(value[1])
          : "memory");
    }
    const std::size_t consumed = vectors * 2;
    source += consumed;
    dest += consumed;
    remaining -= consumed;
  }

  for (std::size_t index = static_cast<std::size_t>(lane); index < remaining;
       index += 32) {
    std::uint32_t value;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=r"(value)
                 : "l"(source + index)
                 : "memory");
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;"
                 :
                 : "l"(dest + index), "r"(value)
                 : "memory");
  }

  asm volatile("fence.acq_rel.sys;" ::: "memory");
  __syncwarp();
#else
  (void)multicast_dest;
  (void)multicast_source;
  (void)count;
  (void)team_rank;
  (void)team_size;
#endif
}

__device__ __forceinline__ std::size_t nvls_rank_count(
    std::size_t count, int rank, int size) {
  std::size_t per_rank = count / static_cast<std::size_t>(size);
  return per_rank +
         (rank == size - 1 ? count % static_cast<std::size_t>(size) : 0);
}

__device__ __forceinline__ std::size_t nvls_rank_offset(
    std::size_t count, int rank, int size) {
  return count / static_cast<std::size_t>(size) *
         static_cast<std::size_t>(rank);
}

__device__ __forceinline__ void nvls_sum_reduce_scatter_warp(
    float* local_dest, const float* multicast_source, std::size_t count,
    int team_rank, int team_size) {
#if defined(__CUDA_ARCH__)
  const unsigned int lane = nvls_lane_id();
  const std::size_t rank_offset =
      nvls_rank_offset(count, team_rank, team_size);
  const std::size_t local_count =
      nvls_rank_count(count, team_rank, team_size);
  const float* source = multicast_source + rank_offset;

  const std::size_t vectors = local_count / 4;
  for (std::size_t index = lane; index < vectors; index += 32) {
    std::uint32_t value[4];
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
        : "l"(reinterpret_cast<const int4*>(source) + index)
        : "memory");
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(reinterpret_cast<int4*>(local_dest) + index), "r"(value[0]),
          "r"(value[1]), "r"(value[2]), "r"(value[3])
        : "memory");
  }
  for (std::size_t index = vectors * 4 + lane; index < local_count;
       index += 32) {
    std::uint32_t value;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
        : "=r"(value)
        : "l"(source + index)
        : "memory");
    asm volatile(
        "st.global.b32 [%0], %1;"
        :
        : "l"(local_dest + index), "r"(value)
        : "memory");
  }
  asm volatile("fence.acq_rel.sys;" ::: "memory");
  __syncwarp();
#else
  (void)local_dest;
  (void)multicast_source;
  (void)count;
  (void)team_rank;
  (void)team_size;
#endif
}

__device__ __forceinline__ void nvls_allgather_warp(
    float* multicast_dest, const float* local_source, std::size_t count,
    int team_rank, int team_size) {
#if defined(__CUDA_ARCH__)
  const unsigned int lane = nvls_lane_id();
  const std::size_t rank_offset =
      nvls_rank_offset(count, team_rank, team_size);
  const std::size_t local_count =
      nvls_rank_count(count, team_rank, team_size);
  float* destination = multicast_dest + rank_offset;

  const std::size_t vectors = local_count / 4;
  for (std::size_t index = lane; index < vectors; index += 32) {
    int4 value = reinterpret_cast<const int4*>(local_source)[index];
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 "
        "[%0], {%1, %2, %3, %4};"
        :
        : "l"(reinterpret_cast<int4*>(destination) + index), "r"(value.x),
          "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory");
  }
  for (std::size_t index = vectors * 4 + lane; index < local_count;
       index += 32) {
    std::uint32_t value =
        reinterpret_cast<const std::uint32_t*>(local_source)[index];
    asm volatile(
        "multimem.st.relaxed.sys.global.f32 [%0], %1;"
        :
        : "l"(destination + index), "r"(value)
        : "memory");
  }
  asm volatile("fence.acq_rel.sys;" ::: "memory");
  __syncwarp();
#else
  (void)multicast_dest;
  (void)local_source;
  (void)count;
  (void)team_rank;
  (void)team_size;
#endif
}

}  // namespace detail
}  // namespace liger_cute
