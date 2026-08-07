// comm_schedule.cuh — CORE-INTERNAL communication schedule (device tables + host setup).
//
// Ported from LigerCommKernels' nvshmem_helpers.{h,cuh}. NOT part of the flat
// ABI: it declares __constant__ device storage and a device accessor for the
// dispatch/combine kernels, plus the host-side functions that populate them.
// The flat ABI wrapper (liger_cute_init_comm_schedule, in nvshmem.h) calls the
// host setup below; the binding TU never includes this header.
//
// ─── Communication schedule ──────────────────────────────────────────────────
// At each step t, source PE r targets dest_table[(t + r) mod NM]. The table is
// constructed with host pattern A,B,...,N-1,A,B,...  (period N) so cyclic shifts
// preserve the per-step IB count = M*(N-1) and each NIC is hit at most once per
// step (full permutation property).
//
// rank_table is the inverse permutation: rank_table[r] = index of r in
// dest_table. Useful for "at what step do I receive from peer p" lookups.
#pragma once

namespace liger_cute {
namespace detail {

// MoE typical sizes top out at 256 experts → 256 PEs; 512 leaves room.
constexpr int kMaxPEs = 512;

// __constant__ storage for the schedule tables, defined in nvshmem.cu. Declared
// extern here so any kernel that includes this header can read them through the
// inline accessor below without each TU re-defining the symbol. Resolved across
// TUs by RDC (CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS, set on
// the core target).
extern __constant__ int g_dest_table[kMaxPEs];
extern __constant__ int g_rank_table[kMaxPEs];

#ifdef __CUDACC__
// Position of `rank` in the dest_table (inverse permutation). Lets a PE compute
// "at what step does peer p target me" via
//   step = (g_rank_table[my_rank] - p + num_pes) % num_pes
// where num_pes is the kernel's team size (= N*M). A single broadcast-friendly
// constant-cache load.
__device__ __forceinline__ int comm_slot_of(int rank) {
  return g_rank_table[rank];
}
#endif  // __CUDACC__

// ─── Host-side setup (defined in nvshmem.cu) ─────────────────────────────────

// Pure (no-CUDA) computation of the comm schedule permutation, exposed for unit
// testing. Fills host_dest[0..N*M) (position->rank) and host_rank[0..N*M)
// (rank->position, the inverse). Caller-owned arrays of size >= N*M. Throws
// (liger_cute::Error) if N or M is non-positive. No kMaxPEs cap — that bounds
// only the device upload in init_comm_schedule.
void build_comm_schedule(int N, int M, int* host_dest, int* host_rank);

// Build the schedule for an (N hosts × M GPUs-per-host) layout and upload it to
// the g_dest_table / g_rank_table device constant memory. Collective wrt host
// setup — call once after nvshmem_init, with the same (N, M) on every PE.
// Throws (liger_cute::Error / CudaError) on a bad layout or a failed upload.
void init_comm_schedule(int N, int M);

}  // namespace detail
}  // namespace liger_cute
