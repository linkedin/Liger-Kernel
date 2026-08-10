// nvshmem.h — flat extern "C" ABI for NVSHMEM bootstrap, teams, and the comm schedule.
//
// Ported from LigerCommKernels' nvshmem_helpers.h, adapted to the liger_cute
// boundary contract (see liger_cute.h): every entry point returns a
// liger_cute_status_t, results come back through out-pointers, and the failure
// detail is read with liger_cute_last_error_string(). No exceptions cross the
// .so boundary; <nvshmem.h> is included only by the implementation (nvshmem.cu),
// never here — this header stays free of CUDA/NVSHMEM types so the binding TU
// can include it without pulling in the device runtime.
//
// Team handles are passed as int64 so Python (and any non-NVSHMEM TU) never
// needs an nvshmem_team_t. Translating a torch ProcessGroup to an NVSHMEM team
// is done host-side: split NVSHMEM_TEAM_WORLD with the group's (start, stride,
// size) via liger_cute_nvshmem_team_split_strided, then carry the returned
// handle into the MoE entry points.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "liger_cute/export.h"
#include "liger_cute/liger_cute.h"  // liger_cute_status_t, liger_cute_last_error_string

#ifdef __cplusplus
extern "C" {
#endif

// ─── Bootstrap / init ────────────────────────────────────────────────────────

// Number of bytes in an nvshmemx_uniqueid_t. Stable across a given NVSHMEM
// version; exposed so Python can allocate a CPU byte buffer of the right size
// before broadcasting via torch.distributed. Does not require NVSHMEM init.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_uniqueid_nbytes(size_t* out);

// Fill `out` (must be at least liger_cute_nvshmem_uniqueid_nbytes() bytes) with
// a fresh unique id. Call on rank 0 of the bootstrap group, then broadcast the
// bytes to peers.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_get_uniqueid(void* out);

// Initialise NVSHMEM using a previously broadcast unique id. `rank` / `nranks`
// describe the calling PE's position in the bootstrap group; `uid` must point to
// the same bytes on every rank.
LIGER_CUTE_API liger_cute_status_t
liger_cute_nvshmem_init_with_uniqueid(int rank, int nranks, const void* uid);

// Initialise NVSHMEM under the host launcher's PMI bootstrap (NVSHMEM_BOOTSTRAP
// = PMI/PMI-2 + srun/mpirun). Mirror of the static-inline nvshmem_init() from
// <host/nvshmem_api.h>; provided here so TUs that don't pull in <nvshmem.h>'s
// device helpers can still bootstrap NVSHMEM.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_init_pmi(void);

// Shutdown — drains the core's symmetric-memory caches (which own
// nvshmem_malloc'd allocations) BEFORE nvshmem_finalize(), so their singleton
// destructors don't run at program exit after the runtime is gone and abort in
// nvshmem_free. Mirror of nvshmem_finalize() otherwise.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_finalize(void);

// ─── PE queries (world) ──────────────────────────────────────────────────────
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_my_pe(int* out);
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_n_pes(int* out);

// ─── Teams ───────────────────────────────────────────────────────────────────

// NVSHMEM_TEAM_WORLD as an int64 handle.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_team_world(int64_t* out);

// Split a team strided (start + stride + size). Writes the new int64 team handle
// to *out, or -1 on ranks that aren't members of the new team.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_team_split_strided(
    int64_t parent_handle, int start, int stride, int size, int64_t* out);

// Destroy a team handle previously returned by split. NVSHMEM_TEAM_WORLD and
// invalid handles are ignored.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_team_destroy(int64_t team_handle);

// Per-team PE queries — useful for verifying that NVSHMEM's team-local PE
// numbering matches the torch process group's rank assignment.
LIGER_CUTE_API liger_cute_status_t
liger_cute_nvshmem_team_my_pe(int64_t team_handle, int* out);
LIGER_CUTE_API liger_cute_status_t
liger_cute_nvshmem_team_n_pes(int64_t team_handle, int* out);

// Translate a PE id from the source team to the destination team. Used by the
// kernel to map team-local → world PE before nvshmem_get/put. Writes the
// equivalent PE id in `dst_team_handle` to *out, or NVSHMEM_INVALID_PE (-1) if
// the PE is not a member of the destination team.
LIGER_CUTE_API liger_cute_status_t liger_cute_nvshmem_team_translate_pe(
    int64_t src_team_handle, int src_pe, int64_t dst_team_handle, int* out);

// ─── Communication schedule ──────────────────────────────────────────────────

// Build the schedule for an (N hosts × M GPUs-per-host) layout and upload it to
// device constant memory (g_dest_table / g_rank_table). Collective wrt host
// setup — call once after nvshmem init with the same (N, M) on every PE.
// (moe_configure_symmetric calls this internally; exposed for direct setup.)
// Fails with LIGER_CUTE_ERR_INVALID_ARGUMENT if N*M exceeds the schedule
// capacity, or LIGER_CUTE_ERR_CUDA on a failed upload.
LIGER_CUTE_API liger_cute_status_t liger_cute_init_comm_schedule(int N, int M);

// Pure (no-CUDA) computation of the comm schedule permutation, exposed for unit
// testing. Fills host_dest[0..N*M) (position->rank) and host_rank[0..N*M)
// (rank->position, the inverse). Caller-owned arrays of size >= N*M.
LIGER_CUTE_API liger_cute_status_t
liger_cute_build_comm_schedule(int N, int M, int* host_dest, int* host_rank);

// ─── Symmetric / device memory pools ─────────────────────────────────────────
//
// The core owns two process-wide caches that hold nvshmem_malloc'd allocations:
// the SymmetricMemoryStack (X/Y/offset buffers persisting fwd->bwd) and the
// BufferPool (per-name device + symmetric scratch). These shims let the torch
// layer drain them without including the device-header-pulling allocator.

// Drain BOTH caches (SymmetricMemoryStack + BufferPool). Collective wrt the
// symmetric frees — every PE must call it in the same order. Does NOT finalize
// NVSHMEM; use between runs to reset cached state. (finalize() calls this
// internally before tearing down the runtime.)
LIGER_CUTE_API liger_cute_status_t liger_cute_pool_clear_all(void);

// Drain ONLY the BufferPool (per-name device + symmetric scratch); leave the
// SymmetricMemoryStack intact. For dropping per-shape cached buffers between
// autotune sweeps without forcing a moe_configure_symmetric re-init.
LIGER_CUTE_API liger_cute_status_t liger_cute_pool_clear_buffers(void);

#ifdef __cplusplus
}  // extern "C"
#endif
