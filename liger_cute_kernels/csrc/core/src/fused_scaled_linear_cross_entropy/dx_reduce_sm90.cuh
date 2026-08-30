#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy — dX reduction support.
//
// The fused kernel itself uses the NVSHMEM-free primitives in
// liger_cute/detail/nvls.cuh. This header owns only the one-time RDC setup that
// resolves ordinary symmetric pointers into multicast aliases.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "dx_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

// ───────────────────────────────────────────────────────────────────────────
// TMA store completion helpers used by the dZ and dW epilogues.
// ───────────────────────────────────────────────────────────────────────────

// cp.async.bulk.wait_group N, always the full form and never `.read`.
//
// `.read` only waits for the SMEM source of a bulk store to be re-usable, and
// it *retires the group from the pending set*. A staging pipeline that drains
// with `.read` and then asks for `wait_group 0` therefore gets no guarantee at
// all about the already-retired groups' global writes. Every tile here is
// published to another warp the moment it is stored, so the epilogue has to
// wait for completion, not merely for the source to be free.
template <int Pending>
__device__ __forceinline__ void dx_wait_tma_store() {
	asm volatile("cp.async.bulk.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ void dx_wait_tma_store_complete() {
	dx_wait_tma_store<0>();
}

// ───────────────────────────────────────────────────────────────────────────
// Host-side multicast mapping (implemented in dx_reduce_sm90.cu)
// ───────────────────────────────────────────────────────────────────────────

struct DxNvlsMapping {
	float* multicast_partial;
	float* multicast_reduced;
	std::uint64_t* multicast_sync;
	float* local_multicast_partial;
	float* local_multicast_reduced;
	std::uint64_t* local_multicast_sync;
	int team_rank;
	int team_size;
	int local_rank;
	int local_size;
	int interhost_rank;
	int interhost_size;
};

struct DxFallbackMapping {
	const float* const* peer_partial;
	std::uint64_t* const* peer_sync;
	const int* world_pes;
	int all_peers_direct;
	int team_rank;
	int team_size;
};

struct DxHierarchicalMapping {
	float* local_multicast_partial;
	float* local_multicast_reduced;
	std::uint64_t* local_multicast_sync;
	float* interhost;
	float* inbox;
	std::uint64_t* signals;
	int local_rank;
	int local_size;
	int interhost_rank;
	int interhost_size;
	int interhost_peer_world;
};

// Collective across parent_team. Resolves the team's multicast aliases when
// available, records direct peer mappings for the non-NVLS path, and clears
// the signal storage.
void configure_dx_nvls_mapping(
	std::int64_t parent_team,
	float* partial,
	float* reduced,
	float* hierarchical,
	float* hierarchical_inbox,
	std::uint64_t* hierarchical_signals,
	std::uint64_t* sync,
	std::size_t sync_bytes,
	float** peer_partial_storage,
	std::uint64_t** peer_sync_storage,
	int* world_pe_storage);

void prepare_dx_reduce_launch(
	const DxReduceWorkspace<float>& workspace,
	cudaStream_t stream);
void complete_dx_reduce_launch(cudaStream_t stream);
void reset_dx_nvls_mapping();

DxNvlsMapping dx_nvls_mapping();
DxFallbackMapping dx_fallback_mapping();
DxHierarchicalMapping dx_local_mapping();
DxHierarchicalMapping dx_hierarchical_mapping();
bool dx_nvls_available();
bool dx_hierarchical_available();

namespace fused_nvshmem {

__device__ __forceinline__ void dx_wait_tma_store_complete() {
	asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void dx_wait_tma_store_read() {
	asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void dx_publish_fence() {
	asm volatile("fence.proxy.async.global;" ::: "memory");
	__threadfence_system();
}

__device__ __forceinline__ void dx_acquire_fence() {
	__threadfence_system();
}

}  // namespace fused_nvshmem

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
