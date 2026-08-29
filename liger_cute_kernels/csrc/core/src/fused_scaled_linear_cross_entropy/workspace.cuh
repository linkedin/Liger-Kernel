#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Backward workspace ownership.
//
// ── Allocator ownership ───────────────────────────────────────────────────
// Nothing here allocates memory itself: there is no local allocator and no raw
// nvshmem_malloc / nvshmem_free lifecycle anywhere in the backward path. Every
// buffer comes from liger::global_buffer_pool(), which is the shared MoE
// utility for fixed, reusable, internal staging that never escapes the
// operation.
//
//   name                allocator  symmetric?  why
//   -----------------   ---------  ----------  ---------------------------
//   kDxPartial          pool       yes   FP32 NVLS SUM source
//   kDxReduced          pool       yes   FP32 NVLS SUM destination
//   kDxSync             pool       yes   per-slot ready/completion epochs
//   kDzWorkspace        pool       no    one wave of dZ, BF16
//   kGridBarrier        pool       no    persistent-kernel phase counter
//
// liger::global_symmetric_stack() is deliberately NOT used. Nothing produced by
// the forward has to be carried to the backward as symmetric state: the
// forward's two all-reduces leave every rank holding identical NLL / LSE /
// entropy, so those are ordinary local CUDA outputs saved through autograd.
// Backward consumes them exactly like any other saved tensor.
//
// ── Collective configuration ──────────────────────────────────────────────
// configure_backward_tp_symmetric() must run on every PE with the same values
// before the first launch. It fixes the capacity of every allocation named
// here and resolves their aliases in the TP team's multicast mapping.
// Capacities are immutable afterwards: every later request asks for the
// *configured* capacity, never the per-call size, so the byte count handed to
// the pool is identical on every call and on every PE.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "backward_gemm_sm90.cuh"
#include "dx_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

// Centralized allocation names. Nothing else in the backward path may name a
// pooled buffer.
struct BackwardSymmetricNames {
	// global_buffer_pool() — symmetric (remotely addressed by NVSHMEM).
	static constexpr const char* kDxPartial =
		"fused_scaled_linear_cross_entropy_tp_dx_partial";
	static constexpr const char* kDxReduced =
		"fused_scaled_linear_cross_entropy_tp_dx_reduced";
	static constexpr const char* kDxHierarchical =
		"fused_scaled_linear_cross_entropy_tp_dx_hierarchical";
	static constexpr const char* kDxHierarchicalInbox =
		"fused_scaled_linear_cross_entropy_tp_dx_hierarchical_inbox";
	static constexpr const char* kDxHierarchicalSignals =
		"fused_scaled_linear_cross_entropy_tp_dx_hierarchical_signals";
	static constexpr const char* kDxSync =
		"fused_scaled_linear_cross_entropy_tp_dx_sync";

	// global_buffer_pool() — device private (never remotely addressed).
	static constexpr const char* kDzWorkspace =
		"fused_scaled_linear_cross_entropy_tp_dz_workspace";
	static constexpr const char* kGridBarrier =
		"fused_scaled_linear_cross_entropy_tp_grid_barrier";
	static constexpr const char* kDxPeerPartialPointers =
		"fused_scaled_linear_cross_entropy_tp_dx_peer_partial_ptrs";
	static constexpr const char* kDxPeerSyncPointers =
		"fused_scaled_linear_cross_entropy_tp_dx_peer_sync_ptrs";
	static constexpr const char* kDxPeerWorldPes =
		"fused_scaled_linear_cross_entropy_tp_dx_peer_world_pes";
};

// The immutable collective configuration.
struct BackwardTpCapacity {
	int max_local_vocab;
	int max_tiles_per_reduce;
	int max_comm_channels;
	int max_resident_ctas;
	int max_stages;
	int team_size;
	std::int64_t team_handle;
};

// Collective. Must be called on every PE with the same values before the first
// launch. CTA-owned NVLS staging is sized for full residency;
// `max_comm_channels` provisions duplicate teams for the accepted fused path.
void configure_backward_tp_symmetric(
	int max_local_vocab,
	int max_tiles_per_reduce,
	int max_comm_channels,
	std::int64_t team_handle);

// Symmetric footprint at the configured maximum. The fused path reuses the
// larger CTA-owned staging allocation, so channels add no buffer bytes.
std::size_t backward_tp_pool_symmetric_bytes(
	int max_tiles_per_reduce, int max_comm_channels);

// Total device-private pool footprint. CTA ring state is in shared memory.
std::size_t backward_tp_pool_device_bytes(int max_local_vocab);

// Collective: every PE of the configured team must call this with the same
// arguments. Returns local and multicast staging pointers plus TP metadata.
// Every pool request asks for the *configured* capacity,
// so the byte count is identical on every PE and on every call; only the
// descriptive fields of the returned workspace follow the launch grid.
//
// The knobs are runtime here on purpose: the compile-time TilesPerReduce /
// NumStages of the kernel template are validated against them in the launcher,
// which keeps a single allocation path for every instantiation.
DxReduceWorkspace<float> reserve_dx_reduce_workspace(
	int tiles_per_reduce, int num_stages, int num_ctas);

// Bytes in one of the two FP32 staging allocations at configured capacity.
std::size_t backward_dx_staging_bytes(int max_tiles_per_reduce);
std::size_t backward_dx_configured_staging_bytes();

// Full resident CTA capacity used by symmetric staging on the current device.
int backward_dx_resident_cta_capacity();

int backward_dx_team_size();

namespace fused_nvshmem {

DxReduceWorkspace<__nv_bfloat16> reserve_dx_reduce_workspace(
	int tiles_per_reduce, int num_stages, int num_comm_channels);

void reset_dx_reduce_signals(
	const DxReduceWorkspace<__nv_bfloat16>& workspace,
	cudaStream_t stream);

}  // namespace fused_nvshmem

// The pooled dZ wave workspace and grid-barrier counter for this shape.
struct BackwardScratch {
	void* dz_workspace;
	std::size_t dz_workspace_bytes;
	int* grid_barrier;
};

BackwardScratch reserve_backward_scratch(int local_vocab);

// Zeroes the grid-barrier counter. Must be enqueued before every launch.
void reset_backward_scratch(
	const BackwardScratch& scratch, cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
