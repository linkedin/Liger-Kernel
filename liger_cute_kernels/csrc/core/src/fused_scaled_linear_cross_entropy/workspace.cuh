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
//   kDxReducedShard     pool       yes   dX shard or SM100 forward wave slots
//   kDxRemoteInbox      pool       yes   two inter-host ring receive slots
//   kDxRemoteSignals    pool       yes   ready/consumed signal pairs
//   kDxSync             pool       yes   per-slot ready/completion epochs
//   kDzWorkspace        pool       no    one wave of dZ, BF16
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

#if LIGER_CUTE_DISPATCH_COMPUTE == 100
#include "backward_gemm_sm100.cuh"
#else
#include "backward_gemm_sm90.cuh"
#endif
#include "dx_reduce.cuh"
#include "state.h"

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
	static constexpr const char* kDxReducedShard =
		"fused_scaled_linear_cross_entropy_tp_dx_reduced_shard";
	static constexpr const char* kDxRemoteInbox =
		"fused_scaled_linear_cross_entropy_tp_dx_remote_inbox";
	static constexpr const char* kDxRemoteSignals =
		"fused_scaled_linear_cross_entropy_tp_dx_remote_signals";
	static constexpr const char* kDxSync =
		"fused_scaled_linear_cross_entropy_tp_dx_sync";

	// global_buffer_pool() — device private (never remotely addressed).
	static constexpr const char* kDzWorkspace =
		"fused_scaled_linear_cross_entropy_tp_dz_workspace";
	static constexpr const char* kDxPeerPartialPointers =
		"fused_scaled_linear_cross_entropy_tp_dx_peer_partial_ptrs";
	static constexpr const char* kDxPeerSyncPointers =
		"fused_scaled_linear_cross_entropy_tp_dx_peer_sync_ptrs";
	static constexpr const char* kDxLaunchEpoch =
		"fused_scaled_linear_cross_entropy_tp_dx_launch_epoch";
	// Fixed device-private signal block for the fused SM100 backward: the
	// monotonic full-grid barrier counter, the per-wave dX reduce-scatter
	// completion counter and the inter-host ring completion epoch.
	static constexpr const char* kBackwardSm100Signals =
		"fused_scaled_linear_cross_entropy_tp_backward_sm100_signals";
	static constexpr const char* kBackwardSm100DzTileReady =
		"fused_scaled_linear_cross_entropy_tp_backward_sm100_dz_tile_ready";
	static constexpr const char* kBackwardSm100Diagnostics =
		"fused_scaled_linear_cross_entropy_tp_backward_sm100_diagnostics";
};

// The immutable collective configuration.
struct BackwardTpCapacity {
	int max_tokens;
	int max_hidden;
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
// `max_comm_channels` is retained for API compatibility; the CTA-owned
// production path does not allocate per-channel teams or buffers.
void configure_backward_tp_symmetric(
	int max_tokens,
	int max_hidden,
	int max_local_vocab,
	int max_tiles_per_reduce,
	int max_comm_channels,
	std::int64_t team_handle);

// Before configuration, returns a conservative topology-independent estimate.
// After configuration, the arguments must exactly match the immutable capacity
// and the exact configured symmetric footprint is returned. The fused path
// reuses the larger CTA-owned staging allocation, so channels add no bytes.
std::size_t backward_tp_pool_symmetric_bytes(
	int max_tokens,
	int max_hidden,
	int max_tiles_per_reduce,
	int max_comm_channels);

// Before configuration, returns a conservative device-private estimate. After
// configuration, max_local_vocab must exactly match the immutable capacity.
// CTA ring state is in shared memory.
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
std::size_t backward_dx_configured_durable_bytes();
std::size_t backward_dx_configured_packed_durable_bytes();
std::size_t tp_reduced_shard_configured_bytes();
std::size_t tp_remote_inbox_slot_configured_bytes();

void validate_backward_tp_shape(
	int tokens, int hidden, int local_vocab);

// Full resident CTA capacity used by symmetric staging on the current device.
int backward_dx_resident_cta_capacity();

int backward_dx_team_size();
std::int64_t backward_dx_team_handle();
int backward_tp_max_local_vocab();

// The pooled dZ wave workspace for this shape.
struct BackwardScratch {
	void* dz_workspace;
	std::size_t dz_workspace_bytes;
};

BackwardScratch reserve_backward_scratch(int local_vocab);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
