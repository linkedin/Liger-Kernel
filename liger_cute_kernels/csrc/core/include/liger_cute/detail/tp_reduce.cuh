#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace liger_cute {
namespace detail {

inline constexpr int kMaxTpReduceTeamSize = 512;
inline constexpr int kRemoteRingWarpSize = 32;
inline constexpr int kMaxRemoteRingWorkerWarpsPerBlock = 32;
inline constexpr unsigned int kRemoteRingFullWarpMask = 0xffffffffu;
inline constexpr int kRemoteRingInboxSlots = 2;
inline constexpr int kRemoteRingSignalSlots = 2;
inline constexpr int kRemoteRingSignalCount =
	2 * kRemoteRingSignalSlots;
inline constexpr std::uint64_t kRemoteRingStepMask = 0xffffu;
inline constexpr std::uint64_t kRemoteSumEpochSuffix = 0x30000000u;
inline constexpr std::uint64_t kForwardRemoteEpochSuffix = 0x40000000u;
static_assert(kRemoteRingInboxSlots == 2);
static_assert(
	(kRemoteRingWarpSize & (kRemoteRingWarpSize - 1)) == 0);
static_assert(kMaxTpReduceTeamSize - 1 <= kRemoteRingStepMask);
static_assert((kRemoteSumEpochSuffix & kRemoteRingStepMask) == 0);
static_assert((kForwardRemoteEpochSuffix & kRemoteRingStepMask) == 0);

template <int NumWorkerWarpsPerBlock>
__host__ __device__ constexpr int
remote_ring_worker_threads_per_block() {
	static_assert(
		NumWorkerWarpsPerBlock >= 1 &&
			NumWorkerWarpsPerBlock <=
				kMaxRemoteRingWorkerWarpsPerBlock);
	return NumWorkerWarpsPerBlock * kRemoteRingWarpSize;
}

template <int NumWorkerWarpsPerBlock>
__host__ __device__ constexpr bool remote_ring_uses_grid_sync() {
	return NumWorkerWarpsPerBlock > 1;
}

__host__ __device__ constexpr int remote_ring_global_worker_index(
		int block,
		int block_threads,
		int thread) {
	return block * block_threads + thread;
}

__host__ __device__ constexpr int remote_ring_global_worker_count(
		int grid_blocks,
		int block_threads) {
	return grid_blocks * block_threads;
}

__host__ __device__ constexpr int remote_ring_step_count(int team_size) {
	return team_size > 1 ? team_size - 1 : 0;
}

__host__ __device__ constexpr int remote_ring_slot(int step) {
	return step & (kRemoteRingInboxSlots - 1);
}

__host__ __device__ constexpr bool remote_ring_uses_consumed(
		int team_size) {
	return team_size > 2;
}

__host__ __device__ constexpr int remote_ring_transport_slot(
		int team_size, int step, int operation_sequence) {
	return team_size == 2
		? operation_sequence & (kRemoteRingInboxSlots - 1)
		: remote_ring_slot(step);
}

__host__ __device__ constexpr int remote_ring_reused_step(int step) {
	return step >= kRemoteRingInboxSlots
		? step - kRemoteRingInboxSlots
		: -1;
}

__host__ __device__ constexpr int remote_ring_forwarded_step(int step) {
	return step > 0 ? step - 1 : -1;
}

__host__ __device__ constexpr int remote_ring_drain_begin_step(
		int team_size) {
	int steps = remote_ring_step_count(team_size);
	return steps > kRemoteRingInboxSlots
		? steps - kRemoteRingInboxSlots
		: 0;
}

__host__ __device__ constexpr int remote_ring_last_step_for_slot(
		int team_size, int slot) {
	int steps = remote_ring_step_count(team_size);
	if (slot < 0 || slot >= kRemoteRingInboxSlots || slot >= steps) {
		return -1;
	}
	int last = steps - 1;
	return remote_ring_slot(last) == slot ? last : last - 1;
}

__host__ __device__ constexpr int remote_ring_previous_rank(
		int rank, int team_size) {
	return rank == 0 ? team_size - 1 : rank - 1;
}

__host__ __device__ constexpr int remote_ring_next_rank(
		int rank, int team_size) {
	return rank + 1 == team_size ? 0 : rank + 1;
}

__host__ __device__ constexpr std::uint64_t remote_ring_signal_value(
		std::uint64_t launch_epoch,
		std::uint64_t operation_suffix,
		int step) {
	return launch_epoch | operation_suffix |
		static_cast<std::uint64_t>(step + 1);
}

constexpr std::size_t remote_ring_inbox_bytes(
		std::size_t payload_bytes) {
	return kRemoteRingInboxSlots * payload_bytes;
}

constexpr std::size_t remote_ring_signal_bytes() {
	return kRemoteRingSignalCount * sizeof(std::uint64_t);
}

enum class LocalReduceBackend : std::uint8_t {
	kNvls,
	kDirectPeer,
};

enum class ReduceOp : std::uint8_t {
	kSum,
	kMax,
};

// Minimal device view for the local NVLS reduce-scatter/all-gather path.
struct NvlsReduceView {
	float* multicast_partial;
	float* multicast_reduced;
	std::uint64_t* multicast_sync;
	float* reduced_shard;
	int rank;
	int size;
};

// Minimal device view for the non-NVLS direct-peer all-reduce path.
struct DirectPeerReduceView {
	const float* const* peer_partial;
	std::uint64_t* const* peer_sync;
	int available;
	int rank;
	int size;
};

// Device view for the inter-host ring among matching node-local GPU ranks.
struct RemoteReduceView {
	float* reduced_shard;
	// Capacity may include several fused-forward source slots plus one
	// persistent running state; standalone dX/SM90 paths use its prefix.
	std::size_t reduced_shard_elements;
	float* inbox;
	std::uint64_t* ready;
	std::uint64_t* consumed;
	std::size_t inbox_slot_elements;
	int rank;
	int size;
	int previous_world;
	int next_world;
	int qp_handle;
	int team_handle;

	bool enabled() const {
		return reduced_shard != nullptr && inbox != nullptr &&
			ready != nullptr && consumed != nullptr &&
			reduced_shard_elements > 0 &&
			inbox_slot_elements > 0 &&
			size > 1 && rank >= 0 && rank < size &&
			previous_world >= 0 && next_world >= 0 &&
			team_handle >= 0;
	}
};

// Host dispatch plan. Only the selected backend-specific view is passed to a
// kernel; RequiresRemote remains a compile-time launcher parameter.
struct TpReducePlan {
	LocalReduceBackend backend;
	NvlsReduceView nvls;
	DirectPeerReduceView direct;
	RemoteReduceView remote;
	int team_size;
};

struct TpReduceTopology {
	int team_size;
	int local_size;
};

__host__ __device__ constexpr bool tp_reduce_uses_remote_ring(
		int team_size,
		int local_size,
		int remote_size) {
	return team_size > local_size &&
		local_size >= 1 &&
		remote_size > 1 &&
		local_size * remote_size == team_size;
}

__host__ __device__ constexpr int tp_reduce_host_rank(
		int team_rank, int local_size) {
	return team_rank / local_size;
}

__host__ __device__ constexpr int tp_reduce_local_rank(
		int team_rank, int local_size) {
	return team_rank % local_size;
}

__host__ __device__ constexpr int tp_reduce_parent_rank(
		int host_rank, int local_rank, int local_size) {
	return host_rank * local_size + local_rank;
}

struct TpReduceBuffers {
	float* partial;
	float* reduced;
	float* reduced_shard;
	std::size_t reduced_shard_bytes;
	float* remote_inbox;
	std::uint64_t* remote_signals;
	std::size_t remote_inbox_slot_bytes;
	std::uint64_t* sync;
	std::size_t sync_bytes;
	float** peer_partial_storage;
	std::uint64_t** peer_sync_storage;
};

#if defined(__CUDACC__)
__device__ __forceinline__ void publish_local_reduce_source() {
	asm volatile("fence.proxy.async.global;" ::: "memory");
	__threadfence_system();
}
#endif

// Collective across parent_team. Resolves NVLS and direct-peer mappings once
// and selects the local backend. An inter-host ring stage is represented
// separately in TpReducePlan::remote.
void configure_tp_reduce(
	std::int64_t parent_team,
	const TpReduceBuffers& buffers);

TpReduceTopology query_tp_reduce_topology(std::int64_t parent_team);
TpReducePlan tp_reduce_plan();

void begin_tp_reduce(
	const std::uint64_t* launch_epoch,
	cudaStream_t stream);
void synchronize_tp_reduce(cudaStream_t stream);
void end_tp_reduce(cudaStream_t stream);
void reset_tp_reduce();

// Separate RDC follow-up stage for the packed shard produced by the local
// reduction. This is intentionally outside the hot non-RDC WGMMA kernel.
void launch_remote_reduce(
	const RemoteReduceView& remote,
	const std::uint64_t* launch_epoch,
	std::size_t count,
	cudaStream_t stream);

}  // namespace detail
}  // namespace liger_cute
