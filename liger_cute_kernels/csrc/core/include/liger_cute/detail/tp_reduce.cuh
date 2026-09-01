#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace liger_cute {
namespace detail {

inline constexpr int kMaxTpReduceTeamSize = 512;

enum class LocalReduceBackend : std::uint8_t {
	kNvls,
	kDirectPeer,
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

// Host-only description of the optional follow-up remote reduction.
struct RemoteReduceView {
	float* reduced_shard;
	float* inbox;
	std::uint64_t* ready;
	std::uint64_t* consumed;
	int rank;
	int size;
	int peer_world;

	bool enabled() const {
		return peer_world >= 0;
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

struct TpReduceBuffers {
	float* partial;
	float* reduced;
	float* reduced_shard;
	float* remote_inbox;
	std::uint64_t* remote_signals;
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
// and selects the local backend. A two-host remote stage is represented
// separately in TpReducePlan::remote.
void configure_tp_reduce(
	std::int64_t parent_team,
	const TpReduceBuffers& buffers);

TpReduceTopology query_tp_reduce_topology(std::int64_t parent_team);
TpReducePlan tp_reduce_plan();

void begin_tp_reduce(
	const std::uint64_t* launch_epoch,
	cudaStream_t stream);
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
