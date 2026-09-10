#include "liger_cute/detail/tp_reduce.cuh"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "remote_all_reduce.cuh"
#include "liger_cute/check.h"

namespace liger_cute {
namespace detail {
namespace {

struct RawNvlsMapping {
	float* multicast_partial;
	float* multicast_reduced;
	std::uint64_t* multicast_sync;
	float* node_multicast_partial;
	float* node_multicast_reduced;
	std::uint64_t* node_multicast_sync;
	int team_rank;
	int team_size;
	int node_rank;
	int node_size;
	int remote_rank;
	int remote_size;
};

RawNvlsMapping g_raw = {};
TpReducePlan g_plan = {};
TpReduceBuffers g_buffers = {};
bool g_configured = false;
std::int64_t g_parent_team = 0;

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"tensor-parallel reduction: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

__global__ void query_nvls_mapping(
		nvshmem_team_t team,
		float* partial,
		float* reduced,
		std::uint64_t* sync,
		RawNvlsMapping* output) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	output->multicast_partial =
		static_cast<float*>(nvshmemx_mc_ptr(team, partial));
	output->multicast_reduced =
		static_cast<float*>(nvshmemx_mc_ptr(team, reduced));
	output->multicast_sync =
		static_cast<std::uint64_t*>(nvshmemx_mc_ptr(team, sync));
	output->node_multicast_partial =
		static_cast<float*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, partial));
	output->node_multicast_reduced =
		static_cast<float*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, reduced));
	output->node_multicast_sync =
		static_cast<std::uint64_t*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, sync));
	output->team_rank = nvshmem_team_my_pe(team);
	output->team_size = nvshmem_team_n_pes(team);
	output->node_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
	output->node_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
	output->remote_rank =
		nvshmem_team_my_pe(NVSHMEMX_TEAM_SAME_MYPE_NODE);
	output->remote_size =
		nvshmem_team_n_pes(NVSHMEMX_TEAM_SAME_MYPE_NODE);
}

__global__ void advance_launch_epoch(std::uint64_t* launch_epoch) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*launch_epoch += std::uint64_t{1} << 32;
	}
}

bool same_buffers(const TpReduceBuffers& lhs, const TpReduceBuffers& rhs) {
	return lhs.partial == rhs.partial &&
		lhs.reduced == rhs.reduced &&
		lhs.reduced_shard == rhs.reduced_shard &&
		lhs.reduced_shard_bytes == rhs.reduced_shard_bytes &&
		lhs.remote_inbox == rhs.remote_inbox &&
		lhs.remote_signals == rhs.remote_signals &&
		lhs.remote_inbox_slot_bytes == rhs.remote_inbox_slot_bytes &&
		lhs.sync == rhs.sync &&
		lhs.sync_bytes == rhs.sync_bytes &&
		lhs.peer_partial_storage == rhs.peer_partial_storage &&
		lhs.peer_sync_storage == rhs.peer_sync_storage;
}

}  // namespace

TpReduceTopology query_tp_reduce_topology(std::int64_t parent_team) {
	nvshmem_team_t team = static_cast<nvshmem_team_t>(parent_team);
	int team_size = nvshmem_team_n_pes(team);
	int node_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
	LIGER_CHECK(
		team_size >= 1 && team_size <= kMaxTpReduceTeamSize,
		"invalid tensor-parallel reduction team size ",
		team_size);
	LIGER_CHECK(node_size >= 1, "invalid NVSHMEM node-team size");
	return {
		team_size,
		team_size < node_size ? team_size : node_size,
	};
}

void configure_tp_reduce(
		std::int64_t parent_team,
		const TpReduceBuffers& buffers) {
	LIGER_CHECK(
		buffers.partial != nullptr && buffers.reduced != nullptr &&
			buffers.reduced_shard != nullptr &&
			buffers.remote_inbox != nullptr &&
			buffers.remote_signals != nullptr &&
			buffers.sync != nullptr,
		"tensor-parallel reduction buffers must be non-null");
	LIGER_CHECK(
		buffers.sync_bytes > 0,
		"tensor-parallel reduction signal storage must be non-empty");
	LIGER_CHECK(
		buffers.reduced_shard_bytes >= sizeof(float),
		"tensor-parallel reduced shard must contain float values");
	LIGER_CHECK(
		buffers.remote_inbox_slot_bytes >= sizeof(float),
		"tensor-parallel remote inbox slot must contain float values");
	LIGER_CHECK(
		buffers.peer_partial_storage != nullptr &&
			buffers.peer_sync_storage != nullptr,
		"tensor-parallel direct-peer mapping storage must be non-null");

	if (g_configured) {
		LIGER_CHECK(
			parent_team == g_parent_team &&
				same_buffers(buffers, g_buffers),
			"tensor-parallel reduction mapping is immutable once configured");
		return;
	}

	nvshmem_team_t team = static_cast<nvshmem_team_t>(parent_team);
	int team_size = query_tp_reduce_topology(parent_team).team_size;

	check_cuda(
		cudaMemset(buffers.sync, 0, buffers.sync_bytes),
		"cudaMemset(reduction signals)");
	check_cuda(
		cudaMemset(
			buffers.remote_signals,
			0,
			remote_ring_signal_bytes()),
		"cudaMemset(remote reduction signals)");

	RawNvlsMapping* device_mapping = nullptr;
	check_cuda(
		cudaMalloc(&device_mapping, sizeof(RawNvlsMapping)),
		"cudaMalloc(reduction mapping)");
	query_nvls_mapping<<<1, 1>>>(
		team,
		buffers.partial,
		buffers.reduced,
		buffers.sync,
		device_mapping);
	check_cuda(cudaGetLastError(), "query_nvls_mapping launch");
	check_cuda(
		cudaMemcpy(
			&g_raw,
			device_mapping,
			sizeof(RawNvlsMapping),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(reduction mapping)");
	check_cuda(cudaFree(device_mapping), "cudaFree(reduction mapping)");

	LIGER_CHECK(
		g_raw.team_rank >= 0 &&
			g_raw.team_rank < g_raw.team_size &&
			g_raw.team_size == team_size,
		"inconsistent tensor-parallel reduction team metadata");

	std::vector<float*> peer_partial(team_size);
	std::vector<std::uint64_t*> peer_sync(team_size);
	std::vector<unsigned char> world_members(
		static_cast<std::size_t>(nvshmem_n_pes()), 0);
	std::vector<unsigned char> node_world_members(
		static_cast<std::size_t>(nvshmem_n_pes()), 0);
	for (int rank = 0; rank < g_raw.node_size; ++rank) {
		int world_pe = nvshmem_team_translate_pe(
			NVSHMEMX_TEAM_NODE, rank, NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			world_pe >= 0 &&
				world_pe < static_cast<int>(node_world_members.size()),
			"failed to translate node-local rank ",
			rank,
			" to NVSHMEM_TEAM_WORLD");
		node_world_members[world_pe] = 1;
	}
	bool direct_available = true;
	bool parent_covers_world = team_size == nvshmem_n_pes();
	bool parent_spans_nodes = false;
	int my_world_pe = nvshmem_my_pe();
	for (int rank = 0; rank < team_size; ++rank) {
		int world_pe = nvshmem_team_translate_pe(
			team, rank, NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			world_pe >= 0 &&
				world_pe < static_cast<int>(world_members.size()),
			"failed to translate tensor-parallel rank ",
			rank,
			" to NVSHMEM_TEAM_WORLD");
		if (world_members[world_pe] != 0) {
			parent_covers_world = false;
		} else {
			world_members[world_pe] = 1;
		}
		parent_spans_nodes =
			parent_spans_nodes || node_world_members[world_pe] == 0;
		peer_partial[rank] = world_pe == my_world_pe
			? buffers.partial
			: static_cast<float*>(
				nvshmem_ptr(buffers.partial, world_pe));
		peer_sync[rank] = world_pe == my_world_pe
			? buffers.sync
			: static_cast<std::uint64_t*>(
				nvshmem_ptr(buffers.sync, world_pe));
		direct_available =
			direct_available && peer_partial[rank] != nullptr &&
			peer_sync[rank] != nullptr;
	}
	check_cuda(
		cudaMemcpy(
			buffers.peer_partial_storage,
			peer_partial.data(),
			peer_partial.size() * sizeof(float*),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(direct-peer partial pointers)");
	check_cuda(
		cudaMemcpy(
			buffers.peer_sync_storage,
			peer_sync.data(),
			peer_sync.size() * sizeof(std::uint64_t*),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(direct-peer signal pointers)");
	bool nvls_available =
		team_size == 1 ||
		(g_raw.multicast_partial != nullptr &&
			g_raw.multicast_reduced != nullptr &&
			g_raw.multicast_sync != nullptr);
	bool remote_topology_valid =
		tp_reduce_uses_remote_ring(
			team_size,
			g_raw.node_size,
			g_raw.remote_size,
			parent_covers_world) &&
		g_raw.node_rank >= 0 &&
		g_raw.node_rank < g_raw.node_size &&
		g_raw.remote_rank >= 0 &&
		g_raw.remote_rank < g_raw.remote_size &&
		g_raw.node_multicast_partial != nullptr &&
		g_raw.node_multicast_reduced != nullptr &&
		g_raw.node_multicast_sync != nullptr;
	if (!nvls_available && parent_spans_nodes) {
		LIGER_CHECK(
			remote_topology_valid,
			"cross-host tensor-parallel reduction requires a world-covering "
			"team, uniform node sizes, node-local NVLS mappings, and matching-"
			"rank remote teams (node size ",
			g_raw.node_size,
			", remote size ",
			g_raw.remote_size,
			", TP size ",
			team_size,
			")");
	}
	bool remote_available =
		!nvls_available && parent_spans_nodes &&
		remote_topology_valid;

	g_plan = {};
	g_plan.remote.previous_world = -1;
	g_plan.remote.next_world = -1;
	g_plan.team_size = g_raw.team_size;
	g_plan.direct = {
		buffers.peer_partial_storage,
		buffers.peer_sync_storage,
		static_cast<int>(direct_available),
		g_raw.team_rank,
		g_raw.team_size};

	if (nvls_available) {
		g_plan.backend = LocalReduceBackend::kNvls;
		g_plan.nvls = {
			team_size == 1 ? buffers.partial : g_raw.multicast_partial,
			team_size == 1 ? buffers.reduced : g_raw.multicast_reduced,
			team_size == 1 ? buffers.sync : g_raw.multicast_sync,
			buffers.reduced_shard,
			g_raw.team_rank,
			g_raw.team_size};
	} else if (remote_available) {
		g_plan.backend = LocalReduceBackend::kNvls;
		g_plan.nvls = {
			g_raw.node_multicast_partial,
			g_raw.node_multicast_reduced,
			g_raw.node_multicast_sync,
			buffers.reduced_shard,
			g_raw.node_rank,
			g_raw.node_size};
		int previous_rank = remote_ring_previous_rank(
			g_raw.remote_rank, g_raw.remote_size);
		int next_rank = remote_ring_next_rank(
			g_raw.remote_rank, g_raw.remote_size);
		int previous_world = nvshmem_team_translate_pe(
			NVSHMEMX_TEAM_SAME_MYPE_NODE,
			previous_rank,
			NVSHMEM_TEAM_WORLD);
		int next_world = nvshmem_team_translate_pe(
			NVSHMEMX_TEAM_SAME_MYPE_NODE,
			next_rank,
			NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			previous_world >= 0 && next_world >= 0,
			"failed to translate remote ring neighbors to world ranks");
		g_plan.remote = {
			buffers.reduced_shard,
			buffers.reduced_shard_bytes / sizeof(float),
			buffers.remote_inbox,
			buffers.remote_signals,
			buffers.remote_signals + kRemoteRingSignalSlots,
			buffers.remote_inbox_slot_bytes / sizeof(float),
			g_raw.remote_rank,
			g_raw.remote_size,
			previous_world,
			next_world,
			static_cast<int>(NVSHMEMX_QP_DEFAULT)};
	} else {
		g_plan.backend = LocalReduceBackend::kDirectPeer;
	}

	int barrier_status = nvshmem_barrier(team);
	LIGER_CHECK(
		barrier_status == 0,
		"tensor-parallel reduction setup barrier failed with status ",
		barrier_status);

	g_parent_team = parent_team;
	g_buffers = buffers;
	g_configured = true;
}

TpReducePlan tp_reduce_plan() {
	LIGER_CHECK(
		g_configured,
		"configure_tp_reduce() must be called before requesting a plan");
	return g_plan;
}

void begin_tp_reduce(
		const std::uint64_t* launch_epoch,
		cudaStream_t stream) {
	LIGER_CHECK(g_configured, "tensor-parallel reduction is not configured");
	LIGER_CHECK(launch_epoch != nullptr, "launch epoch is null");
	advance_launch_epoch<<<1, 1, 0, stream>>>(
		const_cast<std::uint64_t*>(launch_epoch));
	check_cuda(cudaGetLastError(), "advance_launch_epoch launch");
}

void end_tp_reduce(cudaStream_t stream) {
	LIGER_CHECK(g_configured, "tensor-parallel reduction is not configured");
	if (g_plan.team_size > 1) {
		nvshmemx_barrier_on_stream(
			static_cast<nvshmem_team_t>(g_parent_team), stream);
	}
}

void launch_remote_reduce(
		const RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		std::size_t count,
		cudaStream_t stream) {
	LIGER_CHECK(remote.enabled(), "remote reduction is not configured");
	LIGER_CHECK(
		remote.size > 1 && remote.rank >= 0 && remote.rank < remote.size,
		"invalid remote reduction topology");
	LIGER_CHECK(
		count <= remote.inbox_slot_elements &&
			count <= remote.reduced_shard_elements,
		"remote reduction payload exceeds its symmetric buffers");
	launch_remote_ring_all_reduce(
		remote,
		remote.reduced_shard,
		remote.reduced_shard,
		count,
		launch_epoch,
		kRemoteSumEpochSuffix,
		stream);
}

void reset_tp_reduce() {
	g_raw = {};
	g_plan = {};
	g_plan.remote.previous_world = -1;
	g_plan.remote.next_world = -1;
	g_buffers = {};
	g_configured = false;
	g_parent_team = 0;
}

}  // namespace detail
}  // namespace liger_cute
