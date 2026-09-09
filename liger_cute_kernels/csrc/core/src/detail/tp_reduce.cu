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
		lhs.remote_inbox == rhs.remote_inbox &&
		lhs.remote_signals == rhs.remote_signals &&
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
			2 * sizeof(std::uint64_t)),
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
	bool direct_available = true;
	bool parent_covers_world = team_size == nvshmem_n_pes();
	int my_world_pe = nvshmem_my_pe();
	for (int rank = 0; rank < team_size; ++rank) {
		int world_pe = nvshmem_team_translate_pe(
			team, rank, NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			world_pe >= 0,
			"failed to translate tensor-parallel rank ",
			rank,
			" to NVSHMEM_TEAM_WORLD");
		if (world_pe >= static_cast<int>(world_members.size()) ||
			world_members[world_pe] != 0) {
			parent_covers_world = false;
		} else {
			world_members[world_pe] = 1;
		}
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
	bool remote_available =
		!nvls_available && parent_covers_world &&
		g_raw.node_size > 1 &&
		g_raw.remote_size == 2 &&
		g_raw.node_size * g_raw.remote_size == team_size &&
		g_raw.node_multicast_partial != nullptr &&
		g_raw.node_multicast_reduced != nullptr &&
		g_raw.node_multicast_sync != nullptr;

	g_plan = {};
	g_plan.remote.peer_world = -1;
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
		int peer_rank = 1 - g_raw.remote_rank;
		int peer_world = nvshmem_team_translate_pe(
			NVSHMEMX_TEAM_SAME_MYPE_NODE,
			peer_rank,
			NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			peer_world >= 0,
			"failed to translate remote reduction peer to world rank");
		g_plan.remote = {
			buffers.reduced_shard,
			buffers.remote_inbox,
			buffers.remote_signals,
			buffers.remote_signals + 1,
			g_raw.remote_rank,
			g_raw.remote_size,
			peer_world};
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
		remote.size == 2 && remote.rank >= 0 && remote.rank < remote.size,
		"invalid remote reduction topology");
	launch_remote_pair_all_reduce(
		remote.reduced_shard,
		remote.inbox,
		remote.reduced_shard,
		count,
		remote.ready,
		remote.consumed,
		launch_epoch,
		0x30000000u,
		remote.peer_world,
		stream);
}

void reset_tp_reduce() {
	g_raw = {};
	g_plan = {};
	g_plan.remote.peer_world = -1;
	g_buffers = {};
	g_configured = false;
	g_parent_team = 0;
}

}  // namespace detail
}  // namespace liger_cute
