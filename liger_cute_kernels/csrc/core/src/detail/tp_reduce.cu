#include "liger_cute/detail/tp_reduce.cuh"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "remote_all_reduce.cuh"
#include "liger_cute/check.h"

namespace liger_cute {
namespace detail {
namespace {

struct RawNvlsMapping {
	float* local_multicast_partial;
	float* local_multicast_reduced;
	std::uint64_t* local_multicast_sync;
	int team_rank;
	int team_size;
	int local_rank;
	int local_size;
	int remote_rank;
	int remote_size;
};

struct TpReduceContext {
	RawNvlsMapping raw = {};
	TpReducePlan plan = {};
	TpReduceBuffers buffers = {};
	bool configured = false;
	nvshmem_team_t local_team = NVSHMEM_TEAM_INVALID;
	nvshmem_team_t remote_team = NVSHMEM_TEAM_INVALID;
	bool owns_hierarchical_teams = false;
};

std::map<std::int64_t, TpReduceContext> g_contexts;
std::vector<std::int64_t> g_creation_order;
thread_local std::int64_t g_selected_team = -1;

TpReduceContext& selected_context() {
	auto found = g_contexts.find(tp_reduce_team_handle());
	LIGER_CHECK(
		found != g_contexts.end() && found->second.configured,
		"tensor-parallel reduction context is not configured");
	return found->second;
}

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
		nvshmem_team_t local_team,
		nvshmem_team_t remote_team,
		float* partial,
		float* reduced,
		std::uint64_t* sync,
		RawNvlsMapping* output) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	output->local_multicast_partial =
		static_cast<float*>(nvshmemx_mc_ptr(local_team, partial));
	output->local_multicast_reduced =
		static_cast<float*>(nvshmemx_mc_ptr(local_team, reduced));
	output->local_multicast_sync =
		static_cast<std::uint64_t*>(nvshmemx_mc_ptr(local_team, sync));
	output->team_rank = nvshmem_team_my_pe(team);
	output->team_size = nvshmem_team_n_pes(team);
	output->local_rank = nvshmem_team_my_pe(local_team);
	output->local_size = nvshmem_team_n_pes(local_team);
	if (remote_team == NVSHMEM_TEAM_INVALID) {
		output->remote_rank = 0;
		output->remote_size = 1;
	} else {
		output->remote_rank = nvshmem_team_my_pe(remote_team);
		output->remote_size = nvshmem_team_n_pes(remote_team);
	}
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

int count_parent_members_on_node(nvshmem_team_t team) {
	int team_size = nvshmem_team_n_pes(team);
	int local_size = 0;
	for (int rank = 0; rank < team_size; ++rank) {
		local_size += nvshmem_team_translate_pe(
			team, rank, NVSHMEMX_TEAM_NODE) >= 0;
	}
	return local_size;
}

void parent_team_min_max(
		nvshmem_team_t team,
		std::uint64_t* symmetric_scratch,
		int value,
		int& minimum,
		int& maximum) {
	int* scratch = reinterpret_cast<int*>(symmetric_scratch);
	check_cuda(
		cudaMemcpy(
			scratch,
			&value,
			sizeof(value),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(topology consensus source)");
	int min_status =
		nvshmem_int_min_reduce(team, scratch + 1, scratch, 1);
	LIGER_CHECK(
		min_status == 0,
		"tensor-parallel topology minimum reduction failed with status ",
		min_status);
	int max_status =
		nvshmem_int_max_reduce(team, scratch + 2, scratch, 1);
	LIGER_CHECK(
		max_status == 0,
		"tensor-parallel topology maximum reduction failed with status ",
		max_status);
	check_cuda(
		cudaMemcpy(
			&minimum,
			scratch + 1,
			sizeof(minimum),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(topology minimum)");
	check_cuda(
		cudaMemcpy(
			&maximum,
			scratch + 2,
			sizeof(maximum),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(topology maximum)");
}

bool parent_team_rows_are_node_local(
		nvshmem_team_t team, int local_size) {
	int team_rank = nvshmem_team_my_pe(team);
	int local_begin =
		tp_reduce_host_rank(team_rank, local_size) * local_size;
	int team_size = nvshmem_team_n_pes(team);
	for (int rank = 0; rank < team_size; ++rank) {
		bool same_node = nvshmem_team_translate_pe(
			team, rank, NVSHMEMX_TEAM_NODE) >= 0;
		bool same_row =
			rank >= local_begin && rank < local_begin + local_size;
		if (same_node != same_row) return false;
	}
	return true;
}

}  // namespace

TpReduceTopology query_tp_reduce_topology(std::int64_t parent_team) {
	nvshmem_team_t team = static_cast<nvshmem_team_t>(parent_team);
	int team_size = nvshmem_team_n_pes(team);
	LIGER_CHECK(
		team_size >= 1 && team_size <= kMaxTpReduceTeamSize,
		"invalid tensor-parallel reduction team size ",
		team_size);
	int local_size = count_parent_members_on_node(team);
	LIGER_CHECK(
		local_size >= 1 && local_size <= team_size,
		"invalid number of node-local members in tensor-parallel team: ",
		local_size,
		" of ",
		team_size);
	return {
		team_size,
		local_size,
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
		buffers.sync_bytes >= 3 * sizeof(int),
		"tensor-parallel reduction signal storage must contain topology "
		"consensus scratch");
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

	auto inserted = g_contexts.try_emplace(parent_team);
	if (inserted.second) g_creation_order.push_back(parent_team);
	auto& context = inserted.first->second;
	if (context.configured) {
		LIGER_CHECK(
			same_buffers(buffers, context.buffers),
			"tensor-parallel reduction mapping is immutable once configured");
		return;
	}

	nvshmem_team_t team = static_cast<nvshmem_team_t>(parent_team);
	TpReduceTopology topology = query_tp_reduce_topology(parent_team);
	int team_size = topology.team_size;
	int local_size = topology.local_size;

	check_cuda(
		cudaMemset(buffers.sync, 0, buffers.sync_bytes),
		"cudaMemset(reduction signals)");
	check_cuda(
		cudaMemset(
			buffers.remote_signals,
			0,
			remote_ring_signal_bytes()),
		"cudaMemset(remote reduction signals)");

	int minimum_local_size = 0;
	int maximum_local_size = 0;
	parent_team_min_max(
		team,
		buffers.sync,
		local_size,
		minimum_local_size,
		maximum_local_size);
	LIGER_CHECK(
		minimum_local_size == maximum_local_size,
		"cross-host tensor-parallel teams must select the same number of "
		"GPUs on every host, got a range of ",
		minimum_local_size,
		"..",
		maximum_local_size);
	local_size = minimum_local_size;
	LIGER_CHECK(
		team_size % local_size == 0,
		"tensor-parallel team size ",
		team_size,
		" is not divisible by its uniform local size ",
		local_size);

	bool parent_spans_nodes = local_size < team_size;
	context.local_team = team;
	context.remote_team = NVSHMEM_TEAM_INVALID;
	context.owns_hierarchical_teams = false;
	if (parent_spans_nodes) {
		int local_layout_valid =
			parent_team_rows_are_node_local(team, local_size) ? 1 : 0;
		int minimum_layout_valid = 0;
		int maximum_layout_valid = 0;
		parent_team_min_max(
			team,
			buffers.sync,
			local_layout_valid,
			minimum_layout_valid,
			maximum_layout_valid);
		LIGER_CHECK(
			minimum_layout_valid == 1 &&
				maximum_layout_valid == 1,
			"cross-host tensor-parallel team ranks must be ordered as "
			"contiguous, equally sized host rows");

		nvshmem_team_config_t local_config = {};
		nvshmem_team_config_t remote_config = {};
		int split_status = nvshmem_team_split_2d(
			team,
			local_size,
			&local_config,
			0,
			&context.local_team,
			&remote_config,
			0,
			&context.remote_team);
		LIGER_CHECK(
			split_status == 0 &&
				context.local_team != NVSHMEM_TEAM_INVALID &&
				context.remote_team != NVSHMEM_TEAM_INVALID,
			"failed to split tensor-parallel team into parent-relative "
			"local and matching-rank remote teams (status ",
			split_status,
			")");
		context.owns_hierarchical_teams = true;
	}

	RawNvlsMapping* device_mapping = nullptr;
	check_cuda(
		cudaMalloc(&device_mapping, sizeof(RawNvlsMapping)),
		"cudaMalloc(reduction mapping)");
	query_nvls_mapping<<<1, 1>>>(
		team,
		context.local_team,
		context.remote_team,
		buffers.partial,
		buffers.reduced,
		buffers.sync,
		device_mapping);
	check_cuda(cudaGetLastError(), "query_nvls_mapping launch");
	check_cuda(
		cudaMemcpy(
			&context.raw,
			device_mapping,
			sizeof(RawNvlsMapping),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(reduction mapping)");
	check_cuda(cudaFree(device_mapping), "cudaFree(reduction mapping)");

	LIGER_CHECK(
		context.raw.team_rank >= 0 &&
			context.raw.team_rank < context.raw.team_size &&
			context.raw.team_size == team_size &&
			context.raw.local_rank >= 0 &&
			context.raw.local_rank < context.raw.local_size &&
			context.raw.local_size == local_size,
		"inconsistent tensor-parallel reduction team metadata");

	std::vector<float*> peer_partial(team_size);
	std::vector<std::uint64_t*> peer_sync(team_size);
	bool direct_available = true;
	int my_world_pe = nvshmem_my_pe();
	for (int rank = 0; rank < team_size; ++rank) {
		int world_pe = nvshmem_team_translate_pe(
			team, rank, NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			world_pe >= 0 &&
				world_pe < nvshmem_n_pes(),
			"failed to translate tensor-parallel rank ",
			rank,
			" to NVSHMEM_TEAM_WORLD");
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
	bool local_nvls_available =
		local_size == 1 ||
		(context.raw.local_multicast_partial != nullptr &&
			context.raw.local_multicast_reduced != nullptr &&
			context.raw.local_multicast_sync != nullptr);
	bool remote_topology_valid =
		tp_reduce_uses_remote_ring(
			team_size,
			context.raw.local_size,
			context.raw.remote_size) &&
		context.raw.remote_rank >= 0 &&
		context.raw.remote_rank < context.raw.remote_size;
	if (parent_spans_nodes) {
		LIGER_CHECK(
			local_nvls_available && remote_topology_valid,
			"cross-host tensor-parallel reduction requires uniform, "
			"parent-relative node rows with local NVLS mappings and "
			"matching-rank remote teams (local size ",
			context.raw.local_size,
			", remote size ",
			context.raw.remote_size,
			", TP size ",
			team_size,
			")");
	}
	bool remote_available =
		parent_spans_nodes && remote_topology_valid;
	bool nvls_available =
		!parent_spans_nodes && local_nvls_available;

	context.plan = {};
	context.plan.remote.previous_world = -1;
	context.plan.remote.next_world = -1;
	context.plan.remote.team_handle = -1;
	context.plan.team_size = context.raw.team_size;
	context.plan.direct = {
		buffers.peer_partial_storage,
		buffers.peer_sync_storage,
		static_cast<int>(direct_available),
		context.raw.team_rank,
		context.raw.team_size};
	// Direct-peer all-reduce produces a full tile on each rank, so its
	// downstream packed/scatter layout is the local-size-one layout.
	context.plan.nvls = {
		buffers.partial, buffers.reduced, buffers.sync,
		buffers.reduced_shard, 0, 1};

	if (nvls_available || remote_available) {
		context.plan.backend = LocalReduceBackend::kNvls;
		context.plan.nvls = {
			local_size == 1
				? buffers.partial
				: context.raw.local_multicast_partial,
			local_size == 1
				? buffers.reduced
				: context.raw.local_multicast_reduced,
			local_size == 1
				? buffers.sync
				: context.raw.local_multicast_sync,
			buffers.reduced_shard,
			context.raw.local_rank,
			context.raw.local_size};
	}
	if (remote_available) {
		int previous_rank = remote_ring_previous_rank(
			context.raw.remote_rank, context.raw.remote_size);
		int next_rank = remote_ring_next_rank(
			context.raw.remote_rank, context.raw.remote_size);
		int previous_world = nvshmem_team_translate_pe(
			context.remote_team,
			previous_rank,
			NVSHMEM_TEAM_WORLD);
		int next_world = nvshmem_team_translate_pe(
			context.remote_team,
			next_rank,
			NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			previous_world >= 0 && next_world >= 0,
			"failed to translate remote ring neighbors to world ranks");
		context.plan.remote = {
			buffers.reduced_shard,
			buffers.reduced_shard_bytes / sizeof(float),
			buffers.remote_inbox,
			buffers.remote_signals,
			buffers.remote_signals + kRemoteRingSignalSlots,
			buffers.remote_inbox_slot_bytes / sizeof(float),
			context.raw.remote_rank,
			context.raw.remote_size,
			previous_world,
			next_world,
			static_cast<int>(NVSHMEMX_QP_DEFAULT),
			static_cast<int>(context.remote_team)};
	} else if (!nvls_available) {
		context.plan.backend = LocalReduceBackend::kDirectPeer;
	}

	check_cuda(
		cudaMemset(buffers.sync, 0, buffers.sync_bytes),
		"cudaMemset(reduction signals after topology consensus)");
	int barrier_status = nvshmem_barrier(team);
	LIGER_CHECK(
		barrier_status == 0,
		"tensor-parallel reduction setup barrier failed with status ",
		barrier_status);

	context.buffers = buffers;
	context.configured = true;
}

std::int64_t tp_reduce_team_handle() {
	if (g_selected_team >= 0) return g_selected_team;
	LIGER_CHECK(
		g_contexts.size() == 1 && g_contexts.begin()->second.configured,
		"select a prepared tensor-parallel team for this call");
	return g_contexts.begin()->first;
}

TpReduceContextScope::TpReduceContextScope(std::int64_t parent_team)
		: previous_(g_selected_team) {
	if (parent_team < 0) parent_team = tp_reduce_team_handle();
	auto found = g_contexts.find(parent_team);
	LIGER_CHECK(
		found != g_contexts.end() && found->second.configured,
		"tensor-parallel team ", parent_team,
		" is not prepared; configure its context collectively before execution");
	g_selected_team = parent_team;
}

TpReduceContextScope::~TpReduceContextScope() {
	g_selected_team = previous_;
}

TpReducePlan tp_reduce_plan() {
	return selected_context().plan;
}

void begin_tp_reduce(
		const std::uint64_t* launch_epoch,
		cudaStream_t stream) {
	(void)selected_context();
	LIGER_CHECK(launch_epoch != nullptr, "launch epoch is null");
	advance_launch_epoch<<<1, 1, 0, stream>>>(
		const_cast<std::uint64_t*>(launch_epoch));
	check_cuda(cudaGetLastError(), "advance_launch_epoch launch");
	// A remote peer may still be using the shared inbox for another TP team.
	// All members must finish their prior work before this team starts puts.
	if (selected_context().plan.remote.enabled()) synchronize_tp_reduce(stream);
}

void synchronize_tp_reduce(cudaStream_t stream) {
	if (selected_context().plan.team_size > 1) {
		nvshmemx_barrier_on_stream(
			static_cast<nvshmem_team_t>(tp_reduce_team_handle()), stream);
	}
}

void end_tp_reduce(cudaStream_t stream) {
	synchronize_tp_reduce(stream);
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

void release_tp_reduce(std::int64_t parent_team) {
	auto found = g_contexts.find(parent_team);
	if (found == g_contexts.end()) return;
	auto& context = found->second;
	if (context.owns_hierarchical_teams) {
		nvshmem_team_destroy(context.remote_team);
		nvshmem_team_destroy(context.local_team);
	}
	g_contexts.erase(found);
	g_creation_order.erase(
		std::remove(g_creation_order.begin(), g_creation_order.end(), parent_team),
		g_creation_order.end());
}

void reset_tp_reduce() {
	while (!g_creation_order.empty()) release_tp_reduce(g_creation_order.front());
	g_selected_team = -1;
}

}  // namespace detail
}  // namespace liger_cute
