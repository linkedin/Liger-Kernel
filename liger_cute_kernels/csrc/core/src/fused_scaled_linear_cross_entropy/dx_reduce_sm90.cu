// One-time NVLS multicast-pointer setup for the fused backward dX reduction.

#include "dx_reduce_sm90.cuh"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "liger_cute/check.h"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

DxNvlsMapping g_mapping = {};
DxFallbackMapping g_fallback_mapping = {};
DxHierarchicalMapping g_hierarchical_mapping = {};
bool g_nvls_available = false;
bool g_hierarchical_available = false;
bool g_configured = false;
std::int64_t g_parent_team = 0;
float* g_partial = nullptr;
float* g_reduced = nullptr;
float* g_hierarchical = nullptr;
float* g_hierarchical_inbox = nullptr;
std::uint64_t* g_hierarchical_signals = nullptr;
std::uint64_t* g_sync = nullptr;
std::size_t g_sync_bytes = 0;
float** g_peer_partial_storage = nullptr;
std::uint64_t** g_peer_sync_storage = nullptr;
int* g_world_pe_storage = nullptr;

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy backward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

__global__ void query_dx_nvls_mapping(
		nvshmem_team_t team,
		float* partial,
		float* reduced,
		std::uint64_t* sync,
		DxNvlsMapping* output) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	output->multicast_partial =
		static_cast<float*>(nvshmemx_mc_ptr(team, partial));
	output->multicast_reduced =
		static_cast<float*>(nvshmemx_mc_ptr(team, reduced));
	output->multicast_sync =
		static_cast<std::uint64_t*>(nvshmemx_mc_ptr(team, sync));
	output->local_multicast_partial =
		static_cast<float*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, partial));
	output->local_multicast_reduced =
		static_cast<float*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, reduced));
	output->local_multicast_sync =
		static_cast<std::uint64_t*>(
			nvshmemx_mc_ptr(NVSHMEMX_TEAM_NODE, sync));
	output->team_rank = nvshmem_team_my_pe(team);
	output->team_size = nvshmem_team_n_pes(team);
	output->local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
	output->local_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
	output->interhost_rank =
		nvshmem_team_my_pe(NVSHMEMX_TEAM_SAME_MYPE_NODE);
	output->interhost_size =
		nvshmem_team_n_pes(NVSHMEMX_TEAM_SAME_MYPE_NODE);
}

__global__ void advance_dx_launch_epoch(std::uint64_t* launch_epoch) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*launch_epoch += std::uint64_t{1} << 32;
	}
}

}  // namespace

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
		int* world_pe_storage) {
	LIGER_CHECK(
		partial != nullptr && reduced != nullptr &&
			hierarchical != nullptr && hierarchical_inbox != nullptr &&
			hierarchical_signals != nullptr && sync != nullptr,
		"fused_scaled_linear_cross_entropy backward: symmetric reduction "
		"pointers must be non-null");
	LIGER_CHECK(sync_bytes > 0, "NVLS signal storage must be non-empty");
	LIGER_CHECK(
		peer_partial_storage != nullptr && peer_sync_storage != nullptr &&
			world_pe_storage != nullptr,
		"fused_scaled_linear_cross_entropy backward: fallback peer mapping "
		"storage must be non-null");

	if (g_configured) {
		LIGER_CHECK(
			parent_team == g_parent_team && partial == g_partial &&
				reduced == g_reduced &&
				hierarchical == g_hierarchical &&
				hierarchical_inbox == g_hierarchical_inbox &&
				hierarchical_signals == g_hierarchical_signals &&
				sync == g_sync &&
				sync_bytes == g_sync_bytes &&
				peer_partial_storage == g_peer_partial_storage &&
				peer_sync_storage == g_peer_sync_storage &&
				world_pe_storage == g_world_pe_storage,
			"fused_scaled_linear_cross_entropy backward: the reduction mapping "
			"is immutable once configured");
		return;
	}

	nvshmem_team_t team = static_cast<nvshmem_team_t>(parent_team);
	int team_size = nvshmem_team_n_pes(team);
	LIGER_CHECK(
		team_size >= 1 && team_size <= kMaxDxTeamSize,
		"fused_scaled_linear_cross_entropy backward: invalid tensor-parallel "
		"team size ",
		team_size);

	check_cuda(cudaMemset(sync, 0, sync_bytes), "cudaMemset(NVLS signals)");
	check_cuda(
		cudaMemset(
			hierarchical_signals, 0, 2 * sizeof(std::uint64_t)),
		"cudaMemset(hierarchical signals)");

	DxNvlsMapping* device_mapping = nullptr;
	check_cuda(
		cudaMalloc(&device_mapping, sizeof(DxNvlsMapping)),
		"cudaMalloc(NVLS mapping)");
	query_dx_nvls_mapping<<<1, 1>>>(
		team, partial, reduced, sync, device_mapping);
	check_cuda(cudaGetLastError(), "query_dx_nvls_mapping launch");
	check_cuda(
		cudaMemcpy(
			&g_mapping,
			device_mapping,
			sizeof(DxNvlsMapping),
			cudaMemcpyDeviceToHost),
		"cudaMemcpy(NVLS mapping)");
	check_cuda(cudaFree(device_mapping), "cudaFree(NVLS mapping)");

	LIGER_CHECK(
		g_mapping.team_rank >= 0 &&
			g_mapping.team_rank < g_mapping.team_size &&
			g_mapping.team_size == team_size,
		"fused_scaled_linear_cross_entropy backward: inconsistent NVLS team "
		"metadata");
	g_nvls_available =
		team_size == 1 ||
		(g_mapping.multicast_partial != nullptr &&
			g_mapping.multicast_reduced != nullptr &&
			g_mapping.multicast_sync != nullptr);

	std::vector<float*> peer_partial(team_size);
	std::vector<std::uint64_t*> peer_sync(team_size);
	std::vector<int> world_pes(team_size);
	std::vector<unsigned char> world_members(
		static_cast<std::size_t>(nvshmem_n_pes()), 0);
	bool all_peers_direct = true;
	bool parent_covers_world = team_size == nvshmem_n_pes();
	int my_world_pe = nvshmem_my_pe();
	for (int rank = 0; rank < team_size; ++rank) {
		int world_pe = nvshmem_team_translate_pe(
			team, rank, NVSHMEM_TEAM_WORLD);
		LIGER_CHECK(
			world_pe >= 0,
			"fused_scaled_linear_cross_entropy backward: failed to translate "
			"team rank ",
			rank,
			" to NVSHMEM_TEAM_WORLD");
		if (world_pe >= static_cast<int>(world_members.size()) ||
			world_members[world_pe] != 0) {
			parent_covers_world = false;
		} else {
			world_members[world_pe] = 1;
		}
		world_pes[rank] = world_pe;
		peer_partial[rank] = world_pe == my_world_pe
			? partial
			: static_cast<float*>(nvshmem_ptr(partial, world_pe));
		peer_sync[rank] = world_pe == my_world_pe
			? sync
			: static_cast<std::uint64_t*>(nvshmem_ptr(sync, world_pe));
		all_peers_direct =
			all_peers_direct && peer_partial[rank] != nullptr &&
			peer_sync[rank] != nullptr;
	}
	g_hierarchical_available =
		!g_nvls_available && parent_covers_world &&
		g_mapping.local_size > 1 &&
		g_mapping.interhost_size == 2 &&
		g_mapping.local_size * g_mapping.interhost_size == team_size &&
		g_mapping.local_multicast_partial != nullptr &&
		g_mapping.local_multicast_reduced != nullptr &&
		g_mapping.local_multicast_sync != nullptr;
	check_cuda(
		cudaMemcpy(
			peer_partial_storage,
			peer_partial.data(),
			peer_partial.size() * sizeof(float*),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(fallback partial peer pointers)");
	check_cuda(
		cudaMemcpy(
			peer_sync_storage,
			peer_sync.data(),
			peer_sync.size() * sizeof(std::uint64_t*),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(fallback sync peer pointers)");
	check_cuda(
		cudaMemcpy(
			world_pe_storage,
			world_pes.data(),
			world_pes.size() * sizeof(int),
			cudaMemcpyHostToDevice),
		"cudaMemcpy(fallback world ranks)");

	g_fallback_mapping.peer_partial = peer_partial_storage;
	g_fallback_mapping.peer_sync = peer_sync_storage;
	g_fallback_mapping.world_pes = world_pe_storage;
	g_fallback_mapping.all_peers_direct =
		static_cast<int>(all_peers_direct);
	g_fallback_mapping.team_rank = g_mapping.team_rank;
	g_fallback_mapping.team_size = g_mapping.team_size;
	g_hierarchical_mapping.local_multicast_partial =
		g_mapping.local_multicast_partial;
	g_hierarchical_mapping.local_multicast_reduced =
		g_mapping.local_multicast_reduced;
	g_hierarchical_mapping.local_multicast_sync =
		g_mapping.local_multicast_sync;
	g_hierarchical_mapping.interhost = hierarchical;
	g_hierarchical_mapping.inbox = hierarchical_inbox;
	g_hierarchical_mapping.signals = hierarchical_signals;
	g_hierarchical_mapping.local_rank = g_mapping.local_rank;
	g_hierarchical_mapping.local_size = g_mapping.local_size;
	g_hierarchical_mapping.interhost_rank = g_mapping.interhost_rank;
	g_hierarchical_mapping.interhost_size = g_mapping.interhost_size;
	if (g_hierarchical_available) {
		int peer_rank = 1 - g_mapping.interhost_rank;
		g_hierarchical_mapping.interhost_peer_world =
			nvshmem_team_translate_pe(
				NVSHMEMX_TEAM_SAME_MYPE_NODE,
				peer_rank,
				NVSHMEM_TEAM_WORLD);
	} else {
		g_hierarchical_mapping.interhost_peer_world = -1;
	}

	// Every PE must finish clearing its local signal replicas before any fused
	// launch can publish a multicast epoch.
	int barrier_status = nvshmem_barrier(team);
	LIGER_CHECK(
		barrier_status == 0,
		"fused_scaled_linear_cross_entropy backward: NVLS setup barrier "
		"failed with status ",
		barrier_status);

	g_parent_team = parent_team;
	g_partial = partial;
	g_reduced = reduced;
	g_hierarchical = hierarchical;
	g_hierarchical_inbox = hierarchical_inbox;
	g_hierarchical_signals = hierarchical_signals;
	g_sync = sync;
	g_sync_bytes = sync_bytes;
	g_peer_partial_storage = peer_partial_storage;
	g_peer_sync_storage = peer_sync_storage;
	g_world_pe_storage = world_pe_storage;
	g_configured = true;
}

void prepare_dx_reduce_launch(
		const DxReduceWorkspace<float>& workspace,
		cudaStream_t stream) {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	LIGER_CHECK(
		workspace.launch_epoch != nullptr,
		"fused_scaled_linear_cross_entropy backward: launch epoch is null");
	advance_dx_launch_epoch<<<1, 1, 0, stream>>>(
		const_cast<std::uint64_t*>(workspace.launch_epoch));
	check_cuda(cudaGetLastError(), "advance_dx_launch_epoch launch");
}

void complete_dx_reduce_launch(cudaStream_t stream) {
	if (g_mapping.team_size > 1) {
		// One tail barrier keeps the staging and signal generations from being
		// reused until every peer has completed the current launch. It is
		// stream ordered and captured into CUDA Graphs.
		nvshmemx_barrier_on_stream(
			static_cast<nvshmem_team_t>(g_parent_team), stream);
	}
}

DxNvlsMapping dx_nvls_mapping() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	return g_mapping;
}

DxFallbackMapping dx_fallback_mapping() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	return g_fallback_mapping;
}

DxHierarchicalMapping dx_local_mapping() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	DxHierarchicalMapping mapping = g_hierarchical_mapping;
	if (g_nvls_available) {
		mapping.local_rank = g_mapping.team_rank;
		mapping.local_size = g_mapping.team_size;
		if (mapping.local_size == 1) {
			mapping.local_multicast_partial = g_partial;
			mapping.local_multicast_reduced = g_reduced;
			mapping.local_multicast_sync = g_sync;
		} else {
			mapping.local_multicast_partial = g_mapping.multicast_partial;
			mapping.local_multicast_reduced = g_mapping.multicast_reduced;
			mapping.local_multicast_sync = g_mapping.multicast_sync;
		}
		return mapping;
	}
	LIGER_CHECK(
		g_hierarchical_mapping.local_multicast_partial != nullptr &&
			g_hierarchical_mapping.local_multicast_reduced != nullptr &&
			g_hierarchical_mapping.local_multicast_sync != nullptr,
		"fused_scaled_linear_cross_entropy backward: local NVLS mapping "
		"is unavailable");
	return mapping;
}

DxHierarchicalMapping dx_hierarchical_mapping() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	LIGER_CHECK(
		g_hierarchical_available,
		"fused_scaled_linear_cross_entropy backward: hierarchical "
		"NVLS+remote mapping is unavailable");
	return g_hierarchical_mapping;
}

bool dx_nvls_available() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	return g_nvls_available;
}

bool dx_hierarchical_available() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric() before the first launch");
	return g_hierarchical_available;
}

void reset_dx_nvls_mapping() {
	g_mapping = {};
	g_fallback_mapping = {};
	g_hierarchical_mapping = {};
	g_nvls_available = false;
	g_hierarchical_available = false;
	g_configured = false;
	g_parent_team = 0;
	g_partial = nullptr;
	g_reduced = nullptr;
	g_hierarchical = nullptr;
	g_hierarchical_inbox = nullptr;
	g_hierarchical_signals = nullptr;
	g_sync = nullptr;
	g_sync_bytes = 0;
	g_peer_partial_storage = nullptr;
	g_peer_sync_storage = nullptr;
	g_world_pe_storage = nullptr;
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
