#include "workspace.cuh"

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "buffer_pool.cuh"
#include "dx_reduce_sm90.cuh"
#include "liger_cute/check.h"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

using Config = BackwardGemmConfigSm90<90>;
using Launch = BackwardGemmLaunchSm90<90>;

// The ring depth is a compile-time constant shared with the kernel template
// (dx_reduce.cuh): the symmetric capacity and CTA-local mbarrier indexing have
// to match what the kernel uses.
constexpr int kConfiguredStages = kDxRingStages;

BackwardTpCapacity g_capacity = {};
bool g_configured = false;
std::uint64_t g_launch_epoch = 0;

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* name) {
	LIGER_CHECK(
		rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
		name,
		" size overflow");
	return lhs * rhs;
}

std::size_t staging_bytes_at(int tiles_per_reduce, int num_ctas) {
	std::size_t tile_elements = checked_multiply(
		static_cast<std::size_t>(Config::kTileM),
		static_cast<std::size_t>(Config::kDxTileN),
		"dX tile");
	std::size_t elements = checked_multiply(
		tile_elements,
		static_cast<std::size_t>(tiles_per_reduce),
		"dX group");
	elements = checked_multiply(
		elements, static_cast<std::size_t>(kConfiguredStages), "dX ring");
	elements = checked_multiply(
		elements,
		static_cast<std::size_t>(kDxCommWarpsPerChannel),
		"dX comm warps");
	elements = checked_multiply(
		elements, static_cast<std::size_t>(num_ctas), "dX resident CTAs");
	return checked_multiply(elements, sizeof(float), "dX staging");
}

std::size_t sync_bytes_at(int num_ctas, int team_size) {
	std::size_t entries = checked_multiply(
		static_cast<std::size_t>(num_ctas),
		static_cast<std::size_t>(kDxCommWarpsPerChannel),
		"dX sync CTAs");
	entries = checked_multiply(
		entries, static_cast<std::size_t>(kConfiguredStages), "dX sync ring");
	entries = checked_multiply(
		entries, static_cast<std::size_t>(kDxSyncPhases), "dX sync phases");
	entries = checked_multiply(
		entries, static_cast<std::size_t>(team_size), "dX sync team");
	return checked_multiply(entries, sizeof(std::uint64_t), "dX sync");
}

void ensure_configured() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric(max_local_vocab, "
		"max_tiles_per_reduce, max_comm_channels, team_handle) collectively on "
		"every PE before the first launch");
}

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy backward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

int resident_cta_capacity() {
	int device = 0;
	check_cuda(cudaGetDevice(&device), "cudaGetDevice");
	cudaDeviceProp properties = {};
	check_cuda(
		cudaGetDeviceProperties(&properties, device),
		"cudaGetDeviceProperties");
	LIGER_CHECK(
		properties.multiProcessorCount > 0 &&
			properties.multiProcessorCount <= kMaxDxResidentCtas,
		"fused_scaled_linear_cross_entropy backward: SM90 device reports ",
		properties.multiProcessorCount,
		" SMs, but the CTA-owned dX workspace supports at most ",
		kMaxDxResidentCtas);
	return properties.multiProcessorCount;
}

}  // namespace

void configure_backward_tp_symmetric(
		int max_local_vocab,
		int max_tiles_per_reduce,
		int max_comm_channels,
		std::int64_t team_handle) {
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	LIGER_CHECK(
		max_tiles_per_reduce >= 1,
		"max_tiles_per_reduce must be positive");
	LIGER_CHECK(
		max_comm_channels >= 1,
		"max_comm_channels must be positive");

	int max_resident_ctas = resident_cta_capacity();
	nvshmem_team_t team = static_cast<nvshmem_team_t>(team_handle);
	int team_size = nvshmem_team_n_pes(team);
	LIGER_CHECK(
		team_size >= 1 && team_size <= kMaxDxTeamSize,
		"fused_scaled_linear_cross_entropy backward: invalid tensor-parallel "
		"team size ",
		team_size);

	if (g_configured) {
		LIGER_CHECK(
			max_local_vocab <= g_capacity.max_local_vocab &&
				max_tiles_per_reduce <= g_capacity.max_tiles_per_reduce &&
				max_comm_channels <= g_capacity.max_comm_channels &&
				max_resident_ctas == g_capacity.max_resident_ctas &&
				team_size == g_capacity.team_size &&
				team_handle == g_capacity.team_handle,
			"fused_scaled_linear_cross_entropy backward: the symmetric "
			"capacity is immutable once allocated (configured for vocab ",
			g_capacity.max_local_vocab,
			", TilesPerReduce ",
			g_capacity.max_tiles_per_reduce,
			", communication channels ",
			g_capacity.max_comm_channels,
			", resident CTAs ",
			g_capacity.max_resident_ctas,
			"). Configure the maximum upfront.");
		return;
	}

	using Names = BackwardSymmetricNames;
	auto& pool = global_buffer_pool();

	// One fixed order on every PE: symmetric first, then device private.
	std::size_t staging = staging_bytes_at(
		max_tiles_per_reduce, max_resident_ctas);
	auto* partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, staging));
	auto* reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, staging));
	auto* hierarchical = static_cast<float*>(
		pool.get_symmetric(Names::kDxHierarchical, staging));
	auto* hierarchical_inbox = static_cast<float*>(
		pool.get_symmetric(Names::kDxHierarchicalInbox, staging));
	auto* hierarchical_signals = static_cast<std::uint64_t*>(
		pool.get_symmetric(
			Names::kDxHierarchicalSignals,
			2 * sizeof(std::uint64_t)));
	std::size_t sync_bytes = sync_bytes_at(max_resident_ctas, team_size);
	auto* sync = static_cast<std::uint64_t*>(
		pool.get_symmetric(Names::kDxSync, sync_bytes));
	auto** peer_partial_storage = static_cast<float**>(
		pool.get_device(
			Names::kDxPeerPartialPointers,
			static_cast<std::size_t>(team_size) * sizeof(float*)));
	auto** peer_sync_storage = static_cast<std::uint64_t**>(
		pool.get_device(
			Names::kDxPeerSyncPointers,
			static_cast<std::size_t>(team_size) *
				sizeof(std::uint64_t*)));
	auto* world_pe_storage = static_cast<int*>(
		pool.get_device(
			Names::kDxPeerWorldPes,
			static_cast<std::size_t>(team_size) * sizeof(int)));

	pool.get_device(
		Names::kDzWorkspace, Launch::dz_workspace_bytes(max_local_vocab));
	pool.get_device(Names::kGridBarrier, sizeof(int));

	configure_dx_nvls_mapping(
		team_handle,
		partial,
		reduced,
		hierarchical,
		hierarchical_inbox,
		hierarchical_signals,
		sync,
		sync_bytes,
		peer_partial_storage,
		peer_sync_storage,
		world_pe_storage);
	if (dx_nvls_available()) {
		fused_nvshmem::configure_dx_comm_channels(
			max_comm_channels, team_handle);
	}

	g_capacity.max_local_vocab = max_local_vocab;
	g_capacity.max_tiles_per_reduce = max_tiles_per_reduce;
	g_capacity.max_comm_channels = max_comm_channels;
	g_capacity.max_resident_ctas = max_resident_ctas;
	g_capacity.max_stages = kConfiguredStages;
	g_capacity.team_size = team_size;
	g_capacity.team_handle = team_handle;
	g_configured = true;
}

std::size_t backward_tp_pool_symmetric_bytes(
		int max_tiles_per_reduce, int max_comm_channels) {
	LIGER_CHECK(
		max_comm_channels >= 1,
		"max_comm_channels must be positive");
	int max_ctas =
		g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
	int team_size = g_configured ? g_capacity.team_size : kMaxDxTeamSize;
	return 4u * staging_bytes_at(max_tiles_per_reduce, max_ctas) +
		sync_bytes_at(max_ctas, team_size) +
		2 * sizeof(std::uint64_t);
}

std::size_t backward_tp_pool_device_bytes(int max_local_vocab) {
	int team_size = g_configured ? g_capacity.team_size : kMaxDxTeamSize;
	std::size_t peer_mapping =
		static_cast<std::size_t>(team_size) *
		(2 * sizeof(void*) + sizeof(int));
	return Launch::dz_workspace_bytes(max_local_vocab) + sizeof(int) +
		peer_mapping;
}

DxReduceWorkspace<float> reserve_dx_reduce_workspace(
		int tiles_per_reduce,
		int num_stages,
		int num_ctas) {
	ensure_configured();
	LIGER_CHECK(
		tiles_per_reduce >= 1 &&
			tiles_per_reduce <= g_capacity.max_tiles_per_reduce,
		"fused_scaled_linear_cross_entropy backward: TilesPerReduce ",
		tiles_per_reduce,
		" exceeds the configured maximum ",
		g_capacity.max_tiles_per_reduce);
	LIGER_CHECK(
		num_stages == g_capacity.max_stages,
		"fused_scaled_linear_cross_entropy backward: the staging ring depth ",
		num_stages,
		" must equal the configured ",
		g_capacity.max_stages,
		"; CTA-local mbarriers use the compile-time depth");
	LIGER_CHECK(
		num_ctas >= 1 && num_ctas <= g_capacity.max_resident_ctas,
		"fused_scaled_linear_cross_entropy backward: launch grid ",
		num_ctas,
		" exceeds resident CTA capacity ",
		g_capacity.max_resident_ctas);
	using Names = BackwardSymmetricNames;
	auto& pool = global_buffer_pool();
	// Always the configured capacity, never the per-call size.
	std::size_t staging = staging_bytes_at(
		g_capacity.max_tiles_per_reduce, g_capacity.max_resident_ctas);
	std::size_t sync_bytes = sync_bytes_at(
		g_capacity.max_resident_ctas, g_capacity.team_size);
	DxNvlsMapping mapping = dx_nvls_mapping();
	LIGER_CHECK(
		g_launch_epoch < std::numeric_limits<std::uint32_t>::max(),
		"fused_scaled_linear_cross_entropy backward: NVLS launch epoch "
		"exhausted");
	std::uint64_t epoch_base = (++g_launch_epoch) << 32;

	DxReduceWorkspace<float> workspace = {};
	workspace.partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, staging));
	workspace.reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, staging));
	workspace.multicast_partial = mapping.multicast_partial;
	workspace.multicast_reduced = mapping.multicast_reduced;
	workspace.sync = static_cast<std::uint64_t*>(
		pool.get_symmetric(Names::kDxSync, sync_bytes));
	workspace.multicast_sync = mapping.multicast_sync;
	workspace.epoch_base = epoch_base;
	workspace.team_rank = mapping.team_rank;
	workspace.team_size = mapping.team_size;
	return workspace;
}

std::size_t backward_dx_staging_bytes(int max_tiles_per_reduce) {
	int max_ctas =
		g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
	return staging_bytes_at(max_tiles_per_reduce, max_ctas);
}

std::size_t backward_dx_configured_staging_bytes() {
	ensure_configured();
	return staging_bytes_at(
		g_capacity.max_tiles_per_reduce, g_capacity.max_resident_ctas);
}

int backward_dx_resident_cta_capacity() {
	return g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
}

int backward_dx_team_size() {
	ensure_configured();
	return g_capacity.team_size;
}

namespace fused_nvshmem {

DxReduceWorkspace<__nv_bfloat16> reserve_dx_reduce_workspace(
		int tiles_per_reduce,
		int num_stages,
		int num_comm_channels) {
	auto storage =
		::liger::fused_scaled_linear_cross_entropy::
			reserve_dx_reduce_workspace(
				tiles_per_reduce, num_stages, num_comm_channels);

	DxReduceWorkspace<__nv_bfloat16> workspace = {};
	workspace.partial =
		reinterpret_cast<__nv_bfloat16*>(storage.partial);
	workspace.reduced =
		reinterpret_cast<__nv_bfloat16*>(storage.reduced);
	workspace.ready_tiles = reinterpret_cast<int*>(storage.sync);
	workspace.done_groups =
		workspace.ready_tiles + num_comm_channels * num_stages;
	workspace.num_channels = num_comm_channels;
	workspace.num_stages = num_stages;
	workspace.tiles_per_reduce = tiles_per_reduce;
	workspace.tile_elements = Config::kTileM * Config::kDxTileN;
	workspace.tp_rank = dx_comm_team_rank();
	workspace.tp_size = dx_comm_team_size();
	workspace.teams = dx_comm_teams(num_comm_channels);
	return workspace;
}

void reset_dx_reduce_signals(
		const DxReduceWorkspace<__nv_bfloat16>& workspace,
		cudaStream_t stream) {
	std::size_t counters =
		static_cast<std::size_t>(workspace.num_channels) *
		(static_cast<std::size_t>(workspace.num_stages) + 1u);
	check_cuda(
		cudaMemsetAsync(
			workspace.ready_tiles, 0, counters * sizeof(int), stream),
		"cudaMemsetAsync(fused dX signals)");
}

}  // namespace fused_nvshmem

BackwardScratch reserve_backward_scratch(int local_vocab) {
	ensure_configured();
	LIGER_CHECK(
		local_vocab > 0 && local_vocab <= g_capacity.max_local_vocab,
		"fused_scaled_linear_cross_entropy backward: local_vocab ",
		local_vocab,
		" exceeds the configured maximum ",
		g_capacity.max_local_vocab);

	using Names = BackwardSymmetricNames;
	auto& pool = global_buffer_pool();
	std::size_t dz_bytes =
		Launch::dz_workspace_bytes(g_capacity.max_local_vocab);

	BackwardScratch scratch = {};
	scratch.dz_workspace = pool.get_device(Names::kDzWorkspace, dz_bytes);
	scratch.dz_workspace_bytes = dz_bytes;
	scratch.grid_barrier =
		static_cast<int*>(pool.get_device(Names::kGridBarrier, sizeof(int)));
	return scratch;
}

void reset_backward_scratch(
		const BackwardScratch& scratch, cudaStream_t stream) {
	LIGER_CHECK(
		scratch.grid_barrier != nullptr,
		"the grid barrier counter must be allocated before reset");
	check_cuda(
		cudaMemsetAsync(scratch.grid_barrier, 0, sizeof(int), stream),
		"cudaMemsetAsync(grid barrier)");
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
