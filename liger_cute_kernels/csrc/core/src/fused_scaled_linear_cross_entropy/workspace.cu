#include "workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>

#include "buffer_pool.cuh"
#include "forward_reduce.cuh"
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
#include "forward_gemm_sm100.cuh"
#endif
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

#if LIGER_CUTE_DISPATCH_COMPUTE == 100
using Config = BackwardGemmConfigSm100<100>;
using Launch = BackwardGemmLaunchSm100<100>;
#else
using Config = BackwardGemmConfigSm90<90>;
using Launch = BackwardGemmLaunchSm90<90>;
#endif

// The ring depth is a compile-time constant shared with the kernel template
// (dx_reduce.cuh): the symmetric capacity and CTA-local mbarrier indexing have
// to match what the kernel uses.
constexpr int kConfiguredStages = kDxRingStages;

BackwardTpCapacity g_capacity = {};
bool g_configured = false;
std::size_t g_staging_bytes = 0;
std::size_t g_durable_bytes = 0;
std::size_t g_reduced_bytes = 0;
std::size_t g_reduced_shard_bytes = 0;
std::size_t g_remote_payload_bytes = 0;
std::size_t g_sync_bytes = 0;
struct BackwardTpContext {
	std::int64_t slot = 0;
	std::int64_t team_handle = -1;
	int team_size = 0;
	std::size_t packed_durable_bytes = 0;
};

std::map<std::int64_t, BackwardTpContext> g_slots;
std::map<std::int64_t, BackwardTpContext*> g_team_contexts;

BackwardTpContext& selected_context() {
	auto found = g_team_contexts.find(liger_cute::detail::tp_reduce_team_handle());
	LIGER_CHECK(found != g_team_contexts.end(), "FLSCE TP workspace is not configured");
	return *found->second;
}

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

std::size_t durable_bytes_at(int tokens, int hidden) {
	LIGER_CHECK(tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(hidden > 0, "max_hidden must be positive");
	std::size_t waves =
		(static_cast<std::size_t>(tokens) + Config::kWaveRows - 1) /
		Config::kWaveRows;
	std::size_t rows = checked_multiply(
		waves, static_cast<std::size_t>(Config::kWaveRows), "dX durable rows");
	std::size_t n_tiles =
		(static_cast<std::size_t>(hidden) + Config::kDxTileN - 1) /
		Config::kDxTileN;
	std::size_t columns = checked_multiply(
		n_tiles,
		static_cast<std::size_t>(Config::kDxTileN),
		"dX durable columns");
	return checked_multiply(
		checked_multiply(rows, columns, "dX durable elements"),
		sizeof(float),
		"dX durable");
}

std::size_t forward_wave_state_bytes_at(
		int tokens, int local_partition_size) {
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	std::size_t rows_per_rank =
		(static_cast<std::size_t>(tokens) +
			static_cast<std::size_t>(local_partition_size) - 1) /
		static_cast<std::size_t>(local_partition_size);
	return checked_multiply(
		checked_multiply(
			rows_per_rank,
			static_cast<std::size_t>(
				forward_reduced_state_fields<true>()),
			"forward wave state"),
		sizeof(float),
		"forward wave state");
#else
	(void)tokens;
	(void)local_partition_size;
	return 0;
#endif
}

void ensure_configured() {
	LIGER_CHECK(
		g_configured,
		"fused_scaled_linear_cross_entropy backward: call "
		"configure_backward_tp_symmetric(max_tokens, max_hidden, "
		"max_local_vocab, "
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
		"fused_scaled_linear_cross_entropy backward: device reports ",
		properties.multiProcessorCount,
		" SMs, but the CTA-owned dX workspace supports at most ",
		kMaxDxResidentCtas);
	return properties.multiProcessorCount;
}

void validate_configured_symmetric_query(
		int max_tokens,
		int max_hidden,
		int max_tiles_per_reduce,
		int max_comm_channels) {
	if (!g_configured) return;
	LIGER_CHECK(
		max_tokens == g_capacity.max_tokens &&
			max_hidden == g_capacity.max_hidden &&
			max_tiles_per_reduce == g_capacity.max_tiles_per_reduce &&
			max_comm_channels == g_capacity.max_comm_channels,
		"fused_scaled_linear_cross_entropy backward workspace query must "
		"match the configured capacity (configured tokens ",
		g_capacity.max_tokens,
		", hidden ",
		g_capacity.max_hidden,
		", TilesPerReduce ",
		g_capacity.max_tiles_per_reduce,
		", communication channels ",
		g_capacity.max_comm_channels,
		"; requested tokens ",
		max_tokens,
		", hidden ",
		max_hidden,
		", TilesPerReduce ",
		max_tiles_per_reduce,
		", communication channels ",
		max_comm_channels,
		")");
}

void validate_configured_device_query(int max_local_vocab) {
	if (!g_configured) return;
	LIGER_CHECK(
		max_local_vocab == g_capacity.max_local_vocab,
		"fused_scaled_linear_cross_entropy backward workspace query must "
		"match the configured capacity (configured local vocabulary ",
		g_capacity.max_local_vocab,
		"; requested ",
		max_local_vocab,
		")");
}

}  // namespace

std::int64_t tp_context_slot() {
	if (g_slots.empty()) return 0;
	return selected_context().slot;
}

std::string buffer_name(const char* name, std::int64_t slot) {
	if (slot < 0) slot = tp_context_slot();
	if (slot == 0) return name;
	return "fslce_context_" + std::to_string(slot) + "/" + name;
}

void configure_backward_tp_symmetric(
		int max_tokens, int max_hidden, int max_local_vocab,
		int max_tiles_per_reduce, int max_comm_channels,
		std::int64_t team_handle) {
	configure_backward_tp_context(
		max_tokens, max_hidden, max_local_vocab,
		max_tiles_per_reduce, max_comm_channels, team_handle, -1);
}

void configure_backward_tp_context(
		int max_tokens,
		int max_hidden,
		int max_local_vocab,
		int max_tiles_per_reduce,
		int max_comm_channels,
		std::int64_t team_handle,
		std::int64_t context_slot) {
	bool prepare_forward = context_slot >= 0;
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_hidden > 0, "max_hidden must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	LIGER_CHECK(
		max_tiles_per_reduce >= 1,
		"max_tiles_per_reduce must be positive");
	LIGER_CHECK(
		max_comm_channels >= 1,
		"max_comm_channels must be positive");

	int max_resident_ctas = resident_cta_capacity();
	if (g_configured) {
		LIGER_CHECK(
			max_tokens <= g_capacity.max_tokens &&
				max_hidden <= g_capacity.max_hidden &&
				max_local_vocab <= g_capacity.max_local_vocab &&
				max_tiles_per_reduce <= g_capacity.max_tiles_per_reduce &&
				max_comm_channels <= g_capacity.max_comm_channels &&
				max_resident_ctas == g_capacity.max_resident_ctas,
			"fused_scaled_linear_cross_entropy backward: the symmetric "
			"capacity is immutable once allocated (configured for vocab ",
			g_capacity.max_local_vocab,
			", tokens ",
			g_capacity.max_tokens,
			", hidden ",
			g_capacity.max_hidden,
			", TilesPerReduce ",
			g_capacity.max_tiles_per_reduce,
			", communication channels ",
			g_capacity.max_comm_channels,
			", resident CTAs ",
			g_capacity.max_resident_ctas,
			"). Configure the maximum upfront.");
	}

	if (context_slot < 0) {
		if (g_team_contexts.count(team_handle)) return;
		LIGER_CHECK(
			g_slots.empty(),
			"unprepared FLSCE team; configure additional contexts collectively before execution");
		context_slot = 0;
	}
	auto existing_slot = g_slots.find(context_slot);
	if (existing_slot != g_slots.end()) {
		LIGER_CHECK(
			existing_slot->second.team_handle == team_handle &&
				g_team_contexts.count(team_handle),
			"FLSCE context slot cannot be reassigned");
		liger_cute::detail::TpReduceContextScope selected(team_handle);
		configure_forward_tp_workspace(g_capacity.max_tokens, g_capacity.max_local_vocab);
		return;
	}
	if (!g_configured) {
		g_capacity = {max_tokens, max_hidden, max_local_vocab,
			max_tiles_per_reduce, max_comm_channels,
			max_resident_ctas, kConfiguredStages};
		g_staging_bytes = staging_bytes_at(max_tiles_per_reduce, max_resident_ctas);
		g_durable_bytes = durable_bytes_at(max_tokens, max_hidden);
	}
	max_tokens = g_capacity.max_tokens;
	max_local_vocab = g_capacity.max_local_vocab;
	auto topology = liger_cute::detail::query_tp_reduce_topology(team_handle);
	int team_size = topology.team_size;

	BackwardTpContext context;
	context.slot = context_slot;
	context.team_handle = team_handle;
	context.team_size = team_size;
	using Names = BackwardSymmetricNames;
	auto& pool = global_buffer_pool();

	// Reserve bulk scratch once for any supported TP topology. Signals alone
	// use collective slots; their allocation order must match on every PE.
	context.packed_durable_bytes =
		(g_durable_bytes + static_cast<std::size_t>(topology.local_size) - 1) /
		static_cast<std::size_t>(topology.local_size);
	std::size_t forward_wave_state_bytes =
		forward_wave_state_bytes_at(max_tokens, 1);
	std::size_t forward_wave_storage_bytes =
		checked_multiply(
			forward_wave_state_bytes,
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
			static_cast<std::size_t>(
				kForwardWaveSourceSlotsSm100 + 1),
#else
			0u,
#endif
			"forward wave source and running state");
	g_reduced_shard_bytes =
		g_durable_bytes > forward_wave_storage_bytes
		? g_durable_bytes
		: forward_wave_storage_bytes;
	g_remote_payload_bytes =
		g_durable_bytes > forward_wave_state_bytes
		? g_durable_bytes
		: forward_wave_state_bytes;
	g_reduced_bytes = g_staging_bytes > g_durable_bytes
		? g_staging_bytes : g_durable_bytes;
	auto* partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, g_staging_bytes));
	auto* reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, g_reduced_bytes));
	auto* reduced_shard = static_cast<float*>(
		pool.get_symmetric(
			Names::kDxReducedShard, g_reduced_shard_bytes));
	std::size_t remote_inbox_bytes = checked_multiply(
		g_remote_payload_bytes,
		static_cast<std::size_t>(
			liger_cute::detail::kRemoteRingInboxSlots),
		"dX remote inbox");
	auto* remote_inbox = static_cast<float*>(
		pool.get_symmetric(
			Names::kDxRemoteInbox, remote_inbox_bytes));
	auto* remote_signals = static_cast<std::uint64_t*>(
		pool.get_symmetric(
			buffer_name(Names::kDxRemoteSignals, context_slot),
			liger_cute::detail::remote_ring_signal_bytes()));
	int max_team_size = nvshmem_n_pes() < liger_cute::detail::kMaxTpReduceTeamSize
		? nvshmem_n_pes() : liger_cute::detail::kMaxTpReduceTeamSize;
	g_sync_bytes = sync_bytes_at(max_resident_ctas, max_team_size);
	auto* sync = static_cast<std::uint64_t*>(
		pool.get_symmetric(buffer_name(Names::kDxSync, context_slot), g_sync_bytes));
	auto** peer_partial_storage = static_cast<float**>(
		pool.get_device(
			buffer_name(Names::kDxPeerPartialPointers, context_slot),
			static_cast<std::size_t>(team_size) * sizeof(float*)));
	auto** peer_sync_storage = static_cast<std::uint64_t**>(
		pool.get_device(
			buffer_name(Names::kDxPeerSyncPointers, context_slot),
			static_cast<std::size_t>(team_size) *
				sizeof(std::uint64_t*)));
	pool.get_device(
		Names::kDzWorkspace, Launch::dz_workspace_bytes(max_local_vocab));
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	pool.get_device(buffer_name(Names::kBackwardSm100Signals, context_slot),
		static_cast<std::size_t>(kBackwardSignalEntries) * sizeof(std::uint64_t));
	pool.get_device(buffer_name(Names::kBackwardSm100Diagnostics, context_slot),
		static_cast<std::size_t>(kBackwardDiagnosticEntries) * sizeof(std::uint64_t));
	if constexpr (kBackwardSyncVariantSm100 == 2) {
		pool.get_device(Names::kBackwardSm100DzTileReady,
			static_cast<std::size_t>(Launch::num_waves(max_tokens)) *
			static_cast<std::size_t>(Launch::num_dz_cluster_pairs(max_local_vocab)) *
			sizeof(std::uint32_t));
	}
#endif
	auto* launch_epoch = static_cast<std::uint64_t*>(
		pool.get_device(buffer_name(Names::kDxLaunchEpoch, context_slot), sizeof(std::uint64_t)));
	check_cuda(
		cudaMemset(launch_epoch, 0, sizeof(std::uint64_t)),
		"cudaMemset(dX launch epoch)");

	// Keep a reservation even on cache-hit PEs: skipping its allocations would
	// desynchronize the symmetric heap from peers preparing a new subgroup.
	auto* stored = &g_slots.emplace(context_slot, context).first->second;
	if (g_team_contexts.count(team_handle)) return;
	liger_cute::detail::configure_tp_reduce(
		team_handle,
		{
			partial,
			reduced,
			reduced_shard,
			g_reduced_shard_bytes,
			remote_inbox,
			remote_signals,
			g_remote_payload_bytes,
			sync,
			g_sync_bytes,
			peer_partial_storage,
			peer_sync_storage,
		});
	g_team_contexts.emplace(team_handle, stored);
	g_configured = true;
	if (prepare_forward) {
		liger_cute::detail::TpReduceContextScope selected(team_handle);
		configure_forward_tp_workspace(max_tokens, max_local_vocab);
	}
}

std::size_t backward_tp_pool_symmetric_bytes(
		int max_tokens,
		int max_hidden,
		int max_tiles_per_reduce,
		int max_comm_channels) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_hidden > 0, "max_hidden must be positive");
	LIGER_CHECK(
		max_tiles_per_reduce >= 1,
		"max_tiles_per_reduce must be positive");
	LIGER_CHECK(
		max_comm_channels >= 1,
		"max_comm_channels must be positive");
	validate_configured_symmetric_query(
		max_tokens,
		max_hidden,
		max_tiles_per_reduce,
		max_comm_channels);
	int max_ctas =
		g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
	std::size_t staging = staging_bytes_at(max_tiles_per_reduce, max_ctas);
	std::size_t durable = durable_bytes_at(max_tokens, max_hidden);
	std::size_t reduced = staging > durable ? staging : durable;
	std::size_t forward_wave_state =
		forward_wave_state_bytes_at(max_tokens, 1);
	std::size_t forward_wave_storage =
		checked_multiply(
			forward_wave_state,
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
			static_cast<std::size_t>(
				kForwardWaveSourceSlotsSm100 + 1),
#else
			0u,
#endif
			"forward wave source and running state");
	std::size_t reduced_shard =
		durable > forward_wave_storage
		? durable
		: forward_wave_storage;
	std::size_t remote_payload =
		durable > forward_wave_state
		? durable
		: forward_wave_state;
	std::size_t slots = g_configured ? g_slots.size() : 1;
	std::size_t sync_bytes = g_configured ? g_sync_bytes
		: sync_bytes_at(max_ctas, liger_cute::detail::kMaxTpReduceTeamSize);
	return staging + reduced + reduced_shard +
		checked_multiply(
			remote_payload,
			static_cast<std::size_t>(
				liger_cute::detail::kRemoteRingInboxSlots),
			"remote inbox buffers") +
		slots * (sync_bytes + liger_cute::detail::remote_ring_signal_bytes());
}

std::size_t backward_tp_pool_device_bytes(int max_local_vocab, int max_tokens) {
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	validate_configured_device_query(max_local_vocab);
	std::size_t shared_bytes = Launch::dz_workspace_bytes(max_local_vocab);
	std::size_t bytes_per_slot = sizeof(std::uint64_t);
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	bytes_per_slot += static_cast<std::size_t>(
		kBackwardSignalEntries + kBackwardDiagnosticEntries) * sizeof(std::uint64_t);
	if constexpr (kBackwardSyncVariantSm100 == 2) {
		int tokens = g_configured ? g_capacity.max_tokens : max_tokens;
		int waves = tokens > 0 ? Launch::num_waves(tokens) : kBackwardMaxWavesSm100;
		shared_bytes += static_cast<std::size_t>(waves) *
			static_cast<std::size_t>(Launch::num_dz_cluster_pairs(max_local_vocab)) *
			sizeof(std::uint32_t);
	}
#else
	(void)max_tokens;
#endif
	if (!g_configured) {
		return shared_bytes + bytes_per_slot +
			liger_cute::detail::kMaxTpReduceTeamSize * (2 * sizeof(void*));
	}
	std::size_t bytes = shared_bytes;
	for (const auto& slot : g_slots) {
		bytes += bytes_per_slot + static_cast<std::size_t>(slot.second.team_size) *
			(2 * sizeof(void*));
	}
	return bytes;
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
	DxReduceWorkspace<float> workspace = {};
	workspace.partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, g_staging_bytes));
	workspace.reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, g_reduced_bytes));
	workspace.sync = static_cast<std::uint64_t*>(
		pool.get_symmetric(buffer_name(Names::kDxSync), g_sync_bytes));
	workspace.launch_epoch = static_cast<const std::uint64_t*>(
		pool.get_device(buffer_name(Names::kDxLaunchEpoch), sizeof(std::uint64_t)));
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

std::size_t backward_dx_configured_durable_bytes() {
	ensure_configured();
	return g_durable_bytes;
}

std::size_t backward_dx_configured_packed_durable_bytes() {
	ensure_configured();
	return selected_context().packed_durable_bytes;
}

std::size_t tp_reduced_shard_configured_bytes() {
	ensure_configured();
	return g_reduced_shard_bytes;
}

std::size_t tp_remote_inbox_slot_configured_bytes() {
	ensure_configured();
	return g_remote_payload_bytes;
}

int backward_tp_max_local_vocab() {
	ensure_configured();
	return g_capacity.max_local_vocab;
}

void validate_backward_tp_shape(
		int tokens, int hidden, int local_vocab) {
	ensure_configured();
	LIGER_CHECK(
		tokens > 0 && tokens <= g_capacity.max_tokens,
		"fused_scaled_linear_cross_entropy backward: tokens ",
		tokens,
		" exceed the configured maximum ",
		g_capacity.max_tokens);
	LIGER_CHECK(
		hidden > 0 && hidden <= g_capacity.max_hidden,
		"fused_scaled_linear_cross_entropy backward: hidden ",
		hidden,
		" exceeds the configured maximum ",
		g_capacity.max_hidden);
	LIGER_CHECK(
		local_vocab > 0 && local_vocab <= g_capacity.max_local_vocab,
		"fused_scaled_linear_cross_entropy backward: local_vocab ",
		local_vocab,
		" exceeds the configured maximum ",
		g_capacity.max_local_vocab);
	LIGER_CHECK(
		durable_bytes_at(tokens, hidden) <=
			backward_dx_configured_durable_bytes(),
		"fused_scaled_linear_cross_entropy backward: cluster dX durable "
		"workspace capacity exceeded");
}

int backward_dx_resident_cta_capacity() {
	return g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
}

int backward_dx_team_size() {
	ensure_configured();
	return selected_context().team_size;
}

std::int64_t backward_dx_team_handle() {
	ensure_configured();
	return selected_context().team_handle;
}

void reset_fslce_tp_configuration() {
	liger_cute::detail::reset_tp_reduce();
	g_capacity = {};
	g_configured = false;
	g_staging_bytes = 0;
	g_durable_bytes = 0;
	g_reduced_bytes = 0;
	g_reduced_shard_bytes = 0;
	g_remote_payload_bytes = 0;
	g_sync_bytes = 0;
	g_team_contexts.clear();
	g_slots.clear();
	reset_forward_tp_workspace_configuration();
}

void release_fslce_tp_team(std::int64_t team_handle) {
	auto found = g_team_contexts.find(team_handle);
	if (found == g_team_contexts.end()) return;
	check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(release TP team)");
	liger_cute::detail::release_tp_reduce(team_handle);
	g_team_contexts.erase(found);
}

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
	return scratch;
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
