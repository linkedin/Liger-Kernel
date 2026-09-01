#include "workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "buffer_pool.cuh"
#include "forward_reduce.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"

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
std::size_t g_staging_bytes = 0;
std::size_t g_durable_bytes = 0;
std::size_t g_packed_durable_bytes = 0;
std::size_t g_reduced_bytes = 0;

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
		"fused_scaled_linear_cross_entropy backward: SM90 device reports ",
		properties.multiProcessorCount,
		" SMs, but the CTA-owned dX workspace supports at most ",
		kMaxDxResidentCtas);
	return properties.multiProcessorCount;
}

}  // namespace

void configure_backward_tp_symmetric(
		int max_tokens,
		int max_hidden,
		int max_local_vocab,
		int max_tiles_per_reduce,
		int max_comm_channels,
		std::int64_t team_handle) {
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
	liger_cute::detail::TpReduceTopology topology =
		liger_cute::detail::query_tp_reduce_topology(team_handle);
	int team_size = topology.team_size;

	if (g_configured) {
		LIGER_CHECK(
			max_tokens <= g_capacity.max_tokens &&
				max_hidden <= g_capacity.max_hidden &&
				max_local_vocab <= g_capacity.max_local_vocab &&
				max_tiles_per_reduce <= g_capacity.max_tiles_per_reduce &&
				max_comm_channels <= g_capacity.max_comm_channels &&
				max_resident_ctas == g_capacity.max_resident_ctas &&
				team_size == g_capacity.team_size &&
				team_handle == g_capacity.team_handle,
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
		return;
	}

	using Names = BackwardSymmetricNames;
	auto& pool = global_buffer_pool();

	// One fixed order on every PE: symmetric first, then device private.
	g_staging_bytes = staging_bytes_at(
		max_tiles_per_reduce, max_resident_ctas);
	g_durable_bytes = durable_bytes_at(max_tokens, max_hidden);
	int local_partition_size = topology.local_size;
	g_packed_durable_bytes =
		(g_durable_bytes + static_cast<std::size_t>(local_partition_size) - 1) /
		static_cast<std::size_t>(local_partition_size);
	g_reduced_bytes = local_partition_size > 1
		? (g_staging_bytes > g_durable_bytes
			? g_staging_bytes
			: g_durable_bytes)
		: g_staging_bytes;
	auto* partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, g_staging_bytes));
	auto* reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, g_reduced_bytes));
	auto* reduced_shard = static_cast<float*>(
		pool.get_symmetric(
			Names::kDxReducedShard, g_packed_durable_bytes));
	auto* remote_inbox = static_cast<float*>(
		pool.get_symmetric(
			Names::kDxRemoteInbox, g_packed_durable_bytes));
	auto* remote_signals = static_cast<std::uint64_t*>(
		pool.get_symmetric(
			Names::kDxRemoteSignals,
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
	pool.get_device(
		Names::kDzWorkspace, Launch::dz_workspace_bytes(max_local_vocab));
	auto* launch_epoch = static_cast<std::uint64_t*>(
		pool.get_device(Names::kDxLaunchEpoch, sizeof(std::uint64_t)));
	check_cuda(
		cudaMemset(launch_epoch, 0, sizeof(std::uint64_t)),
		"cudaMemset(dX launch epoch)");

	liger_cute::detail::configure_tp_reduce(
		team_handle,
		{
			partial,
			reduced,
			reduced_shard,
			remote_inbox,
			remote_signals,
			sync,
			sync_bytes,
			peer_partial_storage,
			peer_sync_storage,
		});
	g_capacity.max_tokens = max_tokens;
	g_capacity.max_hidden = max_hidden;
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
		int max_tokens,
		int max_hidden,
		int max_tiles_per_reduce,
		int max_comm_channels) {
	LIGER_CHECK(
		max_comm_channels >= 1,
		"max_comm_channels must be positive");
	int max_ctas =
		g_configured ? g_capacity.max_resident_ctas : resident_cta_capacity();
	int team_size = g_configured
		? g_capacity.team_size
		: liger_cute::detail::kMaxTpReduceTeamSize;
	if (g_configured) {
		return g_staging_bytes + g_reduced_bytes +
			2u * g_packed_durable_bytes +
			sync_bytes_at(max_ctas, team_size) +
			2 * sizeof(std::uint64_t);
	}
	std::size_t staging = staging_bytes_at(max_tiles_per_reduce, max_ctas);
	std::size_t durable = durable_bytes_at(max_tokens, max_hidden);
	std::size_t reduced = staging > durable ? staging : durable;
	return staging + reduced + 2u * durable +
		sync_bytes_at(max_ctas, team_size) +
		2 * sizeof(std::uint64_t);
}

std::size_t backward_tp_pool_device_bytes(int max_local_vocab) {
	int team_size = g_configured
		? g_capacity.team_size
		: liger_cute::detail::kMaxTpReduceTeamSize;
	std::size_t peer_mapping =
		static_cast<std::size_t>(team_size) *
		(2 * sizeof(void*));
	return Launch::dz_workspace_bytes(max_local_vocab) +
		sizeof(std::uint64_t) + peer_mapping;
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
	std::size_t sync_bytes = sync_bytes_at(
		g_capacity.max_resident_ctas, g_capacity.team_size);
	DxReduceWorkspace<float> workspace = {};
	workspace.partial = static_cast<float*>(
		pool.get_symmetric(Names::kDxPartial, g_staging_bytes));
	workspace.reduced = static_cast<float*>(
		pool.get_symmetric(Names::kDxReduced, g_reduced_bytes));
	workspace.sync = static_cast<std::uint64_t*>(
		pool.get_symmetric(Names::kDxSync, sync_bytes));
	workspace.launch_epoch = static_cast<const std::uint64_t*>(
		pool.get_device(Names::kDxLaunchEpoch, sizeof(std::uint64_t)));
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
	return g_capacity.team_size;
}

std::int64_t backward_dx_team_handle() {
	ensure_configured();
	return g_capacity.team_handle;
}

void reset_fslce_tp_configuration() {
	liger_cute::detail::reset_tp_reduce();
	g_capacity = {};
	g_configured = false;
	g_staging_bytes = 0;
	g_durable_bytes = 0;
	g_packed_durable_bytes = 0;
	g_reduced_bytes = 0;
	reset_forward_tp_workspace_configuration();
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
