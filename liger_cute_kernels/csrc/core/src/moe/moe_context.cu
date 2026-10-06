#include <nvshmem.h>
#include <cuda_runtime.h>

#include <limits>
#include <map>

#include "moe_symm_config.cuh"
#include "moe_nonrdc_module.h"
#include "liger_cute/check.h"
#include "liger_cute/detail/symmetric_memory.h"

namespace liger {
namespace {
MoeSymmConfig g_capacity{};
std::map<std::int64_t, MoeSymmConfig> g_contexts;
thread_local std::int64_t g_selected_team = -1;

std::string slot_name(const char* name, std::int64_t slot) {
	return slot < 0 ? name : "moe_context_" + std::to_string(slot) + "/" + name;
}

constexpr const char* kRetained[] = {
	"x_sorted", "y_buf", "all_expert_offsets", "all_expert_counts"
};
}  // namespace

MoeSymmConfig& get_symm_config() {
	if (g_contexts.empty() && g_capacity.slot < 0) return g_capacity;
	auto it = g_contexts.find(g_selected_team);
	if (it != g_contexts.end()) return it->second;
	LIGER_CHECK(g_selected_team < 0 && g_contexts.size() == 1,
		"MoE team is not prepared; configure its context before execution");
	return g_contexts.begin()->second;
}

MoeContextScope::MoeContextScope(std::int64_t team) : previous_(g_selected_team) {
	LIGER_CHECK((g_contexts.empty() && g_capacity.slot < 0) || g_contexts.count(team) ||
		(team < 0 && g_contexts.size() == 1),
		"MoE team is not prepared; configure its context before execution");
	g_selected_team = team;
}

MoeContextScope::~MoeContextScope() { g_selected_team = previous_; }

std::string moe_buffer_name(const char* name) {
	return slot_name(name, get_symm_config().slot);
}

void moe_check_forward_capacity() {
	const auto& cfg = get_symm_config();
	if (cfg.slot < 0) return;  // Legacy WORLD-collective allocator.
	auto& stack = liger_cute::detail::global_symmetric_stack();
	for (const char* name : kRetained) {
		LIGER_CHECK(stack.available(moe_buffer_name(name)),
			"MoE retained forward capacity exhausted; configure max_inflight upfront");
	}
}

void release_moe_team(std::int64_t team) {
	auto it = g_contexts.find(team);
	if (it == g_contexts.end()) return;
	auto& stack = liger_cute::detail::global_symmetric_stack();
	for (const char* name : kRetained) {
		LIGER_CHECK(!stack.active(slot_name(name, it->second.slot)),
			"cannot destroy a MoE team with live forward intermediates");
	}
	g_contexts.erase(it);
}

void reset_moe_configuration(bool clear_capacity) {
	g_contexts.clear();
	if (clear_capacity || g_capacity.slot >= 0) g_capacity = {};
	g_selected_team = -1;
}

void moe_configure_context(int max_tokens, int hidden_dim, int max_num_experts,
		int max_top_k, int num_hosts, int gpus_per_host, int max_inflight,
		std::int64_t team, std::int64_t slot) {
	using namespace liger_cute::detail;
	LIGER_CHECK(max_tokens > 0 && hidden_dim > 0 && max_num_experts > 0 &&
		max_top_k > 0 && max_top_k <= max_num_experts && max_inflight > 0 &&
		num_hosts > 0 && gpus_per_host > 0 && slot >= 0,
		"invalid MoE context capacity or topology");
	const int num_pes = nvshmem_team_n_pes(static_cast<nvshmem_team_t>(team));
	LIGER_CHECK(num_pes > 0 && nvshmem_team_my_pe(static_cast<nvshmem_team_t>(team)) >= 0 &&
		static_cast<std::int64_t>(num_hosts) * gpus_per_host == num_pes,
		"MoE topology must match the selected EP team");
	const std::int64_t slots = static_cast<std::int64_t>(max_tokens) * max_top_k +
		static_cast<std::int64_t>(max_num_experts) * kMaxMoeCommTileM;
	LIGER_CHECK(slots <= std::numeric_limits<int>::max(), "MoE slot capacity exceeds int32");
	auto existing = g_contexts.find(team);
	LIGER_CHECK(existing == g_contexts.end() ||
		(existing->second.num_hosts == num_hosts &&
		 existing->second.gpus_per_host == gpus_per_host),
		"MoE topology for a prepared team cannot change");
	if (g_capacity.initialized) {
		LIGER_CHECK(slots == g_capacity.max_total_slots &&
			hidden_dim == g_capacity.hidden_dim &&
			max_num_experts == g_capacity.max_num_experts &&
			max_top_k == g_capacity.max_top_k &&
			max_inflight == g_capacity.max_inflight,
			"MoE process-wide capacities cannot change; reserve maxima upfront");
	}
	MoeSymmConfig cfg{};
	cfg.max_total_slots = static_cast<int>(slots);
	cfg.hidden_dim = hidden_dim;
	cfg.max_num_experts = max_num_experts;
	cfg.max_top_k = max_top_k;
	cfg.max_inflight = max_inflight;
	cfg.num_pes = num_pes;
	cfg.num_hosts = num_hosts;
	cfg.gpus_per_host = gpus_per_host;
	cfg.experts_per_pe = max_num_experts / num_pes;
	cfg.team = static_cast<nvshmem_team_t>(team);
	cfg.slot = slot;
	cfg.initialized = true;

	int device = 0, sms = 0;
	LIGER_CHECK(cudaGetDevice(&device) == cudaSuccess &&
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess,
		"cannot query CUDA device for MoE capacity");
	const std::size_t payload = static_cast<std::size_t>(slots) * hidden_dim * 2;
	const std::size_t staging = static_cast<std::size_t>(sms) *
		kMaxMoeCommStages * kMaxMoeCommTileM * hidden_dim * 2;
	auto& pool = global_buffer_pool();
	pool.get_symmetric("moe_src_staging", staging);
	pool.get_symmetric("moe_dst_staging", staging);
	pool.get_symmetric("moe_sort_offsets", (max_num_experts + 1) * sizeof(int));
	pool.get_symmetric("moe_sort_counts", max_num_experts * sizeof(int));
	pool.get_symmetric("moe_bwd_dy_sorted", payload);
	pool.get_symmetric("moe_bwd_dx_sorted", payload);
	pool.get_symmetric("moe_bwd_x_staging", staging);
	pool.get_symmetric("moe_bwd_dy_staging", staging);
	pool.get_symmetric("moe_bwd_dx_staging", staging);

	// Every PE reserves the slot, including cache hits in a mixed partition.
	// WORLD-sized metadata permits any later EP size under the same maxima.
	auto& stack = global_symmetric_stack();
	const std::size_t peers = nvshmem_n_pes();
	stack.reserve(slot_name(kRetained[0], slot), payload, max_inflight);
	stack.reserve(slot_name(kRetained[1], slot), payload, max_inflight);
	stack.reserve(slot_name(kRetained[2], slot),
		peers * (max_num_experts + 1) * sizeof(int), max_inflight);
	stack.reserve(slot_name(kRetained[3], slot),
		peers * max_num_experts * sizeof(int), max_inflight);
#if LIGER_CUTE_DISPATCH_COMPUTE == 90
	configure_sm90_nonrdc_moe();
#endif
	g_capacity = cfg;
	if (!g_contexts.count(team)) g_contexts.emplace(team, cfg);
}
}  // namespace liger
