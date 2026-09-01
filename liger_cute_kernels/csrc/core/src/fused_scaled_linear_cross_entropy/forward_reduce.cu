#include "forward_reduce.cuh"

#include <cstddef>

#include "buffer_pool.cuh"
#include "liger_cute/check.h"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

int g_capacity_tokens = 0;
std::size_t g_capacity_split_partials_bytes = 0;

std::size_t token_bytes_at(int tokens) {
	return static_cast<std::size_t>(tokens) * sizeof(float);
}

std::size_t packed_bytes_at(int tokens) {
	return static_cast<std::size_t>(kForwardReducedFields) *
		token_bytes_at(tokens);
}

std::size_t split_partials_bytes_at(int tokens, int local_vocab) {
	return ForwardGemmLaunchSm90<90>::workspace_bytes(
		tokens, local_vocab, /*return_entropy=*/true);
}

void ensure_capacity(int tokens) {
	LIGER_CHECK(tokens > 0, "tokens must be positive");
	LIGER_CHECK(
		g_capacity_tokens != 0,
		"fused_scaled_linear_cross_entropy forward: call "
		"configure_forward_tp_workspace(max_tokens, max_local_vocab) before "
		"the first launch");
	LIGER_CHECK(
		tokens <= g_capacity_tokens,
		"fused_scaled_linear_cross_entropy forward workspace was configured "
		"for ",
		g_capacity_tokens,
		" tokens but ",
		tokens,
		" were requested");
}

}  // namespace

void configure_forward_tp_workspace(int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");

	std::size_t split_bytes =
		split_partials_bytes_at(max_tokens, max_local_vocab);
	if (max_tokens <= g_capacity_tokens &&
		split_bytes <= g_capacity_split_partials_bytes) {
		return;
	}
	LIGER_CHECK(
		g_capacity_tokens == 0,
		"fused_scaled_linear_cross_entropy forward workspace is immutable "
		"after its first configuration");

	using Names = ForwardBufferNames;
	std::size_t token_bytes = token_bytes_at(max_tokens);
	std::size_t packed_bytes = packed_bytes_at(max_tokens);
	auto& pool = global_buffer_pool();

	pool.get_device(Names::kLocalMax, token_bytes);
	pool.get_device(Names::kGlobalMax, token_bytes);
	pool.get_device(Names::kPacked, packed_bytes);
	pool.get_device(Names::kReduced, packed_bytes);
	pool.get_device(Names::kGemmSplitPartials, split_bytes);
	pool.get_device(Names::kLocalSum, token_bytes);
	pool.get_device(Names::kLocalTarget, token_bytes);
	pool.get_device(Names::kLocalWeighted, token_bytes);

	g_capacity_tokens = max_tokens;
	g_capacity_split_partials_bytes = split_bytes;
}

std::size_t forward_tp_workspace_device_bytes(
		int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	return 5 * token_bytes_at(max_tokens) +
		2 * packed_bytes_at(max_tokens) +
		split_partials_bytes_at(max_tokens, max_local_vocab);
}

void reset_forward_tp_workspace_configuration() {
	g_capacity_tokens = 0;
	g_capacity_split_partials_bytes = 0;
}

template <bool ReturnEntropy>
ForwardTpWorkspace reserve_forward_tp_workspace(int tokens) {
	ensure_capacity(tokens);

	using Names = ForwardBufferNames;
	std::size_t token_bytes = token_bytes_at(g_capacity_tokens);
	std::size_t packed_bytes = packed_bytes_at(g_capacity_tokens);
	auto& pool = global_buffer_pool();

	ForwardTpWorkspace workspace = {};
	workspace.local.local_max = static_cast<float*>(
		pool.get_device(Names::kLocalMax, token_bytes));
	workspace.global_max = static_cast<float*>(
		pool.get_device(Names::kGlobalMax, token_bytes));
	workspace.packed = static_cast<float*>(
		pool.get_device(Names::kPacked, packed_bytes));
	workspace.reduced = static_cast<float*>(
		pool.get_device(Names::kReduced, packed_bytes));
	workspace.gemm_split_partials = pool.get_device(
		Names::kGemmSplitPartials, g_capacity_split_partials_bytes);
	workspace.gemm_split_partials_bytes =
		g_capacity_split_partials_bytes;
	workspace.local.local_sum = static_cast<float*>(
		pool.get_device(Names::kLocalSum, token_bytes));
	workspace.local.local_target = static_cast<float*>(
		pool.get_device(Names::kLocalTarget, token_bytes));
	workspace.local.local_weighted_sum = static_cast<float*>(
		pool.get_device(Names::kLocalWeighted, token_bytes));
	workspace.tokens = tokens;
	workspace.fields = forward_reduced_fields(ReturnEntropy);
	return workspace;
}

template ForwardTpWorkspace reserve_forward_tp_workspace<false>(int);
template ForwardTpWorkspace reserve_forward_tp_workspace<true>(int);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
