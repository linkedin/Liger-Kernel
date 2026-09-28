// ═══════════════════════════════════════════════════════════════════════════
// SM90 tensor-parallel fused scaled linear cross entropy — forward launcher.
//
// Sole production host call graph:
//
//   fused_linear_scaled_cross_entropy_forward
//     -> forward_gemm_kernel_sm90
//        -> split last-arrival epilogue
//        -> local NVSHMEM MAX
//        -> corrected local NVSHMEM SUM
//     -> optional sharded remote NVSHMEM state merge
//     -> finalize_forward_kernel
//
// The WGMMA translation unit is deliberately non-RDC. NVSHMEM collective
// kernels remain in forward_reduce.cu, which is compiled with RDC.
// ═══════════════════════════════════════════════════════════════════════════

// Must precede every other CUTLASS/CuTe include: it opts the TU into CuTe's
// extended SM90 MMA shapes, which is where the N160 WGMMA atom lives.
#include "forward_gemm_mainloop_sm90.cuh"

#include "forward_gemm_sm90.cuh"
#include "forward_reduce.cuh"
#include "gemm_sm90.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"
#include "online_softmax.cuh"
#include "workspace.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy forward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

}  // namespace

template <bool ReturnEntropy, int Compute>
void fused_linear_scaled_cross_entropy_forward(
		const ForwardTpParamsSm90<Compute>& params,
		cudaStream_t stream) {
	using Traits = ForwardGemmTraitsSm90<Compute>;
	using Config = typename Traits::Config;
	using Launch = ForwardGemmLaunchSm90<Compute>;
	using Element = typename Traits::Element;
	using Smem = ForwardGemmSmemSm90<Compute, ReturnEntropy>;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	const ForwardGemmParamsSm90<Compute>& input = params.gemm;
	LIGER_CHECK(input.tokens >= 0, "tokens must be non-negative");
	if (input.tokens == 0) return;
	LIGER_CHECK(input.hidden > 0, "hidden must be positive");
	LIGER_CHECK(input.local_vocab > 0, "local_vocab must be positive");
	LIGER_CHECK(
		input.x != nullptr && input.weight != nullptr &&
			input.target != nullptr,
		"forward inputs must be non-null");
	LIGER_CHECK(
		params.nll != nullptr && params.lse != nullptr,
		"forward TP outputs nll and lse must be non-null");
	if constexpr (ReturnEntropy) {
		LIGER_CHECK(
			params.entropy != nullptr,
			"entropy output is required when ReturnEntropy=true");
	}
	LIGER_CHECK(
		params.team_handle == backward_dx_team_handle(),
		"forward TP team must match the configured reduction team");
	LIGER_CHECK(
		reinterpret_cast<std::uintptr_t>(input.x) % 16 == 0 &&
			reinterpret_cast<std::uintptr_t>(input.weight) % 16 == 0,
		"forward GEMM operands must be 16 B aligned for TMA");
	LIGER_CHECK(
		(static_cast<std::size_t>(input.hidden) * sizeof(Element)) % 16 == 0,
		"hidden * sizeof(bfloat16) must be a multiple of 16 B for TMA");
	LIGER_CHECK(
		input.tuning.split_n >= 0 && input.tuning.base_split_n >= 0 &&
			input.tuning.extra_m_pairs >= 0 &&
			input.tuning.target_cluster_pairs >= 0,
		"split and cluster tuning values must be non-negative");
	LIGER_CHECK(input.tuning.max_split_n >= 1, "max_split_n must be positive");
	LIGER_CHECK(
		input.tuning.extra_m_pairs == 0 ||
			input.tuning.base_split_n != 0,
		"extra_m_pairs requires an explicit base_split_n");

	ForwardTpWorkspace reduce_workspace =
		reserve_forward_tp_workspace<ReturnEntropy>(input.tokens);

	ForwardGemmParamsSm90<Compute> gemm_params = input;
	gemm_params.output = reduce_workspace.local;
	gemm_params.workspace = reduce_workspace.gemm_split_partials;
	gemm_params.workspace_bytes =
		reduce_workspace.gemm_split_partials_bytes;

	LIGER_CHECK(
		gemm_params.output.local_max != nullptr &&
			gemm_params.output.local_sum != nullptr &&
			gemm_params.output.local_target != nullptr,
		"forward GEMM local statistics buffers must be non-null");
	if constexpr (ReturnEntropy) {
		LIGER_CHECK(
			gemm_params.output.local_weighted_sum != nullptr,
			"local_weighted_sum is required when ReturnEntropy=true");
	}

	// Exact CuTe-DSL operand ordering: X is an ordinary TMA load and each
	// N160 weight panel is multicast over the cluster's M mode.
	auto tma_load_x = sm90::TmaSm90<Compute>::template make_load<
		typename Traits::SmemLayoutX1,
		Traits::kTileM,
		Traits::kTileK>(
			static_cast<const Element*>(gemm_params.x),
			gemm_params.tokens,
			gemm_params.hidden);
	auto tma_load_w =
		sm90::TmaSm90<Compute>::template make_load_multicast<
			typename Traits::SmemLayoutW1,
			Traits::kAccumulatorTileN,
			Traits::kTileK,
			Traits::kClusterM>(
				static_cast<const Element*>(gemm_params.weight),
				gemm_params.local_vocab,
				gemm_params.hidden);

	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	static_assert(kSmemBytes <= kHopperMaxSmemBytes,
		"SM90 forward shared-memory footprint exceeds the Hopper limit");
	static_assert(kSmemBytes <= Launch::smem_bytes(ReturnEntropy),
		"ForwardGemmLaunchSm90::smem_bytes must bound the real footprint");

	using Cluster = sm90::ClusterLaunchSm90<Compute>;
	liger_cute::detail::TpReducePlan reduce =
		liger_cute::detail::tp_reduce_plan();
	DxReduceWorkspace<float> comm = {};
	float* remote_source = reduce.remote.enabled()
		? reduce.remote.reduced_shard
		: nullptr;

	auto launch_local = [&](auto mapping, auto backend_tag, auto remote_tag) {
		constexpr auto Backend = decltype(backend_tag)::value;
		constexpr bool RequiresRemote = decltype(remote_tag)::value;
		using Mapping = std::decay_t<decltype(mapping)>;
		auto* kernel_fn = &forward_gemm_tp_kernel_sm90<
			ReturnEntropy,
			Compute,
			RequiresRemote,
			Backend,
			decltype(tma_load_x),
			decltype(tma_load_w),
			Mapping>;

		check_cuda(
			Cluster::prepare(kernel_fn, kSmemBytes),
			"cudaFuncSetAttribute(MaxDynamicSharedMemorySize / "
			"NonPortableClusterSizeAllowed)");
		int max_active_cluster_pairs = Cluster::max_active_clusters(
			kernel_fn,
			Config::kNumThreads,
			kSmemBytes,
			Config::kClusterM);
		ForwardGemmSplitSm90<Compute> split = Launch::resolve_split(
			gemm_params.tuning,
			gemm_params.tokens,
			gemm_params.local_vocab,
			max_active_cluster_pairs);
		LIGER_CHECK(
			split.num_cluster_pairs <= max_active_cluster_pairs,
			"forward local reduction requires the complete cluster grid "
			"to remain resident");
		LIGER_CHECK(
			split.base_split_n >= 1 &&
				split.split_n <= split.num_logical_n_tiles,
			"split_n must be between 1 and the number of logical vocabulary "
			"tiles");
		LIGER_CHECK(
			split.extra_m_pairs < split.num_m_pairs &&
				split.split_n ==
					split.base_split_n +
						(split.extra_m_pairs != 0 ? 1 : 0),
			"uneven split tuning must use split_n=base_split_n+1 for a "
			"proper subset of M pairs");

		std::size_t partial_rows =
			static_cast<std::size_t>(split.num_m_tiles) *
			static_cast<std::size_t>(split.split_n) *
			static_cast<std::size_t>(Config::kTileM);
		std::size_t partial_bytes = partial_rows * sizeof(float);
		std::size_t required_bytes =
			partial_bytes * (ReturnEntropy ? 4u : 3u);
		LIGER_CHECK(
			gemm_params.workspace != nullptr &&
				gemm_params.workspace_bytes >= required_bytes,
			"forward GEMM needs a ",
			required_bytes,
			" B split-partial workspace, got ",
			gemm_params.workspace_bytes);
		LIGER_CHECK(
			reinterpret_cast<std::uintptr_t>(gemm_params.workspace) % 16 ==
				0,
			"forward GEMM workspace must be 16 B aligned");

		float* scratch = static_cast<float*>(gemm_params.workspace);
		ForwardGemmPartialsSm90<Compute> partials;
		partials.partial_max = scratch + 0 * partial_rows;
		partials.partial_sum = scratch + 1 * partial_rows;
		partials.partial_target = scratch + 2 * partial_rows;
		partials.partial_weighted =
			ReturnEntropy ? scratch + 3 * partial_rows : nullptr;

		int grid_ctas = split.num_cluster_pairs * Config::kClusterM;
		comm = reserve_dx_reduce_workspace(
			1, kDxRingStages, grid_ctas);
		std::size_t forward_comm_elements =
			static_cast<std::size_t>(split.num_m_tiles) *
			kForwardReducedFields *
			Config::kTileM;
		LIGER_CHECK(
			forward_comm_elements * sizeof(float) <=
				backward_dx_configured_staging_bytes(),
			"forward local reduction exceeds the configured TP staging "
			"capacity");
		if constexpr (RequiresRemote) {
			static_assert(
				Backend ==
					liger_cute::detail::LocalReduceBackend::kNvls);
			int rows_per_rank = ceil_div(
				gemm_params.tokens, mapping.size);
			std::size_t remote_elements =
				static_cast<std::size_t>(rows_per_rank) *
				static_cast<std::size_t>(
					1 + reduce_workspace.fields);
			LIGER_CHECK(
				remote_elements * sizeof(float) <=
					backward_dx_configured_packed_durable_bytes(),
				"forward remote reduction exceeds the configured "
				"symmetric durable capacity");
			std::size_t gathered_elements =
				remote_elements *
				static_cast<std::size_t>(mapping.size);
			LIGER_CHECK(
				gathered_elements * sizeof(float) <=
					backward_dx_configured_staging_bytes(),
				"forward remote all-gather exceeds the configured "
				"symmetric staging capacity");
		}
		LIGER_CHECK(
			reduce_workspace.split_ready != nullptr,
			"forward split completion counters must be non-null");

		check_cuda(
			cudaMemsetAsync(
				reduce_workspace.split_ready,
				0,
				static_cast<std::size_t>(split.num_m_tiles) *
					sizeof(int),
				stream),
			"cudaMemsetAsync(forward split counters)");
		liger_cute::detail::begin_tp_reduce(
			comm.launch_epoch, stream);
		check_cuda(
			Cluster::launch(
				kernel_fn,
				dim3(
					static_cast<unsigned>(Config::kClusterM),
					1u,
					static_cast<unsigned>(split.num_cluster_pairs)),
				Config::kNumThreads,
				kSmemBytes,
				Config::kClusterM,
				stream,
				tma_load_x,
				tma_load_w,
				gemm_params,
				partials,
				split,
				comm,
				mapping,
				reduce_workspace.split_ready,
				reduce_workspace.global_max,
				reduce_workspace.reduced,
				remote_source),
			"cudaLaunchKernelEx(forward_gemm_tp_kernel_sm90)");
	};

	if (reduce.backend ==
		liger_cute::detail::LocalReduceBackend::kNvls) {
		if (reduce.remote.enabled()) {
			launch_local(
				reduce.nvls,
				std::integral_constant<
					liger_cute::detail::LocalReduceBackend,
					liger_cute::detail::LocalReduceBackend::kNvls>{},
				std::true_type{});
		} else {
			launch_local(
				reduce.nvls,
				std::integral_constant<
					liger_cute::detail::LocalReduceBackend,
					liger_cute::detail::LocalReduceBackend::kNvls>{},
				std::false_type{});
		}
	} else {
		LIGER_CHECK(
			reduce.direct.available != 0,
			"forward TP requires NVLS or complete direct-peer mappings");
		launch_local(
			reduce.direct,
			std::integral_constant<
				liger_cute::detail::LocalReduceBackend,
				liger_cute::detail::LocalReduceBackend::kDirectPeer>{},
			std::false_type{});
	}

	launch_forward_remote_finalize(
		ReturnEntropy,
		reduce.remote,
		reduce.nvls,
		comm,
		comm.launch_epoch,
		reduce_workspace,
		gemm_params.target,
		params.nll,
		params.lse,
		params.entropy,
		gemm_params.tokens,
		gemm_params.ignore_index,
		stream);
	liger_cute::detail::end_tp_reduce(stream);
}

template void fused_linear_scaled_cross_entropy_forward<false, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);
template void fused_linear_scaled_cross_entropy_forward<true, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
