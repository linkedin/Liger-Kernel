#include "forward_gemm_kernel_sm100.cuh"

#include "forward_reduce.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"
#include "workspace.cuh"

#include <cuda_runtime.h>
#include <cute/atom/copy_traits_sm100_tma.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

void check_cuda_sm100(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy SM100 forward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

struct ClusterLaunchSm100 {
	template <class Kernel>
	static void prepare(Kernel kernel, int smem_bytes) {
		int device = 0;
		int optin = 0;
		check_cuda_sm100(
			cudaGetDevice(&device),
			"cudaGetDevice");
		check_cuda_sm100(
			cudaDeviceGetAttribute(
				&optin,
				cudaDevAttrMaxSharedMemoryPerBlockOptin,
				device),
			"cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");
		LIGER_CHECK(
			smem_bytes <= optin,
			"SM100 forward requires ",
			smem_bytes,
			" B dynamic shared memory, but the device supports ",
			optin,
			" B");
		check_cuda_sm100(
			cudaFuncSetAttribute(
				kernel,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				smem_bytes),
			"cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
	}

	static cudaLaunchConfig_t config(
			dim3 grid,
			int threads,
			int smem_bytes,
			cudaStream_t stream,
			cudaLaunchAttribute& cluster_attribute) {
		cluster_attribute = {};
		cluster_attribute.id =
			cudaLaunchAttributeClusterDimension;
		cluster_attribute.val.clusterDim.x = 2;
		cluster_attribute.val.clusterDim.y = 1;
		cluster_attribute.val.clusterDim.z = 1;
		cudaLaunchConfig_t launch = {};
		launch.gridDim = grid;
		launch.blockDim =
			dim3(static_cast<unsigned>(threads), 1, 1);
		launch.dynamicSmemBytes = smem_bytes;
		launch.stream = stream;
		launch.attrs = &cluster_attribute;
		launch.numAttrs = 1;
		return launch;
	}

	template <class Kernel>
	static int max_active_clusters(
			Kernel kernel,
			int threads,
			int smem_bytes) {
		cudaLaunchAttribute cluster_attribute = {};
		cudaLaunchConfig_t launch = config(
			dim3(2, 1, 1),
			threads,
			smem_bytes,
			nullptr,
			cluster_attribute);
		int clusters = 0;
		if (cudaOccupancyMaxActiveClusters(
				&clusters,
				kernel,
				&launch) != cudaSuccess) {
			cudaGetLastError();
			return 0;
		}
		return clusters;
	}

	template <class Kernel, class... Args>
	static cudaError_t launch(
			Kernel kernel,
			dim3 grid,
			int threads,
			int smem_bytes,
			cudaStream_t stream,
			const Args&... args) {
		cudaLaunchAttribute cluster_attribute = {};
		cudaLaunchConfig_t launch = config(
			grid,
			threads,
			smem_bytes,
			stream,
			cluster_attribute);
		return cudaLaunchKernelEx(
			&launch,
			kernel,
			args...);
	}

	template <class Kernel, class... Args>
	static cudaError_t launch_cooperative(
			Kernel kernel,
			dim3 grid,
			int threads,
			int smem_bytes,
			cudaStream_t stream,
			const Args&... args) {
		cudaLaunchAttribute attributes[2] = {};
		attributes[0].id =
			cudaLaunchAttributeClusterDimension;
		attributes[0].val.clusterDim.x = 2;
		attributes[0].val.clusterDim.y = 1;
		attributes[0].val.clusterDim.z = 1;
		attributes[1].id =
			cudaLaunchAttributeCooperative;
		attributes[1].val.cooperative = 1;
		cudaLaunchConfig_t launch = {};
		launch.gridDim = grid;
		launch.blockDim =
			dim3(static_cast<unsigned>(threads), 1, 1);
		launch.dynamicSmemBytes = smem_bytes;
		launch.stream = stream;
		launch.attrs = attributes;
		launch.numAttrs = 2;
		return cudaLaunchKernelEx(
			&launch,
			kernel,
			args...);
	}
};

template <bool ReturnEntropy, int Compute>
void launch_forward_sm100(
		const ForwardTpParamsSm100<Compute>& params,
		cudaStream_t stream) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Launch = ForwardGemmLaunchSm100<Compute>;
	using Split = ForwardGemmSplitSm100<Compute>;
	using Partials = ForwardGemmPartialsSm100<Compute>;
	using Smem =
		ForwardGemmSmemSm100<
			Compute,
			ReturnEntropy>;
	using Element = typename Traits::Element;

	const ForwardGemmParamsSm100<Compute>& input =
		params.gemm;
	LIGER_CHECK(
		input.tokens >= 0,
		"tokens must be non-negative");
	if (input.tokens == 0) return;
	LIGER_CHECK(
		input.hidden > 0,
		"hidden must be positive");
	LIGER_CHECK(
		input.local_vocab > 0,
		"local_vocab must be positive");
	LIGER_CHECK(
		input.x != nullptr &&
			input.weight != nullptr &&
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
	validate_backward_tp_shape(
		input.tokens,
		input.hidden,
		input.local_vocab);
	LIGER_CHECK(
		reinterpret_cast<std::uintptr_t>(input.x) % 16 == 0 &&
			reinterpret_cast<std::uintptr_t>(
				input.weight) %
				16 ==
			0,
		"forward GEMM operands must be 16 B aligned for TMA");
	LIGER_CHECK(
		(static_cast<std::size_t>(input.hidden) *
			sizeof(Element)) %
				16 ==
			0,
		"hidden * sizeof(bfloat16) must be a multiple of 16 B for TMA");
	LIGER_CHECK(
		input.tuning.split_n >= 0 &&
			input.tuning.base_split_n >= 0 &&
			input.tuning.extra_m_pairs >= 0 &&
			input.tuning.target_cluster_pairs >= 0,
		"split and cluster tuning values must be non-negative");
	LIGER_CHECK(
		input.tuning.max_split_n >= 1,
		"max_split_n must be positive");
	LIGER_CHECK(
		input.tuning.extra_m_pairs == 0 ||
			input.tuning.base_split_n != 0,
		"extra_m_pairs requires an explicit base_split_n");

	ForwardTpWorkspace reduce_workspace =
		reserve_forward_tp_workspace<ReturnEntropy>(
			input.tokens);
	int uniform_vocab_capacity =
		backward_tp_max_local_vocab();
	LIGER_CHECK(
		input.local_vocab <=
			reduce_workspace.vocab_capacity,
		"forward local_vocab ",
		input.local_vocab,
		" exceeds the configured forward capacity ",
		reduce_workspace.vocab_capacity);
	ForwardGemmParamsSm100<Compute> gemm_params =
		input;
	gemm_params.output.local_max =
		reduce_workspace.local.local_max;
	gemm_params.output.local_sum =
		reduce_workspace.local.local_sum;
	gemm_params.output.local_target =
		reduce_workspace.local.local_target;
	gemm_params.output.local_weighted_sum =
		reduce_workspace.local.local_weighted_sum;
	gemm_params.workspace =
		reduce_workspace.gemm_split_partials;
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

	auto tensor_x = make_tensor(
		make_gmem_ptr(
			static_cast<const Element*>(
				gemm_params.x)),
		make_shape(
			static_cast<int64_t>(gemm_params.tokens),
			static_cast<int64_t>(gemm_params.hidden)),
		make_stride(
			static_cast<int64_t>(gemm_params.hidden),
			Int<1>{}));
	auto tensor_w = make_tensor(
		make_gmem_ptr(
			static_cast<const Element*>(
				gemm_params.weight)),
		make_shape(
			static_cast<int64_t>(
				gemm_params.local_vocab),
			static_cast<int64_t>(gemm_params.hidden)),
		make_stride(
			static_cast<int64_t>(gemm_params.hidden),
			Int<1>{}));
	auto tma_load_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_x,
		typename Traits::SmemLayoutX1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});
	auto tma_load_w = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_w,
		typename Traits::SmemLayoutW1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});

	constexpr int kSmemBytes =
		static_cast<int>(sizeof(Smem));
	liger_cute::detail::TpReducePlan reduce =
		liger_cute::detail::tp_reduce_plan();
	if (reduce.remote.enabled()) {
		LIGER_CHECK(
			reduce_workspace.vocab_capacity >=
				uniform_vocab_capacity,
			"forward workspace max_local_vocab ",
			reduce_workspace.vocab_capacity,
			" is smaller than the collective capacity ",
			uniform_vocab_capacity,
			" required by the SM100 remote schedule");
	}
	DxReduceWorkspace<float> comm = {};
	ForwardFinalOutputsSm100 outputs{
		params.nll,
		params.lse,
		params.entropy};

	auto launch_local =
		[&](auto mapping, auto backend_tag, auto remote_tag) {
			constexpr auto Backend =
				decltype(backend_tag)::value;
			constexpr bool RequiresRemote =
				decltype(remote_tag)::value;
			using Mapping =
				std::decay_t<decltype(mapping)>;
			auto* kernel =
				&forward_gemm_tp_kernel_sm100<
					ReturnEntropy,
					Compute,
					RequiresRemote,
					Backend,
					decltype(tma_load_x),
					decltype(tma_load_w),
					Mapping>;

			ClusterLaunchSm100::prepare(
				kernel,
				kSmemBytes);
			int max_active_cluster_pairs =
				ClusterLaunchSm100::max_active_clusters(
					kernel,
					Config::kNumThreads,
					kSmemBytes);
			LIGER_CHECK(
				max_active_cluster_pairs > 0,
				"SM100 forward could not determine a resident "
				"cluster capacity");
			Split split = Launch::resolve_split(
				gemm_params.tuning,
				gemm_params.tokens,
				RequiresRemote
					? uniform_vocab_capacity
					: gemm_params.local_vocab,
				max_active_cluster_pairs);
			if constexpr (RequiresRemote) {
				LIGER_CHECK(
					forward_wave_count_supported_sm100(
						split.num_waves),
					"configured local vocabulary capacity requires ",
					split.num_waves,
					" communication waves, but at most ",
					kForwardMaxWavesSm100,
					" are supported");
			}
			LIGER_CHECK(
				split.num_cluster_pairs <=
					max_active_cluster_pairs,
				"forward local reduction requires the complete "
				"cluster grid to remain resident");
			LIGER_CHECK(
				split.num_m_tiles <=
					backward_dx_resident_cta_capacity(),
				"forward wave synchronization exceeds the configured "
				"resident-CTA signal capacity");
			LIGER_CHECK(
				split.base_split_n >= 1 &&
					split.split_n <=
						split.num_logical_n_tiles &&
					split.split_n <=
						Config::kDefaultMaxSplitN,
				"split_n must be between 1 and the number "
				"of logical vocabulary tiles and fit the configured "
				"split workspace");
			LIGER_CHECK(
				split.extra_m_pairs <
					split.num_m_pairs &&
					split.split_n ==
						split.base_split_n +
							(split.extra_m_pairs != 0
								? 1
								: 0),
				"uneven split tuning must use "
				"split_n=base_split_n+1 for a proper subset "
				"of M pairs");

			std::size_t partial_rows =
				static_cast<std::size_t>(
					kForwardWaveSourceSlotsSm100) *
				static_cast<std::size_t>(
					split.num_m_tiles) *
				static_cast<std::size_t>(
					split.split_n) *
				static_cast<std::size_t>(
					Config::kCtaTileM);
			std::size_t required_bytes =
				partial_rows * sizeof(float) *
				(ReturnEntropy ? 4u : 3u);
			LIGER_CHECK(
				gemm_params.workspace != nullptr &&
					gemm_params.workspace_bytes >=
						required_bytes,
				"forward GEMM needs a ",
				required_bytes,
				" B split-partial workspace, got ",
				gemm_params.workspace_bytes);
			LIGER_CHECK(
				reinterpret_cast<std::uintptr_t>(
					gemm_params.workspace) %
						16 ==
					0,
				"forward GEMM workspace must be 16 B aligned");

			float* scratch =
				static_cast<float*>(
					gemm_params.workspace);
			Partials partials;
			partials.partial_max =
				scratch + 0 * partial_rows;
			partials.partial_sum =
				scratch + 1 * partial_rows;
			partials.partial_target =
				scratch + 2 * partial_rows;
			partials.partial_weighted =
				ReturnEntropy
				? scratch + 3 * partial_rows
				: nullptr;
			partials.ready =
				reduce_workspace.wave_partial_ready;

			int grid_ctas =
				split.num_cluster_pairs *
				Config::kClusterM;
			comm = reserve_dx_reduce_workspace(
				1,
				kDxRingStages,
				grid_ctas);
			std::size_t forward_comm_elements =
				static_cast<std::size_t>(
					RequiresRemote
						? kForwardWaveSourceSlotsSm100
						: 1) *
				static_cast<std::size_t>(
					split.num_m_tiles) *
				kForwardReducedFields *
				Config::kCtaTileM;
			LIGER_CHECK(
				forward_comm_elements * sizeof(float) <=
					backward_dx_configured_staging_bytes(),
				"forward local reduction exceeds the configured "
				"TP staging capacity");
			ForwardWaveWorkspaceSm100<Compute>
				wave_workspace = {};
			wave_workspace.diagnostics =
				reduce_workspace.diagnostics;
			if constexpr (RequiresRemote) {
				static_assert(
					Backend ==
					liger_cute::detail::
						LocalReduceBackend::kNvls);
				LIGER_CHECK(
					reduce_workspace.wave_partial_ready !=
							nullptr &&
						reduce_workspace.wave_tile_ready !=
							nullptr &&
						reduce_workspace.wave_slot_released !=
							nullptr,
					"forward remote wave epochs must be non-null");
				int rows_per_rank = ceil_div(
					gemm_params.tokens,
					mapping.size);
				int padded_tokens =
					rows_per_rank * mapping.size;
				std::size_t remote_elements =
					static_cast<std::size_t>(
						rows_per_rank) *
					static_cast<std::size_t>(
						1 +
						reduce_workspace.fields);
				std::size_t source_and_running_elements =
					remote_elements *
					static_cast<std::size_t>(
						kForwardWaveSourceSlotsSm100 + 1);
				LIGER_CHECK(
					source_and_running_elements <=
						reduce.remote.
							reduced_shard_elements &&
						source_and_running_elements *
								sizeof(float) <=
							tp_reduced_shard_configured_bytes(),
					"forward remote source slots and running state "
					"exceed the configured symmetric shard capacity");
				LIGER_CHECK(
					remote_elements <=
						reduce.remote.
							inbox_slot_elements &&
						remote_elements * sizeof(float) <=
							tp_remote_inbox_slot_configured_bytes(),
					"one forward wave exceeds the configured remote "
					"inbox slot capacity");
				std::size_t gathered_elements =
					remote_elements *
					static_cast<std::size_t>(
						mapping.size);
				LIGER_CHECK(
					gathered_elements * sizeof(float) <=
						backward_dx_configured_staging_bytes(),
					"forward remote all-gather exceeds the "
					"configured symmetric staging capacity");
				wave_workspace.tile_ready =
					reduce_workspace.wave_tile_ready;
				wave_workspace.slot_released =
					reduce_workspace.wave_slot_released;
				wave_workspace.launch_epoch =
					comm.launch_epoch;
				wave_workspace.source_slots =
					reduce.remote.reduced_shard;
				wave_workspace.running_state =
					wave_workspace.source_slots +
					static_cast<std::size_t>(
						kForwardWaveSourceSlotsSm100) *
						remote_elements;
				wave_workspace.source_slot_elements =
					remote_elements;
				wave_workspace.rows_per_rank =
					rows_per_rank;
				wave_workspace.padded_tokens =
					padded_tokens;
			}

			if constexpr (!RequiresRemote) {
				LIGER_CHECK(
					reduce_workspace.split_ready != nullptr,
					"forward split completion counters must be non-null");
				check_cuda_sm100(
					cudaMemsetAsync(
						reduce_workspace.split_ready,
						0,
						static_cast<std::size_t>(
							split.num_m_tiles) *
							sizeof(int),
						stream),
					"cudaMemsetAsync(forward split counters)");
			}
			if constexpr (kForwardDiagnosticTimestampsSm100) {
				LIGER_CHECK(
					wave_workspace.diagnostics != nullptr,
					"forward diagnostic storage must be non-null");
				check_cuda_sm100(
					cudaMemsetAsync(
						wave_workspace.diagnostics,
						0,
						static_cast<std::size_t>(
							kForwardDiagnosticEntriesSm100) *
							sizeof(std::uint64_t),
						stream),
					"cudaMemsetAsync(forward diagnostics)");
				check_cuda_sm100(
					cudaMemsetAsync(
						wave_workspace.diagnostics +
							kForwardDiagnosticKernelStartSm100,
						0xff,
						sizeof(std::uint64_t),
						stream),
					"cudaMemsetAsync(forward diagnostic start)");
			}
			liger_cute::detail::begin_tp_reduce(
				comm.launch_epoch,
				stream);
			dim3 grid(
				static_cast<unsigned>(Config::kClusterM),
				1u,
				static_cast<unsigned>(
					split.num_cluster_pairs));
			if constexpr (RequiresRemote) {
				liger_cute::detail::synchronize_tp_reduce(
					stream);
				check_cuda_sm100(
					ClusterLaunchSm100::launch_cooperative(
						kernel,
						grid,
						Config::kNumThreads,
						kSmemBytes,
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
						wave_workspace,
						reduce.remote,
						outputs),
					"cudaLaunchKernelEx(cooperative "
					"forward_gemm_tp_kernel_sm100)");
			} else {
				check_cuda_sm100(
					ClusterLaunchSm100::launch(
						kernel,
						grid,
						Config::kNumThreads,
						kSmemBytes,
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
						wave_workspace,
						reduce.remote,
						outputs),
					"cudaLaunchKernelEx("
					"forward_gemm_tp_kernel_sm100)");
			}
		};

	if (
		reduce.backend ==
		liger_cute::detail::LocalReduceBackend::kNvls) {
		if (reduce.remote.enabled()) {
			launch_local(
				reduce.nvls,
				std::integral_constant<
					liger_cute::detail::LocalReduceBackend,
					liger_cute::detail::
						LocalReduceBackend::kNvls>{},
				std::true_type{});
		} else {
			launch_local(
				reduce.nvls,
				std::integral_constant<
					liger_cute::detail::LocalReduceBackend,
					liger_cute::detail::
						LocalReduceBackend::kNvls>{},
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
				liger_cute::detail::
					LocalReduceBackend::kDirectPeer>{},
			std::false_type{});
	}

	liger_cute::detail::end_tp_reduce(stream);
}

}  // namespace

template <bool ReturnEntropy, int Compute>
void fused_linear_scaled_cross_entropy_forward_sm100(
		const ForwardTpParamsSm100<Compute>& params,
		cudaStream_t stream) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	launch_forward_sm100<ReturnEntropy, Compute>(
		params,
		stream);
}

template void fused_linear_scaled_cross_entropy_forward_sm100<
	false,
	100>(
	const ForwardTpParamsSm100<100>&,
	cudaStream_t);
template void fused_linear_scaled_cross_entropy_forward_sm100<
	true,
	100>(
	const ForwardTpParamsSm100<100>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
