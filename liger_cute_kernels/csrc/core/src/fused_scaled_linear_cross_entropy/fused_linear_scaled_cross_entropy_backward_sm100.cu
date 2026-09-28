// ═══════════════════════════════════════════════════════════════════════════
// SM100 fused scaled linear cross entropy backward host dispatch.
//
// One persistent cluster-2 launch runs every wave; there is no per-wave kernel
// launch and no separate finalizer. The launcher only builds the nine TMA
// descriptors, resolves the reduction plan, enforces the full-residency
// invariant the in-kernel grid barrier depends on, and clears the device-side
// signal block.
// ═══════════════════════════════════════════════════════════════════════════

#include "backward_launch_sm100.cuh"

#include "backward_gemm_sm100.cuh"
#include "buffer_pool.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"
#include "workspace.cuh"

#include <cuda_runtime.h>
#include <cute/atom/copy_traits_sm100_tma.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

using namespace cute;
using backward_sm100::Bundle;
using backward_sm100::check_cuda;
using backward_sm100::ClusterLaunch;
using backward_sm100::HostCommConfig;
using backward_sm100::HostConfig;
using backward_sm100::HostElement;
using backward_sm100::HostLaunch;
using backward_sm100::HostTraits;

// The device-private signal block: one monotonic grid-barrier counter, one
// monotonic dX reduce-scatter completion counter and one inter-host ring
// completion epoch. Fixed size, pooled, never per-launch allocated.
BackwardWaveWorkspaceSm100<100> reserve_backward_signals_sm100() {
	auto& pool = liger_cute::detail::global_buffer_pool();
	auto* signals = static_cast<std::uint64_t*>(pool.get_device(
		BackwardSymmetricNames::kBackwardSm100Signals,
		static_cast<std::size_t>(kBackwardSignalEntries) *
			sizeof(std::uint64_t)));
	BackwardWaveWorkspaceSm100<100> workspace = {};
	workspace.grid_barrier = signals + kBackwardSignalGridBarrier;
	workspace.dx_scatter_ready = signals + kBackwardSignalScatterBase;
	workspace.dx_remote_received =
		signals + kBackwardSignalRemoteReceived;
	workspace.dx_remote_ready = signals + kBackwardSignalRemoteBase;
	workspace.dx_remote_merge_arrived =
		signals + kBackwardSignalRemoteMergeArrived;
	if constexpr (kBackwardDiagnosticTimestampsSm100) {
		workspace.diagnostics =
			static_cast<std::uint64_t*>(pool.get_device(
				BackwardSymmetricNames::kBackwardSm100Diagnostics,
				static_cast<std::size_t>(
					kBackwardDiagnosticEntries) *
					sizeof(std::uint64_t)));
	}
	return workspace;
}

void validate(const BackwardTpParamsSm100<100>& params) {
	const auto& gemm = params.gemm;
	LIGER_CHECK(gemm.tokens >= 0, "tokens must be non-negative");
	LIGER_CHECK(gemm.hidden > 0, "hidden must be positive");
	LIGER_CHECK(gemm.local_vocab > 0, "local_vocab must be positive");
	LIGER_CHECK(
		gemm.x != nullptr && gemm.weight != nullptr &&
			gemm.target != nullptr && gemm.grad_output != nullptr &&
			gemm.lse != nullptr,
		"backward inputs x / weight / target / grad_output / lse must be "
		"non-null");
	LIGER_CHECK(
		gemm.grad_input != nullptr && gemm.grad_weight != nullptr,
		"backward outputs grad_input and grad_weight must be non-null");
	LIGER_CHECK(
		gemm.dz_workspace != nullptr,
		"the dZ wave workspace must be reserved");
	LIGER_CHECK(
		params.tiles_per_reduce == 1 || params.tiles_per_reduce == 2 ||
			params.tiles_per_reduce == 4,
		"TilesPerReduce must be 1, 2 or 4, got ",
		params.tiles_per_reduce);
	LIGER_CHECK(
		params.num_comm_channels >= 1,
		"num_comm_channels must be positive (legacy compatibility argument)");

	auto aligned = [](const void* pointer) {
		return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
	};
	LIGER_CHECK(
		aligned(gemm.x) && aligned(gemm.weight) &&
			aligned(gemm.grad_weight) && aligned(gemm.dz_workspace),
		"backward TMA BF16 operands must be 16 B aligned");
	LIGER_CHECK(
		reinterpret_cast<std::uintptr_t>(gemm.grad_input) % 8 == 0,
		"grad_input must be 8 B aligned for vectorized BF16 stores");
	LIGER_CHECK(
		(static_cast<std::size_t>(gemm.hidden) * sizeof(HostElement)) % 16 ==
			0,
		"hidden * sizeof(bfloat16) must be a multiple of 16 B for TMA");
	// The MN-major dX / dW operands address W and X transposed, so the leading
	// dimension of the transposed view must also satisfy the TMA box rules.
	LIGER_CHECK(
		gemm.hidden % HostConfig::kTileK == 0 ||
			gemm.hidden % 8 == 0,
		"hidden must be a multiple of 8 for the transposed TMA views");
	validate_backward_tp_shape(gemm.tokens, gemm.hidden, gemm.local_vocab);

	std::size_t needed = HostLaunch::dz_workspace_bytes(gemm.local_vocab);
	LIGER_CHECK(
		gemm.dz_workspace_bytes >= needed,
		"the dZ wave workspace needs ",
		needed,
		" B, got ",
		gemm.dz_workspace_bytes);
	LIGER_CHECK(
		backward_wave_count_supported_sm100(
			HostLaunch::num_waves(gemm.tokens)),
		"the fused SM100 backward supports at most ",
		kBackwardMaxWavesSm100,
		" token waves");
}

template <bool ReturnEntropy, bool EnableLocalReduce, bool RequiresRemote>
void launch_instance(
		const BackwardTpParamsSm100<100>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		cudaStream_t stream) {
	using Smem = BackwardGemmSmemSm100<100, ReturnEntropy>;

	const auto& gemm = params.gemm;
	int padded_vocab = HostLaunch::padded_vocab(gemm.local_vocab);
	int num_waves = HostLaunch::num_waves(gemm.tokens);

	auto* kernel = &backward_gemm_tp_kernel_sm100<
		ReturnEntropy,
		100,
		RequiresRemote,
		EnableLocalReduce,
		HostCommConfig,
		Bundle>;
	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	ClusterLaunch::prepare(kernel, kSmemBytes);

	int max_active_cluster_pairs =
		ClusterLaunch::max_active_clusters(
			kernel, HostConfig::kNumThreads, kSmemBytes);
	LIGER_CHECK(
		max_active_cluster_pairs > 0,
		"SM100 backward could not determine a resident cluster capacity");

	int dz_pairs = HostLaunch::num_dz_cluster_pairs(gemm.local_vocab);
	int dx_pairs = HostLaunch::num_dx_cluster_pairs(gemm.hidden);
	int dw_pairs =
		HostLaunch::num_dw_cluster_pairs(gemm.hidden, gemm.local_vocab);
	int work_pairs = dz_pairs;
	if (dx_pairs > work_pairs) work_pairs = dx_pairs;
	if (dw_pairs > work_pairs) work_pairs = dw_pairs;
	int cluster_pairs = work_pairs < max_active_cluster_pairs
		? work_pairs
		: max_active_cluster_pairs;
	LIGER_CHECK(
		cluster_pairs >= 1,
		"SM100 backward requires at least one resident cluster pair");
	int grid_ctas = cluster_pairs * HostConfig::kClusterM;
	// Strict full-residency invariant: the in-kernel software grid barrier is
	// only safe if every launched cluster is co-resident.
	LIGER_CHECK(
		cluster_pairs <= max_active_cluster_pairs,
		"SM100 backward requires the complete cluster grid to stay resident");
	LIGER_CHECK(
		grid_ctas <= backward_dx_resident_cta_capacity(),
		"SM100 backward grid ",
		grid_ctas,
		" exceeds the configured resident CTA staging capacity ",
		backward_dx_resident_cta_capacity());

	DxReduceWorkspace<float> comm =
		reserve_dx_reduce_workspace(
			params.tiles_per_reduce, kDxRingStages, grid_ctas);
	// The CTA-owned dX staging arena must cover the highest slot this grid can
	// address, otherwise the epilogue's FP32 stores run off the end of the
	// pooled buffer.
	std::size_t staging_high_water =
		(dx_slot_offset<HostCommConfig>(
			grid_ctas - 1, kDxCommWarpsPerChannel - 1, kDxRingStages - 1) +
			static_cast<std::size_t>(HostCommConfig::kTileElements)) *
		sizeof(float);
	LIGER_CHECK(
		staging_high_water <= backward_dx_configured_staging_bytes(),
		"SM100 backward dX staging needs ",
		staging_high_water,
		" B for a ",
		grid_ctas,
		"-CTA grid, configured ",
		backward_dx_configured_staging_bytes());

	backward_sm100::TmaOperands tma_operands =
		backward_sm100::operands_of(gemm);
	tma_operands.dx_staging = comm.partial;
	tma_operands.dx_staging_rows = backward_sm100::dx_staging_rows();
	Bundle bundle = backward_sm100::make_tma_bundle(tma_operands);

	BackwardWaveWorkspaceSm100<100> wave_workspace =
		reserve_backward_signals_sm100();
	wave_workspace.launch_epoch = comm.launch_epoch;
	wave_workspace.packed_shard = reduce.nvls.reduced_shard;
	wave_workspace.grid_ctas = grid_ctas;
	wave_workspace.staging_rows = tma_operands.dx_staging_rows;
	wave_workspace.num_waves = num_waves;
	if constexpr (kBackwardSyncVariantSm100 == 2) {
		wave_workspace.dz_tile_ready_entries =
			static_cast<std::size_t>(num_waves) *
			static_cast<std::size_t>(dz_pairs);
		auto& pool = liger_cute::detail::global_buffer_pool();
		wave_workspace.dz_tile_ready =
			static_cast<std::uint32_t*>(pool.get_device(
				BackwardSymmetricNames::kBackwardSm100DzTileReady,
				wave_workspace.dz_tile_ready_entries *
					sizeof(std::uint32_t)));
		check_cuda(
			cudaMemsetAsync(
				wave_workspace.dz_tile_ready,
				0,
				wave_workspace.dz_tile_ready_entries *
					sizeof(std::uint32_t),
				stream),
			"cudaMemsetAsync(SM100 backward dZ tile epochs)");
	}

	std::size_t packed_elements =
		static_cast<std::size_t>(num_waves) *
		static_cast<std::size_t>(
			HostLaunch::dx_tiles_per_wave(gemm.hidden)) *
		static_cast<std::size_t>(HostCommConfig::kTileElements) /
		static_cast<std::size_t>(reduce.nvls.size);
	wave_workspace.packed_shard_elements = packed_elements;
	LIGER_CHECK(
		wave_workspace.packed_shard != nullptr,
		"SM100 backward requires the pooled packed dX shard");
	LIGER_CHECK(
		packed_elements * sizeof(float) <=
			tp_reduced_shard_configured_bytes(),
		"SM100 backward packed dX shard needs ",
		packed_elements * sizeof(float),
		" B, configured ",
		tp_reduced_shard_configured_bytes());
	std::size_t full_elements =
		packed_elements * static_cast<std::size_t>(reduce.nvls.size);
	LIGER_CHECK(
		full_elements * sizeof(float) <=
			backward_dx_configured_durable_bytes(),
		"SM100 backward dX all-gather destination exceeds the configured "
		"durable workspace");

	if constexpr (RequiresRemote) {
		std::size_t chunk_elements = packed_elements /
			static_cast<std::size_t>(num_waves);
		LIGER_CHECK(
			reduce.remote.enabled(),
			"SM100 backward remote dispatch requires a configured "
			"inter-host ring");
		LIGER_CHECK(
			chunk_elements <= reduce.remote.inbox_slot_elements &&
				chunk_elements * sizeof(float) <=
					tp_remote_inbox_slot_configured_bytes(),
			"one SM100 backward dX chunk exceeds the configured remote inbox "
			"slot capacity");
		LIGER_CHECK(
			packed_elements <= reduce.remote.reduced_shard_elements,
			"the SM100 backward packed dX shard exceeds the configured "
			"symmetric shard capacity");
		// Source/result slot accounting for the chunk pipeline. Warp 0 defers
		// each chunk's all-gather by exactly one chunk, so at most
		// kBackwardDxChunkPipelineDepth chunks are live at any time. Both the
		// packed shard and the all-gather destination are addressed by
		// absolute chunk index, so the slot space is `num_waves` deep and a
		// live chunk's storage is never recycled underneath it.
		LIGER_CHECK(
			num_waves >= 1,
			"the SM100 backward chunk pipeline needs at least one chunk");
		LIGER_CHECK(
			static_cast<std::size_t>(num_waves) * chunk_elements <=
				reduce.remote.reduced_shard_elements,
			"the SM100 backward chunk slot space must hold every chunk of the "
			"launch so an in-flight chunk is never overwritten");
		static_assert(
			kBackwardDxChunkPipelineDepth <=
				liger_cute::detail::kRemoteRingInboxSlots,
			"the inter-host ring must buffer every live dX chunk");
	}

	check_cuda(
		cudaMemsetAsync(
			wave_workspace.grid_barrier,
			0,
			static_cast<std::size_t>(kBackwardSignalEntries) *
				sizeof(std::uint64_t),
			stream),
		"cudaMemsetAsync(SM100 backward signal block)");
	if constexpr (kBackwardDiagnosticTimestampsSm100) {
		check_cuda(
			cudaMemsetAsync(
				wave_workspace.diagnostics,
				0,
				static_cast<std::size_t>(
					kBackwardDiagnosticEntries) *
					sizeof(std::uint64_t),
				stream),
			"cudaMemsetAsync(SM100 backward diagnostics)");
	}
	if constexpr (EnableLocalReduce) {
		if (reduce.nvls.size > 1 || RequiresRemote) {
			liger_cute::detail::begin_tp_reduce(
				comm.launch_epoch, stream);
		}
	}
	dim3 grid(
		static_cast<unsigned>(HostConfig::kClusterM),
		1u,
		static_cast<unsigned>(cluster_pairs));
	if constexpr (RequiresRemote) {
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)
		liger_cute::detail::synchronize_tp_reduce(stream);
		check_cuda(
			ClusterLaunch::launch_cooperative(
				kernel,
				grid,
				HostConfig::kNumThreads,
				kSmemBytes,
				stream,
				bundle,
				gemm,
				comm,
				reduce.nvls,
				reduce.remote,
				wave_workspace),
			"cudaLaunchKernelEx(cooperative "
			"backward_gemm_tp_kernel_sm100)");
#else
		LIGER_CHECK(
			false,
			"SM100 backward was built without the inter-host ring; rebuild "
			"with LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM=1");
#endif
	} else {
		check_cuda(
			ClusterLaunch::launch(
				kernel,
				grid,
				HostConfig::kNumThreads,
				kSmemBytes,
				stream,
				bundle,
				gemm,
				comm,
				reduce.nvls,
				reduce.remote,
				wave_workspace),
			"cudaLaunchKernelEx(backward_gemm_tp_kernel_sm100)");
	}
	if constexpr (EnableLocalReduce) {
		if (reduce.nvls.size > 1 || RequiresRemote) {
			liger_cute::detail::end_tp_reduce(stream);
		}
	}
}

template <bool ReturnEntropy>
void dispatch_instance(
		const BackwardTpParamsSm100<100>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		cudaStream_t stream) {
	LIGER_CHECK(
		reduce.backend == liger_cute::detail::LocalReduceBackend::kNvls,
		"the fused SM100 backward requires a node-local NVLS team");
	if (reduce.remote.enabled()) {
		LIGER_CHECK(
			(reduce.nvls.size == 1 ||
				reduce.nvls.size == 2 ||
				reduce.nvls.size == 4 ||
				reduce.nvls.size == 8) &&
				reduce.remote.size == 2,
			"the SM100 remote backward requires two uniform hosts with "
			"1, 2, 4, or 8 selected GPUs per host, got TP",
			reduce.team_size,
			" with local size ",
			reduce.nvls.size,
			" and remote size ",
			reduce.remote.size);
		LIGER_CHECK(
			reduce.team_size ==
				reduce.nvls.size * reduce.remote.size,
			"inconsistent SM100 hierarchical reduction topology");
		launch_instance<ReturnEntropy, true, true>(
			params, reduce, stream);
	} else {
		LIGER_CHECK(
			reduce.nvls.size == 1 || reduce.nvls.size == 2 ||
				reduce.nvls.size == 4 || reduce.nvls.size == 8,
			"the SM100 node-local backward supports TP1/2/4/8, got ",
			reduce.nvls.size);
		launch_instance<ReturnEntropy, true, false>(
			params, reduce, stream);
	}
}

}  // namespace

template <bool ReturnEntropy, int Compute>
void fused_linear_scaled_cross_entropy_backward_sm100(
		const BackwardTpParamsSm100<Compute>& params, cudaStream_t stream) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	if (params.gemm.tokens == 0) return;
	validate(params);
	LIGER_CHECK(
		params.team_handle == 0 ||
			params.team_handle == backward_dx_team_handle(),
		"backward TP team must match the configured reduction team");

	liger_cute::detail::TpReducePlan reduce =
		liger_cute::detail::tp_reduce_plan();
	dispatch_instance<ReturnEntropy>(params, reduce, stream);
}

template void fused_linear_scaled_cross_entropy_backward_sm100<false, 100>(
	const BackwardTpParamsSm100<100>&,
	cudaStream_t);
template void fused_linear_scaled_cross_entropy_backward_sm100<true, 100>(
	const BackwardTpParamsSm100<100>&,
	cudaStream_t);

void fused_linear_scaled_cross_entropy_backward_diagnostics_sm100(
		std::uint64_t* output,
		std::size_t entries,
		cudaStream_t stream) {
	LIGER_CHECK(
		output != nullptr && entries >= kBackwardDiagnosticEntries,
		"SM100 backward diagnostic output must hold ",
		kBackwardDiagnosticEntries,
		" uint64 entries");
	auto& pool = liger_cute::detail::global_buffer_pool();
	auto* diagnostics =
		static_cast<std::uint64_t*>(pool.get_device(
			BackwardSymmetricNames::kBackwardSm100Diagnostics,
			static_cast<std::size_t>(kBackwardDiagnosticEntries) *
				sizeof(std::uint64_t)));
	check_cuda(
		cudaMemcpyAsync(
			output,
			diagnostics,
			static_cast<std::size_t>(kBackwardDiagnosticEntries) *
				sizeof(std::uint64_t),
			cudaMemcpyDeviceToDevice,
			stream),
		"cudaMemcpyAsync(SM100 backward diagnostics)");
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
