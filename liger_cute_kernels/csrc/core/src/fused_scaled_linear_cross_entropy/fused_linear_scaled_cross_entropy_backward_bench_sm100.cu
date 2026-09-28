// ═══════════════════════════════════════════════════════════════════════════
// SM100 fused backward — benchmark-only per-phase launch paths.
//
// Each entry point runs exactly one of the three GEMM phases through the
// production kernel template with a single phase bit set. Tile shapes, the
// mainloop pipeline, the accumulator pipeline, the TMEM plan, the epilogues,
// the wave loop and both full-grid barriers are the production ones; only the
// other two phases and the dX reduction/communication dependencies are
// compiled out. The TMA descriptors come from the same shared helper the
// production launcher uses, so an isolated phase cannot measure a different
// operand layout or box size than production executes.
//
// Required epilogue work is retained in full:
//   dZ   softmax / entropy gradient transform, BF16 convert, TMA store to the
//        dZ wave workspace (including the zeroed padded columns and rows).
//   dX   TMEM -> register -> FP32 stores into the CTA-owned staging arena.
//   dW   TMEM -> BF16 -> SMEM -> TMA store (wave 0) / TMA reduce-add (later).
// ═══════════════════════════════════════════════════════════════════════════

#include "backward_launch_sm100.cuh"

#include "backward_gemm_sm100.cuh"
#include "buffer_pool.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"
#include "workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

using namespace cute;
using backward_sm100::Bundle;
using backward_sm100::check_cuda;
using backward_sm100::ClusterLaunch;
using backward_sm100::HostCommConfig;
using backward_sm100::HostConfig;
using backward_sm100::HostLaunch;

BackwardWaveWorkspaceSm100<100> reserve_bench_signals() {
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
	return workspace;
}

template <bool ReturnEntropy, int PhaseMask>
void launch_phase(
		const BackwardTpParamsSm100<100>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		cudaStream_t stream) {
	static_assert(
		backward_phase_is_isolated_sm100(PhaseMask),
		"unsupported benchmark phase mask");
	using Smem = BackwardGemmSmemSm100<100, ReturnEntropy>;

	const auto& gemm = params.gemm;
	int num_waves = HostLaunch::num_waves(gemm.tokens);
	auto* kernel = &backward_gemm_tp_kernel_sm100<
		ReturnEntropy,
		100,
		false,
		false,
		HostCommConfig,
		Bundle,
		PhaseMask>;
	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	ClusterLaunch::prepare(kernel, kSmemBytes);

	int max_active_cluster_pairs =
		ClusterLaunch::max_active_clusters(
			kernel, HostConfig::kNumThreads, kSmemBytes);
	LIGER_CHECK(
		max_active_cluster_pairs > 0,
		"SM100 backward benchmark could not determine a resident cluster "
		"capacity");
	// Production grid: the fused kernel sizes its grid by the widest phase, so
	// an isolated measurement must launch the same grid to reproduce the
	// occupancy and tail behaviour production actually sees.
	int work_pairs =
		backward_sm100::work_cluster_pairs(gemm, kBackwardPhaseAll);
	int cluster_pairs = work_pairs < max_active_cluster_pairs
		? work_pairs
		: max_active_cluster_pairs;
	LIGER_CHECK(
		cluster_pairs >= 1,
		"SM100 backward benchmark needs at least one resident cluster pair");
	int grid_ctas = cluster_pairs * HostConfig::kClusterM;
	LIGER_CHECK(
		grid_ctas <= backward_dx_resident_cta_capacity(),
		"SM100 backward benchmark grid exceeds the resident CTA capacity");

	DxReduceWorkspace<float> comm = reserve_dx_reduce_workspace(
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

	BackwardWaveWorkspaceSm100<100> wave_workspace = reserve_bench_signals();
	wave_workspace.launch_epoch = comm.launch_epoch;
	wave_workspace.packed_shard = reduce.nvls.reduced_shard;
	wave_workspace.grid_ctas = grid_ctas;
	wave_workspace.staging_rows = tma_operands.dx_staging_rows;
	wave_workspace.num_waves = num_waves;
	if constexpr (kBackwardSyncVariantSm100 == 2) {
		int dz_pairs =
			HostLaunch::num_dz_cluster_pairs(gemm.local_vocab);
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
			"cudaMemsetAsync(SM100 backward benchmark dZ tile epochs)");
	}

	check_cuda(
		cudaMemsetAsync(
			wave_workspace.grid_barrier,
			0,
			static_cast<std::size_t>(kBackwardSignalEntries) *
				sizeof(std::uint64_t),
			stream),
		"cudaMemsetAsync(SM100 backward benchmark signal block)");
	// No reduction runs here, so the benchmark deliberately does not enter the
	// tensor-parallel epoch protocol: repeated back-to-back begin/end pairs
	// are a launch-rate hazard and contribute nothing to the measurement.
	dim3 grid(
		static_cast<unsigned>(HostConfig::kClusterM),
		1u,
		static_cast<unsigned>(cluster_pairs));
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
		"cudaLaunchKernelEx(backward_gemm_tp_kernel_sm100 benchmark)");
}

template <bool ReturnEntropy>
void dispatch_phase(
		const BackwardTpParamsSm100<100>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		int phase_mask,
		cudaStream_t stream) {
	switch (phase_mask) {
		case 0:
			launch_phase<ReturnEntropy, 0>(
				params, reduce, stream);
			return;
		case kBackwardPhaseDz:
			launch_phase<ReturnEntropy, kBackwardPhaseDz>(
				params, reduce, stream);
			return;
		case kBackwardPhaseDx:
			launch_phase<ReturnEntropy, kBackwardPhaseDx>(
				params, reduce, stream);
			return;
		case kBackwardPhaseDw:
			launch_phase<ReturnEntropy, kBackwardPhaseDw>(
				params, reduce, stream);
			return;
		case kBackwardPhaseDz | kBackwardPhaseDx:
			launch_phase<
				ReturnEntropy, kBackwardPhaseDz | kBackwardPhaseDx>(
					params, reduce, stream);
			return;
		case kBackwardPhaseDz | kBackwardPhaseDw:
			launch_phase<
				ReturnEntropy, kBackwardPhaseDz | kBackwardPhaseDw>(
					params, reduce, stream);
			return;
		case kBackwardPhaseDx | kBackwardPhaseDw:
			launch_phase<
				ReturnEntropy, kBackwardPhaseDx | kBackwardPhaseDw>(
					params, reduce, stream);
			return;
		case kBackwardPhaseDx | kBackwardPhaseDw |
			kBackwardPhaseAuditSkipDxDwBarrier:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDx | kBackwardPhaseDw |
					kBackwardPhaseAuditSkipDxDwBarrier>(
						params, reduce, stream);
			return;
		case kBackwardPhaseDz | kBackwardPhaseDx |
			kBackwardPhaseAuditSkipDzGridBarrier:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDz | kBackwardPhaseDx |
					kBackwardPhaseAuditSkipDzGridBarrier>(
						params, reduce, stream);
			return;
		case kBackwardPhaseDz | kBackwardPhaseDw |
			kBackwardPhaseAuditSkipDzGridBarrier:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDz | kBackwardPhaseDw |
					kBackwardPhaseAuditSkipDzGridBarrier>(
						params, reduce, stream);
			return;
		case kBackwardPhaseDz | kBackwardPhaseDx |
			kBackwardPhaseAuditSkipDzDrain:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDz | kBackwardPhaseDx |
					kBackwardPhaseAuditSkipDzDrain>(
						params, reduce, stream);
			return;
		case kBackwardPhaseDz |
			kBackwardPhaseAuditForceDzGridBarrier:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDz |
					kBackwardPhaseAuditForceDzGridBarrier>(
						params, reduce, stream);
			return;
		case kBackwardPhaseDx |
			kBackwardPhaseAuditForceDxDwBarrier:
			launch_phase<
				ReturnEntropy,
				kBackwardPhaseDx |
					kBackwardPhaseAuditForceDxDwBarrier>(
						params, reduce, stream);
			return;
		default:
			LIGER_CHECK(
				false,
				"SM100 backward benchmark phase must be empty (0), dZ (1), "
				"dX (2), dZ|dX (3), dW (4), dZ|dW (5), dX|dW (6), or "
				"audit dX|dW without handoff (14), got ",
				phase_mask);
	}
}

}  // namespace

void fused_linear_scaled_cross_entropy_backward_phase_bench_sm100(
		const BackwardTpParamsSm100<100>& params,
		bool return_entropy,
		int phase_mask,
		cudaStream_t stream) {
	if (params.gemm.tokens == 0) return;
	LIGER_CHECK(
		params.gemm.hidden > 0 && params.gemm.local_vocab > 0,
		"benchmark shapes must be positive");
	LIGER_CHECK(
		params.gemm.dz_workspace != nullptr,
		"the dZ wave workspace must be reserved");
	validate_backward_tp_shape(
		params.gemm.tokens, params.gemm.hidden, params.gemm.local_vocab);
	LIGER_CHECK(
		backward_wave_count_supported_sm100(
			HostLaunch::num_waves(params.gemm.tokens)),
		"the fused SM100 backward supports at most ",
		kBackwardMaxWavesSm100,
		" token waves");

	liger_cute::detail::TpReducePlan reduce =
		liger_cute::detail::tp_reduce_plan();
	if (return_entropy) {
		dispatch_phase<true>(params, reduce, phase_mask, stream);
	} else {
		dispatch_phase<false>(params, reduce, phase_mask, stream);
	}
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
