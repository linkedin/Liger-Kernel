#pragma once

// SM100 forward kernel entry point. As in the fused MoE kernels, the top-level
// body owns synchronization and makes the warp-role split explicit; role
// implementations live in forward_gemm_roles_sm100.cuh.

#include "forward_gemm_roles_sm100.cuh"

#include <cute/arch/tmem_allocator_sm100.hpp>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

template <
	bool ReturnEntropy,
	int Compute,
	bool RequiresRemote,
	liger_cute::detail::LocalReduceBackend Backend,
	class TmaLoadX,
	class TmaLoadW,
	class Mapping>
__global__ __launch_bounds__(
	ForwardGemmConfigSm100<
		Compute>::kNumThreads,
	1) __cluster_dims__(2, 1, 1)
void forward_gemm_tp_kernel_sm100(
		__grid_constant__ const TmaLoadX tma_load_x,
		__grid_constant__ const TmaLoadW tma_load_w,
		__grid_constant__ const ForwardGemmParamsSm100<
			Compute> params,
		__grid_constant__ const ForwardGemmPartialsSm100<
			Compute> partials,
		__grid_constant__ const ForwardGemmSplitSm100<
			Compute> split,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const Mapping mapping,
		int* split_ready,
		float* global_max,
		float* reduced,
		__grid_constant__ const ForwardWaveWorkspaceSm100<
			Compute> wave_workspace,
		__grid_constant__ const liger_cute::detail::
			RemoteReduceView remote,
		__grid_constant__ const ForwardFinalOutputsSm100 outputs) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	static_assert(
		!RequiresRemote ||
			Backend ==
				liger_cute::detail::
					LocalReduceBackend::kNvls,
		"the fused SM100 remote path requires node-local NVLS");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Smem = ForwardGemmSmemSm100<
		Compute,
		ReturnEntropy>;

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);
	int warp_id =
		static_cast<int>(threadIdx.x) / kWarpSize;
	ForwardWarpRoleSm100 role =
		forward_warp_role_sm100(warp_id);

	cute::prefetch_tma_descriptor(
		tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_w.get_tma_descriptor());
	auto pipe =
		forward_make_pipe_sm100<Compute>(
			smem.pipeline);

	cute::TMEM::Allocator2Sm tmem_allocator;
	// Every role participates in startup before the reduction and communication
	// warps diverge from the compute pipeline.
	cute::cluster_sync();
	if (role == ForwardWarpRoleSm100::kEpilogue &&
			warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.allocate(
			Config::kTmemColumns,
			&smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();

	ForwardGemmWorkSm100<Compute> work =
		forward_gemm_assign_work_sm100<Compute>(
			split,
			static_cast<int>(blockIdx.z),
			static_cast<int>(
				cute::block_rank_in_cluster()));
	int num_k_tiles =
		ForwardGemmLaunchSm100<
			Compute>::num_k_tiles(params.hidden);
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (
			role == ForwardWarpRoleSm100::kTmaProducer &&
			(threadIdx.x & (kWarpSize - 1)) == 0) {
			forward_diagnostic_min_sm100(
				wave_workspace.diagnostics,
				kForwardDiagnosticKernelStartSm100,
				forward_globaltimer_sm100());
		}
	}

	typename Traits::PipelineState state;
	switch (role) {
	case ForwardWarpRoleSm100::kLocalReduceControl:
		if constexpr (RequiresRemote) {
			forward_reduce_waves_sm100<
				ReturnEntropy,
				Compute,
				Backend>(
					params,
					partials,
					split,
					comm,
					mapping,
					wave_workspace,
					work);
		}
		break;

	case ForwardWarpRoleSm100::kRemoteCommunication:
		if constexpr (RequiresRemote) {
			if (
				blockIdx.x == 0 &&
				blockIdx.y == 0 &&
				blockIdx.z == 0 &&
				cute::block_rank_in_cluster() == 0) {
#if defined(LIGER_CUTE_FSLCE_SM100_ENABLE_NVSHMEM)
				forward_communicate_waves_sm100<
					ReturnEntropy,
					Compute>(
						params,
						split,
						comm,
						mapping,
						remote,
						wave_workspace,
						outputs);
#else
				__trap();
#endif
			}
		}
		break;

	case ForwardWarpRoleSm100::kTmaProducer:
		state =
			cutlass::make_producer_start_state<
				typename Traits::MainloopPipeline>();
		forward_tma_role_sm100<
			ReturnEntropy,
			RequiresRemote,
			Compute>(
				pipe,
				state,
				smem,
				tma_load_x,
				tma_load_w,
				params,
				work,
				split,
				wave_workspace,
				num_k_tiles);
		break;

	case ForwardWarpRoleSm100::kUmmaProducer:
		forward_compute_role_sm100<
			ForwardWarpRoleSm100::kUmmaProducer,
			ReturnEntropy,
			RequiresRemote,
			Compute>(
				pipe,
				state,
				smem,
				params,
				partials,
				work,
				split,
				wave_workspace,
				num_k_tiles);
		break;

	case ForwardWarpRoleSm100::kEpilogue:
		forward_compute_role_sm100<
			ForwardWarpRoleSm100::kEpilogue,
			ReturnEntropy,
			RequiresRemote,
			Compute>(
				pipe,
				state,
				smem,
				params,
				partials,
				work,
				split,
				wave_workspace,
				num_k_tiles);
		break;

	case ForwardWarpRoleSm100::kInactive:
		break;
	}

	if constexpr (RequiresRemote) {
		bool compute_role =
			role == ForwardWarpRoleSm100::kUmmaProducer ||
			role == ForwardWarpRoleSm100::kEpilogue;
		if (compute_role) {
			// Warps 0 and 1 continue their reduction/communication pipelines,
			// while warp 2 has already drained the TMA producer tail.
			constexpr int kUmmaEpilogueThreads =
				(Config::kLastEpilogueWarp -
					Config::kUmmaWarp + 1) *
				kWarpSize;
			static_assert(kUmmaEpilogueThreads == 288);
			cutlass::arch::NamedBarrier::sync(
				kUmmaEpilogueThreads,
				Config::kComputeDoneBarrierId);
			if (
				warp_id ==
				Config::kFirstEpilogueWarp) {
				tmem_allocator.release_allocation_lock();
				tmem_allocator.free(
					smem.tmem_base,
					Config::kTmemColumns);
			}
		}
		return;
	}

	// The node-local path finalizes after every warp has left its role.
	__syncthreads();
	cute::cluster_sync();
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(
			smem.tmem_base,
			Config::kTmemColumns);
	}
	__syncthreads();

	forward_finalize_splits_and_reduce_local_sm100<
		ReturnEntropy,
		Compute,
		Backend>(
			smem,
			params,
			partials,
			split,
			comm,
			mapping,
			split_ready,
			global_max,
			reduced,
			wave_workspace.diagnostics,
			outputs);
	cute::cluster_sync();
#else
	__trap();
#endif
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
