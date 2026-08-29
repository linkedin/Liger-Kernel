#pragma once

#include <cuda_runtime.h>

#include "backward_gemm_sm90.cuh"
#include "dx_reduce_sm90.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

void launch_backward_dx_dw_fallback_wave_sm90(
	bool return_entropy,
	int tiles_per_reduce,
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const DxFallbackMapping& mapping,
	int grid,
	cudaStream_t stream,
	int wave);

void launch_backward_dx_dw_hierarchical_wave_sm90(
	bool return_entropy,
	int tiles_per_reduce,
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const DxHierarchicalMapping& mapping,
	int grid,
	cudaStream_t stream,
	int wave);

void finalize_backward_dx_cluster_local_sm90(
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const DxHierarchicalMapping& mapping,
	int grid,
	int num_waves,
	cudaStream_t stream);

void finalize_backward_dx_cluster_hierarchical_sm90(
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const DxHierarchicalMapping& mapping,
	int grid,
	int num_waves,
	cudaStream_t stream);

void finalize_backward_dx_hierarchical_sm90(
	int tiles_per_reduce,
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const DxHierarchicalMapping& mapping,
	int grid,
	int num_waves,
	cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
