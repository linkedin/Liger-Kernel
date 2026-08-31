#pragma once

// FSLCE-specific adapters around the reusable tensor-parallel reduction API.

#include <cuda_runtime.h>

#include "backward_gemm_sm90.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

void launch_backward_dx_dw_direct_peer_wave_sm90(
	bool return_entropy,
	int tiles_per_reduce,
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const liger_cute::detail::DirectPeerReduceView& mapping,
	int grid,
	cudaStream_t stream,
	int wave);

template <bool RequiresRemote>
void finalize_backward_dx_cluster_sm90(
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const liger_cute::detail::NvlsReduceView& mapping,
	const liger_cute::detail::RemoteReduceView& remote,
	int grid,
	int num_waves,
	cudaStream_t stream);

extern template void finalize_backward_dx_cluster_sm90<false>(
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const liger_cute::detail::NvlsReduceView& mapping,
	const liger_cute::detail::RemoteReduceView& remote,
	int grid,
	int num_waves,
	cudaStream_t stream);
extern template void finalize_backward_dx_cluster_sm90<true>(
	const BackwardGemmParamsSm90<90>& params,
	const DxReduceWorkspace<float>& comm,
	const liger_cute::detail::NvlsReduceView& mapping,
	const liger_cute::detail::RemoteReduceView& remote,
	int grid,
	int num_waves,
	cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
