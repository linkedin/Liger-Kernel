// ═══════════════════════════════════════════════════════════════════════════
// SM90 tensor-parallel fused scaled linear cross entropy — forward launcher.
//
// Sole production host call graph:
//
//   fused_linear_scaled_cross_entropy_forward
//     -> forward_gemm_kernel_sm90
//     -> forward_split_reduce_kernel_sm90
//     -> NCCL MAX
//     -> pack_forward_reduction_kernel
//     -> NCCL SUM
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
#include "online_softmax.cuh"

#include <nccl.h>

#include <cstddef>
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

template <bool ReturnEntropy>
__global__ void pack_forward_reduction_kernel(
		const float* local_max,
		const float* global_max,
		const float* local_sum,
		const float* local_target,
		const float* local_weighted_sum,
		float* packed,
		int tokens) {
	int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	if (row >= tokens) return;

	float correction = local_sum[row] == 0.0f
		? 0.0f
		: expf(local_max[row] - global_max[row]);
	packed[kForwardReducedSumField * tokens + row] =
		local_sum[row] * correction;
	packed[kForwardReducedTargetField * tokens + row] =
		local_target[row];
	if constexpr (ReturnEntropy) {
		packed[kForwardReducedWeightedField * tokens + row] =
			local_weighted_sum[row] * correction;
	}
}

template <bool ReturnEntropy>
__global__ void finalize_forward_kernel(
		const float* global_max,
		const float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index) {
	int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	if (row >= tokens) return;

	float global_weighted = 0.0f;
	if constexpr (ReturnEntropy) {
		global_weighted =
			reduced[kForwardReducedWeightedField * tokens + row];
	}
	FinalizedSoftmax result = finalize_softmax<ReturnEntropy>(
		global_max[row],
		reduced[kForwardReducedSumField * tokens + row],
		reduced[kForwardReducedTargetField * tokens + row],
		global_weighted,
		target[row] == ignore_index);
	nll[row] = result.nll;
	lse[row] = result.lse;
	if constexpr (ReturnEntropy) {
		entropy[row] = result.entropy;
	}
}

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy forward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

void check_nccl(ncclResult_t result, const char* what) {
	LIGER_CHECK(
		result == ncclSuccess,
		"fused_scaled_linear_cross_entropy forward: ",
		what,
		" failed: ",
		ncclGetErrorString(result));
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
		params.nccl_comm_handle != 0,
		"forward TP requires a valid ncclComm_t handle");
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
			gemm_params.output.local_lse != nullptr &&
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

	auto* kernel_fn = &forward_gemm_kernel_sm90<
		ReturnEntropy, Compute, decltype(tma_load_x), decltype(tma_load_w)>;

	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	static_assert(kSmemBytes <= kHopperMaxSmemBytes,
		"SM90 forward shared-memory footprint exceeds the Hopper limit");
	static_assert(kSmemBytes <= Launch::smem_bytes(ReturnEntropy),
		"ForwardGemmLaunchSm90::smem_bytes must bound the real footprint");

	using Cluster = sm90::ClusterLaunchSm90<Compute>;
	check_cuda(
		Cluster::prepare(kernel_fn, kSmemBytes),
		"cudaFuncSetAttribute(MaxDynamicSharedMemorySize / "
		"NonPortableClusterSizeAllowed)");

	int max_active_cluster_pairs = Cluster::max_active_clusters(
		kernel_fn, Config::kNumThreads, kSmemBytes, Config::kClusterM);
	ForwardGemmSplitSm90<Compute> split = Launch::resolve_split(
		gemm_params.tuning,
		gemm_params.tokens,
		gemm_params.local_vocab,
		max_active_cluster_pairs);
	LIGER_CHECK(
		split.base_split_n >= 1 &&
			split.split_n <= split.num_logical_n_tiles,
		"split_n must be between 1 and the number of logical vocabulary tiles");
	LIGER_CHECK(
		split.extra_m_pairs < split.num_m_pairs &&
			split.split_n ==
				split.base_split_n + (split.extra_m_pairs != 0 ? 1 : 0),
		"uneven split tuning must use split_n=base_split_n+1 for a proper "
		"subset of M pairs");

	// Split partials are laid out [statistic][M tile][split][row].
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
		reinterpret_cast<std::uintptr_t>(gemm_params.workspace) % 16 == 0,
		"forward GEMM workspace must be 16 B aligned");

	float* scratch = static_cast<float*>(gemm_params.workspace);
	ForwardGemmPartialsSm90<Compute> partials;
	partials.partial_max = scratch + 0 * partial_rows;
	partials.partial_sum = scratch + 1 * partial_rows;
	partials.partial_target = scratch + 2 * partial_rows;
	partials.partial_weighted =
		ReturnEntropy ? scratch + 3 * partial_rows : nullptr;

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
			split),
		"cudaLaunchKernelEx(forward_gemm_kernel_sm90)");

	forward_split_reduce_kernel_sm90<ReturnEntropy, Compute>
		<<<split.num_m_tiles, Config::kTileM, 0, stream>>>(
			gemm_params, partials, split);
	check_cuda(
		cudaGetLastError(), "forward_split_reduce_kernel_sm90 launch");

	ncclComm_t comm =
		reinterpret_cast<ncclComm_t>(params.nccl_comm_handle);
	check_nccl(
		ncclAllReduce(
			reduce_workspace.local.local_max,
			reduce_workspace.global_max,
			gemm_params.tokens,
			ncclFloat32,
			ncclMax,
			comm,
			stream),
		"ncclAllReduce(MAX)");

	constexpr int kPostThreads = 256;
	int post_blocks = ceil_div(gemm_params.tokens, kPostThreads);
	pack_forward_reduction_kernel<ReturnEntropy>
		<<<post_blocks, kPostThreads, 0, stream>>>(
			reduce_workspace.local.local_max,
			reduce_workspace.global_max,
			reduce_workspace.local.local_sum,
			reduce_workspace.local.local_target,
			reduce_workspace.local.local_weighted_sum,
			reduce_workspace.packed,
			gemm_params.tokens);
	check_cuda(
		cudaGetLastError(), "pack_forward_reduction_kernel launch");

	check_nccl(
		ncclAllReduce(
			reduce_workspace.packed,
			reduce_workspace.reduced,
			static_cast<std::size_t>(reduce_workspace.fields) *
				static_cast<std::size_t>(gemm_params.tokens),
			ncclFloat32,
			ncclSum,
			comm,
			stream),
		"ncclAllReduce(SUM)");

	finalize_forward_kernel<ReturnEntropy>
		<<<post_blocks, kPostThreads, 0, stream>>>(
			reduce_workspace.global_max,
			reduce_workspace.reduced,
			gemm_params.target,
			params.nll,
			params.lse,
			params.entropy,
			gemm_params.tokens,
			gemm_params.ignore_index);
	check_cuda(
		cudaGetLastError(), "finalize_forward_kernel launch");
}

template void fused_linear_scaled_cross_entropy_forward<false, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);
template void fused_linear_scaled_cross_entropy_forward<true, 90>(
	const ForwardTpParamsSm90<90>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
