#include "forward_reduce.cuh"

#if LIGER_CUTE_DISPATCH_COMPUTE == 100
#include "forward_gemm_sm100.cuh"
#else
#include "forward_gemm_sm90.cuh"
#endif

#include <cstddef>
#include <cstdint>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "buffer_pool.cuh"
#include "forward_remote_reduce.cuh"
#include "liger_cute/check.h"

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

int g_capacity_tokens = 0;
int g_capacity_local_vocab = 0;
std::size_t g_capacity_split_partials_bytes = 0;

#if LIGER_CUTE_DISPATCH_COMPUTE == 100
using ForwardWorkspaceLaunch = ForwardGemmLaunchSm100<100>;
#else
using ForwardWorkspaceLaunch = ForwardGemmLaunchSm90<90>;
#endif

std::size_t token_bytes_at(int tokens) {
	return static_cast<std::size_t>(tokens) * sizeof(float);
}

std::size_t packed_bytes_at(int tokens) {
	return static_cast<std::size_t>(kForwardReducedFields) *
		token_bytes_at(tokens);
}

std::size_t split_partials_bytes_at(int tokens, int local_vocab) {
	return ForwardWorkspaceLaunch::workspace_bytes(
		tokens, local_vocab, /*return_entropy=*/true);
}

std::size_t split_ready_bytes_at(int tokens) {
	return static_cast<std::size_t>(
		ForwardWorkspaceLaunch::num_m_tiles(tokens)) * sizeof(int);
}

#if LIGER_CUTE_DISPATCH_COMPUTE == 100
std::size_t wave_partial_ready_bytes_at(
		int tokens, int local_vocab) {
	return ForwardWorkspaceLaunch::wave_partial_ready_entries(
		tokens, local_vocab) * sizeof(std::uint64_t);
}

std::size_t wave_tile_ready_bytes_at(int tokens) {
	return ForwardWorkspaceLaunch::wave_tile_ready_entries(tokens) *
		sizeof(std::uint64_t);
}

constexpr std::size_t wave_slot_released_bytes() {
	return static_cast<std::size_t>(
		kForwardWaveSourceSlotsSm100) *
		sizeof(std::uint64_t);
}

constexpr std::size_t diagnostic_bytes() {
	return static_cast<std::size_t>(
		kForwardDiagnosticEntriesSm100) *
		sizeof(std::uint64_t);
}
#else
std::size_t wave_partial_ready_bytes_at(int, int) {
	return 0;
}

std::size_t wave_tile_ready_bytes_at(int) {
	return 0;
}

constexpr std::size_t wave_slot_released_bytes() {
	return 0;
}

constexpr std::size_t diagnostic_bytes() {
	return 0;
}
#endif

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

template <bool ReturnEntropy>
__device__ __forceinline__ void store_finalized_row(
		const float* global_max,
		const float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index,
		int row) {
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

template <bool ReturnEntropy>
__global__ void local_finalize_forward_kernel(
		const float* global_max,
		const float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index) {
	for (int row = static_cast<int>(
				blockIdx.x * blockDim.x + threadIdx.x);
			row < tokens;
			row += static_cast<int>(blockDim.x * gridDim.x)) {
		store_finalized_row<ReturnEntropy>(
			global_max,
			reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			row);
	}
}

template <bool ReturnEntropy, int NumWorkerWarpsPerBlock>
__global__ void remote_finalize_forward_kernel(
		__grid_constant__ const liger_cute::detail::RemoteReduceView remote,
		__grid_constant__ const liger_cute::detail::NvlsReduceView local,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		const std::uint64_t* launch_epoch,
		float* remote_source,
		float* global_max,
		float* reduced,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index) {
	constexpr int kFields =
		forward_reduced_fields<ReturnEntropy>();
	constexpr int kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	int rows_per_rank =
		(tokens + local.size - 1) / local.size;
	int padded_tokens = rows_per_rank * local.size;
	std::size_t rows_owned =
		static_cast<std::size_t>(rows_per_rank);

	int lane = liger_cute::detail::remote_ring_lane();
	reduce_forward_state_ring<
		ReturnEntropy,
		NumWorkerWarpsPerBlock>(
		remote, launch_epoch, remote_source, rows_owned);

	int warp = static_cast<int>(threadIdx.x) / 32;
	allgather_forward_reduced_state_warp<ReturnEntropy>(
		local,
		comm,
		launch_epoch,
		remote_source,
		padded_tokens,
		warp,
		/*communication_warp=*/0);
	if constexpr (
			liger_cute::detail::remote_ring_uses_grid_sync<
				NumWorkerWarpsPerBlock>()) {
		__syncthreads();
	}

	constexpr int kWorkers =
		liger_cute::detail::remote_ring_worker_threads_per_block<
			NumWorkerWarpsPerBlock>();
	int worker = NumWorkerWarpsPerBlock == 1
		? lane
		: static_cast<int>(threadIdx.x);
	for (int row = worker;
			row < tokens;
			row += kWorkers) {
		std::size_t state =
			static_cast<std::size_t>(row) *
				static_cast<std::size_t>(kStateFields);
		global_max[row] = comm.reduced[state];
		for (int field = 0; field < kFields; ++field) {
			reduced[field * tokens + row] =
				comm.reduced[state + 1 + field];
		}
		store_finalized_row<ReturnEntropy>(
			global_max,
			reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			row);
	}
}

}  // namespace

void configure_forward_tp_workspace(int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");

	std::size_t split_bytes =
		split_partials_bytes_at(max_tokens, max_local_vocab);
	if (max_tokens <= g_capacity_tokens &&
		max_local_vocab <= g_capacity_local_vocab &&
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
	pool.get_device(Names::kReduced, packed_bytes);
	pool.get_device(Names::kGemmSplitPartials, split_bytes);
	pool.get_device(Names::kSplitReady, split_ready_bytes_at(max_tokens));
	if constexpr (LIGER_CUTE_DISPATCH_COMPUTE == 100) {
		auto* partial_ready = pool.get_device(
			Names::kWavePartialReady,
			wave_partial_ready_bytes_at(
				max_tokens, max_local_vocab));
		auto* tile_ready = pool.get_device(
			Names::kWaveTileReady,
			wave_tile_ready_bytes_at(max_tokens));
		auto* slot_released = pool.get_device(
			Names::kWaveSlotReleased,
			wave_slot_released_bytes());
		auto* diagnostics = pool.get_device(
			Names::kDiagnostics,
			diagnostic_bytes());
		LIGER_CHECK(
			cudaMemset(
				partial_ready,
				0,
				wave_partial_ready_bytes_at(
					max_tokens, max_local_vocab)) ==
				cudaSuccess,
			"cudaMemset(forward wave partial epochs) failed");
		LIGER_CHECK(
			cudaMemset(
				tile_ready,
				0,
				wave_tile_ready_bytes_at(max_tokens)) ==
				cudaSuccess,
			"cudaMemset(forward wave tile epochs) failed");
		LIGER_CHECK(
			cudaMemset(
				slot_released,
				0,
				wave_slot_released_bytes()) ==
				cudaSuccess,
			"cudaMemset(forward wave release epochs) failed");
		LIGER_CHECK(
			cudaMemset(
				diagnostics,
				0,
				diagnostic_bytes()) ==
				cudaSuccess,
			"cudaMemset(forward diagnostics) failed");
	}
	pool.get_device(Names::kLocalSum, token_bytes);
	pool.get_device(Names::kLocalTarget, token_bytes);
	pool.get_device(Names::kLocalWeighted, token_bytes);

	g_capacity_tokens = max_tokens;
	g_capacity_local_vocab = max_local_vocab;
	g_capacity_split_partials_bytes = split_bytes;
}

std::size_t forward_tp_workspace_device_bytes(
		int max_tokens, int max_local_vocab) {
	LIGER_CHECK(max_tokens > 0, "max_tokens must be positive");
	LIGER_CHECK(max_local_vocab > 0, "max_local_vocab must be positive");
	return 5 * token_bytes_at(max_tokens) +
		packed_bytes_at(max_tokens) +
		split_partials_bytes_at(max_tokens, max_local_vocab) +
		split_ready_bytes_at(max_tokens) +
		wave_partial_ready_bytes_at(max_tokens, max_local_vocab) +
		wave_tile_ready_bytes_at(max_tokens) +
		wave_slot_released_bytes() +
		diagnostic_bytes();
}

void reset_forward_tp_workspace_configuration() {
	g_capacity_tokens = 0;
	g_capacity_local_vocab = 0;
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
	workspace.reduced = static_cast<float*>(
		pool.get_device(Names::kReduced, packed_bytes));
	workspace.split_ready = static_cast<int*>(
		pool.get_device(Names::kSplitReady, split_ready_bytes_at(g_capacity_tokens)));
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	workspace.wave_partial_ready = static_cast<std::uint64_t*>(
		pool.get_device(
			Names::kWavePartialReady,
			wave_partial_ready_bytes_at(
				g_capacity_tokens,
				g_capacity_local_vocab)));
	workspace.wave_tile_ready = static_cast<std::uint64_t*>(
		pool.get_device(
			Names::kWaveTileReady,
			wave_tile_ready_bytes_at(g_capacity_tokens)));
	workspace.wave_slot_released = static_cast<std::uint64_t*>(
		pool.get_device(
			Names::kWaveSlotReleased,
			wave_slot_released_bytes()));
	workspace.diagnostics = static_cast<std::uint64_t*>(
		pool.get_device(
			Names::kDiagnostics,
			diagnostic_bytes()));
#endif
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
	workspace.fields = forward_reduced_fields(ReturnEntropy);
	workspace.vocab_capacity = g_capacity_local_vocab;
	return workspace;
}

template ForwardTpWorkspace reserve_forward_tp_workspace<false>(int);
template ForwardTpWorkspace reserve_forward_tp_workspace<true>(int);

int forward_tp_diagnostic_entries() {
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	return kForwardDiagnosticEntriesSm100;
#else
	return 0;
#endif
}

void copy_forward_tp_diagnostics(
		std::uint64_t* output,
		int entries,
		cudaStream_t stream) {
	ensure_capacity(1);
	LIGER_CHECK(output != nullptr, "forward diagnostic output is null");
	int required = forward_tp_diagnostic_entries();
	LIGER_CHECK(
		entries >= required,
		"forward diagnostic output has ",
		entries,
		" entries but ",
		required,
		" are required");
#if LIGER_CUTE_DISPATCH_COMPUTE == 100
	auto& pool = global_buffer_pool();
	auto* diagnostics = static_cast<std::uint64_t*>(
		pool.get_device(
			ForwardBufferNames::kDiagnostics,
			diagnostic_bytes()));
	cudaError_t error = cudaMemcpyAsync(
		output,
		diagnostics,
		diagnostic_bytes(),
		cudaMemcpyDeviceToDevice,
		stream);
	LIGER_CHECK(
		error == cudaSuccess,
		"cudaMemcpyAsync(forward diagnostics) failed: ",
		cudaGetErrorString(error));
#else
	(void)stream;
#endif
}

template <bool ReturnEntropy>
void launch_forward_remote_finalize_typed(
		const liger_cute::detail::RemoteReduceView& remote,
		const liger_cute::detail::NvlsReduceView& local,
		const DxReduceWorkspace<float>& comm,
		const std::uint64_t* launch_epoch,
		const ForwardTpWorkspace& workspace,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index,
		cudaStream_t stream) {
	if (!remote.enabled()) {
		constexpr int kThreads = 256;
		int blocks = (tokens + kThreads - 1) / kThreads;
		local_finalize_forward_kernel<ReturnEntropy>
			<<<blocks, kThreads, 0, stream>>>(
			workspace.global_max,
			workspace.reduced,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index);
	} else {
		constexpr int kStateFields =
			forward_reduced_state_fields<ReturnEntropy>();
		std::size_t rows_owned = static_cast<std::size_t>(
			(tokens + local.size - 1) / local.size);
		std::size_t payload_elements =
			rows_owned * static_cast<std::size_t>(kStateFields);
		LIGER_CHECK(
			payload_elements <= remote.inbox_slot_elements &&
				payload_elements <=
					remote.reduced_shard_elements,
			"forward remote reduction payload exceeds its symmetric buffers");
		liger_cute::detail::RemoteReduceView remote_arg = remote;
		liger_cute::detail::NvlsReduceView local_arg = local;
		DxReduceWorkspace<float> comm_arg = comm;
		const std::uint64_t* launch_epoch_arg = launch_epoch;
		float* remote_source = remote.reduced_shard;
		float* global_max = workspace.global_max;
		float* reduced = workspace.reduced;
		const std::int64_t* target_arg = target;
		float* nll_arg = nll;
		float* lse_arg = lse;
		float* entropy_arg = entropy;
		int tokens_arg = tokens;
		std::int64_t ignore_index_arg = ignore_index;
		void* args[] = {
			&remote_arg,
			&local_arg,
			&comm_arg,
			&launch_epoch_arg,
			&remote_source,
			&global_max,
			&reduced,
			&target_arg,
			&nll_arg,
			&lse_arg,
			&entropy_arg,
			&tokens_arg,
			&ignore_index_arg};
		auto* kernel =
			&remote_finalize_forward_kernel<
				ReturnEntropy,
				kForwardStandaloneRemoteWorkerWarpsPerBlock>;
		// Every member of the active TP subgroup enters this blocking NVSHMEM
		// kernel with identical launch geometry and ordering.
		int status = nvshmemx_collective_launch(
			reinterpret_cast<const void*>(kernel),
			dim3(1, 1, 1),
			dim3(
				liger_cute::detail::
					remote_ring_worker_threads_per_block<
						kForwardStandaloneRemoteWorkerWarpsPerBlock>(),
				1,
				1),
			args,
			0,
			stream);
		LIGER_CHECK(
			status == 0,
			"forward remote finalize collective launch failed with status ",
			status);
	}
	cudaError_t error = cudaGetLastError();
	LIGER_CHECK(
		error == cudaSuccess,
		"forward finalize launch failed: ",
		cudaGetErrorString(error));
}

void launch_forward_remote_finalize(
		bool return_entropy,
		const liger_cute::detail::RemoteReduceView& remote,
		const liger_cute::detail::NvlsReduceView& local,
		const DxReduceWorkspace<float>& comm,
		const std::uint64_t* launch_epoch,
		const ForwardTpWorkspace& workspace,
		const std::int64_t* target,
		float* nll,
		float* lse,
		float* entropy,
		int tokens,
		std::int64_t ignore_index,
		cudaStream_t stream) {
	if (return_entropy) {
		launch_forward_remote_finalize_typed<true>(
			remote,
			local,
			comm,
			launch_epoch,
			workspace,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			stream);
	} else {
		launch_forward_remote_finalize_typed<false>(
			remote,
			local,
			comm,
			launch_epoch,
			workspace,
			target,
			nll,
			lse,
			entropy,
			tokens,
			ignore_index,
			stream);
	}
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
