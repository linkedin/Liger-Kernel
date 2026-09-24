#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// dX communication primitives for the fused SM100 backward.
//
// Three warp-scoped stages, all driven from inside the persistent kernel:
//
//   1. warp 0, interleaved with the dX UMMA — node-local NVLS reduce-scatter
//      of one completed M128xN256 FP32 staging tile into this rank's packed
//      shard of the durable dX. Runs per tile; it never waits for the whole
//      wave.
//   2. warp 1, overlapped with the dW UMMA — inter-host matching-rank IB ring
//      reduce-scatter over the wave's packed shard, using remote_ring.cuh's
//      QP put-signal/wait transport verbatim (NumWorkerWarpsPerBlock = 1, so
//      the ring never touches a block or grid barrier).
//   3. warp 0, after the ring — node-local NVLS all-gather of the packed shard
//      back into the full FP32 tile, converted once to X.dtype and scattered
//      into the caller's row-major grad_input.
//
// Stages 1 and 3 mirror `backward_dx_cluster2_gemm_wave_kernel_sm90`'s
// communication warps and `cluster_local_allgather_scatter_kernel`; stage 2
// mirrors the SM90 `launch_remote_reduce` follow-up, moved into the fused
// kernel so it overlaps the dW GEMM instead of following it.
//
// Every signal carries the launch epoch, so a replayed CUDA graph can never
// observe a stale generation.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "backward_gemm_sm100.cuh"
#include "dx_reduce.cuh"
#include "liger_cute/detail/nvls.cuh"
#include "liger_cute/detail/tp_reduce.cuh"
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)
#include "liger_cute/detail/remote_ring.cuh"
#endif

namespace liger {
namespace fused_scaled_linear_cross_entropy {

#if defined(__CUDACC__)

struct alignas(8) BackwardBfloat16x4Sm100 {
	__nv_bfloat162 low;
	__nv_bfloat162 high;
};

__device__ __forceinline__ std::uint64_t backward_load_acquire_system_sm100(
		const std::uint64_t* address) {
	std::uint64_t value = 0;
	asm volatile(
		"ld.acquire.sys.global.u64 %0, [%1];"
		: "=l"(value)
		: "l"(address)
		: "memory");
	return value;
}

__device__ __forceinline__ void backward_store_release_system_sm100(
		std::uint64_t* address, std::uint64_t value) {
	asm volatile(
		"st.release.sys.global.u64 [%0], %1;"
		:
		: "l"(address), "l"(value)
		: "memory");
}

__device__ __forceinline__ std::uint64_t backward_load_acquire_device_sm100(
		const std::uint64_t* address) {
	std::uint64_t value = 0;
	asm volatile(
		"ld.acquire.gpu.global.u64 %0, [%1];"
		: "=l"(value)
		: "l"(address)
		: "memory");
	return value;
}

// Warp-uniform spin until `*address` reaches `value`. Used only by warps 0
// and 1, which never take part in a CTA-wide barrier.
__device__ __forceinline__ void backward_wait_ge_warp_sm100(
		const std::uint64_t* address, std::uint64_t value) {
	unsigned int active = __activemask();
	for (;;) {
		bool ready = backward_load_acquire_device_sm100(address) >= value;
		if (__all_sync(active, ready)) break;
		__nanosleep(64);
	}
	__syncwarp(active);
}

__device__ __forceinline__ void backward_wait_epoch_warp_sm100(
		const std::uint64_t* address, std::uint64_t epoch) {
	unsigned int active = __activemask();
	for (;;) {
		bool ready =
			backward_load_acquire_system_sm100(address) == epoch;
		if (__all_sync(active, ready)) break;
		__nanosleep(64);
	}
	__syncwarp(active);
}

// ───────────────────────────────────────────────────────────────────────────
// Durable layout
//
// The packed shard holds `tile_elements / team_size` contiguous FP32 values
// per dX tile, tiles ordered `wave * tiles_per_wave + m_tile * n_tiles +
// n_tile`. Identical on every PE, so the ring payload for one wave is one
// contiguous message.
// ───────────────────────────────────────────────────────────────────────────

__host__ __device__ inline std::size_t backward_dx_durable_tile_sm100(
		int wave, int tiles_per_wave, int m_tile, int n_tile,
		int num_n_tiles) {
	return static_cast<std::size_t>(wave) *
			static_cast<std::size_t>(tiles_per_wave) +
		static_cast<std::size_t>(m_tile) *
			static_cast<std::size_t>(num_n_tiles) +
		static_cast<std::size_t>(n_tile);
}

// ───────────────────────────────────────────────────────────────────────────
// Stage 1 — node-local NVLS reduce-scatter of one staged tile
// ───────────────────────────────────────────────────────────────────────────

template <typename CommConfig, int Compute = 100>
__device__ __forceinline__ void backward_dx_reduce_scatter_warp_sm100(
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		float* packed_destination,
		int cta,
		int stage,
		int wave,
		int pass) {
	constexpr std::size_t kTileElements = CommConfig::kTileElements;
	std::size_t base = dx_slot_offset<CommConfig>(cta, 0, stage);
	unsigned int lane = liger_cute::detail::nvls_lane_id();

	if (mapping.size == 1) {
		constexpr std::size_t kVectors = kTileElements / 4;
		const auto* source = reinterpret_cast<const float4*>(
			comm.partial + base);
		auto* destination = reinterpret_cast<float4*>(packed_destination);
		for (std::size_t vector = lane; vector < kVectors;
				vector += kWarpSize) {
			destination[vector] = source[vector];
		}
		__syncwarp();
		return;
	}

	std::size_t ready_offset = dx_sync_offset<CommConfig>(
		cta, 0, stage, kDxReadyPhase, mapping.size);
	std::size_t complete_offset = dx_sync_offset<CommConfig>(
		cta, 0, stage, kDxCompletePhase, mapping.size);
	std::uint64_t epoch = backward_wave_epoch_sm100(
			dx_epoch_base(comm),
			kBackwardDxScatterEpochSuffixSm100,
			wave) |
		static_cast<std::uint64_t>(pass + 1);

	liger_cute::detail::nvls_barrier_warp(
		comm.sync + ready_offset,
		mapping.multicast_sync + ready_offset,
		mapping.rank,
		mapping.size,
		epoch);
	liger_cute::detail::nvls_sum_reduce_scatter_warp(
		packed_destination,
		mapping.multicast_partial + base,
		kTileElements,
		mapping.rank,
		mapping.size);
	liger_cute::detail::nvls_barrier_warp(
		comm.sync + complete_offset,
		mapping.multicast_sync + complete_offset,
		mapping.rank,
		mapping.size,
		epoch);
}

// ───────────────────────────────────────────────────────────────────────────
// Stage 3 — node-local NVLS all-gather plus the BF16 scatter into grad_input
// ───────────────────────────────────────────────────────────────────────────

template <typename CommConfig, int Compute = 100>
__device__ __forceinline__ void backward_dx_allgather_scatter_warp_sm100(
		const BackwardGemmParamsSm100<Compute>& params,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& mapping,
		const float* packed_source,
		std::size_t durable_tile,
		int cta,
		int stage,
		int wave,
		int pass,
		int m_tile,
		int n_tile) {
	using Config = BackwardGemmConfigSm100<Compute>;
	constexpr std::size_t kTileElements = CommConfig::kTileElements;
	constexpr int kTileM = CommConfig::kTileM;
	constexpr int kTileN = CommConfig::kTileN;
	unsigned int lane = liger_cute::detail::nvls_lane_id();

	const float* values = packed_source;
	std::size_t full_offset =
		durable_tile * static_cast<std::size_t>(kTileElements);
	if (mapping.size > 1) {
		std::size_t ready_offset = dx_sync_offset<CommConfig>(
			cta, 0, stage, kDxReadyPhase, mapping.size);
		std::size_t complete_offset = dx_sync_offset<CommConfig>(
			cta, 0, stage, kDxCompletePhase, mapping.size);
		std::uint64_t epoch = backward_wave_epoch_sm100(
				dx_epoch_base(comm),
				kBackwardDxAllgatherEpochSuffixSm100,
				wave) |
			static_cast<std::uint64_t>(pass + 1);
		liger_cute::detail::nvls_barrier_warp(
			comm.sync + ready_offset,
			mapping.multicast_sync + ready_offset,
			mapping.rank,
			mapping.size,
			epoch);
		liger_cute::detail::nvls_allgather_warp(
			mapping.multicast_reduced + full_offset,
			packed_source,
			kTileElements,
			mapping.rank,
			mapping.size);
		liger_cute::detail::nvls_barrier_warp(
			comm.sync + complete_offset,
			mapping.multicast_sync + complete_offset,
			mapping.rank,
			mapping.size,
			epoch);
		values = comm.reduced + full_offset;
	}

	auto* grad_input = static_cast<__nv_bfloat16*>(params.grad_input);
	constexpr std::size_t kVectors = kTileElements / 4;
	constexpr int kVectorsPerRow = kTileN / 4;
	int row_base = wave * Config::kWaveRows + m_tile * kTileM;
	int column_base = n_tile * kTileN;
	for (std::size_t vector = lane; vector < kVectors;
			vector += kWarpSize) {
		int row = static_cast<int>(vector) / kVectorsPerRow;
		int column = (static_cast<int>(vector) % kVectorsPerRow) * 4;
		int output_row = row_base + row;
		int output_column = column_base + column;
		if (output_row >= params.tokens ||
			output_column >= params.hidden) {
			continue;
		}
		float4 value = reinterpret_cast<const float4*>(values)[vector];
		BackwardBfloat16x4Sm100 packed{
			__floats2bfloat162_rn(value.x, value.y),
			__floats2bfloat162_rn(value.z, value.w)};
		auto* output = reinterpret_cast<BackwardBfloat16x4Sm100*>(
			grad_input +
			static_cast<std::size_t>(output_row) * params.hidden +
			output_column);
		*output = packed;
	}
	__syncwarp();
}

// ───────────────────────────────────────────────────────────────────────────
// Stage 2 — inter-host matching-rank IB ring over one wave's packed shard
// ───────────────────────────────────────────────────────────────────────────

#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)

// FP32 SUM merge for remote_ring_reduce's per-step callback.
struct BackwardDxRemoteSumMerge {
	float* accumulator;

	__device__ __forceinline__ void operator()(
			const float* contribution,
			std::size_t count,
			int worker,
			int workers) const {
		std::size_t vectors = count / 4;
		for (std::size_t vector = static_cast<std::size_t>(worker);
				vector < vectors;
				vector += static_cast<std::size_t>(workers)) {
			float4 lhs = reinterpret_cast<float4*>(accumulator)[vector];
			float4 rhs =
				reinterpret_cast<const float4*>(contribution)[vector];
			lhs.x += rhs.x;
			lhs.y += rhs.y;
			lhs.z += rhs.z;
			lhs.w += rhs.w;
			reinterpret_cast<float4*>(accumulator)[vector] = lhs;
		}
		for (std::size_t index =
					vectors * 4 + static_cast<std::size_t>(worker);
				index < count;
				index += static_cast<std::size_t>(workers)) {
			accumulator[index] += contribution[index];
		}
	}
};

__device__ __forceinline__ void backward_dx_ring_transport_warp_sm100(
		const liger_cute::detail::RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		const float* packed_wave,
		std::size_t count,
		int wave) {
	if (remote.size != 2) __trap();
	std::uint64_t operation_suffix = backward_wave_epoch_sm100(
		0ull, kBackwardDxRemoteEpochSuffixSm100, wave);
	std::uint64_t previous_operation_suffix = wave > 0
		? backward_wave_epoch_sm100(
			0ull, kBackwardDxRemoteEpochSuffixSm100, wave - 1)
		: 0ull;
	liger_cute::detail::remote_ring_transport_step_warp(
		remote,
		launch_epoch,
		operation_suffix,
		previous_operation_suffix,
		wave,
		0,
		packed_wave,
		count,
		liger_cute::detail::remote_ring_lane());
}

__device__ __forceinline__ void backward_dx_ring_finish_warp_sm100(
		const liger_cute::detail::RemoteReduceView& remote,
		const std::uint64_t* launch_epoch,
		int wave,
		bool last_wave) {
	std::uint64_t operation_suffix = backward_wave_epoch_sm100(
		0ull, kBackwardDxRemoteEpochSuffixSm100, wave);
	liger_cute::detail::remote_ring_finish_warp(
		remote,
		launch_epoch,
		operation_suffix,
		liger_cute::detail::remote_ring_lane(),
		last_wave);
}

__device__ __forceinline__ void backward_dx_remote_merge_workers_sm100(
		float* packed_wave,
		const float* contribution,
		std::size_t count,
		int worker,
		int workers) {
	BackwardDxRemoteSumMerge{packed_wave}(
		contribution, count, worker, workers);
}

#endif  // LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM

#endif  // __CUDACC__

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
