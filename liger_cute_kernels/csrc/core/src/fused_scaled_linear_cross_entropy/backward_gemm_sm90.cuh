#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy — backward GEMM contract.
//
// Structural port of
//   src/liger_kernel/ops/cutedsl/ops/_fused_scaled_cross_entropy_backward_fused_sm90.py
// (the CuTe-DSL `_FusedBackwardSM90` one-kernel backward). Constant names below
// map onto the reference module constants:
//
//   THREADS_PER_CTA 384   tile_m 128    tile_n 256    tile_k 64   stages 3
//   m_tiles_per_wave 8    wave_rows 1024
//   dz_store_stages 1 -> 2                dw_store_stages 4 -> 2
//   epi_registers 88 -> 104               mma_registers 208 -> 200
//
// The kernel keeps the reference's three fused phases and its wave loop:
//
//   phase dZ   Z = X @ W_local^T, softmax/entropy epilogue -> dZ wave buffer
//   grid barrier
//   phase dX   dX_partial = dZ @ W_local, epilogue -> CTA-owned symmetric
//              staging, the same CTA's warps 1..2 all-reduce and store dX
//   grid barrier
//   phase dW   dW_local = dZ^T @ X, TMA store (wave 0) / reduce-add (wave > 0)
//   grid barrier (unless this was the last wave)
//
// ── Deliberate deltas from the single-GPU reference ───────────────────────
//  * Warp roles are fixed by the tensor-parallel contract: warp 0 is the TMA
//    producer for all three phases, warps 1..2 reduce that same CTA's dX groups,
//    warp 3 is reserved and idle on every path, warps 4..11 are the two WGMMA
//    consumer warp groups. The reference gave warp group 0 (warps 0..3) the dZ
//    softmax epilogue; here warps 1..3 are unavailable, so the epilogue runs in
//    the consumer warp groups straight out of the FP32 WGMMA accumulators. The
//    reference's FP16 logit staging round-trip is reproduced in register
//    (float -> half -> float) so the numerics are unchanged.
//  * CLUSTER_M is 1, so the dX B operand is not multicast. The resident grid is
//    launched through nvshmemx_collective_launch() for cross-PE co-scheduling;
//    that API takes no cluster launch attribute. The reference's cluster-pair
//    M multicast is therefore not expressible and is dropped.
//  * The epilogue store tile is N64 rather than the reference's N32 for dZ/dW.
//    A BF16 N64 row is exactly 128 B, which is what the SW128 store staging
//    used by dZ and dW requires. dW already used N64 in the reference.
//  * dX stays FP32 from WGMMA through a compact symmetric staging ring and the
//    NVLS SUM. The communication warps convert once to X.dtype while
//    scattering the final reduced result into the caller's row-major tensor
//    (BF16 for this SM90 specialization). dW is untouched and stays rank-local.
//
// This header is deliberately CuTe-free: launch geometry, the wave/tile
// schedule, the warp plan and the launcher declarations only, so torch-free
// consumers can include the umbrella header without CUTLASS. The executable
// producer/consumer/mainloop lives in backward_gemm_mainloop_sm90.cuh and is
// instantiated by fused_linear_scaled_cross_entropy_backward.cu.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backward.cuh"
#include "config.cuh"
#include "dx_reduce.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

inline constexpr float kBackwardLog2E = 1.4426950408889634f;
inline constexpr int kBackwardMaxSmemBytes = 227 * 1024;

// exp2 with the reference's fastmath=True lowering (llvm.nvvm.ex2.approx.ftz).
__host__ __device__ inline float backward_exp2_sm90(float value) {
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return result;
#else
	return exp2f(value);
#endif
}

template <int Compute = 90>
struct BackwardGemmConfigSm90 {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	static constexpr int kCompute = Compute;

	static constexpr int kTileM = 128;
	static constexpr int kTileK = 64;
	static constexpr int kDzTileN = 256;
	static constexpr bool kDzNoTmaStore = false;
	static constexpr int kDzLogicalTileN = kDzTileN;
	static constexpr int kDxTileN = 256;
	static constexpr int kDwTileN = 256;
	static constexpr int kDwTileM = kTileM;
	static constexpr int kDwLogicalTileN = kDwTileN;
	static constexpr int kMainloopStages = 3;
	static constexpr int kMTilesPerWave = 8;
	static constexpr int kWaveRows = kTileM * kMTilesPerWave;

	static constexpr int kDzStoreTileN = 64;
	static constexpr int kDzStoreStages = 2;
	static constexpr int kDwStoreTileN = 64;
	static constexpr int kDwStoreStages = 2;

	static constexpr int kClusterM = 1;
	static constexpr int kNumThreads =
		fused_scaled_linear_cross_entropy::kNumThreads;
	static constexpr int kNumMmaWarpGroups = 2;
	static constexpr int kWarpsPerWarpGroup = 4;
	static constexpr int kWarpGroupSize = 128;
	static constexpr int kConsumerThreads =
		kNumMmaWarpGroups * kWarpGroupSize;
	static constexpr int kFirstConsumerThread = kWarpGroupSize;

	// The dZ workspace K extent must be a whole number of K tiles so the dX /
	// dW mainloops never see a ragged K tail on an operand they own.
	static constexpr int kVocabAlign = kTileK;

	// setmaxnreg budget. Warps 0..3 share one warp group, so the communication
	// warps take the producer allocation. The selected host-waved combined
	// dX+dW specialization uses 56/224; separate phase-specialized kernels
	// retain their budgets below.
	static constexpr int kProducerRegisters = 56;
	static constexpr int kMmaRegisters = 224;
	static constexpr int kDzProducerRegisters = 24;
	static constexpr int kDzMmaRegisters = 240;
	static constexpr int kDxDwProducerRegisters = 72;
	static constexpr int kDxDwMmaRegisters = 216;
	static constexpr int kUsableRegisterBudget = 64512;

	// Hardware named barriers. 0 is __syncthreads().
	static constexpr int kDzEpilogueBarrierId = 4;   // consumer warp groups
	static constexpr int kDxEpilogueBarrierId = 5;   // consumer warp groups
	static constexpr int kDwStoreBarrierId = 6;      // consumer warp groups
	// Hands the aliased A/B and store-staging arenas from the dX phase to the
	// dW phase. Deliberately NOT __syncthreads(): warps 1..2 are still
	// draining the dX reduction ring and must not be pulled into it, and warp
	// 3 is reserved. Only the producer warp and the two consumer warp groups
	// touch the aliased storage, so only they take part.
	static constexpr int kComputeBarrierId = 8;
	static constexpr int kComputeBarrierThreads =
		(1 + kNumMmaWarpGroups * kWarpsPerWarpGroup) * kWarpSize;

	static_assert(kNumThreads == (kNumMmaWarpGroups + 1) * kWarpGroupSize);
	static_assert(kWaveRows % kTileM == 0);
	static_assert(kDzTileN % kDzStoreTileN == 0);
	static_assert(kComputeBarrierThreads == 288,
		"the compute barrier covers warp 0 and warps 4..11");
	static_assert(kComputeBarrierThreads % kWarpSize == 0);
	static_assert(kComputeBarrierThreads < kNumThreads,
		"the compute barrier must exclude the communication and reserved "
		"warps, otherwise it degenerates into __syncthreads()");
	static_assert(kDwTileN % kDwStoreTileN == 0);
	static_assert(backward_warp_role(0) == BackwardWarpRole::kProducer);
	static_assert(backward_warp_role(1) == BackwardWarpRole::kDxCommunication);
	static_assert(backward_warp_role(2) == BackwardWarpRole::kDxCommunication);
	static_assert(backward_warp_role(3) == BackwardWarpRole::kReserved);
	static_assert(backward_warp_role(4) == BackwardWarpRole::kConsumer);
	static_assert(backward_warp_role(11) == BackwardWarpRole::kConsumer);

	__host__ __device__ static constexpr int register_total() {
		return kWarpGroupSize * kProducerRegisters +
			kNumMmaWarpGroups * kWarpGroupSize * kMmaRegisters;
	}
	static_assert(register_total() <= kUsableRegisterBudget,
		"SM90 backward register budget exceeds the usable per-CTA budget");
	static_assert(
		kWarpGroupSize * kDzProducerRegisters +
			kNumMmaWarpGroups * kWarpGroupSize * kDzMmaRegisters <=
			kUsableRegisterBudget,
		"SM90 dZ register budget exceeds the usable per-CTA budget");
	static_assert(
		kWarpGroupSize * kDxDwProducerRegisters +
			kNumMmaWarpGroups * kWarpGroupSize * kDxDwMmaRegisters <=
			kUsableRegisterBudget,
		"SM90 dX+dW register budget exceeds the usable per-CTA budget");
};

// The reference's per-phase geometry, kept as named contracts so a future
// SM100 port can specialise them without touching the mainloop.
template <int Compute = 90>
struct DzGemmContractSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileN = Config::kDzTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStoreTileN = Config::kDzStoreTileN;
	static constexpr int kStoreStages = Config::kDzStoreStages;
};

template <int Compute = 90>
struct DxGemmContractSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileN = Config::kDxTileN;
	static constexpr int kTileK = Config::kTileK;
};

template <int Compute = 90>
struct DwGemmContractSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileN = Config::kDwTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStoreTileN = Config::kDwStoreTileN;
	static constexpr int kStoreStages = Config::kDwStoreStages;
};

template <int Compute = 90>
struct BackwardWarpPlanSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	static constexpr int kProducerWarp = 0;
	static constexpr int kFirstDxCommWarp = 1;
	static constexpr int kLastDxCommWarp = 2;
	static constexpr int kReservedWarp = 3;
	static constexpr int kFirstConsumerWarp = 4;
	static constexpr int kLastConsumerWarp = 11;
	// The consumer warp that issues every TMA store (the reference's `issuer`).
	static constexpr int kIssuerWarp = kFirstConsumerWarp;

	static_assert(Config::kNumThreads == kNumThreads);
	static_assert(kLastConsumerWarp + 1 == kNumWarps);
	static_assert(backward_warp_role(kReservedWarp) ==
		BackwardWarpRole::kReserved);
};

// ───────────────────────────────────────────────────────────────────────────
// Problem description
// ───────────────────────────────────────────────────────────────────────────

// The vocabulary shard is contiguous: local column c is global vocabulary index
// `vocab_start + c`, and targets arrive as *global* int64 indices. `lse` and
// `entropy` are the forward's already-all-reduced per-token outputs, so they
// are ordinary local tensors — nothing symmetric is carried from forward.
template <int Compute = 90>
struct BackwardGemmParamsSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	const void* x = nullptr;               // BF16 [tokens, hidden]
	const void* weight = nullptr;          // BF16 [local_vocab, hidden]
	const std::int64_t* target = nullptr;  // int64 [tokens], global indices
	const float* grad_output = nullptr;    // FP32 [tokens], upstream NLL scale
	const float* lse = nullptr;            // FP32 [tokens], global LSE
	const float* entropy = nullptr;        // FP32 [tokens], global entropy
	const float* entropy_grad = nullptr;   // FP32 [tokens], upstream entropy scale

	void* grad_input = nullptr;   // X.dtype [tokens, hidden], BF16 on SM90
	void* grad_weight = nullptr;  // BF16 [local_vocab, hidden], rank local

	// Internal scratch, launcher owned (see BackwardSymmetricNames).
	void* dz_workspace = nullptr;  // BF16 [kWaveRows, padded_local_vocab]
	std::size_t dz_workspace_bytes = 0;

	int tokens = 0;
	int hidden = 0;
	int local_vocab = 0;
	std::int64_t vocab_start = 0;
	std::int64_t ignore_index = -100;
	float inverse_temperature = 1.0f;
	// Selftest-only skew injection; production callers leave the defaults.
	int dx_comm_delay_warp = -1;
	int dx_comm_delay_iterations = 0;
};

// ───────────────────────────────────────────────────────────────────────────
// Launch geometry
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct BackwardGemmLaunchSm90 {
	using Config = BackwardGemmConfigSm90<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");

	__host__ __device__ static constexpr int num_waves(int tokens) {
		return ceil_div(tokens, Config::kWaveRows);
	}

	// Vocabulary padded to a whole K tile so dX/dW never see a ragged K tail.
	__host__ __device__ static constexpr int padded_vocab(int local_vocab) {
		return ceil_div(local_vocab, Config::kVocabAlign) * Config::kVocabAlign;
	}

	__host__ __device__ static constexpr int num_dz_n_tiles(int local_vocab) {
		return ceil_div(
			padded_vocab(local_vocab), Config::kDzLogicalTileN);
	}

	__host__ __device__ static constexpr int num_dx_n_tiles(int hidden) {
		return ceil_div(hidden, Config::kDxTileN);
	}

	__host__ __device__ static constexpr int num_dx_k_tiles(int local_vocab) {
		return padded_vocab(local_vocab) / Config::kTileK;
	}

	__host__ __device__ static constexpr int num_dz_k_tiles(int hidden) {
		return ceil_div(hidden, Config::kTileK);
	}

	__host__ __device__ static constexpr int num_dw_m_tiles(int local_vocab) {
		return ceil_div(padded_vocab(local_vocab), Config::kDwTileM);
	}

	__host__ __device__ static constexpr int num_dw_n_tiles(int hidden) {
		return ceil_div(hidden, Config::kDwLogicalTileN);
	}

	__host__ __device__ static constexpr int num_dw_k_tiles() {
		return Config::kWaveRows / Config::kTileK;
	}

	// One wave of dZ, BF16.
	__host__ __device__ static std::size_t dz_workspace_bytes(
			int local_vocab) {
		return static_cast<std::size_t>(Config::kWaveRows) *
			static_cast<std::size_t>(padded_vocab(local_vocab)) * 2u;
	}

	// Number of dX reduction groups produced by the whole launch, for a given
	// TilesPerReduce. Waves are always full (`kMTilesPerWave` M tiles) so the
	// group count is a pure function of the shape on every PE.
	__host__ __device__ static int num_dx_groups(
			int tokens, int hidden, int tiles_per_reduce) {
		int groups_per_m = ceil_div(num_dx_n_tiles(hidden), tiles_per_reduce);
		return num_waves(tokens) * Config::kMTilesPerWave * groups_per_m;
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Host launcher
//
// `params.grad_input` is filled with the TP-reduced dX and `params.grad_weight`
// with this rank's dW. Collective: every PE of the configured TP team calls
// this with the
// same shapes and TilesPerReduce.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct BackwardTpParamsSm90 {
	static constexpr int kCompute = Compute;

	BackwardGemmParamsSm90<Compute> gemm = {};

	// Compile-time knob selected at run time; 1, 2 (default) or 4.
	int tiles_per_reduce = 2;
	// Retained for FFI compatibility; CTA ownership ignores this value.
	int num_comm_channels = 4;
};

__host__ __device__ constexpr int backward_dx_split_k(
		int hidden, int local_vocab) {
	// Split-K2 requires an even number of K64 vocabulary tiles. Ragged odd
	// tile counts use split-K1 so the final tile is never dropped.
	int k_tiles = (local_vocab + 63) / 64;
	return hidden == 2048 && k_tiles % 2 == 0 ? 2 : 1;
}

template <bool ReturnEntropy, int Compute = 90>
void fused_linear_scaled_cross_entropy_backward(
	const BackwardTpParamsSm90<Compute>& params,
	cudaStream_t stream);

extern template void fused_linear_scaled_cross_entropy_backward<false, 90>(
	const BackwardTpParamsSm90<90>&,
	cudaStream_t);
extern template void fused_linear_scaled_cross_entropy_backward<true, 90>(
	const BackwardTpParamsSm90<90>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
