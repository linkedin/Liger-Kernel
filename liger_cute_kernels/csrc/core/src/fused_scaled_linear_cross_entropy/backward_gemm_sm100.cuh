#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// SM100 fused scaled linear cross entropy — backward contract.
//
// One persistent 384-thread, cluster-2 kernel fuses dZ, dX and dW together
// with the device-side token-wave loop. The SM90 pair of wave kernels
// (backward_dz_handoff_wave_kernel_sm90 + backward_dx_cluster2_gemm_wave_
// kernel_sm90) is the algorithmic and synchronization reference; the
// executable machinery is the SM100 UMMA/TMEM/TMA machinery already used by
// forward_gemm_roles_sm100.cuh and the SM100 MoE kernels.
//
// ── Per-wave schedule ─────────────────────────────────────────────────────
//   phase dZ    Z = X @ W_local^T over this wave's token rows; the epilogue
//               applies the softmax / entropy gradient and TMA-stores BF16 dZ
//               into the wave workspace. Padded vocabulary columns and rows
//               past `tokens` are written as exact zeros so the dX and dW
//               mainloops never see a ragged K tail.
//   grid sync   full-grid software barrier: dZ publication.
//   phase dX    dX = dZ @ W_local. FP32 TMEM fragments stage through shared
//               memory and are converted to BF16 by the epilogue warps.
//   phase dW    dW_local = dZ^T @ X with the validated six-stage vertical
//               schedule. Wave 0 stores; later waves TMA-reduce-add.
//   grid sync   full-grid software barrier: dZ workspace reuse (skipped after
//               the final wave).
//
// ── Warp plan ─────────────────────────────────────────────────────────────
//   warp 0        reserved for node-local dX reduction; idle in TP1.
//   warp 1        reserved for remote communication; idle in TP1.
//   warp 2        every TMA producer, for all three phases.
//   warp 3        every UMMA issue, for all three phases.
//   warps 4..11   epilogues and TMEM loads. Warp 4 owns the single
//                 Allocator2Sm TMEM allocation for the whole kernel.
//
// One TMEM allocation sized by the max-over-phases column count is reused by
// dZ, dX and dW. The operand arenas form a phase-serial union; pipelines,
// mbarriers and the TMEM handle live outside it. The three phase pipelines are
// initialized once and their states are carried across every token wave.
//
// This header is deliberately CuTe-free — geometry, the wave/tile schedule,
// the warp plan, the parameter block and the launcher declaration only — so
// torch-free consumers can include the umbrella header without CUTLASS. The
// executable producer/consumer/mainloop lives in
// backward_gemm_mainloop_sm100.cuh and is instantiated by
// fused_linear_scaled_cross_entropy_backward_sm100.cu.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backward.cuh"
#include "config.cuh"
#include "dx_reduce.cuh"
#include "liger_cute/detail/tp_reduce.cuh"

#ifndef LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES
#define LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES 5
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS
#define LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS 4096
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT
#define LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT 0
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS
#define LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS 0
#endif

// Diagnostic only. Forces the chunk-granular, one-chunk-deferred dX schedule
// even when no inter-host ring is configured, so the state machine (stage P's
// collective gate, stage F's deferral, the epoch ordering and the all-gather
// replay) can be exercised on a single host. The transport itself is still
// skipped: on one host the node-local reduce-scatter already produced the
// complete shard. Production dispatch is unaffected and continues to select the
// schedule from the configured reduction plan.
#ifndef LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_CHUNK_PIPELINE
#define LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_CHUNK_PIPELINE 0
#endif

namespace liger {
namespace fused_scaled_linear_cross_entropy {

inline constexpr float kBackwardLog2ESm100 = 1.4426950408889634f;
inline constexpr bool kBackwardTp1ComputeOnlySm100 = false;
inline constexpr bool kBackwardNodeLocalReduceSm100 = true;
inline constexpr bool kBackwardRemoteReduceSm100 = true;
inline constexpr bool kBackwardDxConsumedAfterFinalStoreSm100 = true;
inline constexpr int kBackwardSyncVariantSm100 =
	LIGER_CUTE_FSLCE_SM100_BACKWARD_SYNC_VARIANT;
inline constexpr bool kBackwardDiagnosticTimestampsSm100 =
	LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_TIMESTAMPS != 0;
static_assert(
	kBackwardSyncVariantSm100 >= 0 && kBackwardSyncVariantSm100 <= 3,
	"SM100 backward sync variant must be A=0, B=1, C=2, or D=3");

inline constexpr int kBackwardDiagnosticKernelStart = 0;
inline constexpr int kBackwardDiagnosticKernelEnd = 1;
inline constexpr int kBackwardDiagnosticDzTmaStart = 2;
inline constexpr int kBackwardDiagnosticDzTmaEnd = 3;
inline constexpr int kBackwardDiagnosticDzMmaStart = 4;
inline constexpr int kBackwardDiagnosticDzMmaFirstReady = 5;
inline constexpr int kBackwardDiagnosticDzMmaEnd = 6;
inline constexpr int kBackwardDiagnosticDzEpiStart = 7;
inline constexpr int kBackwardDiagnosticDzEpiFirstReady = 8;
inline constexpr int kBackwardDiagnosticDzEpiEnd = 9;
inline constexpr int kBackwardDiagnosticDzGridWaitMax = 10;
inline constexpr int kBackwardDiagnosticDxTmaStart = 11;
inline constexpr int kBackwardDiagnosticDxTmaEnd = 12;
inline constexpr int kBackwardDiagnosticDxMmaStart = 13;
inline constexpr int kBackwardDiagnosticDxMmaFirstReady = 14;
inline constexpr int kBackwardDiagnosticDxMmaEnd = 15;
inline constexpr int kBackwardDiagnosticDxEpiStart = 16;
inline constexpr int kBackwardDiagnosticDxEpiFirstReady = 17;
inline constexpr int kBackwardDiagnosticDxEpiEnd = 18;
inline constexpr int kBackwardDiagnosticDxDwWaitMax = 19;
inline constexpr int kBackwardDiagnosticDwTmaStart = 20;
inline constexpr int kBackwardDiagnosticDwTmaEnd = 21;
inline constexpr int kBackwardDiagnosticDwMmaStart = 22;
inline constexpr int kBackwardDiagnosticDwMmaFirstReady = 23;
inline constexpr int kBackwardDiagnosticDwMmaEnd = 24;
inline constexpr int kBackwardDiagnosticDwEpiStart = 25;
inline constexpr int kBackwardDiagnosticDwEpiFirstReady = 26;
inline constexpr int kBackwardDiagnosticDwEpiEnd = 27;
inline constexpr int kBackwardDiagnosticLocalFirstReady = 28;
inline constexpr int kBackwardDiagnosticLocalReduceEnd = 29;
inline constexpr int kBackwardDiagnosticLocalStoreEnd = 30;
inline constexpr int kBackwardDiagnosticRemoteStart = 31;
inline constexpr int kBackwardDiagnosticRemoteEnd = 32;
inline constexpr int kBackwardDiagnosticRemoteTransportEnd = 33;
inline constexpr int kBackwardDiagnosticEntries = 34;

// exp2 with the SM90 backward's fastmath lowering, so both architectures
// produce bit-identical dZ for the same accumulator.
__host__ __device__ inline float backward_exp2_sm100(float value) {
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return result;
#else
	return exp2f(value);
#endif
}

enum class BackwardWarpRoleSm100 : std::uint8_t {
	kDxLocalReduce,
	kRemoteCommunication,
	kTmaProducer,
	kUmmaProducer,
	kEpilogue,
	kInactive,
};

__host__ __device__ constexpr BackwardWarpRoleSm100 backward_warp_role_sm100(
		int warp_id) {
	return warp_id == 0
		? BackwardWarpRoleSm100::kDxLocalReduce
		: (warp_id == 1
			? BackwardWarpRoleSm100::kRemoteCommunication
			: (warp_id == 2
				? BackwardWarpRoleSm100::kTmaProducer
				: (warp_id == 3
					? BackwardWarpRoleSm100::kUmmaProducer
					: (warp_id >= 4 && warp_id < kNumWarps
						? BackwardWarpRoleSm100::kEpilogue
						: BackwardWarpRoleSm100::kInactive))));
}

// ───────────────────────────────────────────────────────────────────────────
// Geometry
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 100>
struct BackwardGemmConfigSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	static constexpr int kCompute = Compute;

	// Paired-CTA UMMA: the joined tile is M256, each CTA owns 128 rows.
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileM = 256;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kClusterM = 2;

	// All phases use the same joined M256xN256xK64 UMMA tile and one
	// max-over-phases TMEM allocation. Their measured mainloop depths remain
	// phase specific.
	static constexpr int kDzTileN = kTileN;
	static constexpr int kDzMainloopStages = 5;
	static constexpr int kDxTileN = kTileN;
	static constexpr int kDxLogicalTileN = 2 * kDxTileN;
	static constexpr int kDxMainloopStages = 4;
	static constexpr int kDwTileN = kTileN;
	static constexpr int kDwMainloopStages = 6;
	static constexpr int kDwTileM = kTileM;
	static constexpr int kDzLogicalTileN = kDzTileN;
	static constexpr int kDwLogicalTileN = kDwTileN;

	static_assert(
		LIGER_CUTE_FSLCE_SM100_BACKWARD_STAGES == kDzMainloopStages,
		"SM100 backward dZ is fixed to the validated five-stage pipeline");
	static constexpr int kMainloopStages = kDzMainloopStages;

	// TMEM: two N256 accumulator stages, reused by dZ, dX and dW. This is the
	// max over phases, so exactly one Allocator2Sm allocation is taken.
	static constexpr int kAccumulatorStages = 2;
	static constexpr int kTmemStageColumns = kTileN;
	static constexpr int kTmemColumns =
		kAccumulatorStages * kTmemStageColumns;

	static constexpr int kEpilogueChunkN = 32;
	static constexpr int kEpilogueWarpgroups = 2;
	static constexpr int kEpilogueWarps = 8;
	static constexpr int kWarpgroupSize = 4 * kWarpSize;
	static constexpr int kEpilogueThreads = kEpilogueWarps * kWarpSize;
	static constexpr int kWarpgroupTileN = kTileN / kEpilogueWarpgroups;
	static constexpr int kChunksPerWarpgroup =
		kWarpgroupTileN / kEpilogueChunkN;
	static constexpr int kNumThreads =
		fused_scaled_linear_cross_entropy::kNumThreads;

	// Compile-time ablation knob. The production default is 4096 rows, which
	// gives dW a K4096 single-pass store with no inter-wave reduce-add.
	static constexpr int kWaveRows =
		LIGER_CUTE_FSLCE_SM100_BACKWARD_WAVE_ROWS;
	static constexpr int kMTilesPerWave = kWaveRows / kCtaTileM;
	static constexpr int kMPairsPerWave = kMTilesPerWave / kClusterM;
	static constexpr int kDzWorkspaceSlots =
		kBackwardSyncVariantSm100 == 3 && kWaveRows < 4096 ? 2 : 1;

	// The dZ workspace K extent is a whole number of K tiles so the dX and dW
	// mainloops never see a ragged K tail on an operand they own.
	static constexpr int kVocabAlign = kTileK;
	static constexpr int kDwKTiles = kWaveRows / kTileK;

	static constexpr int kDxLocalReduceWarp = 0;
	static constexpr int kRemoteCommunicationWarp = 1;
	static constexpr int kTmaWarp = 2;
	static constexpr int kUmmaWarp = 3;
	static constexpr int kFirstEpilogueWarp = 4;
	static constexpr int kLastEpilogueWarp = 11;

	// Named barriers. 0 is __syncthreads() and is never used in the per-wave
	// hot path. Every barrier below is compute-only: warps 0 and 1 are
	// excluded by construction so the communication warps are never dragged
	// into a GEMM-side rendezvous.
	static constexpr int kWarpgroup0BarrierId = 1;
	static constexpr int kWarpgroup1BarrierId = 2;
	static constexpr int kMmaEpilogueBarrierId = 3;
	static constexpr int kEpilogueBarrierId = 4;
	static constexpr int kComputeBarrierId = 5;
	static constexpr int kDwMmaEpilogueBarrierId = 6;
	static constexpr int kDxMmaEpilogueBarrierId = 7;

	// warps 3..11
	static constexpr int kMmaEpilogueThreads =
		(kLastEpilogueWarp - kUmmaWarp + 1) * kWarpSize;
	// warps 2..11 — owns the phase-serial operand/store arena handoff and the
	// full-grid software barrier.
	static constexpr int kComputeThreads =
		(kLastEpilogueWarp - kTmaWarp + 1) * kWarpSize;
	static constexpr int kDwEpilogueThreads = 4 * kWarpSize;
	static constexpr int kDwMmaEpilogueThreads =
		kWarpSize + kDwEpilogueThreads;
	static constexpr int kDxEpilogueThreads = 4 * kWarpSize;
	static constexpr int kDxMmaEpilogueThreads =
		kWarpSize + kDxEpilogueThreads;

	static_assert(kCtaTileM * kClusterM == kTileM);
	static_assert(kNumThreads == 384);
	static_assert(kDzMainloopStages == 5);
	static_assert(kDxMainloopStages == 4);
	static_assert(kDwMainloopStages == 6);
	static_assert(kWarpgroupTileN % kEpilogueChunkN == 0);
	static_assert(kTmemColumns == 512);
	static_assert(kWaveRows % kTileK == 0);
	static_assert(
		kWaveRows == 1024 || kWaveRows == 2048 || kWaveRows == 4096,
		"SM100 backward wave rows must be 1024, 2048, or 4096");
	static_assert(kWaveRows % kCtaTileM == 0);
	static_assert(kMTilesPerWave % kClusterM == 0);
	static_assert(kMmaEpilogueThreads == 288);
	static_assert(kComputeThreads == 320);
	static_assert(kDwMmaEpilogueThreads == 160);
	static_assert(kDxMmaEpilogueThreads == 160);
	static_assert(kComputeThreads < kNumThreads,
		"the compute barrier must exclude warps 0 and 1, otherwise it "
		"degenerates into __syncthreads()");
	static_assert(
		backward_warp_role_sm100(kDxLocalReduceWarp) ==
		BackwardWarpRoleSm100::kDxLocalReduce);
	static_assert(
		backward_warp_role_sm100(kRemoteCommunicationWarp) ==
		BackwardWarpRoleSm100::kRemoteCommunication);
	static_assert(
		backward_warp_role_sm100(kTmaWarp) ==
		BackwardWarpRoleSm100::kTmaProducer);
	static_assert(
		backward_warp_role_sm100(kUmmaWarp) ==
		BackwardWarpRoleSm100::kUmmaProducer);
	static_assert(
		backward_warp_role_sm100(kFirstEpilogueWarp) ==
		BackwardWarpRoleSm100::kEpilogue);
	static_assert(
		backward_warp_role_sm100(kLastEpilogueWarp) ==
		BackwardWarpRoleSm100::kEpilogue);
};

// Per-phase geometry, kept as named contracts so a phase can be retuned
// without touching the mainloop.
template <int Compute = 100>
struct DzGemmContractSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileN = Config::kDzTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStages = Config::kDzMainloopStages;
};

template <int Compute = 100>
struct DxGemmContractSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kTileM;
	static constexpr int kTileN = Config::kDxTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStages = Config::kDxMainloopStages;
};

template <int Compute = 100>
struct DwGemmContractSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;
	static constexpr int kTileM = Config::kDwTileM;
	static constexpr int kTileN = Config::kDwTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStages = Config::kDwMainloopStages;
};

template <int Compute = 100>
struct BackwardWarpPlanSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;

	static constexpr int kDxLocalReduceWarp = Config::kDxLocalReduceWarp;
	static constexpr int kRemoteCommunicationWarp =
		Config::kRemoteCommunicationWarp;
	static constexpr int kTmaWarp = Config::kTmaWarp;
	static constexpr int kUmmaWarp = Config::kUmmaWarp;
	static constexpr int kFirstEpilogueWarp = Config::kFirstEpilogueWarp;
	static constexpr int kLastEpilogueWarp = Config::kLastEpilogueWarp;
	// The epilogue warp that owns the whole-kernel TMEM allocation.
	static constexpr int kTmemOwnerWarp = Config::kFirstEpilogueWarp;

	static_assert(Config::kNumThreads == kNumThreads);
	static_assert(kLastEpilogueWarp + 1 == kNumWarps);
};

// ───────────────────────────────────────────────────────────────────────────
// Epoch layout for the fused wave loop
//
// Every device-visible signal used by the backward carries the launch epoch so
// a replayed CUDA graph can never observe a stale generation. The wave index
// occupies bits [16, 28) and the operation suffix bits [28, 32), which keeps
// both clear of remote_ring.cuh's step mask.
// ───────────────────────────────────────────────────────────────────────────

inline constexpr int kBackwardWaveEpochShiftSm100 = 16;
inline constexpr int kBackwardWaveEpochBitsSm100 = 12;
inline constexpr int kBackwardMaxWavesSm100 =
	(1 << kBackwardWaveEpochBitsSm100) - 1;
inline constexpr std::uint64_t kBackwardWaveEpochMaskSm100 =
	static_cast<std::uint64_t>(kBackwardMaxWavesSm100)
	<< kBackwardWaveEpochShiftSm100;
inline constexpr std::uint64_t kBackwardDxScatterEpochSuffixSm100 =
	0x10000000u;
inline constexpr std::uint64_t kBackwardDxAllgatherEpochSuffixSm100 =
	0x20000000u;
inline constexpr std::uint64_t kBackwardDxRemoteEpochSuffixSm100 =
	0x50000000u;

static_assert(
	(kBackwardWaveEpochMaskSm100 &
		liger_cute::detail::kRemoteRingStepMask) == 0);
static_assert(
	(kBackwardDxScatterEpochSuffixSm100 &
		kBackwardWaveEpochMaskSm100) == 0);
static_assert(
	(kBackwardDxAllgatherEpochSuffixSm100 &
		kBackwardWaveEpochMaskSm100) == 0);
static_assert(
	(kBackwardDxRemoteEpochSuffixSm100 &
		kBackwardWaveEpochMaskSm100) == 0);
static_assert(
	(kBackwardDxRemoteEpochSuffixSm100 &
		liger_cute::detail::kRemoteRingStepMask) == 0);
static_assert(
	!liger_cute::detail::remote_ring_uses_grid_sync<1>(),
	"the SM100 backward communication warp must not enter a block or grid "
	"barrier");

__host__ __device__ constexpr std::uint64_t backward_wave_epoch_sm100(
		std::uint64_t launch_epoch, std::uint64_t suffix, int wave) {
	return launch_epoch | suffix |
		(static_cast<std::uint64_t>(wave + 1)
			<< kBackwardWaveEpochShiftSm100);
}

__host__ __device__ constexpr bool backward_wave_count_supported_sm100(
		int waves) {
	return waves >= 1 && waves <= kBackwardMaxWavesSm100;
}

// ───────────────────────────────────────────────────────────────────────────
// dX chunk pipeline depth
//
// A chunk is live from the moment its first tile is reduce-scattered until
// warp 0 finalizes it. Because stage F is deferred by exactly one chunk, at
// most two chunks are live at once: chunk k in its IB/finalize tail and chunk
// k+1 in its tile-granular reduce-scatter.
//
// Source and result slots are *not* recycled inside a launch: the packed dX
// shard and the all-gather destination are both addressed by absolute chunk
// index, so the slot space is `num_chunks` deep and a live chunk's storage can
// never be reused underneath it. `kBackwardDxChunkPipelineDepth` is the
// contract the ring transport and the epoch layout have to satisfy, and it is
// asserted against the remote inbox depth below.
// ───────────────────────────────────────────────────────────────────────────

inline constexpr bool kBackwardDiagnosticChunkPipelineSm100 =
	LIGER_CUTE_FSLCE_SM100_BACKWARD_DIAGNOSTIC_CHUNK_PIPELINE != 0;

inline constexpr int kBackwardDxChunkPipelineDepth = 2;
static_assert(
	kBackwardDxChunkPipelineDepth >= 2,
	"deferring the dX finalize by one chunk needs two live chunks");
static_assert(
	kBackwardDxChunkPipelineDepth <=
		liger_cute::detail::kRemoteRingInboxSlots,
	"the inter-host ring must be able to hold every live chunk; "
	"remote_ring.cuh alternates inbox slots by operation sequence");

// Absolute source/result slot of a chunk. Distinct for every chunk of a
// launch, so `kBackwardDxChunkPipelineDepth` live chunks always have disjoint
// storage.
__host__ __device__ constexpr int backward_dx_chunk_slot_sm100(int chunk) {
	return chunk;
}

// The number of chunks whose finalize may still be outstanding when chunk
// `chunk` starts its reduce-scatter.
__host__ __device__ constexpr int backward_dx_pending_chunks_sm100(
		int chunk) {
	return chunk > 0 ? 1 : 0;
}

__host__ __device__ constexpr int backward_remote_merge_worker_sm100(
		int cta, int lane) {
	return cta * kWarpSize + lane;
}

__host__ __device__ constexpr int backward_remote_merge_workers_sm100(
		int grid_ctas) {
	return grid_ctas * kWarpSize;
}

__host__ __device__ constexpr unsigned long long
backward_remote_merge_target_sm100(int chunk, int grid_ctas) {
	return static_cast<unsigned long long>(chunk + 1) *
		static_cast<unsigned long long>(grid_ctas);
}

// ───────────────────────────────────────────────────────────────────────────
// Persistent device-side wave state
//
// Every buffer named here comes from liger::global_buffer_pool(): a fixed
// internal device-private signal block plus the already-configured symmetric
// dX staging / packed shard. Nothing is allocated per launch, no raw
// nvshmem_malloc / nvshmem_free is used and no SymmetricMemoryStack is
// involved: `lse` and `entropy` are ordinary CUDA tensors saved by autograd.
// ───────────────────────────────────────────────────────────────────────────

// Signal block layout (device private, uint64 entries).
inline constexpr int kBackwardSignalGridBarrier = 0;
inline constexpr int kBackwardSignalScatterBase = 8;
inline constexpr int kBackwardSignalRemoteReceived = 16;
inline constexpr int kBackwardSignalRemoteBase = 17;
inline constexpr int kBackwardSignalRemoteMergeArrived = 18;
inline constexpr int kBackwardSignalEntries = 24;

template <int Compute = 100>
struct BackwardWaveWorkspaceSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	// Monotone full-grid arrive/wait counter. Only compute warps touch it.
	std::uint64_t* grid_barrier = nullptr;
	// Reserved for the later local/remote dX reduction phases. TP1 leaves
	// warps 0 and 1 idle and never reads these fields.
	std::uint64_t* dx_scatter_ready = nullptr;
	std::uint64_t* dx_remote_received = nullptr;
	std::uint64_t* dx_remote_ready = nullptr;
	std::uint64_t* dx_remote_merge_arrived = nullptr;
	const std::uint64_t* launch_epoch = nullptr;
	float* packed_shard = nullptr;
	std::size_t packed_shard_elements = 0;
	std::uint32_t* dz_tile_ready = nullptr;
	std::size_t dz_tile_ready_entries = 0;
	std::uint64_t* diagnostics = nullptr;
	int grid_ctas = 0;
	int num_waves = 0;
	int staging_rows = 0;
};

// ───────────────────────────────────────────────────────────────────────────
// Problem description
//
// Identical contract to the SM90 backward: the vocabulary shard is contiguous,
// targets arrive as global int64 indices, and `lse` / `entropy` are the
// forward's already-all-reduced per-token outputs held as ordinary local
// tensors.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 100>
struct BackwardGemmParamsSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;

	const void* x = nullptr;               // BF16 [tokens, hidden]
	const void* weight = nullptr;          // BF16 [local_vocab, hidden]
	const std::int64_t* target = nullptr;  // int64 [tokens], global indices
	const float* grad_output = nullptr;    // FP32 [tokens], upstream NLL scale
	const float* lse = nullptr;            // FP32 [tokens], global LSE
	const float* entropy = nullptr;        // FP32 [tokens], global entropy
	const float* entropy_grad = nullptr;   // FP32 [tokens], entropy scale

	void* grad_input = nullptr;   // X.dtype [tokens, hidden], BF16
	void* grad_weight = nullptr;  // BF16 [local_vocab, hidden], rank local

	void* dz_workspace = nullptr;  // BF16 [kWaveRows, padded_local_vocab]
	std::size_t dz_workspace_bytes = 0;

	int tokens = 0;
	int hidden = 0;
	int local_vocab = 0;
	std::int64_t vocab_start = 0;
	std::int64_t ignore_index = -100;
	float inverse_temperature = 1.0f;
};

// ───────────────────────────────────────────────────────────────────────────
// Launch geometry
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 100>
struct BackwardGemmLaunchSm100 {
	using Config = BackwardGemmConfigSm100<Compute>;
	static constexpr int kCompute = Compute;

	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	__host__ __device__ static constexpr int num_waves(int tokens) {
		return ceil_div(tokens, Config::kWaveRows);
	}

	__host__ __device__ static constexpr int padded_vocab(int local_vocab) {
		return ceil_div(local_vocab, Config::kVocabAlign) *
			Config::kVocabAlign;
	}

	// dZ: joined M256 x N256 tiles over the padded vocabulary.
	__host__ __device__ static constexpr int num_dz_n_tiles(
			int local_vocab) {
		return ceil_div(padded_vocab(local_vocab), Config::kDzTileN);
	}

	__host__ __device__ static constexpr int num_dz_k_tiles(int hidden) {
		return ceil_div(hidden, Config::kTileK);
	}

	// dX: joined M256 x N256 tiles over hidden, K over the padded vocabulary.
	__host__ __device__ static constexpr int num_dx_n_tiles(int hidden) {
		return ceil_div(hidden, Config::kDxTileN);
	}

	__host__ __device__ static constexpr int num_dx_wide_n_tiles(int hidden) {
		return ceil_div(hidden, Config::kDxLogicalTileN);
	}

	__host__ __device__ static constexpr int num_dx_k_tiles(
			int local_vocab) {
		return padded_vocab(local_vocab) / Config::kTileK;
	}

	// dW: joined M256 over the padded vocabulary, N256 over hidden, K over the
	// wave's token rows.
	__host__ __device__ static constexpr int num_dw_m_pairs(
			int local_vocab) {
		return ceil_div(padded_vocab(local_vocab), Config::kDwTileM);
	}

	__host__ __device__ static constexpr int num_dw_n_tiles(int hidden) {
		return ceil_div(hidden, Config::kDwLogicalTileN);
	}

	__host__ __device__ static constexpr int num_dw_k_tiles() {
		return Config::kDwKTiles;
	}

	// Cluster-pair work counts. Every phase rasters cluster pairs, so a
	// fully resident grid of `num_cluster_pairs` clusters covers all of them.
	__host__ __device__ static constexpr int num_dz_cluster_pairs(
			int local_vocab) {
		return Config::kMPairsPerWave * num_dz_n_tiles(local_vocab);
	}

	__host__ __device__ static constexpr int num_dx_cluster_pairs(
			int hidden) {
		return Config::kMPairsPerWave * num_dx_wide_n_tiles(hidden);
	}

	__host__ __device__ static constexpr int num_dx_cluster_pairs_split(
			int hidden, int local_vocab) {
		(void)local_vocab;
		return num_dx_cluster_pairs(hidden);
	}

	__host__ __device__ static constexpr int num_dw_cluster_pairs(
			int hidden, int local_vocab) {
		return num_dw_m_pairs(local_vocab) *
			ceil_div(num_dw_n_tiles(hidden), 2);
	}

	// One wave of dZ, BF16.
	__host__ __device__ static std::size_t dz_workspace_bytes(
			int local_vocab) {
		return static_cast<std::size_t>(Config::kDzWorkspaceSlots) *
			static_cast<std::size_t>(Config::kWaveRows) *
			static_cast<std::size_t>(padded_vocab(local_vocab)) * 2u;
	}

	// Number of CTA-owned FP32 dX tiles produced per wave. Waves are always a
	// whole `kMTilesPerWave` so this is a pure function of the shape on every
	// PE of the team.
	__host__ __device__ static constexpr int dx_tiles_per_wave(int hidden) {
		return Config::kMTilesPerWave * num_dx_n_tiles(hidden);
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Phase selection
//
// The fused production kernel runs all three phases. The benchmark paths
// instantiate the *same* kernel with a single phase enabled, so the tile
// shapes, mainloop pipeline, accumulator pipeline, TMEM plan, epilogue and
// wave-loop scaffolding are bit-identical to production; only the other two
// phases and the dX reduction/communication dependencies are compiled out.
// Nothing else is special-cased, so a benchmark number cannot drift away from
// what production executes.
// ───────────────────────────────────────────────────────────────────────────

inline constexpr int kBackwardPhaseDz = 1;
inline constexpr int kBackwardPhaseDx = 2;
inline constexpr int kBackwardPhaseDw = 4;
inline constexpr int kBackwardPhaseAll =
	kBackwardPhaseDz | kBackwardPhaseDx | kBackwardPhaseDw;
inline constexpr int kBackwardPhaseAuditSkipDxDwBarrier = 8;
inline constexpr int kBackwardPhaseAuditSkipDzGridBarrier = 16;
inline constexpr int kBackwardPhaseAuditSkipDzDrain = 32;
inline constexpr int kBackwardPhaseAuditForceDzGridBarrier = 64;
inline constexpr int kBackwardPhaseAuditForceDxDwBarrier = 128;

// Benchmark-admissible masks. A single phase isolates that GEMM completely.
// The dZ-prefixed pairs keep the production dZ -> grid barrier -> phase
// ordering so a phase that must observe a published dZ workspace can be
// measured differentially against dZ alone.
__host__ __device__ constexpr bool backward_phase_is_isolated_sm100(
		int phase_mask) {
	return phase_mask == 0 ||
		phase_mask == kBackwardPhaseDz ||
		phase_mask == kBackwardPhaseDx ||
		phase_mask == kBackwardPhaseDw ||
		phase_mask == (kBackwardPhaseDz | kBackwardPhaseDx) ||
		phase_mask == (kBackwardPhaseDz | kBackwardPhaseDw) ||
		phase_mask == (kBackwardPhaseDx | kBackwardPhaseDw) ||
		phase_mask ==
			(kBackwardPhaseDx | kBackwardPhaseDw |
				kBackwardPhaseAuditSkipDxDwBarrier) ||
		phase_mask ==
			(kBackwardPhaseDz | kBackwardPhaseDx |
				kBackwardPhaseAuditSkipDzGridBarrier) ||
		phase_mask ==
			(kBackwardPhaseDz | kBackwardPhaseDw |
				kBackwardPhaseAuditSkipDzGridBarrier) ||
		phase_mask ==
			(kBackwardPhaseDz | kBackwardPhaseDx |
				kBackwardPhaseAuditSkipDzDrain) ||
		phase_mask ==
			(kBackwardPhaseDz |
				kBackwardPhaseAuditForceDzGridBarrier) ||
		phase_mask ==
			(kBackwardPhaseDx |
				kBackwardPhaseAuditForceDxDwBarrier);
}

// Multiply-accumulate count the kernel actually issues for one phase, using
// the padded extents the tiling really covers (whole M256/N256 tiles over the
// K-tile-aligned vocabulary). The benchmark harness compares against a cuBLAS
// GEMM of exactly these extents.
__host__ __device__ inline double backward_phase_flops_sm100(
		int phase_mask, int tokens, int hidden, int local_vocab);

__host__ __device__ inline double backward_phase_flops_sm100(
		int phase_mask, int tokens, int hidden, int local_vocab) {
	using Config = BackwardGemmConfigSm100<100>;
	using Launch = BackwardGemmLaunchSm100<100>;
	double waves = Launch::num_waves(tokens);
	double wave_rows = Config::kWaveRows;
	double padded_vocab = Launch::padded_vocab(local_vocab);
	double dz_n = static_cast<double>(
		Launch::num_dz_n_tiles(local_vocab)) * Config::kDzTileN;
	double dx_n = static_cast<double>(
		Launch::num_dx_n_tiles(hidden)) * Config::kDxTileN;
	double dw_m = static_cast<double>(
		Launch::num_dw_m_pairs(local_vocab)) * Config::kDwTileM;
	double dw_n = static_cast<double>(
		Launch::num_dw_n_tiles(hidden)) * Config::kDwLogicalTileN;
	if (phase_mask == kBackwardPhaseDz) {
		return 2.0 * waves * wave_rows * dz_n * hidden;
	}
	if (phase_mask == kBackwardPhaseDx) {
		return 2.0 * waves * wave_rows * dx_n * padded_vocab;
	}
	if (phase_mask == kBackwardPhaseDw) {
		return 2.0 * waves * dw_m * dw_n * wave_rows;
	}
	return 0.0;
}

// ───────────────────────────────────────────────────────────────────────────
// Host launcher
//
// `params.grad_input` receives the TP-reduced dX and `params.grad_weight` this
// rank's dW. Collective: every PE of the configured TP team calls this with
// the same shapes and TilesPerReduce.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 100>
struct BackwardTpParamsSm100 {
	static constexpr int kCompute = Compute;

	BackwardGemmParamsSm100<Compute> gemm = {};

	// Retained for FFI parity with SM90. The fused SM100 dX message is exactly
	// one M128xN256 tile; this knob still selects the reserved staging
	// capacity and is validated against the configured maximum.
	int tiles_per_reduce = 2;
	// Retained for SM90 API compatibility. The SM100 persistent kernel assigns
	// warp 0 to node-local reduction and warp 1 to remote communication.
	int num_comm_channels = 4;
	// NVSHMEM team used by the matching forward. The SM100 IB path supports two
	// uniform hosts with 1, 2, 4, or 8 selected GPUs per host.
	std::int64_t team_handle = 0;
};

// Collective over params.team_handle. Produces globally reduced BF16 dX and
// rank-local BF16 dW from the forward's saved FP32 lse/entropy tensors. The
// SM100 specialization runs dZ, dX, dW, local NVLS and any inter-host ring in
// one persistent cluster-2 launch.
template <bool ReturnEntropy, int Compute = 100>
void fused_linear_scaled_cross_entropy_backward_sm100(
	const BackwardTpParamsSm100<Compute>& params,
	cudaStream_t stream);

extern template void
fused_linear_scaled_cross_entropy_backward_sm100<false, 100>(
	const BackwardTpParamsSm100<100>&,
	cudaStream_t);
extern template void
fused_linear_scaled_cross_entropy_backward_sm100<true, 100>(
	const BackwardTpParamsSm100<100>&,
	cudaStream_t);

// Benchmark-only entry point. Runs exactly one of the three GEMM phases with
// the production geometry and epilogue and without any reduction or
// communication dependency, so the phase can be compared against a matched
// cuBLAS BF16 GEMM. `phase_mask` must be one of kBackwardPhaseDz /
// kBackwardPhaseDx / kBackwardPhaseDw.
void fused_linear_scaled_cross_entropy_backward_phase_bench_sm100(
	const BackwardTpParamsSm100<100>& params,
	bool return_entropy,
	int phase_mask,
	cudaStream_t stream);

// Copies the latest diagnostic timestamp/counter block to `output`.
// `entries` must be at least kBackwardDiagnosticEntries.
void fused_linear_scaled_cross_entropy_backward_diagnostics_sm100(
	std::uint64_t* output,
	std::size_t entries,
	cudaStream_t stream);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
