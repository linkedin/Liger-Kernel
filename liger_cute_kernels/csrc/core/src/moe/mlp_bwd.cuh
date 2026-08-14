#pragma once

// ═══════════════════════════════════════════════════════════════════
// Fused MLP Backward — batch-interleaved Phase 1 + Phase 2
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors forward mlp.cuh: pipeline construction + tile/batch loop.
// Called once for local tiles, once for remote tiles.
//
// Per batch (gridDim.x tiles):
//   Phase 1: one tile per CTA (a→b→c→d sub-phases)
//   Global barrier
//   Phase 2: all CTAs do weight grads for this batch's K-blocks
//   Global barrier
//
// ═══════════════════════════════════════════════════════════════════

#include "mlp.cuh"           // kMlpBarrierId, mlp1 pipe helpers
#include "mlp1_fused_act.cuh"  // bwd Phase 1a — TileM=128 cooperative-M-split
#include "mlp2_t_fused.cuh"  // bwd Phase 1b' — single-m-tile entry over mlp2_t.cuh
#include "mlp5.cuh"
#include "mlp5_fused.cuh"    // mlp5_fused_producer/consumer (single-m-tile entry)
#include "mlp3.cuh"          // Mlp3Traits, Mlp3FusedSmem
#include "mlp4.cuh"          // Mlp4Traits, Mlp4FusedSmem
#include "silu_bwd_fused.cuh"  // silu_bwd_pair_tile — separate elementwise phase
#include "cta_barrier.cuh"   // target-based CtaCounterBarrier

#include <climits>           // INT_MAX for Phase 2 per-batch range scan

namespace liger {

using namespace cute;

// MLP-warps barrier count for the bwd kernel. Hopper uses the historical
// 9-warp partition (w0 + w4-11). Blackwell mirrors the forward path: comm
// uses only warps 1-2 and warp 3 joins MLP as the dedicated UMMA producer.
template <int Compute>
static constexpr int kMlpBwdBarrierThreadsFor = (Compute == 100 ? 10 : 9) * 32;
static_assert(kMlpBwdBarrierThreadsFor<90> == 288);
static_assert(kMlpBwdBarrierThreadsFor<100> == 320);

// Compute == 100 (Blackwell/SM100) always means the Phase-2 GEMMs (mlp3 AND
// mlp4) run the SM100 ClusterM=2 paired-CTA path; Compute == 90 (Hopper/SM90)
// always means plain 1SM. There is no per-phase or per-trait choice anymore
// (see mlp3.cuh / mlp4.cuh's unified mlp{3,4}_producer<Traits, Compute> /
// mlp{3,4}_consumer<Traits, Compute>), so `Compute == 100` alone drives the
// TMEM allocator choice (Allocator2Sm vs Allocator1Sm) and the extra
// tmem_pair_barrier below: both mlp3 and mlp4 share ONE TMEM allocation per
// CTA-pair, so the pair must rendezvous around the allocate/free calls.

// ── Compute == 100 (paired-CTA) caller contract ──
//
// This header only builds the __device__ Phase-2 GEMM bodies; it does not
// launch the kernel or build TMA descriptors, so the following invariants
// must hold at the call site (host launcher, e.g. moe_bwd.cu) whenever
// Compute == 100 is selected — violating any of them is a silent-corruption
// or launch-failure risk, not something this file can assert at compile time:
//   • Grid must be launched EVEN (gridDim.x·gridDim.y divisible by
//     Traits{3,4}::ClusterM) — the flat cell-walk above (cell_start =
//     flat_id / ClusterM, cell_stride = total_phase2_ctas / ClusterM)
//     assumes every CTA has a live cluster peer.
//   • Launch via cudaLaunchKernelEx with clusterDim = (ClusterM, 1, 1) (or
//     the ClusterShape Traits{3,4} defines) and the
//     cudaLaunchAttributeClusterDimension /
//     NonPortableClusterSizeAllowed attribute set (ClusterM > 8 or any
//     non-default shape needs the "non-portable" opt-in).
//   • The dYT/Z (mlp3) and X/dU^T/dV^T (mlp4) TMA descriptors passed in
//     (TmaLoadDYT3/TmaLoadZT3/... below) must be built with
//     make_tma_copy_A_sm100 / make_tma_copy_B_sm100 (2-CTA-aware multicast
//     descriptors) — a plain SM90 make_tma_copy descriptor does not carry
//     the multicast-mask semantics mlp{3,4}_producer's Compute == 100
//     `copy(tma.with(barrier, mcast_mask), ...)` calls rely on.
//   • MlpFusedBwdSmem's static size (larger for the Compute == 100
//     Mlp3FusedSmem2Sm variant) must fit within the dynamically-queried
//     cudaDevAttrMaxSharedMemoryPerBlockOptin limit, and the kernel's
//     cudaFuncAttributeMaxDynamicSharedMemorySize must be opted in to that
//     size before launch — this header assumes the smem struct it defines
//     already fits whatever the host allocated.
// See Compute == 100's use below (TMEM Allocator2Sm + tmem_pair_barrier) and
// the Compute == 90/100 static_asserts in mlp_fused_bwd_run_batches /
// mlp_fused_bwd / mlp_fused_bwd_dual for the invariants this file DOES
// enforce.

// ═══════════════════════════════════════════════════════════════════
// Phase-1 sub-batch grouping (mlp1/2/5 small chunks, mlp3/4 one big batch)
// ═══════════════════════════════════════════════════════════════════
//
// Remote BWD groups kBwdSubBatch Phase-1 sub-batches before running ONE
// Phase-2 (mlp3/mlp4) over all of them at once. Phase 1 (mlp1/2/5) prefers
// SMALL per-batch chunks (less CTA barrier-stall); Phase 2 (weight grads,
// dA/dB/dC) prefers a LONG K-reduction so the cooperative cell-walk fills
// the grid (deeper outer_split / k_split). Grouping serves both.
//
// Buffer survival: the SubBatch sub-batches keep their X/dY comm-staging slots
// un-released until the grouped Phase 2 consumes them, and their Z/dU/dV scratch
// slots must not be overwritten. This needs the staging ring to be DEEPER than
// the group span — remote_num_m_tiles > SubBatch·grid_x — so the group's live
// slots fit with comm prefetch headroom (host-checked in moe_bwd.cu). No ring/
// group ALIGNMENT is required: the grouped Phase-2 window wraps the ring exactly
// as the original per-batch loop does (a window that runs past the ring end is
// only correct when it does not actually straddle — i.e. global_total ≤ ring or
// grid_x·SubBatch divides the ring — same property the un-grouped loop has).
// SubBatch == 1 reduces the grouped loop EXACTLY to the original per-batch loop.
//
// The sub-batch count is a pure TEMPLATE ARGUMENT (SubBatch) on moe_fused_bwd /
// mlp_fused_bwd_dual, sourced from MoeBwdConfig::kSubBatch (whose default is a
// literal — see moe_bwd.cu). Distinct SubBatch instantiations coexist in one
// binary; there is no preprocessor knob.
static constexpr int kBwdSubBatchMax = 8;   // upper bound for the per-group save arrays

// ═══════════════════════════════════════════════════════════════════
// Fused backward shared memory — union of all phases
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5, int Compute = 90>
struct MlpFusedBwdSmem {
	union {
		Mlp1FusedActSmem<Traits1> mlp1;     // TileM=128 cooperative-M-split
		Mlp2TFusedSmem<Traits2T>  mlp2t;    // TileM=128 cooperative-M-split (mlp2_t.cuh)
		Mlp5Smem<Traits5>         mlp5;     // TileM=128 cooperative-M-split
		// mlp4's fused smem is traits-conditional internally (see mlp4.cuh),
		// so it needs no separate 2SM specialization here. mlp3 instead uses
		// a wholly distinct 2SM smem layout (Mlp3FusedSmem2Sm — paired-CTA
		// accumulator pipeline in place of mlp3's normal AccumulatorPipeline),
		// selected purely by Compute == 100 (Blackwell always pairs mlp3's
		// Phase-2 GEMM across CTAs; Compute == 90 never does).
		cute::conditional_t<
			Compute == 100,
			Mlp3FusedSmem2Sm<Traits3>,
			Mlp3FusedSmem<Traits3>> mlp3;
		Mlp4FusedSmem<Traits4>    mlp4;
	};

	// Phase 2 per-expert K-range scratch lives in gmem (MlpBwdBufs:
	// expert_k_starts / expert_k_ends). Allocated to
	// exactly `experts_per_pe` ints — no compile-time cap.

	// Phase 1a, 1b', 1d: single fused pipe each (1c silu_bwd is
	// elementwise gmem→gmem, no pipe).
	typename Mlp1MainloopPipelineFor<Traits1, Compute>::SharedStorage pa_pipe;
	typename Mlp2TMainloopPipelineFor<Traits2T, Compute>::SharedStorage pb_pipe;
	typename Mlp5MainloopPipelineFor<Traits5, Compute>::SharedStorage pd_pipe;

	// Phase 2: single fused pipe each for mlp3 and mlp4. Outputs (dA,
	// dB, dC) are accumulated via TMA_REDUCE_ADD to gmem — no prev-pipe
	// needed.
	typename Mlp3MainloopPipelineFor<Traits3, Compute>::SharedStorage p3_pipe;
	typename Mlp4MainloopPipelineFor<Traits4, Compute>::SharedStorage p4_pipe;
	alignas(16) uint32_t tmem_base;
	// Rendezvous barrier for the paired-CTA TMEM allocation window
	// (Compute == 100 only — one CTA-pair shares one TMEM allocation, so
	// both peers must arrive before either frees it). Collapses to a plain
	// uint32_t placeholder (no ClusterBarrier construction/arrive/wait) on
	// Compute == 90, which never pairs CTAs.
	alignas(16) cute::conditional_t<
		Compute == 100,
		cutlass::arch::ClusterBarrier,
		uint32_t> tmem_pair_barrier;
};

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5>
struct MlpBwdTmemColumns {
	static constexpr int kMlp1Cols  = Traits1::AccStages  * (2 * Traits1::TileN);
	static constexpr int kMlp2TCols = Traits2T::AccStages * Traits2T::TileN;
	static constexpr int kMlp5Cols  = Traits5::AccStages  * Traits5::TileN;
	static constexpr int kMlp3Cols  = Traits3::AccStages  * Traits3::TileN;
	static constexpr int kMlp4Cols  = Traits4::AccStages * (2 * Traits4::WgTileN);

	static constexpr int kPhase15Cols = (kMlp1Cols > kMlp5Cols)
		? kMlp1Cols : kMlp5Cols;
	static constexpr int kPhase2T3Cols = (kMlp2TCols > kMlp3Cols)
		? kMlp2TCols : kMlp3Cols;
	static constexpr int kPhase1235Cols = (kPhase15Cols > kPhase2T3Cols)
		? kPhase15Cols : kPhase2T3Cols;
	static constexpr int value = (kPhase1235Cols > kMlp4Cols)
		? kPhase1235Cols : kMlp4Cols;
	static_assert(value <= 512, "BWD SM100 TMEM allocation must fit one CTA's 512 columns");
};

// ═══════════════════════════════════════════════════════════════════
// MlpBwdDims — invariant backward dimensions (__grid_constant__)
// ═══════════════════════════════════════════════════════════════════

struct MlpBwdDims {
	int hidden_dim;
	int intermediate_dim;

	// Flat 1-D launch geometry (host-computed). The grid is launched as
	// dim3(num_blocks) with num_blocks = num_sms, so gridDim is no longer
	// the logical grid. Phase 1 + comm partition over the GEMM-active subset:
	//   n_gemm = floor_NS = largest multiple of NSplit ≤ num_sms
	//   grid_x = n_gemm / NSplit                 (Phase 1 column count)
	// CTAs with flat_id ≥ n_gemm fall through Phase 1/comm but still run
	// Phase 2 (grid-stride over all num_blocks CTAs) and every barrier.
	int grid_x;
	int n_gemm;

	// Phase 1a: recompute
	int total_n_rows_1;
	int num_n_tiles_1;
	int num_k_tiles_1;

	// Phase 1b: dZ = dY @ A
	int total_k_cols_2t;
	int num_n_tiles_2t;
	int num_k_tiles_2t;

	// Phase 1d: dX = dU@B + dV@C
	int total_k_cols_5;
	int num_n_tiles_5;
	int num_k_tiles_5;

	// Phase 2: mlp3 (dA)
	int num_m_tiles_3;
	int num_n_tiles_3;
	int total_n_rows_3;

	// Phase 2: mlp4 (dB, dC)
	int num_m_tiles_4;
	int num_n_tiles_4;
	int total_m_rows_4;

	int* phase_counter;

	// Scalar params (in __grid_constant__ to avoid register pressure)
	int num_tokens;
	int num_m_tiles;            // Phase 1 output buffer height (in m-tiles).
	                            // Local: total tile capacity (= max_total_slots/TileM).
	                            // Remote: gridDim.x * CommNumStages (the staging ring).
	                            // Phase 2 wraps bk_start modulo this when computing
	                            // the slot index into z_buf/du_buf/dv_buf — for the
	                            // local pass that's a no-op; for the remote pass it
	                            // brings batches ≥ NumStages back into the live ring.
	int total_m_tiles_in_pass;  // Phase 2 K-block iteration bound (in m-tiles).
	                            // Local: equals num_m_tiles (existing semantics — Phase 2
	                            // walks the full padded buffer and skips fake K-blocks
	                            // via the expert_for_k_block sentinel).
	                            // Remote: equals smem.comm.global_total — strictly less
	                            // than num_m_tiles when per_cta_tiles > NumStages, so
	                            // batches that would walk past the ring still run Phase 2
	                            // (with bk_start wrapped) instead of being silently
	                            // clamped to bk_tiles=0.
	int num_batches;
	int local_expert_start;
	int experts_per_pe;  // Phase 3/4 use this to skip k-blocks whose expert
	                     // falls outside [local_expert_start,
	                     // local_expert_start + experts_per_pe). Required
	                     // because expert_for_k_block (built post-sort) holds
	                     // GLOBAL expert IDs spanning all PEs, but dA/dB/dC
	                     // on this PE only have slots for the local subset.

	// Phase 2 virtual-grid block count along the non-NSplit2 axis:
	//   phase2_num_blocks = (gridDim.x * gridDim.y) / NSplit2.
	// Host-known (depends only on launch grid + template constant NSplit2),
	// so storing here lets every batch's mlp3/mlp4 call read it from grid
	// constant memory instead of recomputing `gridDim.x * gridDim.y /
	// NSplit2` per batch.
	int phase2_num_blocks;

	// Remote-pass overrides for num_tokens / num_m_tiles. Both are host-
	// known (gridDim.x * CommNumStages and that × Traits1::TileM); kept here
	// so moe_fused_bwd's remote_dims build can just assign from grid constant
	// instead of computing on device. The remote-only fields stay = 0 for
	// the local pass (the local pass doesn't read them).
	int remote_num_m_tiles;
	int remote_num_tokens;

	// 1 when the bwd TMA-GET comm path is active (NVLink P2P + divisible
	// hidden_dim + bounce smem fits). Selects the comm warp layout AND the
	// remote MLP iterator's producer/consumer counts (1·NC get / 2·NC put
	// vs 2·NC get / 1·NC put). Host-set; 0 keeps the original getmem layout.
	int tma_get_enabled;

	// (efkb_stride_3 / efkb_stride_4 moved to template parameters on
	// mlp_fused_bwd — see EfkbStride3/EfkbStride4 below. They were always
	// compile-time constants per call site (local: 1, remote:
	// Traits1::TileM/Traits{3,4}::TileK), so passing them as runtime ints
	// just kept Traits1::TileM and the divides live in registers across the
	// whole batch loop. Hoisting to templates lets the compiler constant-
	// fold the `/1` (local) into a no-op and the remote ratios into a shift.
};

// ═══════════════════════════════════════════════════════════════════
// MlpBwdBufs — buffer pointers (__grid_constant__)
// ═══════════════════════════════════════════════════════════════════

template <typename Element>
struct MlpBwdBufs {
	// du_buf holds U' (silu-bwd coefficient) after mlp1, then dU after silu_bwd.
	// dv_buf holds V' (silu-bwd coefficient) after mlp1, then dV after silu_bwd.
	// dz_buf holds dZ between mlp2_t (Phase 1b') and silu_bwd (Phase 1c).
	Element* du_buf;
	Element* dv_buf;
	Element* dz_buf;

	// Expert id per K-block, read by Phase 2's mlp4 and mlp3 respectively.
	// They hold the same logical content — for K-block k, the local expert
	// id of the tile that K-block belongs to — but the source array differs
	// because each Phase-2 sub-pass has a different freshness requirement.
	//
	//   expert_for_k_block_mlp4  is read by mlp4 (which reads X). In the
	//                            remote pass it points at
	//                            CommBuffersBwd::tile_expert_ids_x — comm
	//                            won't overwrite it until MLP's release_src,
	//                            which is deferred past mlp4.
	//
	//   expert_for_k_block_mlp3  is read by mlp3 (which reads dY). In the
	//                            remote pass it points at
	//                            CommBuffersBwd::tile_expert_ids_dy — comm
	//                            won't overwrite it until MLP's release_dy,
	//                            which is deferred past mlp3.
	//
	// In the local pass and the standalone bwd kernel, both pointers alias
	// the same post-sort expert_for_k_block built by build_expert_for_k_block
	// — there is no comm race to defend against, and a single source is
	// sufficient.
	const int* expert_for_k_block_mlp4;
	const int* expert_for_k_block_mlp3;

	// Per-pass per-expert K-range scratch for Phase 2. Filled once at
	// the start of each pass (in the bwd preamble) and consumed by both
	// mlp3 and mlp4 batches — they require Traits3::TileK == Traits4::
	// TileK (the codebase already enforces this since a single
	// expert_for_k_block indexes both), so one shared scan suffices.
	// Sized exactly `experts_per_pe` ints each.
	int* expert_k_starts;
	int* expert_k_ends;

	Element const* gmem_da;
	Element const* gmem_db;
	Element const* gmem_dc;
	int* barrier_counter;
};

// ═══════════════════════════════════════════════════════════════════
// MLP-warps-only barriers (BWD)
// ═══════════════════════════════════════════════════════════════════
//
// Both barriers use the target-based CtaCounterBarrier (see
// cta_barrier.cuh). The earlier modulo variants were susceptible to a
// CTA-skew deadlock where a fast CTA could push the counter past the
// `% stride == 0` window before slow CTAs polled.
//
//   • MlpBwdGlobalBarrier:  cross-CTA, stride = gridDim.x * gridDim.y.
//                           Counter is a single int in device memory.
//   • MlpBwdXBarrier:       per-blockIdx.x, stride = gridDim.y.
//                           Counter is a gmem array of length gridDim.x.
template <int Compute>
using MlpBwdCtaBarrierT =
	CtaCounterBarrier<kMlpBwdBarrierThreadsFor<Compute>, kMlpBarrierId>;

// ═══════════════════════════════════════════════════════════════════
// Per-phase inline runners
// ═══════════════════════════════════════════════════════════════════
//
// Each phase of the bwd batch loop (1a / 1b' / 1d / 2-mlp4 / 2-mlp3) is
// pulled out into its own __forceinline__ device function. They take the
// per-phase PipelineState by REFERENCE — the caller (`mlp_fused_bwd`)
// holds the persistent state at function scope, and the inline body sees
// it as a direct register reference once inlined. Using actual functions
// instead of capture-all-by-reference lambdas avoids forcing the captured
// state into stack/local memory (CUDA closures pin captured vars to
// addressable storage, even when __forceinline__ would otherwise let
// them stay in registers).
//
// State persistence: the in-smem pipeline barriers carry a phase counter
// that advances each iteration and persists across phase invocations. To
// keep correctness with construction-inside-the-phase, the outer scope
// persists only a monotonic `uint32_t count` per phase (1 reg vs the
// 3-reg full PipelineState). The phase function:
//   1. Constructs a fresh PipelineState locally (warp-role-aware initial
//      phase: producer starts phase=1, consumer starts phase=0).
//   2. Calls `advance(count)` to align it with the actual smem-barrier
//      phase as left by the previous invocation.
//   3. Runs the phase, naturally advancing local state.
//   4. Writes `count = state.count()` back through the reference.
// This shrinks the cross-phase live set from 5×3 = 15 regs to 5×1 = 5 regs
// without re-introducing the desync bug we hit when state was lambda-local
// (T4096_D4096_I14336_E16_K4: dA off by 10% on expert 7 when num_iter %
// (2*Stages) != 0).

template <typename Pipeline>
__device__ __forceinline__ cutlass::PipelineState<Pipeline::Stages>
mlp_bwd_resume_state(uint32_t count, bool is_producer) {
	auto s = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: cutlass::PipelineState<Pipeline::Stages>{};
	s.advance(count);
	return s;
}

template <typename Traits1, int Compute = 90, typename TmaLoadX, typename TmaLoadW1,
          typename TmaStoreU, typename TmaStoreV, typename TmaStoreZ>
__device__ __forceinline__ void mlp_bwd_run_phase_1a(
		Mlp1MainloopPipelineFor<Traits1, Compute>& pa_pipe,
		uint32_t& pa_count,
		Mlp1FusedActSmem<Traits1>& smem_mlp1,
		TmaLoadX  const& tma_load_x,
		TmaLoadW1 const& tma_load_b_fwd,
		TmaLoadW1 const& tma_load_c_fwd,
		TmaStoreU const& tma_store_du,
		TmaStoreV const& tma_store_dv,
		TmaStoreZ const& tma_store_z,
		int warp_id, int m, int expert,
		const MlpBwdDims& dims,
		uint32_t tmem_base,
		int split_idx = -1, int num_splits = -1) {
	int ns = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int nc = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	auto s = mlp_bwd_resume_state<Mlp1MainloopPipelineFor<Traits1, Compute>>(
		pa_count, warp_id == 0);
	if (warp_id == 0)
		mlp1_fused_act_producer<Traits1, Compute == 90>(pa_pipe, s,
			smem_mlp1, tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
			m, (Compute == 90) ? expert : expert * dims.num_n_tiles_1,
			dims.num_tokens, dims.hidden_dim, dims.intermediate_dim,
			dims.experts_per_pe, dims.total_n_rows_1,
			dims.num_n_tiles_1, dims.num_k_tiles_1,
			ns, nc);
	if constexpr (Compute == 100) smem_mlp1.tmem_base = tmem_base;
	constexpr int kFirstMlp12ConsumerWarp = (Compute == 100) ? 3 : 4;
	if (warp_id >= kFirstMlp12ConsumerWarp)
		mlp1_fused_act_consumer<Traits1, Compute>(pa_pipe, s,
			smem_mlp1, tma_store_du, tma_store_dv, tma_store_z,
			m, dims.intermediate_dim, dims.num_m_tiles,
			dims.num_n_tiles_1, dims.num_k_tiles_1,
			ns, nc);
	pa_count = s.count();
}

template <typename Traits2T, int Compute = 90, typename TmaLoadDY, typename TmaLoadW2T,
          typename TmaStoreDZ>
__device__ __forceinline__ void mlp_bwd_run_phase_1b(
		Mlp2TMainloopPipelineFor<Traits2T, Compute>& pb_pipe,
		uint32_t& pb_count,
		Mlp2TFusedSmem<Traits2T>& smem_mlp2t,
		TmaLoadDY  const& tma_load_dy,
		TmaLoadW2T const& tma_load_a_col,
		TmaStoreDZ const& tma_store_dz,
		int warp_id, int m, int expert,
		const MlpBwdDims& dims,
		uint32_t tmem_base,
		int split_idx = -1, int num_splits = -1) {
	int ns = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int nc = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	auto s = mlp_bwd_resume_state<Mlp2TMainloopPipelineFor<Traits2T, Compute>>(
		pb_count, warp_id == 0);
	if (warp_id == 0)
		mlp2_t_fused_producer<Traits2T, Compute == 90>(pb_pipe, s,
			smem_mlp2t, tma_load_dy, tma_load_a_col,
			m, (Compute == 90) ? expert : expert * dims.num_k_tiles_2t,
			dims.num_tokens, dims.hidden_dim, dims.intermediate_dim,
			dims.experts_per_pe, dims.total_k_cols_2t,
			dims.num_n_tiles_2t, dims.num_k_tiles_2t,
			ns, nc);
	if constexpr (Compute == 100) smem_mlp2t.tmem_base = tmem_base;
	constexpr int kFirstMlp12ConsumerWarp = (Compute == 100) ? 3 : 4;
	if (warp_id >= kFirstMlp12ConsumerWarp)
		mlp2_t_fused_consumer<Traits2T, Compute>(pb_pipe, s,
			smem_mlp2t, tma_store_dz,
			m, dims.intermediate_dim, dims.num_m_tiles,
			dims.num_n_tiles_2t, dims.num_k_tiles_2t,
			ns, nc);
	pb_count = s.count();
}

template <typename Traits5, int Compute = 90, typename TmaLoadDU, typename TmaLoadDV,
          typename TmaLoadB, typename TmaLoadC, typename TmaStoreDX>
__device__ __forceinline__ void mlp_bwd_run_phase_1d(
		Mlp5MainloopPipelineFor<Traits5, Compute>& pd_pipe,
		uint32_t& pd_count,
		Mlp5Smem<Traits5>& smem_mlp5,
		TmaLoadDU  const& tma_load_du,
		TmaLoadDV  const& tma_load_dv,
		TmaLoadB   const& tma_load_b_col,
		TmaLoadC   const& tma_load_c_col,
		TmaStoreDX const& tma_store_dx,
		int warp_id, int m, int expert,
		const MlpBwdDims& dims,
		uint32_t tmem_base,
		int split_idx = -1, int num_splits = -1) {
	int ns = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int nc = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	auto s = mlp_bwd_resume_state<Mlp5MainloopPipelineFor<Traits5, Compute>>(
		pd_count, warp_id == 0);
	if (warp_id == 0)
		mlp5_fused_producer<Traits5, Compute == 90>(pd_pipe, s,
			smem_mlp5,
			tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col,
			m, (Compute == 90) ? expert : expert * dims.num_k_tiles_5,
			dims.num_tokens, dims.hidden_dim, dims.intermediate_dim,
			dims.experts_per_pe, dims.total_k_cols_5,
			dims.num_n_tiles_5, dims.num_k_tiles_5,
			ns, nc);
	if constexpr (Compute == 100) smem_mlp5.tmem_base = tmem_base;
	constexpr int kFirstMlp5ConsumerWarp = (Compute == 100) ? 3 : 4;
	if (warp_id >= kFirstMlp5ConsumerWarp)
		mlp5_fused_consumer<Traits5, Compute>(pd_pipe, s,
			smem_mlp5, tma_store_dx,
			m, dims.hidden_dim,
			dims.num_m_tiles, dims.num_n_tiles_5, dims.num_k_tiles_5,
			ns, nc);
	pd_count = s.count();
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2 per-batch build-ranges scan
// ═══════════════════════════════════════════════════════════════════
//
// Warp 0 fills (k_start, k_end) per local expert for the current batch.
// Empty experts stay at (INT_MAX, 0); the chunk loop skips them via
// `k_end <= k_start`. Ends with a NamedBarrier::sync so the caller can
// proceed straight into the producer/consumer without extra plumbing.
//
// Called once per (batch, phase). mlp3 and mlp4 use different
// expert_for_k_block arrays and strides, so each phase scans its own.

// ring_kb (K-blocks in the staging ring): when > 0 the window [k_offset,
// k_offset+num_k_in_batch) is in UNWRAPPED coordinates, and the physical ring
// slot for the expert-id lookup is (k_offset+kb) % ring_kb — while the recorded
// k_start/k_end ranges stay UNWRAPPED so each expert's range is contiguous even
// when the grouped window straddles the ring wrap. ring_kb == 0 (default) means
// no wrap (the un-grouped / local-pass callers, whose window already fits).
template <int Compute = 90>
__device__ __forceinline__ void mlp_bwd_phase_2_build_ranges(
		int* k_starts,
		int* k_ends,
		const int* expert_for_k_block,
		int k_offset,
		int num_k_in_batch,
		int experts_per_pe,
		int local_expert_start,
		int efkb_stride,
		int warp_id,
		int ring_kb = 0) {
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;
	if (warp_id == 0) {
		int lane = threadIdx.x;
		for (int e = lane; e < experts_per_pe; e += 32) {
			k_starts[e] = INT_MAX;
			k_ends[e]   = 0;
		}
		__syncwarp();
		for (int kb = lane; kb < num_k_in_batch; kb += 32) {
			int kb_abs  = k_offset + kb;                                 // unwrapped
			int kb_phys = (ring_kb > 0) ? (kb_abs % ring_kb) : kb_abs;   // ring slot
			int eg = expert_for_k_block[kb_phys / efkb_stride];
			int el = eg - local_expert_start;
			if (el >= 0 && el < experts_per_pe) {
				atomicMin(&k_starts[el], kb_abs);
				atomicMax(&k_ends[el],   kb_abs + 1);
			}
		}
	}
	cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
}

// ── mlp4 Phase-2 runner ──
//
// Unified over Compute: Compute == 90 (Hopper) drives the adaptive kMSplit
// ternary walk over plain per-CTA cells; Compute == 100 (Blackwell) always
// drives the SM100 ClusterM=2 paired-CTA path (fixed M-split — the paired
// UMMA atom only supports the shared-N/split-M cooperative layout — and
// cell = CTA-PAIR, not CTA). Calls the unified mlp4_producer<Traits4,
// Compute> / mlp4_consumer<Traits4, Compute> (mlp4.cuh), which do their own
// internal Compute == 100 dispatch to the paired-CTA multicast TMA path.
// Reuses the plain Mlp4FusedSmem<Traits4> smem type (see MlpFusedBwdSmem —
// mlp4.cuh's Mlp4FusedSmem is traits-compatible internally, no separate
// Compute-conditional smem needed here).
template <typename Traits1, typename Traits4, int NSplit2, int Compute = 90,
          typename TmaLoadXT4, typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaReduceDB, typename TmaReduceDC>
__device__ __forceinline__ void mlp_bwd_run_phase_2_mlp4(
		Mlp4MainloopPipelineFor<Traits4, Compute>& p4_pipe,
		uint32_t& p4_count,
		Mlp4FusedSmem<Traits4>& smem_mlp4,
		const int* k_starts,
		const int* k_ends,
		TmaLoadXT4   const& tma_load_xt4,
		TmaLoaddUT4  const& tma_load_dut4,
		TmaLoaddVT4  const& tma_load_dvt4,
		TmaReduceDB  const& tma_reduce_db,
		TmaReduceDC  const& tma_reduce_dc,
		int warp_id, int bk_start, int bk_tiles,
		int phase2_div, int phase2_mod,
		const MlpBwdDims& dims,
		uint32_t tmem_base,
		int ring_kb = 0) {
	static_assert(Compute == 90 || Compute == 100,
		"mlp_bwd_run_phase_2_mlp4: Compute must be 90 (Hopper 1SM) or "
		"100 (Blackwell ClusterM=2 paired-CTA) — only Compute == 100 "
		"executes the paired-CTA (ClusterM/NumPairs) code below.");
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;
	auto s = mlp_bwd_resume_state<Mlp4MainloopPipelineFor<Traits4, Compute>>(
		p4_count, warp_id == 0);

	int  experts_per_pe = dims.experts_per_pe;

	// Batch's K-window in mlp4's TileK_4 grain. Producer/consumer
	// intersect this with the global per-expert ranges in `k_starts`/
	// `k_ends` (filled once per pass in the bwd preamble).
	int batch_kb_start = (bk_start * Traits1::TileM) / Traits4::TileK;
	int batch_kb_end   = batch_kb_start
		+ (bk_tiles * Traits1::TileM) / Traits4::TileK;

	// Flat cell index over all Phase 2 CTAs. cell_start = phase2_div·NSplit2
	// + phase2_mod = flat_id ∈ [0,num_blocks) (the div/mod from the caller
	// recombine to the flat id), so coverage is complete and disjoint for
	// ANY num_blocks — no NSplit2 divisibility.
	int flat_id = phase2_div * NSplit2 + phase2_mod;
	// Phase 2 uses ALL launched CTAs (flat grid-stride over cells), so the
	// cell stride is the full launched count gridDim.x·gridDim.y = num_blocks
	// on Compute == 90; Compute == 100 pairs CTAs (ClusterM=2), so the stride
	// is in CTA-PAIRS instead.
	int total_phase2_ctas = (int)gridDim.x * (int)gridDim.y;
	// Size outer_split from the experts LIVE in this batch's K-window, NOT the
	// global expert set. Sizing from experts_per_pe under-splits when only ~1
	// expert is live per batch (large T) — the dead experts inflate the sizing
	// input, so outer_split stays small and the lone live expert's work spreads
	// over too few CTAs. k_starts/k_ends are per-expert ranges in TileK grain.
	int live_experts_4 = 0;
	for (int e = 0; e < experts_per_pe; ++e)
		if (k_starts[e] < batch_kb_end && k_ends[e] > batch_kb_start) live_experts_4++;
	if (live_experts_4 < 1) live_experts_4 = 1;

	int cell_start, cell_stride, outer_split_4, k_split_4;
	if constexpr (Compute == 100) {
		// SM100 ClusterM=2 paired-CTA path: fixed M-split (no kMSplit
		// ternary — the paired-CTA MMA atom only supports the shared-N/
		// split-M cooperative layout), cell = CTA-PAIR.
		int num_clusters = total_phase2_ctas / Traits4::ClusterM;
		int total_chunks = live_experts_4 * dims.num_n_tiles_4;
		int walk_extent   = dims.num_m_tiles_4;
		outer_split_4 = 1;
		{
			int target_cells = 2 * num_clusters;
			while (outer_split_4 * 2 * Traits4::NumPairs <= walk_extent
			       && total_chunks * outer_split_4 < target_cells)
				outer_split_4 *= 2;
			while (walk_extent % outer_split_4 != 0)
				outer_split_4 /= 2;
			if (outer_split_4 < 1) outer_split_4 = 1;
		}
		// K-split (KS): split each cell's K-loop across k_split_4 CTA-pairs
		// when the live cells don't fill the grid. TMA_REDUCE_ADD sums
		// partials.
		k_split_4 = 1;
		{
			int active_cells  = total_chunks * outer_split_4;
			int batch_k_tiles = batch_kb_end - batch_kb_start;
			constexpr int kKMin = 8;
			int max_by_depth = batch_k_tiles / kKMin;
			if (max_by_depth < 1) max_by_depth = 1;
			if (active_cells > 0 && active_cells < num_clusters) {
				k_split_4 = (num_clusters + active_cells - 1) / active_cells;
				if (k_split_4 > max_by_depth) k_split_4 = max_by_depth;
				if (k_split_4 < 1) k_split_4 = 1;
			}
		}
		cell_start  = flat_id / Traits4::ClusterM;
		cell_stride = num_clusters;
	} else {
		// SM90 (Hopper) 1SM path. mlp4 holds the SHARED operand and walks
		// the cooperation (split) axis (see mlp4.cuh):
		//   kMSplit=true  (M-split): chunk = (e, n_tile), lane partitions num_m_tiles_4
		//   kMSplit=false (N-split): chunk = (e, m_tile), lane partitions num_n_tiles_4
		// Adaptive split: largest pow2 divisor of walk_extent so total_cells
		// covers all Phase 2 CTAs.
		int total_chunks_4    = Traits4::kMSplit
			? (live_experts_4 * dims.num_n_tiles_4)
			: (live_experts_4 * dims.num_m_tiles_4);
		int walk_extent_4 = Traits4::kMSplit ? dims.num_m_tiles_4 : dims.num_n_tiles_4;
		outer_split_4 = 1;
		{
			int target_cells = 2 * total_phase2_ctas;
			while (outer_split_4 * 2 <= walk_extent_4
			       && total_chunks_4 * outer_split_4 < target_cells) {
				outer_split_4 *= 2;
			}
			while (walk_extent_4 % outer_split_4 != 0) outer_split_4 /= 2;
			if (outer_split_4 < 1) outer_split_4 = 1;
		}
		// K-split (KS): split each cell's K-loop across k_split_4 CTAs when
		// the live cells don't fill the grid. TMA_REDUCE_ADD sums partials.
		k_split_4 = 1;
		{
			int active_cells  = total_chunks_4 * outer_split_4;
			int batch_k_tiles = batch_kb_end - batch_kb_start;
			constexpr int kKMin = 8;
			int max_by_depth = batch_k_tiles / kKMin;
			if (max_by_depth < 1) max_by_depth = 1;
			if (active_cells > 0 && active_cells < total_phase2_ctas) {
				k_split_4 = (total_phase2_ctas + active_cells - 1) / active_cells;
				if (k_split_4 > max_by_depth) k_split_4 = max_by_depth;
				if (k_split_4 < 1) k_split_4 = 1;
			}
		}
		cell_start  = flat_id;
		cell_stride = total_phase2_ctas;
	}

	if (warp_id == 0)
		mlp4_producer<Traits4, Compute>(
			p4_pipe, s,
			smem_mlp4, tma_load_xt4, tma_load_dut4, tma_load_dvt4,
			k_starts, k_ends, experts_per_pe,
			dims.intermediate_dim, dims.hidden_dim, dims.num_tokens,
			dims.num_m_tiles_4, dims.num_n_tiles_4, outer_split_4,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, k_split_4, ring_kb);
	if constexpr (Compute == 100) smem_mlp4.tmem_base = tmem_base;
	constexpr int kFirstMlp4ConsumerWarp = (Compute == 100) ? 3 : 4;
	if (warp_id >= kFirstMlp4ConsumerWarp)
		mlp4_consumer<Traits4, Compute, Compute == 90>(
			p4_pipe, s,
			smem_mlp4, tma_reduce_db, tma_reduce_dc,
			k_starts, k_ends, experts_per_pe,
			dims.intermediate_dim, dims.hidden_dim, dims.total_m_rows_4,
			dims.num_m_tiles_4, dims.num_n_tiles_4, outer_split_4,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, k_split_4);

	cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	p4_count = s.count();
}

// ── mlp3 Phase-2 runner ──
//
// Unified over Compute: Compute == 90 (Hopper) drives the adaptive kMSplit
// ternary walk over plain per-CTA cells; Compute == 100 (Blackwell) always
// drives the SM100 ClusterM=2 paired-CTA path (fixed M-split — the paired
// UMMA atom only supports the shared-N/split-M cooperative layout — and
// cell = CTA-PAIR, not CTA), and uses the distinct Mlp3FusedSmem2Sm smem
// layout (paired-CTA accumulator pipeline; see MlpFusedBwdSmem). Calls the
// unified mlp3_producer<Traits3, Compute> / mlp3_consumer<Traits3, Compute>
// (mlp3.cuh), which do their own internal Compute == 100 dispatch to the
// paired-CTA multicast TMA path.
template <typename Traits1, typename Traits3, int NSplit2, int Compute = 90,
          typename TmaLoadDYT3, typename TmaLoadZT3, typename TmaReduceDA>
__device__ __forceinline__ void mlp_bwd_run_phase_2_mlp3(
		Mlp3MainloopPipelineFor<Traits3, Compute>& p3_pipe,
		uint32_t& p3_count,
		cute::conditional_t<Compute == 100,
			Mlp3FusedSmem2Sm<Traits3>, Mlp3FusedSmem<Traits3>>& smem_mlp3,
		const int* k_starts,
		const int* k_ends,
		TmaLoadDYT3 const& tma_load_dyt3,
		TmaLoadZT3  const& tma_load_zt3,
		TmaReduceDA const& tma_reduce_da,
		int warp_id, int bk_start, int bk_tiles,
		int phase2_div, int phase2_mod,
		const MlpBwdDims& dims,
		uint32_t tmem_base,
		int ring_kb = 0) {
	static_assert(Compute == 90 || Compute == 100,
		"mlp_bwd_run_phase_2_mlp3: Compute must be 90 (Hopper 1SM) or "
		"100 (Blackwell ClusterM=2 paired-CTA) — only Compute == 100 "
		"executes the paired-CTA (ClusterM/NumPairs) code below.");
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;
	auto s = mlp_bwd_resume_state<Mlp3MainloopPipelineFor<Traits3, Compute>>(
		p3_count, warp_id == 0);

	int  experts_per_pe = dims.experts_per_pe;

	// Batch's K-window in mlp3's TileK_3 grain (see mlp4 helper above).
	int batch_kb_start = (bk_start * Traits1::TileM) / Traits3::TileK;
	int batch_kb_end   = batch_kb_start
		+ (bk_tiles * Traits1::TileM) / Traits3::TileK;

	// Flat cell index over all Phase 2 CTAs. cell_start = phase2_div·NSplit2
	// + phase2_mod = flat_id ∈ [0,num_blocks) (the div/mod from the caller
	// recombine to the flat id), so coverage is complete and disjoint for
	// ANY num_blocks — no NSplit2 divisibility.
	int flat_id = phase2_div * NSplit2 + phase2_mod;
	// All launched CTAs (gridDim.x·gridDim.y = num_blocks) — see mlp4 helper.
	int total_phase2_ctas = (int)gridDim.x * (int)gridDim.y;
	// Size outer_split from the experts LIVE in this batch's K-window (see the
	// mlp4 helper above for the rationale) — not the global experts_per_pe.
	int live_experts_3 = 0;
	for (int e = 0; e < experts_per_pe; ++e)
		if (k_starts[e] < batch_kb_end && k_ends[e] > batch_kb_start) live_experts_3++;
	if (live_experts_3 < 1) live_experts_3 = 1;

	int cell_start, cell_stride, outer_split_3, k_split_3;
	if constexpr (Compute == 100) {
		// SM100 ClusterM=2 paired-CTA path: fixed M-split, cell = CTA-PAIR.
		int total_phase2_pairs = total_phase2_ctas / Traits3::ClusterM;
		int total_chunks = live_experts_3 * dims.num_n_tiles_3;
		outer_split_3 = 1;
		{
			int target_cells = 2 * total_phase2_pairs;
			while (outer_split_3 * 2 * Traits3::NumPairs <= dims.num_m_tiles_3
			       && total_chunks * outer_split_3 < target_cells)
				outer_split_3 *= 2;
			while (dims.num_m_tiles_3 % (outer_split_3 * Traits3::NumPairs) != 0)
				outer_split_3 /= 2;
			outer_split_3 = max(outer_split_3, 1);
		}
		// K-split (KS): split each cell's K-loop across k_split_3 CTA-pairs
		// when the live cells don't fill the grid. TMA_REDUCE_ADD sums
		// partials.
		k_split_3 = 1;
		{
			int active_cells = total_chunks * outer_split_3;
			int batch_k_tiles = batch_kb_end - batch_kb_start;
			constexpr int kKMin = 8;
			int max_by_depth = max(batch_k_tiles / kKMin, 1);
			if (active_cells > 0 && active_cells < total_phase2_pairs) {
				k_split_3 = (total_phase2_pairs + active_cells - 1) / active_cells;
				k_split_3 = min(k_split_3, max_by_depth);
			}
		}
		cell_start  = flat_id / Traits3::ClusterM;
		cell_stride = total_phase2_pairs;
	} else {
		// SM90 (Hopper) 1SM path. mlp3 holds the SHARED operand and walks
		// the cooperation (split) axis:
		//   kMSplit=true  : chunk = (e, n_tile), lane partitions num_m_tiles
		//   kMSplit=false : chunk = (e, m_tile), lane partitions num_n_tiles
		// Adaptive split: pick largest pow2 divisor of walk_extent so
		// total_cells covers all Phase 2 CTAs.
		int total_chunks_3    = Traits3::kMSplit
			? (live_experts_3 * dims.num_n_tiles_3)
			: (live_experts_3 * dims.num_m_tiles_3);
		int walk_extent_3 = Traits3::kMSplit ? dims.num_m_tiles_3 : dims.num_n_tiles_3;
		outer_split_3 = 1;
		{
			int target_cells = 2 * total_phase2_ctas;
			while (outer_split_3 * 2 <= walk_extent_3
			       && total_chunks_3 * outer_split_3 < target_cells) {
				outer_split_3 *= 2;
			}
			while (walk_extent_3 % outer_split_3 != 0) outer_split_3 /= 2;
			if (outer_split_3 < 1) outer_split_3 = 1;
		}
		// K-split (KS): when the live cells (total_chunks_3 × outer_split_3)
		// don't fill the grid, split each cell's K-loop across k_split_3
		// CTAs and let the TMA_REDUCE_ADD epilogue sum the partials.
		// Bounded so each slice keeps >= kKMin K-tiles (else the Stages-deep
		// pipeline can't amortize its fill).
		k_split_3 = 1;
		{
			int active_cells  = total_chunks_3 * outer_split_3;
			int batch_k_tiles = batch_kb_end - batch_kb_start;
			constexpr int kKMin = 8;
			int max_by_depth = batch_k_tiles / kKMin;
			if (max_by_depth < 1) max_by_depth = 1;
			if (active_cells > 0 && active_cells < total_phase2_ctas) {
				k_split_3 = (total_phase2_ctas + active_cells - 1) / active_cells;
				if (k_split_3 > max_by_depth) k_split_3 = max_by_depth;
				if (k_split_3 < 1) k_split_3 = 1;
			}
		}
		cell_start  = flat_id;
		cell_stride = total_phase2_ctas;
	}

	if (warp_id == 0)
		mlp3_producer<Traits3, Compute>(
			p3_pipe, s,
			smem_mlp3, tma_load_dyt3, tma_load_zt3,
			k_starts, k_ends, experts_per_pe,
			dims.hidden_dim, dims.intermediate_dim, dims.num_tokens,
			dims.num_m_tiles_3, dims.num_n_tiles_3, outer_split_3,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, k_split_3, ring_kb);
	if constexpr (Compute == 100) smem_mlp3.tmem_base = tmem_base;
	constexpr int kFirstMlp3ConsumerWarp = (Compute == 100) ? 3 : 4;
	if (warp_id >= kFirstMlp3ConsumerWarp)
		mlp3_consumer<Traits3, Compute, Compute == 90>(
			p3_pipe, s,
			smem_mlp3, tma_reduce_da,
			k_starts, k_ends, experts_per_pe,
			dims.hidden_dim, dims.intermediate_dim, dims.total_n_rows_3,
			dims.num_m_tiles_3, dims.num_n_tiles_3, outer_split_3,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, k_split_3);

	cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	p3_count = s.count();
}

// ═══════════════════════════════════════════════════════════════════
// mlp_fused_bwd_run_batches — batch loop body extracted
// ═══════════════════════════════════════════════════════════════════
//
// Inner per-pass batch loop. Caller (mlp_fused_bwd or mlp_fused_bwd_dual)
// owns pipeline construction + the persistent per-phase counts. This lets
// the dual entry share ONE set of pipelines / counts across the local
// and remote passes — they sit at the dual-function's stack scope and
// PTXAS can analyze both batch loops as sequential code in one CFG,
// reusing spill slots between them.
//
// Templated on TileIter / TMA types / EfkbStride{3,4} — gets inlined into
// the caller's body. When called from mlp_fused_bwd_dual twice, the two
// inlined copies appear as two adjacent code regions in the same outer
// function (vs two separate inlined instances of the entire mlp_fused_bwd,
// which is the current 2-call shape).

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5,
          int NSplit2,
          int Compute = 90,
          typename TileIter,
          typename TmaLoadX, typename TmaLoadW1,
          typename TmaStoreZ, typename TmaStoreDU, typename TmaStoreDV,
          typename TmaLoadDY, typename TmaLoadW2T,
          typename TmaStoreDZ,
          typename TmaLoadDU, typename TmaLoadDV,
          typename TmaLoadB, typename TmaLoadC, typename TmaStoreDX,
          typename TmaLoadDYT3, typename TmaLoadZT3, typename TmaReduceDA,
          typename TmaLoadXT4, typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaReduceDB, typename TmaReduceDC>
__device__ __forceinline__ void mlp_fused_bwd_run_batches(
		MlpFusedBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute>& smem,
		Mlp1MainloopPipelineFor<Traits1, Compute>& pa_pipe,
		Mlp2TMainloopPipelineFor<Traits2T, Compute>& pb_pipe,
		Mlp5MainloopPipelineFor<Traits5, Compute>&  pd_pipe,
		Mlp3MainloopPipelineFor<Traits3, Compute>& p3_pipe,
		Mlp4MainloopPipelineFor<Traits4, Compute>& p4_pipe,
		uint32_t& pa_count, uint32_t& pb_count, uint32_t& pd_count,
		uint32_t& p3_count, uint32_t& p4_count,
		TileIter& iter,
		TmaLoadX const& tma_load_x, TmaLoadW1 const& tma_load_b_fwd,
		TmaLoadW1 const& tma_load_c_fwd, TmaStoreZ const& tma_store_z,
		TmaStoreDU const& tma_store_du, TmaStoreDV const& tma_store_dv,
		TmaLoadDY const& tma_load_dy, TmaLoadW2T const& tma_load_a_col,
		TmaStoreDZ const& tma_store_dz,
		TmaLoadDU const& tma_load_du, TmaLoadDV const& tma_load_dv,
		TmaLoadB const& tma_load_b_col, TmaLoadC const& tma_load_c_col,
		TmaStoreDX const& tma_store_dx,
		TmaLoadDYT3 const& tma_load_dyt3, TmaLoadZT3 const& tma_load_zt3,
		TmaReduceDA const& tma_reduce_da,
		TmaLoadXT4 const& tma_load_xt4,
		TmaLoaddUT4 const& tma_load_dut4, TmaLoaddVT4 const& tma_load_dvt4,
		TmaReduceDB const& tma_reduce_db, TmaReduceDC const& tma_reduce_dc,
		int efkb_stride_3, int efkb_stride_4,
		const MlpBwdDims& dims,
		const MlpBwdBufs<typename Traits1::Element>& bufs,
		MlpBwdCtaBarrierT<Compute>& global_barrier,
		MlpBwdCtaBarrierT<Compute>& x_barrier,
		int warp_id) {

	using Element = typename Traits1::Element;
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;

	// Phase 2 per-expert K-range scan is done PER BATCH over the current
	// batch's K-window (below), NOT once per pass. mlp3 and mlp4 share TileK,
	// so a single per-batch scan (over the X-pipe expert-id array) feeds both.
	//
	// A once-per-pass global scan over [0, num_tokens/TileK) is only correct
	// when each physical slot maps to a SINGLE expert for the whole pass —
	// true for the local pass (fully-materialized buffer) and for a remote
	// ring that never wraps (global_total ≤ L). When the remote staging ring
	// wraps to a 2nd+ lap (global_total > L = MC·CommNumStages), a physical
	// slot is REUSED for a different expert each lap, so a global min/max
	// range mis-attributes the reused slots — corrupting the experts whose
	// tiles straddle the lap-reuse boundary (the ring's first/last experts).
	// Scanning the LIVE tile_expert_ids per batch (which holds this batch's
	// residency, sync-guaranteed by Phase 1's acquire) attributes every slot
	// correctly regardless of lap. This restores the pre-#102 per-batch scan
	// semantics under #102's cooperative cell-walk mlp3/mlp4.
	static_assert(Traits3::TileK == Traits4::TileK,
		"Phase 2 shared k_starts/ends require Traits3::TileK == Traits4::TileK");
	// The Compute == 100 ClusterM=2 paired-CTA Phase-2 path issues tcgen05
	// 2x1SM UMMA and cross-CTA cluster-scope mbarriers that only exist on
	// SM100. Compute is the SOLE selector (see mlp_bwd_run_phase_2_mlp3 /
	// mlp_bwd_run_phase_2_mlp4 above) — catch a bogus Compute value here at
	// compile time rather than as a cryptic pipeline/SharedStorage type
	// mismatch further down.
	static_assert(Compute == 90 || Compute == 100,
		"mlp_fused_bwd_run_batches: Compute must be 90 (SM90/Hopper, 1SM "
		"Phase-2) or 100 (SM100/Blackwell, ClusterM=2 paired-CTA Phase-2).");

	for (int batch = 0; batch < dims.num_batches; ++batch) {

		int saved_m = -1, saved_expert = -1;
		bool has_tile = false;

		// Phase 1: recompute + silu coefs, dZ via dY @ A, silu_bwd
		if (iter.has_next()) {
			cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);

			iter.acquire_src();
			auto tile = iter.next();
			int expert = tile.expert - dims.local_expert_start;
			int m = tile.y_m;

			saved_m = m;
			saved_expert = expert;
			has_tile = true;

			mlp_bwd_run_phase_1a<Traits1, Compute>(
				pa_pipe, pa_count, smem.mlp1,
				tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
				tma_store_du, tma_store_dv, tma_store_z,
				warp_id, m, expert, dims, smem.tmem_base);
			cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);

			iter.acquire_dy();
			mlp_bwd_run_phase_1b<Traits2T, Compute>(
				pb_pipe, pb_count, smem.mlp2t,
				tma_load_dy, tma_load_a_col, tma_store_dz,
				warp_id, m, expert, dims, smem.tmem_base);
			x_barrier.wait();
			x_barrier.wait();

			silu_bwd_pair_tile<Element>(
				bufs.dz_buf, bufs.du_buf, bufs.dv_buf,
				bufs.du_buf, bufs.dv_buf,
				m, Traits1::TileM, Traits2T::TileN, dims.intermediate_dim,
				blockIdx.y, gridDim.y);
			cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
		}

		// Global barrier: Z, dU, dV all visible
		global_barrier.wait();

		// Phase 1d: dX = dU@B + dV@C
		if (has_tile) {
			iter.acquire_dst();
			mlp_bwd_run_phase_1d<Traits5, Compute>(
				pd_pipe, pd_count, smem.mlp5,
				tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col,
				tma_store_dx,
				warp_id, saved_m, saved_expert, dims, smem.tmem_base);
			__threadfence();  // system-scope dX flush for the NIC put is done by release_dst
			cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
			iter.release_dst(threadIdx.x);
		}

		// Phase 2: weight grads — mlp4 first, then mlp3
		int phase2_flat_cta = (int)blockIdx.x + (int)blockIdx.y * (int)gridDim.x;
		int phase2_div      = phase2_flat_cta / NSplit2;
		int phase2_mod      = phase2_flat_cta % NSplit2;

		// Per-batch K-window = the tiles Phase 1 delivered THIS batch. Phase 1
		// advances each of grid_x columns by one tile per batch, so batch b
		// delivers global tiles [b·grid_x, (b+1)·grid_x). Phase 2 must reduce
		// exactly those — striding by grid_x (= n_gemm/NSplit), NOT gridDim.x.
		// Under the flat 1-D launch gridDim.x = num_blocks (= num_sms) ≠ grid_x,
		// so using gridDim.x here over-strides and over-spans: in the remote
		// pass it reads slots from other laps of the circular staging ring
		// (mis-attributed tile_expert_ids → corrupt dA/dB/dC at the ring-wrap
		// boundary experts). The local pass tolerated it (fully-materialized
		// non-circular buffer, harmless no-op tail batches), which is why 1-PE
		// passed and multi-PE failed. num_batches is already sized ceil(total /
		// grid_x), so grid_x is the matching stride.
		int bk_start_unwrapped = batch * dims.grid_x;
		int bk_tiles = min(dims.grid_x,
		                   dims.total_m_tiles_in_pass - bk_start_unwrapped);
		int bk_start = bk_start_unwrapped % dims.num_m_tiles;

		// Per-batch per-expert K-range scan over THIS batch's window of the
		// (possibly wrapped) ring. Reads the LIVE X-pipe expert-id array
		// (tile_expert_ids_x in the remote pass; release_src is deferred past
		// mlp4, so it stays valid here AND for mlp3 since the ranges are cached
		// into k_starts/ends before the scan's barrier). bk_start is already
		// wrapped mod num_m_tiles, so the window indexes physical ring slots —
		// matching the kb coordinates the mlp3/mlp4 producers feed to TMA.
		if (bk_tiles > 0) {
			int batch_kb_off = (bk_start * Traits1::TileM) / Traits4::TileK;
			int batch_kb_cnt = (bk_tiles * Traits1::TileM) / Traits4::TileK;
			mlp_bwd_phase_2_build_ranges<Compute>(
				bufs.expert_k_starts, bufs.expert_k_ends,
				bufs.expert_for_k_block_mlp4,
				batch_kb_off, batch_kb_cnt,
				dims.experts_per_pe, dims.local_expert_start,
				efkb_stride_4, warp_id);
		}

		// Cross-CTA barrier after the per-batch scan: k_starts/k_ends is shared
		// by all CTAs (each runs build_ranges = reset + atomicMin/Max over the
		// same window), so mlp4 must not read it until every CTA's scan is done,
		// else producer/consumer tile-count desync → pipeline deadlock.
		if (bk_tiles > 0)
			global_barrier.wait();

		if (bk_tiles > 0) {
			mlp_bwd_run_phase_2_mlp4<
				Traits1, Traits4, NSplit2, Compute>(
				p4_pipe, p4_count, smem.mlp4,
				bufs.expert_k_starts, bufs.expert_k_ends,
				tma_load_xt4, tma_load_dut4, tma_load_dvt4,
				tma_reduce_db, tma_reduce_dc,
				warp_id, bk_start, bk_tiles,
				phase2_div, phase2_mod,
				dims, smem.tmem_base);
		}

		global_barrier.wait();
		if (has_tile)
			iter.release_src(threadIdx.x);

		if (bk_tiles > 0) {
			mlp_bwd_run_phase_2_mlp3<
				Traits1, Traits3, NSplit2, Compute>(
				p3_pipe, p3_count, smem.mlp3,
				bufs.expert_k_starts, bufs.expert_k_ends,
				tma_load_dyt3, tma_load_zt3, tma_reduce_da,
				warp_id, bk_start, bk_tiles,
				phase2_div, phase2_mod,
				dims, smem.tmem_base);
		}

		global_barrier.wait();
		if (has_tile)
			iter.release_dy(threadIdx.x);
	}
}

// ═══════════════════════════════════════════════════════════════════
// mlp_fused_bwd — pipeline construction + batch loop (single pass)
// ═══════════════════════════════════════════════════════════════════
//
// Generic over TileIter. Called once for local, once for remote.
// Mirrors forward mlp_fused_fwd: prologue → tile/batch loop.

// Per-phase N-split parameters (autotuning surface).
// Phase 1's effective NSplits (1a, 1b', 1c, 1d) are tied to gridDim.y by
// construction: those phases iterate n-tiles via blockIdx.y/gridDim.y. To
// retune them, change the launch grid shape at the host level.
// Phase 2 (mlp3, mlp4) uses NSplit2 with virtual (vx, vy) remapping — it
// can differ from gridDim.y as long as gridDim.x * gridDim.y is divisible
// by NSplit2.
template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5,
          int NSplit2 = 4,            // Phase 2 (mlp3/mlp4) virtual n-split
          // Phase 2 expert-id array stride: divisor applied to the K-block
          // index before reading expert_for_k_block_mlp{3,4}.
          //   Local pass:  array built per-K-block by build_expert_for_k_block
          //                → stride = 1 (the divide folds away).
          //   Remote pass: array aliases comm's tile_expert_ids (one entry per
          //                Traits1::TileM comm tile) → stride =
          //                Traits1::TileM / Traits{3,4}::TileK. Without this,
          //                TM=128+TK=64 would read two K-blocks of slot N as
          //                belonging to experts N and N+1 (mis-attribution).
          int EfkbStride3 = 1,
          int EfkbStride4 = 1,
          int Compute = 90,
          typename TileIter,
          // Phase 1 TMA descriptors
          typename TmaLoadX, typename TmaLoadW1,
          typename TmaStoreZ, typename TmaStoreDU, typename TmaStoreDV,
          typename TmaLoadDY, typename TmaLoadW2T,
          typename TmaStoreDZ,
          typename TmaLoadDU, typename TmaLoadDV,
          typename TmaLoadB, typename TmaLoadC, typename TmaStoreDX,
          // Phase 2 TMA descriptors (reduce-add variant: no prev-loads)
          typename TmaLoadDYT3, typename TmaLoadZT3, typename TmaReduceDA,
          typename TmaLoadXT4, typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaReduceDB, typename TmaReduceDC>
__device__ __forceinline__ void mlp_fused_bwd(
		MlpFusedBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute>& smem,
		TileIter& iter,
		// Phase 1 TMA
		TmaLoadX const& tma_load_x, TmaLoadW1 const& tma_load_b_fwd,
		TmaLoadW1 const& tma_load_c_fwd, TmaStoreZ const& tma_store_z,
		TmaStoreDU const& tma_store_du, TmaStoreDV const& tma_store_dv,
		TmaLoadDY const& tma_load_dy, TmaLoadW2T const& tma_load_a_col,
		TmaStoreDZ const& tma_store_dz,
		TmaLoadDU const& tma_load_du, TmaLoadDV const& tma_load_dv,
		TmaLoadB const& tma_load_b_col, TmaLoadC const& tma_load_c_col,
		TmaStoreDX const& tma_store_dx,
		// Phase 2 TMA
		TmaLoadDYT3 const& tma_load_dyt3, TmaLoadZT3 const& tma_load_zt3,
		TmaReduceDA const& tma_reduce_da,
		TmaLoadXT4 const& tma_load_xt4,
		TmaLoaddUT4 const& tma_load_dut4, TmaLoaddVT4 const& tma_load_dvt4,
		TmaReduceDB const& tma_reduce_db, TmaReduceDC const& tma_reduce_dc,
		// Dims + buffers
		const MlpBwdDims& dims,
		const MlpBwdBufs<typename Traits1::Element>& bufs,
		// Target-based barriers — constructed ONCE by the caller in
		// moe_fused_bwd and threaded through here so target counters
		// accumulate monotonically across local/cross-phase/remote calls.
		MlpBwdCtaBarrierT<Compute>& global_barrier,
		MlpBwdCtaBarrierT<Compute>& x_barrier) {

	using Element = typename Traits1::Element;
	int warp_id = threadIdx.x / Traits1::WarpSize;
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;
	// See mlp_fused_bwd_run_batches — Compute is the SOLE selector of the
	// Compute == 100 ClusterM=2 paired-CTA Phase-2 path; the paired-CTA
	// UMMA + cluster mbarrier rendezvous below is SM100-only.
	static_assert(Compute == 90 || Compute == 100,
		"mlp_fused_bwd: Compute must be 90 (SM90/Hopper, 1SM Phase-2) or "
		"100 (SM100/Blackwell, ClusterM=2 paired-CTA Phase-2).");

	// ── Prefetch TMA descriptors ──
	cute::prefetch_tma_descriptor(tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_b_fwd.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c_fwd.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_du.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dv.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dy.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_a_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dz.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_du.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dv.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_b_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dx.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dyt3.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_zt3.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_da.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_xt4.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dut4.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dvt4.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_db.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_dc.get_tma_descriptor());

	// ── Prologue: construct pipelines ──
	// Pipeline constructor requires __syncthreads (all 384 threads).
	// Must happen before comm warps exit. All five phases (1a/1b/1d,
	// 2-mlp3, 2-mlp4) use a single cooperative fused pipe each; phase
	// 1b additionally uses a tiny scale-load aux pipe (Stages=1) to
	// carry the silu-bwd coefficients U'/V' produced by mlp1.
	auto pa_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp1_make_pipe_umma<Traits1>(smem.pa_pipe);
		else
			return mlp1_make_pipe<Traits1>(smem.pa_pipe);
	}();
	auto pb_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp2_t_make_pipe_umma<Traits2T>(smem.pb_pipe);
		else
			return mlp2_t_make_pipe<Traits2T>(smem.pb_pipe);
	}();
	auto pd_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp5_make_pipe_umma<Traits5>(smem.pd_pipe);
		else
			return mlp5_make_pipe<Traits5>(smem.pd_pipe);
	}();

	// Single fused pipe for mlp3 carrying dYT + Z per acquire. On SM100,
	// warp 3 issues UMMA and epilogue WGs 4-11 drain the same output tile.
	// Compute == 100 always drives the SM100 ClusterM=2 paired-CTA pipe
	// (mlp3_make_pipe_umma_2sm); Compute == 90 drives the existing Hopper
	// pipe (mlp3_make_pipe).
	auto p3_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp3_make_pipe_umma_2sm<Traits3>(smem.p3_pipe);
		else
			return mlp3_make_pipe<Traits3>(smem.p3_pipe);
	}();
	// Single fused pipe for mlp4 carrying X + dU^T + dV^T per acquire. On
	// SM100, warp 3 issues the two UMMAs and epilogue WGs 4-11 reduce-add
	// dB/dC. Compute == 100 always drives the SM100 ClusterM=2 paired-CTA
	// pipe (mlp4_make_pipe_umma_2sm); Compute == 90 drives the existing
	// Hopper pipe (mlp4_make_pipe).
	auto p4_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp4_make_pipe_umma_2sm<Traits4>(smem.p4_pipe);
		else
			return mlp4_make_pipe<Traits4>(smem.p4_pipe);
	}();

	// Comm warps exit. Hopper keeps the historical MLP partition (warps 0,4-11)
	// and drops warp 3 immediately. Blackwell mirrors the forward path: comm
	// uses only warps 1-2 and warp 3 joins MLP as the dedicated UMMA producer.
	if constexpr (Compute == 90) {
		if (warp_id == 3) return;
	}
	if (warp_id == 1 || warp_id == 2) return;

	cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	cute::conditional_t<
		Compute == 100,
		cute::TMEM::Allocator2Sm,
		cute::TMEM::Allocator1Sm> tmem_alloc{};
	if constexpr (Compute == 100) {
		constexpr int kTmemColumns =
			MlpBwdTmemColumns<Traits1, Traits2T, Traits3, Traits4, Traits5>::value;
		if (warp_id == 3) {
			tmem_alloc.allocate(kTmemColumns, &smem.tmem_base);
			if constexpr (Compute == 100) {
				if (cute::elect_one_sync())
					smem.tmem_pair_barrier.init(
						Traits1::WarpSize);
			}
			__syncwarp();
		}
		cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	}

	// Keep phase-role tests at their call sites. Hoisting producer/MMA/epilogue
	// booleans across every batch keeps them live through all phase mainloops,
	// increasing register pressure in a kernel that already spills.

	// ── Persistent per-phase pipeline counts ─────────────────────
	//
	// We persist only the monotonic iteration count per phase (1 uint32_t)
	// rather than the full PipelineState (3 ints: index_, phase_, count_).
	// Each phase function reconstructs the live state on entry by calling
	// `make_producer_start_state().advance(count)` (or the consumer
	// equivalent) — this is correct because `advance(N)` deterministically
	// computes the (index, phase) tuple that ++state would arrive at after
	// N iterations, and that tuple matches the in-smem barrier phase the
	// previous invocation left behind.
	//
	// PipelineState itself must NOT be persistent across phase invocations
	// if we want construction inside the phase — but persisting only the
	// count (5 regs across 5 phases vs 15 regs for full states) trims the
	// cross-phase live set while keeping correctness. We hit a desync bug
	// on T4096_D4096_I14336_E16_K4 (dA off 10% on expert 7) when state was
	// reset fresh per batch without preserving the count.
	uint32_t pa_count = 0;
	uint32_t pb_count = 0;
	uint32_t pd_count = 0;
	uint32_t p3_count = 0;
	uint32_t p4_count = 0;

	mlp_fused_bwd_run_batches<
		Traits1, Traits2T, Traits3, Traits4, Traits5,
		NSplit2, Compute>(
		smem, pa_pipe, pb_pipe, pd_pipe, p3_pipe, p4_pipe,
		pa_count, pb_count, pd_count, p3_count, p4_count,
		iter,
		tma_load_x, tma_load_b_fwd, tma_load_c_fwd,
		tma_store_z, tma_store_du, tma_store_dv,
		tma_load_dy, tma_load_a_col, tma_store_dz,
		tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col, tma_store_dx,
		tma_load_dyt3, tma_load_zt3, tma_reduce_da,
		tma_load_xt4, tma_load_dut4, tma_load_dvt4, tma_reduce_db, tma_reduce_dc,
		EfkbStride3, EfkbStride4,
		dims, bufs, global_barrier, x_barrier, warp_id);

	if constexpr (Compute == 100) {
		constexpr int kTmemColumns =
			MlpBwdTmemColumns<Traits1, Traits2T, Traits3, Traits4, Traits5>::value;
		cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
		if (warp_id == 3) {
			tmem_alloc.release_allocation_lock();
			if constexpr (Compute == 100) {
				uint32_t rank = cute::block_rank_in_cluster();
				uint32_t peer = rank ^ 1;
				bool leader = (rank % 2) == 0;
				smem.tmem_pair_barrier.arrive(peer, !leader);
				smem.tmem_pair_barrier.wait(0);
				smem.tmem_pair_barrier.arrive(peer, leader);
			}
			tmem_alloc.free(smem.tmem_base, kTmemColumns);
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// mlp_fused_bwd_dual — single-instance local + remote passes
// ═══════════════════════════════════════════════════════════════════
//
// Builds pipelines + per-phase counts ONCE, runs the local batch loop,
// crosses the global barrier, and runs the remote batch loop in the
// SAME function body. The aim is to let PTXAS analyze both batch loops
// as sequential code in one CFG, reusing spill slots between them
// (lifetimes don't overlap — separated by global_barrier.wait()). This
// is the alternative to the current pattern in moe_fused_bwd of two
// inlined `mlp_fused_bwd` calls, each constructing its own pipelines
// and re-initializing the smem barriers.
//
// Pipeline barriers in smem are constructed ONCE here. The persistent
// `*_count` state is shared across passes and consumed by the count→
// PipelineState resume logic inside each phase helper (see
// mlp_bwd_resume_state) — this matches the smem barriers' phase
// counters that advance across both passes.
template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5,
          int NSplit2, int SubBatch,
          // SubTiles = CommTileM / GemmTileM ∈ {1,2}. >1 → Phase-1 (mlp1a/mlp2t/
          // silu/mlp5) steps SubTiles GemmTileM-row sub-tiles per 128 comm slot;
          // Phase-2 sees num_m_tiles·SubTiles GemmTileM token-tiles with
          // tile_expert_ids replicated SubTiles× (the K-block count is invariant).
          // 1 → unchanged single-tile path.
          int SubTiles = 1,
          int Compute = 90,
          typename FusedIter,
          // Shared TMA (Phase 1 weights, Phase 1d B/C, Phase 2 reduce + Z/U/V)
          typename TmaLoadW1, typename TmaStoreZ, typename TmaStoreDU,
          typename TmaStoreDV, typename TmaLoadW2T, typename TmaStoreDZ,
          typename TmaLoadDU, typename TmaLoadDV,
          typename TmaLoadB, typename TmaLoadC,
          typename TmaLoadZT3, typename TmaReduceDA,
          typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaReduceDB, typename TmaReduceDC,
          // Per-role TMA types — local and remote instances share the
          // same C++ type (same cute Layout: dynamic shape, _1 col stride,
          // identical smem layout). Only the runtime gmem pointer differs.
          typename TmaLoadX, typename TmaLoadDY, typename TmaStoreDX,
          typename TmaLoadDYT3, typename TmaLoadXT4>
__device__ __forceinline__ void mlp_fused_bwd_dual(
		MlpFusedBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute>& smem,
		FusedIter& iter,
		bool remote_active,
		int efkb_stride_4_remote,
		// Phase 1 TMA — weights are shared across tiles; X/dY/dX (the *_remote
		// instances below) come from the comm staging ring, where EVERY tile,
		// local or remote, is staged.
		TmaLoadW1 const& tma_load_b_fwd, TmaLoadW1 const& tma_load_c_fwd,
		TmaStoreZ const& tma_store_z,
		TmaStoreDU const& tma_store_du, TmaStoreDV const& tma_store_dv,
		TmaLoadW2T const& tma_load_a_col, TmaStoreDZ const& tma_store_dz,
		TmaLoadDU const& tma_load_du, TmaLoadDV const& tma_load_dv,
		TmaLoadB const& tma_load_b_col, TmaLoadC const& tma_load_c_col,
		// Phase 2 TMA (shared)
		TmaLoadZT3 const& tma_load_zt3, TmaReduceDA const& tma_reduce_da,
		TmaLoaddUT4 const& tma_load_dut4, TmaLoaddVT4 const& tma_load_dvt4,
		TmaReduceDB const& tma_reduce_db, TmaReduceDC const& tma_reduce_dc,
		// Staging-ring data TMAs: X/dY/dX (Phase 1) + dYᵀ/Xᵀ (Phase 2).
		TmaLoadX const& tma_load_x_remote,
		TmaLoadDY const& tma_load_dy_remote,
		TmaStoreDX const& tma_store_dx_remote,
		TmaLoadDYT3 const& tma_load_dyt3_remote,
		TmaLoadXT4 const& tma_load_xt4_remote,
		const MlpBwdDims& remote_dims,
		const MlpBwdBufs<typename Traits1::Element>& remote_bufs,
		MlpBwdCtaBarrierT<Compute>& global_barrier,
		MlpBwdCtaBarrierT<Compute>& x_barrier,
		// Flat 1-D launch coordinates (computed once in moe_fused_bwd):
		//   flat_id     = blockIdx.x
		//   col         = flat_id / NSplit       (Phase 1 logical column)
		//   grid_x      = n_gemm / NSplit        (Phase 1 column count)
		//   split       = flat_id % NSplit       (N-split index)
		//   num_splits  = NSplit
		//   gemm_active = flat_id < n_gemm  → this CTA runs Phase 1 + comm;
		//                 gap CTAs skip Phase 1 but still run Phase 2 + barriers.
		int flat_id, int col, int grid_x, int split, int num_splits,
		bool gemm_active) {

	static_assert(SubBatch >= 1 && SubBatch <= kBwdSubBatchMax,
		"SubBatch (MoeBwdConfig::kSubBatch) must be in [1, kBwdSubBatchMax]");
	// See mlp_fused_bwd_run_batches — Compute is the SOLE selector of the
	// Compute == 100 ClusterM=2 paired-CTA Phase-2 path; the paired-CTA
	// UMMA + cluster mbarrier rendezvous below is SM100-only.
	static_assert(Compute == 90 || Compute == 100,
		"mlp_fused_bwd_dual: Compute must be 90 (SM90/Hopper, 1SM Phase-2) "
		"or 100 (SM100/Blackwell, ClusterM=2 paired-CTA Phase-2).");

	int warp_id = threadIdx.x / Traits1::WarpSize;
	constexpr int kMlpBwdBarrierThreads = kMlpBwdBarrierThreadsFor<Compute>;

	// Prefetch the TMA descriptors this pass uses: shared weights/intermediates
	// + the staging-ring data descriptors.
	cute::prefetch_tma_descriptor(tma_load_b_fwd.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c_fwd.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_du.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dv.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_a_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dz.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_du.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dv.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_b_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c_col.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_zt3.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_da.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dut4.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dvt4.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_db.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_dc.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_x_remote.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dy_remote.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_dx_remote.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dyt3_remote.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_xt4_remote.get_tma_descriptor());

	// Build pipelines ONCE (smem barriers initialized in one place — the
	// remote pass picks up smem-barrier phase from where local left off).
	auto pa_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp1_make_pipe_umma<Traits1>(smem.pa_pipe);
		else
			return mlp1_make_pipe<Traits1>(smem.pa_pipe);
	}();
	auto pb_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp2_t_make_pipe_umma<Traits2T>(smem.pb_pipe);
		else
			return mlp2_t_make_pipe<Traits2T>(smem.pb_pipe);
	}();
	auto pd_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp5_make_pipe_umma<Traits5>(smem.pd_pipe);
		else
			return mlp5_make_pipe<Traits5>(smem.pd_pipe);
	}();

	// Single fused pipe for mlp3 carrying dYT + Z per acquire. Compute ==
	// 100 always drives the SM100 ClusterM=2 paired-CTA pipe
	// (mlp3_make_pipe_umma_2sm); Compute == 90 drives the existing Hopper
	// pipe (mlp3_make_pipe).
	auto p3_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp3_make_pipe_umma_2sm<Traits3>(smem.p3_pipe);
		else
			return mlp3_make_pipe<Traits3>(smem.p3_pipe);
	}();
	// Single fused pipe for mlp4 carrying X + dU^T + dV^T per acquire.
	// Compute == 100 always drives the SM100 ClusterM=2 paired-CTA pipe
	// (mlp4_make_pipe_umma_2sm); Compute == 90 drives the existing Hopper
	// pipe (mlp4_make_pipe).
	auto p4_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp4_make_pipe_umma_2sm<Traits4>(smem.p4_pipe);
		else
			return mlp4_make_pipe<Traits4>(smem.p4_pipe);
	}();

	if constexpr (Compute == 90) {
		if (warp_id == 3) return;
	}
	if (warp_id == 1 || warp_id == 2) return;

	cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	cute::conditional_t<
		Compute == 100,
		cute::TMEM::Allocator2Sm,
		cute::TMEM::Allocator1Sm> tmem_alloc{};
	if constexpr (Compute == 100) {
		constexpr int kTmemColumns =
			MlpBwdTmemColumns<Traits1, Traits2T, Traits3, Traits4, Traits5>::value;
		if (warp_id == 3) {
			tmem_alloc.allocate(kTmemColumns, &smem.tmem_base);
			if constexpr (Compute == 100) {
				if (cute::elect_one_sync())
					smem.tmem_pair_barrier.init(
						Traits1::WarpSize);
			}
			__syncwarp();
		}
		cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
	}

	// Counts persist across both passes — smem barriers continue advancing
	// from local into remote, and PipelineState::advance(count) re-derives
	// the right (index, phase) for the resume.
	uint32_t pa_count = 0;
	uint32_t pb_count = 0;
	uint32_t pd_count = 0;
	uint32_t p3_count = 0;
	uint32_t p4_count = 0;

	using Element = typename Traits1::Element;

	// Phase 2 flat-cell coords are CTA-invariant — hoist out of the loops.
	int phase2_div = flat_id / NSplit2;
	int phase2_mod = flat_id % NSplit2;

	// Shared k_starts/k_ends require equal TileK (one scan feeds mlp3+mlp4).
	static_assert(Traits3::TileK == Traits4::TileK,
		"Phase 2 shared k_starts/ends require Traits3::TileK == Traits4::TileK");

	// Nothing to do if this PE has no tiles (remote_active is grid-uniform, so
	// every CTA agrees and the early-out keeps the barriers below well-defined).
	if (!remote_active) {
		if constexpr (Compute == 100) {
			constexpr int kTmemColumns =
				MlpBwdTmemColumns<Traits1, Traits2T, Traits3, Traits4, Traits5>::value;
			cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
			if (warp_id == 3) {
				tmem_alloc.release_allocation_lock();
				if constexpr (Compute == 100) {
					uint32_t rank =
						cute::block_rank_in_cluster();
					uint32_t peer = rank ^ 1;
					bool leader = (rank % 2) == 0;
					smem.tmem_pair_barrier.arrive(
						peer, !leader);
					smem.tmem_pair_barrier.wait(0);
					smem.tmem_pair_barrier.arrive(
						peer, leader);
				}
				tmem_alloc.free(smem.tmem_base, kTmemColumns);
			}
		}
		return;
	}

	// One cross-CTA sync so all MLP CTAs enter the grouped loop together:
	// SM90 synchronizes w0+w4-11 (288 threads); SM100 synchronizes
	// w0+w3-11 (320 threads). Comm warps 1-2 have already returned.
	global_barrier.wait();

	// ═══════════════════════════════════════════════════════════════
	// REMOTE GROUPED PASS — kBwdSubBatch Phase-1 sub-batches, then ONE
	// Phase-2 (mlp4+mlp3) over their combined kBwdSubBatch·grid_x K-window.
	// ═══════════════════════════════════════════════════════════════
	// Phase 1 (mlp1/2/5) runs in SMALL per-sub-batch chunks (less CTA
	// barrier-stall); Phase 2 (weight grads dA/dB/dC) runs ONCE over the whole
	// group so its cooperative cell-walk has a long K-reduction to fill the grid
	// (deeper outer_split / k_split). Each sub-batch keeps its X/dY staging slot
	// un-released until the grouped Phase 2 has read X^T (mlp4) and dY^T (mlp3);
	// the saved slots are freed in a batch after each Phase-2 sub-pass. The host
	// guarantees CommNumStages is a multiple of kBwdSubBatch and ≥ it, so the
	// group window never straddles the ring wrap (per-expert k-ranges stay
	// contiguous) and comm has prefetch headroom. kBwdSubBatch == 1 reduces this
	// to the original per-batch loop.
	const int remote_batches = remote_dims.num_batches;

	// Adaptive group size: a group of `g` sub-batches holds g·grid_x live staging
	// slots that must map to DISTINCT ring slots, so g·grid_x ≤ ring. Cap the
	// group by how many grid_x-windows the ring holds — deep rings get the full
	// SubBatch; shallow rings degrade to fewer (down to 1 = original per-batch
	// loop). Grid-uniform (num_m_tiles, grid_x are host-set), so every CTA agrees.
	const int eff_subbatch =
		max(1, min((int)SubBatch, remote_dims.num_m_tiles / grid_x));

	// Phase-1 GEMM consumers + Phase-2 see the token axis in GemmTileM
	// (=Traits1::TileM) tiles: num_m_tiles (Phase-1 buffer height) and num_tokens
	// (M extent) scale by SubTiles = CommTileM/GemmTileM. The comm loop control
	// (eff_subbatch, group_start, bk_tiles clamp, slot releases) stays in
	// 128-comm-slot units via remote_dims. At SubTiles=1 the two are identical.
	// (Other dims fields — N/K extents, expert counts — are M-axis-independent.)
	MlpBwdDims gemm_dims = remote_dims;
	gemm_dims.num_m_tiles *= SubTiles;
	gemm_dims.num_tokens  *= SubTiles;

	for (int g0 = 0; g0 < remote_batches; g0 += eff_subbatch) {
		const int gcount = min(eff_subbatch, remote_batches - g0);

		// Per-sub-batch staging slots, freed together after Phase 2. cur_slot in
		// the iterator only tracks the LAST acquire, so releasing the earlier
		// sub-batches needs their saved slot (release indexes *_consumed[slot]).
		int  saved_slot[SubBatch];
		bool saved_live[SubBatch];

		// ── Phase 1: gcount sub-batches (mlp1a / mlp2t / silu_bwd / mlp5) ──
		for (int s = 0; s < gcount; ++s) {
			int  saved_m = -1, saved_expert = -1;
			bool has_tile = false;

			// Gated by gemm_active — gap CTAs (flat_id ≥ n_gemm) skip Phase 1 and
			// the per-column x_barrier, but still hit every global barrier below.
			if (gemm_active && iter.has_next()) {
				cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				iter.acquire_src();
				iter.acquire_src();
				auto tile = iter.next();
				int expert = tile.expert - remote_dims.local_expert_start;
				int m = tile.y_m;

				saved_m = m;
				saved_expert = expert;
				has_tile = true;

				// One 128-row comm slot feeds SubTiles GemmTileM-row sub-tiles.
				// gemm coordinate of sub-tile `sub` = m·SubTiles + sub (the
				// Phase-1 kernels multiply by Traits1::TileM=GemmTileM → rows
				// m·128 + sub·64). gemm_dims carries the SubTiles-scaled
				// num_m_tiles / num_tokens so the buffer tensor shapes match.
				// Acquire/release of the comm slot stays once per 128 tile.
				for (int sub = 0; sub < SubTiles; ++sub) {
					mlp_bwd_run_phase_1a<Traits1, Compute>(
						pa_pipe, pa_count, smem.mlp1,
						tma_load_x_remote, tma_load_b_fwd, tma_load_c_fwd,
						tma_store_du, tma_store_dv, tma_store_z,
						warp_id, m * SubTiles + sub, expert, gemm_dims, smem.tmem_base, split, num_splits);
					if (sub + 1 < SubTiles)
						cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				}
				cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);

				iter.acquire_dy();
				for (int sub = 0; sub < SubTiles; ++sub) {
					mlp_bwd_run_phase_1b<Traits2T, Compute>(
						pb_pipe, pb_count, smem.mlp2t,
						tma_load_dy_remote, tma_load_a_col, tma_store_dz,
						warp_id, m * SubTiles + sub, expert, gemm_dims, smem.tmem_base, split, num_splits);
					if (sub + 1 < SubTiles)
						cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				}

				x_barrier.wait();

				for (int sub = 0; sub < SubTiles; ++sub) {
					silu_bwd_pair_tile<Element>(
						remote_bufs.dz_buf, remote_bufs.du_buf, remote_bufs.dv_buf,
						remote_bufs.du_buf, remote_bufs.dv_buf,
						m * SubTiles + sub, Traits1::TileM, Traits2T::TileN,
						remote_dims.intermediate_dim, split, num_splits);
					if (sub + 1 < SubTiles)
						cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				}
				cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
			}

			// Save this sub-batch's staging slot (== tile.y_m == iter cur_slot)
			// for the deferred grouped release. -1 / false when this CTA had no
			// tile (gap CTA or column exhausted).
			saved_slot[s] = saved_m;
			saved_live[s] = has_tile;

			// Global barrier: this sub-batch's Z, dU, dV visible to all CTAs.
			global_barrier.wait();

			// Phase 1d: dX = dU@B + dV@C (per sub-batch — Phase 2 never reads dX,
			// so it is emitted and its dst slot released immediately).
			if (has_tile) {
				iter.acquire_dst();
				for (int sub = 0; sub < SubTiles; ++sub) {
					mlp_bwd_run_phase_1d<Traits5, Compute>(
						pd_pipe, pd_count, smem.mlp5,
						tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col,
						tma_store_dx_remote,
						warp_id, saved_m * SubTiles + sub, saved_expert, gemm_dims,
						smem.tmem_base, split, num_splits);
					if (sub + 1 < SubTiles)
						cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				}
				__threadfence();  // dX flush for the NIC put (release_dst signals)
				cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
				iter.release_dst(threadIdx.x);
			}
		}

		// ── Phase 2: ONE pass over the group's gcount·grid_x K-window ──
		// Phase 1 advances each of grid_x columns by one tile per sub-batch, so
		// the group delivered global tiles [g0·grid_x, (g0+gcount)·grid_x).
		// bk_start is UNWRAPPED (group_start) so the group's K-window stays
		// contiguous even when it straddles the staging-ring wrap; build_ranges
		// and the producers wrap the physical slot by % ring_kb internally
		// (ring_kb > 0 enables it). The group's gcount·grid_x live slots are
		// guaranteed distinct (host check: ring > SubBatch·grid_x).
		// group_start / bk_tiles_comm are in 128-comm-slot units (the iterator's
		// tile granularity). Phase-2 works in GemmTileM tiles, so scale by
		// SubTiles: the K-window ratio (× Traits1::TileM/TileK) then yields the
		// same K-block count as the old CommTileM×comm-slot product. ring_kb uses
		// gemm_dims.num_m_tiles (already ×SubTiles). SubTiles=1 → unchanged.
		int group_start   = g0 * grid_x;
		int bk_tiles_comm = min(gcount * grid_x,
		                        remote_dims.total_m_tiles_in_pass - group_start);
		int bk_start = group_start   * SubTiles;
		int bk_tiles = bk_tiles_comm * SubTiles;
		int ring_kb  = (gemm_dims.num_m_tiles * Traits1::TileM) / Traits4::TileK;

		// Per-group per-expert K-range scan over the group's window. Reads the
		// LIVE X-pipe expert ids — none of the group's slots have been released
		// yet, so all gcount·grid_x are valid. Feeds mlp3+mlp4 (unwrapped ranges).
		if (bk_tiles > 0) {
			int batch_kb_off = (bk_start * Traits1::TileM) / Traits4::TileK;
			int batch_kb_cnt = (bk_tiles * Traits1::TileM) / Traits4::TileK;
			mlp_bwd_phase_2_build_ranges<Compute>(
				remote_bufs.expert_k_starts, remote_bufs.expert_k_ends,
				remote_bufs.expert_for_k_block_mlp4,
				batch_kb_off, batch_kb_cnt,
				remote_dims.experts_per_pe, remote_dims.local_expert_start,
				efkb_stride_4_remote, warp_id, ring_kb);
			// Cross-CTA barrier: k_starts/k_ends is shared by all CTAs (each
			// resets + atomicMin/Max), so no producer may read it mid-reset.
			global_barrier.wait();
		}

		if (bk_tiles > 0) {
			mlp_bwd_run_phase_2_mlp4<
				Traits1, Traits4, NSplit2, Compute>(
				p4_pipe, p4_count, smem.mlp4,
				remote_bufs.expert_k_starts, remote_bufs.expert_k_ends,
				tma_load_xt4_remote, tma_load_dut4, tma_load_dvt4,
				tma_reduce_db, tma_reduce_dc,
				warp_id, bk_start, bk_tiles,
				phase2_div, phase2_mod,
				gemm_dims, smem.tmem_base, ring_kb);
		}

		global_barrier.wait();
		// mlp4 done reading X^T — free every X slot the group held.
		{
			__threadfence();
			for (int s = 0; s < gcount; ++s)
				if (saved_live[s]) iter.release_src_slot(saved_slot[s], threadIdx.x);
		}

		if (bk_tiles > 0) {
			mlp_bwd_run_phase_2_mlp3<
				Traits1, Traits3, NSplit2, Compute>(
				p3_pipe, p3_count, smem.mlp3,
				remote_bufs.expert_k_starts, remote_bufs.expert_k_ends,
				tma_load_dyt3_remote, tma_load_zt3, tma_reduce_da,
				warp_id, bk_start, bk_tiles,
				phase2_div, phase2_mod,
				gemm_dims, smem.tmem_base, ring_kb);
		}

		global_barrier.wait();
		// mlp3 done reading dY^T — free every dY slot the group held.
		{
			__threadfence();
			for (int s = 0; s < gcount; ++s)
				if (saved_live[s]) iter.release_dy_slot(saved_slot[s], threadIdx.x);
		}
	}

	if constexpr (Compute == 100) {
		constexpr int kTmemColumns =
			MlpBwdTmemColumns<Traits1, Traits2T, Traits3, Traits4, Traits5>::value;
		cutlass::arch::NamedBarrier::sync(kMlpBwdBarrierThreads, kMlpBarrierId);
		if (warp_id == 3) {
			tmem_alloc.release_allocation_lock();
			if constexpr (Compute == 100) {
				uint32_t rank = cute::block_rank_in_cluster();
				uint32_t peer = rank ^ 1;
				bool leader = (rank % 2) == 0;
				smem.tmem_pair_barrier.arrive(peer, !leader);
				smem.tmem_pair_barrier.wait(0);
				smem.tmem_pair_barrier.arrive(peer, leader);
			}
			tmem_alloc.free(smem.tmem_base, kTmemColumns);
		}
	}
}


} // namespace liger
