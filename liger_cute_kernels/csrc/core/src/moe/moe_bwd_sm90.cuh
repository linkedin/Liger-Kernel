#pragma once

// ═══════════════════════════════════════════════════════════════════
// Fused MoE Backward: MLP compute for local + remote tiles
// ═══════════════════════════════════════════════════════════════════
//
// MLP-only device function — called by non-comm warps. In BWD comm uses
// warps 1-2; warp 3 enters the MLP path on SM100 and exits immediately on SM90.
//
// Execution flow (UNIFIED — local tile loop removed, mirrors moe_fused_fwd):
//   EVERY tile (local or remote) flows through the comm get→staging→MLP→put
//   path, driven by mlp_fused_bwd_dual over the RemoteMlpTileIteratorBwd /
//   staging pipeline (there is no separate materialized local pass).
//
// mlp_fused_bwd_dual runs the grouped batch loop:
//   SubBatch × (Phase 1 tile) → Phase 2 weight grads over the grouped window
//
// ═══════════════════════════════════════════════════════════════════

#include "mlp_bwd_sm90.cuh"
#include "tile_iterator_bwd_sm90.cuh"

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// MoE backward shared memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5, int Compute = 90>
struct MoeBwdSmem {
	MlpFusedBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute> mlp;
	CommSmem comm;
	MlpBwdDims runtime_dims;
	MlpBwdDims gemm_dims;
	MlpBwdBufs<typename Traits1::Element> runtime_bufs;
	RemoteMlpTileIteratorBwdSharedState iterator_state;
};

// ═══════════════════════════════════════════════════════════════════
// moe_fused_bwd — local phase → barrier → remote phase
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5,
          int CommNumStages, int NSplit2, int SubBatch,
          // SubTiles = CommTileM / GemmTileM. >1 → Phase-1 (mlp1a/mlp2t/
          // silu/mlp5) steps through SubTiles GemmTileM-row sub-tiles per
          // comm staging slot; the mlp3/4 ratios use Traits1::TileM (= GemmTileM).
          int SubTiles,
          int Compute = 90,
          int WarpGroupRole = 0,
          typename FusedIter,
          typename TmaBundle>
__device__ __forceinline__ void moe_fused_bwd(
		MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute>& smem,
		const TmaBundle& tma,
		// Comm pipe pointers (for building remote iter)
		int* x_src_ready, int* x_src_consumed,
		int* dy_src_ready, int* dy_src_consumed,
		int* dst_ready, int* dst_consumed,
		// Two per-pipe expert-id arrays (see CommBuffersBwd docstring).
		// remote_tile_expert_ids_x is the X-pipe array used for the iter's
		// tile.expert lookup (and Phase 2 mlp4 in this remote run).
		// remote_tile_expert_ids_dy is the dY-pipe array used for Phase 2
		// mlp3 — gated by release_dy, so it stays stable across mlp3 even
		// when release_src has already let comm refill the X-pipe array.
		const int* remote_tile_expert_ids_x,
		const int* remote_tile_expert_ids_dy,
		const int* remote_tile_valid_rows_x,
		// Runtime-adjusted dimensions and buffers.
		const MlpBwdDims& remote_dims,
		const MlpBwdDims& gemm_dims,
		const MlpBwdBufs<typename Traits1::Element>& remote_bufs,
		int col,
		int grid_x,
		int split,
		int runtime_nsplit,
		bool gemm_active) {

	// Target-based barriers — constructed ONCE so the per-CTA target
	// counters accumulate monotonically across:
	//   local pass → global_barrier.wait() → remote pass
	// Constructing a fresh barrier per wait() resets target=0 and is racy.
	// Flat 1-D launch coordinates. blockIdx.x = flat_id ∈ [0, num_blocks).
	//   global_barrier counts ALL launched CTAs (num_blocks = gridDim.x).
	//   x_barrier is per logical Phase-1 column (col), stride runtime_nsplit.
	int flat_id     = (int)blockIdx.x;

	MlpBwdCtaBarrierT<Compute> global_barrier(
		remote_bufs.barrier_counter,
		(int)gridDim.x * (int)gridDim.y);
	MlpBwdCtaBarrierT<Compute> x_barrier(
		&remote_dims.phase_counter[col],
		runtime_nsplit);

	FusedIter iter;
	// Initialize the fused iter's remote sub-iter from comm pipe pointers.
	// (Local sub-iter was init'd by the caller in moe_bwd_kernel.) Comm warps
	// filled staging during this CTA's local pass. Ticket-based iterator
	// keys on absolute slot index, so pointer offsets are NOT pre-applied
	// (mirrors RemoteMlpTileIterator's init in moe.cuh).
#if defined(LIGER_CUTE_SM90_NONRDC_SPLIT)
	iter.init_remote(
		&smem.iterator_state,
		/*is_leader=*/(threadIdx.x == 0));
#else
	iter.init_remote(
		remote_tile_expert_ids_x,
		remote_tile_valid_rows_x,
		smem.comm.per_cta_tiles,
		col,
		remote_dims.n_gemm,
		x_src_ready,  x_src_consumed,
		dy_src_ready, dy_src_consumed,
		dst_ready,    dst_consumed,
		/*is_leader=*/(threadIdx.x == 0),
		runtime_nsplit,
		/*tma_enabled=*/(remote_dims.tma_get_enabled != 0));
#endif

	// Grid-uniform activity predicate. The unified path stages local tiles even
	// at world size one, so every valid non-empty launch is active.
	bool remote_active =
		(remote_dims.num_pes > 0) && (remote_dims.num_tokens > 0);

	// The remote staging ring (L = MC·CommNumStages tiles, MC = n_gemm/NC) is a
	// STREAMING buffer: when global_total > L it wraps and physical slots are
	// reused across laps. This is correct — mlp_fused_bwd_dual rescans the live
	// per-slot expert ids PER BATCH (see its loop), so reused slots are always
	// attributed to the expert currently resident, and each slot's reduction is
	// committed before comm refills it. No L ≥ global_total requirement.

	// UNIFIED PATH: there is no separate materialized local MLP pass. The shared
	// TileIterator enumerates LOCAL experts (p=0) alongside remote ones; the comm
	// get warps stage local X/dY from local symmetric memory and the put warps
	// drain local dX back to local dx_sorted, so EVERY tile (local or remote)
	// flows through the RemoteMlpTileIteratorBwd / staging pipeline that
	// mlp_fused_bwd_dual drives. Only the staging-ring (remote) TMAs and dims/bufs
	// are passed; the weight/intermediate descriptors are shared across all tiles.
	// efkb_stride maps a (GemmTileM-granular) Phase-2 K-block to its parent
	// COMM-tile slot in tile_expert_ids (which the comm warps emit once per slot).
	// = CommTileM/TileK = Traits1::TileM·SubTiles/Traits4::TileK. The integer
	// division floors each GEMM sub-tile to its parent communication slot's expert (the
	// interpolation). SubTiles=1 → Traits1::TileM/TileK, unchanged.
	constexpr int kEfkbStride4Remote = Traits1::TileM * SubTiles / Traits4::TileK;
	mlp_fused_bwd_dual<
		Traits1, Traits2T, Traits3, Traits4, Traits5,
		NSplit2, SubBatch, SubTiles, Compute, WarpGroupRole>(
		smem.mlp, iter, remote_active,
		kEfkbStride4Remote,
		tma,
		remote_dims, gemm_dims, remote_bufs,
		global_barrier, x_barrier,
		flat_id, col, grid_x, split, /*num_splits=*/runtime_nsplit,
		gemm_active, nullptr);
}

} // namespace liger
