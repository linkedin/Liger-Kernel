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

#include "mlp_bwd.cuh"

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
};

// ═══════════════════════════════════════════════════════════════════
// moe_fused_bwd — local phase → barrier → remote phase
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2T, typename Traits3,
          typename Traits4, typename Traits5,
          int CommNumStages, int NSplit2, int SubBatch,
          // SubTiles = CommTileM / GemmTileM ∈ {1,2}. >1 → Phase-1 (mlp1a/mlp2t/
          // silu/mlp5) steps through SubTiles GemmTileM-row sub-tiles per 128-wide
          // comm staging slot; the mlp3/4 ratios use Traits1::TileM (= GemmTileM).
          int SubTiles,
          int Compute = 90,
          typename FusedIter,
          // Phase 1 TMA
          typename TmaLoadX, typename TmaLoadW1,
          typename TmaStoreZ, typename TmaStoreU, typename TmaStoreV,
          typename TmaLoadDY, typename TmaLoadW2T, typename TmaStoreDZ,
          typename TmaLoadDU, typename TmaLoadDV,
          typename TmaLoadB, typename TmaLoadC, typename TmaStoreDX,
          // Phase 2 TMA
          typename TmaLoadDYT3, typename TmaLoadZT3, typename TmaStoredA,
          typename TmaLoadXT4, typename TmaLoaddUT4, typename TmaLoaddVT4,
          typename TmaStoredB, typename TmaStoredC,
          // Remote TMA: SAME C++ types as their local counterparts
          // (the cute Layouts for both x_sorted and the staging ring use
          // dynamic shape + _1 col stride + identical smem layout), so we
          // can reuse the TmaLoadX/DY/StoreDX/etc. template params. Only
          // the runtime gmem pointer differs between local and remote.
          typename TmaLoadXR = TmaLoadX,
          typename TmaLoadDYR = TmaLoadDY,
          typename TmaStoreDXR = TmaStoreDX,
          typename TmaLoadDYT3R = TmaLoadDYT3,
          typename TmaLoadXT4R = TmaLoadXT4>
__device__ __forceinline__ void moe_fused_bwd(
		MoeBwdSmem<Traits1, Traits2T, Traits3, Traits4, Traits5, Compute>& smem,
		FusedIter& iter,
		// Phase 1 TMA (local)
		TmaLoadX const& tma_load_x, TmaLoadW1 const& tma_load_b_fwd,
		TmaLoadW1 const& tma_load_c_fwd, TmaStoreZ const& tma_store_z,
		TmaStoreU const& tma_store_u, TmaStoreV const& tma_store_v,
		TmaLoadDY const& tma_load_dy, TmaLoadW2T const& tma_load_a_col,
		TmaStoreDZ const& tma_store_dz,
		TmaLoadDU const& tma_load_du, TmaLoadDV const& tma_load_dv,
		TmaLoadB const& tma_load_b_col, TmaLoadC const& tma_load_c_col,
		TmaStoreDX const& tma_store_dx,
		// Phase 2 TMA
		TmaLoadDYT3 const& tma_load_dyt3, TmaLoadZT3 const& tma_load_zt3,
		TmaStoredA const& tma_store_da,
		TmaLoadXT4 const& tma_load_xt4,
		TmaLoaddUT4 const& tma_load_dut4, TmaLoaddVT4 const& tma_load_dvt4,
		TmaStoredB const& tma_store_db, TmaStoredC const& tma_store_dc,
		// Remote TMA
		TmaLoadXR const& tma_load_x_remote,
		TmaLoadDYR const& tma_load_dy_remote,
		TmaStoreDXR const& tma_store_dx_remote,
		// Remote Phase 2 TMA: transposed staging sources for dA (Phase 3)
		// and dB/dC (Phase 4). Without these, the remote phase reuses the
		// local-only x_sorted/dy_sorted descriptors and produces wrong
		// gradients for tokens routed across PEs.
		TmaLoadDYT3R const& tma_load_dyt3_remote,
		TmaLoadXT4R  const& tma_load_xt4_remote,
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
		// Dims + buffers
		const MlpBwdDims& dims,
		const MlpBwdBufs<typename Traits1::Element>& bufs,
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

	MlpBwdCtaBarrierT<Compute> global_barrier(bufs.barrier_counter,
		(int)gridDim.x * (int)gridDim.y);
	MlpBwdCtaBarrierT<Compute> x_barrier(&dims.phase_counter[col],
		runtime_nsplit);

	// Initialize the fused iter's remote sub-iter from comm pipe pointers.
	// (Local sub-iter was init'd by the caller in moe_bwd_kernel.) Comm warps
	// filled staging during this CTA's local pass. Ticket-based iterator
	// keys on absolute slot index, so pointer offsets are NOT pre-applied
	// (mirrors RemoteMlpTileIterator's init in moe.cuh).
	iter.init_remote(
		remote_tile_expert_ids_x,
		smem.comm.per_cta_tiles,
		col,
		dims.n_gemm,
		x_src_ready,  x_src_consumed,
		dy_src_ready, dy_src_consumed,
		dst_ready,    dst_consumed,
		/*is_leader=*/(threadIdx.x == 0),
		runtime_nsplit,
			/*tma_enabled=*/(dims.tma_get_enabled != 0));

	// Remote dims override: num_tokens / num_m_tiles use the staging-
	// ring shape; total_m_tiles_in_pass uses the actual remote tile
	// count (allows batches ≥ NumStages to run with bk_start wrapped
	// into the live ring window). local_expert_start = 0 because
	// RemoteIter reads tile_expert_ids written by nvshmem_comm_main_bwd
	// (= TileIterator's LOCAL expert index).
	MlpBwdDims remote_dims = dims;
	remote_dims.num_tokens             = dims.remote_num_tokens;
	remote_dims.num_m_tiles            = dims.remote_num_m_tiles;
	remote_dims.total_m_tiles_in_pass  = smem.comm.global_total;
	remote_dims.num_batches            =
		(smem.comm.global_total + grid_x - 1) / grid_x;
	remote_dims.local_expert_start     = 0;

	// Alias per-pipe expert-id arrays into MlpBwdBufs. Phase 2 mlp4
	// reads X-pipe gated array (stable across mlp4 because release_src
	// is deferred past mlp4). Phase 2 mlp3 reads dY-pipe gated array.
	MlpBwdBufs<typename Traits1::Element> remote_bufs = bufs;
	remote_bufs.expert_for_k_block_mlp4 = remote_tile_expert_ids_x;
	remote_bufs.expert_for_k_block_mlp3 = remote_tile_expert_ids_dy;

	// Volatile load: written by warp 2 of this CTA in nvshmem_comm_prologue_bwd
	// then made visible via __syncthreads() in moe_bwd_kernel. Without volatile
	// the compiler may hoist above the syncthreads, returning a stale 0 →
	// multi-PE deadlock (this CTA skips remote while peers wait at the
	// remote-pass mlp_global_barrier).
	bool remote_active = (*reinterpret_cast<int volatile*>(&smem.comm.global_total) != 0);

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
	// COMM-tile slot in tile_expert_ids (which the comm warps emit one-per-128).
	// = CommTileM/TileK = Traits1::TileM·SubTiles/Traits4::TileK. The integer
	// division floors each gemm sub-tile to its parent 128-slot's expert (the
	// interpolation). SubTiles=1 → Traits1::TileM/TileK, unchanged.
	constexpr int kEfkbStride4Remote = Traits1::TileM * SubTiles / Traits4::TileK;
	mlp_fused_bwd_dual<
		Traits1, Traits2T, Traits3, Traits4, Traits5, NSplit2, SubBatch, SubTiles, Compute>(
		smem.mlp, iter, remote_active,
		kEfkbStride4Remote,
		// Phase 1 shared weights/intermediates
		tma_load_b_fwd, tma_load_c_fwd,
		tma_store_z, tma_store_u, tma_store_v,
		tma_load_a_col, tma_store_dz,
		tma_load_du, tma_load_dv, tma_load_b_col, tma_load_c_col,
		// Phase 2 shared
		tma_load_zt3, tma_store_da,
		tma_load_dut4, tma_load_dvt4, tma_store_db, tma_store_dc,
		// Staging-ring data TMAs (X/dY/dX, dYᵀ/Xᵀ)
		tma_load_x_remote, tma_load_dy_remote, tma_store_dx_remote,
		tma_load_dyt3_remote, tma_load_xt4_remote,
		remote_dims, remote_bufs,
		global_barrier, x_barrier,
		flat_id, col, grid_x, split, /*num_splits=*/runtime_nsplit, gemm_active);
}

} // namespace liger
