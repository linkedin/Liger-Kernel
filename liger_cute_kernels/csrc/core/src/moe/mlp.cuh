#pragma once

// ═══════════════════════════════════════════════════════════════════
// Fused MLP:  Y = (Z @ A^T)  where  Z = SiLU(B @ X) * (C @ X)
// ═══════════════════════════════════════════════════════════════════
//
// Pipelines are constructed once in the prologue (requires all threads).
// Pipeline states are persistent across tiles — cooperative 2-WG +
// fused pipes reduce the state register footprint to two states per
// phase (producer + consumer), so hoisting them out of the tile loop
// still fits comfortably in registers.
//
// The tile loop is driven by a TileIterator (local or remote).
// The intermediate Z buffer is double-buffered: z_m alternates
// between blockIdx.x*ZBufferSlots + (iter % ZBufferSlots).
//
// WG0 warps 1-3 do NOT participate — reserved for NVSHMEM comm.
//
// ═══════════════════════════════════════════════════════════════════

#include "mlp1_fused.cuh"
#include "mlp2_fused.cuh"
#include "tile_iterator.cuh"
#include "cta_barrier.cuh"   // target-based CtaCounterBarrier

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Fused shared memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2, int Compute = 90>
struct MlpFusedSmem {
	union {
		Mlp1FusedSmem<Traits1> mlp1;
		Mlp2FusedSmem<Traits2> mlp2;
	};

	// Pipeline barrier storage — persistent across phases.
	// Both phases use a single fused pipe (one mbarrier per stage).
	typename Mlp1MainloopPipelineFor<Traits1, Compute>::SharedStorage p1_pipe;
	typename Mlp2MainloopPipelineFor<Traits2, Compute>::SharedStorage p2_pipe;
	alignas(16) uint32_t tmem_base;
};

static constexpr int kMlpBarrierId      = 14;

// Target-based monotonic barrier for the FWD MLP warps. Count is Compute-aware:
// Hopper 288 (w0 + w4-11); Blackwell 320 (w0 + w3-11, warp 3 = MLP1 UMMA). Must
// match the in-fn kMlpBarrierThreads — both fire kMlpBarrierId. See cta_barrier.cuh
// for why the modulo variant deadlocks under CTA skew.
template <int Compute>
using MlpFwdCtaBarrierT = CtaCounterBarrier<(Compute == 100 ? 10 : 9) * 32, kMlpBarrierId>;

// ═══════════════════════════════════════════════════════════════════
// MlpDims — invariant MLP dimensions, passed as __grid_constant__
// ═══════════════════════════════════════════════════════════════════

struct MlpDims {
	int hidden_dim;
	int intermediate_dim;
	int num_experts;         // local experts represented by the weight tensors
	int total_n_rows_1;     // physical rows spanned by the strided B/C views
	int total_n_rows_2;     // experts_per_pe * hidden_dim
	int expert_n_stride_1;  // B/C expert stride in TileN1 units
	int num_n_tiles_1;      // intermediate_dim / TileN1
	int num_n_tiles_2;      // ceildiv(hidden_dim/TileN2, NSub) — per-WG outer steps
	int num_k_tiles_1;      // hidden_dim / TileK1
	int num_k_tiles_2;      // intermediate_dim / TileK2
	int y_buf_m_tiles;      // max_total_slots / TileM — row-tile extent of the
	                        // symmetric local_output (y_buf) descriptor, used as
	                        // num_y_m_tiles for the direct-to-peer Y store.
	int* phase_counter;
};

// ═══════════════════════════════════════════════════════════════════
// mlp_fused_fwd — generic over TileIterator
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2,
          int ZBufferSlots = 2,
          // SubTiles = CommTileM/GemmTileM ∈ {1,2}. When 2, each 128-wide comm
          // tile (one acquire_src/release_src/acquire_dst/release_dst) is
          // consumed as 2 GemmTileM(=64)-row sub-tiles: mlp1 runs both sub-tiles
          // (into 2 Z regions), then mlp2 runs both. The comm semaphores stay
          // 128-wide; only the TMA + wgmma granularity is finer. 1 → unchanged.
          int SubTiles = 1,
          int Compute = 90,
          typename TileIter,
          typename TmaLoadX, typename TmaLoadW1, typename TmaStoreZ,
          typename TmaLoadZ, typename TmaLoadW2, typename TmaStoreY>
__device__ __forceinline__ void mlp_fused_fwd(
		MlpFusedSmem<Traits1, Traits2, Compute>& smem,
		TileIter& iter,
		// MLP1 TMA descriptors
		TmaLoadX  const& tma_load_x,
		TmaLoadW1 const& tma_load_b,
		TmaLoadW1 const& tma_load_c,
		TmaStoreZ const& tma_store_z,
		// MLP2 TMA descriptors
		TmaLoadZ  const& tma_load_z,
		TmaLoadW2 const& tma_load_a,
		TmaStoreY const& tma_store_y,
		// Invariant MLP dims — const ref into __grid_constant__ .param space
		const MlpDims& dims,
		// Per-phase params (change between local and remote)
		int num_tokens,
		int num_m_tiles,
		// Persistent per-CTA x_barrier — constructed once in moe_fused_fwd
		// and threaded through here so target accumulates across all 3
		// mlp_fused_fwd calls (A1 / B / A2). Standalone callers (no
		// remote phase) still pass their own barrier instance.
		MlpFwdCtaBarrierT<Compute> & x_barrier,
		int local_expert_start = 0,
		// Logical column / grid_x for the flat-grid fused launch. Defaults
		// (-1) fall back to blockIdx.x / gridDim.x for standalone 2-D-grid
		// callers (mlp.cu bench kernel).
		int col = -1,
		int grid_x = -1,
		// N-split identity for the GEMM producers/consumers. Defaults (-1)
		// fall back to blockIdx.y / gridDim.y (standalone 2-D-grid mlp.cu).
		int split_idx = -1,
		int num_splits = -1,
		// Direct-to-peer Y store: array of per-peer TMA descriptors over
		// nvshmem_ptr(local_output, peer) (indexed by MlpTileInfo::peer_rank).
		// For is_local tiles MLP2 stores Y straight into the destination's
		// symmetric local_output instead of dst_staging (tma_store_y). nullptr →
		// staging path only (standalone mlp.cu callers have no peer descriptors).
		const TmaStoreY* y_peer_desc = nullptr) {

	using Pipeline1 = Mlp1MainloopPipelineFor<Traits1, Compute>;
	using PipeState1 = typename Traits1::PipelineState;
	using Pipeline2 = Mlp2MainloopPipelineFor<Traits2, Compute>;
	using PipeState2 = typename Traits2::PipelineState;

	int warp_id = threadIdx.x / Traits1::WarpSize;

	// MLP-warp NamedBarrier arrival count — must equal MlpFwdCtaBarrierT<Compute>'s
	// template count (both fire kMlpBarrierId). Hopper: w0+w4-11 (288). Blackwell:
	// w0+w3-11 (320, warp 3 = MLP1 UMMA).
	constexpr int kMlpBarrierThreads = (Compute == 100 ? 10 : 9) * 32;

	cute::prefetch_tma_descriptor(tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_y.get_tma_descriptor());

	// ── Prologue: construct all pipelines once ──────────
	// Pipeline constructors set per-thread role (Producer / Consumer /
	// NonParticipant) so warps 1-3 can safely skip the MLP body below.
	auto p1_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp1_make_pipe_umma<Traits1>(smem.p1_pipe);
		else
			return mlp1_make_pipe<Traits1>(smem.p1_pipe);
	}();
	auto p2_pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp2_make_pipe_umma<Traits2>(smem.p2_pipe);
		else
			return mlp2_make_pipe<Traits2>(smem.p2_pipe);
	}();

	if constexpr (Compute == 90) {
		if (warp_id == 3) {
			return;
		}
	} 

	// Sync MLP warps only (excludes comm warps 1-3 which may be in
	// nvshmem_comm_main when called from moe_fused_fwd).
	cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);

	cute::TMEM::Allocator1Sm tmem_alloc{};
	if constexpr (Compute == 100) {
		constexpr int kMlp1Cols = Traits1::AccStages * (2 * Traits1::TileN);
		constexpr int kMlp2Cols = Traits2::AccStages * Traits2::TileN;
		constexpr int kTmemColumns = (kMlp1Cols > kMlp2Cols)
			? kMlp1Cols : kMlp2Cols;
		if (warp_id == 3) {
			tmem_alloc.allocate(kTmemColumns, &smem.tmem_base);
			__syncwarp();
		}
		cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
	}

	// Z buffer: [grid_x, ZBufferSlots, SubTiles, GemmTileM, intermediate_dim].
	// Indexed in GemmTileM-row tiles: SubTiles Z regions per comm-tile slot so
	// mlp1 can stage both sub-tiles before mlp2 drains them. z_rows is unchanged
	// (grid_x·ZBuf·128 == grid_x·ZBuf·SubTiles·GemmTileM), only the indexing splits.
	int z_col    = (col    >= 0) ? col    : (int)blockIdx.x;
	int z_grid_x = (grid_x >= 0) ? grid_x : (int)gridDim.x;
	int num_z_m_tiles = z_grid_x * ZBufferSlots * SubTiles;
	int z_base = z_col * ZBufferSlots * SubTiles;
	// N-split identity for the GEMM producers/consumers (flat-grid: NOT
	// blockIdx.y / gridDim.y, which are 0 / 1). Each CTA computes only its
	// stripe of N-tiles (n = n_split, n_split+n_count, ...).
	int n_split = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_count = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	int tile_iter = 0;

	// ── Persistent pipeline states (hoisted out of the tile loop) ──
	// With single fused pipes + cooperative consumer, we have only
	// 2 states per phase. They persist across tiles since k-tile
	// counts are fixed — the barrier phase bit cycles back to 0
	// after each tile's k-loop completes.
	PipeState1 p1_state = (warp_id == 0) ?
		cutlass::make_producer_start_state<Pipeline1>() : PipeState1{};
	PipeState1 p1_cons_state;
	PipeState2 p2_state = (warp_id == 0) ?
		cutlass::make_producer_start_state<Pipeline2>() : PipeState2{};
	PipeState2 p2_cons_state;

	// x_barrier is passed in by reference (constructed once per CTA
	// in moe_fused_fwd) so target advances monotonically across
	// the A1 / B / A2 phases that all share dims.phase_counter.

	// ── Tile loop ───────────────────────────────────────
	while (iter.has_next()) {
		// Barrier: ensure previous phase 2 is done before reusing smem union.
		cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
		iter.acquire_src();
		auto tile = iter.next();

		int expert = tile.expert - local_expert_start;
		// Base Z region for this comm tile. SubTiles GemmTileM-row regions per
		// slot, indexed z_m_base + sub. tile.y_m is the comm slot index in
		// CommTileM(=GemmTileM·SubTiles) units; the GemmTileM-unit row coordinate
		// of sub-tile `sub` into the 128-wide slot is tile.y_m·SubTiles + sub.
		int z_m_base = z_base + (tile_iter % ZBufferSlots) * SubTiles;

		// Phase 1: for each GemmTileM sub-tile, Z[z_m_base+sub] = SiLU(B@X)·(C@X).
		// One comm semaphore (acquire_src above) covers all SubTiles; the GEMM
		// just steps finer. SubTiles==1 collapses to the original single tile.
		for (int sub = 0; sub < SubTiles; ++sub) {
			int x_mt = tile.y_m * SubTiles + sub;
			int z_m  = z_m_base + sub;
			if constexpr (Compute == 100) smem.mlp1.tmem_base = smem.tmem_base;
			if (warp_id == 0) {
				mlp1_fused_producer<Traits1, Compute == 90>(
					p1_pipe, p1_state,
					smem.mlp1, tma_load_x, tma_load_b, tma_load_c,
					x_mt, (Compute == 90) ? expert : expert * dims.expert_n_stride_1,
					num_tokens, dims.hidden_dim, dims.intermediate_dim,
					dims.num_experts, dims.total_n_rows_1,
					dims.num_n_tiles_1, dims.num_k_tiles_1, n_split, n_count);
			}

			// Fused MoE CTA warp map: warp 0 = TMA, warps 1..3 = NVSHMEM comm
			// (warp 3 = put), warps 4..11 = MLP GEMM consumers. Warp 3 is a comm
			// warp here, so the fused MLP1 keeps the ORIGINAL warp-4 UMMA design
			// (single-warp UMMA + warps 4..11 epilogue). The warp-3 dual-WG
			// rewrite is used ONLY by the isolated MLP1 kernel (no comm warps),
			// so it neither frees warp 3 from comm nor changes the shared MLP
			// barrier count — MLP2 and the rest of the fused pipeline are
			// untouched. Compile-time flag (no register cost). Enabling warp-3
			// here would require the comm-warp + barrier migration (pipeline.md
			// §5) — intentionally out of scope.
			
			if (warp_id >= 3 && warp_id <= 11) {
				mlp1_fused_consumer<Traits1, Compute>(
					p1_pipe, p1_cons_state,
					smem.mlp1, tma_store_z, z_m, dims.intermediate_dim,
					num_z_m_tiles, dims.num_n_tiles_1, dims.num_k_tiles_1, n_split, n_count);
			}

			// Barrier between sub-tiles so the smem mlp1 union is free to reuse.
			// The final sub-tile reuses the shared phase-1-done barrier below.
			if (sub + 1 < SubTiles)
				cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
		}

		// Intra-CTA sync: all MLP warps done with phase 1 (all sub-tiles).
		cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);

		// Release src staging (remote: free slot; local: no-op).
		// Pass threadIdx.x directly so only thread 0 (TMA producer warp
		// leader) has lane==0 and fires the atomicAdd exactly once per
		// CTA — StagePipe expects NumConsumers = NC releases per slot.
		iter.release_src(threadIdx.x);

		// Cross-CTA sync: wait for ALL y-CTAs sharing this blockIdx.x
		// to finish phase 1 for this M-tile (all sub-tiles' Z regions).
		x_barrier.wait();

		// Acquire dst staging (remote: wait for put slot; local: no-op).
		iter.acquire_dst();

		// Y-store target select. Same-host (is_local) tiles store Y DIRECTLY
		// into the destination peer's symmetric local_output via its per-peer
		// descriptor at the token-tile row; all other tiles store to the
		// staging ring (tma_store_y) at the slot row. mlp1's X-load above still
		// uses tile.y_m (the staging slot) regardless — input path unchanged.
		// Tile counts are in CommTileM units → scale by SubTiles for GemmTileM boxes.
		bool y_direct = (y_peer_desc != nullptr && tile.is_local);
		const TmaStoreY& y_desc = y_direct ? y_peer_desc[tile.peer_rank] : tma_store_y;
		int y_base   = y_direct ? tile.y_store_m : tile.y_m;
		int y_ntiles = (y_direct ? dims.y_buf_m_tiles : num_m_tiles) * SubTiles;
		// Phase 2: for each sub-tile, Y[y_base·SubTiles+sub] = Z[z_m_base+sub] @ A^T.
		// Cooperative-M-split: both consumer WGs cooperate on the same (m,n) tile.
		// Cooperative-M-split: both consumer WGs cooperate on the same (m,n) tile.
		for (int sub = 0; sub < SubTiles; ++sub) {
			int z_m     = z_m_base + sub;
			int y_coord = y_base * SubTiles + sub;
			if constexpr (Compute == 100) smem.mlp2.tmem_base = smem.tmem_base;
			if (warp_id == 0) {
				mlp2_fused_producer<Traits2, Compute == 90>(
					p2_pipe, p2_state,
					smem.mlp2, tma_load_z, tma_load_a,
					z_m, (Compute == 90) ? expert : expert * dims.num_n_tiles_2,
					num_z_m_tiles * Traits1::TileM, dims.hidden_dim,
					dims.intermediate_dim, dims.num_experts, dims.total_n_rows_2,
					dims.num_n_tiles_2, dims.num_k_tiles_2, n_split, n_count);
			}
			
			if (warp_id >= 3 && warp_id <= 11) {
				mlp2_fused_consumer<Traits2, Compute>(
					p2_pipe, p2_cons_state,
					smem.mlp2, y_desc,
					y_coord, dims.hidden_dim,
					y_ntiles, dims.num_n_tiles_2, dims.num_k_tiles_2, n_split, n_count);
			}

			// Barrier between sub-tiles so the smem mlp2 union is free to reuse.
			// The final sub-tile reuses the shared post-phase-2 barrier below.
			if (sub + 1 < SubTiles)
				cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
		}

		// Release dst staging (remote: cross-CTA barrier + signal; local: no-op).
		// Pass threadIdx.x directly — exactly one atomicAdd per CTA
		// (threadIdx.x == 0), matching NumProducers = NC on dst_pipe.
		// Device-scope: the system-scope flush of Y to HBM for the NIC put is
		// done by RemoteIter::release_dst below (all threads). This is just the
		// intra-CTA ordering before the MLP-warp barrier.
		__threadfence();
		cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
		iter.release_dst(threadIdx.x);

		tile_iter++;
	}

	if constexpr (Compute == 100) {
		constexpr int kMlp1Cols = Traits1::AccStages * (2 * Traits1::TileN);
		constexpr int kMlp2Cols = Traits2::AccStages * Traits2::TileN;
		constexpr int kTmemColumns = (kMlp1Cols > kMlp2Cols)
			? kMlp1Cols : kMlp2Cols;
		cutlass::arch::NamedBarrier::sync(kMlpBarrierThreads, kMlpBarrierId);
		if (warp_id == 3) {
			tmem_alloc.release_allocation_lock();
			tmem_alloc.free(smem.tmem_base, kTmemColumns);
		}
	}
}

} // namespace liger
