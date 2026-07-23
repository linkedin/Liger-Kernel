#pragma once

// ═══════════════════════════════════════════════════════════════════
// Fused MoE: MLP compute for local + remote tiles
// ═══════════════════════════════════════════════════════════════════
//
// MLP-only device function — called by MLP warps (0,1,4-11).
// Comm warps (2-3) are handled separately at the kernel level to
// keep CommBuffers out of the MLP register scope.
//
// Execution flow (when remote tiles exist):
//   a. mlp_fused_fwd with LocalMlpTileIterator (first half of column)
//   b. NamedBarrier (MLP warps only)
//   c. mlp_fused_fwd with RemoteMlpTileIterator
//   d. NamedBarrier (MLP warps only)
//   e. mlp_fused_fwd with LocalMlpTileIterator (second half of column)
//
// Splitting the local pass around the remote pass lets the comm warps
// overlap remote gets with the first local half and remote puts with
// the second local half. When there are no remote tiles, the kernel
// runs a single full local pass and returns.
//
// For remote tiles:
//   - X input comes from src_staging (filled by comm get)
//   - Y output goes to dst_staging (drained by comm put)
//   - The RemoteMlpTileIterator acquire/release hooks synchronize
//     with comm warps via StagePipe
// ═══════════════════════════════════════════════════════════════════

#include "mlp.cuh"

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// SymmetricMemoryStack keys for the persistent fwd→bwd buffers.
// Centralized so set_size / put / pop / top sites all agree.
// ═══════════════════════════════════════════════════════════════════
inline constexpr const char* kSymmKeyXSorted          = "x_sorted";
inline constexpr const char* kSymmKeyYBuf             = "y_buf";
inline constexpr const char* kSymmKeyAllExpertOffsets = "all_expert_offsets";

// ═══════════════════════════════════════════════════════════════════
// MoE shared memory — union of MLP smem + comm smem
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2, int Compute = 90>
struct MoeSmem {
	// MLP shared memory (fused phase 1 + 2 with pipeline barriers).
	MlpFusedSmem<Traits1, Traits2, Compute> mlp;

	// Comm shared memory (total_tiles count + remote_offsets pointer).
	CommSmem comm;
};

// ═══════════════════════════════════════════════════════════════════
// moe_fused_fwd — MLP warps only (no CommBuffers in scope)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits1, typename Traits2,
          int ZBufferSlots,
          int CommNumStages,
          int NC,
          // kSubTiles = CommTileM / GemmTileM ∈ {1,2}. >1 means the GEMM steps
          // through SubTiles sub-tiles of Traits1::TileM (=GemmTileM) rows per
          // 128-wide comm staging slot. 1 → unchanged single-tile path.
          int SubTiles,
          int Compute = 90,
          typename LocalIter, typename RemoteIter,
          typename TmaLoadX, typename TmaLoadW1, typename TmaStoreZ,
          typename TmaLoadZ, typename TmaLoadW2, typename TmaStoreY>
__device__ __forceinline__ void moe_fused_fwd(
		MoeSmem<Traits1, Traits2, Compute>& smem,
		LocalIter& local_iter,
		// Remote-phase comm pointers (no CommBuffers — keeps MLP register scope clean)
		int* src_ready,
		int* src_consumed,
		int* dst_ready,
		int* dst_consumed,
		// RemoteIter re-derives (expert, pe) and the per-tile is_local fence-scope
		// predicate from remote_offsets (SMEM) using these — no per-slot HBM.
		int experts_per_pe,
		int num_pes,
		int my_pe,
		int gpus_per_node,
		// MLP1 TMA descriptors
		TmaLoadX  const& tma_load_x,
		TmaLoadW1 const& tma_load_b,
		TmaLoadW1 const& tma_load_c,
		TmaStoreZ const& tma_store_z,
		// MLP2 TMA descriptors
		TmaLoadZ  const& tma_load_z,
		TmaLoadW2 const& tma_load_a,
		TmaStoreY const& tma_store_y,
		// Remote-phase TMA descriptors (X loaded from staging, Y stored to staging)
		TmaLoadX  const& tma_load_x_remote,
		TmaStoreY const& tma_store_y_remote,
		// Invariant MLP dims — const ref into __grid_constant__ .param space
		const MlpDims& dims,
		// Per-phase params (change between local and remote)
		int num_tokens,
		int num_m_tiles,
		// Flat-grid logical coordinates: col = flat_id / NSplit (this CTA's
		// GEMM column), grid_x = n_gemm / NSplit (number of columns). Under
		// the flat launch gridDim.x = num_blocks and gridDim.y = 1, so these
		// replace every blockIdx.x / gridDim.x / gridDim.y read below.
		int col,
		int grid_x,
		int split,            // this CTA's N-split index = flat_id % runtime_nsplit
		int runtime_nsplit,
		int local_expert_start = 0,
		// Direct-to-peer Y store: per-peer TMA descriptors over
		// nvshmem_ptr(local_output, peer). Gated per-tile on is_local downstream.
		const TmaStoreY* y_peer_desc = nullptr) {

	int warp_id = threadIdx.x / Traits1::WarpSize;

	// Construct the per-CTA x_barrier ONCE here so its target counter
	// accumulates monotonically across all 3 mlp_fused_fwd calls
	// (A1, B, A2). Each phase invokes mlp_fused_fwd which calls
	// x_barrier.wait() inside its tile loop — they must share the
	// same barrier object so target stays aligned with the shared
	// device counter. Constructing the barrier inside mlp_fused_fwd
	// would reset target=0 per call while the counter keeps growing,
	// and the second mlp_fused_fwd would see counter > target on
	// its very first poll and exit without actually waiting.
	MlpFwdCtaBarrierT<Compute> x_barrier(&dims.phase_counter[col],
		runtime_nsplit);

	// UNIFIED PATH: the local MLP pass has been removed. The comm-side
	// TileIterator now enumerates LOCAL experts (p=0) alongside remote ones,
	// and the get warp stages local X from local symmetric memory (do_get
	// is_local TMA path) exactly like remote tiles. So EVERY tile this column
	// owns — local or remote — flows through the RemoteMlpTileIterator /
	// src_staging→MLP→dst_staging pipeline below. local_iter is unused now.
	// smem.comm.per_cta_tiles is this column's total tile count (stride
	// grid_x); a column with no tiles returns immediately.
	(void)local_iter; (void)tma_load_x; (void)tma_store_y; (void)num_tokens;
	(void)num_m_tiles; (void)local_expert_start;
	int total_remote = smem.comm.per_cta_tiles;
	if (total_remote == 0) return;

	bool is_leader = (warp_id == 0 && (threadIdx.x % Traits1::WarpSize) == 0);

	// Build RemoteMlpTileIterator. The staging is a flat ring of
	// L = MC · CommNumStages slots, where MC = (gridDim.x · gridDim.y) / NC,
	// indexed by
	//     slot = (blockIdx.x + j · gridDim.x) mod L   (MLP, ticket-based)
	//     slot = xc          + (j mod CommNumStages) · MC
	//                                                 (comm, NumStages cycle).
	// Both sides resolve to the same slot for a given global tile T and
	// cover the full L. No divisibility relationship is required between
	// NC, NSplit and CommNumStages (runtime: NC must divide gridDim.x ·
	// gridDim.y so MC is integer).
	// n_gemm = grid_x · runtime_nsplit — the GEMM/comm-active CTA count.
	int n_gemm = grid_x * runtime_nsplit;
	RemoteIter remote_iter;
	remote_iter.init(
		// remote_offsets in SMEM drives the embedded walker (null → no derivation).
		smem.comm.remote_offsets, experts_per_pe, num_pes, my_pe, gpus_per_node,
		smem.comm.per_cta_tiles,
		col,
		n_gemm,
		src_ready, src_consumed,
		dst_ready, dst_consumed,
		is_leader,
		runtime_nsplit);

	// Remote MLP dimensions: TMA descriptors cover the full flat staging
	// list = MC · CommNumStages · TileM rows. MC = gridDim.x / NC — MUST match
	// the ring size in allocate_moe_buffers and the consumer's ring_len
	// (RemoteMlpTileIterator mc = gridDim.x/NC); sizing on n_gemm here would
	// leave the wrap-around slots [n_gemm/NC·CS, gridDim.x/NC·CS) outside the
	// staging coverage (OOB) when gridDim.x/NC > n_gemm/NC.
	int mc = gridDim.x / NC;
	int remote_num_m_tiles = mc * CommNumStages;

	// NOTE: phase_counter is monotonic across phase A and phase B —
	// the modulo-gridDim.y check in mlp_fused_fwd is invariant
	// under any shared starting value. The host zeroes phase_counter
	// before each kernel launch (see moe.cu); resetting it here would
	// race across the gridDim.y CTAs in the column.

	// num_tokens = total staging rows = remote_num_m_tiles COMM tiles × CommTileM.
	// CommTileM = Traits1::TileM (=GemmTileM) × SubTiles. num_m_tiles passed is the
	// COMM tile count; mlp_fused_fwd scales it by SubTiles for the GEMM sub-tiles.
	mlp_fused_fwd<Traits1, Traits2, ZBufferSlots, SubTiles, Compute>(
		smem.mlp, remote_iter,
		tma_load_x_remote, tma_load_b, tma_load_c, tma_store_z,
		tma_load_z, tma_load_a, tma_store_y_remote,
		dims,
		remote_num_m_tiles * Traits1::TileM * SubTiles, remote_num_m_tiles, x_barrier,
		/*local_expert_start=*/0, col, grid_x, split, runtime_nsplit,
		/*y_peer_desc=*/y_peer_desc);
}

} // namespace liger
