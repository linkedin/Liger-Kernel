#pragma once

// ═══════════════════════════════════════════════════════════════════
// Backward MLP tile iterators
// ═══════════════════════════════════════════════════════════════════
//
// Both iterators mirror their forward counterparts in tile_iterator.cuh
// and add acquire_dy/release_dy hooks for the bwd-only dY pipe. Kept in
// a separate header from the forward iterators because they depend on
// mlp_comms_bwd.cuh (CommBuffersBwd), which forward code does not need.

#include "tile_iterator.cuh"
#include "mlp_comms_bwd.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// LocalMlpTileIteratorBwd
// ═══════════════════════════════════════════════════════════════════
//
// Backward local tile iterator. Mirrors LocalMlpTileIterator (including
// set_range / m_end sub-range support) and adds acquire_dy/release_dy
// hooks (no-ops for local tiles).
//
// Tiles are visited in sort order (expert-contiguous). The caller can
// detect expert boundaries from tile.expert if needed.

template <typename Element, int TileM = 128>
struct LocalMlpTileIteratorBwd {
	const Element* x_base;
	const int* expert_ids;
	int num_m_tiles;
	int hidden_dim;
	int tile_offset;
	int col_stride;  // flat-grid column stride (= grid_x); gridDim.x fallback

	int m;
	int m_end;  // exclusive upper bound for the current sub-range

	// Flat-grid launch: start_col_ = flat_id / NSplit (this CTA's logical
	// column), col_stride_ = grid_x (= n_gemm / NSplit). Defaults (-1) fall
	// back to blockIdx.x / gridDim.x for legacy 2-D-grid standalone callers.
	__device__ void init(const Element* x_base_, const int* expert_ids_,
	                     int num_m_tiles_, int hidden_dim_,
	                     int tile_offset_ = 0,
	                     int start_col_ = -1, int col_stride_ = -1) {
		x_base = x_base_;
		expert_ids = expert_ids_;
		num_m_tiles = num_m_tiles_;
		hidden_dim = hidden_dim_;
		tile_offset = tile_offset_;
		col_stride = (col_stride_ >= 0) ? col_stride_ : (int)gridDim.x;
		m = (start_col_ >= 0) ? start_col_ : (int)blockIdx.x;
		m_end = num_m_tiles;
	}

	// Restrict iteration to [m_start, m_end_). Mirrors fwd
	// LocalMlpTileIterator::set_range — kept for symmetry even though
	// moe_fused_bwd currently only does a single full local pass.
	__device__ void set_range(int m_start, int m_end_) {
		m = m_start;
		m_end = m_end_;
	}

	__device__ bool has_next() const {
		return m < m_end;
	}

	__device__ MlpTileInfo next() {
		MlpTileInfo info;
		info.x_ptr  = x_base + (size_t)m * TileM * hidden_dim;
		info.expert = expert_ids[m];
		info.y_m    = tile_offset + m;

		m += col_stride;
		return info;
	}

	// All staging hooks are no-ops for local tiles.
	__device__ void acquire_src() {}
	__device__ void release_src(int /*lane*/) {}
	__device__ void acquire_dy() {}
	__device__ void release_dy(int /*lane*/) {}
	__device__ void acquire_dst() {}
	__device__ void release_dst(int /*lane*/) {}
};

// ═══════════════════════════════════════════════════════════════════
// RemoteMlpTileIteratorBwd — ticket-based slot synchronization
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors the FWD RemoteMlpTileIterator's ticket-based design. Three
// pipes (X, dY, dX) all key on the same per-iteration (slot, ticket)
// pair, computed once in acquire_src() and reused by acquire_dy /
// acquire_dst / release_dst / release_src / release_dy. The release
// hooks for src and dy are deferred past Phase 2 mlp4 and mlp3
// respectively (callers in mlp_bwd.cuh) — both reuse the same cached
// (cur_slot, cur_ticket) so deferral does not require any extra state.
//
// Ring shape:
//   L          = MC · NumStages
//   MC         = (gridDim.x · gridDim.y) / NC
//   T_j        = m_base + j · gridDim.x        (column's j-th global tile)
//   cur_slot   = T_j mod L
//   cur_ticket = T_j /  L
//
// Producer/consumer counts per slot:
//   src (X):  2·NC get warps produce, N_SPLIT GEMM CTAs consume.
//   dy:       2·NC get warps produce, N_SPLIT GEMM CTAs consume.
//   dst (dX): N_SPLIT GEMM CTAs produce, NC put warps consume.
//
// Wait thresholds (atomic poll by leader, fence-syncwarp for non-leaders):
//   acquire_src(): x_src_ready[slot]  >= (ticket+1) · 2NC
//   acquire_dy():  dy_src_ready[slot] >= (ticket+1) · 2NC
//   acquire_dst(): dst_consumed[slot] >= ticket     · NC
//                  (first visit per slot — ticket == 0 — has no
//                  outstanding put to drain, skip the spin)
//
// Release increments (per consuming/producing CTA):
//   release_src(): atomicAdd(&x_src_consumed[slot], 1)
//   release_dy():  atomicAdd(&dy_src_consumed[slot], 1)
//   release_dst(): atomicAdd(&dst_ready[slot], 1)
//
// Decoupling: since the ticket comes from the global tile index, no
// divisibility relationship between NumStages, N_SPLIT and NC is
// required. The only host-side correctness requirement is that NC
// evenly divides gridDim.x · gridDim.y so MC is integer.

template <typename Element, int NumStages, int NC = 2, int TileM = 128>
struct RemoteMlpTileIteratorBwd {
	// Slot's producer/consumer counts. RUNTIME (not constexpr) because the
	// comm warp layout switches on the TMA-GET enable flag:
	//   getmem  : 2·NC get warps produce X/dY, NC put warps consume dX
	//   TMA     : 1·NC get warp  produces X/dY, 2·NC put warps consume dX
	// Set in init() from tma_enabled so the acquire thresholds match the
	// actual number of producer_release / consumer_release calls per slot.
	int NumProducersSrc;   // get warps × NC
	int NumConsumersDst;   // put warps × NC

	const int* tile_expert_ids;  // [L] — comm writes expert id at slot index
	int total_tiles;
	int m_base;       // logical column (flat_id / runtime NS) — tile-seq start
	int col_stride;   // grid_x = n_gemm / runtime NS (column stride)
	int ring_len;     // L = MC · NumStages, set at init
	int num_splits;

	int* x_src_ready;
	int* x_src_consumed;
	int* dy_src_ready;
	int* dy_src_consumed;
	int* dst_ready;
	int* dst_consumed;

	bool is_leader;   // per-warp leader (lane 0) — does the atomic poll

	int idx;          // iteration counter within this column
	int cur_slot;     // cached slot for current acquire/release cycle
	int cur_ticket;   // cached ticket for current acquire/release cycle

	// Flat-grid launch: m_base_ = flat_id / NSplit (logical column),
	// n_gemm_ = floor_NS (= grid_x · NSplit). col_stride = n_gemm_ / N_SPLIT,
	// MC = n_gemm_ / NC. n_gemm_ < 0 falls back to gridDim.x · gridDim.y for
	// legacy 2-D-grid standalone callers.
	__device__ void init(const int* tile_expert_ids_,
	                     int total_tiles_,
	                     int m_base_,
	                     int n_gemm_,
	                     int* x_src_ready_, int* x_src_consumed_,
	                     int* dy_src_ready_, int* dy_src_consumed_,
	                     int* dst_ready_, int* dst_consumed_,
	                     bool is_leader_,
	                     int runtime_nsplit_,
	                     bool tma_enabled_ = false) {
		// Producer/consumer counts follow the comm-side warp layout, which is
		// chosen by the same tma_enabled flag (see nvshmem_comm_main_bwd).
		NumProducersSrc = (tma_enabled_ ? kNumGetWarpsBwd : kNumGetWarpsPerCta) * NC;
		NumConsumersDst = (tma_enabled_ ? kNumPutWarpsBwd : 1) * NC;
		tile_expert_ids = tile_expert_ids_;
		total_tiles = total_tiles_;
		m_base = m_base_;
		num_splits = runtime_nsplit_;
		idx = 0;

		int n_gemm = (n_gemm_ >= 0) ? n_gemm_
		                            : ((int)gridDim.x * (int)gridDim.y);
		col_stride = n_gemm / num_splits;
		// L = MC · NumStages, MC = launched NC-complete comm columns.
		int mc = (int)gridDim.x / NC;
		ring_len = mc * NumStages;

		x_src_ready    = x_src_ready_;
		x_src_consumed = x_src_consumed_;
		dy_src_ready   = dy_src_ready_;
		dy_src_consumed= dy_src_consumed_;
		dst_ready      = dst_ready_;
		dst_consumed   = dst_consumed_;
		is_leader      = is_leader_;
		cur_slot   = 0;
		cur_ticket = 0;
	}

	__device__ bool has_next() const {
		return idx < total_tiles;
	}

	// Compute and cache (slot, ticket) for the current iteration.
	// Called by acquire_src() — the rest of the per-tile API (next,
	// acquire_dy, release_*, acquire_dst) reuses the cached values.
	__device__ void compute_slot() {
		int T = m_base + idx * col_stride;
		cur_slot   = T % ring_len;
		cur_ticket = T / ring_len;
	}

	// Called AFTER acquire_src(), so do_get_bwd's __threadfence has
	// made tile_expert_ids[slot] visible.
	__device__ MlpTileInfo next() {
		MlpTileInfo info;
		info.x_ptr  = nullptr;  // not used — TMA loads by coordinate
		info.expert = tile_expert_ids[cur_slot];
		info.y_m    = cur_slot;
		idx++;
		return info;
	}

	// Wait for comm warps to finish X get into staging.
	// Producer (comm) signals ready += 2·NC per tile-write at the slot,
	// so after ticket K the wait threshold is (K+1)·2NC.
	__device__ void acquire_src() {
		compute_slot();
		int target = (cur_ticket + 1) * NumProducersSrc;
		if (is_leader) {
			while (atomicAdd(&x_src_ready[cur_slot], 0) < target) {}
		}
		__syncwarp();
		__threadfence();  // observe NIC-written X (inbound visibility needs GDRCopy, not a fence)
	}

	// Free X slot for comm warps to reuse (deferred past Phase 2 mlp4).
	// Per tile (column-of-N_SPLIT CTAs), x_src_consumed advances by N_SPLIT.
	__device__ void release_src(int lane) {
		__threadfence();
		if (lane == 0) {
			atomicAdd(&x_src_consumed[cur_slot], 1);
		}
	}

	// Sub-batch-grouped variant: release an EXPLICIT slot rather than the
	// cached cur_slot. Used when Phase 1 acquires S sub-batches before the
	// grouped Phase 2 — by release time cur_slot holds the LAST sub-batch's
	// slot, so the earlier sub-batches must be freed by their saved slot.
	// release_src/release_dy only ever index x_src_consumed[slot] /
	// dy_src_consumed[slot] (the ticket is unused), so the saved slot alone
	// is sufficient. Caller emits ONE __threadfence (shared across the batch
	// of releases) before invoking these per saved slot.
	__device__ void release_src_slot(int slot, int lane) {
		if (lane == 0) {
			atomicAdd(&x_src_consumed[slot], 1);
		}
	}

	// Wait for comm warps to finish dY get into staging.
	__device__ void acquire_dy() {
		int target = (cur_ticket + 1) * NumProducersSrc;
		if (is_leader) {
			while (atomicAdd(&dy_src_ready[cur_slot], 0) < target) {}
		}
		__syncwarp();
		__threadfence();  // observe NIC-written dY (inbound visibility needs GDRCopy, not a fence)
	}

	// Free dY slot for comm warps to reuse (deferred past Phase 2 mlp3).
	__device__ void release_dy(int lane) {
		__threadfence();
		if (lane == 0) {
			atomicAdd(&dy_src_consumed[cur_slot], 1);
		}
	}

	// Sub-batch-grouped variant (see release_src_slot).
	__device__ void release_dy_slot(int slot, int lane) {
		if (lane == 0) {
			atomicAdd(&dy_src_consumed[slot], 1);
		}
	}

	// Wait for previous dX put to finish (staging slot free).
	// ticket==0 → no prior put.
	__device__ void acquire_dst() {
		if (cur_ticket >= 1) {
			int target = cur_ticket * NumConsumersDst;
			if (is_leader) {
				while (atomicAdd(&dst_consumed[cur_slot], 0) < target) {}
			}
			__syncwarp();
			__threadfence();
		}
	}

	// Signal comm warps that dX is ready to be put. Per tile, dst_ready
	// advances by N_SPLIT.
	__device__ void release_dst(int lane) {
		__threadfence();  // device-scope (NVLink-only build); IB put needs __threadfence_system()
		if (lane == 0) {
			atomicAdd(&dst_ready[cur_slot], 1);
		}
	}
};

// ═══════════════════════════════════════════════════════════════════
// FusedMlpTileIteratorBwd
// ═══════════════════════════════════════════════════════════════════
//
// Thin wrapper over a single RemoteMlpTileIteratorBwd. Every BWD tile —
// local or remote — flows through the comm get→staging→MLP→put path (the
// separate materialized local pass was removed), so there is only the remote
// staging-ring sub-iterator and all accessors forward straight to it. The
// wrapper exists only to keep a stable iterator type at the moe_fused_bwd /
// mlp_fused_bwd_dual call sites.

template <typename Element, int NumStages, int NC = 2, int TileM = 128>
struct FusedMlpTileIteratorBwd {
	RemoteMlpTileIteratorBwd<Element, NumStages, NC, TileM> remote;

	__device__ void init_remote(const int* tile_expert_ids,
	                             int total_tiles,
	                             int m_base,
	                             int n_gemm,
	                             int* x_src_ready, int* x_src_consumed,
	                             int* dy_src_ready, int* dy_src_consumed,
	                             int* dst_ready, int* dst_consumed,
	                             bool is_leader,
	                             int runtime_nsplit,
	                             bool tma_enabled = false) {
		remote.init(tile_expert_ids, total_tiles, m_base, n_gemm,
			x_src_ready, x_src_consumed, dy_src_ready, dy_src_consumed,
			dst_ready, dst_consumed, is_leader, runtime_nsplit, tma_enabled);
	}

	// All accessors forward to the remote sub-iterator (local pass removed).
	__device__ bool has_next() const { return remote.has_next(); }
	__device__ MlpTileInfo next() { return remote.next(); }
	__device__ void acquire_src() { remote.acquire_src(); }
	__device__ void release_src(int lane) { remote.release_src(lane); }
	__device__ void release_src_slot(int slot, int lane) { remote.release_src_slot(slot, lane); }
	__device__ void acquire_dy() { remote.acquire_dy(); }
	__device__ void release_dy(int lane) { remote.release_dy(lane); }
	__device__ void release_dy_slot(int slot, int lane) { remote.release_dy_slot(slot, lane); }
	__device__ void acquire_dst() { remote.acquire_dst(); }
	__device__ void release_dst(int lane) { remote.release_dst(lane); }
};

} // namespace liger
