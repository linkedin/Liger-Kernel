#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Tile Iterators
// ═══════════════════════════════════════════════════════════════════
//
// Provide a uniform interface for iterating over M-tiles in the
// fused MLP kernel. Each iterator yields MlpTileInfo per tile.
//
// Both iterators support staging acquire/release hooks:
//   acquire_src / release_src — around phase 1 (X read)
//   acquire_dst / release_dst — around phase 2 (Y write)
// For local tiles these are no-ops. For remote tiles they
// synchronize with the comm warps via StagePipe.
//
// ═══════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cuda/atomic>
#include "mlp_comms.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// MlpTileInfo — what the M-loop body needs per tile
// ═══════════════════════════════════════════════════════════════════

struct MlpTileInfo {
	const void* x_ptr;  // pointer to the start of this M-tile's X data
	int expert;         // expert ID for weight selection
	int y_m;            // staging slot — X-load coord AND staging Y-store coord
	// ── Direct-to-peer Y store (cur_is_local tiles) ──────────────────────
	int y_store_m;      // token-tile index into the destination's local_output
	                    // (= token_offset / TileM); the TMA store row.
	int peer_rank;      // destination peer's intra-host local rank — selects the
	                    // per-peer TMA descriptor built over nvshmem_ptr(y_buf, pe).
	bool is_local;      // same-host (NVLink) → GEMM TMA-stores Y straight into the
	                    // peer's symmetric local_output, bypassing dst_staging+put.
};

// ═══════════════════════════════════════════════════════════════════
// FlatMlpWorkItem — per-(m, n-chunk) work item for the flat scheduler
// ═══════════════════════════════════════════════════════════════════
//
// The flat scheduler hands these out per stage per phase. A CTA loops
//   while (iter.has_next()) { auto item = iter.next(); ... }
// processing each item with the per-(m, [n_start, n_end)) chunk
// producer/consumer.

struct FlatMlpWorkItem {
	int m;            // global M-tile index (into expert_ids / arrival counters)
	int expert;       // expert ID after local_expert_start subtraction
	int z_m;          // Z-buffer M-slot index (double-buffered across stages)
	int n_start;      // chunk start n-tile (inclusive)
	int n_end;        // chunk end n-tile (exclusive); ≤ n_start + NS
	int m_prev_wait;  // global m whose mlp2_arrival must reach num_n_tiles_2
	                  // before this chunk can write Z[z_m]. -1 if no wait
	                  // (first ZBufferSlots stages have no prior occupant).
	int phase;        // 0 = MLP1, 1 = MLP2
};

// ═══════════════════════════════════════════════════════════════════
// LocalMlpTileIterator
// ═══════════════════════════════════════════════════════════════════

template <typename Element, int TileM = 128>
struct LocalMlpTileIterator {
	const Element* x_base;
	const int* expert_ids;
	int num_m_tiles;
	int hidden_dim;
	int tile_offset;  // global tile offset (for TMA coordinates)

	int m;
	int m_end;  // exclusive upper bound for the current sub-range
	int col_stride; // logical grid_x — column stride between this CTA's tiles
	                // (flat-grid launch: NOT gridDim.x, which is num_blocks)

	__device__ void init(const Element* x_base_, const int* expert_ids_,
	                     int num_m_tiles_, int hidden_dim_,
	                     int col_stride_, int start_col_,
	                     int tile_offset_ = 0) {
		x_base = x_base_;
		expert_ids = expert_ids_;
		num_m_tiles = num_m_tiles_;
		hidden_dim = hidden_dim_;
		tile_offset = tile_offset_;
		// Logical column / stride passed explicitly (flat-grid launch) instead
		// of reading blockIdx.x / gridDim.x. The fused driver overrides m via
		// set_range() before iterating; standalone callers use m = start_col.
		col_stride = col_stride_;
		m = start_col_;
		m_end = num_m_tiles;
	}

	// Restrict iteration to [m_start, m_end_). Used by moe_fused_fwd to
	// split the local M-loop around the remote phase so per-tile compute
	// overlaps with remote get/put traffic.
	__device__ void set_range(int m_start, int m_end_) {
		m = m_start;
		m_end = m_end_;
	}

	__device__ bool has_next() const {
		return m < m_end;
	}

	// y_m is the global M-tile index (for TMA coordinates into full buffer).
	__device__ MlpTileInfo next() {
		MlpTileInfo info;
		info.x_ptr  = x_base + (size_t)m * TileM * hidden_dim;
		info.expert = expert_ids[m];
		info.y_m    = tile_offset + m;
		// Local (non-fused) iteration never uses the direct-to-peer store path.
		info.y_store_m = info.y_m;
		info.peer_rank = 0;
		info.is_local  = false;

		m += col_stride;
		return info;
	}

	// No-op staging hooks for local tiles.
	__device__ void acquire_src() {}
	__device__ void release_src(int /*lane*/) {}
	__device__ void acquire_dst() {}
	__device__ void release_dst(int /*lane*/) {}
};

// ═══════════════════════════════════════════════════════════════════
// LocalFlatMlpTileIterator — chunked-flat scheduler work-item iterator
// ═══════════════════════════════════════════════════════════════════
//
// 1D iteration interface. The iterator's internal state machine walks
// stages and phases (MLP1 then MLP2 per stage); each next() returns a
// FlatMlpWorkItem with m / n_start / n_end / expert / z_m / m_prev_wait
// / phase already populated. has_next() is mutating — it advances the
// state machine past phase / stage boundaries until either it finds
// the next work item or signals end-of-iteration.
//
// Grid is 2D: dim3(num_m_per_stage, NC). Total CTAs = num_m_per_stage·NC.
// Per-CTA flat index = blockIdx.x + blockIdx.y · gridDim.x; stride
// over the per-phase work pool = gridDim.x · gridDim.y.
//
// Usage:
//   iter.init(expert_ids, num_m_tiles, num_m_per_stage,
//             num_n_tiles_1, NS1, num_n_tiles_2, NS2);
//   while (iter.has_next()) {
//       FlatMlpWorkItem item = iter.next();
//       if (item.phase == 0) { /* MLP1 */ } else { /* MLP2 */ }
//   }

template <typename Element, int TileM = 128, int ZBufferSlots = 2>
struct LocalFlatMlpTileIterator {
	const int* expert_ids;
	int num_m_tiles;
	int num_m_per_stage;
	int num_stages;
	int local_expert_start;

	// Phase parameters captured at init.
	int num_n_tiles_1;
	int NS1;
	int num_n_chunks_1;
	int num_n_tiles_2;
	int NS2;
	int num_n_chunks_2;

	// CTA identity (decoded once at init).
	int flat_cta_idx;       // blockIdx.x + blockIdx.y · gridDim.x
	int total_ctas;         // gridDim.x · gridDim.y

	// State machine.
	int stage;              // current stage index (-1 before first has_next)
	int stage_m_start;
	int stage_m_count;
	bool in_mlp1;           // true → MLP1 active, false → MLP2 active
	int total_work_phase;
	int work_idx;

	__device__ void init(const int* expert_ids_,
	                     int num_m_tiles_,
	                     int num_m_per_stage_,
	                     int num_n_tiles_1_,
	                     int NS1_,
	                     int num_n_tiles_2_,
	                     int NS2_,
	                     int local_expert_start_ = 0) {
		expert_ids = expert_ids_;
		num_m_tiles = num_m_tiles_;
		num_m_per_stage = num_m_per_stage_;
		num_stages = (num_m_tiles_ + num_m_per_stage_ - 1) / num_m_per_stage_;
		local_expert_start = local_expert_start_;
		num_n_tiles_1 = num_n_tiles_1_;
		NS1 = NS1_;
		num_n_chunks_1 = (num_n_tiles_1_ + NS1_ - 1) / NS1_;
		num_n_tiles_2 = num_n_tiles_2_;
		NS2 = NS2_;
		num_n_chunks_2 = (num_n_tiles_2_ + NS2_ - 1) / NS2_;
		flat_cta_idx = blockIdx.x + blockIdx.y * gridDim.x;
		total_ctas = gridDim.x * gridDim.y;

		// Pre-state: first has_next() will advance into (stage=0, MLP1).
		stage = -1;
		stage_m_start = 0;
		stage_m_count = 0;
		in_mlp1 = false;
		total_work_phase = 0;
		work_idx = flat_cta_idx;
	}

	// Advance to the next (stage, phase) that has any work for this CTA.
	// Returns true if there is a work item to emit at the current state,
	// false when iteration is exhausted.
	__device__ bool has_next() {
		while (work_idx >= total_work_phase) {
			if (in_mlp1) {
				// Finished MLP1 in this stage — start MLP2 (same stage).
				in_mlp1 = false;
				total_work_phase = stage_m_count * num_n_chunks_2;
				work_idx = flat_cta_idx;
			} else {
				// Finished MLP2 (or pre-state) — advance to next stage's MLP1.
				++stage;
				if (stage >= num_stages) return false;
				stage_m_start = stage * num_m_per_stage;
				int rem = num_m_tiles - stage_m_start;
				stage_m_count = (rem < num_m_per_stage) ? rem : num_m_per_stage;
				in_mlp1 = true;
				total_work_phase = stage_m_count * num_n_chunks_1;
				work_idx = flat_cta_idx;
			}
		}
		return true;
	}

	__device__ FlatMlpWorkItem next() {
		FlatMlpWorkItem item;
		int num_n_chunks_phase = in_mlp1 ? num_n_chunks_1 : num_n_chunks_2;
		int num_n_tiles_phase  = in_mlp1 ? num_n_tiles_1  : num_n_tiles_2;
		int NS_phase           = in_mlp1 ? NS1            : NS2;
		int m_in_stage = work_idx / num_n_chunks_phase;
		int chunk_idx  = work_idx % num_n_chunks_phase;
		item.m       = stage_m_start + m_in_stage;
		item.expert  = expert_ids[item.m] - local_expert_start;
		item.z_m     = (stage % ZBufferSlots) * num_m_per_stage + m_in_stage;
		item.n_start = chunk_idx * NS_phase;
		int n_end    = item.n_start + NS_phase;
		item.n_end   = (n_end < num_n_tiles_phase) ? n_end : num_n_tiles_phase;
		if (in_mlp1 && stage >= ZBufferSlots) {
			item.m_prev_wait = (stage - ZBufferSlots) * num_m_per_stage + m_in_stage;
		} else {
			item.m_prev_wait = -1;
		}
		item.phase = in_mlp1 ? 0 : 1;
		work_idx += total_ctas;
		return item;
	}

	// No-op staging hooks for local (standalone) iteration; RemoteFlat
	// variant will wire these to NVSHMEM src/dst pipes.
	__device__ void acquire_src() {}
	__device__ void release_src(int /*lane*/) {}
	__device__ void acquire_dst() {}
	__device__ void release_dst(int /*lane*/) {}
};

// ═══════════════════════════════════════════════════════════════════
// RemoteMlpTileIterator — ticket-based slot synchronization
// ═══════════════════════════════════════════════════════════════════
//
// Iterates over remote tiles prefetched by comm warps. Staging area
// is a flat ring of L = MC · NumStages slots, where MC = (gridDim.x ·
// gridDim.y) / NC is the M-tile count produced per comm pipeline stage.
//
// Producer/consumer counts per slot (unchanged from comm side):
//   src: 2·NC get warps produce, runtime NS GEMM CTAs consume.
//   dst: runtime NS GEMM CTAs produce, NC put warps consume.
//
// ── Ring walk ──────────────────────────────────────────────────────
// Each column blockIdx.x walks its tile sequence with stride gridDim.x
// and wraps the slot index modulo L:
//   T_j   = blockIdx.x + j * gridDim.x       (global tile index for j-th iter)
//   slot  = T_j mod L                         (physical staging slot)
//   ticket = T_j /  L                         (pass count over the ring)
//
// Comm writes tile T to the same slot (= T mod L), so producer and
// consumer always agree on the slot. There is no per-stage-id pipe
// state: the ticket carries enough information to compute the wait
// threshold directly.
//
//   acquire_src(): wait for src_ready[slot]    >= (ticket+1) · 2NC
//   release_src(): atomic increment src_consumed[slot] by 1 (per CTA)
//   acquire_dst(): wait for dst_consumed[slot] >= ticket · NC
//                  (first visit per slot — ticket == 0 — has no
//                  outstanding put to drain, skip the spin)
//   release_dst(): atomic increment dst_ready[slot] by 1 (per CTA)
//
// With runtime NS CTAs cooperating per column, each contributes one
// atomic per release, so src_consumed and dst_ready advance by
// runtime NS per consumed/produced tile — exactly what the comm-side
// StagePipe expects.
//
// ── Decoupling from comm-side constants ────────────────────────────
// Because the ticket comes from the global tile index (not a
// per-CTA cyclic counter), NumStages, runtime NS and NC need not share
// any divisibility relationship. The only correctness requirement is
// that NC evenly divides gridDim.x · gridDim.y so MC is integer
// (host-side concern, not a template constraint).

template <typename Element, int NumStages, int NC = 2, int TileM = 128>
struct RemoteMlpTileIterator {
	// Slot's producer/consumer counts (compile-time, used for ticket math).
	// FWD steady-state GET uses kNumGetWarpsFwd active warps/CTA (the TMA path
	// saturates with one), so the MLP waits for kNumGetWarpsFwd·NC src signals.
	static constexpr int NumProducersSrc = kNumGetWarpsFwd * NC;  // active get warps × NC
	static constexpr int NumConsumersDst = kNumPutWarpsFwd * NC;  // put warps × NC

	// (expert, pe) are re-derived in SMEM, NOT read from HBM: the MLP's j-th tile
	// is global flat tile T_j = m_base + j·col_stride — the same flat position the
	// comm placed at slot=T_j%L, ticket=T_j/L (the StagePipe ticket handshake
	// guarantees the rendezvous). So an embedded TileIterator init(m_base,
	// col_stride) walked in lockstep with idx reproduces what the comm wrote,
	// with zero per-tile HBM load and fewer registers than holding HBM pointers.
	TileIterator<TileM> tit;     // embedded comm-side walker
	bool derive_on;              // false → no remote_offsets (test harness) → local
	int gpus_per_node;           // same-host divisor for is_local
	int iter_my_pe;              // this PE (team space) for is_local
	int cur_is_local;            // cached per-tile fence-scope predicate

	int total_tiles;
	int m_base;       // logical column index (flat_id / runtime NS) — column's
	                  // tile-sequence start
	int col_stride;   // logical grid_x (= n_gemm / runtime NS) — tile-sequence
	                  // stride; NOT gridDim.x under the flat-grid launch
	int ring_len;     // L = MC · NumStages, set at init
	int num_splits;

	int* src_ready;
	int* src_consumed;
	int* dst_ready;
	int* dst_consumed;

	bool is_leader;   // per-warp leader (lane 0) — does the atomic poll

	int idx;          // iteration counter within this column
	int cur_slot;     // cached slot for current acquire/release cycle
	int cur_ticket;   // cached ticket for current acquire/release cycle

	// remote_offsets (SMEM) drives the embedded walker; pass nullptr to disable
	// derivation (e.g. the standalone handshake test, which only exercises the
	// ticket/slot protocol). experts_per_pe/num_pes/my_pe/gpus_per_node feed the
	// walker and the same-host is_local predicate.
	__device__ void init(const int* remote_offsets,
	                     int experts_per_pe,
	                     int num_pes,
	                     int my_pe,
	                     int gpus_per_node_,
	                     int total_tiles_,
	                     int m_base_,
	                     int n_gemm_,
	                     int* src_ready_, int* src_consumed_,
	                     int* dst_ready_, int* dst_consumed_,
	                     bool is_leader_,
	                     int runtime_nsplit_) {
		total_tiles = total_tiles_;
		m_base = m_base_;
		num_splits = runtime_nsplit_;
		col_stride = n_gemm_ / num_splits;
		idx = 0;

		// L = MC · NumStages, MC = n_gemm / NC.
		int mc = gridDim.x / NC;
		ring_len = mc * NumStages;

		src_ready    = src_ready_;
		src_consumed = src_consumed_;
		dst_ready    = dst_ready_;
		dst_consumed = dst_consumed_;
		is_leader    = is_leader_;
		cur_slot   = 0;
		cur_ticket = 0;

		gpus_per_node = gpus_per_node_;
		iter_my_pe    = my_pe;
		cur_is_local  = 1;
		derive_on = (remote_offsets != nullptr);
		if (derive_on) {
			tit.remote_offsets = remote_offsets;
			tit.offsets_stride = experts_per_pe + 1;
			tit.experts_per_pe = experts_per_pe;
			tit.num_pes        = num_pes;
			tit.my_pe          = my_pe;
			// Walk the MLP column's flat-tile subsequence (start m_base, stride
			// col_stride) — identical positions to compute_slot()'s T_j.
			tit.init(m_base_, col_stride);
		}
	}

	__device__ bool has_next() const {
		return idx < total_tiles;
	}

	// Compute and cache (slot, ticket) for the current iteration.
	// Called by acquire_src() — the rest of the per-tile API (next,
	// release_src, acquire_dst, release_dst) reuses the cached values.
	__device__ void compute_slot() {
		int T = m_base + idx * col_stride;
		cur_slot   = T % ring_len;
		cur_ticket = T / ring_len;
	}

	// Re-derive (expert, pe) for this tile from SMEM offsets — the embedded
	// walker advances in lockstep with idx and reproduces what the comm wrote at
	// this slot (proven: same global flat position). No HBM load.
	__device__ MlpTileInfo next() {
		MlpTileInfo info;
		info.x_ptr  = nullptr;  // not used — TMA loads by coordinate
		info.y_m    = cur_slot;
		if (derive_on) {
			TileInfo ti = tit.next();
			info.expert  = ti.expert;
			bool loc = (gpus_per_node > 0)
				? ((ti.pe / gpus_per_node) == (iter_my_pe / gpus_per_node))
				: true;
			cur_is_local   = (int)loc;
			info.is_local  = loc;
			// Destination peer's intra-host local rank → per-peer y_buf descriptor.
			info.peer_rank = (gpus_per_node > 0) ? (ti.pe % gpus_per_node) : 0;
			// Direct-store row = this tile's token offset in the peer's local_output.
			info.y_store_m = ti.token_offset / TileM;
		} else {
			// Test harness (no SMEM offsets): no derivation, no direct store.
			info.expert    = 0;
			cur_is_local   = 1;
			info.is_local  = false;
			info.peer_rank = 0;
			info.y_store_m = cur_slot;
		}
		idx++;
		return info;
	}

	// Wait for comm's get to finish writing this tile to its slot.
	// Producer (comm) signals ready += 2·NC per tile-write at the slot,
	// so after ticket K the wait threshold is (K+1)·2·NC.
	__device__ void acquire_src() {
		compute_slot();
		int target = (cur_ticket + 1) * NumProducersSrc;
		if (is_leader) {
			// Relaxed spin (ld.relaxed.gpu — cheapest read, no RMW, no per-iter
			// fence); the __threadfence below establishes the acquire ordering
			// once for all lanes on exit. atomic_ref ctor is zero-cost/hoisted.
			cuda::atomic_ref<int, cuda::thread_scope_device> r(src_ready[cur_slot]);
			while (r.load(cuda::memory_order_relaxed) < target) {}
		}
		__syncwarp();
		// Device-scope: observing NIC-written src. The NIC's inbound-write
		// visibility is not fixable by a fence (needs GDRCopy), so system scope
		// here only costs bandwidth. Device fence covers the GPU-internal view.
		__threadfence();
	}

	// Signal that this CTA has consumed the tile at cur_slot. Per
	// tile (column-of-runtime-NS CTAs), src_consumed advances by runtime NS.
	__device__ void release_src(int lane) {
		__threadfence();
		if (lane == 0) {
			atomicAdd(&src_consumed[cur_slot], 1);
		}
	}

	// Wait for the previous put from cur_slot to finish (so we can
	// reuse the slot for a fresh Y write). ticket==0 → no prior put.
	__device__ void acquire_dst() {
		if (cur_ticket >= 1) {
			int target = cur_ticket * NumConsumersDst;
			if (is_leader) {
				cuda::atomic_ref<int, cuda::thread_scope_device> r(dst_consumed[cur_slot]);
				while (r.load(cuda::memory_order_relaxed) < target) {}
			}
			__syncwarp();
			__threadfence();
		}
	}

	// Signal that Y is ready in cur_slot for the comm put. Per tile,
	// dst_ready advances by runtime NS.
	__device__ void release_dst(int lane) {
		// Same-host tiles store Y directly into the peer's symmetric
		// local_output, so the put warp has no staged data to read. Cross-host
		// tiles use dst_staging + putmem and require a system-scope fence before
		// the put warp observes dst_ready.
		if (!cur_is_local) __threadfence_system();
		if (lane == 0) {
			atomicAdd(&dst_ready[cur_slot], 1);
		}
	}
};

} // namespace liger
