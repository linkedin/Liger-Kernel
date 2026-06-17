#pragma once

// ═══════════════════════════════════════════════════════════════════
// Fused MoE MLP with NVSHMEM communication
// ═══════════════════════════════════════════════════════════════════
//
// Combines nvshmem get/put with MLP compute in a single persistent
// kernel. Warps 1-3 of WG0 handle nvshmem communication while the
// rest of the CTA runs the MLP1 pipeline.
//
// Warp roles (384 threads = 12 warps):
//
//   WG0 (warps 0–3):
//     warp 0   TMA producer (mlp1 pipelines)
//     warp 1   NVSHMEM get warp 0 (fetches X into src_staging)
//     warp 2   NVSHMEM get warp 1 (fetches X into src_staging)
//     warp 3   NVSHMEM put warp (sends Y from dst_staging)
//
//   WG1 (warps 4–7):  Consumer A (mlp1 WGMMA)
//   WG2 (warps 8–11): Consumer B (mlp1 WGMMA)
//
// Staging areas (per blockIdx.x, shared by both blockIdx.y):
//   src_staging[NumStages][TileM][hidden_dim]
//   dst_staging[NumStages][TileM][intermediate_dim]
//
// Schedule: the two get warps and the put warp each run an
// independent flat loop over the same tile sequence. The NumStages
// staging ring bounds how far the get warps can lead the put warp.
//
// NVSHMEM get throughput is roughly half of put; we counteract that
// by issuing 2× as many gets per tile. Each tile is chunked across
// 2·NC warps on the get side (2 warps per CTA × NC CTAs) but only NC
// warps on the put side. Get chunk = tile_bytes / (2·NC); put chunk
// = tile_bytes / NC.
//
// Transport-dependent chunking:
//   * NVLink P2P (nvshmem_ptr != nullptr): all 2·NC get warps / NC put
//     warps cooperate per tile, each copying its chunk with coalesced
//     int4 loads/stores — many warps maximise NVLink throughput.
//   * IB (nvshmem_ptr == nullptr): the per-warp chunking is counter-
//     productive (small RDMA messages, per-tile rendezvous). Instead a
//     SINGLE warp issues one whole-tile getmem/putmem, and the warp that
//     issues rotates round-robin across the 2·NC (get) / NC (put) warps
//     keyed by the tile sequence index, so consecutive tiles are issued
//     by different warps. Their blocking transfers then overlap (bounded
//     by the NumStages staging ring). The non-issuing warps do no I/O and
//     just rubber-stamp the StagePipe so the producer/consumer counters
//     stay intact.
//
// ═══════════════════════════════════════════════════════════════════

#include <cuda.h>            // CUtensorMap (raw TMA descriptor)
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/ptx>          // typed TMA / mbarrier intrinsics (replaces inline PTX)
#include <cuda_bf16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "nvshmem_helpers.cuh"  // g_dest_table

#include "cta_barrier.cuh"  // SyncThreadsCtaCounterBarrier

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// StagePipe — bidirectional acquire/release for a staging buffer
// ═══════════════════════════════════════════════════════════════════
//
// Wraps a pair of atomic counters (ready[S], consumed[S]) shared
// between a producer and a consumer. Each side calls acquire/release:
//
//   Producer: acquire() waits on consumed[] (slot free), then
//             release() signals ready[] (data written).
//   Consumer: acquire() waits on ready[] (data available), then
//             release() signals consumed[] (slot freed).
//
// Internally tracks stage index and monotonic thresholds.
// is_leader is set at init — only the leader thread polls atomics.

template <int NumStages, int NumProducers, int NumConsumers>
struct StagePipe {
	int* ready;       // producer signals, consumer waits
	int* consumed;    // consumer signals, producer waits
	int  stride;      // step in slots between consecutive logical stages
	bool is_leader;

	int prod_count;
	int cons_count;
	int prod_expected[NumStages];  // threshold for consumed[] (producer waits on)
	int cons_expected[NumStages];  // threshold for ready[] (consumer waits on)

	// Stride lets the pipe address into a flat slot list with non-unit
	// stage spacing (e.g. comm pipes use stride = MC, MLP pipes use
	// stride = gridDim.x). Default stride = 1 preserves the original
	// contiguous-ring layout used by the bwd path.
	__device__ void init(int* ready_, int* consumed_, bool is_leader_,
	                     int stride_ = 1) {
		ready = ready_;
		consumed = consumed_;
		stride = stride_;
		is_leader = is_leader_;
		prod_count = 0;
		cons_count = 0;
		for (int i = 0; i < NumStages; ++i) {
			prod_expected[i] = 0;
			cons_expected[i] = 0;
		}
	}

	// ── Producer side ───────────────────────────────────────

	__device__ int producer_stage() const {
		return prod_count % NumStages;
	}

	// Wait until this slot is free (consumer has released it).
	//
	// Acquire-side memory ordering: only the leader does the atomic poll
	// (cheaper than 32 lanes spinning), but ALL lanes need to see the
	// counterparty's writes after this returns. __syncwarp alone does NOT
	// establish global-memory ordering — non-leader lanes may still hit
	// stale L1/L2. We add a __threadfence after the leader's poll AND after
	// __syncwarp so every lane has a release/acquire pair.
	__device__ void producer_acquire() {
		int s = producer_stage();
		if (prod_count >= NumStages) {
			if (is_leader) {
				cuda::atomic_ref<int, cuda::thread_scope_device> r(consumed[s * stride]);
				while (r.load(cuda::memory_order_relaxed) < prod_expected[s]) {}
			}
			__syncwarp();
			__threadfence();
		}
		prod_expected[s] += NumConsumers;
	}

	// Signal that data has been written to this slot.
	__device__ void producer_release(int lane) {
		int s = producer_stage();
		// Device-scope: this is the get-side publish (NIC-written src → MLP) or
		// a slot-free signal — both GPU-internal w.r.t. the consumer. The NIC's
		// inbound-write visibility is NOT fixable by a fence (needs GDRCopy), so
		// system scope here only costs bandwidth without buying correctness. The
		// one publish that genuinely crosses to the NIC (dst → put warp) keeps
		// system scope in RemoteIter::release_dst.
		__threadfence();
		if (lane == 0) {
			atomicAdd(&ready[s * stride], 1);
		}
		prod_count++;
	}

	// ── Consumer side ───────────────────────────────────────

	__device__ int consumer_stage() const {
		return cons_count % NumStages;
	}

	// Wait until this slot has data (producer has released it).
	//
	// Acquire-side memory ordering: see comment on producer_acquire above.
	// __syncwarp alone does NOT establish global-memory visibility for
	// non-leader lanes — they may load stale data from L1/L2 even after
	// the leader has confirmed the producer's release. The __threadfence
	// after __syncwarp gives every lane in the warp a fresh view of the
	// producer's writes (HBM coherence) before the warp-collective load
	// in the subsequent put.
	__device__ void consumer_acquire() {
		int s = consumer_stage();
		if (is_leader) {
			cuda::atomic_ref<int, cuda::thread_scope_device> r(ready[s * stride]);
			while (r.load(cuda::memory_order_relaxed) < cons_expected[s] + NumProducers) {}
		}
		__syncwarp();
		// Device-scope: the put warp observes dst_staging (Y), already flushed
		// to HBM by the MLP's system-scope release_dst. Observe is GPU-internal.
		__threadfence();
		cons_expected[s] += NumProducers;
	}

	// Signal that this slot has been consumed and is free.
	__device__ void consumer_release(int lane) {
		int s = consumer_stage();
		__threadfence();
		if (lane == 0) {
			atomicAdd(&consumed[s * stride], 1);
		}
		cons_count++;
	}
};

// ═══════════════════════════════════════════════════════════════════
// CommSmem — per-CTA shared memory (smem-local, not shared across CTAs)
// ═══════════════════════════════════════════════════════════════════

struct CommSmem {
	// Total number of remote tiles across all CTAs. Initialized to -1.
	// Comm warps write the actual count; consumers wait for >= 0.
	int total_tiles;

	// Per-CTA tile count (this blockIdx.x's share of total_tiles).
	int per_cta_tiles;

	// Grid-wide remote tile count (sum across all CTAs/columns). Same
	// value on every CTA in the grid; used by moe_fused_bwd to gate the
	// remote phase consistently — the inner mlp_fused_bwd's cross-CTA
	// mlp_global_barrier requires every CTA to make the same enter/skip
	// decision, so per-CTA per_cta_tiles can't be the gate.
	int global_total;

	// Pre-fetched remote expert offsets, assigned to dynamic smem at runtime.
	// Layout: [num_pes][experts_per_pe + 1]
	int* remote_offsets;
};

// ═══════════════════════════════════════════════════════════════════
// PeSync — device-level phase synchronization with monotonic counters
// ═══════════════════════════════════════════════════════════════════
//
// Single monotonic CTA counter with modular check for cross-CTA barriers.
// Two remote signal arrays (dispatch_done, mlp_done) for PE-level sync.
//
// Protocol per kernel call:
//   1. barrier(): after sort — cross-CTA visibility of sorted_token_ids
//   2. signal_dispatch(): barrier + signal dispatch_done to remote PEs
//   3. wait_dispatch(): wait for remote dispatch_done signals
//   4. signal_mlp(): barrier + signal mlp_done to remote PEs
//   5. wait_mlp(): wait for remote mlp_done signals
//   6. advance_iteration_counter() at the very end of the kernel —
//      CTA 0 thread 0 atomically increments the global iteration counter.
//
// The iteration counter lives in device memory (allocated from the
// buffer pool). Every CTA reads it via signal_phase / wait_phase;
// exactly one CTA increments it at the very end. The cross-CTA
// barrier()s upstream of the end ensure no CTA is still reading
// `iter_counter` when the atomicAdd fires. The motivation for moving
// the counter from a host atomic to GPU memory is CUDA graph
// capture: a captured kernel replays the SAME launch on every replay,
// so a host-set iteration value would be baked in and never increase
// across replays — breaking the wait-until-greater-or-equal protocol.
// A GPU-resident counter that the kernel itself atomicAdds at the end
// gets re-evaluated on every replay, producing a fresh monotonic
// target each time.
//
// Host calls nvshmemx_barrier_on_stream(team, stream) after each
// kernel for safety.

struct PeSync {
	uint64_t* dispatch_done;     // [num_pes] symmetric — monotonic
	uint64_t* mlp_done;          // [num_pes] symmetric — monotonic
	int* cta_counter;            // [1] device memory — shared monotonic counter
	uint64_t* iter_counter;      // [1] device memory — global iteration counter
	int num_pes;
	int my_pe;
	nvshmem_team_t team;

	// ── barrier: cross-CTA global memory fence ──────────
	// Defers to a persistent SyncThreadsCtaCounterBarrier owned by the
	// caller — see cta_barrier.cuh for why the barrier object must be
	// constructed ONCE per kernel and reused (target accumulates).
	// PeSync cannot own the barrier itself because it lives inside a
	// __grid_constant__ MoeParams; the caller builds the barrier on
	// the kernel stack and threads it into every signal/wait call.
	__device__ void barrier(SyncThreadsCtaCounterBarrier& b) const {
		b.wait();
	}

	// Load the current iteration value. All callers must see the same
	// value within a single kernel — guaranteed because
	// advance_iteration_counter() runs only at the very end, after the
	// barrier()s that enclose every signal/wait phase.
	__device__ uint64_t iteration() const {
		return *iter_counter;
	}

	// ── signal_phase: barrier + remote PE signal ────────
	// Cross-CTA barrier, then CTA 0 signals remote PEs.
	__device__ void signal_phase(SyncThreadsCtaCounterBarrier& b,
	                             uint64_t* signal) const {
		b.wait();

		// CTA 0: signal remote PEs.
		int cta_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (cta_id == 0 && threadIdx.x == 0) {
			uint64_t target = iteration() + 1;
			for (int pe_ = 1; pe_ < num_pes; ++pe_) {
				// Route through g_dest_table to follow the canonical comm
				// schedule; constant-cache lookup is broadcast-friendly.
				int team_pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + pe_) % num_pes];
				int global_pe = nvshmem_team_translate_pe(team, team_pe, NVSHMEM_TEAM_WORLD);
				nvshmemx_signal_op(&signal[my_pe], target,
					NVSHMEM_SIGNAL_SET, global_pe);
			}
			signal[my_pe] = target;
		}
		__syncthreads();
	}

	// ── wait_phase: wait for all remote PEs' signal ─────
	__device__ void wait_phase(uint64_t* signal) const {
		uint64_t target = iteration() + 1;
		if (threadIdx.x == 0) {
			for (int pe_ = 1; pe_ < num_pes; ++pe_) {
				int pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + pe_) % num_pes];
				nvshmem_uint64_wait_until(&signal[pe], NVSHMEM_CMP_GE, target);
			}
		}
		__syncthreads();
	}

	// ── advance_iteration_counter: end-of-kernel atomic bump ──────────
	// Called once at the very end of the kernel by CTA 0 thread 0 only.
	// Increments the global iteration counter so the NEXT kernel launch
	// (or graph replay) reads a fresh value. Cross-CTA barriers earlier
	// in the kernel guarantee every CTA has finished using iteration()
	// before this fires.
	__device__ void advance_iteration_counter() const {
		int cta_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (cta_id == 0 && threadIdx.x == 0) {
			atomicAdd(reinterpret_cast<unsigned long long*>(iter_counter), 1ULL);
		}
	}

	// ── Convenience wrappers ────────────────────────────

	__device__ void signal_dispatch(SyncThreadsCtaCounterBarrier& b) const {
		signal_phase(b, dispatch_done);
	}

	__device__ void wait_dispatch() const {
		wait_phase(dispatch_done);
	}

	__device__ void signal_mlp(SyncThreadsCtaCounterBarrier& b) const {
		signal_phase(b, mlp_done);
	}

	__device__ void wait_mlp() const {
		wait_phase(mlp_done);
	}
};

// ═══════════════════════════════════════════════════════════════════
// CommBuffers — pointers passed to the kernel
// ═══════════════════════════════════════════════════════════════════

struct CommBuffers {
	// Staging areas: per-blockIdx.x, shared by both blockIdx.y CTAs.
	// Layout: [gridDim.x][NumStages][TileM][dim]
	void* src_staging;              // [gridDim.x * S * TileM * hidden_dim]
	void* dst_staging;              // [gridDim.x * S * TileM * intermediate_dim]

	// Stage pipe signals in device memory (HBM), per-blockIdx.x.
	// Layout: [gridDim.x][NumStages] — shared by both blockIdx.y CTAs.
	int* src_ready;                 // nvshmem → consumer: tile data written
	int* src_consumed;              // consumer → nvshmem: tile data read
	int* dst_ready;                 // consumer → nvshmem: tile data written
	int* dst_consumed;              // nvshmem → consumer: put completed, slot free

	const int* expert_offsets;      // [E + 1] local (from radix sort, TileM-aligned, symmetric)
	const int* sorted_token_ids;    // [total_slots] local sorted token IDs

	void* local_tokens;            // [T, hidden_dim] local input (symmetric)
	void* local_output;            // [total_slots, intermediate_dim] local output (symmetric)

	// Per-staging-slot expert IDs in HBM. Written by do_get, read by MLP.
	// Layout: [gridDim.x][NumStages]
	int* tile_expert_ids;

	// All PEs' expert offsets, gathered into local symmetric memory.
	// Layout: [num_pes][num_experts + 1] — symmetric.
	int* all_expert_offsets;

	// Cross-PE synchronization for offset broadcast/consume.
	PeSync pe_sync;

	int hidden_dim;
	int num_experts;          // total global experts
	int experts_per_pe;       // num_experts / num_pes (evenly split)
	int all_expert_offsets_stride;  // max_num_experts + 1 (fixed across configs)
	// GEMM/comm-active CTA count = floor_NS (largest multiple of NSplit <=
	// num_sms). The flat-grid launch may run more CTAs than this (grid-stride
	// phases use them); GEMM and comm gate to the first n_gemm CTAs so the
	// comm↔MLP staging ring geometry (grid_x = n_gemm/NSplit, MC = n_gemm/NC)
	// stays consistent. Replaces the old gridDim.x·gridDim.y product.
	int n_gemm;

	// Convenience accessors for fields stored in pe_sync.
	__host__ __device__ __forceinline__ int my_pe()  const { return pe_sync.my_pe; }
	__host__ __device__ __forceinline__ int num_pes() const { return pe_sync.num_pes; }
	__host__ __device__ __forceinline__ nvshmem_team_t team() const { return pe_sync.team; }
};

// ═══════════════════════════════════════════════════════════════════
// TileInfo — describes one tile to get or put
// ═══════════════════════════════════════════════════════════════════

struct TileInfo {
	int pe;              // remote PE
	int token_offset;    // offset into remote PE's token buffer (in tokens)
	int expert;          // local expert index (for weight selection)
};

// ═══════════════════════════════════════════════════════════════════
// TileIterator — enumerates tiles: expert-outer, PE-inner, strided
// ═══════════════════════════════════════════════════════════════════
//
// Tiles within each (expert, pe) pair are strided by gridDim.x.
// Each blockIdx.x starts at its own offset and advances by gridDim.x.

template <int TileM_>
struct TileIterator {
	static constexpr int TileM = TileM_;

	const int* remote_offsets;  // smem: [num_pes][experts_per_pe + 1]
	int offsets_stride;         // experts_per_pe + 1
	int experts_per_pe;
	int num_pes;
	int my_pe;

	int stride;         // distance between consecutive tiles

	// Current position in the iteration.
	int le;             // local expert index
	int pe_idx;         // 1..num_pes-1 (skips my_pe)
	int tile;           // tile within current (expert, pe) pair
	int remote_tiles;   // total tiles for current (expert, pe) pair
	int remote_start;   // start offset for current (expert, pe)

	int total_tiles;    // this iterator's tile count

	__device__ void init(int start, int str) {
		stride = str;
		le = 0;
		pe_idx = 0;
		total_tiles = 0;

		// Count total tiles across all (expert, pe) pairs and
		// compute this iterator's share.
		// NOTE: p starts at 0 → the LOCAL PE (my_pe, via g_dest_table slot
		// offset 0) is enumerated first for each expert, alongside the remote
		// PEs (p=1..num_pes-1). Unified path: local experts are staged by the
		// comm get warp from local symmetric memory just like remote tiles, so
		// the separate local MLP pass is gone (see moe.cuh moe_fused_fwd).
		int global_total = 0;
		for (int e = 0; e < experts_per_pe; ++e) {
			for (int p = 0; p < num_pes; ++p) {
				int pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + p) % num_pes];
				int rs = remote_offsets[pe * offsets_stride + e];
				int re = remote_offsets[pe * offsets_stride + e + 1];
				global_total += (re - rs) / TileM;
			}
		}
		for (int t = start; t < global_total; t += stride)
			total_tiles++;

		// Find the starting position: walk the flat tile space
		// to find where start lands.
		int skip = start;
		for (int e = 0; e < experts_per_pe; ++e) {
			for (int p = 0; p < num_pes; ++p) {
				int pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + p) % num_pes];
				int rs = remote_offsets[pe * offsets_stride + e];
				int re = remote_offsets[pe * offsets_stride + e + 1];
				int ntiles = (re - rs) / TileM;
				if (skip < ntiles) {
					le = e;
					pe_idx = p;
					tile = skip;
					remote_start = rs;
					remote_tiles = ntiles;
					return;
				}
				skip -= ntiles;
			}
		}
		// No tiles for this iterator.
		le = experts_per_pe;
	}

	__device__ bool has_next() const {
		return le < experts_per_pe;
	}

	__device__ TileInfo next() {
		int pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + pe_idx) % num_pes];
		TileInfo info;
		info.pe = pe;
		info.token_offset = remote_start + tile * TileM;
		info.expert = le;

		// Advance by stride in the flat tile space.
		// Carry the remainder across (expert, pe) boundaries.
		tile += stride;
		while (tile >= remote_tiles && le < experts_per_pe) {
			int remaining = tile - remote_tiles;
			pe_idx++;
			if (pe_idx >= num_pes) {
				pe_idx = 0;
				le++;
			}
			if (le >= experts_per_pe) break;
			int next_pe = liger::g_dest_table[(liger::comm_slot_of(my_pe) + pe_idx) % num_pes];
			remote_start = remote_offsets[next_pe * offsets_stride + le];
			int remote_end = remote_offsets[next_pe * offsets_stride + le + 1];
			remote_tiles = (remote_end - remote_start) / TileM;
			tile = remaining;
		}
		return info;
	}
};

// ═══════════════════════════════════════════════════════════════════
// NVSHMEM Comm device functions — dedicated get/put warps
// ═══════════════════════════════════════════════════════════════════
//
// Per CTA: warp 2 is the getter, warp 3 is the putter.
//
// The physical grid is launched as (gridDim.x, N_SPLIT) to match the
// GEMM's cooperative-WG layout. Comms re-indexes the flat CTA id into
// a (MC, NC) logical grid where NC must divide N_SPLIT:
//
//   cta_id = blockIdx.x * gridDim.y + blockIdx.y
//   xc     = cta_id / NC            in [0, MC)
//   yc     = cta_id % NC            in [0, NC)    — chunk index
//   MC     = gridDim.x * (gridDim.y / NC)
//
// NC CTAs (same xc, different yc) cooperate on one tile and chunk its
// bytes NC ways. MC groups stride through the remote-tile list, so
// more tiles are in flight per kernel launch when NC < N_SPLIT.
//
// A CTA's comm tile is chosen by xc; its GEMM tile is chosen by
// blockIdx.x. These can diverge when NC < N_SPLIT (the whole point of
// the re-indexing) — the staging ring is keyed by xc so MLP and comm
// rendezvous through the ring, not through shared CTA coordinates.
//
// The two roles run independent flat loops over all tiles. No 1F1B —
// the staging ring (NumStages) bounds how far the getter can lead
// the putter via the shared src/dst StagePipe signals.

// Prologue + bwd use warps 2-3 only (kept for backward compatibility
// with the bwd path that has not been migrated to the 2-getter layout).
static constexpr int kCommWarpStart    = 2;
static constexpr int kCommWarpEnd      = 3;
static constexpr int kCommWarpsPerCta  = kCommWarpEnd - kCommWarpStart + 1;  // 2
static constexpr int kCommThreadStart  = kCommWarpStart * 32;

// Fwd comm layout: 2 get warps + 1 put warp per CTA.
//   warp 1 → get warp 0
//   warp 2 → get warp 1 (also prologue leader)
//   warp 3 → put warp
static constexpr int kCommGetWarp0     = 1;
static constexpr int kCommGetWarp1     = 2;
static constexpr int kCommPutWarp      = 3;
static constexpr int kNumGetWarpsPerCta = 2;  // shared w/ bwd — do NOT change

// FWD-only active get-warp count. With the TMA GET (DMA engine, issue-bound
// not thread-bound), ONE warp saturates the per-SM TMA unit, so warp 2 is
// freed (idle in steady state; still helps the epilogue last-tile put drain).
// kNumGetWarpsPerCta stays 2 for the bwd path and for the 2-warp epilogue
// drain (kPutW); only the steady-state FWD producer count drops to this.
static constexpr int kNumGetWarpsFwd = 1;

// FWD-only put-warp count. The get warp freed by the TMA path (above) joins
// the put side: warps 2+3 are two symmetric put warps. putmem is warp-LD/ST
// (thread-bound) so 2 warps ≈ 2× put BW. Each puts 1/(kNumPutWarpsFwd·NC) of
// a tile and drives the dst pipe; RemoteMlpTileIterator's NumConsumersDst and
// the dst StagePipe NumConsumers both follow this.
static constexpr int kNumPutWarpsFwd = 2;

// Range of warps participating in fwd comm (used by callers to filter
// MLP vs comm warps).
static constexpr int kCommFwdWarpStart = kCommGetWarp0;
static constexpr int kCommFwdWarpEnd   = kCommPutWarp;

// Get side has kNumGetWarpsPerCta × NC warps per tile (chunk =
// tile_bytes / (kNumGetWarpsPerCta × NC)); put side has NC warps per
// tile (chunk = tile_bytes / NC).

// ═══════════════════════════════════════════════════════════════════
// GET-via-TMA: fetch X over NVLink with the TMA engine, not warp LD/ST.
// ═══════════════════════════════════════════════════════════════════
//
// The dedicated get warp drives a TMA load (peer X → smem) followed by a
// TMA store (smem → local src_staging HBM). TMA can't go global→global on
// Hopper, so the bounce through smem is mandatory; the store TARGET is
// LOCAL HBM, so completion (cp.async.bulk.wait_group) + a device-scope
// __threadfence in producer_release makes it visible to the MLP reader —
// the safe-completion path (unlike a store to a PEER, which failed before).
//
// Validated in microbench: ~370 GB/s, in- and out-of-kernel correct.
//
// kGetKChunk is the contiguous column box width (elements). Each of the
// kNumGetWarpsPerCta·NC warps owns a TileM/(kNumGetWarpsPerCta·NC) row
// band of the tile and loops over hidden_dim/kGetKChunk column boxes.
// Box = [kGetBoxRows, kGetKChunk]; descriptors built host-side in moe.cu.
static constexpr int kGetKChunk = 256;

// Per-peer source descriptors + one local-staging dst descriptor, built on
// the host (nvshmem_ptr(x_sorted, peer)) and passed __grid_constant__.
// FWD-only: NOT referenced by bwd — kept out of CommBuffers so the shared
// struct's ABI is unchanged and moe_bwd.cu need not recompile.
//
// Multi-host: only SAME-HOST peers are P2P-reachable, so src_x_desc is indexed
// by intra-host local rank (pe % gpus_per_node), NOT by global team PE — the
// array therefore only ever needs gpus_per_node (≤ kGetMaxPes) entries
// regardless of total PE count. Remote (IB) peers fall back to getmem and never
// touch this array. Assumes ranks are host-contiguous and host-aligned to
// gpus_per_node (true for a WORLD-contiguous team); asserted host-side.
static constexpr int kGetMaxPes = 8;   // upper bound on GPUs per host
struct GetTmaDescs {
	CUtensorMap src_x_desc[kGetMaxPes];  // index by (TileInfo::pe % gpus_per_node)
	CUtensorMap dst_staging_desc;        // local src_staging
	int  enabled;                        // 1 = NVLink P2P TMA path available; 0 = getmem
	int  gpus_per_node;                  // local host PE count — host/local-rank divisor
	int  my_pe;                          // this PE in team space (for same-host test)
};

// Same-host predicate (pure arithmetic): true iff peer `pe` shares this PE's
// host. Used both to gate the TMA get path and to pick the release_dst fence
// scope (device for same-host NVLink, system for cross-host IB).
__host__ __device__ __forceinline__ bool get_pe_is_local(
		const GetTmaDescs* descs, int pe) {
	if (descs == nullptr || descs->gpus_per_node <= 0) return true;  // single-host/test
	return (pe / descs->gpus_per_node) == (descs->my_pe / descs->gpus_per_node);
}

// Per-peer Y-store descriptors for the direct GEMM→peer TMA store (FWD-only).
// desc[r] is a CUTE TMA-store object built host-side over
// nvshmem_ptr(local_output, peer) for the same-host peer at intra-host local
// rank r (self included, rank = my_pe % gpus_per_node). When a tile is_local,
// MLP2 stores Y straight into the destination peer's symmetric local_output via
// desc[MlpTileInfo::peer_rank] instead of dst_staging — the put warp then only
// rubber-stamps. The direct store is gated per-tile on is_local (same-host →
// P2P guaranteed), so no runtime enable flag is needed. Templated on the CUTE
// store type so the same object the wrapper passes as tma_store_y populates it.
template <typename TmaStoreY>
struct PeerYStoreDescs {
	TmaStoreY desc[kGetMaxPes];  // index by (TileInfo::pe % gpus_per_node)
};

// Smem bounce: per get-warp single box + its mbarrier, 128-aligned so the
// TMA dst smem address is aligned. Stride between the two get warps' regions.
template <typename Element, int TileM, int NC>
struct GetBounce {
	static constexpr int kWarpsPerTile = kNumGetWarpsFwd * NC;
	static constexpr int kBoxRows      = TileM / kWarpsPerTile;
	static constexpr int kBoxElems     = kBoxRows * kGetKChunk;
	static constexpr int kBoxBytes     = kBoxElems * (int)sizeof(Element);
	// [box (128-aligned)] [mbar:8B], next warp's region 128-aligned.
	static constexpr int kPerWarpStride = ((kBoxBytes + 8) + 127) & ~127;
	static constexpr int kTotalBytes    = kPerWarpStride * kNumGetWarpsFwd;
};

template <typename Element, int TileM, int K, int NC, int N_SPLIT>
__device__ __forceinline__ void do_get(
		StagePipe<K, kNumGetWarpsFwd * NC, N_SPLIT>& src_pipe,
		const CommBuffers& bufs,
		const TileInfo& info,
		Element* src_staging,  // base of flat staging buffer
		int src_tile_elems,
		int xc,                // start slot in flat list
		int MC,                // stride between consecutive comm stages
		int chunk_idx,         // 0..kNumGetWarpsPerCta * NC - 1
		int& ib_seq,           // IB-tile counter (round-robin turn; advanced here)
		int lane,
		const GetTmaDescs* descs,  // per-peer src + local dst TMA descriptors (nullptr → getmem)
		char* bounce_warp) {       // this warp's GetBounce sub-region (128-aligned)

	using Bounce = GetBounce<Element, TileM, NC>;
	constexpr int kWarpsPerTile = Bounce::kWarpsPerTile;
	constexpr int kBoxRows      = Bounce::kBoxRows;

	src_pipe.producer_acquire();

	// No per-tile signal wait — wait_dispatch() already confirmed
	// all dispatch writes on all PEs are complete.

	int slot = xc + src_pipe.producer_stage() * MC;

	// Locality of this tile's source PE (same-host NVLink vs cross-host IB).
	// Pure arithmetic; gates the TMA get path below. The MLP's release_dst
	// recomputes the SAME predicate independently (embedded TileIterator), so no
	// per-slot flag is published here.
	bool is_local = get_pe_is_local(descs, info.pe);

	// TMA path: peer X → smem → local src_staging, via the TMA engine.
	// Gated on same-host NVLink P2P (is_local), TMA availability (descs->enabled),
	// and hidden_dim divisibility; cross-host IB, odd shapes, or a null descs
	// (e.g. the standalone test) fall back to getmem.
	bool tma_ok = descs && descs->enabled && is_local
		&& (bufs.hidden_dim % kGetKChunk == 0);
	if (tma_ok) {
		// Typed TMA / mbarrier intrinsics from <cuda/ptx>. These compile to the
		// same SASS as the hand-written PTX they replace: the peer-X→smem load
		// is the exact `cp.async.bulk.tensor.2d.shared::cluster.global.tile`
		// instruction, and the smem→staging store the `.global.shared::cta.tile`
		// one (`.tile` is the default tiled mode → identical encoding).
		namespace ptx = cuda::ptx;
		Element*  sbuf = reinterpret_cast<Element*>(bounce_warp);
		uint64_t* mbar = reinterpret_cast<uint64_t*>(bounce_warp + Bounce::kBoxBytes);

		if (lane == 0)
			ptx::mbarrier_init(mbar, 1u);
		__syncwarp();
		// Make the barrier init visible to the async (TMA) proxy before use.
		ptx::fence_proxy_async(ptx::space_shared);

		// Tile coords: each warp owns its kBoxRows-tall row band.
		int y_src = info.token_offset + chunk_idx * kBoxRows;  // peer X row
		int y_dst = slot * TileM       + chunk_idx * kBoxRows;  // staging row
		// Same-host (tma_ok ⇒ is_local) → index by intra-host local rank so the
		// descriptor array stays gpus_per_node-sized regardless of total PEs.
		const CUtensorMap* src_map = &descs->src_x_desc[info.pe % descs->gpus_per_node];
		const CUtensorMap* dst_map = &descs->dst_staging_desc;

		const int n_k = bufs.hidden_dim / kGetKChunk;
		const uint32_t txbytes = (uint32_t)Bounce::kBoxBytes;
		uint32_t phase = 0;
		for (int k = 0; k < n_k; ++k) {
			int x = k * kGetKChunk;
			if (lane == 0) {
				// Arm the barrier for the incoming box, then issue the
				// peer-X → smem TMA load it tracks.
				(void)ptx::mbarrier_arrive_expect_tx(
					ptx::sem_release, ptx::scope_cta, ptx::space_shared,
					mbar, txbytes);
				int32_t load_crd[2] = {x, y_src};
				ptx::cp_async_bulk_tensor(
					ptx::space_cluster, ptx::space_global,
					sbuf, src_map, load_crd, mbar);
				// Spin until the load completes (single-buffer bounce).
				while (!ptx::mbarrier_try_wait_parity(mbar, phase)) {}
				// smem now holds the loaded box; push it to local staging.
				int32_t store_crd[2] = {x, y_dst};
				ptx::cp_async_bulk_tensor(
					ptx::space_global, ptx::space_shared,
					dst_map, store_crd, sbuf);
				ptx::cp_async_bulk_commit_group();
				// Single-buffer: drain the store before the next load overwrites smem.
				ptx::cp_async_bulk_wait_group(ptx::n32_t<0>{});
			}
			phase ^= 1;
			__syncwarp();
		}

		if (bufs.tile_expert_ids && chunk_idx == 0 && lane == 0)
			bufs.tile_expert_ids[slot] = info.expert;

		// producer_release's __threadfence (device scope) orders the completed
		// local TMA stores before the ready signal — MLP reader sees fresh X.
		__syncwarp();
		src_pipe.producer_release(lane);
		return;
	}

	// ── Fallback: warp getmem (IB, or hidden_dim not divisible by kGetKChunk) ──
	int chunk = src_tile_elems / kWarpsPerTile;
	int offset = chunk_idx * chunk;
	auto* local_tokens = static_cast<Element*>(bufs.local_tokens);
	Element* remote_base = local_tokens + info.token_offset * bufs.hidden_dim;
	Element* local_base = src_staging + slot * src_tile_elems;
	int global_pe = nvshmem_team_translate_pe(bufs.team(), info.pe, NVSHMEM_TEAM_WORLD);
	nvshmemx_getmem_warp(local_base + offset, remote_base + offset,
		(size_t)chunk * sizeof(Element), global_pe);
	if (bufs.tile_expert_ids && chunk_idx == 0 && lane == 0) {
		bufs.tile_expert_ids[slot] = info.expert;
	}
	src_pipe.producer_release(lane);
}

template <typename Element, int TileM, int K, int NC, int N_SPLIT>
__device__ __forceinline__ void do_put(
		StagePipe<K, N_SPLIT, kNumPutWarpsFwd * NC>& dst_pipe,
		const CommBuffers& bufs,
		const TileInfo& info,
		Element* dst_staging,
		int dst_tile_elems,
		int xc,                // start slot in flat list
		int MC,                // stride between consecutive comm stages
		int yc,                // 0..NC-1 — CTA's chunk index within the tile
		int& ib_seq,           // IB-tile counter (round-robin turn; advanced here)
		int lane,
		int put_local_idx,     // 0..kNumPutWarpsFwd-1 — put warp within the CTA
		const GetTmaDescs* descs) {  // null → standalone test (real put); else skip local

	// Two put warps per CTA: split the tile kNumPutWarpsFwd·NC ways, this
	// warp owns slice (yc·kNumPutWarpsFwd + put_local_idx).
	constexpr int kWarpsPerTile = kNumPutWarpsFwd * NC;
	int chunk = dst_tile_elems / kWarpsPerTile;
	int offset = (yc * kNumPutWarpsFwd + put_local_idx) * chunk;
	int my_bytes = chunk * sizeof(Element);

	dst_pipe.consumer_acquire();

	// Direct-store: for same-host (NVLink) tiles the GEMM has already TMA-stored
	// Y straight into the peer's symmetric local_output, so this put warp must
	// NOT copy dst_staging over it (that would clobber the correct Y with a stale
	// staging slot). It only RUBBER-STAMPS the dst pipe so the StagePipe counters
	// stay consistent and the GEMM's acquire_dst never stalls. Cross-host (IB)
	// tiles still need the real putmem. descs==nullptr is the standalone comm
	// test (no direct store) → always do the real put.
	bool skip_put = (descs != nullptr) && get_pe_is_local(descs, info.pe);
	if (!skip_put) {
		auto* local_output = static_cast<Element*>(bufs.local_output);
		int slot = xc + dst_pipe.consumer_stage() * MC;
		Element* local_base = dst_staging + slot * dst_tile_elems;
		Element* remote_base = local_output + info.token_offset * bufs.hidden_dim;
		int global_pe = nvshmem_team_translate_pe(bufs.team(), info.pe, NVSHMEM_TEAM_WORLD);
		nvshmemx_putmem_warp(remote_base + offset, local_base + offset,
			my_bytes, global_pe);
	}

	dst_pipe.consumer_release(lane);
}

// Chunk-only put for the cooperative LAST-tile drain (epilogue get-warp put
// swap). The dedicated put warp owns the pipe (acquire/release); each of the
// PutW comm warps just writes its share — NO pipe ops here, so no extra
// dst_ready/dst_consumed signalling. P2P splits the tile PutW*NC ways; IB
// sends the whole tile from rank 0 only. dst_tile_elems must be divisible by
// PutW*NC (TileM*D divisible by 2*NC holds for the supported shapes).
template <typename Element, int PutW, int NC>
__device__ __forceinline__ void do_put_chunk(
		const CommBuffers& bufs, const TileInfo& info,
		Element* dst_staging, int dst_tile_elems,
		int slot, int yc, int put_rank, int lane) {
	constexpr int kWarpsPerTile = PutW * NC;
	int put_idx = yc * PutW + put_rank;
	int chunk   = dst_tile_elems / kWarpsPerTile;
	int offset  = put_idx * chunk;
	Element* local_base  = dst_staging + (size_t)slot * dst_tile_elems;
	Element* remote_base = static_cast<Element*>(bufs.local_output)
	                     + (size_t)info.token_offset * bufs.hidden_dim;
	int global_pe = nvshmem_team_translate_pe(bufs.team(), info.pe, NVSHMEM_TEAM_WORLD);
	// nbi: the last tile's put has no downstream consumer waiting on it, so it
	// need not block — completion is guaranteed by the nvshmem_quiet() each
	// comm warp issues after comm_main returns (moe.cu). Single per-chunk put,
	// no transport branch (NVLink-only build) → one call site.
	nvshmemx_putmem_nbi_warp(remote_base + offset, local_base + offset,
		(size_t)chunk * sizeof(Element), global_pe);
}

// ═══════════════════════════════════════════════════════════════════
// broadcast_expert_offsets — called after sort, before MLP phase.
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ void broadcast_expert_offsets(
		const CommBuffers& bufs) {

	int cta_id = blockIdx.x + blockIdx.y * gridDim.x;
	int num_experts    = bufs.num_experts;
	int num_pes        = bufs.num_pes();
	int my_pe          = bufs.my_pe();
	int offsets_stride  = num_experts + 1;
	const int* my_offsets = bufs.expert_offsets;

	// CTA 0: copy into own local all_expert_offsets[my_pe][:].
	if (cta_id == 0 && threadIdx.x == 0) {
		int* local_dst = bufs.all_expert_offsets + my_pe * offsets_stride;
		for (int i = 0; i <= num_experts; ++i)
			local_dst[i] = my_offsets[i];
	}

	// CTA i puts offsets to PE i (if valid remote PE).
	// No signal here — signal_dispatch() after dispatch covers it.
	// NVSHMEM ordering: when dispatch signal arrives, this put is visible.
	int target_pe = cta_id;
	if (target_pe >= num_pes || target_pe == my_pe) return;

	int global_target = nvshmem_team_translate_pe(bufs.team(), target_pe, NVSHMEM_TEAM_WORLD);
	int* remote_dst = bufs.all_expert_offsets + my_pe * offsets_stride;
	nvshmemx_int_put_nbi_block(
		remote_dst, my_offsets,
		offsets_stride,
		global_target);
}

// ═══════════════════════════════════════════════════════════════════
// Prologue: waits for remote offsets, computes total_tiles.
// ═══════════════════════════════════════════════════════════════════
//
// Called by all threads — comm warps do the work, others pass through.
// Must be followed by __syncthreads() before nvshmem_comm_main.

template <int TileM, int NC = 2, int N_SPLIT = NC>
__device__ __forceinline__ void nvshmem_comm_prologue(
		CommSmem& smem,
		const CommBuffers& bufs) {

	int warp_id = threadIdx.x / 32;
	if (warp_id < kCommWarpStart || warp_id > kCommWarpEnd)
		return;

	int lane = threadIdx.x % 32;
	int local_warp = warp_id - kCommWarpStart;
	bool is_leader = (local_warp == 0 && lane == 0);

	int experts_per_pe = bufs.experts_per_pe;
	int num_pes        = bufs.num_pes();
	int my_pe          = bufs.my_pe();

	if (bufs.all_expert_offsets == nullptr) {
		// Standalone test (no nvshmem offsets) — no tiles.
		if (is_leader) {
			smem.total_tiles = 0;
			smem.per_cta_tiles = 0;
		}
		__syncwarp();
		return;
	}
	// NOTE: num_pes == 1 is NO LONGER short-circuited. The unified TileIterator
	// enumerates the LOCAL PE (p=0), so a single-GPU run stages its local tiles
	// through the comm get/put path exactly like the multi-GPU remote path —
	// this is the apples-to-apples isolation of the per-tile staging cost.

	// wait_dispatch() was called before entering the MLP phase,
	// guaranteeing all remote PEs' offsets are visible.

	// Extract this PE's local expert offsets from each remote PE's full array.
	// all_expert_offsets layout: [num_pes][num_experts + 1].
	// smem layout: [num_pes][experts_per_pe + 1] — only this PE's experts.
	int local_e_start = my_pe * experts_per_pe;
	int src_stride = bufs.num_experts + 1;
	int dst_stride = experts_per_pe + 1;
	{
		int tid = threadIdx.x - kCommThreadStart;
		for (int pe = 0; pe < num_pes; ++pe) {
			for (int i = tid; i <= experts_per_pe; i += (kCommWarpsPerCta * 32))
				smem.remote_offsets[pe * dst_stride + i] =
					bufs.all_expert_offsets[pe * src_stride + local_e_start + i];
		}
	}
	__syncwarp();

	// Re-index the flat cta_id to the comm (xc, yc) grid. Comm walks
	// tiles owned by (xc, yc) with stride MC. MLP walks tiles for this
	// blockIdx.x column with stride gridDim.x — a different start AND
	// a different stride, so the two counts must be computed separately
	// (uneven distribution can make them differ by more than the ratio
	// of the strides per CTA).
	// Flat-grid launch: blockIdx.x is the flat cta_id; the logical comm/MLP
	// grid comes from n_gemm + compile-time NC/N_SPLIT, not gridDim.
	//   comm:  xc = cta_id / NC,            MC     = n_gemm / NC
	//   MLP:   col = cta_id / N_SPLIT,       grid_x = n_gemm / N_SPLIT
	int cta_id = blockIdx.x;
	// Comm grid is NC-aligned and DECOUPLED from the GEMM's NS-aligned n_gemm:
	// recompute the comm count n_comm from the launched grid (NOT n_gemm). The
	// comm side uses all NC-complete CTAs; the leftover remainder (xc >= MC)
	// gets comm_count == 0 and self-gates. The MLP count below keeps n_gemm.
	int n_comm = (gridDim.x / NC) * NC;
	int xc = cta_id / NC;
	int MC = n_comm / NC;

	TileIterator<TileM> iter;
	iter.remote_offsets = smem.remote_offsets;
	iter.offsets_stride = dst_stride;
	iter.experts_per_pe = experts_per_pe;
	iter.num_pes = num_pes;
	iter.my_pe = my_pe;

	// Comm count: this CTA's per-(xc, yc) workload.
	iter.init(xc, MC);
	int comm_count = iter.total_tiles;

	// MLP count: this column's workload, stride grid_x.
	iter.init(cta_id / N_SPLIT, bufs.n_gemm / N_SPLIT);
	int mlp_count = iter.total_tiles;

	if (is_leader) {
		smem.total_tiles   = comm_count;  // read by comm_main
		smem.per_cta_tiles = mlp_count;   // read by MLP iterator
	}
	__syncwarp();
}

// Main loop: warps 1, 2 (getters) and warp 3 (putter) each run an
// independent flat loop over the same tile sequence. Called after
// __syncthreads().
//
// CTA re-indexing:
//   cta_id = blockIdx.x * gridDim.y + blockIdx.y
//   xc     = cta_id / NC                  (tile-stream index in [0, MC))
//   yc     = cta_id % NC                  (CTA's slot within its xc group)
//   MC     = (gridDim.x * gridDim.y) / NC (multiply before divide so MC is
//                                          exact when NC ∤ gridDim.y)
//
// Each xc group strides through the remote-tile list by MC; all NC
// CTAs with the same xc cooperate on the same tile. Get side splits
// each tile kNumGetWarpsPerCta × NC ways, with
// chunk_idx = yc * kNumGetWarpsPerCta + local_get_idx. Put side
// splits NC ways with chunk_idx = yc.
//
// Signal arrays and staging are a flat list of L = MC * NumStages
// slots. Comm CTA at xc walks slots {xc, xc+MC, …, xc+(NumStages-1)·MC}
// (stride MC, NumStages entries — full column coverage of L). MLP at
// blockIdx.x walks slots {blockIdx.x, blockIdx.x+gridDim.x, …} (stride
// gridDim.x, L/gridDim.x = NumStages·N_SPLIT/NC entries — looping back
// across the full L). Both sides cover all L slots, and per-column
// subsets are disjoint across blockIdx.x (gridDim.x | L, since
// L = gridDim.x · NumStages · N_SPLIT/NC and NC | NumStages·N_SPLIT).
//
// The flat list bounds how far the getters can lead the putter: the
// getter blocks in producer_acquire once its NumStages-slot subset is
// pending MLP consumption, and the putter blocks in consumer_acquire
// until MLP releases a dst slot.
template <typename Element, int TileM, int NumStages, int NC, int N_SPLIT>
__device__ __forceinline__ void nvshmem_comm_main(
		CommSmem& smem,
		const CommBuffers& bufs,
		const GetTmaDescs* get_descs,  // per-peer X load + local staging store (nullptr → getmem)
		char* get_bounce) {            // base of this CTA's GetBounce smem region

	// Comm-side pipe depth is NumStages — one comm CTA cycle covers
	// the full MC × NumStages staging area, independent of N_SPLIT and
	// NC. The MLP side is ticket-based (RemoteMlpTileIterator) and
	// shares no per-stage_id state, so no divisibility relationship is
	// required between NumStages, N_SPLIT and NC.
	constexpr int K = NumStages;

	int warp_id = threadIdx.x / 32;
	if (warp_id < kCommFwdWarpStart || warp_id > kCommFwdWarpEnd)
		return;

	int lane = threadIdx.x % 32;
	// Flat-grid launch: blockIdx.x is the flat cta_id. (xc, yc) comm grid and
	// MC derive from n_gemm + compile-time NC, not gridDim.
	int cta_id = blockIdx.x;
	// n_comm: NC-aligned comm grid, decoupled from n_gemm, recomputed from the
	// launched grid to match the prologue's comm_count.
	int n_comm = (gridDim.x / NC) * NC;
	int xc = cta_id / NC;
	int yc = cta_id % NC;
	int MC = n_comm / NC;
	if (cta_id >= n_comm) return;

	// Per-warp leader: each comm warp must independently spin in
	// StagePipe acquire(). __syncwarp() is warp-local.
	bool is_leader = (lane == 0);

	int total = smem.total_tiles;
	if (total == 0) return;

	int tile_elems = TileM * bufs.hidden_dim;

	// ── Tile iterator: MC-strided through all remote tiles, starting at xc ──
	// All three roles visit the same tile sequence in the same order.
	TileIterator<TileM> iter;
	iter.remote_offsets = smem.remote_offsets;
	iter.offsets_stride = bufs.experts_per_pe + 1;
	iter.experts_per_pe = bufs.experts_per_pe;
	iter.num_pes = bufs.num_pes();
	iter.my_pe = bufs.my_pe();
	iter.init(xc, MC);

	int my_total = iter.total_tiles;
	if (my_total == 0) return;

	auto* src_staging_base = static_cast<Element*>(bufs.src_staging);
	auto* dst_staging_base = static_cast<Element*>(bufs.dst_staging);

	// Warp roles (FWD, TMA GET): warp 1 = single TMA get warp; warps 2+3 =
	// two symmetric put warps. The TMA GET saturates with one warp (DMA
	// engine, issue-bound), so the freed get warp moves to the put side —
	// putmem is warp-LD/ST (thread-bound), so 2 put warps ≈ 2× put BW. The
	// old cooperative last-tile drain is GONE: every tile is now split across
	// kNumPutWarpsFwd·NC put warps that each drive the dst pipe directly.
	if (warp_id == kCommPutWarp || warp_id == kCommGetWarp1) {
		// Two put warps per CTA. put_local_idx ∈ {0,1}; each puts its
		// 1/(kNumPutWarpsFwd·NC) slice and independently drives the pipe.
		// NumProducers = N_SPLIT (GEMM CTAs produce Y), NumConsumers =
		// kNumPutWarpsFwd·NC (2 put warps × NC CTAs per ring).
		const int put_local_idx = (warp_id == kCommPutWarp) ? 0 : 1;
		StagePipe<K, N_SPLIT, kNumPutWarpsFwd * NC> dst_pipe;
		dst_pipe.init(bufs.dst_ready + xc, bufs.dst_consumed + xc, is_leader, MC);

		int ib_seq = 0;  // advances only on IB tiles (inside do_put)
		for (int i = 0; i < my_total; ++i) {
			TileInfo info = iter.next();
			do_put<Element, TileM, K, NC, N_SPLIT>(
				dst_pipe, bufs, info,
				dst_staging_base, tile_elems, xc, MC, yc, ib_seq, lane,
				put_local_idx, get_descs);
		}
	} else {  // warp_id == kCommGetWarp0 — the single TMA get warp
		const int chunk_idx = yc;  // kNumGetWarpsFwd == 1 → yc·1 + 0

		// NumProducers = kNumGetWarpsFwd·NC (= NC: one get warp × NC CTAs),
		// NumConsumers = N_SPLIT (all GEMM CTAs consume cooperatively).
		StagePipe<K, kNumGetWarpsFwd * NC, N_SPLIT> src_pipe;
		src_pipe.init(bufs.src_ready + xc, bufs.src_consumed + xc, is_leader, MC);
		char* bounce_warp = get_bounce;  // single warp → slice 0

		int ib_seq = 0;  // advances only on IB tiles (inside do_get)
		for (int i = 0; i < my_total; ++i) {
			TileInfo info = iter.next();
			do_get<Element, TileM, K, NC, N_SPLIT>(
				src_pipe, bufs, info,
				src_staging_base, tile_elems, xc, MC, chunk_idx, ib_seq, lane,
				get_descs, bounce_warp);
		}
	}
}

} // namespace liger
