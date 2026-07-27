#pragma once

// ═══════════════════════════════════════════════════════════════════
// Backward MoE communication: gets X + dY, puts dX
// ═══════════════════════════════════════════════════════════════════
//
// Warp roles:
//   warp 1 → X/dY get warp
//   warp 2 → dX put warp
//   warp 3 → MLP-side SM100 UMMA producer (exits early on SM90)
//
// Grid is launched 2D as (gridDim.x, NSplit) where NSplit is the
// MLP-side cooperative count. Comms re-indexes the flat CTA id into
// a (MC, NC) logical grid:
//
//   cta_id = blockIdx.x * gridDim.y + blockIdx.y
//   xc     = cta_id / NC                in [0, MC)
//   yc     = cta_id % NC                in [0, NC)
//   MC     = (gridDim.x * gridDim.y) / NC   (multiply before divide so
//                                            MC is exact when NC ∤ NSplit)
//
// NC CTAs (same xc, different yc) cooperate on one tile per get/put.
// MC > gridDim.x when NC < NSplit so excess comm CTAs run ahead and
// load the next tiles while the MLP is still on the current one.
//
// ── Tile-level locking (same as fwd) ───────────────────────────────
//
// All three pipes (X, dY, dX) use the per-tile StagePipe over a flat
// [L = MC * NumStages]-sized signal array. Get pipes have
// NumProducers = kNumGetWarpsPerCta·NC, NumConsumers = NSplit. Put
// pipe has NumProducers = NSplit, NumConsumers = NC. Comm strides MC
// across stages (NumStages entries — full ring cycle, independent of
// NSplit and NC); MLP is ticket-based (RemoteMlpTileIteratorBwd) and
// keys on the absolute slot index = (m_base + idx·gridDim.x) mod L.
// Both sides cover the same flat L slots.
//
// ═══════════════════════════════════════════════════════════════════

#include "mlp_comms.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// Bwd comm warp constants
// ═══════════════════════════════════════════════════════════════════
//
// Main loop runs on warps 1-2 only. kCommBwdGetWarp1 remains for legacy
// prologue compatibility with shared helpers; it is not a main-loop role.

static constexpr int kCommBwdGetWarp0  = 1;
static constexpr int kCommBwdGetWarp1  = 2;  // prologue compatibility only
static constexpr int kCommBwdPutWarp   = 2;
static constexpr int kCommBwdWarpStart = kCommBwdGetWarp0;
static constexpr int kCommBwdWarpEnd   = kCommBwdPutWarp;

// ── TMA-GET (NVLink) warp rebalance ─────────────────────────────────
// Mirrors the fwd #111 layout: one warp saturates the TMA GET path, so warp 1
// gets X+dY and warp 2 puts dX. Warp 3 enters the MLP-side branch and exits
// there for Compute != 100. The MLP-side iterator picks producer/consumer
// counts from the same runtime tma_enabled flag.
static constexpr int kNumGetWarpsBwd = 1;  // TMA mode: 1 get warp (X+dY)
static constexpr int kNumPutWarpsBwd = 1;  // TMA mode: 1 put warp (dX)

// ═══════════════════════════════════════════════════════════════════
// GetTmaDescsBwd — raw CUtensorMap descriptors for the bwd TMA GET
// ═══════════════════════════════════════════════════════════════════
//
// Per-same-host-peer source maps over nvshmem_ptr(x_sorted, peer) and
// nvshmem_ptr(dy_sorted, peer); one local dst map each over x_staging /
// dy_staging. Built host-side in moe_bwd.cu, passed __grid_constant__.
// Same as fwd GetTmaDescs: local peers index by pe % gpus_per_node; remote
// peers fall back to IB getmem in do_get_bwd_tma.
struct GetTmaDescsBwd {
	CUtensorMap src_x_desc[kGetMaxPes];   // index by (TileInfo::pe % gpus_per_node)
	CUtensorMap src_dy_desc[kGetMaxPes];
	CUtensorMap dst_x_staging_desc;       // local x_staging
	CUtensorMap dst_dy_staging_desc;      // local dy_staging
	int enabled;                          // 1 = local-peer TMA path available
	int gpus_per_node;
	int my_pe;
};

__host__ __device__ __forceinline__ bool get_pe_is_local_bwd(
		const GetTmaDescsBwd* descs, int pe) {
	if (descs == nullptr || descs->gpus_per_node <= 0) return true;
	return (pe / descs->gpus_per_node) == (descs->my_pe / descs->gpus_per_node);
}

// Box height (rows) of a single bwd TMA transfer. The bwd smem union sits
// near the 228 KiB cap (mlp1_act ≈ 216 KiB), so unlike the fwd's 32 KiB
// [64×256] box, the bwd uses a small [kGetBoxRowsBwd × kGetKChunk] box and
// the get warp loops over row-bands. kGetBoxRowsBwd must divide the warp's
// row band (TileM / NC). 16 divides 64/32/16 for NC ∈ {2,4,8} at TileM=128.
// Box = 16·256·2 = 8 KiB, which fits the ~12 KiB headroom of the default
// config (216 KiB union); tighter tuned configs fall back to getmem.
static constexpr int kGetBoxRowsBwd = 16;

// Smem bounce: one [kGetBoxRowsBwd × kGetKChunk] box + its mbarrier, both
// 128-aligned. Single get warp → one region.
template <typename Element>
struct GetBounceBwd {
	static constexpr int kBoxRows  = kGetBoxRowsBwd;
	static constexpr int kBoxElems = kBoxRows * kGetKChunk;
	static constexpr int kBoxBytes = kBoxElems * (int)sizeof(Element);
	static constexpr int kPerWarpStride = ((kBoxBytes + 8) + 127) & ~127;
	static constexpr int kTotalBytes    = kPerWarpStride;
};

// do_get_bwd_tma is defined below, after CommBuffersBwd (it needs the
// complete struct type for bufs.hidden_dim).

// ═══════════════════════════════════════════════════════════════════
// CommBuffersBwd — backward-specific buffer pointers
// ═══════════════════════════════════════════════════════════════════

struct CommBuffersBwd {
	// Staging: flat slot list [L][TileM][hidden_dim] where
	// L = gridDim.x * NumStages. Indexed by slot ∈ [0, L).
	void* x_staging;           // gets: X from remote PE
	void* dy_staging;          // gets: dY from remote PE
	void* dx_staging;          // puts: dX back to remote PE

	// Per-tile signals ([L] entries each), one counter pair per slot —
	// same granularity as fwd src/dst pipes.
	int* x_src_ready;
	int* x_src_consumed;
	int* dy_src_ready;
	int* dy_src_consumed;
	int* dst_ready;
	int* dst_consumed;

	// Per-slot expert IDs ([L] each), one array per get pipe.
	//
	// They hold the SAME value for any given live slot — info.expert at the
	// time the comm warp wrote that slot — but each is gated by its own
	// pipe's StagePipe release window:
	//
	//   tile_expert_ids_x  is overwritten between MLP's release_src and
	//                      comm's next-lap producer_release on the X pipe.
	//                      Phase 2 mlp4 reads this (mlp4 reads X) and is
	//                      safe as long as release_src is deferred past mlp4.
	//
	//   tile_expert_ids_dy is overwritten between MLP's release_dy and
	//                      comm's next-lap producer_release on the dY pipe.
	//                      Phase 2 mlp3 reads this (mlp3 reads dY) and is
	//                      safe as long as release_dy is deferred past mlp3.
	//
	// Splitting the array is what lets release_src happen between mlp4 and
	// mlp3 without races: comm refilling tile_expert_ids_x in that window
	// can't poison mlp3, because mlp3 reads the dY-pipe array.
	int* tile_expert_ids_x;
	int* tile_expert_ids_dy;

	// Remote data pointers (symmetric memory)
	void* remote_x;            // get source: X on remote PE
	void* remote_dy;           // get source: dY on remote PE
	void* remote_dx;           // put destination: dX on remote PE

	// Reused from forward
	int* all_expert_offsets;
	int all_expert_offsets_stride;

	int my_pe;
	int num_pes;
	nvshmem_team_t team;

	int hidden_dim;
	int num_experts;
	int experts_per_pe;

	// GEMM/comm-active CTA count = floor_NS = grid_x · NSplit (NOT the
	// launched grid size). Under the flat 1-D launch gridDim.x = num_blocks
	// (= num_sms) ≥ n_gemm, so MC = n_gemm / NC and the MLP column count
	// grid_x = n_gemm / NSplit must come from here, not from gridDim.
	int n_gemm;

	// Used by the dummy_consumer_bwd in mlp_comms_bwd.cu when it replicates
	// the real BWD's batch loop (mlp_global_barrier + mlp_x_barrier). Not
	// used by the comm warps themselves.
	int* barrier_counter;
	int* phase_counter;
};

// ═══════════════════════════════════════════════════════════════════
// do_get_bwd_tma — TMA GET of one buffer (X or dY) for one tile
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors fwd do_get's TMA path (cuda::ptx intrinsics): peer → smem box →
// local staging, via TMA load + TMA store. Because the bounce box is small
// (kGetBoxRowsBwd rows), the single get warp loops over its TileM/NC-row
// band in kGetBoxRowsBwd-tall sub-bands × kGetKChunk-wide column boxes.
// chunk_idx == yc (the CTA's slot within its NC-cooperating group); with
// kNumGetWarpsBwd == 1 the per-warp row band is TileM/NC.
template <typename Element, int TileM, int K, int NC>
__device__ __forceinline__ void do_get_bwd_tma(
		StagePipe<K, kNumGetWarpsBwd * NC, 1>& src_pipe,
		const CommBuffersBwd& bufs,
		const TileInfo& info,
		Element* remote_base,
		Element* staging_base,
		int tile_elems,
		int xc,
		int MC,
		int chunk_idx,                 // = yc (0..NC-1)
		int& ib_seq,                   // advanced once per actual IB get call
		int lane,
		int* tile_expert_ids_dst,
		const GetTmaDescsBwd* descs,
		const CUtensorMap* src_desc_per_pe,  // descs->src_x_desc or src_dy_desc
		const CUtensorMap* dst_desc,         // descs->dst_x_staging or dst_dy_staging
		char* bounce_warp) {

	namespace ptx = cuda::ptx;
	using Bounce = GetBounceBwd<Element>;
	constexpr int kWarpsPerTile = kNumGetWarpsBwd * NC;  // = NC
	constexpr int kRowBand      = TileM / kWarpsPerTile;  // rows this warp owns
	constexpr int kBoxRows      = Bounce::kBoxRows;

	src_pipe.producer_acquire();

	int slot = xc + src_pipe.producer_stage() * MC;
	Element* local_tile  = staging_base + slot * tile_elems;
	Element* remote_tile = remote_base + info.token_offset * bufs.hidden_dim;

	bool is_local = descs && descs->enabled && get_pe_is_local_bwd(descs, info.pe);
	bool tma_ok = is_local && (bufs.hidden_dim % kGetKChunk == 0) && bounce_warp != nullptr;

	if (tma_ok) {
		Element*  sbuf = reinterpret_cast<Element*>(bounce_warp);
		uint64_t* mbar = reinterpret_cast<uint64_t*>(bounce_warp + Bounce::kBoxBytes);

		if (lane == 0)
			ptx::mbarrier_init(mbar, 1u);
		__syncwarp();
		ptx::fence_proxy_async(ptx::space_shared);

		const int n_k = bufs.hidden_dim / kGetKChunk;
		const uint32_t txbytes = (uint32_t)Bounce::kBoxBytes;
		int y_src_base = info.token_offset + chunk_idx * kRowBand;  // peer row
		int y_dst_base = slot * TileM       + chunk_idx * kRowBand;  // staging row
		const int local_peer = info.pe % descs->gpus_per_node;

		uint32_t phase = 0;
		for (int rb = 0; rb < kRowBand; rb += kBoxRows) {
			for (int k = 0; k < n_k; ++k) {
				int x = k * kGetKChunk;
				if (lane == 0) {
					(void)ptx::mbarrier_arrive_expect_tx(
						ptx::sem_release, ptx::scope_cta, ptx::space_shared,
						mbar, txbytes);
					int32_t load_crd[2] = {x, y_src_base + rb};
					ptx::cp_async_bulk_tensor(
						ptx::space_cluster, ptx::space_global,
						sbuf, &src_desc_per_pe[local_peer], load_crd, mbar);
					while (!ptx::mbarrier_try_wait_parity(mbar, phase)) {}
					int32_t store_crd[2] = {x, y_dst_base + rb};
					ptx::cp_async_bulk_tensor(
						ptx::space_global, ptx::space_shared,
						dst_desc, store_crd, sbuf);
					ptx::cp_async_bulk_commit_group();
					ptx::cp_async_bulk_wait_group(ptx::n32_t<0>{});
				}
				phase ^= 1;
				__syncwarp();
			}
		}
	} else {
		constexpr int kWarpsPerTile = kNumGetWarpsBwd * NC;  // one get warp per CTA
		int chunk = tile_elems / kWarpsPerTile;
		int offset = chunk_idx * chunk;
		int global_pe = nvshmem_team_translate_pe(bufs.team, info.pe, NVSHMEM_TEAM_WORLD);
		void* p2p_remote = nvshmem_ptr(remote_tile, global_pe);
		if (p2p_remote != nullptr) {
			nvshmemx_getmem_warp(local_tile + offset, remote_tile + offset,
				(size_t)chunk * sizeof(Element), global_pe);
		} else {
			if (chunk_idx == ib_seq % kWarpsPerTile) {
				nvshmemx_getmem_warp(local_tile, remote_tile,
					(size_t)tile_elems * sizeof(Element), global_pe);
			}
			++ib_seq;
		}
	}

	if (tile_expert_ids_dst != nullptr && chunk_idx == 0 && lane == 0)
		tile_expert_ids_dst[slot] = info.expert;

	__syncwarp();
	src_pipe.producer_release(lane);
}

// ═══════════════════════════════════════════════════════════════════
// do_get_bwd — gets one buffer (X or dY) for one tile
// ═══════════════════════════════════════════════════════════════════
//
// kWarpsPerTile = kNumGetWarpsPerCta * NC = 2·NC (warps 1+2 across
// NC cooperating CTAs). Slot in flat list = xc + producer_stage*MC.

// `tile_expert_ids_dst` is the per-pipe expert-id array to write to (one of
// bufs.tile_expert_ids_x or bufs.tile_expert_ids_dy, depending on which pipe
// this call is for), or nullptr to skip the write. Exactly one comm warp per
// tile is expected to pass non-null — caller picks the writer to avoid double
// writes and false sharing.
//
// The write happens AFTER the getmem completes and BEFORE producer_release,
// so it is part of the slot's atomic publish: any consumer that's seen
// producer_release sees both the getmem'd staging bytes AND the expert id.
template <typename Element, int TileM, int K, int NC>
__device__ __forceinline__ void do_get_bwd(
		StagePipe<K, kNumGetWarpsPerCta * NC, 1>& src_pipe,
		const CommBuffersBwd& bufs,
		const TileInfo& info,
		Element* remote_base,
		Element* staging_base,    // base of flat [L][TileM][hidden_dim]
		int tile_elems,
		int xc,                   // start slot in flat list
		int MC,                   // stride between consecutive comm slots
		int chunk_idx,            // 0..2·NC-1
		int& ib_seq,              // IB-tile counter (round-robin turn; advanced here)
		int lane,
		int* tile_expert_ids_dst) {

	constexpr int kWarpsPerTile = kNumGetWarpsPerCta * NC;
	int chunk = tile_elems / kWarpsPerTile;
	int offset = chunk_idx * chunk;

	src_pipe.producer_acquire();

	// Slot in the flat list. Comm strides MC across pipe stages;
	// MLP strides gridDim.x. Both resolve to the same physical
	// slot for a given global tile since L = MC·K = gridDim.x·NumStages.
	int slot = xc + src_pipe.producer_stage() * MC;

	int global_pe = nvshmem_team_translate_pe(bufs.team, info.pe, NVSHMEM_TEAM_WORLD);

	// Tile base (offset 0). P2P chunks add `offset`; the IB whole-tile
	// transfer addresses the base directly.
	Element* local_tile  = staging_base + slot * tile_elems;
	Element* remote_tile = remote_base + info.token_offset * bufs.hidden_dim;

	// Fast path: NVLink P2P → all 2·NC warps cooperate, each copying its
	// chunk with direct int4 loads. IB path: one whole-tile getmem issued
	// by a single warp, chosen round-robin across IB tiles only (ib_seq
	// skips local/P2P tiles and is advanced in lockstep by every warp).
	// See the matching fwd do_get for the full rationale.
	void* p2p_remote = nvshmem_ptr(remote_tile, global_pe);
	if (p2p_remote != nullptr) {
		static_assert(sizeof(Element) == 2,
			"P2P int4 copy assumes 2-byte elements (bf16/half)");
		constexpr int kElemsPerVec = sizeof(int4) / sizeof(Element);
		int n_vec = chunk / kElemsPerVec;
		const int4* src_v = reinterpret_cast<const int4*>(
			static_cast<Element*>(p2p_remote) + offset);
		int4*       dst_v = reinterpret_cast<int4*>(local_tile + offset);
		#pragma unroll 4
		for (int i = lane; i < n_vec; i += 32) {
			dst_v[i] = src_v[i];
		}
		__syncwarp();
	} else {
		if (chunk_idx == ib_seq % kWarpsPerTile) {
			nvshmemx_getmem_warp(local_tile, remote_tile,
				(size_t)tile_elems * sizeof(Element), global_pe);
		}
		++ib_seq;
	}

	if (tile_expert_ids_dst != nullptr && chunk_idx == 0 && lane == 0) {
		tile_expert_ids_dst[slot] = info.expert;
	}
	__threadfence();

	src_pipe.producer_release(lane);
}

// ═══════════════════════════════════════════════════════════════════
// do_put_bwd — puts dX for one tile back to remote PE
// ═══════════════════════════════════════════════════════════════════
//
// kWarpsPerTile = NC. NumProducers = NSplit (MLP-side),
// NumConsumers = NC (comm-side).

// num_put_warps / put_local_idx keep the single-put-warp layout
// (kWarpsPerTile = NC, slice = yc): warp 2 owns dX puts while warp 3 is
// reserved for the SM100 MLP path.
template <typename Element, int TileM, int K, int NC>
__device__ __forceinline__ void do_put_bwd(
		StagePipe<K, 1, NC>& dst_pipe,
		const CommBuffersBwd& bufs,
		const TileInfo& info,
		Element* dx_staging,      // base of flat [L][TileM][hidden_dim]
		int tile_elems,
		int xc,
		int MC,
		int chunk_idx,            // 0..NC-1 (= yc)
		int& ib_seq,              // IB-tile counter (round-robin turn; advanced here)
		int lane,
		int num_put_warps = 1,    // 1 (fallback) or kNumPutWarpsBwd (TMA mode)
		int put_local_idx = 0) {  // 0..num_put_warps-1

	int kWarpsPerTile = num_put_warps * NC;
	int put_idx = chunk_idx * num_put_warps + put_local_idx;  // 0..kWarpsPerTile-1
	int chunk = tile_elems / kWarpsPerTile;
	int offset = put_idx * chunk;
	int my_bytes = chunk * sizeof(Element);

	dst_pipe.consumer_acquire();

	int slot = xc + dst_pipe.consumer_stage() * MC;
	auto* remote_dx = static_cast<Element*>(bufs.remote_dx);
	Element* local_tile  = dx_staging + slot * tile_elems;
	Element* remote_tile = remote_dx + info.token_offset * bufs.hidden_dim;

	int global_pe = nvshmem_team_translate_pe(bufs.team, info.pe, NVSHMEM_TEAM_WORLD);

	// NVLink P2P peers keep the cooperative split; IB peers send the whole
	// tile as one RDMA issued by a single put warp, round-robin over IB
	// tiles only (see do_get_bwd / fwd do_put).
	void* p2p_remote = nvshmem_ptr(remote_tile, global_pe);
	if (p2p_remote != nullptr) {
		nvshmemx_putmem_warp(remote_tile + offset, local_tile + offset,
			my_bytes, global_pe);
	} else {
		if (put_idx == ib_seq % kWarpsPerTile) {
			nvshmemx_putmem_warp(remote_tile, local_tile,
				(size_t)tile_elems * sizeof(Element), global_pe);
		}
		++ib_seq;
	}

	dst_pipe.consumer_release(lane);
}

// ═══════════════════════════════════════════════════════════════════
// nvshmem_comm_prologue_bwd — fetch remote offsets, compute tile counts
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors fwd nvshmem_comm_prologue. Computes both counts since uneven
// distribution can give comm and MLP different per-CTA tile counts:
//   smem.total_tiles   = comm count  (start = xc, stride = MC)
//   smem.per_cta_tiles = MLP  count  (start = blockIdx.x, stride = gridDim.x)
//
// Runs on warps 2-3 (kCommWarpStart..kCommWarpEnd). Grid is 2D
// (gridDim.x, NSplit).

__device__ __forceinline__ int select_runtime_nsplit_bwd_from_tiles(
		int total_tiles,
		int num_n_tiles_1, int num_n_tiles_2t, int num_n_tiles_5,
		int num_k_tiles_1, int num_k_tiles_2t, int num_k_tiles_5,
		int experts_per_pe,
		int num_blocks, int static_nsplit) {
	if (total_tiles <= 0) return static_nsplit;
	constexpr int candidates[5] = {2, 4, 6, 8, 16};
	int best_ns = static_nsplit;
	long long best_cost = -1;
	long long best_waste = -1;
	int best_n_gemm = -1;
	long long w1  = 2LL * num_k_tiles_1;   // mlp1 recomputes gate + up
	long long w2t = (long long)num_k_tiles_2t;
	long long w5  = 2LL * num_k_tiles_5;   // mlp5 computes dU@B + dV@C
	#pragma unroll
	for (int i = 0; i < 5; ++i) {
		int ns = candidates[i];
		if (ns > num_blocks) continue;
		int n_gemm = (num_blocks / ns) * ns;
		int ms = n_gemm / ns;
		if (ms <= 0) continue;
		int m_waves = (total_tiles + ms - 1) / ms;
		int n1_waves  = (num_n_tiles_1  + ns - 1) / ns;
		int n2t_waves = (num_n_tiles_2t + ns - 1) / ns;
		int n5_waves  = (num_n_tiles_5  + ns - 1) / ns;
		long long phase_cost = w1 * n1_waves + w2t * n2t_waves + w5 * n5_waves;
		long long cost = (long long)m_waves * phase_cost;
		int min_n_tiles = min(num_n_tiles_1, min(num_n_tiles_2t, num_n_tiles_5));
		bool fragmented_experts = (experts_per_pe >= 8) && (min_n_tiles <= 8);
		if (fragmented_experts && ns < 4) {
			long long frag = ((long long)experts_per_pe * 1024LL)
				/ (long long)max(total_tiles, 1);
			cost += (cost * frag) / 128LL;
		}
		long long waste1 =
			(long long)m_waves * ms * n1_waves * ns
			- (long long)total_tiles * num_n_tiles_1;
		long long waste2t =
			(long long)m_waves * ms * n2t_waves * ns
			- (long long)total_tiles * num_n_tiles_2t;
		long long waste5 =
			(long long)m_waves * ms * n5_waves * ns
			- (long long)total_tiles * num_n_tiles_5;
		long long waste = w1 * waste1 + w2t * waste2t + w5 * waste5;
		if (best_cost < 0 ||
		    cost < best_cost ||
		    (cost == best_cost && waste < best_waste) ||
		    (cost == best_cost && waste == best_waste && n_gemm > best_n_gemm)) {
			best_cost = cost;
			best_waste = waste;
			best_n_gemm = n_gemm;
			best_ns = ns;
		}
	}
	return best_ns;
}

template <int TileM, int NC = 2>
__device__ __forceinline__ void nvshmem_comm_prologue_bwd(
		CommSmem& smem,
		const CommBuffersBwd& bufs,
		int static_nsplit) {

	int warp_id = threadIdx.x / 32;
	if (warp_id < kCommWarpStart || warp_id > kCommWarpEnd)
		return;

	int lane = threadIdx.x % 32;
	int local_warp = warp_id - kCommWarpStart;
	bool is_leader = (local_warp == 0 && lane == 0);

	int experts_per_pe = bufs.experts_per_pe;
	int num_pes        = bufs.num_pes;
	int my_pe          = bufs.my_pe;

	if (bufs.all_expert_offsets == nullptr) {
		if (is_leader) {
			smem.total_tiles   = 0;
			smem.per_cta_tiles = 0;
			smem.global_total  = 0;
			smem.runtime_nsplit = static_nsplit;
			smem.runtime_n_gemm = bufs.n_gemm;
			smem.runtime_grid_x = bufs.n_gemm / static_nsplit;
		}
		__threadfence_block();
		__syncwarp();
		return;
	}
	// NOTE: num_pes == 1 is NO LONGER short-circuited (mirrors fwd
	// nvshmem_comm_prologue). The shared TileIterator enumerates the LOCAL PE
	// (p=0), so a single-GPU bwd run stages its local X/dY through the comm
	// get path and puts dX back through the comm put path exactly like the
	// multi-GPU remote path — the separate local MLP pass is gone.

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

	// Flat 1-D launch: cta_id = flat_id = blockIdx.x (gridDim.y == 1).
	// Comm uses all NC-complete launched CTAs, decoupled from the MLP runtime
	// NS. MLP count below is only the static-seed default; moe_bwd_kernel
	// overwrites it after selecting runtime NS from the actual tile count.
	int cta_id = blockIdx.x * gridDim.y + blockIdx.y;
	int n_comm = (gridDim.x / NC) * NC;
	int xc = cta_id / NC;
	int MC = n_comm / NC;
	int grid_x = bufs.n_gemm / static_nsplit;
	int mlp_col = cta_id / static_nsplit;

	TileIterator<TileM> iter;
	iter.remote_offsets = smem.remote_offsets;
	iter.offsets_stride = dst_stride;
	iter.experts_per_pe = experts_per_pe;
	iter.num_pes = num_pes;
	iter.my_pe = my_pe;

	// Comm count: this CTA's per-(xc, yc) workload.
	iter.init(xc, MC);
	int comm_count = iter.total_tiles;

	// MLP count: this CTA's logical column (mlp_col) workload, stride grid_x.
	iter.init(mlp_col, grid_x);
	int mlp_count = iter.total_tiles;

	// Grid-wide remote tile count (start=0, stride=1 → counts every tile
	// once). Same value on every CTA — used by moe_fused_bwd to gate the
	// remote phase consistently across the grid.
	iter.init(0, 1);
	int grid_total = iter.total_tiles;

	if (is_leader) {
		smem.total_tiles   = comm_count;   // read by comm_main_bwd
		smem.per_cta_tiles = mlp_count;    // read by MLP iterator
		smem.global_total  = grid_total;   // read by moe_fused_bwd gate
		smem.runtime_nsplit = static_nsplit;
		smem.runtime_n_gemm = bufs.n_gemm;
		smem.runtime_grid_x = grid_x;
	}
	// Block-scope fence so the leader's smem writes above are visible
	// to all threads in this CTA (including MLP warps that early-
	// returned at the top) AFTER the caller's __syncthreads(). Without
	// this, the compiler/HW may reorder MLP warps' loads of
	// smem.global_total to before the comm warp's store, giving stale
	// (zero-initialized) values and causing CTA(0,0) to skip the remote
	// phase while peer CTAs proceed — manifests as a multi-PE deadlock.
	__threadfence_block();
	__syncwarp();
}

// ═══════════════════════════════════════════════════════════════════
// nvshmem_comm_main_bwd — independent flat loops per warp role
// ═══════════════════════════════════════════════════════════════════
//
// All three pipes are per-tile StagePipes (mirrors fwd). Comm at xc
// walks slots {xc, xc+MC, ..., xc+(K-1)·MC} of the [L]-sized signal
// arrays (stride MC). Get pipes use NumProducers = 2·NC,
// NumConsumers = NSplit; put pipe uses NumProducers = NSplit,
// NumConsumers = NC.

template <typename Element, int TileM, int NumStages, int NC = 2>
__device__ __forceinline__ void nvshmem_comm_main_bwd(
		CommSmem& smem,
		const CommBuffersBwd& bufs,
		const GetTmaDescsBwd* descs = nullptr,  // nullptr/disabled → getmem layout
		char* get_bounce = nullptr) {           // bwd TMA bounce smem region

	// Comm-side pipe depth is NumStages — one comm CTA cycle covers
	// the full MC × NumStages staging area, independent of NSplit and
	// NC. The MLP side is ticket-based (RemoteMlpTileIteratorBwd) and
	// shares no per-stage_id state, so no divisibility relationship is
	// required between NumStages, NSplit and NC.
	constexpr int K = NumStages;

	int warp_id = threadIdx.x / 32;
	if (warp_id != kCommBwdGetWarp0 && warp_id != kCommBwdPutWarp)
		return;

	int lane = threadIdx.x % 32;
	bool is_leader = (lane == 0);

	// Flat 1-D launch: cta_id = flat_id = blockIdx.x (gridDim.y == 1).
	int cta_id = blockIdx.x * gridDim.y + blockIdx.y;
	int n_comm = (gridDim.x / NC) * NC;
	if (cta_id >= n_comm) return;
	int xc = cta_id / NC;
	int yc = cta_id % NC;
	int MC = n_comm / NC;
	int runtime_nsplit = smem.runtime_nsplit;

	int total = smem.total_tiles;
	if (total == 0) return;

	int tile_elems = TileM * bufs.hidden_dim;

	TileIterator<TileM> iter;
	iter.remote_offsets = smem.remote_offsets;
	iter.offsets_stride = bufs.experts_per_pe + 1;
	iter.experts_per_pe = bufs.experts_per_pe;
	iter.num_pes = bufs.num_pes;
	iter.my_pe = bufs.my_pe;
	iter.init(xc, MC);

	int my_total = iter.total_tiles;
	if (my_total == 0) return;

	auto* x_staging  = static_cast<Element*>(bufs.x_staging);
	auto* dy_staging = static_cast<Element*>(bufs.dy_staging);
	auto* dx_staging = static_cast<Element*>(bufs.dx_staging);
	auto* remote_x_base  = static_cast<Element*>(bufs.remote_x);
	auto* remote_dy_base = static_cast<Element*>(bufs.remote_dy);

	// Unified layout: warp 1 gets X then dY for each tile; warp 2 puts dX.
	// do_get_bwd_tma chooses per tile: same-host TMA, P2P fallback, or IB
	// whole-tile round-robin. This keeps X(t), dY(t), X(t+1), ... in one loop.
	if (warp_id == kCommBwdGetWarp0) {
		StagePipe<K, kNumGetWarpsBwd * NC, 1> x_pipe;
		x_pipe.init(bufs.x_src_ready + xc, bufs.x_src_consumed + xc,
		            is_leader, MC, kNumGetWarpsBwd * NC, runtime_nsplit);

		StagePipe<K, kNumGetWarpsBwd * NC, 1> dy_pipe;
		dy_pipe.init(bufs.dy_src_ready + xc, bufs.dy_src_consumed + xc,
		             is_leader, MC, kNumGetWarpsBwd * NC, runtime_nsplit);

		int chunk_idx = yc;  // kNumGetWarpsBwd == 1
		// Exactly one writer per tile per pipe: warp 1 (kCommBwdGetWarp0),
		// yc=0, lane=0. The X-pipe get writes tile_expert_ids_x; the dY-pipe
		// get writes tile_expert_ids_dy. Both arrays end up holding the same
		// info.expert for any given live slot — but each is gated by its
		// own pipe's release window, so MLP can use the X-pipe array for
		// Phase 2 mlp4 and the dY-pipe array for Phase 2 mlp3 with
		// asymmetric release timing (release_src after mlp4, release_dy
		// after mlp3) and no race in either read.
		bool is_expert_id_writer =
			(warp_id == kCommBwdGetWarp0) && (yc == 0);
		int* x_expert_dst  = is_expert_id_writer ? bufs.tile_expert_ids_x  : nullptr;
		int* dy_expert_dst = is_expert_id_writer ? bufs.tile_expert_ids_dy : nullptr;

		// One IB-tile counter shared by the X and dY gets, advanced per IB
		// getmem so X(t), dY(t), X(t+1), … rotate across the NC get warps.
		int ib_seq = 0;
		for (int i = 0; i < my_total; ++i) {
			TileInfo info = iter.next();

			do_get_bwd_tma<Element, TileM, K, NC>(
				x_pipe, bufs, info, remote_x_base, x_staging, tile_elems,
				xc, MC, chunk_idx, ib_seq, lane, x_expert_dst, descs,
				descs ? descs->src_x_desc : nullptr,
				descs ? &descs->dst_x_staging_desc : nullptr,
				get_bounce);

			do_get_bwd_tma<Element, TileM, K, NC>(
				dy_pipe, bufs, info, remote_dy_base, dy_staging, tile_elems,
				xc, MC, chunk_idx, ib_seq, lane, dy_expert_dst, descs,
				descs ? descs->src_dy_desc : nullptr,
				descs ? &descs->dst_dy_staging_desc : nullptr,
				get_bounce);
		}
	} else {  // warp 2 → dX put warp
		StagePipe<K, 1, NC> dst_pipe;
		dst_pipe.init(bufs.dst_ready + xc, bufs.dst_consumed + xc,
		              is_leader, MC, runtime_nsplit, kNumPutWarpsBwd * NC);

		int put_local_idx = 0;
		int chunk_idx = yc;
		int ib_seq = 0;  // advances only on IB tiles (inside do_put_bwd)
		for (int i = 0; i < my_total; ++i) {
			TileInfo info = iter.next();
			do_put_bwd<Element, TileM, K, NC>(
				dst_pipe, bufs, info,
				dx_staging, tile_elems, xc, MC, chunk_idx, ib_seq, lane,
				kNumPutWarpsBwd, put_local_idx);
		}
	}
}

} // namespace liger
