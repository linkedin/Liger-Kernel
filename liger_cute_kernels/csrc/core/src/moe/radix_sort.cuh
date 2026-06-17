#pragma once

#include "models.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// Stable counting sort: groups tokens by expert assignment
// ═══════════════════════════════════════════════════════════════════
//
// Each CTA owns a contiguous range of M-tiles [tile_start, tile_end).
// Uses 2D grid with linearized cta_id = blockIdx.x + blockIdx.y * gridDim.x.
//
// Algorithm:
//   0. Per-warp tile histogram building (parallel, no __syncthreads)
//   1. Each CTA sums its tile counts → cta_sums[bid * E], signals done
//   2. CTA 0 does exclusive prefix sum over cta_sums + computes
//      aligned expert_offsets, signals completion
//   3. Each CTA reads its prefix from cta_sums
//   4. Parallel ballot-based scatter (see comments below).
//
// cta_done[] flags:
//   0 → not started
//   1 → histogram + cta_sums written
//   2 → prefix sum + expert_offsets ready (set by CTA 0)
//
// Step 4 details — parallel ballot scatter (replaces the older
// "one thread per expert sequentially scans tiles" design). For each
// tile, every thread owns one (token, k) entry; a warp-level
// __match_any_sync groups same-expert entries within the warp. Each
// warp publishes its per-expert counts into smem; one thread per
// expert does a cross-warp exclusive prefix scan, then reserves the
// chunk's span in the per-CTA expert cursor. Every thread then writes
// its entry to the correct position.
//
// Stability is preserved because:
//   - tiles are processed serially within a CTA;
//   - within a tile, chunks are processed serially;
//   - within a chunk, the cross-warp scan gives every warp an
//     exclusive prefix offset, so warp w's lanes always land
//     before warp w+1's for the same expert;
//   - within a warp, rank = popc(mask & lane_below_bits) preserves
//     lane order, which corresponds to the (token, k) iteration order.

static constexpr int kWarpSize = 32;

// Required smem ints for sort_tokens<TileM, NumThreads> at a given E.
// Layout:
//   [0, NumWarps*E)              warp_expert_count + step-0 tile histograms
//   [NumWarps*E, NumWarps*E + E) expert_pos (per-CTA cursor for step 4)
//   [NumWarps*E + E, +E)         chunk_base (per-chunk reservation base)
constexpr int sort_tokens_smem_ints(int num_threads, int num_experts) {
	return (num_threads / kWarpSize + 2) * num_experts;
}

template <int TileM, int NumThreads>
__device__ __forceinline__ void sort_tokens(
		int* __restrict__ sort_smem,            // see smem layout above
		const int* __restrict__ expert_indices, // [T * K] from router
		SortBuffers sort,
		int num_tokens, int num_experts, int top_k,
		int num_m_tiles) {
	if (threadIdx.x >= NumThreads) {
		return;
	}

	static constexpr int NumWarps = NumThreads / kWarpSize;

	int tid      = threadIdx.x;
	int warp_id  = tid / kWarpSize;
	int lane     = tid % kWarpSize;
	int bid      = blockIdx.x + blockIdx.y * gridDim.x;
	int num_ctas = gridDim.x * gridDim.y;

	int tiles_per_cta = (num_m_tiles + num_ctas - 1) / num_ctas;
	int tile_start = bid * tiles_per_cta;
	int tile_end   = min(tile_start + tiles_per_cta, num_m_tiles);

	// Smem regions (caller must allocate at least
	// (NumWarps + 2) * num_experts ints):
	//   [0, NumWarps*E)              warp_expert_count + step-0 tile histograms
	//   [NumWarps*E, NumWarps*E + E) expert_pos        (per-CTA cursor for step 4)
	//   [NumWarps*E + E, +E)         chunk_base        (per-chunk reservation base)
	int* warp_expert_count = sort_smem;
	int* expert_pos        = sort_smem + NumWarps * num_experts;
	int* chunk_base        = expert_pos + num_experts;

	int* my_counts = warp_expert_count + warp_id * num_experts;

	// ── Step 0: per-warp tile histograms ─────────────────
	for (int m = tile_start + warp_id; m < tile_end; m += NumWarps) {
		int token_offset = m * TileM;
		int tokens_this = min(TileM, num_tokens - token_offset);
		int num_entries = tokens_this * top_k;

		for (int e = lane; e < num_experts; e += kWarpSize)
			my_counts[e] = 0;
		__syncwarp();

		for (int i = lane; i < num_entries; i += kWarpSize) {
			int idx = token_offset + i / top_k;
			int k   = i % top_k;
			int expert_id = expert_indices[idx * top_k + k];
			atomicAdd(&my_counts[expert_id], 1);
		}
		__syncwarp();

		for (int e = lane; e < num_experts; e += kWarpSize)
			sort.tile_expert_counts[m * num_experts + e] = my_counts[e];
		__syncwarp();
	}
	__syncthreads();

	// ── Step 1: sum tile counts → cta_sums ───────────────
	// Reuse first [E] of warp_expert_count as sort_counts (only first E
	// entries touched; warps' histograms past offset E are stale but unused).
	int* sort_counts = warp_expert_count;

	for (int e = tid; e < num_experts; e += NumThreads)
		sort_counts[e] = 0;
	__syncthreads();

	for (int m = tile_start; m < tile_end; ++m) {
		for (int e = tid; e < num_experts; e += NumThreads)
			sort_counts[e] += sort.tile_expert_counts[m * num_experts + e];
		__syncthreads();
	}

	for (int e = tid; e < num_experts; e += NumThreads)
		sort.cta_sums[bid * num_experts + e] = sort_counts[e];
	__threadfence();
	__syncthreads();

	if (tid == 0)
		atomicExch(&sort.cta_done[bid], 1);

	// ── Step 2: CTA 0 prefix-sum + expert_offsets ────────
	// Stage-1 layout: poll ALL cta_done flags in parallel up front,
	// then run the existing serial scan body. The parallel wait collapses
	// the cost of 130 sequential single-thread polls to ~one round-trip
	// (each thread polls ≤ 1 CTA when NumThreads ≥ num_ctas, which holds
	// for any reasonable persistent-kernel configuration).
	if (bid == 0) {
		// Phase A: parallel wait. Every thread polls cta_done for the
		// CTAs assigned to it (stride NumThreads). The acquire semantics
		// of atomicAdd(., 0) + the post-loop __syncthreads guarantee that
		// every thread in CTA 0 sees every CTA's cta_sums writes.
		for (int c = tid; c < num_ctas; c += NumThreads) {
			while (atomicAdd(&sort.cta_done[c], 0) < 1) {}
		}
		__syncthreads();

		// Phase B: per-expert exclusive prefix scan across c, one warp
		// per expert. Each lane owns a contiguous chunk of c-values
		// (size kMaxElemsPerLane), does a per-lane sum, contributes to
		// a warp-level Kogge-Stone exclusive scan, then writes back
		// exclusive prefixes. The total per expert lands in sort_counts.
		//
		// kMaxElemsPerLane sized for up to 256 CTAs (32 lanes × 8). On
		// H100 (132 SMs → grid = ~130 CTAs) only lanes 0..16 are active.
		constexpr int kMaxElemsPerLane = 8;
		for (int e_block = 0; e_block < num_experts; e_block += NumWarps) {
			int e = e_block + warp_id;
			if (e < num_experts) {
				int vals[kMaxElemsPerLane];
				int lane_total = 0;
				#pragma unroll
				for (int i = 0; i < kMaxElemsPerLane; ++i) {
					int c = lane * kMaxElemsPerLane + i;
					int v = (c < num_ctas) ? sort.cta_sums[c * num_experts + e] : 0;
					vals[i] = v;
					lane_total += v;
				}
				// Warp-level inclusive scan via Kogge-Stone.
				int sum = lane_total;
				#pragma unroll
				for (int offset = 1; offset < 32; offset *= 2) {
					int v = __shfl_up_sync(0xFFFFFFFFu, sum, offset);
					if (lane >= offset) sum += v;
				}
				int lane_prefix = sum - lane_total;            // exclusive
				int total       = __shfl_sync(0xFFFFFFFFu, sum, 31);
				// Write back the exclusive prefixes.
				int acc = lane_prefix;
				#pragma unroll
				for (int i = 0; i < kMaxElemsPerLane; ++i) {
					int c = lane * kMaxElemsPerLane + i;
					if (c < num_ctas) {
						sort.cta_sums[c * num_experts + e] = acc;
						acc += vals[i];
					}
				}
				if (lane == 0)
					sort_counts[e] = total;
			}
		}
		__syncthreads();

		if (tid == 0) {
			sort.expert_offsets[0] = 0;
			for (int e = 0; e < num_experts; ++e) {
				int aligned = ((sort_counts[e] + TileM - 1) / TileM) * TileM;
				sort.expert_offsets[e + 1] = sort.expert_offsets[e] + aligned;
			}
			if (sort.tile_expert_ids != nullptr) {
				for (int e = 0; e < num_experts; ++e) {
					int t_start = sort.expert_offsets[e] / TileM;
					int t_end   = sort.expert_offsets[e + 1] / TileM;
					for (int t = t_start; t < t_end; ++t)
						sort.tile_expert_ids[t] = e;
				}
			}
		}
		__threadfence();
		__syncthreads();
		if (tid == 0)
			atomicExch(&sort.cta_done[0], 2);
	}

	// ── Step 3: read CTA prefix ─────────────────────────
	if (tid == 0)
		while (atomicAdd(&sort.cta_done[0], 0) < 2) {}
	__syncthreads();

	for (int e = tid; e < num_experts; e += NumThreads)
		sort_counts[e] = sort.cta_sums[bid * num_experts + e];
	__syncthreads();

	// ── Step 4 (NEW): parallel ballot scatter ───────────
	// Initialize per-CTA expert cursors from the CTA's prefix.
	for (int e = tid; e < num_experts; e += NumThreads)
		expert_pos[e] = sort.expert_offsets[e] + sort_counts[e];
	__syncthreads();

	for (int m = tile_start; m < tile_end; ++m) {
		int token_offset = m * TileM;
		int tokens_in_tile = min(TileM, num_tokens - token_offset);
		int entries_in_tile = tokens_in_tile * top_k;

		// Process the tile in chunks of NumThreads entries.
		for (int chunk_start = 0; chunk_start < entries_in_tile;
				chunk_start += NumThreads) {

			// Clear warp_expert_count for this chunk. (Scan below reads
			// every (w, e) slot, so leftover values would corrupt it.)
			for (int i = tid; i < NumWarps * num_experts; i += NumThreads)
				warp_expert_count[i] = 0;
			__syncthreads();

			int i = chunk_start + tid;
			bool active = (i < entries_in_tile);
			int token_in_tile = active ? i / top_k : 0;
			int k_idx         = active ? i % top_k : 0;
			int my_idx        = token_offset + token_in_tile;
			// Sentinel for inactive lanes — distinct from all valid expert
			// IDs (which are in [0, num_experts)).
			int my_e = active ? expert_indices[my_idx * top_k + k_idx] : -1;

			// Warp-level same-expert grouping.
			unsigned int mask = __match_any_sync(0xFFFFFFFFu, my_e);
			int rank        = __popc(mask & ((1u << lane) - 1));
			int leader_lane = __ffs((int)mask) - 1;
			int warp_count  = __popc(mask);
			bool is_leader  = (lane == leader_lane) && active;

			if (is_leader)
				warp_expert_count[warp_id * num_experts + my_e] = warp_count;
			__syncthreads();

			// Cross-warp exclusive prefix scan per expert. One thread per
			// expert (E threads total) walks warps 0..NumWarps-1.
			if (tid < num_experts) {
				int sum = 0;
				for (int w = 0; w < NumWarps; ++w) {
					int v = warp_expert_count[w * num_experts + tid];
					warp_expert_count[w * num_experts + tid] = sum;
					sum += v;
				}
				chunk_base[tid] = expert_pos[tid];
				expert_pos[tid] += sum;
			}
			__syncthreads();

			if (active) {
				int my_warp_offset = warp_expert_count[warp_id * num_experts + my_e];
				int write_pos = chunk_base[my_e] + my_warp_offset + rank;
				sort.sorted_token_ids[write_pos] = my_idx;
				sort.token_expert_slots[my_idx * top_k + k_idx] = write_pos;
			}
			__syncthreads();
		}
	}
}

} // namespace liger
