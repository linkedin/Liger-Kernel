#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>          // cp_async_fence / cp_async_wait
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include "math.cuh"   // to_float, from_float, ldg_elem

namespace liger {

static constexpr int kCombineWarpSize = 32;
static constexpr int kCombineMaxTopK  = 16;

// ═══════════════════════════════════════════════════════════════════
// combine_tokens_tma — 1-token-per-warp + cp.async pipeline + TMA store
// ═══════════════════════════════════════════════════════════════════
//
// CTA = 12 warps (384 threads) → 3 WGs of 4 warps each.
// TMA store tile = kCombineWGTokens × kCombineWGHidden = 32 × 256.
//
// Tile decomposition (key for HBM coalescing):
//   - Each warp owns 1 TOKEN at a time → all 32 lanes share slot/weight
//   - 32 lanes spread across N=256: lane l holds elements [l·8, l·8+8)
//   - That's 8 bf16 per lane = 16 B → int4 cp.async load width
//   - Each warp's load is 32 lanes × 16 B = 512 B *contiguous* in one row
//     → one coalesced HBM transaction per (warp, token, k)
//   - 4 warps × 8 tokens (sequential per warp) = 32 tokens per WG tile
//
// Cross-token cp.async pipeline (2 stages, 8 tokens per warp):
//   - Prologue: issue tokens 0 and 1
//   - Iter t = 0..7: issue token t+2 (if any), wait for t, reduce t
//   - Stages cycle: token t → stage (t % 2)
//   - Slots/weights for different tokens are different, so issues are
//     necessarily per-token (cannot bulk across tokens).
//   - In the final partial tile, invalid token rows use cp.async zero-fill;
//     their output rows are zero in SMEM and discarded by the TMA OOB store.

static constexpr int kCombineWarpsPerWG    = 4;
static constexpr int kCombineWGTokens      = 32;
static constexpr int kCombineWGHidden      = 256;
static constexpr int kCombineTokensPerWarp = kCombineWGTokens / kCombineWarpsPerWG;  // 8
static constexpr int kCombineElemsPerLane  = kCombineWGHidden / kCombineWarpSize;    // 8
static constexpr int kCombineLaneBytes     = kCombineElemsPerLane * 2;               // 16 (= int4)
static constexpr int kCombineStages        = 2;     // output staging double-buffer
static constexpr int kCpAsyncStages        = 2;     // input cp.async ring (SMEM-bound)
static constexpr int kCombineTmaMaxTopK    = 8;     // SMEM sized for this max

template <typename Element, int NumWGs>
struct CombineSmem {
	static constexpr int kStoreElems = kCombineWGTokens * kCombineWGHidden;
	static constexpr int kWGThreads  = kCombineWarpsPerWG * kCombineWarpSize;
	// Output TMA-store staging.
	alignas(128) Element store_buf[NumWGs][kCombineStages][kStoreElems];
	// cp.async input staging:
	//   [WG][stage][warp_in_wg][k][lane] → one int4 (16 B) each
	// One stage holds all K rows for ONE token (across one warp).
	alignas(128) int4 load_buf[NumWGs][kCpAsyncStages]
	                          [kCombineWarpsPerWG]
	                          [kCombineTmaMaxTopK]
	                          [kCombineWarpSize];
};

template <typename Element>
using CombineSmemLayoutSlot = cute::Layout<
	cute::Shape <cute::Int<kCombineWGTokens>, cute::Int<kCombineWGHidden>>,
	cute::Stride<cute::Int<kCombineWGHidden>, cute::_1>>;

// 16-byte cp.async loads use cute::SM80_CP_ASYNC_CACHEGLOBAL<int4>.
// Each cp.async target address is unique within a work item.

template <typename Element, int NumThreads, class TmaStore>
__device__ __forceinline__ void combine_tokens_tma(
        const Element* __restrict__ expert_out,        // [total_slots, hidden_dim]
        const Element* __restrict__ expert_weights,    // [num_tokens, top_k]
        const int* __restrict__ token_expert_slots,    // [num_tokens, top_k]
        TmaStore const& tma_store,
        CombineSmem<Element, NumThreads / (kCombineWarpsPerWG * kCombineWarpSize)>& smem,
        int hidden_dim,
        int num_tokens,
        int top_k) {

	using namespace cute;

	static_assert(sizeof(Element) == 2, "combine_tokens_tma requires 2-byte element type");
	static constexpr int kNumWarps     = NumThreads / kCombineWarpSize;
	static constexpr int kNumWGs       = kNumWarps / kCombineWarpsPerWG;
	static constexpr int kWGThreads    = kCombineWarpsPerWG * kCombineWarpSize;   // 128
	static_assert(kCombineTokensPerWarp == 8,
		"Pipeline unrolled for 8 tokens per warp.");
	static_assert(kCombineWGHidden % kCombineWarpSize == 0,
		"kCombineWGHidden must be lane-strided in N.");

	const int warp_id        = threadIdx.x / kCombineWarpSize;
	const int lane           = threadIdx.x % kCombineWarpSize;
	const int wg_id          = warp_id / kCombineWarpsPerWG;
	const int warp_in_wg     = warp_id % kCombineWarpsPerWG;
	const int wg_leader_tid  = wg_id * kWGThreads;
	const bool is_wg_leader  = (threadIdx.x == wg_leader_tid);
	const int wg_barrier_id  = wg_id + 1;

	// All quantities are in int4 (16-byte) units along N — that's the
	// cp.async granularity. One int4 = 8 bf16 elements.
	const int int4_per_row   = hidden_dim / kCombineElemsPerLane;
	const int n_n_chunks     = hidden_dim / kCombineWGHidden;
	const int n_m_tiles      =
		(num_tokens + kCombineWGTokens - 1) / kCombineWGTokens;
	const int n_work_items   = n_m_tiles * n_n_chunks;

	const int4* src_int4 = reinterpret_cast<const int4*>(expert_out);

	auto mY = tma_store.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma = tma_store.get_slice(Int<0>{});

	const int cta_id = blockIdx.x + blockIdx.y * gridDim.x;
	const int n_ctas = gridDim.x * gridDim.y;

	int my_work     = cta_id * kNumWGs + wg_id;
	int work_stride = n_ctas * kNumWGs;

	bool store_in_flight = false;
	int  out_stage = 0;

	for (; my_work < n_work_items; my_work += work_stride) {
		const int m_tile  = my_work / n_n_chunks;
		const int n_chunk = my_work % n_n_chunks;

		// Base int4 index in N for this work-item.
		const int n_base_int4 = n_chunk * (kCombineWGHidden / kCombineElemsPerLane);  // n_chunk · 32

		// Token of THIS warp's t-th step within the WG tile:
		//   token(t) = m_tile · 32 + warp_in_wg · 8 + t
		const int warp_token_base = m_tile * kCombineWGTokens
		                          + warp_in_wg * kCombineTokensPerWarp;
		const bool warp_all_valid =
			warp_token_base + kCombineTokensPerWarp <= num_tokens;

		// Helper: issue cp.async for token (warp's t) into ring stage S.
		// Each lane issues K cp.asyncs (one per slot, 16 B each).
		// `.cg` skips L1, leaving room for slot/weight L1 hits.
		auto issue_token = [&] (int T, int S) {
			const int my_token    = warp_token_base + T;
			const int weight_base = my_token * top_k;
			int4* stage_base = &smem.load_buf[wg_id][S][warp_in_wg][0][0];
			if (warp_all_valid) {
				for (int k = 0; k < top_k; ++k) {
					const int slot = __ldg(&token_expert_slots[weight_base + k]);
					const int4* src = &src_int4[slot * int4_per_row
					                            + n_base_int4 + lane];
					int4* dst = stage_base + k * kCombineWarpSize + lane;
					cute::SM80_CP_ASYNC_CACHEGLOBAL<int4>::copy(*src, *dst);
				}
			} else {
				const bool token_valid = my_token < num_tokens;
				for (int k = 0; k < top_k; ++k) {
					const int slot = token_valid
						? __ldg(&token_expert_slots[weight_base + k])
						: 0;
					const int4* src = &src_int4[slot * int4_per_row
					                            + n_base_int4 + lane];
					int4* dst = stage_base + k * kCombineWarpSize + lane;
					cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<int4>::copy(
						*src, *dst, token_valid);
				}
			}
			cute::cp_async_fence();
		};

		// Helper: reduce token T from ring stage S → write to output SMEM.
		auto reduce_token = [&] (int T, int S, Element* slot_ptr) {
			const int my_token    = warp_token_base + T;
			const int weight_base = my_token * top_k;
			int4* stage_base = &smem.load_buf[wg_id][S][warp_in_wg][0][0];

			float acc[kCombineElemsPerLane] = {};
			if (warp_all_valid || my_token < num_tokens) {
				for (int k = 0; k < top_k; ++k) {
					const float w = to_float<Element>(
						ldg_elem<Element>(&expert_weights[weight_base + k]));
					const int4 raw = stage_base[k * kCombineWarpSize + lane];
					const Element* e = reinterpret_cast<const Element*>(&raw);
					CUTE_UNROLL
					for (int j = 0; j < kCombineElemsPerLane; ++j)
						acc[j] += w * to_float<Element>(e[j]);
				}
			}

			// Write 8 elements as one int4 to the output SMEM tile.
			int4 out_chunk;
			Element* oe = reinterpret_cast<Element*>(&out_chunk);
			CUTE_UNROLL
			for (int j = 0; j < kCombineElemsPerLane; ++j)
				oe[j] = from_float<Element>(acc[j]);
			// Row in tile = warp_in_wg·8 + T, lane = int4 col idx.
			const int row = warp_in_wg * kCombineTokensPerWarp + T;
			int4* tile_int4 = reinterpret_cast<int4*>(slot_ptr);
			tile_int4[row * (kCombineWGHidden / kCombineElemsPerLane) + lane] = out_chunk;
		};

		// ═══ Wait for prior TMA store on this output stage to drain ═══
		if (store_in_flight && is_wg_leader) {
			cute::tma_store_wait<kCombineStages - 1>();
		}
		cutlass::arch::NamedBarrier::sync(kWGThreads, wg_barrier_id);

		Element* slot_ptr = &smem.store_buf[wg_id][out_stage][0];

		// ═══ Cross-token cp.async pipeline (2 stages, 8 tokens) ═══
		// Prologue: pre-issue tokens 0, 1.
		issue_token(0, 0);
		issue_token(1, 1);

		// t=0..5: wait t, reduce t, issue t+2 (in stage t%2 — safe to
		// reuse since reduce_token finished reading it).
		CUTE_UNROLL
		for (int t = 0; t < kCombineTokensPerWarp - 2; ++t) {
			cute::cp_async_wait<1>();
			reduce_token(t, t % kCpAsyncStages, slot_ptr);
			issue_token(t + 2, (t + 2) % kCpAsyncStages);
		}

		// t=6: only token 7 should still be in flight after wait.
		cute::cp_async_wait<1>();
		reduce_token(kCombineTokensPerWarp - 2,
			(kCombineTokensPerWarp - 2) % kCpAsyncStages, slot_ptr);

		// t=7: drain the last in-flight commit, reduce.
		cute::cp_async_wait<0>();
		reduce_token(kCombineTokensPerWarp - 1,
			(kCombineTokensPerWarp - 1) % kCpAsyncStages, slot_ptr);

		// ═══ TMA store ═══
		cutlass::arch::NamedBarrier::sync(kWGThreads, wg_barrier_id);

		if (is_wg_leader) {
			auto sStore = make_tensor(make_smem_ptr(slot_ptr),
				CombineSmemLayoutSlot<Element>{});
			auto gY = local_tile(mY,
				make_tile(Int<kCombineWGTokens>{}, Int<kCombineWGHidden>{}),
				make_coord(m_tile, n_chunk));
			cute::tma_store_fence();
			copy(tma_store, cta_tma.partition_S(sStore),
				cta_tma.partition_D(gY));
			cute::tma_store_arrive();
		}

		store_in_flight = true;
		out_stage = (out_stage + 1) % kCombineStages;
	}

	if (is_wg_leader) {
		cute::tma_store_wait<0>();
	}
}

} // namespace liger
