#pragma once

#include "combine.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// combine_tokens_bwd — backward pass of weighted gather-reduce
// ═══════════════════════════════════════════════════════════════════
//
// Forward:  O[t, d] = Σ_k w[t, k] * Y[slot(t,k), d]
//
// Backward:
//   dW[t, k]          = Σ_d dO[t, d] * Y[slot(t,k), d]   (dot over hidden dim)
//   dY[slot(t,k), d]  = dO[t, d] * w[t, k]                (scaled scatter)
//
// One warp per token, grid-stride. int4 vectorization (8 elements per
// load). intermediate_dim % 16 == 0 required (we read pairs of int4 per
// i-iter; tail handles ≤1 leftover int4).
//
// Per-K specialization: at runtime `top_k` dispatches to a fixed-K
// kernel where the slot/weight/dW arrays are sized to exactly K and
// fit cleanly in lane registers. Within fixed-K we:
//   • load dO once per int4-pair (i, i+1), reuse across all K
//   • prefetch dO for the next pair into L2 (single hint, K-independent)
//   • process 2 int4 (32 B) of dO + 2 int4 of Y per pair to expose
//     more memory-level parallelism per warp
//   • K=4 also load-batches Y across all K before any FMA, which the
//     LSU can absorb at that K
//
// History: this layout was selected after a 10-round sweep — see
// `combine_bwd_optimization_journal.md`. Geomean throughput rose from
// ~2120 GB/s (the original k-outer/i-inner walk) to ~2707 GB/s, with
// the largest shape reaching 2885 GB/s ≈ 86 % of HBM3 peak.

namespace detail {

__device__ __forceinline__ void combine_bwd_l2_prefetch(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(ptr));
}

// Standard fixed-K specialization: wider (2× int4 per pair) + dO L2 prefetch.
// Used for K=2 and K=8.
template <int kK, typename Element, int NumThreads>
__device__ __forceinline__ void combine_tokens_bwd_fixed_k_wider(
        const Element* __restrict__ dO,
        const Element* __restrict__ expert_out,
        const Element* __restrict__ expert_weights,
        const int* __restrict__ token_expert_slots,
        Element* __restrict__ dW,
        Element* __restrict__ dY,
        int intermediate_dim,
        int num_tokens) {

    constexpr int kWarpSize     = 32;
    constexpr int kNumWarps     = NumThreads / kWarpSize;
    constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(Element);  // 8

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;
    const int int4_per_row = intermediate_dim / kElemsPerInt4;
    const int int4_pairs   = int4_per_row / 2;
    const int int4_tail    = int4_per_row & 1;

    const int4* dO_base = reinterpret_cast<const int4*>(dO);
    const int4* Y_base  = reinterpret_cast<const int4*>(expert_out);
    int4*       dY_base = reinterpret_cast<int4*>(dY);

    const int cta_id     = blockIdx.x + blockIdx.y * gridDim.x;
    const int grid_warps = gridDim.x * gridDim.y * kNumWarps;
    int token = cta_id * kNumWarps + warp_id;

    for (; token < num_tokens; token += grid_warps) {
        const int wk_base = token * kK;

        int   slot_arr[kK];
        float w_arr[kK];
        float dW_acc[kK];
        CUTE_UNROLL
        for (int k = 0; k < kK; ++k) {
            slot_arr[k] = __ldg(&token_expert_slots[wk_base + k]);
            w_arr[k]    = to_float<Element>(
                ldg_elem<Element>(&expert_weights[wk_base + k]));
            dW_acc[k]   = 0.0f;
        }

        const int dO_row_off = token * int4_per_row;

        for (int p = lane; p < int4_pairs; p += kWarpSize) {
            const int i0 = 2 * p;
            const int i1 = 2 * p + 1;

            const int p_next = p + kWarpSize;
            if (p_next < int4_pairs)
                combine_bwd_l2_prefetch(&dO_base[dO_row_off + 2 * p_next]);

            const int4 dO_raw0 = __ldg(&dO_base[dO_row_off + i0]);
            const int4 dO_raw1 = __ldg(&dO_base[dO_row_off + i1]);
            const Element* dO_e0 = reinterpret_cast<const Element*>(&dO_raw0);
            const Element* dO_e1 = reinterpret_cast<const Element*>(&dO_raw1);

            float dO_f0[kElemsPerInt4], dO_f1[kElemsPerInt4];
            CUTE_UNROLL
            for (int j = 0; j < kElemsPerInt4; ++j) {
                dO_f0[j] = to_float<Element>(dO_e0[j]);
                dO_f1[j] = to_float<Element>(dO_e1[j]);
            }

            CUTE_UNROLL
            for (int k = 0; k < kK; ++k) {
                const int row_off = slot_arr[k] * int4_per_row;
                const int4 Y_raw0 = __ldg(&Y_base[row_off + i0]);
                const int4 Y_raw1 = __ldg(&Y_base[row_off + i1]);
                const Element* Y_e0 = reinterpret_cast<const Element*>(&Y_raw0);
                const Element* Y_e1 = reinterpret_cast<const Element*>(&Y_raw1);

                int4 dY_out0, dY_out1;
                Element* dY_e0 = reinterpret_cast<Element*>(&dY_out0);
                Element* dY_e1 = reinterpret_cast<Element*>(&dY_out1);

                float local = 0.0f;
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j) {
                    float Y_f0 = to_float<Element>(Y_e0[j]);
                    float Y_f1 = to_float<Element>(Y_e1[j]);
                    local   += dO_f0[j] * Y_f0 + dO_f1[j] * Y_f1;
                    dY_e0[j] = from_float<Element>(dO_f0[j] * w_arr[k]);
                    dY_e1[j] = from_float<Element>(dO_f1[j] * w_arr[k]);
                }
                dW_acc[k] += local;
                dY_base[row_off + i0] = dY_out0;
                dY_base[row_off + i1] = dY_out1;
            }
        }

        // Tail: 0 or 1 leftover int4 chunk per lane.
        if (int4_tail) {
            const int i = 2 * int4_pairs + lane;
            if (i < int4_per_row) {
                const int4 dO_raw = __ldg(&dO_base[dO_row_off + i]);
                const Element* dO_e = reinterpret_cast<const Element*>(&dO_raw);
                float dO_f[kElemsPerInt4];
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j)
                    dO_f[j] = to_float<Element>(dO_e[j]);

                CUTE_UNROLL
                for (int k = 0; k < kK; ++k) {
                    const int slot = slot_arr[k];
                    const int4 Y_raw = __ldg(&Y_base[slot * int4_per_row + i]);
                    const Element* Y_e = reinterpret_cast<const Element*>(&Y_raw);
                    int4 dY_out;
                    Element* dY_e = reinterpret_cast<Element*>(&dY_out);
                    float local = 0.0f;
                    CUTE_UNROLL
                    for (int j = 0; j < kElemsPerInt4; ++j) {
                        float Y_f = to_float<Element>(Y_e[j]);
                        local   += dO_f[j] * Y_f;
                        dY_e[j]  = from_float<Element>(dO_f[j] * w_arr[k]);
                    }
                    dW_acc[k] += local;
                    dY_base[slot * int4_per_row + i] = dY_out;
                }
            }
        }

        CUTE_UNROLL
        for (int k = 0; k < kK; ++k) {
            float acc = dW_acc[k];
            CUTE_UNROLL
            for (int offset = 16; offset >= 1; offset >>= 1)
                acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
            if (lane == 0)
                dW[wk_base + k] = from_float<Element>(acc);
        }
    }
}

// Load-batched fixed-K specialization for K=4: load all K Y per pair before
// any FMA, then K FMAs, then K stores. Stresses the LSU's ability to keep
// K Y loads in flight. Net win at K=4 only — at K=2 there's nothing to
// batch and at K=8 the K Y chunks + K dY out chunks exceed the per-lane
// register sweet spot.
template <int kK, typename Element, int NumThreads>
__device__ __forceinline__ void combine_tokens_bwd_fixed_k_wider_loadbatch(
        const Element* __restrict__ dO,
        const Element* __restrict__ expert_out,
        const Element* __restrict__ expert_weights,
        const int* __restrict__ token_expert_slots,
        Element* __restrict__ dW,
        Element* __restrict__ dY,
        int intermediate_dim,
        int num_tokens) {

    constexpr int kWarpSize     = 32;
    constexpr int kNumWarps     = NumThreads / kWarpSize;
    constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(Element);

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;
    const int int4_per_row = intermediate_dim / kElemsPerInt4;
    const int int4_pairs   = int4_per_row / 2;
    const int int4_tail    = int4_per_row & 1;

    const int4* dO_base = reinterpret_cast<const int4*>(dO);
    const int4* Y_base  = reinterpret_cast<const int4*>(expert_out);
    int4*       dY_base = reinterpret_cast<int4*>(dY);

    const int cta_id     = blockIdx.x + blockIdx.y * gridDim.x;
    const int grid_warps = gridDim.x * gridDim.y * kNumWarps;
    int token = cta_id * kNumWarps + warp_id;

    for (; token < num_tokens; token += grid_warps) {
        const int wk_base = token * kK;

        int   slot_arr[kK];
        float w_arr[kK];
        float dW_acc[kK];
        CUTE_UNROLL
        for (int k = 0; k < kK; ++k) {
            slot_arr[k] = __ldg(&token_expert_slots[wk_base + k]);
            w_arr[k]    = to_float<Element>(
                ldg_elem<Element>(&expert_weights[wk_base + k]));
            dW_acc[k]   = 0.0f;
        }

        const int dO_row_off = token * int4_per_row;

        for (int p = lane; p < int4_pairs; p += kWarpSize) {
            const int i0 = 2 * p;
            const int i1 = 2 * p + 1;

            const int p_next = p + kWarpSize;
            if (p_next < int4_pairs)
                combine_bwd_l2_prefetch(&dO_base[dO_row_off + 2 * p_next]);

            const int4 dO_raw0 = __ldg(&dO_base[dO_row_off + i0]);
            const int4 dO_raw1 = __ldg(&dO_base[dO_row_off + i1]);
            const Element* dO_e0 = reinterpret_cast<const Element*>(&dO_raw0);
            const Element* dO_e1 = reinterpret_cast<const Element*>(&dO_raw1);

            float dO_f0[kElemsPerInt4], dO_f1[kElemsPerInt4];
            CUTE_UNROLL
            for (int j = 0; j < kElemsPerInt4; ++j) {
                dO_f0[j] = to_float<Element>(dO_e0[j]);
                dO_f1[j] = to_float<Element>(dO_e1[j]);
            }

            // ── Load all K Y for this pair ─────────────────────────
            int4 Y_raw0[kK], Y_raw1[kK];
            CUTE_UNROLL
            for (int k = 0; k < kK; ++k) {
                const int row_off = slot_arr[k] * int4_per_row;
                Y_raw0[k] = __ldg(&Y_base[row_off + i0]);
                Y_raw1[k] = __ldg(&Y_base[row_off + i1]);
            }

            // ── Compute K FMAs ─────────────────────────────────────
            int4 dY_out0[kK], dY_out1[kK];
            CUTE_UNROLL
            for (int k = 0; k < kK; ++k) {
                const Element* Y_e0 = reinterpret_cast<const Element*>(&Y_raw0[k]);
                const Element* Y_e1 = reinterpret_cast<const Element*>(&Y_raw1[k]);
                Element* dY_e0 = reinterpret_cast<Element*>(&dY_out0[k]);
                Element* dY_e1 = reinterpret_cast<Element*>(&dY_out1[k]);

                float local = 0.0f;
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j) {
                    float Y_f0 = to_float<Element>(Y_e0[j]);
                    float Y_f1 = to_float<Element>(Y_e1[j]);
                    local   += dO_f0[j] * Y_f0 + dO_f1[j] * Y_f1;
                    dY_e0[j] = from_float<Element>(dO_f0[j] * w_arr[k]);
                    dY_e1[j] = from_float<Element>(dO_f1[j] * w_arr[k]);
                }
                dW_acc[k] += local;
            }

            // ── Issue all K dY stores ──────────────────────────────
            CUTE_UNROLL
            for (int k = 0; k < kK; ++k) {
                const int row_off = slot_arr[k] * int4_per_row;
                dY_base[row_off + i0] = dY_out0[k];
                dY_base[row_off + i1] = dY_out1[k];
            }
        }

        if (int4_tail) {
            const int i = 2 * int4_pairs + lane;
            if (i < int4_per_row) {
                const int4 dO_raw = __ldg(&dO_base[dO_row_off + i]);
                const Element* dO_e = reinterpret_cast<const Element*>(&dO_raw);
                float dO_f[kElemsPerInt4];
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j)
                    dO_f[j] = to_float<Element>(dO_e[j]);

                CUTE_UNROLL
                for (int k = 0; k < kK; ++k) {
                    const int slot = slot_arr[k];
                    const int4 Y_raw = __ldg(&Y_base[slot * int4_per_row + i]);
                    const Element* Y_e = reinterpret_cast<const Element*>(&Y_raw);
                    int4 dY_out;
                    Element* dY_e = reinterpret_cast<Element*>(&dY_out);
                    float local = 0.0f;
                    CUTE_UNROLL
                    for (int j = 0; j < kElemsPerInt4; ++j) {
                        float Y_f = to_float<Element>(Y_e[j]);
                        local   += dO_f[j] * Y_f;
                        dY_e[j]  = from_float<Element>(dO_f[j] * w_arr[k]);
                    }
                    dW_acc[k] += local;
                    dY_base[slot * int4_per_row + i] = dY_out;
                }
            }
        }

        CUTE_UNROLL
        for (int k = 0; k < kK; ++k) {
            float acc = dW_acc[k];
            CUTE_UNROLL
            for (int offset = 16; offset >= 1; offset >>= 1)
                acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
            if (lane == 0)
                dW[wk_base + k] = from_float<Element>(acc);
        }
    }
}

// Generic fall-back used when top_k is not 2 / 4 / 8. Same shape as the
// fixed-K kernels but uses a kMaxTopK-sized register array (template-bounded).
template <typename Element, int NumThreads, int kMaxTopK>
__device__ __forceinline__ void combine_tokens_bwd_generic(
        const Element* __restrict__ dO,
        const Element* __restrict__ expert_out,
        const Element* __restrict__ expert_weights,
        const int* __restrict__ token_expert_slots,
        Element* __restrict__ dW,
        Element* __restrict__ dY,
        int intermediate_dim,
        int num_tokens,
        int top_k) {

    constexpr int kWarpSize     = 32;
    constexpr int kNumWarps     = NumThreads / kWarpSize;
    constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(Element);

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;
    const int int4_per_row = intermediate_dim / kElemsPerInt4;

    const int4* dO_base = reinterpret_cast<const int4*>(dO);
    const int4* Y_base  = reinterpret_cast<const int4*>(expert_out);
    int4*       dY_base = reinterpret_cast<int4*>(dY);

    const int cta_id     = blockIdx.x + blockIdx.y * gridDim.x;
    const int grid_warps = gridDim.x * gridDim.y * kNumWarps;
    int token = cta_id * kNumWarps + warp_id;

    for (; token < num_tokens; token += grid_warps) {
        const int wk_base = token * top_k;

        int   slot_arr[kMaxTopK];
        float w_arr[kMaxTopK];
        float dW_acc[kMaxTopK];
        CUTE_UNROLL
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) {
                slot_arr[k] = __ldg(&token_expert_slots[wk_base + k]);
                w_arr[k]    = to_float<Element>(
                    ldg_elem<Element>(&expert_weights[wk_base + k]));
            }
            dW_acc[k] = 0.0f;
        }

        const int dO_row_off = token * int4_per_row;

        for (int i = lane; i < int4_per_row; i += kWarpSize) {
            const int4 dO_raw = __ldg(&dO_base[dO_row_off + i]);
            const Element* dO_e = reinterpret_cast<const Element*>(&dO_raw);

            float dO_f[kElemsPerInt4];
            CUTE_UNROLL
            for (int j = 0; j < kElemsPerInt4; ++j)
                dO_f[j] = to_float<Element>(dO_e[j]);

            CUTE_UNROLL
            for (int k = 0; k < kMaxTopK; ++k) {
                if (k >= top_k) break;
                const int slot = slot_arr[k];
                const int4 Y_raw = __ldg(&Y_base[slot * int4_per_row + i]);
                const Element* Y_e = reinterpret_cast<const Element*>(&Y_raw);

                int4 dY_out;
                Element* dY_e = reinterpret_cast<Element*>(&dY_out);

                float local = 0.0f;
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j) {
                    const float Y_f = to_float<Element>(Y_e[j]);
                    local   += dO_f[j] * Y_f;
                    dY_e[j]  = from_float<Element>(dO_f[j] * w_arr[k]);
                }
                dW_acc[k] += local;
                dY_base[slot * int4_per_row + i] = dY_out;
            }
        }

        CUTE_UNROLL
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k >= top_k) break;
            float acc = dW_acc[k];
            CUTE_UNROLL
            for (int offset = 16; offset >= 1; offset >>= 1)
                acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
            if (lane == 0)
                dW[wk_base + k] = from_float<Element>(acc);
        }
    }
}

} // namespace detail

// Public entry — same signature as the original baseline. Per-K dispatch
// at the top routes to the layout that won the 10-round sweep:
//   K=2: wider + dO L2 prefetch
//   K=4: wider + dO L2 prefetch + load-batched k-loop
//   K=8: wider + dO L2 prefetch
// Other K fall back to a generic kMaxTopK=8 implementation.
template <typename Element, int NumThreads>
__device__ __forceinline__ void combine_tokens_bwd(
        const Element* __restrict__ dO,
        const Element* __restrict__ expert_out,
        const Element* __restrict__ expert_weights,
        const int* __restrict__ token_expert_slots,
        Element* __restrict__ dW,
        Element* __restrict__ dY,
        int intermediate_dim,
        int num_tokens,
        int top_k) {

    static_assert(sizeof(Element) == 2, "combine_tokens_bwd requires 2-byte element type");

    switch (top_k) {
        case 2:
            detail::combine_tokens_bwd_fixed_k_wider<2, Element, NumThreads>(
                dO, expert_out, expert_weights, token_expert_slots,
                dW, dY, intermediate_dim, num_tokens);
            return;
        case 4:
            detail::combine_tokens_bwd_fixed_k_wider_loadbatch<4, Element, NumThreads>(
                dO, expert_out, expert_weights, token_expert_slots,
                dW, dY, intermediate_dim, num_tokens);
            return;
        case 8:
            detail::combine_tokens_bwd_fixed_k_wider<8, Element, NumThreads>(
                dO, expert_out, expert_weights, token_expert_slots,
                dW, dY, intermediate_dim, num_tokens);
            return;
        default:
            detail::combine_tokens_bwd_generic<Element, NumThreads, /*kMaxTopK=*/8>(
                dO, expert_out, expert_weights, token_expert_slots,
                dW, dY, intermediate_dim, num_tokens, top_k);
            return;
    }
}

} // namespace liger
