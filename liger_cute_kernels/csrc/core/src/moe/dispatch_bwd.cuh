#pragma once

#include "combine.cuh"

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// dispatch_tokens_bwd — backward pass of token scatter (dispatch)
// ═══════════════════════════════════════════════════════════════════
//
// Forward:  X[slot(t,k), d] = I[t, d]          (scatter to top_k slots)
//
// Backward:
//   dI[t, d] = Σ_k dX[slot(t,k), d]            (gather-sum over top_k)
//
// One warp per token, grid-stride. int4 vectorization (8 elements per
// load). intermediate_dim % 8 == 0 required. All warps participate.
// fp32 accumulation across K slots, convert back to Element at store.

static constexpr int kDispatchBwdWarpSize = 32;

template <typename Element, int NumThreads>
__device__ __forceinline__ void dispatch_tokens_bwd(
        const Element* __restrict__ dX,                // [total_slots, intermediate_dim]
        const int* __restrict__ token_expert_slots,    // [num_tokens, top_k]
        Element* __restrict__ dI,                      // [num_tokens, intermediate_dim]
        int intermediate_dim,
        int num_tokens,
        int top_k) {

    static_assert(sizeof(Element) == 2, "dispatch_tokens_bwd requires 2-byte element type");
    static constexpr int kNumWarps     = NumThreads / kDispatchBwdWarpSize;
    static constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(Element);  // 8

    const int warp_id = threadIdx.x / kDispatchBwdWarpSize;
    const int lane    = threadIdx.x % kDispatchBwdWarpSize;
    const int int4_per_row = intermediate_dim / kElemsPerInt4;

    const int4* dX_base = reinterpret_cast<const int4*>(dX);
    int4*       dI_base = reinterpret_cast<int4*>(dI);

    const int cta_id     = blockIdx.x + blockIdx.y * gridDim.x;
    const int grid_warps = gridDim.x * gridDim.y * kNumWarps;
    int token = cta_id * kNumWarps + warp_id;

    for (; token < num_tokens; token += grid_warps) {
        const int slot_base = token * top_k;

        for (int i = lane; i < int4_per_row; i += kDispatchBwdWarpSize) {
            float acc[kElemsPerInt4] = {};

            for (int k = 0; k < top_k; ++k) {
                const int   slot = __ldg(&token_expert_slots[slot_base + k]);
                const int4  raw  = __ldg(&dX_base[slot * int4_per_row + i]);
                const Element* e = reinterpret_cast<const Element*>(&raw);
                CUTE_UNROLL
                for (int j = 0; j < kElemsPerInt4; ++j)
                    acc[j] += to_float<Element>(e[j]);
            }

            int4 out_chunk;
            Element* oe = reinterpret_cast<Element*>(&out_chunk);
            CUTE_UNROLL
            for (int j = 0; j < kElemsPerInt4; ++j)
                oe[j] = from_float<Element>(acc[j]);
            dI_base[token * int4_per_row + i] = out_chunk;
        }
    }
}

} // namespace liger
