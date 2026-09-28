#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace liger {

// ═══════════════════════════════════════════════════════════════════
// Token dispatch: scatter tokens to sorted positions (int4 vectorized)
// ═══════════════════════════════════════════════════════════════════
//
// Copies tokens from the input [T, D] tensor to sorted positions in
// the destination buffer [total_slots, D]. Each sorted_token_ids[i]
// gives the original token index to place at row i. Padding slots
// (id == -1) are zero-filled.
//
// One warp per token row within each tile. All warps in the CTA
// collaborate on one tile at a time.
//
// Element must be 2 bytes (bf16, fp16). int4 = 16 bytes = 8 elements.

static constexpr int kDispatchWarpSize = 32;

template <typename Element, int TileM, int NumThreads>
__device__ __forceinline__ void dispatch_tokens(
		const Element* __restrict__ tokens,             // [T, D]
		Element* __restrict__ dst,                       // [total_slots, D]
		const int* __restrict__ sorted_token_ids,        // [total_slots]
		int hidden_dim,
		int total_slots) {

	static_assert(sizeof(Element) == 2, "dispatch_tokens requires 2-byte element type");
	static constexpr int NumWarps = NumThreads / kDispatchWarpSize;
	static constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(Element);  // 8

	int warp_id = threadIdx.x / kDispatchWarpSize;
	int lane    = threadIdx.x % kDispatchWarpSize;

	int int4_per_row = hidden_dim / kElemsPerInt4;

	const int4* src_base = reinterpret_cast<const int4*>(tokens);
	int4* dst_base       = reinterpret_cast<int4*>(dst);

	const int4 zero = {0, 0, 0, 0};

	int cta_id   = blockIdx.x + blockIdx.y * gridDim.x;
	int num_ctas = gridDim.x * gridDim.y;
	int num_tiles = total_slots / TileM;

	// Iterate by tile, grid-strided. All warps in CTA share the tile.
	for (int tile = cta_id; tile < num_tiles; tile += num_ctas) {
		int tile_row_start = tile * TileM;

		// Each warp handles a subset of rows within the tile.
		for (int r = warp_id; r < TileM; r += NumWarps) {
			int row = tile_row_start + r;
			int src_token = __ldg(&sorted_token_ids[row]);

			int dst_off = row * int4_per_row;

			if (src_token >= 0) {
				int src_off = src_token * int4_per_row;
				for (int i = lane; i < int4_per_row; i += kDispatchWarpSize)
					dst_base[dst_off + i] = __ldg(&src_base[src_off + i]);
			} else {
				for (int i = lane; i < int4_per_row; i += kDispatchWarpSize)
					dst_base[dst_off + i] = zero;
			}
		}
	}
}

} // namespace liger
