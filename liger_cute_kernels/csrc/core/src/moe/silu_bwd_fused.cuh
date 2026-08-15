#pragma once

// ═══════════════════════════════════════════════════════════════════
// SiLU backward (fused tile):
//   dV = dZ * SiLU(U)
//   dU = dZ * V * grad_SiLU(U)
// ═══════════════════════════════════════════════════════════════════
//
// Element-wise backward of Z = SiLU(U) * V.  Uses all MLP warps
// (0, 1, 4–11 = 320 threads), excluding communication warps 2–3.
// Vectorized with uint4 (8 bf16 per load/store).
//
// num_elems is a runtime constant (tokens × intermediate_dim).

#include "math.cuh"

namespace liger {

static constexpr int kSiluBwdThreads = 320;  // 10 warps × 32

template <typename Element>
__device__ __forceinline__ void silu_bwd_fused(
		const Element* __restrict__ dZ,
		const Element* __restrict__ U,
		const Element* __restrict__ V,
		Element* __restrict__ dU,
		Element* __restrict__ dV,
		int num_elems) {

	constexpr int kVecSize = sizeof(uint4) / sizeof(Element);

	// Dense thread ID across 10 participating warps (0,1,4-11).
	// Warps 2-3 (comm) have already returned before reaching this call.
	int warp_id    = threadIdx.x / 32;
	int lane       = threadIdx.x % 32;
	int dense_warp = (warp_id < 2) ? warp_id : warp_id - 2;
	int tid        = dense_warp * 32 + lane;

	auto* dZ_v = reinterpret_cast<const uint4*>(dZ);
	auto* U_v  = reinterpret_cast<const uint4*>(U);
	auto* V_v  = reinterpret_cast<const uint4*>(V);
	auto* dU_v = reinterpret_cast<uint4*>(dU);
	auto* dV_v = reinterpret_cast<uint4*>(dV);

	int vec_count = num_elems / kVecSize;

	for (int i = tid; i < vec_count; i += kSiluBwdThreads) {
		uint4 dz_vec = dZ_v[i];
		uint4 u_vec  = U_v[i];
		uint4 v_vec  = V_v[i];

		auto* dz_e = reinterpret_cast<Element*>(&dz_vec);
		auto* u_e  = reinterpret_cast<Element*>(&u_vec);
		auto* v_e  = reinterpret_cast<Element*>(&v_vec);

		uint4 du_out, dv_out;
		auto* du_e = reinterpret_cast<Element*>(&du_out);
		auto* dv_e = reinterpret_cast<Element*>(&dv_out);

		CUTE_UNROLL
		for (int j = 0; j < kVecSize; ++j) {
			float dz = static_cast<float>(dz_e[j]);
			float u  = static_cast<float>(u_e[j]);
			float v  = static_cast<float>(v_e[j]);
			float f = fast_sigmoid(u);
			float df = f * (1.0f + u * (1.0f - f));

			dv_e[j] = from_float<Element>(dz * u * f);
			du_e[j] = from_float<Element>(dz * v * df);
		}

		dV_v[i] = dv_out;
		dU_v[i] = du_out;
	}
}

// ═══════════════════════════════════════════════════════════════════
// N-tile-aware strided variant: processes a [num_rows, num_cols]
// sub-matrix within a row-major [num_rows, row_stride] buffer.
// Handles non-contiguous columns efficiently with full thread
// utilization across all rows × cols.
// ═══════════════════════════════════════════════════════════════════

template <typename Element>
__device__ __forceinline__ void silu_bwd_fused_strided(
		const Element* __restrict__ dZ_base,
		const Element* __restrict__ U_base,
		const Element* __restrict__ V_base,
		Element* __restrict__ dU_base,
		Element* __restrict__ dV_base,
		int num_rows,
		int num_cols,
		int row_stride) {

	constexpr int kVecSize = sizeof(uint4) / sizeof(Element);

	int warp_id    = threadIdx.x / 32;
	int lane       = threadIdx.x % 32;
	int dense_warp = (warp_id < 2) ? warp_id : warp_id - 2;
	int tid        = dense_warp * 32 + lane;

	// Total vectors across all rows
	int cols_per_vec_row = num_cols / kVecSize;
	int total_vecs = num_rows * cols_per_vec_row;

	for (int i = tid; i < total_vecs; i += kSiluBwdThreads) {
		int r = i / cols_per_vec_row;
		int c = i % cols_per_vec_row;
		int off = r * row_stride / kVecSize + c;  // offset in uint4 units

		auto* dZ_v = reinterpret_cast<const uint4*>(dZ_base);
		auto* U_v  = reinterpret_cast<const uint4*>(U_base);
		auto* V_v  = reinterpret_cast<const uint4*>(V_base);
		auto* dU_v = reinterpret_cast<uint4*>(dU_base);
		auto* dV_v = reinterpret_cast<uint4*>(dV_base);

		uint4 dz_vec = dZ_v[off];
		uint4 u_vec  = U_v[off];
		uint4 v_vec  = V_v[off];

		auto* dz_e = reinterpret_cast<Element*>(&dz_vec);
		auto* u_e  = reinterpret_cast<Element*>(&u_vec);
		auto* v_e  = reinterpret_cast<Element*>(&v_vec);

		uint4 du_out, dv_out;
		auto* du_e = reinterpret_cast<Element*>(&du_out);
		auto* dv_e = reinterpret_cast<Element*>(&dv_out);

		CUTE_UNROLL
		for (int j = 0; j < kVecSize; ++j) {
			float dz = static_cast<float>(dz_e[j]);
			float u  = static_cast<float>(u_e[j]);
			float v  = static_cast<float>(v_e[j]);
			float f = fast_sigmoid(u);
			float df = f * (1.0f + u * (1.0f - f));

			dv_e[j] = from_float<Element>(dz * u * f);
			du_e[j] = from_float<Element>(dz * v * df);
		}

		dV_v[off] = dv_out;
		dU_v[off] = du_out;
	}
}

// ═══════════════════════════════════════════════════════════════════
// Pair-multiply variant: dU = dZ * U', dV = dZ * V'
//
// Used by the fused mlp_bwd Phase 1c. Reads U' = V·silu'(U) and
// V' = silu(U) (already stored by Phase 1a) plus dZ (just stored
// by Phase 1b'); writes dU/dV in place.
//
// **n-split**: caller passes (split_idx, num_splits) matching the
// blockIdx.y / gridDim.y partition that Phase 1b' used. Each CTA
// processes ONLY its own n-tile slice — the same slice it wrote in
// Phase 1b'. This keeps Phase 1c reads intra-CTA so an intra-CTA
// NamedBarrier between 1b' and 1c is sufficient. Cross-CTA dU/dV
// visibility for downstream phases is handled by the existing
// mlp_global_barrier.
//
// The elementwise pair phase uses the historical 9-warp work partition
// (warps 0, 4-11 = 288 threads). SM100 warp 3 enters the BWD MLP path for
// UMMA phases, but skips this elementwise helper.
// ═══════════════════════════════════════════════════════════════════

static constexpr int kSiluBwdPairThreads = 288;  // 9 warps

template <typename Element>
__device__ __forceinline__ void silu_bwd_pair_tile(
		const Element* __restrict__ dZ_base,    // gmem [num_tokens, I]
		const Element* __restrict__ Up_base,    // U' coefficient
		const Element* __restrict__ Vp_base,    // V' coefficient
		Element* __restrict__ dU_base,          // output (may alias dZ_base)
		Element* __restrict__ dV_base,          // output (may alias Vp_base)
		int m,
		int TileM,
		int TileN,
		int intermediate_dim,
		int split_idx,                          // = blockIdx.y
		int num_splits) {                       // = gridDim.y

	constexpr int kVecSize = sizeof(uint4) / sizeof(Element);

	// Dense thread ID across 9 participating warps (0, 4-11). Warps 1-2 are
	// BWD comm warps; warp 3 is reserved for SM100 UMMA phases and does not
	// contribute to this elementwise work mapping.
	int warp_id    = threadIdx.x / 32;
	if (warp_id >= 1 && warp_id <= 3) return;
	int lane       = threadIdx.x % 32;
	int dense_warp = (warp_id == 0) ? 0 : warp_id - 3;
	int tid        = dense_warp * 32 + lane;

	int num_n_tiles    = (intermediate_dim + TileN - 1) / TileN;
	int row_start      = m * TileM;
	int vec_per_tile_n = TileN / kVecSize;                  // vecs per row of one n-tile
	int vec_row_stride = intermediate_dim / kVecSize;       // vecs per row in full I-dim
	int vecs_per_tile  = TileM * vec_per_tile_n;            // total vecs in one (TileM, TileN) tile

	auto* dZ_v = reinterpret_cast<const uint4*>(dZ_base);
	auto* Up_v = reinterpret_cast<const uint4*>(Up_base);
	auto* Vp_v = reinterpret_cast<const uint4*>(Vp_base);
	auto* dU_v = reinterpret_cast<uint4*>(dU_base);
	auto* dV_v = reinterpret_cast<uint4*>(dV_base);

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {
		int col_vec_start = n * vec_per_tile_n;
		int valid_col_vecs = min(vec_per_tile_n, vec_row_stride - col_vec_start);

		for (int i = tid; i < vecs_per_tile; i += kSiluBwdPairThreads) {
			int local_row     = i / vec_per_tile_n;
			int local_col_vec = i % vec_per_tile_n;
			if (local_col_vec >= valid_col_vecs) continue;

			size_t off = (size_t)(row_start + local_row) * vec_row_stride
			           + col_vec_start + local_col_vec;

			uint4 dz = dZ_v[off];
			uint4 up = Up_v[off];
			uint4 vp = Vp_v[off];

			auto* dz_e = reinterpret_cast<Element*>(&dz);
			auto* up_e = reinterpret_cast<Element*>(&up);
			auto* vp_e = reinterpret_cast<Element*>(&vp);

			uint4 du, dv;
			auto* du_e = reinterpret_cast<Element*>(&du);
			auto* dv_e = reinterpret_cast<Element*>(&dv);

			CUTE_UNROLL
			for (int j = 0; j < kVecSize; ++j) {
				float dz_f = static_cast<float>(dz_e[j]);
				float up_f = static_cast<float>(up_e[j]);
				float vp_f = static_cast<float>(vp_e[j]);
				du_e[j] = from_float<Element>(dz_f * up_f);
				dv_e[j] = from_float<Element>(dz_f * vp_f);
			}

			dU_v[off] = du;
			dV_v[off] = dv;
		}
	}
}

} // namespace liger
