#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 5 (single-tile API for fused use)
// ═══════════════════════════════════════════════════════════════════
//
// Same kernel body as mlp5.cuh's nested-loop default, exposed as a
// single-(m)-tile producer + consumer so a parent kernel can drive the
// outer m loop itself (e.g. mlp_bwd.cuh composing Phase 1d).
//
// Mirrors how mlp1_fused.cuh exposes single-tile entry points: callers
// own the Pipeline construction (use
// mlp5_make_pipe from mlp5.cuh) and the Mlp5Smem allocation; this
// header just provides the producer/consumer body parameterized for
// ONE m-tile.
//
// Params unique to fused use:
//   - m               : the m-tile this invocation processes
//   - expert_k_offset : expert · num_k_tiles (caller resolves the
//                       expert lookup once per m)
//   - split_idx       : the column index within an N-split (caller
//                       picks; analogous to blockIdx.y in the
//                       standalone driver)
//   - num_splits      : total columns (analogous to gridDim.y)
//
// Output store goes to a global dX tensor view of shape
// (num_m_tiles · TileM, hidden_dim) (same as the standalone consumer).
// ═══════════════════════════════════════════════════════════════════

#include "mlp5.cuh"  // Mlp5Traits, Mlp5Smem, mlp5_make_pipe

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Producer (single m-tile) — fills 2·num_k_tiles slots per (m, n)
// across n ∈ {split_idx, split_idx+num_splits, ...}.
//   Phase 1 (k=0..num_k_tiles-1):  Z=dU, W=B
//   Phase 2 (k=num_k_tiles..2K-1): Z=dV, W=C
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp5_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaLoadZ const& tma_load_du,
		TmaLoadZ const& tma_load_dv,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		int m,
		int expert_k_offset,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int total_k_cols,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

	// int64_t cast for production shapes where T · I or E · I · H can
	// exceed 2^31 elements (same rationale as in mlp1_fused_act.cuh and
	// the int64 commit on the fwd path).
	auto mDU = tma_load_du.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(intermediate_dim)));
	auto mDV = tma_load_dv.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(intermediate_dim)));
	auto mB  = tma_load_b.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(total_k_cols)));
	auto mC  = tma_load_c.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(total_k_cols)));

	auto cta_tma_du = tma_load_du.get_slice(Int<0>{});
	auto cta_tma_b  = tma_load_b.get_slice(Int<0>{});

	auto tZsZ = cta_tma_du.partition_D(sZ);
	auto tWsW = cta_tma_b.partition_D(sW);

	auto gB_all = local_tile(mB,
		make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
		make_coord(_, _));
	auto gC_all = local_tile(mC,
		make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
		make_coord(_, _));
	auto tBgB_all = cta_tma_b.partition_S(gB_all);
	auto tCgC_all = cta_tma_b.partition_S(gC_all);

	auto gDU = local_tile(mDU,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgDU = cta_tma_du.partition_S(gDU);
	auto gDV = local_tile(mDV,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgDV = cta_tma_du.partition_S(gDV);

	const bool is_leader = (threadIdx.x == 0);

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {
		// Phase 1: dU @ B
		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (is_leader) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_du.with(*bar, 0),
					tZgDU(_, _, _, k), tZsZ(_, _, _, state.index()));
				copy(tma_load_b.with(*bar, 0),
					tBgB_all(_, _, _, n, expert_k_offset + k),
					tWsW(_, _, _, state.index()));
			}
			++state;
		}
		// Phase 2: dV @ C
		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (is_leader) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_dv.with(*bar, 0),
					tZgDV(_, _, _, k), tZsZ(_, _, _, state.index()));
				copy(tma_load_c.with(*bar, 0),
					tCgC_all(_, _, _, n, expert_k_offset + k),
					tWsW(_, _, _, state.index()));
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Consumer (single m-tile) — cooperative M-split, dual-phase acc
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename Pipeline, typename TmaStoreDX>
__device__ __forceinline__ void mlp5_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaStoreDX const& tma_store_dx,
		int m,
		int hidden_dim,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

	auto tCsZ = thr_mma.partition_A(sZ);
	auto tCsW = thr_mma.partition_B(sW);

	auto acc = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;
	const int my_barrier_id = 1 + my_wg;
	const bool is_my_wg_leader = (tid_in_wg == 0);

	constexpr int store_slot_elems = Traits::AtomTileM * Traits::EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStore{});

	auto mDX = tma_store_dx.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_m_tiles) * Traits::TileM,
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma_dx = tma_store_dx.get_slice(Int<0>{});

	int total_k = 2 * num_k_tiles;
	constexpr int K_PIPE_MMAS = 1;
	bool store_in_flight = false;

	for (int n = split_idx; n < num_n_tiles; n += num_splits) {

		clear(acc);
		auto state_release = state;
		int prologue_count = (total_k < K_PIPE_MMAS) ? total_k : K_PIPE_MMAS;

		// Prologue
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc);
			warpgroup_arrive();
			gemm(tiled_mma, tCsZ(_, _, _, state.index()),
				tCsW(_, _, _, state.index()), acc);
			warpgroup_commit_batch();
			++state;
		}
		// Steady state
		for (int k = prologue_count; k < total_k; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc);
			warpgroup_arrive();
			gemm(tiled_mma, tCsZ(_, _, _, state.index()),
				tCsW(_, _, _, state.index()), acc);
			warpgroup_commit_batch();

			warpgroup_wait<K_PIPE_MMAS>();
			warpgroup_fence_operand(acc);
			pipe.consumer_release(state_release);
			++state;
			++state_release;
		}
		// Drain
		warpgroup_wait<0>();
		warpgroup_fence_operand(acc);
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_release(state_release);
			++state_release;
		}

		// ── Epilogue ──────────────────────────────────────
		CUTE_UNROLL
		for (int r = 0; r < Traits::NumEpiRounds; ++r) {
			if (store_in_flight)
				cute::tma_store_wait<0>();

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			int chunk_start = r * Traits::EpiChunkN;
			CUTE_UNROLL
			for (int i = 0; i < size(acc); ++i) {
				auto coord = tCcC(i);
				int m_loc = get<0>(coord);
				int n_loc = get<1>(coord);
				int m_local, n_local;
				if constexpr (Traits::kMSplit) {
					m_local = m_loc - my_wg * Traits::AtomTileM;
					n_local = n_loc;
				} else {
					m_local = m_loc;
					n_local = n_loc - my_wg * Traits::WgTileN;
				}
				if (n_local >= chunk_start &&
				    n_local <  chunk_start + Traits::EpiChunkN) {
					int chunk_n = n_local - chunk_start;
					sStore(m_local, chunk_n) = static_cast<Element>(acc(i));
				}
			}

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			if (is_my_wg_leader) {
				cute::tma_store_fence();
				int m_tile_idx, n_tile_idx;
				if constexpr (Traits::kMSplit) {
					m_tile_idx = 2 * m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					m_tile_idx = m;
					n_tile_idx = n * (Traits::TileN / Traits::EpiChunkN)
					           + my_wg * Traits::NumEpiRounds + r;
				}
				auto gDX = local_tile(mDX,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(m_tile_idx, n_tile_idx));
				copy(tma_store_dx, cta_tma_dx.partition_S(sStore),
					cta_tma_dx.partition_D(gDX));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

} // namespace liger
