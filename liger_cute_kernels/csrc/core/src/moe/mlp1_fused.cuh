#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 1 (single-tile, fused):  Z = SiLU(B @ X) * (C @ X)
// ═══════════════════════════════════════════════════════════════════
//
// Single (TileM, TileN) producer/consumer building blocks for callers
// that drive the outer M-tile loop themselves: the standalone MLP1
// benchmark (mlp1.cu), the fused MLP kernel (mlp.cuh), the activation
// variants (mlp1_fused_act.cuh), and the backward path (mlp_bwd.cuh).
//
// Design:
//   - Cooperative 2-WG consumer; split axis selected by TileM:
//       TileM=128  → Layout<_2,_1,_1>, split along M (each WG = 64×TileN)
//       TileM=64   → Layout<_1,_2,_1>, split along N (each WG = 64×TileN/2)
//   - Two accumulators per WG (acc_B = U = B@X, acc_C = V = C@X)
//   - Per-thread SiLU(U)*V fusion (no cross-WG sync)
//   - Per-WG store_buf with EpiChunkN-round epilogue
//   - K_PIPE_MMAS = 1 (CUTLASS-canonical, see
//     sm90_mma_tma_gmma_ss_warpspecialized.hpp:264)
//   - Single fused X+W1+W2 TMA pipe (1 mbarrier per stage, 3 TMA copies)
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for X / B / C)
//   WG0 warps 1-3  : Idle
//   WG1 (warps 4-7)  : Consumer atom 0
//   WG2 (warps 8-11) : Consumer atom 1
//
// Per `expert_ids`: one int per TileM-row m-block.
//
// Mlp1FusedSmem omits the pipeline storage so callers can fuse it with
// other phases' smem (see mlp.cuh). The benchmark wrapper (mlp1.cu) is
// responsible for attaching MainloopPipeline::SharedStorage and running
// the outer for(m) loop over m-tiles.
//
// ═══════════════════════════════════════════════════════════════════

#include "models.cuh"
#include "math.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// Traits
// ═══════════════════════════════════════════════════════════════════

template <
	typename Element_,
	int TileM_      = 128,
	int TileN_      = 128,
	int TileK_      = 64,
	int Stages_     = 4,
	int EpiChunkN_  = 32
>
struct Mlp1Traits {
	using Element      = Element_;
	using ElementAccum = float;

	static constexpr int TileM      = TileM_;
	static constexpr int TileN      = TileN_;
	static constexpr int TileK      = TileK_;
	static constexpr int Stages     = Stages_;
	static constexpr int EpiChunkN  = EpiChunkN_;

	static_assert(TileM == 64 || TileM == 128, "TileM must be 64 or 128");

	// Hopper WGMMA atom has m=64. With TileM=128 the two WGs cooperate
	// along M (each owns 64 rows × TileN). With TileM=64 the WGs
	// cooperate along N (each owns 64 rows × TileN/2). AtomTileM is the
	// per-WG M extent — always 64.
	static constexpr bool kMSplit  = (TileM == 128);
	static constexpr int  AtomTileM = 64;
	static constexpr int  WgTileN   = kMSplit ? TileN : (TileN / 2);

	static_assert(TileN <= 256,             "TileN must fit TMA box limit");
	static_assert(WgTileN % EpiChunkN == 0, "EpiChunkN must divide WgTileN");
	static_assert(kMSplit || (TileN % 2 == 0),
		"TileN must be even when TileM=64 (cooperative N-split halves)");
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;

	// Cooperative 2-WG layout: M-split for TileM=128, N-split for TileM=64.
	// AtomN is the per-WG WGMMA atom N — combined across both WGs it must
	// cover exactly TileN cols (= sW's N extent). With WgLayout=<1,2,1>
	// and AtomN=TileN, the TiledMMA would cover 2·TileN cols and WG1's
	// partition_B(sW) slice would read past sW's TileN-wide buffer → IMA.
	static constexpr int AtomN = kMSplit ? TileN : (TileN / 2);
	using GmmaAtom = typename GmmaSelector<Element, AtomN>::Atom;
	using WgLayout = cute::conditional_t<
		kMSplit,
		Layout<Shape<_2, _1, _1>>,
		Layout<Shape<_1, _2, _1>>>;
	using TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>;

	using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;

	using SmemLayoutX_1 = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileM>, Int<TileK>>{}));
	using SmemLayoutW_1 = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileN>, Int<TileK>>{}));

	using SmemLayoutX = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
	using SmemLayoutW = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileN>, Int<TileK>, Int<Stages>>{}));

	// Per-WG store buffer slot: (AtomTileM, EpiChunkN). One slot per WG.
	// Epi writes the WG's m-strip in EpiChunkN-wide rounds.
	using SmemLayoutStoreSlot = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	using SmemLayoutStore_1 = SmemLayoutStoreSlot;

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	static constexpr int TmaTransBytesX =
		static_cast<int>(size(SmemLayoutX_1{}) * sizeof(Element));
	static constexpr int TmaTransBytesW =
		static_cast<int>(size(SmemLayoutW_1{}) * sizeof(Element));
	// Fused pipeline: X + W1 + W2 share one mbarrier per stage.
	static constexpr int TmaTransBytes = TmaTransBytesX + 2 * TmaTransBytesW;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumConsumers    = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads      = 384;
};

// ═══════════════════════════════════════════════════════════════════
// Expert weight pointers (device-side)
// ═══════════════════════════════════════════════════════════════════

template <typename Element>
struct ExpertWeights1 {
	const Element* B;  // [intermediate_dim, hidden_dim]
	const Element* C;  // [intermediate_dim, hidden_dim]
};

// ═══════════════════════════════════════════════════════════════════
// Fused MLP1 shared memory — no pipeline storage (managed externally)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp1FusedSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_X_size     = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_X[smem_X_size];
	alignas(128) Element smem_W1[smem_W_size];
	alignas(128) Element smem_W2[smem_W_size];
	// Per-WG store buffer: WG1 uses store_buf[0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	CUTE_DEVICE Element* X_data()  { return &smem_X[0]; }
	CUTE_DEVICE Element* W1_data() { return &smem_W1[0]; }
	CUTE_DEVICE Element* W2_data() { return &smem_W2[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Pipeline construction helper — single fused X+W1+W2 pipeline
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
__device__ __forceinline__ auto mlp1_make_pipe(
		typename Traits::MainloopPipeline::SharedStorage& storage) {

	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	typename Pipeline::Params pp;
	pp.transaction_bytes = Traits::TmaTransBytes;
	pp.num_producers = 1;
	pp.num_consumers = Traits::ConsumerThreads;
	if (is_producer) {
		pp.role = Category::Producer;
		pp.is_leader = (threadIdx.x == 0);
	} else if (is_consumer) {
		pp.role = Category::Consumer;
	} else {
		pp.role = Category::NonParticipant;
	}
	return Pipeline(storage, pp, Shape<_1, _1, _1>{},
		cute::true_type{}, cute::true_type{});
}

// ═══════════════════════════════════════════════════════════════════
// Fused producer — single (m), all (n) under blockIdx.y stride.
// Single fused X + W1 + W2 TMA pipe per k-step.
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename Pipeline,
          typename TmaLoadX, typename TmaLoadW>
__device__ __forceinline__ void mlp1_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedSmem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		int m,
		int expert_n_offset,
		int num_tokens,
		int hidden_dim,
		int total_n_rows,
		int num_n_tiles,
		int num_k_tiles,
		// N-split identity (flat-grid launch). Defaults (-1) fall back to
		// blockIdx.y / gridDim.y for standalone 2-D-grid callers (mlp1.cu).
		int split_idx = -1,
		int num_splits = -1) {

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	// int64_t cast: total_n_rows × hidden_dim can exceed INT_MAX at
	// production shapes (epp · I · D > 2^31), and CUTE's layout math
	// otherwise wraps to a negative byte offset → kernel IMA.
	auto mX = tma_load_x.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	auto mB = tma_load_b.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(hidden_dim)));
	auto mC = tma_load_c.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(hidden_dim)));

	auto cta_tma_x = tma_load_x.get_slice(Int<0>{});
	auto cta_tma_b = tma_load_b.get_slice(Int<0>{});
	auto cta_tma_c = tma_load_c.get_slice(Int<0>{});

	auto tXsX   = cta_tma_x.partition_D(sX);
	auto tW1sW1 = cta_tma_b.partition_D(sW1);
	auto tW2sW2 = cta_tma_c.partition_D(sW2);

	auto gX = local_tile(mX,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tXgX = cta_tma_x.partition_S(gX);

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {
		auto gB = local_tile(mB,
			make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
			make_coord(expert_n_offset + n, _));
		auto gC = local_tile(mC,
			make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
			make_coord(expert_n_offset + n, _));
		auto tBgB = cta_tma_b.partition_S(gB);
		auto tCgC = cta_tma_c.partition_S(gC);

		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (threadIdx.x == 0) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_x.with(*bar, 0),
					tXgX(_, _, _, k), tXsX(_, _, _, state.index()));
				copy(tma_load_b.with(*bar, 0),
					tBgB(_, _, _, k), tW1sW1(_, _, _, state.index()));
				copy(tma_load_c.with(*bar, 0),
					tCgC(_, _, _, k), tW2sW2(_, _, _, state.index()));
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Fused consumer — cooperative M-split, dual accumulators, per-WG epi
// ═══════════════════════════════════════════════════════════════════
//
// z_m is the M-tile index into the output Z buffer. The two WGs split
// the (TileM, TileN) output along M: WG1 writes rows [0, AtomTileM)
// of tile z_m, WG2 writes [AtomTileM, TileM).

template <typename Traits, typename Pipeline, typename TmaStoreZ>
__device__ __forceinline__ void mlp1_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedSmem<Traits>& smem,
		TmaStoreZ const& tma_store_z,
		int z_m,                // M-tile index into Z buffer
		int intermediate_dim,
		int num_z_m_tiles,      // number of M-tile slots in Z buffer
		int num_n_tiles,
		int num_k_tiles,
		// N-split identity (flat-grid launch); -1 → blockIdx.y / gridDim.y.
		int split_idx = -1,
		int num_splits = -1) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;   // 0..255
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;    // 0..127
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	auto tCsX  = thr_mma.partition_A(sX);
	auto tCsW1 = thr_mma.partition_B(sW1);
	auto tCsW2 = thr_mma.partition_B(sW2);

	// Two accumulators per WG, each covering the WG's (AtomTileM, TileN) m-strip.
	auto acc_B = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});
	auto acc_C = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	// Identity coords for per-thread acc → store_buf mapping.
	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;  // 0 or 1
	const int my_barrier_id = 1 + my_wg;                          // 1 or 2
	const bool is_my_wg_leader = (tid_in_wg == 0);

	constexpr int store_slot_elems = Traits::AtomTileM * Traits::EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	// int64_t cast: num_z_m_tiles · TileM · intermediate_dim exceeds
	// INT_MAX once grid_x · ZBuf · TileM · I > 2^31 (e.g. ZBuf=16
	// at I=16384 wraps and IMAs).
	auto mZ = tma_store_z.get_tma_tensor(
		make_shape(static_cast<int64_t>(num_z_m_tiles) * Traits::TileM,
		           static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_z = tma_store_z.get_slice(Int<0>{});

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;
	constexpr int K_PIPE_MMAS = 1;  // CUTLASS-matching

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		clear(acc_B);
		clear(acc_C);

		auto state_release = state;
		int prologue_count = (num_k_tiles < K_PIPE_MMAS) ? num_k_tiles : K_PIPE_MMAS;

		// Prologue
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			warpgroup_arrive();
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW1(_, _, _, state.index()), acc_B);
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW2(_, _, _, state.index()), acc_C);
			warpgroup_commit_batch();
			++state;
		}
		// Steady state
		for (int k = prologue_count; k < num_k_tiles; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			warpgroup_arrive();
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW1(_, _, _, state.index()), acc_B);
			gemm(tiled_mma, tCsX(_, _, _, state.index()),
				tCsW2(_, _, _, state.index()), acc_C);
			warpgroup_commit_batch();

			warpgroup_wait<K_PIPE_MMAS>();
			warpgroup_fence_operand(acc_B);
			warpgroup_fence_operand(acc_C);
			pipe.consumer_release(state_release);
			++state;
			++state_release;
		}
		// Drain
		warpgroup_wait<0>();
		warpgroup_fence_operand(acc_B);
		warpgroup_fence_operand(acc_C);
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_release(state_release);
			++state_release;
		}

		// SiLU fusion: acc_B = SiLU(U) * V (per-thread, no sync).
		CUTE_UNROLL
		for (int i = 0; i < size(acc_B); ++i) {
			float u = acc_B(i);
			float v = acc_C(i);
			acc_B(i) = fast_silu(u) * v;
		}

		// ── Epilogue ──────────────────────────────────────
		// Each WG writes its (AtomTileM=64) × EpiChunkN slot to store_buf
		// and issues TMA stores. NumEpiRounds rounds per WG. The per-thread
		// acc element → (m_local, n_local) mapping depends on the split:
		//   TileM=128 (M-split): WG_w owns rows [w*64, (w+1)*64), full N
		//   TileM=64  (N-split): WG_w owns full M, cols [w*WgTileN, ...)
		CUTE_UNROLL
		for (int r = 0; r < Traits::NumEpiRounds; ++r) {
			if (store_in_flight)
				cute::tma_store_wait<0>();

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			int chunk_start = r * Traits::EpiChunkN;
			CUTE_UNROLL
			for (int i = 0; i < size(acc_B); ++i) {
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
					sStore(m_local, chunk_n) = static_cast<Element>(acc_B(i));
				}
			}

			cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

			if (is_my_wg_leader) {
				cute::tma_store_fence();
				int m_tile_idx, n_tile_idx;
				if constexpr (Traits::kMSplit) {
					// TileM=128: row tile space = 2 tiles per logical M
					// (AtomTileM=64). WG_w writes (2*z_m + w, n*Rounds + r).
					m_tile_idx = 2 * z_m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					// TileM=64: 1 row tile per logical M. WG_w handles its
					// N-half: chunk index = n*(TileN/EpiChunkN)
					//                     + w*NumEpiRounds + r.
					m_tile_idx = z_m;
					n_tile_idx = n * (Traits::TileN / Traits::EpiChunkN)
					           + my_wg * Traits::NumEpiRounds + r;
				}
				auto gZ = local_tile(mZ,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(m_tile_idx, n_tile_idx));
				copy(tma_store_z, cta_tma_z.partition_S(sStore),
					cta_tma_z.partition_D(gZ));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

} // namespace liger
