#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 2 (single-tile, fused):  Y = Z @ A^T  (down-projection)
// ═══════════════════════════════════════════════════════════════════
//
// Single (TileM, TileN) producer/consumer building blocks for callers
// that drive the outer M-tile loop themselves: the standalone MLP2
// benchmark (mlp2.cu) and the fused MLP kernel (mlp.cuh).
//
// Design:
//   - Cooperative 2-WG consumer; split axis selected by TileM:
//       TileM=128  → Layout<_2,_1,_1>, split along M (each WG = 64×TileN)
//       TileM=64   → Layout<_1,_2,_1>, split along N (each WG = 64×TileN/2)
//   - One accumulator per WG
//   - Per-WG store_buf with EpiChunkN-round epilogue
//   - K_PIPE_MMAS = 1 (CUTLASS-canonical, see
//     sm90_mma_tma_gmma_ss_warpspecialized.hpp:264)
//   - Single fused Z+W TMA pipe (1 mbarrier per stage, 2 TMA copies)
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for Z / A)
//   WG0 warps 1-3  : Idle
//   WG1 (warps 4-7)  : Consumer atom 0
//   WG2 (warps 8-11) : Consumer atom 1
//
// Per `expert_ids`: one int per TileM-row m-block.
//
// Mlp2FusedSmem omits the pipeline storage so callers can fuse it with
// other phases' smem (see mlp.cuh). The benchmark wrapper (mlp2.cu) is
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
struct Mlp2Traits {
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

	using SmemLayoutZ_1 = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileM>, Int<TileK>>{}));
	using SmemLayoutW_1 = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileN>, Int<TileK>>{}));

	using SmemLayoutZ = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
	using SmemLayoutW = decltype(tile_to_shape(SmemLayoutAtom{},
		Shape<Int<TileN>, Int<TileK>, Int<Stages>>{}));

	// Per-WG store buffer slot: (AtomTileM, EpiChunkN).
	using SmemLayoutStoreSlot = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	using SmemLayoutStore_1 = SmemLayoutStoreSlot;

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	static constexpr int TmaTransBytesZ =
		static_cast<int>(size(SmemLayoutZ_1{}) * sizeof(Element));
	static constexpr int TmaTransBytesW =
		static_cast<int>(size(SmemLayoutW_1{}) * sizeof(Element));
	// Fused pipeline: Z + W share one mbarrier per stage.
	static constexpr int TmaTransBytes = TmaTransBytesZ + TmaTransBytesW;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumConsumers    = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads      = 384;
};

// ═══════════════════════════════════════════════════════════════════
// Fused MLP2 shared memory — no pipeline storage (managed externally)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp2FusedSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element smem_W[smem_W_size];
	// Per-WG store buffer: WG1 uses store_buf[0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
	CUTE_DEVICE Element* W_data() { return &smem_W[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Pipeline construction helper — single fused Z+W pipeline
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
__device__ __forceinline__ auto mlp2_make_pipe(
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
// Single fused Z + W TMA pipe per k-step.
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp2_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2FusedSmem<Traits>& smem,
		TmaLoadZ const& tma_load_z,
		TmaLoadW const& tma_load_a,
		int m,
		int expert_n_offset,
		int num_tokens,
		int intermediate_dim,
		int total_n_rows,
		int num_n_tiles,
		int num_k_tiles,
		// N-split identity (flat-grid launch); -1 → blockIdx.y / gridDim.y.
		int split_idx = -1,
		int num_splits = -1) {

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	// int64_t cast: total_n_rows × intermediate_dim and num_tokens ×
	// intermediate_dim both exceed INT_MAX at production shapes. CUTE's
	// layout math otherwise wraps to a negative byte offset → IMA.
	auto mZ = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(intermediate_dim)));
	auto mA = tma_load_a.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(intermediate_dim)));

	auto cta_tma_z = tma_load_z.get_slice(Int<0>{});
	auto cta_tma_a = tma_load_a.get_slice(Int<0>{});

	auto tZsZ = cta_tma_z.partition_D(sZ);
	auto tWsW = cta_tma_a.partition_D(sW);

	auto gZ = local_tile(mZ,
		make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
		make_coord(m, _));
	auto tZgZ = cta_tma_z.partition_S(gZ);

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {
		auto gW = local_tile(mA,
			make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
			make_coord(expert_n_offset + n, _));
		auto tWgW = cta_tma_a.partition_S(gW);

		for (int k = 0; k < num_k_tiles; ++k) {
			pipe.producer_acquire(state);
			if (threadIdx.x == 0) {
				auto* bar = pipe.producer_get_barrier(state);
				copy(tma_load_z.with(*bar, 0),
					tZgZ(_, _, _, k), tZsZ(_, _, _, state.index()));
				copy(tma_load_a.with(*bar, 0),
					tWgW(_, _, _, k), tWsW(_, _, _, state.index()));
			}
			++state;
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Fused consumer — cooperative M-split, single acc per WG, per-WG epi
// ═══════════════════════════════════════════════════════════════════
//
// y_m is the M-tile index into the output Y buffer. The two WGs split
// the (TileM, TileN) output along M: WG1 writes rows [0, AtomTileM),
// WG2 writes [AtomTileM, TileM).

template <typename Traits, typename Pipeline, typename TmaStoreY>
__device__ __forceinline__ void mlp2_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2FusedSmem<Traits>& smem,
		TmaStoreY const& tma_store_y,
		int y_m,                // M-tile index into Y buffer
		int hidden_dim,
		int num_y_m_tiles,      // number of M-tile slots in Y buffer
		int num_n_tiles,
		int num_k_tiles,
		// N-split identity (flat-grid launch); -1 → blockIdx.y / gridDim.y.
		int split_idx = -1,
		int num_splits = -1) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

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
		typename Traits::SmemLayoutStoreSlot{});

	// int64_t cast: num_y_m_tiles · TileM · hidden_dim can exceed
	// INT_MAX at large output buffers; cast keeps CUTE's layout math
	// in 64-bit so byte offsets don't wrap.
	auto mY = tma_store_y.get_tma_tensor(
		make_shape(static_cast<int64_t>(num_y_m_tiles) * Traits::TileM,
		           static_cast<int64_t>(hidden_dim)));
	auto cta_tma_y = tma_store_y.get_slice(Int<0>{});

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;
	constexpr int K_PIPE_MMAS = 1;  // CUTLASS-matching

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		clear(acc);

		auto state_release = state;
		int prologue_count = (num_k_tiles < K_PIPE_MMAS) ? num_k_tiles : K_PIPE_MMAS;

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
		for (int k = prologue_count; k < num_k_tiles; ++k) {
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
		// TileM=128 (M-split): WG_w owns rows [w*64, (w+1)*64), full N.
		// TileM=64  (N-split): WG_w owns full M, cols [w*WgTileN, ...).
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
					// TileM=128: 2 row tiles per logical M tile.
					m_tile_idx = 2 * y_m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					// TileM=64: 1 row tile per logical M tile; WG_w handles
					// its N-half.
					m_tile_idx = y_m;
					n_tile_idx = n * (Traits::TileN / Traits::EpiChunkN)
					           + my_wg * Traits::NumEpiRounds + r;
				}
				auto gY = local_tile(mY,
					make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
					make_coord(m_tile_idx, n_tile_idx));
				copy(tma_store_y, cta_tma_y.partition_S(sStore),
					cta_tma_y.partition_D(gY));
				cute::tma_store_arrive();
			}
			store_in_flight = true;
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

} // namespace liger
