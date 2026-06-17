#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 3:  dA = dY^T @ Z   (weight gradient, backward)
// ═══════════════════════════════════════════════════════════════════
//
// TMA_REDUCE_ADD variant. Cooperative 2-WG consumer; the LARGER of
// (TileM, TileN) is the cooperation axis. Only two configs supported:
//   (TileM=256, TileN=128) → M-split (Layout<_2,_1,_1>)
//   (TileM=128, TileN=256) → N-split (Layout<_1,_2,_1>)  ← default
//
// Per-WG output footprint = (WgTileM, WgTileN) = (128, 128) in both
// cases — each WG covers 128 rows × 128 cols per K-step via 2 wgmma
// m=64 issues, matching today's non-coop dual-WG per-WG work.
//
// Single fused DYT + Z TMA pipe (one mbarrier per stage, 2 TMA copies).
// Consumer computes a per-WG partial dA[m,n] in registers, casts to
// smem store_buf, then issues SM90_TMA_REDUCE_ADD into gmem — hardware
// atomically adds the partial. dA starts zero (torch::zeros / bwd host).
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for DYT / Z)
//   WG0 warps 1-3  : Idle
//   WG1 (warps 4-7)  : Consumer atom 0
//   WG2 (warps 8-11) : Consumer atom 1
//
// Chunk-fixed walk: each CTA owns ONE (chunk, lane) cell. We hold the
// SHARED (cooperative) operand chunk-constant and walk the cooperation
// (split) axis, maximizing L2 reuse of the operand both WGs consume:
//   kMSplit=true  (M-split, Z shared):    chunk = (e, n_tile), walks m_tile; holds Z(n,*)
//   kMSplit=false (N-split, dY^T shared): chunk = (e, m_tile), walks n_tile; holds dY^T(m,*)
// ═══════════════════════════════════════════════════════════════════

#include "models.cuh"

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
	int TileM_     = 128,
	int TileN_     = 256,
	int TileK_     = 64,
	int Stages_    = 4,
	int EpiChunkN_ = 64
>
struct Mlp3Traits {
	using Element      = Element_;
	using ElementAccum = float;

	static constexpr int TileM     = TileM_;
	static constexpr int TileN     = TileN_;
	static constexpr int TileK     = TileK_;
	static constexpr int Stages    = Stages_;
	static constexpr int EpiChunkN = EpiChunkN_;

	// Cooperative 2-WG layout. Only two configs supported:
	//   (TileM=256, TileN=128) → kMSplit=true,  WgLayout<_2,_1,_1>
	//   (TileM=128, TileN=256) → kMSplit=false, WgLayout<_1,_2,_1>
	// Per-WG output footprint = (WgTileM=128, WgTileN=128) in both. Each
	// WG covers 2 wgmma m=64 atoms per K-step (atom_count_M = 2).
	static_assert((TileM == 256 && TileN == 128) ||
	              (TileM == 128 && TileN == 256),
		"Mlp3Traits supports only (TileM, TileN) = (256, 128) M-split "
		"or (128, 256) N-split");

	static constexpr bool kMSplit   = (TileM > TileN);
	static constexpr int  AtomTileM = 64;
	static constexpr int  WgTileM   = 128;
	static constexpr int  WgTileN   = 128;

	static_assert(WgTileN % EpiChunkN == 0,
		"EpiChunkN must divide WgTileN (=128)");
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;

	static constexpr int AtomN = WgTileN;
	using GmmaAtom = typename GmmaSelectorMNMN<Element, AtomN>::Atom;
	using WgLayout = cute::conditional_t<
		kMSplit,
		Layout<Shape<_2, _1, _1>>,
		Layout<Shape<_1, _2, _1>>>;
	using TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>;

	using SmemLayoutAtom = GMMA::Layout_MN_SW128_Atom<Element>;

	using SmemLayoutDYT_1 = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>>{},
		Step<_2, _1>{}));
	using SmemLayoutZ_1   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>>{},
		Step<_2, _1>{}));
	using SmemLayoutDYT = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>, Int<Stages>>{},
		Step<_2, _1, _3>{}));
	using SmemLayoutZ   = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>, Int<Stages>>{},
		Step<_2, _1, _3>{}));

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	static constexpr int TmaTransBytesDYT =
		static_cast<int>(size(SmemLayoutDYT_1{}) * sizeof(Element));
	static constexpr int TmaTransBytesZ =
		static_cast<int>(size(SmemLayoutZ_1{}) * sizeof(Element));
	// Fused pipeline: DYT + Z share one mbarrier per stage. Both WGs
	// consume the same Z via cooperative TiledMma slicing.
	static constexpr int TmaTransBytes = TmaTransBytesDYT + TmaTransBytesZ;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumConsumers    = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;   // 256
	static constexpr int NumThreads      = 384;

	// Per-WG store buffer slot: (WgTileM, EpiChunkN). One slot per WG; it
	// holds kAtomsPerWg = WgTileM/AtomTileM contiguous atom-rows after the
	// acc→smem fold.
	static constexpr int kAtomsPerWg = WgTileM / AtomTileM;   // 2
	using SmemLayoutStoreSlot = Layout<
		Shape<Int<WgTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	// TMA-reduce descriptor box = ONE wgmma atom-row (AtomTileM × EpiChunkN).
	// At TileM=256 each WG owns 2 *interleaved* atoms (rows {0-63,128-191} /
	// {64-127,192-255}) that map to non-contiguous gmem rows, so a WG issues
	// kAtomsPerWg separate TMA stores per epi round (mirrors mlp1_fused.cuh,
	// which uses an AtomTileM box but only ever has 1 atom/WG).
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

// ═══════════════════════════════════════════════════════════════════
// Shared Memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp3Smem {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size   = cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	// Per-WG store buffer: WG1 uses [0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	typename Traits::MainloopPipeline::SharedStorage pipe_storage;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data()   { return &smem_Z[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Fused MLP3 shared memory (data only — pipe storage lives in the
// outer MlpFusedBwdSmem union; otherwise identical to Mlp3Smem).
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp3FusedSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size   = cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element store_buf[2 * smem_store_size];

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data()   { return &smem_Z[0]; }
};


// ═══════════════════════════════════════════════════════════════════
// Pipe construction helper
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors mlp2/mlp5's make_pipe pattern: hides the role/transaction-
// byte setup so callers (mlp3_fwd, or any fused parent kernel) get a
// consistent Pipeline. Roles follow the standard mlp warp layout —
// warp 0 = TMA producer, warps 4-11 = consumers.

template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipeline
mlp3_make_pipe(typename Traits::MainloopPipeline::SharedStorage& storage) {
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
// Chunk-fixed cooperative producer/consumer
// ═══════════════════════════════════════════════════════════════════
//
// Used by the standalone mlp3 kernel (mlp3.cu) and mlp_bwd Phase 2.
// Each CTA owns ONE (chunk, lane) cell. We hold the SHARED operand and
// walk the cooperation (split) axis:
//   kMSplit=true  (M-split): chunk = (e, n_tile), lane partitions
//                   num_m_tiles, walks m_tile. Z(n_tile, *) — the shared
//                   N-side operand — is the L2-held chunk-constant.
//   kMSplit=false (N-split): chunk = (e, m_tile), lane partitions
//                   num_n_tiles, walks n_tile. dY^T(m_tile, *) — the
//                   shared M-side operand — is L2-held.
//
// Cell decode (chunk-major: cell_idx = chunk_idx × outer_split + lane):
//   total_cells = total_chunks × outer_split
//   cell        →  chunk_idx = cell_idx / outer_split
//                  lane      = cell_idx % outer_split

template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaLoadDYT, typename TmaLoadZ>
__device__ __forceinline__ void mlp3_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaLoadDYT const& tma_load_dyt,
		TmaLoadZ const& tma_load_z,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int cell_start,
		int cell_stride,
		int batch_kb_start,
		int batch_kb_end,
		int k_split,
		// Staging-ring depth in K-blocks. > 0 means batch_kb_*/the per-expert
		// k-ranges are UNWRAPPED and the gmem K-tile coordinate wraps by % ring_kb
		// (grouped grouped Phase-2 may straddle the ring). 0 = no wrap.
		int ring_kb = 0) {

	auto sDYT = make_tensor(make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ   = make_tensor(make_smem_ptr(smem.Z_data()),   typename Traits::SmemLayoutZ{});

	auto mDYT = tma_load_dyt.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(num_tokens)));
	auto mZT  = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));

	auto cta_tma_dyt = tma_load_dyt.get_slice(Int<0>{});
	auto cta_tma_z   = tma_load_z.get_slice(Int<0>{});

	auto tDYTsDYT = cta_tma_dyt.partition_D(sDYT);
	auto tZsZ     = cta_tma_z.partition_D(sZ);

	int total_chunks = num_experts * (Traits::kMSplit ? num_n_tiles : num_m_tiles);
	// K-split: each (chunk, walk-lane) is replicated k_split times; the
	// k_slice handles a 1/k_split sub-range of the expert's K-loop. Partial
	// reductions accumulate into dA via the TMA_REDUCE_ADD epilogue.
	int total_cells  = total_chunks * outer_split * k_split;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {

		int k_slice   = cell_idx % k_split;
		int cell_om   = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane      = cell_om - chunk_idx * outer_split;

		int e, m_chunk, n_chunk, walk_begin, walk_end;
		if constexpr (Traits::kMSplit) {
			// M-split: hold Z(n) shared. chunk = (e, n_tile), walks m_tile.
			e            = chunk_idx / num_n_tiles;
			n_chunk      = chunk_idx - e * num_n_tiles;
			m_chunk      = 0;
			int m_per    = num_m_tiles / outer_split;
			walk_begin   = lane * m_per;
			walk_end     = walk_begin + m_per;
		} else {
			// N-split: hold dY^T(m) shared. chunk = (e, m_tile), walks n_tile.
			e            = chunk_idx / num_m_tiles;
			m_chunk      = chunk_idx - e * num_m_tiles;
			n_chunk      = 0;
			int n_per    = num_n_tiles / outer_split;
			walk_begin   = lane * n_per;
			walk_end     = walk_begin + n_per;
		}

		// Intersect this expert's global K range with the current
		// batch's K-window. Global k_starts/ends were filled once per
		// pass; per-batch we just clip.
		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e],   batch_kb_end);
		if (kb_hi <= kb_lo) continue;

		// K-split: clip to this cell's k_slice sub-range of [kb_lo, kb_hi).
		{
			int k_total = kb_hi - kb_lo;
			int k_per   = (k_total + k_split - 1) / k_split;
			int s_lo    = kb_lo + k_slice * k_per;
			int s_hi    = min(s_lo + k_per, kb_hi);
			if (s_hi <= s_lo) continue;   // empty slice (e.g. straggler)
			kb_lo = s_lo;
			kb_hi = s_hi;
		}

		for (int w = walk_begin; w < walk_end; ++w) {
			int m_tile = Traits::kMSplit ? w       : m_chunk;
			int n_tile = Traits::kMSplit ? n_chunk : w;
			auto gDYT = local_tile(mDYT,
				make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
				make_coord(m_tile, _));
			auto gZ = local_tile(mZT,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(n_tile, _));
			auto tDYTgDYT = cta_tma_dyt.partition_S(gDYT);
			auto tZgZ     = cta_tma_z.partition_S(gZ);

			for (int kb = kb_lo; kb < kb_hi; ++kb) {
				int kb_g = (ring_kb > 0) ? (kb % ring_kb) : kb;  // physical ring slot
				pipe.producer_acquire(state);
				if (threadIdx.x == 0) {
					auto* bar = pipe.producer_get_barrier(state);
					copy(tma_load_dyt.with(*bar, 0),
						tDYTgDYT(_, _, _, kb_g), tDYTsDYT(_, _, _, state.index()));
					copy(tma_load_z.with(*bar, 0),
						tZgZ(_, _, _, kb_g), tZsZ(_, _, _, state.index()));
				}
				++state;
			}
		}
	}
}

template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaReduceAddDA>
__device__ __forceinline__ void mlp3_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int cell_start,
		int cell_stride,
		int batch_kb_start,
		int batch_kb_end,
		int k_split) {

	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;     // 0 or 1
	const int my_barrier_id = 1 + my_wg;                              // 1 or 2
	const int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;       // 0..255
	const int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;        // 0..127
	auto thr_mma = tiled_mma.get_slice(tid_in_mma);

	auto sDYT = make_tensor(make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ   = make_tensor(make_smem_ptr(smem.Z_data()),   typename Traits::SmemLayoutZ{});

	auto tCsDYT = thr_mma.partition_A(sDYT);
	auto tCsZ   = thr_mma.partition_B(sZ);

	auto acc = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	// Identity coords for per-thread acc → store_buf mapping.
	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	constexpr int store_slot_elems = Traits::WgTileM * Traits::EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	const bool is_my_wg_leader = (tid_in_wg == 0);

	auto mdA = tma_reduce_da.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_da = tma_reduce_da.get_slice(Int<0>{});

	int total_chunks = num_experts * (Traits::kMSplit ? num_n_tiles : num_m_tiles);
	int total_cells  = total_chunks * outer_split * k_split;

	constexpr int K_PIPE_MMAS = 1;

	// Carries across walk iters AND across cells for cross-tile K-loop /
	// store overlap. Drained once at the end.
	bool store_in_flight = false;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {

		int k_slice   = cell_idx % k_split;
		int cell_om   = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane      = cell_om - chunk_idx * outer_split;

		int e, m_chunk, n_chunk, walk_begin, walk_end;
		if constexpr (Traits::kMSplit) {
			// M-split: hold Z(n) shared. chunk = (e, n_tile), walks m_tile.
			e            = chunk_idx / num_n_tiles;
			n_chunk      = chunk_idx - e * num_n_tiles;
			m_chunk      = 0;
			int m_per    = num_m_tiles / outer_split;
			walk_begin   = lane * m_per;
			walk_end     = walk_begin + m_per;
		} else {
			// N-split: hold dY^T(m) shared. chunk = (e, m_tile), walks n_tile.
			e            = chunk_idx / num_m_tiles;
			m_chunk      = chunk_idx - e * num_m_tiles;
			n_chunk      = 0;
			int n_per    = num_n_tiles / outer_split;
			walk_begin   = lane * n_per;
			walk_end     = walk_begin + n_per;
		}

		// Intersect this expert's global K range with the current
		// batch's K-window.
		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e],   batch_kb_end);
		if (kb_hi <= kb_lo) continue;

		// K-split: clip to this cell's k_slice sub-range (must match producer).
		{
			int k_total = kb_hi - kb_lo;
			int k_per   = (k_total + k_split - 1) / k_split;
			int s_lo    = kb_lo + k_slice * k_per;
			int s_hi    = min(s_lo + k_per, kb_hi);
			if (s_hi <= s_lo) continue;   // empty slice
			kb_lo = s_lo;
			kb_hi = s_hi;
		}

		int k_count = kb_hi - kb_lo;

		for (int w = walk_begin; w < walk_end; ++w) {
			int m_tile = Traits::kMSplit ? w       : m_chunk;
			int n_tile = Traits::kMSplit ? n_chunk : w;

			clear(acc);
			auto state_release = state;
			int prologue_count = (k_count < K_PIPE_MMAS) ? k_count : K_PIPE_MMAS;

			for (int k = 0; k < prologue_count; ++k) {
				pipe.consumer_wait(state);
				warpgroup_fence_operand(acc);
				warpgroup_arrive();
				gemm(tiled_mma, tCsDYT(_, _, _, state.index()),
					tCsZ(_, _, _, state.index()), acc);
				warpgroup_commit_batch();
				++state;
			}

			for (int k = prologue_count; k < k_count; ++k) {
				pipe.consumer_wait(state);
				warpgroup_fence_operand(acc);
				warpgroup_arrive();
				gemm(tiled_mma, tCsDYT(_, _, _, state.index()),
					tCsZ(_, _, _, state.index()), acc);
				warpgroup_commit_batch();

				warpgroup_wait<K_PIPE_MMAS>();
				warpgroup_fence_operand(acc);
				pipe.consumer_release(state_release);

				++state;
				++state_release;
			}

			warpgroup_wait<0>();
			warpgroup_fence_operand(acc);
			for (int k = 0; k < prologue_count; ++k) {
				pipe.consumer_release(state_release);
				++state_release;
			}

			// ── Epilogue ──────────────────────────────────────
			// Each WG writes its (WgTileM × EpiChunkN) slot to store_buf
			// and issues TMA stores. NumEpiRounds rounds per WG.
			//   kMSplit=true  : M-split. WG_w owns WgTileM contiguous
			//                   rows of TileM. With WgLayout_M=2 and
			//                   atom_count_M = WgTileM/AtomTileM, the
			//                   natural CUTE m-atom positions interleave
			//                   between WGs — formula below maps m_loc
			//                   back to a contiguous local row.
			//   kMSplit=false : N-split. WG_w owns full TileM, cols
			//                   [w*WgTileN, (w+1)*WgTileN).
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
						// CUTE atom-iter interleaves WGs in M direction
						// when atom_count_M > 1 (TileM=256). Unified
						// formula folds 64-row WG strips into a
						// contiguous WgTileM-row local range:
						//   m_atom_global = m_loc / 64    (atom row idx)
						//   m_atom_in_wg  = m_atom_global / 2
						//   m_local       = m_atom_in_wg*64 + (m_loc % 64)
						//                 = (m_loc / 128)*64 + (m_loc % 64)
						// For tile_count_M=1 (TileM=128) the / 128 term
						// is 0 so m_local = m_loc % 64.
						m_local = (m_loc / 128) * 64 + (m_loc % 64);
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
					int da_m = e * num_m_tiles + m_tile;
					// Store box = ONE atom-row (AtomTileM). Each WG issues
					// kAtomsPerWg stores, one per atom it owns. The store_buf
					// holds the WG's atoms in contiguous local rows
					// [a*AtomTileM, +AtomTileM); their gmem atom-rows differ:
					//   M-split: the 2 atoms interleave → 4*da_m + my_wg + 2a
					//            (gmem M-grid is TileM/AtomTileM rows per tile).
					//   N-split: WG owns the full TileM (2 contiguous atoms) and
					//            its own N-half → 2*da_m + a, n offset by my_wg.
					int n_tile_idx;
					if constexpr (Traits::kMSplit) {
						n_tile_idx = n_tile * Traits::NumEpiRounds + r;
					} else {
						n_tile_idx = n_tile * (Traits::TileN / Traits::EpiChunkN)
						           + my_wg * Traits::NumEpiRounds + r;
					}
					constexpr int kRowAtoms = Traits::TileM / Traits::AtomTileM;  // 4 (M-split) / 2 (N-split)
					CUTE_UNROLL
					for (int a = 0; a < Traits::kAtomsPerWg; ++a) {
						int m_atom_row = Traits::kMSplit
							? (kRowAtoms * da_m + my_wg + Traits::kAtomsPerWg * a)
							: (kRowAtoms * da_m + a);
						auto sStore_a = local_tile(sStore,
							make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
							make_coord(a, 0));
						auto gdA = local_tile(mdA,
							make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
							make_coord(m_atom_row, n_tile_idx));
						copy(tma_reduce_da, cta_tma_da.partition_S(sStore_a),
							cta_tma_da.partition_D(gdA));
					}
					cute::tma_store_arrive();
				}
				store_in_flight = true;
			}
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

// ═══════════════════════════════════════════════════════════════════
// mlp3_fwd — single-launch driver, chunk-fixed
// ═══════════════════════════════════════════════════════════════════
//
// One launch covers all experts. blockIdx.x = cell_start, gridDim.x =
// cell_stride. Each CTA fixes one (chunk, lane) and walks the inner
// axis × kb internally; producer/consumer each get ONE call.

template <typename Traits,
          typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceAddDA>
__device__ __forceinline__ void mlp3_fwd(
		Mlp3Smem<Traits>& smem,
		TmaLoadDYT const& tma_load_dyt,
		TmaLoadZ const& tma_load_z,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim, int intermediate_dim,
		int num_tokens, int total_n_rows,
		int num_m_tiles, int num_n_tiles,
		int outer_split) {

	using Pipeline  = typename Traits::MainloopPipeline;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	cute::prefetch_tma_descriptor(tma_load_dyt.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_da.get_tma_descriptor());

	auto pipe = mlp3_make_pipe<Traits>(smem.pipe_storage);

	__syncthreads();

	PipeState s_prod = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState s_cons;

	int cell_start  = (int)blockIdx.x;
	int cell_stride = (int)gridDim.x;
	// Standalone covers the full token range — pass (0, num_k_blocks)
	// so producer/consumer's intersection with the batch window is a
	// no-op vs the global k_starts/ends scratch.
	int batch_kb_start = 0;
	int batch_kb_end   = num_tokens / Traits::TileK;

	if (is_producer) {
		mlp3_producer<Traits>(
			pipe, s_prod, smem,
			tma_load_dyt, tma_load_z,
			expert_k_starts, expert_k_ends, num_experts,
			hidden_dim, intermediate_dim, num_tokens,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, /*k_split=*/1);
	}
	if (is_consumer) {
		mlp3_consumer<Traits>(
			pipe, s_cons, smem,
			tma_reduce_da,
			expert_k_starts, expert_k_ends, num_experts,
			intermediate_dim, total_n_rows,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, /*k_split=*/1);
	}
}

// ═══════════════════════════════════════════════════════════════════
// Host launcher (raw-pointer; defined in mlp3.cu)
// ═══════════════════════════════════════════════════════════════════

// dA = dY^T @ Z, accumulated atomically into the [E, H, I] output via
// TMA_REDUCE_ADD. dA MUST be zero-initialized by the caller.
//
//   dY                : [num_tokens, hidden_dim]
//   Z                 : [num_tokens, intermediate_dim]
//   dA                : [num_experts, hidden_dim, intermediate_dim]
//   expert_for_k_block: [num_tokens / TileK] int32, TileK=64
void mlp3_fwd_bf16_launch(
		const cutlass::bfloat16_t* dY,
		const cutlass::bfloat16_t* Z,
		cutlass::bfloat16_t* dA,
		const int* expert_for_k_block,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int device,
		cudaStream_t stream);

} // namespace liger
