#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 4:  dB = dU^T @ X,  dC = dV^T @ X   (weight grads, bwd)
// ═══════════════════════════════════════════════════════════════════
//
// TMA_REDUCE_ADD variant. Cooperative 2-WG consumer, mirroring mlp3.cuh.
// The two WGs cooperate on ONE output tile at a time, so dB and dC are
// produced SEQUENTIALLY (two phases per cell). This lets a SINGLE A-side
// smem buffer be reused for dU^T (phase dB) then dV^T (phase dC) — vs the
// old design's two simultaneous A buffers — so the footprint matches mlp3
// and fits the 228 KiB cap at 256-wide tiles.
//
// Only two configs supported (mirrors Mlp3Traits):
//   (TileM=256, TileN=128) → M-split (Layout<_2,_1,_1>)  ← default
//   (TileM=128, TileN=256) → N-split (Layout<_1,_2,_1>)
//
// GEMM orientation: dB = dU^T @ X, so C[M=I, N=H], A[M=I,K=T] = dU^T/dV^T,
// B[N=H,K=T] = X. TileM tiles the intermediate (I) dim, TileN the hidden
// (H) dim. Per-WG output footprint = (WgTileM, WgTileN) = (128, 128).
//
// Single fused A + X TMA pipe (one mbarrier per stage, 2 TMA copies). The
// shared operand X is read by both dB and dC phases and by both WGs.
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for A / X)
//   WG0 warps 1-3  : Idle
//   WG1 (warps 4-7)  : Consumer atom 0
//   WG2 (warps 8-11) : Consumer atom 1
//
// Chunk-fixed walk (same principle as mlp3): hold the SHARED operand
// chunk-constant and walk the cooperation (split) axis:
//   kMSplit=true  (M-split, X shared):       chunk = (e, n_tile), walks m_tile; holds X(n,*)
//   kMSplit=false (N-split, dU^T/dV^T shared): chunk = (e, m_tile), walks n_tile; holds A(m,*)
// In the M-split default, X(n_tile) is held across BOTH the m-walk and the
// dB/dC phases — re-read for phase dC hits L2.
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
	int TileM_     = 256,
	int TileN_     = 128,
	int TileK_     = 64,
	int Stages_    = 4,
	int EpiChunkN_ = 64
>
struct Mlp4Traits {
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
		"Mlp4Traits supports only (TileM, TileN) = (256, 128) M-split "
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

	// A-side = dU^T / dV^T (M-side, TileM); X = shared B-side (N-side, TileN).
	using SmemLayoutA_1 = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>>{},
		Step<_2, _1>{}));
	using SmemLayoutX_1 = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>>{},
		Step<_2, _1>{}));
	using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileM>, Int<TileK>, Int<Stages>>{},
		Step<_2, _1, _3>{}));
	using SmemLayoutX = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<TileN>, Int<TileK>, Int<Stages>>{},
		Step<_2, _1, _3>{}));

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	static constexpr int TmaTransBytesA =
		static_cast<int>(size(SmemLayoutA_1{}) * sizeof(Element));
	static constexpr int TmaTransBytesX =
		static_cast<int>(size(SmemLayoutX_1{}) * sizeof(Element));
	// Fused pipeline: X + ONE A operand (dU^T or dV^T, per phase) share one
	// mbarrier per stage. Both WGs consume the same X via cooperative TiledMma.
	static constexpr int TmaTransBytes = TmaTransBytesA + TmaTransBytesX;

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
	// At TileM=256 each WG owns 2 *interleaved* atoms that map to non-contiguous
	// gmem rows, so a WG issues kAtomsPerWg separate TMA stores per epi round
	// (mirrors mlp1_fused.cuh / mlp3.cuh).
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

// ═══════════════════════════════════════════════════════════════════
// Shared Memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp4Smem {
	using Element = typename Traits::Element;

	static constexpr int smem_X_size     = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int smem_A_size     = cosize_v<typename Traits::SmemLayoutA>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	// Single A buffer, reused for dU^T (phase dB) then dV^T (phase dC).
	alignas(128) Element smem_X[smem_X_size];
	alignas(128) Element smem_A[smem_A_size];
	// Per-WG store buffer: WG1 uses [0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	typename Traits::MainloopPipeline::SharedStorage pipe_storage;

	CUTE_DEVICE Element* X_data() { return &smem_X[0]; }
	CUTE_DEVICE Element* A_data() { return &smem_A[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// Fused MLP4 shared memory (data only — pipe storage lives in the
// outer MlpFusedBwdSmem union; otherwise identical to Mlp4Smem).
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp4FusedSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_X_size     = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int smem_A_size     = cosize_v<typename Traits::SmemLayoutA>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_X[smem_X_size];
	alignas(128) Element smem_A[smem_A_size];
	alignas(128) Element store_buf[2 * smem_store_size];

	CUTE_DEVICE Element* X_data() { return &smem_X[0]; }
	CUTE_DEVICE Element* A_data() { return &smem_A[0]; }
};


// ═══════════════════════════════════════════════════════════════════
// Pipe construction helper (mirrors mlp3_make_pipe)
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipeline
mlp4_make_pipe(typename Traits::MainloopPipeline::SharedStorage& storage) {
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
// Chunk-fixed cooperative producer/consumer — sequential dB / dC
// ═══════════════════════════════════════════════════════════════════
//
// Used by the standalone mlp4 kernel (mlp4.cu) and mlp_bwd Phase 2.
// Each CTA owns ONE (chunk, lane) cell and processes it in TWO phases
// (phase 0 = dB via dU^T, phase 1 = dC via dV^T). We hold the SHARED
// operand and walk the cooperation (split) axis:
//   kMSplit=true  (M-split): chunk = (e, n_tile), lane partitions
//                   num_m_tiles, walks m_tile. X(n_tile, *) — the shared
//                   N-side operand — is the L2-held chunk-constant,
//                   reused across the m-walk AND both phases.
//   kMSplit=false (N-split): chunk = (e, m_tile), lane partitions
//                   num_n_tiles, walks n_tile. A(m_tile, *) (dU^T then
//                   dV^T) is L2-held; X is re-streamed per phase.
//
// Cell decode (chunk-major: cell_idx = chunk_idx × outer_split + lane):
//   total_cells = total_chunks × outer_split
//   cell        →  chunk_idx = cell_idx / outer_split
//                  lane      = cell_idx % outer_split

template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaLoadX, typename TmaLoadA>
__device__ __forceinline__ void mlp4_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadA const& tma_load_dut,
		TmaLoadA const& tma_load_dvt,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
		int hidden_dim,
		int num_tokens,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int cell_start,
		int cell_stride,
		int batch_kb_start,
		int batch_kb_end,
		int k_split,
		// Staging-ring depth in K-blocks; > 0 wraps the gmem K-tile coordinate
		// (% ring_kb) for the grouped Phase-2 that may straddle the ring. See
		// mlp3_producer. 0 = no wrap.
		int ring_kb = 0) {

	auto sA = make_tensor(make_smem_ptr(smem.A_data()), typename Traits::SmemLayoutA{});
	auto sX = make_tensor(make_smem_ptr(smem.X_data()), typename Traits::SmemLayoutX{});

	// int64_t casts: H·T, I·T can exceed 2^31 elems at production shapes.
	auto mX   = tma_load_x.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(num_tokens)));
	auto mdUT = tma_load_dut.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));
	auto mdVT = tma_load_dvt.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));

	auto cta_tma_x   = tma_load_x.get_slice(Int<0>{});
	auto cta_tma_dut = tma_load_dut.get_slice(Int<0>{});
	auto cta_tma_dvt = tma_load_dvt.get_slice(Int<0>{});

	auto tXsX = cta_tma_x.partition_D(sX);
	auto tAsA = cta_tma_dut.partition_D(sA);   // dU^T/dV^T share A layout

	int total_chunks = num_experts * (Traits::kMSplit ? num_n_tiles : num_m_tiles);
	// K-split: each (chunk, walk-lane) replicated k_split times; k_slice owns a
	// 1/k_split sub-range of the K-loop. Partials sum via TMA_REDUCE_ADD.
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
			// M-split: hold X(n) shared. chunk = (e, n_tile), walks m_tile.
			e            = chunk_idx / num_n_tiles;
			n_chunk      = chunk_idx - e * num_n_tiles;
			m_chunk      = 0;
			int m_per    = num_m_tiles / outer_split;
			walk_begin   = lane * m_per;
			walk_end     = walk_begin + m_per;
		} else {
			// N-split: hold A(m) shared. chunk = (e, m_tile), walks n_tile.
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

		// K-split: clip to this cell's k_slice sub-range of [kb_lo, kb_hi).
		{
			int k_total = kb_hi - kb_lo;
			int k_per   = (k_total + k_split - 1) / k_split;
			int s_lo    = kb_lo + k_slice * k_per;
			int s_hi    = min(s_lo + k_per, kb_hi);
			if (s_hi <= s_lo) continue;   // empty slice
			kb_lo = s_lo;
			kb_hi = s_hi;
		}

		// Two phases per cell: phase 0 = dB (dU^T), phase 1 = dC (dV^T).
		// dU^T/dV^T share descriptor layout; pair the tma object, its slice,
		// and its gmem tensor consistently per phase.
		CUTE_UNROLL
		for (int phase = 0; phase < 2; ++phase) {
			auto mA               = (phase == 0) ? mdUT        : mdVT;
			const auto& tma_A     = (phase == 0) ? tma_load_dut : tma_load_dvt;
			const auto& cta_tma_A = (phase == 0) ? cta_tma_dut  : cta_tma_dvt;

			for (int w = walk_begin; w < walk_end; ++w) {
				int m_tile = Traits::kMSplit ? w       : m_chunk;
				int n_tile = Traits::kMSplit ? n_chunk : w;
				auto gX = local_tile(mX,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n_tile, _));
				auto gA = local_tile(mA,
					make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
					make_coord(m_tile, _));
				auto tXgX = cta_tma_x.partition_S(gX);
				auto tAgA = cta_tma_A.partition_S(gA);

				for (int kb = kb_lo; kb < kb_hi; ++kb) {
					int kb_g = (ring_kb > 0) ? (kb % ring_kb) : kb;  // physical ring slot
					pipe.producer_acquire(state);
					if (threadIdx.x == 0) {
						auto* bar = pipe.producer_get_barrier(state);
						copy(tma_load_x.with(*bar, 0),
							tXgX(_, _, _, kb_g), tXsX(_, _, _, state.index()));
						copy(tma_A.with(*bar, 0),
							tAgA(_, _, _, kb_g), tAsA(_, _, _, state.index()));
					}
					++state;
				}
			}
		}
	}
}

template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaReduceAddDB, typename TmaReduceAddDC>
__device__ __forceinline__ void mlp4_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDB const& tma_reduce_db,
		TmaReduceAddDC const& tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int total_m_rows,
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

	auto sA = make_tensor(make_smem_ptr(smem.A_data()), typename Traits::SmemLayoutA{});
	auto sX = make_tensor(make_smem_ptr(smem.X_data()), typename Traits::SmemLayoutX{});

	auto tCsA = thr_mma.partition_A(sA);
	auto tCsX = thr_mma.partition_B(sX);

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

	auto mdB = tma_reduce_db.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_m_rows),
		static_cast<int64_t>(hidden_dim)));
	auto mdC = tma_reduce_dc.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_m_rows),
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma_db = tma_reduce_db.get_slice(Int<0>{});
	auto cta_tma_dc = tma_reduce_dc.get_slice(Int<0>{});

	int total_chunks = num_experts * (Traits::kMSplit ? num_n_tiles : num_m_tiles);
	int total_cells  = total_chunks * outer_split * k_split;

	constexpr int K_PIPE_MMAS = 1;

	// Carries across phases, walk iters AND cells for cross-tile K-loop /
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
			// M-split: hold X(n) shared. chunk = (e, n_tile), walks m_tile.
			e            = chunk_idx / num_n_tiles;
			n_chunk      = chunk_idx - e * num_n_tiles;
			m_chunk      = 0;
			int m_per    = num_m_tiles / outer_split;
			walk_begin   = lane * m_per;
			walk_end     = walk_begin + m_per;
		} else {
			// N-split: hold A(m) shared. chunk = (e, m_tile), walks n_tile.
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

		// Two phases per cell: phase 0 = dB, phase 1 = dC.
		CUTE_UNROLL
		for (int phase = 0; phase < 2; ++phase) {

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
					gemm(tiled_mma, tCsA(_, _, _, state.index()),
						tCsX(_, _, _, state.index()), acc);
					warpgroup_commit_batch();
					++state;
				}

				for (int k = prologue_count; k < k_count; ++k) {
					pipe.consumer_wait(state);
					warpgroup_fence_operand(acc);
					warpgroup_arrive();
					gemm(tiled_mma, tCsA(_, _, _, state.index()),
						tCsX(_, _, _, state.index()), acc);
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
				// and TMA_REDUCE_ADDs it into dB (phase 0) or dC (phase 1).
				// Store indexing mirrors mlp3_consumer (M-split / N-split).
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
						int out_m = e * num_m_tiles + m_tile;
						// Store box = ONE atom-row (AtomTileM); kAtomsPerWg stores
						// per WG. M-split: the 2 atoms interleave in gmem
						// (4*out_m + my_wg + 2a). N-split: WG owns the full TileM
						// (2 contiguous atoms) and its own N-half (2*out_m + a).
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
								? (kRowAtoms * out_m + my_wg + Traits::kAtomsPerWg * a)
								: (kRowAtoms * out_m + a);
							auto sStore_a = local_tile(sStore,
								make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
								make_coord(a, 0));
							if (phase == 0) {
								auto gdB = local_tile(mdB,
									make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
									make_coord(m_atom_row, n_tile_idx));
								copy(tma_reduce_db, cta_tma_db.partition_S(sStore_a),
									cta_tma_db.partition_D(gdB));
							} else {
								auto gdC = local_tile(mdC,
									make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
									make_coord(m_atom_row, n_tile_idx));
								copy(tma_reduce_dc, cta_tma_dc.partition_S(sStore_a),
									cta_tma_dc.partition_D(gdC));
							}
						}
						cute::tma_store_arrive();
					}
					store_in_flight = true;
				}
			}
		}
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

// ═══════════════════════════════════════════════════════════════════
// mlp4_fwd — single-launch driver, chunk-fixed
// ═══════════════════════════════════════════════════════════════════
//
// One launch covers all experts. blockIdx.x = cell_start, gridDim.x =
// cell_stride. Each CTA fixes one (chunk, lane), runs both dB/dC phases
// internally; producer/consumer each get ONE call.

template <typename Traits,
          typename TmaLoadX, typename TmaLoadA, typename TmaReduceAddOut>
__device__ __forceinline__ void mlp4_fwd(
		Mlp4Smem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadA const& tma_load_dut,
		TmaLoadA const& tma_load_dvt,
		TmaReduceAddOut const& tma_reduce_db,
		TmaReduceAddOut const& tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim, int hidden_dim,
		int num_tokens, int total_m_rows,
		int num_m_tiles, int num_n_tiles,
		int outer_split) {

	using Pipeline  = typename Traits::MainloopPipeline;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	cute::prefetch_tma_descriptor(tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dut.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_dvt.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_db.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_reduce_dc.get_tma_descriptor());

	auto pipe = mlp4_make_pipe<Traits>(smem.pipe_storage);

	__syncthreads();

	PipeState s_prod = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState s_cons;

	int cell_start  = (int)blockIdx.x;
	int cell_stride = (int)gridDim.x;
	// Standalone covers the full token range — pass (0, num_k_blocks).
	int batch_kb_start = 0;
	int batch_kb_end   = num_tokens / Traits::TileK;

	if (is_producer) {
		mlp4_producer<Traits>(
			pipe, s_prod, smem,
			tma_load_x, tma_load_dut, tma_load_dvt,
			expert_k_starts, expert_k_ends, num_experts,
			intermediate_dim, hidden_dim, num_tokens,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, /*k_split=*/1);
	}
	if (is_consumer) {
		mlp4_consumer<Traits>(
			pipe, s_cons, smem,
			tma_reduce_db, tma_reduce_dc,
			expert_k_starts, expert_k_ends, num_experts,
			hidden_dim, total_m_rows,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			batch_kb_start, batch_kb_end, /*k_split=*/1);
	}
}

// ═══════════════════════════════════════════════════════════════════
// Host launcher (raw-pointer; defined in mlp4.cu)
// ═══════════════════════════════════════════════════════════════════

// dB = dU^T @ X, dC = dV^T @ X. Both outputs accumulated atomically
// via TMA_REDUCE_ADD. dB and dC MUST be zero-initialized by caller.
//
//   dU                : [num_tokens, intermediate_dim]
//   dV                : [num_tokens, intermediate_dim]
//   X                 : [num_tokens, hidden_dim]
//   dB, dC            : [num_experts, intermediate_dim, hidden_dim]
//   expert_for_k_block: [num_tokens / TileK] int32, TileK=64
void mlp4_fwd_bf16_launch(
		const cutlass::bfloat16_t* dU,
		const cutlass::bfloat16_t* dV,
		const cutlass::bfloat16_t* X,
		cutlass::bfloat16_t* dB,
		cutlass::bfloat16_t* dC,
		const int* expert_for_k_block,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
		int device,
		cudaStream_t stream);

} // namespace liger
