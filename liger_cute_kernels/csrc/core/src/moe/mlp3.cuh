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
//   WG0 warps 1-2  : Idle / reserved for fused-kernel communication
//   WG0 warp 3     : Idle on SM90; dedicated UMMA producer on SM100
//   WG1 (warps 4-7)  : WGMMA consumer on SM90; epilogue WG0 on SM100
//   WG2 (warps 8-11) : WGMMA consumer on SM90; epilogue WG1 on SM100
//
// Chunk-fixed walk: each CTA owns ONE (chunk, lane) cell. We hold the
// SHARED (cooperative) operand chunk-constant and walk the cooperation
// (split) axis, maximizing L2 reuse of the operand both WGs consume:
//   kMSplit=true  (M-split, Z shared):    chunk = (e, n_tile), walks m_tile; holds Z(n,*)
//   kMSplit=false (N-split, dY^T shared): chunk = (e, m_tile), walks n_tile; holds dY^T(m,*)
// ═══════════════════════════════════════════════════════════════════

#include "models.cuh"
#include "tmem_load_op.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>

// SM100 (Blackwell / UMMA) support for the Compute=100 consumer specialization
// (Mlp3ConsumerImpl<100>). Header-only and safe to include under an sm_90a
// build — the tcgen05 PTX they emit is guarded by CUTE_ARCH_TCGEN05_* and only
// the Compute=100 body (itself gated on __CUDA_ARCH__ >= 1000) instantiates it.
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>  // PipelineTmaUmmaAsync, PipelineUmmaAsync

namespace liger {

using namespace cute;

template <typename Element, typename ElementAccum, int TileM, int TileN, bool Valid = (TileM == 64 || TileM == 128)>
struct Mlp3UmmaTiledMmaSelector {
	using type = void;
};

template <typename Element, typename ElementAccum, int TileM, int TileN>
struct Mlp3UmmaTiledMmaSelector<Element, ElementAccum, TileM, TileN, true> {
	using type = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, ElementAccum,
		                     TileM, TileN,
		                     UMMA::Major::MN, UMMA::Major::MN>{}));
};

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

	// ── SM100 (Compute=100) pipelines — single CTA, no cluster. ──
	// MainloopPipelineUmma: UMMA-aware TMA pipeline whose consumer_release
	// issues the UMMA-gated smem-buffer arrival. The Blackwell launcher must
	// build it with num_consumers=1 (one umma_arrive per stage releases the
	// buffer). Params/PipelineState/SharedStorage alias the Hopper
	// PipelineTmaAsync's, so producer + Compute=100 consumer drive it unchanged.
	using MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<
		Stages, cute::Shape<cute::_1, cute::_1, cute::_1>,
		cute::Shape<cute::_1, cute::_1, cute::_1>>;
	// Double-buffered TMEM accumulators: MMA for the next (cell, walk) can fill
	// the alternate stage while warps 4..11 drain/reduce-add the previous stage.
	static constexpr int AccStages = 2;
	static_assert(AccStages * TileN <= 512,
		"SM100 MLP3 accumulator stages must fit in 512 TMEM columns");
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccStages>;

	// SM100 UMMA TiledMMA — one 1SM tcgen05 atom spanning (TileM, TileN) with
	// the accumulator in TMEM. mlp3 is the FIRST ported kernel with BOTH
	// operands MN-major: A = dY^T (M=H contiguous) and B = Z (N=I contiguous),
	// exactly matching SmemLayoutDYT / SmemLayoutZ (both Layout_MN_SW128_Atom +
	// Step<_2,_1,_3>). make_umma_desc<Major::MN> reads the MN-contiguous stride
	// from each smem layout, so no transpose is needed (cf. mlp2_t's <K,MN>).
	using TiledMmaUmma = typename Mlp3UmmaTiledMmaSelector<
		Element, ElementAccum, TileM, TileN>::type;

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

template <typename Traits, int Compute>
using Mlp3MainloopPipelineFor = cute::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

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

	// SM100 (Compute=100) only: landing slot for tcgen05.alloc's granted TMEM
	// base address, plus the accumulator pipeline's barrier storage (UMMA→
	// epilogue handoff). The Hopper (Compute=90) path never touches these; they
	// keep the consumer signature identical across Compute values.
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

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
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data()   { return &smem_Z[0]; }
};


// ═══════════════════════════════════════════════════════════════════
// Pipe construction helper
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors mlp2/mlp5's make_pipe pattern: hides the role/transaction-
// byte setup so callers (mlp3_fwd, or any fused parent kernel) get a
// consistent Pipeline. The plain SM90 helper uses warp 0 as TMA producer and
// warps 4-11 as WGMMA consumers; the SM100 helper additionally enrolls warp 3
// as the UMMA consumer of the TMA pipeline.

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

// SM100 (Compute=100) mainloop pipe. Mirrors mlp3_make_pipe but builds the
// UMMA-aware TMA pipeline (PipelineTmaUmmaAsync) whose consumer_release issues
// the UMMA-gated smem-buffer arrival. num_consumers=1: on Blackwell the buffer
// is released by a single umma_arrive per stage (the MMA warp), not per
// consumer thread. The ctor arg pattern is identical to the Hopper pipe, so the
// arch-agnostic mlp3_producer drives it unchanged.
template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipelineUmma
mlp3_make_pipe_umma(typename Traits::MainloopPipelineUmma::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipelineUmma;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 3 && warp_id <= 11);  // warp 3 = UMMA producer

	typename Pipeline::Params pp;
	pp.transaction_bytes = Traits::TmaTransBytes;
	pp.num_producers = 1;
	pp.num_consumers = 1;   // UMMA: one umma_arrive per stage releases the buffer
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

// ═══════════════════════════════════════════════════════════════════
// Consumer — arch-split into Compute=90 (WGMMA) / Compute=100 (UMMA)
// ═══════════════════════════════════════════════════════════════════
//   Compute=90  → Hopper / WGMMA. Both WGs cooperate in ONE TiledMMA that
//                 partitions (TileM,TileN); each thread holds a register acc
//                 slice, scatters into store_buf, and the WG leader issues
//                 SM90_TMA_REDUCE_ADD.
//   Compute=100 → Blackwell / UMMA. One 1SM tcgen05 atom (warp 3) fills ONE
//                 TMEM accumulator over the whole (TileM,TileN); both WGs pull
//                 their N-half from TMEM into the SAME store_buf slots, then the
//                 IDENTICAL REDUCE_ADD leader block stores them.

template <int Compute>
struct Mlp3ConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper WGMMA. Verbatim from the original mlp3_consumer.
// ───────────────────────────────────────────────────────────────────
template <>
struct Mlp3ConsumerImpl<90> {
template <typename Traits,
          int Compute = 90,
          typename Pipeline, typename SmemType,
          typename TmaReduceAddDA>
static __device__ __forceinline__ void run(
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

		(void)Compute;
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
};  // Mlp3ConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Compute=100 — Blackwell / UMMA.
//
// One 1SM tcgen05 atom with M=TileM, N=TileN covers the whole tile. Warp 3
// issues the continuous k-loop into double-buffered TMEM accumulators while
// warps 4-11 drain the previous stage; ScaleOut::Zero on the first MMA of each
// (cell, walk) clears its selected stage, One thereafter.
//
// Epilogue (the mlp3-specific bit): the store path is SM90_TMA_REDUCE_ADD and
// is ARCH-AGNOSTIC (guarded only by __CUDA_ARCH__ in CUTLASS 4.4.1), so the
// entire `is_my_wg_leader` reduce-add block below — the tma_store_fence →
// copy(tma_reduce_da,…) → tma_store_arrive loop, incl. the m_atom_row /
// n_tile_idx index math — is kept BYTE-FOR-BYTE identical to Impl<90>. Only the
// step that FILLS store_buf changes: instead of scattering a register acc, both
// WGs pull their N-half out of TMEM in EpiChunkN chunks and write the SAME
// sStore(m_local, chunk_n) positions.
//
// CORRECTNESS PIN: (m_local, chunk_n) are re-derived from the TMEM-load
// partition_D identity coords (cChunk = identity((WgTileM, EpiChunkN))), NOT
// from the SM90 tCcC — the UMMA thread-value layout differs from WGMMA, so the
// SM90 M-split remap does not carry over. For the supported N-split config
// (TileM=WgTileM=128) each WG owns the full 128 rows and its own WgTileN cols,
// so the map is the plain identity (m_local = m_row, chunk_n = n_col).
// ───────────────────────────────────────────────────────────────────
template <>
struct Mlp3ConsumerImpl<100> {
template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaReduceAddDA>
static __device__ __forceinline__ void run(
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	constexpr int TileM        = Traits::TileM;
	constexpr int TileN        = Traits::TileN;
	constexpr int EpiChunkN    = Traits::EpiChunkN;
	constexpr int WgTileM      = Traits::WgTileM;       // 128
	constexpr int NumEpiRounds = Traits::NumEpiRounds;  // WgTileN / EpiChunkN = 2

	// One 1SM UMMA atom's M is ≤ 128, so only the N-split config (TileM=128)
	// maps to a single accumulator. The M-split config (TileM=256) exceeds a
	// 1SM atom's M and stays on the WGMMA (Compute=90) path.
	static_assert(TileM == WgTileM,
		"Mlp3 Compute=100 (UMMA) supports the N-split config (TileM=128) only; "
		"the M-split config (TileM=256) exceeds a 1SM UMMA atom's M extent.");
	static_assert(!Traits::kMSplit, "Mlp3 Compute=100 is the N-split path");
	static_assert(TileN % 2 == 0, "two consumer warpgroups split TileN");

	// ── Thread identity ─────────────────────────────────────────────
	// Warp 3 is the dedicated, epilogue-free UMMA producer. Warps 4..11 are
	// the two aligned epilogue WGs.
	const int  warp_id         = threadIdx.x / Traits::WarpSize;
	const bool is_mma_warp     = (warp_id == 3);
	const bool is_epilogue     = (warp_id >= 4 && warp_id <= 11);
	const int  tid_in_epi      = threadIdx.x - Traits::WarpGroupSize;        // warps 4..11 -> 0..255
	const int  my_wg           = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	const int  tid_in_wg       = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	const int  my_barrier_id   = 1 + my_wg;                                  // 1 or 2
	const bool is_my_wg_leader = is_epilogue && (tid_in_wg == 0);
	constexpr int kEpilogueThreads = Traits::ConsumerThreads;
	constexpr int kMmaEpiThreads = kEpilogueThreads + Traits::WarpSize;
	static_assert(Traits::WarpGroupSize == 4 * Traits::WarpSize);
	static_assert(kEpilogueThreads == 8 * Traits::WarpSize);
	static_assert(kMmaEpiThreads == 9 * Traits::WarpSize);

	// ── UMMA TiledMMA: one 1SM tcgen05 atom over (TileM, TileN); BOTH operands
	//    MN-major (dY^T: M=H contiguous; Z: N=I contiguous). ──
	typename Traits::TiledMmaUmma tiled_mma;
	auto cta_mma = tiled_mma.get_slice(0);   // 1SM → single CTA, peer-coord 0

	auto sDYT = make_tensor(make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ   = make_tensor(make_smem_ptr(smem.Z_data()),   typename Traits::SmemLayoutZ{});

	// ── One TMEM accumulator covering the whole (TileM, TileN) tile ──
	auto cAccFull = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC     = cta_mma.partition_C(cAccFull);
	auto tCtAcc   = cta_mma.make_fragment_C(tCgC);

	// TMEM is allocated by the outer fused/standalone launcher once per CTA.

	// ── Accumulator pipeline: UMMA producer (warp 3) → epilogue consumers
	//    (warps 4..11, both WGs). AccStages stages allow the next MMA to run
	//    while the previous accumulator stage is drained and TMA-reduced. ──
	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp ? AccPipe::ThreadCategory::Producer
	                              : AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = 1;
	acc_params.initializing_warp  = 4;
	AccPipe acc_pipe(smem.acc_pipe, acc_params,
		cute::Shape<cute::_1, cute::_1, cute::_1>{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	// The launcher hoisted tcgen05.alloc to once-per-CTA (persistent grid; a
	// per-cell alloc traps with phase_invalid_during_alloc). Publish its base.
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
	tCtAcc.data() = smem.tmem_base;

	// ── Per-WG store slot (WgTileM × EpiChunkN) — SAME smem as Hopper ──
	constexpr int store_slot_elems = WgTileM * EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	auto mdA = tma_reduce_da.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_da = tma_reduce_da.get_slice(Int<0>{});

	// ── Epilogue TMEM→reg plumbing (built once). flat_divide (not
	//    zipped_divide) keeps flat (M,N) modes; fragment sized from partition_D
	//    (the DEST coords), NOT partition_S, so it excludes the datapath-lane
	//    dim. Identity coords come from cChunk = identity((WgTileM, EpiChunkN)). ──
	auto epi_tile   = make_tile(Int<WgTileM>{}, Int<EpiChunkN>{});
	auto acc_mn     = tCtAcc(make_coord(_, _), _0{}, _0{});   // (TileM, TileN)
	auto tAcc_epi   = flat_divide(acc_mn, epi_tile);          // (WgTileM,EpiChunkN,1,TileN/EpiChunkN)
	auto t2r        = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAcc_epi(_, _, _0{}, _0{}));
	auto thr_t2r    = t2r.get_slice(tid_in_wg);
	auto tTR_tAcc   = thr_t2r.partition_S(tAcc_epi);          // (Cpy,Cpy_M,Cpy_N,1,nChunks)
	auto cChunk     = make_identity_tensor(make_shape(Int<WgTileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);            // (Cpy,Cpy_M,Cpy_N)
	auto tTR_rAcc   = make_tensor<float>(shape(tTR_cChunk));  // f32 regs

	int total_chunks = num_experts * (Traits::kMSplit ? num_n_tiles : num_m_tiles);
	int total_cells  = total_chunks * outer_split * k_split;

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
			e            = chunk_idx / num_n_tiles;
			n_chunk      = chunk_idx - e * num_n_tiles;
			m_chunk      = 0;
			int m_per    = num_m_tiles / outer_split;
			walk_begin   = lane * m_per;
			walk_end     = walk_begin + m_per;
		} else {
			e            = chunk_idx / num_m_tiles;
			m_chunk      = chunk_idx - e * num_m_tiles;
			n_chunk      = 0;
			int n_per    = num_n_tiles / outer_split;
			walk_begin   = lane * n_per;
			walk_end     = walk_begin + n_per;
		}

		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e],   batch_kb_end);
		if (kb_hi <= kb_lo) continue;

		{
			int k_total = kb_hi - kb_lo;
			int k_per   = (k_total + k_split - 1) / k_split;
			int s_lo    = kb_lo + k_slice * k_per;
			int s_hi    = min(s_lo + k_per, kb_hi);
			if (s_hi <= s_lo) continue;
			kb_lo = s_lo;
			kb_hi = s_hi;
		}

		for (int w = walk_begin; w < walk_end; ++w) {
			int m_tile = Traits::kMSplit ? w       : m_chunk;
			int n_tile = Traits::kMSplit ? n_chunk : w;

			// ── Mainloop: warp 3 issues one continuous UMMA k-loop into the
			//    single TMEM accumulator. ScaleOut::Zero on the first MMA of
			//    this (cell, walk) (clears acc), One thereafter. ──
			if (is_mma_warp) {
				acc_pipe.producer_acquire(acc_prod_state);   // TMEM acc free
				int acc_stage = acc_prod_state.index();
				tCtAcc.data() = smem.tmem_base + uint32_t(acc_stage * TileN);
				bool first = true;
				for (int kb = kb_lo; kb < kb_hi; ++kb) {
					pipe.consumer_wait(state);
					auto tCsDYT = cta_mma.partition_A(sDYT(_, _, state.index()));
					auto tCsZ   = cta_mma.partition_B(sZ(_, _, state.index()));
					auto tCrDYT = cta_mma.make_fragment_A(tCsDYT);
					auto tCrZ   = cta_mma.make_fragment_B(tCsZ);
					CUTE_UNROLL
					for (int ks = 0; ks < size<2>(tCrDYT); ++ks) {
						tiled_mma.accumulate_ = first ? UMMA::ScaleOut::Zero
						                              : UMMA::ScaleOut::One;
						first = false;
						gemm(tiled_mma, tCrDYT(_, _, ks), tCrZ(_, _, ks), tCtAcc);
					}
					pipe.consumer_release(state);   // UMMA-gated smem release
					++state;
				}
				acc_pipe.producer_commit(acc_prod_state);    // umma_arrive
				++acc_prod_state;
			}

			// ── Epilogue: wait for the accumulator, then each WG pulls its
			//    N-half from TMEM into store_buf and the leader reduce-adds. ──
			if (is_epilogue) {
				acc_pipe.consumer_wait(acc_cons_state);
				int acc_stage = acc_cons_state.index();
				tCtAcc.data() = smem.tmem_base + uint32_t(acc_stage * TileN);
				auto acc_mn_stage   = tCtAcc(make_coord(_, _), _0{}, _0{});
				auto tAcc_epi_stage = flat_divide(acc_mn_stage, epi_tile);
				auto tTR_tAcc_stage = thr_t2r.partition_S(tAcc_epi_stage);

				CUTE_UNROLL
				for (int r = 0; r < NumEpiRounds; ++r) {
					// TMEM → registers for this WG's r-th n-chunk (all WgTileM rows).
					int chunk = my_wg * NumEpiRounds + r;   // absolute n-chunk
					copy(t2r, tTR_tAcc_stage(_, _, _, _0{}, chunk), tTR_rAcc);

					if (store_in_flight)
						cute::tma_store_wait<0>();

					cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, my_barrier_id);

					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rAcc); ++i) {
						int m_local = get<0>(tTR_cChunk(i));   // 0..WgTileM
						int chunk_n = get<1>(tTR_cChunk(i));   // 0..EpiChunkN
						sStore(m_local, chunk_n) = static_cast<Element>(tTR_rAcc(i));
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

				// TMEM reads done → release the acc so the next (cell, walk) MMA
				// may reuse it (one elected consumer thread arrives).
				cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
				if (tid_in_epi == 0)
					acc_pipe.consumer_release(acc_cons_state);
				++acc_cons_state;
			}
		}
	}
	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
#else
	// Never dispatched on non-SM100 targets (host picks Compute=90 there).
	__trap();
#endif
}
};  // Mlp3ConsumerImpl<100>

// ───────────────────────────────────────────────────────────────────
// Forwarder — Compute defaults to 90 so existing call sites
// (mlp3_consumer<Traits>(...)) are unchanged; pass mlp3_consumer<Traits, 100>
// for the Blackwell path.
// ───────────────────────────────────────────────────────────────────
template <typename Traits, int Compute = 90,
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
	Mlp3ConsumerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_reduce_da,
		expert_k_starts, expert_k_ends, num_experts,
		intermediate_dim, total_n_rows,
		num_m_tiles, num_n_tiles, outer_split,
		cell_start, cell_stride,
		batch_kb_start, batch_kb_end, k_split);
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
