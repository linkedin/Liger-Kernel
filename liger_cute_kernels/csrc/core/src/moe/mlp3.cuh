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

// SM100 paired-CTA MLP3 support. Header-only, with the same guarding
// rationale as the other SM100 includes above.
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>

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

template <int Compute>
struct Mlp3ProducerImpl;

template <>
struct Mlp3ProducerImpl<90> {
template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaLoadDYT, typename TmaLoadZ>
static __device__ __forceinline__ void run(
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
};  // Mlp3ProducerImpl<90>

template <typename Traits, int Compute = 90,
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
		int ring_kb = 0) {
	Mlp3ProducerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_load_dyt, tma_load_z,
		expert_k_starts, expert_k_ends, num_experts,
		hidden_dim, intermediate_dim, num_tokens,
		num_m_tiles, num_n_tiles, outer_split,
		cell_start, cell_stride,
		batch_kb_start, batch_kb_end, k_split, ring_kb);
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
          bool Expert3D = false,
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
		int hidden_dim,
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

	auto mdA = [&]() {
		if constexpr (Expert3D) {
			return tma_reduce_da.get_tma_tensor(make_shape(
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_reduce_da.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_n_rows),
				static_cast<int64_t>(intermediate_dim)));
		}
	}();
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
					int da_m = Expert3D ? m_tile : e * num_m_tiles + m_tile;
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
						auto gdA = [&]() {
							if constexpr (Expert3D) {
								return local_tile(mdA,
									make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
									make_coord(m_atom_row, n_tile_idx, e));
							} else {
								return local_tile(mdA,
									make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
									make_coord(m_atom_row, n_tile_idx));
							}
						}();
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
// Forwarder — Compute defaults to 90 so existing call sites
// (mlp3_consumer<Traits>(...)) are unchanged; pass mlp3_consumer<Traits, 100>
// for the Blackwell path.
// ───────────────────────────────────────────────────────────────────
template <typename Traits, int Compute = 90, bool Expert3D = false,
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
		int hidden_dim,
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
	Mlp3ConsumerImpl<Compute>::template run<Traits, Expert3D>(
		pipe, state, smem, tma_reduce_da,
		expert_k_starts, expert_k_ends, num_experts,
		hidden_dim, intermediate_dim, total_n_rows,
		num_m_tiles, num_n_tiles, outer_split,
		cell_start, cell_stride,
		batch_kb_start, batch_kb_end, k_split);
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

// ═══════════════════════════════════════════════════════════════════
// SM100 2-SM (paired-CTA) MLP3 — basic ClusterM=2 path
// ═══════════════════════════════════════════════════════════════════
//
// Joins two adjacent CTAs into one cta_group::2 UMMA atom covering a
// 256x256x64 joined tile, doubling the M extent a single CTA-pair can
// produce per wave versus the 1SM path above. Callers must launch with
// cudaLaunchKernelEx, clusterDim=(2,1,1), an even gridDim.x, and
// pair-aware SM100 2SM TMA load descriptors for dY^T / Z. The dA
// descriptor remains an ordinary (non-paired) TMA_REDUCE_ADD.
//
// TileK, Stages, and EpiChunkN stay trait parameters — the validated
// production candidate is TileK=64/Stages=5/EpiChunkN=64 (see
// 2sm.md / 2sm_stage.md), but that tuple is a caller-side choice, not
// hardcoded here. ClusterM is fixed at 2 (one paired-CTA "pair"): the
// ClusterM=4 (two-pair, N-split-across-pairs) and shared-operand TMA
// multicast variants explored alongside this trait did not clear the
// required performance bar (see the mlp3_multicast bench results) and
// are intentionally not ported, nor are the CompactEpilogue and
// EarlyTmemRelease epilogue experiments (see mlp3_s6_bounce /
// mlp3_s5_early_release) — both were rejected and are omitted here.
//
// Host-launcher contract (no launcher lives in this header — these are
// hard requirements on any .cu translation unit that selects Compute=100,
// e.g. an mlp_bwd.cuh fused entry point):
//   1. Grid: gridDim.x must be even (mlp3_fwd __traps if it is not);
//      launch must use cudaLaunchKernelEx with cudaLaunchConfig_t::
//      cudaLaunchAttributeClusterDimension = {ClusterM, 1, 1}.
//   2. The kernel function must first be registered via
//      cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)
//      before the cudaLaunchKernelEx call — the driver rejects a 2-CTA
//      cluster launch of a kernel that exceeds the "portable" static
//      shared-memory budget without this opt-in.
//   3. The dY^T / Z TMA operands passed in as TmaLoadDYT / TmaLoadZ MUST
//      be constructed with cute::make_tma_copy_A_sm100 /
//      cute::make_tma_copy_B_sm100 (NOT the ordinary 1SM make_tma_copy)
//      — only the _sm100 constructors produce the cta_group::2,
//      pair-aware copy atoms that create_tma_multicast_mask /
//      tma_partition below assume. The dA reduce-add operand stays an
//      ordinary (non-paired) TMA_REDUCE_ADD, per above.
//   4. Dynamic shared memory for Mlp3Smem2Sm<Traits> / Mlp3FusedSmem2Sm
//      <Traits> must be sized and opted in at runtime by the launcher —
//      query cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemory
//      PerBlockOptin, device), then cudaFuncSetAttribute(kernel,
//      cudaFuncAttributeMaxDynamicSharedMemorySize, requested) with
//      requested <= optin. This header intentionally has no compile-
//      time static_assert on a fixed smem byte ceiling (only the TMEM
//      column assert on AccStages*TileN below) — the ceiling is a
//      runtime, device-dependent quantity and must not be hardcoded.
template <
	typename Element_,
	int TileM_ = 256,
	int TileN_ = 256,
	int TileK_ = 64,
	int Stages_ = 3,
	int EpiChunkN_ = 64,
	int AccStages_ = 2,
	int ClusterM_ = 2>
struct Mlp3Traits2Sm {
	using Element = Element_;
	using ElementAccum = float;

	static constexpr int TileM = TileM_;
	static constexpr int TileN = TileN_;
	static constexpr int TileK = TileK_;
	static constexpr int Stages = Stages_;
	static constexpr int EpiChunkN = EpiChunkN_;
	static constexpr int AccStages = AccStages_;
	static constexpr int ClusterM = ClusterM_;
	static constexpr int NumPairs = ClusterM / 2;
	static constexpr int CtaTileM = TileM / 2;
	static constexpr int AtomTileM = 64;
	static constexpr int WgTileN = TileN / 2;
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;
	static constexpr int kAtomsPerCta = CtaTileM / AtomTileM;

	static_assert(TileM == 256,
		"MLP3 2SM requires joined TileM=256; joined TileM=128 is invalid");
	static_assert(TileN == 128 || TileN == 256,
		"MLP3 2SM supports TileN=128 or TileN=256");
	static_assert(TileK == 32 || TileK == 64,
		"MLP3 2SM pilot supports TileK=32 or 64");
	static_assert(Stages >= 2 && Stages <= 12,
		"MLP3 2SM supports two to twelve mainloop stages");
	static_assert(CtaTileM == 128, "each peer CTA must own 128 M rows");
	static_assert(WgTileN % EpiChunkN == 0,
		"EpiChunkN must divide each warpgroup's N half");
	static_assert(kAtomsPerCta == 2,
		"each CTA must reduce-add two 64-row atoms");
	static_assert(AccStages >= 2, "MLP3 2SM requires accumulator double buffering");
	static_assert(AccStages * TileN <= 512,
		"MLP3 2SM accumulator stages must fit in 512 TMEM columns");
	static_assert(ClusterM == 2,
		"MLP3 2SM only ports the validated single-pair path; ClusterM=4 "
		"(two-pair N-split, evaluated together with shared-operand "
		"multicast) was rejected -- see the mlp3_multicast bench results");

	using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
	using ClusterShape = Shape<Int<ClusterM>, _1, _1>;
	using AtomThrShape = Shape<_2, _1, _1>;

	using TiledMma2Sm = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element, Element, ElementAccum, TileM, TileN,
			UMMA::Major::MN, UMMA::Major::MN>{}));
	static_assert(size(typename TiledMma2Sm::AtomThrID{}) == 2,
		"MLP3 2SM requires a two-CTA MMA atom");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma2Sm::AtomThrID{})));

	using MmaShapeA_MK = decltype(partition_shape_A(
		TiledMma2Sm{}, make_shape(Int<TileM>{}, Int<TileK>{})));
	using MmaShapeB_NK = decltype(partition_shape_B(
		TiledMma2Sm{}, make_shape(Int<TileN>{}, Int<TileK>{})));

	using SmemLayoutAtom = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutDYT = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeA_MK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutZ = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeB_NK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutDYT_1 =
		decltype(SmemLayoutDYT{}(_, _, _, Int<0>{}));
	using SmemLayoutZ_1 =
		decltype(SmemLayoutZ{}(_, _, _, Int<0>{}));

	static constexpr int TmaTransBytesDYT =
		static_cast<int>(cosize_v<SmemLayoutDYT_1> * sizeof(Element));
	static constexpr int TmaTransBytesZ =
		static_cast<int>(cosize_v<SmemLayoutZ_1> * sizeof(Element));
	static constexpr int TmaTransBytes =
		2 * (TmaTransBytesDYT + TmaTransBytesZ);

	using MainloopPipelineUmma2Sm = cutlass::PipelineTmaUmmaAsync<
		Stages, ClusterShape, AtomThrShape>;
	using MainloopPipeline = MainloopPipelineUmma2Sm;
	using MainloopPipelineUmma = MainloopPipelineUmma2Sm;
	using PipelineState = typename MainloopPipelineUmma2Sm::PipelineState;
	using AccumulatorPipeline2Sm = cutlass::PipelineUmmaAsync<
		AccStages, AtomThrShape>;

	static constexpr int WarpSize = 32;
	static constexpr int WarpGroupSize = 128;
	static constexpr int NumConsumers = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads = 384;

	// One (CtaTileM, EpiChunkN) store slot per epilogue warpgroup; WG0
	// uses [0..S-1], WG1 uses [S..2S-1] (S = smem_store_size, see
	// Mlp3Smem2Sm below). Mirrors the 1SM Mlp3Traits::SmemLayoutStoreSlot,
	// sized CtaTileM (=128) instead of WgTileM.
	using SmemLayoutStoreSlot = Layout<
		Shape<Int<CtaTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

// ═══════════════════════════════════════════════════════════════════
// SM100 2-SM Shared Memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp3Smem2Sm {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size =
		cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size =
		cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size =
		cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element store_buf[2 * smem_store_size];
	alignas(16) typename Traits::MainloopPipelineUmma2Sm::SharedStorage pipe_storage;
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline2Sm::SharedStorage acc_pipe;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
};

// Fused MLP3 2SM shared memory (data only — the mainloop pipe's storage
// lives in the outer fused-kernel smem union; otherwise identical to
// Mlp3Smem2Sm). Mirrors the 1SM Mlp3FusedSmem / Mlp3Smem split.
template <typename Traits>
struct Mlp3FusedSmem2Sm {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size =
		cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size =
		cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size =
		cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element store_buf[2 * smem_store_size];
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline2Sm::SharedStorage acc_pipe;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// SM100 2-SM pipe construction helper
// ═══════════════════════════════════════════════════════════════════
//
// Mirrors mlp3_make_pipe's role/transaction-byte setup, adjusted for the
// paired-CTA cluster: warp 0 is the TMA producer (leader CTA elects the
// arrival), warps 3-11 are UMMA/epilogue consumers.
template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipelineUmma2Sm
mlp3_make_pipe_umma_2sm(
		typename Traits::MainloopPipelineUmma2Sm::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipelineUmma2Sm;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	bool is_consumer = warp_id >= 3 && warp_id <= 11;

	typename Pipeline::Params params;
	params.transaction_bytes = Traits::TmaTransBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = 0;
	if (is_producer) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == 0 &&
			cute::block_rank_in_cluster() % 2 == 0;
	} else if (is_consumer) {
		params.role = Category::Consumer;
	} else {
		params.role = Category::NonParticipant;
	}
	return Pipeline(
		storage, params, typename Traits::ClusterShape{},
		cute::true_type{}, cute::true_type{});
}

// ═══════════════════════════════════════════════════════════════════
// SM100 2-SM producer / consumer
// ═══════════════════════════════════════════════════════════════════
//
// Chunk-fixed walk, same principle as mlp3_producer/mlp3_consumer above:
// each CTA-pair owns ONE (chunk, lane) cell and walks m_tile × kb
// internally. Within a pair, the leader CTA (pair_rank==0) drives the
// paired UMMA and owns the accumulator producer role; both CTAs run the
// epilogue over their own CtaTileM=128 half of the joined 256-row tile.
template <>
struct Mlp3ProducerImpl<100> {
template <
	typename Traits,
	typename Pipeline,
	typename SmemType,
	typename TmaLoadDYT,
	typename TmaLoadZ>
static __device__ __forceinline__ void run(
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
		int ring_kb = 0) {
	// Compute!=100 must stay 1SM: this body issues SM100-only paired-CTA
	// TMA ops (pair-aware tma_partition / create_tma_multicast_mask on a
	// TiledMma2Sm/MainloopPipelineUmma2Sm operand) and must never be live
	// code for an sm_90a compilation pass of this same translation unit.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	auto sDYT = make_tensor(
		make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ = make_tensor(
		make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});

	auto mDYT = tma_load_dyt.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(num_tokens)));
	auto mZT = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));

	typename Traits::TiledMma2Sm tiled_mma;
	int cta_rank = static_cast<int>(cute::block_rank_in_cluster());
	int pair_rank = cta_rank % 2;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	auto pair_layout_vmnk = tiled_divide(
		make_layout(typename Traits::ClusterShape{}),
		make_tile(typename Traits::TiledMma2Sm::AtomThrID{}));
	auto pair_coord_vmnk =
		pair_layout_vmnk.get_flat_coord(pair_rank);
	uint16_t mcast_mask_dyt =
		create_tma_multicast_mask<2>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t mcast_mask_z =
		create_tma_multicast_mask<1>(pair_layout_vmnk, pair_coord_vmnk);

	int total_chunks = num_experts * num_n_tiles;
	int total_cells = total_chunks * outer_split * k_split;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {
		int k_slice = cell_idx % k_split;
		int cell_om = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane = cell_om - chunk_idx * outer_split;

		int e = chunk_idx / num_n_tiles;
		int n_tile = chunk_idx - e * num_n_tiles;
		int walk_begin = lane * num_m_tiles / outer_split;
		int walk_end = (lane + 1) * num_m_tiles / outer_split;

		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e], batch_kb_end);
		if (kb_hi <= kb_lo)
			continue;

		int k_total = kb_hi - kb_lo;
		int k_per = (k_total + k_split - 1) / k_split;
		int slice_lo = kb_lo + k_slice * k_per;
		int slice_hi = min(slice_lo + k_per, kb_hi);
		if (slice_hi <= slice_lo)
			continue;
		kb_lo = slice_lo;
		kb_hi = slice_hi;

		for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
			auto coord = make_coord(m_tile, n_tile, _);
			auto gDYT = local_tile(
				mDYT, typename Traits::TileShape{}, coord,
				Step<_1, X, _1>{});
			auto gZ = local_tile(
				mZT, typename Traits::TileShape{}, coord,
				Step<X, _1, _1>{});
			auto tCgDYT = cta_mma.partition_A(gDYT);
			auto tCgZ = cta_mma.partition_B(gZ);

			auto [tDYTgDYT, tDYTsDYT] = tma_partition(
				tma_load_dyt,
				get<2>(pair_coord_vmnk),
				make_layout(size<2>(pair_layout_vmnk)),
				group_modes<0, 3>(sDYT),
				group_modes<0, 3>(tCgDYT));
			auto [tZgZ, tZsZ] = tma_partition(
				tma_load_z,
				get<1>(pair_coord_vmnk),
				make_layout(size<1>(pair_layout_vmnk)),
				group_modes<0, 3>(sZ),
				group_modes<0, 3>(tCgZ));

			for (int kb = kb_lo; kb < kb_hi; ++kb) {
				int kb_g = ring_kb > 0 ? kb % ring_kb : kb;
				pipe.producer_acquire(state);
				if (cute::elect_one_sync()) {
					auto* barrier = pipe.producer_get_barrier(state);
					copy(
						tma_load_dyt.with(
							*barrier, mcast_mask_dyt),
						tDYTgDYT(_, kb_g),
						tDYTsDYT(_, state.index()));
					copy(
						tma_load_z.with(
							*barrier, mcast_mask_z),
						tZgZ(_, kb_g),
						tZsZ(_, state.index()));
				}
				++state;
			}
		}
	}
	pipe.producer_tail(state);
#else
	__trap();
#endif
}
};  // Mlp3ProducerImpl<100>

template <>
struct Mlp3ConsumerImpl<100> {
template <
	typename Traits,
	bool Expert3D = false,
	typename Pipeline,
	typename SmemType,
	typename TmaReduceAddDA>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
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
	(void)hidden_dim;
	using Element = typename Traits::Element;
	constexpr int TileM = Traits::TileM;
	constexpr int TileN = Traits::TileN;
	constexpr int CtaTileM = Traits::CtaTileM;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int NumEpiRounds = Traits::NumEpiRounds;
	constexpr int kMmaEpiThreads =
		Traits::ConsumerThreads + Traits::WarpSize;

	int cta_rank = static_cast<int>(cute::block_rank_in_cluster());
	int pair_rank = cta_rank % 2;
	bool is_leader_cta = pair_rank == 0;
	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_mma_warp = warp_id == 3;
	bool is_epilogue = warp_id >= 4 && warp_id <= 11;
	int tid_in_epi = threadIdx.x - Traits::WarpGroupSize;
	int wg = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	int tid_in_wg = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	int wg_barrier_id = 1 + wg;
	bool is_wg_leader = is_epilogue && tid_in_wg == 0;

	typename Traits::TiledMma2Sm tiled_mma;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	auto sDYT = make_tensor(
		make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ = make_tensor(
		make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto tCrDYT = cta_mma.make_fragment_A(sDYT);
	auto tCrZ = cta_mma.make_fragment_B(sZ);

	auto cAccFull = make_identity_tensor(
		make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC = cta_mma.partition_C(cAccFull);
	auto tCtAcc = cta_mma.make_fragment_C(tCgC);

	using AccPipe = typename Traits::AccumulatorPipeline2Sm;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp && is_leader_cta
		? AccPipe::ThreadCategory::Producer
		: AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = 2;
	acc_params.initializing_warp = 4;
	AccPipe acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});
	auto acc_prod_state =
		cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
	uint32_t tmem_base = smem.tmem_base;
	tCtAcc.data() = tmem_base;

	constexpr int store_slot_elems = CtaTileM * EpiChunkN;
	Element* store_ptr = smem.store_buf + wg * store_slot_elems;
	auto sStore = make_tensor(
		make_smem_ptr(store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	auto mdA = tma_reduce_da.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_da = tma_reduce_da.get_slice(Int<0>{});

	auto epi_tile =
		make_tile(Int<CtaTileM>{}, Int<EpiChunkN>{});
	auto acc_mn = tCtAcc(make_coord(_, _), _0{}, _0{});
	auto tAcc_epi = flat_divide(acc_mn, epi_tile);
	auto t2r = make_tmem_copy(
		TmemLoadOp<EpiChunkN>{},
		tAcc_epi(_, _, _0{}, _0{}));
	auto thr_t2r = t2r.get_slice(tid_in_wg);
	auto cChunk = make_identity_tensor(
		make_shape(Int<CtaTileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);
	auto tTR_rAcc = make_tensor<float>(shape(tTR_cChunk));

	Layout tmem_warp_layout =
		typename decltype(make_tmem_warp_partitioner(
			tAcc_epi(_, _, _0{}, _0{})))::TiledLayout_TV{};
	constexpr bool predicate_tmem_load =
		size(tmem_warp_layout) != cosize(tmem_warp_layout);

	int total_chunks = num_experts * num_n_tiles;
	int total_cells = total_chunks * outer_split * k_split;
	bool store_in_flight = false;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {
		int k_slice = cell_idx % k_split;
		int cell_om = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane = cell_om - chunk_idx * outer_split;

		int e = chunk_idx / num_n_tiles;
		int n_tile = chunk_idx - e * num_n_tiles;
		int walk_begin = lane * num_m_tiles / outer_split;
		int walk_end = (lane + 1) * num_m_tiles / outer_split;

		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e], batch_kb_end);
		if (kb_hi <= kb_lo)
			continue;

		int k_total = kb_hi - kb_lo;
		int k_per = (k_total + k_split - 1) / k_split;
		int slice_lo = kb_lo + k_slice * k_per;
		int slice_hi = min(slice_lo + k_per, kb_hi);
		if (slice_hi <= slice_lo)
			continue;
		kb_lo = slice_lo;
		kb_hi = slice_hi;

		for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
			if (is_mma_warp && is_leader_cta) {
				acc_pipe.producer_acquire(acc_prod_state);
				int acc_stage = acc_prod_state.index();
				tCtAcc.data() =
					tmem_base + uint32_t(acc_stage * TileN);

				bool first = true;
				for (int kb = kb_lo; kb < kb_hi; ++kb) {
					pipe.consumer_wait(state);
					CUTE_UNROLL
					for (int ks = 0; ks < size<2>(tCrDYT); ++ks) {
						tiled_mma.accumulate_ = first
							? UMMA::ScaleOut::Zero
							: UMMA::ScaleOut::One;
						first = false;
						gemm(
							tiled_mma,
							tCrDYT(_, _, ks, state.index()),
							tCrZ(_, _, ks, state.index()),
							tCtAcc);
					}
					pipe.consumer_release(state);
					++state;
				}
				acc_pipe.producer_commit(acc_prod_state);
				++acc_prod_state;
			}

			if (is_epilogue) {
				acc_pipe.consumer_wait(acc_cons_state);
				int acc_stage = acc_cons_state.index();
				tCtAcc.data() =
					tmem_base + uint32_t(acc_stage * TileN);
				auto acc_mn_stage =
					tCtAcc(make_coord(_, _), _0{}, _0{});
				auto tAcc_epi_stage =
					flat_divide(acc_mn_stage, epi_tile);
				auto tTR_tAcc =
					thr_t2r.partition_S(tAcc_epi_stage);

				CUTE_UNROLL
				for (int round = 0;
				     round < NumEpiRounds;
				     ++round) {
					int chunk = wg * NumEpiRounds + round;
					auto tAccChunk =
						tTR_tAcc(_, _, _, _0{}, chunk);
					bool issue_tmem_load = true;
					if constexpr (predicate_tmem_load) {
						int subpart =
							(tAccChunk.data().dp_ / 32) % 4;
						issue_tmem_load =
							tid_in_wg / Traits::WarpSize == subpart;
					}
					if (issue_tmem_load)
						copy(t2r, tAccChunk, tTR_rAcc);

					int da_m = e * num_m_tiles + m_tile;
					int n_chunk =
						n_tile *
							(Traits::TileN / Traits::EpiChunkN) +
						wg * Traits::NumEpiRounds + round;

					if (store_in_flight)
						cute::tma_store_wait<0>();
					cutlass::arch::NamedBarrier::sync(
						Traits::WarpGroupSize,
						wg_barrier_id);

					if (issue_tmem_load) {
						CUTE_UNROLL
						for (int i = 0;
						     i < size(tTR_rAcc);
						     ++i) {
							int m_local =
								get<0>(tTR_cChunk(i));
							int n_local =
								get<1>(tTR_cChunk(i));
							sStore(m_local, n_local) =
								static_cast<Element>(
									tTR_rAcc(i));
						}
					}
					cutlass::arch::NamedBarrier::sync(
						Traits::WarpGroupSize,
						wg_barrier_id);

					if (is_wg_leader) {
						cute::tma_store_fence();
						CUTE_UNROLL
						for (int atom = 0;
						     atom < Traits::kAtomsPerCta;
						     ++atom) {
							int m_atom =
								(Traits::TileM /
								 Traits::AtomTileM) *
									da_m +
								Traits::kAtomsPerCta *
									pair_rank +
								atom;
							auto sStoreAtom =
								local_tile(
									sStore,
									make_tile(
										Int<Traits::AtomTileM>{},
										Int<Traits::EpiChunkN>{}),
									make_coord(atom, 0));
							auto gdA = local_tile(
								mdA,
								make_tile(
									Int<Traits::AtomTileM>{},
									Int<Traits::EpiChunkN>{}),
								make_coord(m_atom, n_chunk));
							copy(
								tma_reduce_da,
								cta_tma_da.partition_S(
									sStoreAtom),
								cta_tma_da.partition_D(gdA));
						}
						cute::tma_store_arrive();
					}
					store_in_flight = true;
				}

				// TMEM reads done → release the acc so the next
				// (cell, walk) MMA may reuse it (one elected consumer
				// thread arrives). Always released after all epilogue
				// rounds' register→SMEM fold + TMA_REDUCE_ADD store
				// work has been issued for this m_tile — matches the
				// 1SM mlp3_consumer's release placement. (The donor's
				// EarlyTmemRelease experiment moved this earlier for
				// one accumulator stage's worth of extra overlap; it
				// was rejected — see mlp3_s5_early_release — so is not
				// ported.)
				cutlass::arch::NamedBarrier::sync(
					Traits::ConsumerThreads, 0);
				if (tid_in_epi == 0)
					acc_pipe.consumer_release(acc_cons_state);
				++acc_cons_state;
			}
		}
	}

	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
#else
	__trap();
#endif
}
};  // Mlp3ConsumerImpl<100>

// ═══════════════════════════════════════════════════════════════════
// Single-launch driver, chunk-fixed and architecture-dispatched
// ═══════════════════════════════════════════════════════════════════
//
// Compute=90 uses the original per-CTA Hopper pipeline and cell mapping.
// Compute=100 uses a paired-CTA pipeline, pair-granular cell mapping, a
// shared Allocator2Sm allocation, and cluster synchronization.
template <
	typename Traits,
	int Compute = 90,
	bool Expert3D = false,
	typename TmaLoadDYT,
	typename TmaLoadZ,
	typename TmaReduceAddDA>
__device__ __forceinline__ void mlp3_fwd(
		cute::conditional_t<
			Compute == 100,
			Mlp3Smem2Sm<Traits>,
			Mlp3Smem<Traits>>& smem,
		TmaLoadDYT const& tma_load_dyt,
		TmaLoadZ const& tma_load_z,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int batch_kb_start = 0,
		int batch_kb_end = -1,
		int k_split = 1,
		int ring_kb = 0) {
	static_assert(Compute == 90 || Compute == 100,
		"mlp3_fwd supports Compute=90 or Compute=100");
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	constexpr int kFirstConsumerWarp = Compute == 100 ? 3 : 4;
	bool is_consumer =
		warp_id >= kFirstConsumerWarp && warp_id <= 11;

	cute::prefetch_tma_descriptor(
		tma_load_dyt.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_reduce_da.get_tma_descriptor());

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp3_make_pipe_umma_2sm<Traits>(
				smem.pipe_storage);
		else
			return mlp3_make_pipe<Traits>(smem.pipe_storage);
	}();
	using Pipeline = decltype(pipe);

	if constexpr (Compute == 100) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
		cute::TMEM::Allocator2Sm tmem_allocator;
		if (gridDim.x % Traits::ClusterM != 0)
			__trap();
		cute::cluster_sync();
		constexpr int kTmemColumns =
			Traits::AccStages * Traits::TileN;
		if (warp_id == 4) {
			tmem_allocator.allocate(
				kTmemColumns, &smem.tmem_base);
			__syncwarp();
		}
		__syncthreads();
		cute::cluster_sync();
#else
		__trap();
#endif
	} else {
		__syncthreads();
	}

	PipeState producer_state = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState consumer_state;

	int cell_start = static_cast<int>(blockIdx.x);
	int cell_stride = static_cast<int>(gridDim.x);
	if constexpr (Compute == 100) {
		cell_start /= Traits::ClusterM;
		cell_stride /= Traits::ClusterM;
	}
	int kb_end = batch_kb_end >= 0
		? batch_kb_end
		: num_tokens / Traits::TileK;

	if (is_producer) {
		mlp3_producer<Traits, Compute>(
			pipe,
			producer_state,
			smem,
			tma_load_dyt,
			tma_load_z,
			expert_k_starts,
			expert_k_ends,
			num_experts,
			hidden_dim,
			intermediate_dim,
			num_tokens,
			num_m_tiles,
			num_n_tiles,
			outer_split,
			cell_start,
			cell_stride,
			batch_kb_start,
			kb_end,
			k_split,
			ring_kb);
	} else if (is_consumer) {
		if constexpr (Compute == 100) {
			mlp3_consumer<Traits, Compute, Expert3D>(
				pipe,
				consumer_state,
				smem,
				tma_reduce_da,
				expert_k_starts,
				expert_k_ends,
				num_experts,
				hidden_dim,
				intermediate_dim,
				total_n_rows,
				num_m_tiles,
				num_n_tiles,
				outer_split,
				cell_start,
				cell_stride,
				batch_kb_start,
				kb_end,
				k_split);
		} else {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
			mlp3_consumer<Traits, Compute, Expert3D>(
				pipe,
				consumer_state,
				smem,
				tma_reduce_da,
				expert_k_starts,
				expert_k_ends,
				num_experts,
				hidden_dim,
				intermediate_dim,
				total_n_rows,
				num_m_tiles,
				num_n_tiles,
				outer_split,
				cell_start,
				cell_stride,
				batch_kb_start,
				kb_end,
				k_split);
#else
			__trap();
#endif
		}
	}

	if constexpr (Compute == 100) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
		cute::TMEM::Allocator2Sm tmem_allocator;
		__syncthreads();
		cute::cluster_sync();
		constexpr int kTmemColumns =
			Traits::AccStages * Traits::TileN;
		if (warp_id == 4) {
			tmem_allocator.release_allocation_lock();
			tmem_allocator.free(
				smem.tmem_base, kTmemColumns);
		}
#endif
	}
}

} // namespace liger
