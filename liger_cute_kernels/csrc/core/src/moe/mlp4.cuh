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
//   WG0 warps 1-2  : Idle / reserved for fused-kernel communication
//   WG0 warp 3     : Idle on SM90; dedicated UMMA producer on SM100
//   WG1 (warps 4-7)  : WGMMA consumer on SM90; epilogue WG0 on SM100
//   WG2 (warps 8-11) : WGMMA consumer on SM90; epilogue WG1 on SM100
//
// Chunk-fixed walk (same principle as mlp3): hold the SHARED operand
// chunk-constant and walk the cooperation (split) axis:
//   kMSplit=true  (M-split, X shared):       chunk = (e, n_tile), walks m_tile; holds X(n,*)
//   kMSplit=false (N-split, dU^T/dV^T shared): chunk = (e, m_tile), walks n_tile; holds A(m,*)
// In the M-split default, X(n_tile) is held across BOTH the m-walk and the
// dB/dC phases — re-read for phase dC hits L2.
// ═══════════════════════════════════════════════════════════════════

#include "models.cuh"
#include "tmem_load_op.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>

// ── SM100 (Compute=100 / Blackwell UMMA) headers ────────────────────
// Header-only; safe to include under an sm_90a build. The tcgen05 PTX they
// emit is guarded by CUTE_ARCH_TCGEN05_* and only the Compute=100 body (itself
// gated on __CUDA_ARCH__ >= 1000) ever instantiates it. Mirrors mlp5.cuh /
// mlp1_fused.cuh.
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>  // PipelineTmaUmmaAsync, PipelineUmmaAsync

// ── SM100 2-SM (paired-CTA cluster) headers ─────────────────────────
// Needed by the paired MLP4 path below: cluster synchronization/rank queries
// for the (2,1,1) cluster and the 2SM TMA copy traits used by tma_partition().
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>

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

	// ── SM100 (Compute=100 / Blackwell UMMA) ────────────────────────
	// The mainloop pipeline is the UMMA-aware TMA pipeline (its consumer_release
	// issues the UMMA-gated smem-buffer arrival). Params/PipelineState/
	// SharedStorage alias the Hopper PipelineTmaAsync's, so smem.pipe_storage
	// (typed MainloopPipeline::SharedStorage) drives BOTH — the standalone
	// launcher passes it to mlp4_make_pipe_umma unchanged. num_consumers=1 on
	// Blackwell (one umma_arrive per stage releases the buffer).
	using MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<
		Stages, cute::Shape<cute::_1, cute::_1, cute::_1>,
		cute::Shape<cute::_1, cute::_1, cute::_1>>;
	// Per-WG UMMA atom: SM100_MMA_F16BF16_SS with M=WgTileM=128, N=WgTileN=128,
	// BOTH operands MN-major (A = dU^T/dV^T is M-major in smem; X is N-major).
	// Warp 3 issues TWO of these per k-step into two independent
	// (128,128) TMEM accumulators (one per consumer WG) — see
	// Mlp4ConsumerImpl<100>. UMMA requires M ∈ {64,128}, so the 256-wide tile
	// cannot use one atom; the per-WG (128,128) atom is used for both configs.
	using TiledMmaUmma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, float, WgTileM, WgTileN,
		                     UMMA::Major::MN, UMMA::Major::MN>{}));
	// Double-buffered TMEM accumulators: each stage owns one (WgTileM,WgTileN)
	// accumulator per epilogue WG. The two dB/dC phases still clear
	// independently — this only lets MMA for the next (phase,walk) run while
	// the epilogue drains the previous accumulator stage.
	static constexpr int AccStages = 2;
	static_assert(AccStages * (2 * WgTileN) <= 512,
		"SM100 MLP4 needs AccStages x two per-WG accumulators to fit in 512 TMEM columns");
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccStages>;

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

template <typename Traits, int Compute>
using Mlp4MainloopPipelineFor = cute::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

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

	// SM100 (Compute=100) only: landing slot for tcgen05.alloc's granted TMEM
	// base address, plus the accumulator pipeline's barrier storage (UMMA→
	// epilogue handoff). The Hopper (Compute=90) path never touches these;
	// SM100 launchers allocate/free TMEM outside the consumer and publish the
	// base here; the accumulator pipeline storage still lives in this smem.
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

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
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

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

// SM100 (Compute=100) mainloop pipe. Mirrors mlp4_make_pipe but builds the
// UMMA-aware TMA pipeline (PipelineTmaUmmaAsync) whose consumer_release issues
// the UMMA-gated smem-buffer arrival. num_consumers=1: on Blackwell the buffer
// is released by a single umma_arrive per stage (the MMA warp), not per
// consumer thread. Params/PipelineState/SharedStorage alias the Hopper
// PipelineTmaAsync's, and the ctor arg pattern is identical, so the (shared)
// mlp4_producer and the Compute=100 consumer drive it unchanged.
template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipelineUmma
mlp4_make_pipe_umma(typename Traits::MainloopPipelineUmma::SharedStorage& storage) {
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

template <int Compute>
struct Mlp4ProducerImpl;

template <>
struct Mlp4ProducerImpl<90> {
template <typename Traits,
          typename Pipeline, typename SmemType,
          typename TmaLoadX, typename TmaLoadA>
static __device__ __forceinline__ void run(
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
};  // Mlp4ProducerImpl<90>

template <typename Traits, int Compute = 90,
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
		int ring_kb = 0) {
	Mlp4ProducerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_load_x, tma_load_dut, tma_load_dvt,
		expert_k_starts, expert_k_ends, num_experts,
		intermediate_dim, hidden_dim, num_tokens,
		num_m_tiles, num_n_tiles, outer_split,
		cell_start, cell_stride,
		batch_kb_start, batch_kb_end, k_split, ring_kb);
}

// ═══════════════════════════════════════════════════════════════════
// Consumer — architecture-specialized on `int Compute`
// ═══════════════════════════════════════════════════════════════════
//
// Primary template is undefined; one full specialization per supported compute
// capability provides a `run(...)` member (a function-template on the Traits/
// Pipeline/Smem/Tma types, since function templates can't be partially
// specialized). The free function `mlp4_consumer` below forwards to the right
// specialization — call sites (mlp_bwd.cuh's mlp4_consumer<Traits4>(...)) stay
// unchanged (Compute defaults to 90).
//
//   Compute=90  → Hopper / WGMMA (cooperative 2-WG, register acc, M-atom
//                 interleave + (m_loc/128)*64+(m_loc%64) store remap)
//   Compute=100 → Blackwell / UMMA (warp 3 issues 2 per-WG (128,128) UMMAs
//                 into double-buffered TMEM stages; warps 4-11 run the
//                 epilogue, with independent per-phase clear and contiguous
//                 per-WG store — no remap)

template <int Compute>
struct Mlp4ConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper / WGMMA. (Verbatim from the original mlp4_consumer.)
// ───────────────────────────────────────────────────────────────────
template <>
struct Mlp4ConsumerImpl<90> {
template <typename Traits,
          bool Expert3D = false,
          int Compute = 90,
          typename Pipeline, typename SmemType,
          typename TmaReduceAddDB, typename TmaReduceAddDC>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDB const& tma_reduce_db,
		TmaReduceAddDC const& tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
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

	(void)Compute;
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

	auto mdB = [&]() {
		if constexpr (Expert3D) {
			return tma_reduce_db.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_reduce_db.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_m_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();
	auto mdC = [&]() {
		if constexpr (Expert3D) {
			return tma_reduce_dc.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_reduce_dc.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_m_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();
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
						int out_m = Expert3D ? m_tile : e * num_m_tiles + m_tile;
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
								auto gdB = [&]() {
									if constexpr (Expert3D) {
										return local_tile(mdB,
											make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
											make_coord(m_atom_row, n_tile_idx, e));
									} else {
										return local_tile(mdB,
											make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
											make_coord(m_atom_row, n_tile_idx));
									}
								}();
								copy(tma_reduce_db, cta_tma_db.partition_S(sStore_a),
									cta_tma_db.partition_D(gdB));
							} else {
								auto gdC = [&]() {
									if constexpr (Expert3D) {
										return local_tile(mdC,
											make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
											make_coord(m_atom_row, n_tile_idx, e));
									} else {
										return local_tile(mdC,
											make_tile(Int<Traits::AtomTileM>{}, Int<Traits::EpiChunkN>{}),
											make_coord(m_atom_row, n_tile_idx));
									}
								}();
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
};  // struct Mlp4ConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Free-function forwarder. Existing call sites (mlp4_consumer<Traits>(...))
// are unchanged (Compute defaults to 90); pass mlp4_consumer<Traits, 100>(...)
// to select the Blackwell path.
// ───────────────────────────────────────────────────────────────────
template <typename Traits, int Compute = 90, bool Expert3D = false,
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
		int intermediate_dim,
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
	Mlp4ConsumerImpl<Compute>::template run<Traits, Expert3D>(
		pipe, state, smem, tma_reduce_db, tma_reduce_dc,
		expert_k_starts, expert_k_ends, num_experts,
		intermediate_dim, hidden_dim, total_m_rows,
		num_m_tiles, num_n_tiles, outer_split,
		cell_start, cell_stride, batch_kb_start, batch_kb_end, k_split);
}

// ═══════════════════════════════════════════════════════════════════
// SM100 2-SM (paired-CTA) MLP4 — Mlp4Traits2Sm, pipe/producer/consumer
// ═══════════════════════════════════════════════════════════════════
//
// A cluster-(2,1,1) SM100 UMMA path: two CTAs jointly compute one
// (TileM=256, TileN=256) tile via SM100_MMA_F16BF16_2x1SM_SS, each CTA
// owning CtaTileM=128 M-rows (kAtomsPerCta=2 wgmma-style 64-row atoms).
// Reuses the generic Mlp4Smem<Traits> / Mlp4FusedSmem<Traits> above —
// Mlp4Traits2Sm supplies the same nested type names (SmemLayoutA,
// SmemLayoutX, SmemLayoutStoreSlot, MainloopPipeline, AccumulatorPipeline)
// so no separate 2SM smem struct is needed.
//
// Scope: only the BASIC ClusterM=2 (single paired-CTA) configuration
// required by the fused backward kernel is ported here. Intentionally
// EXCLUDED (upstream "topology experiment" scope, not ported): the
// multi-pair (ClusterM=4) path, shared-operand multicast
// (MulticastShared_), N-split, the compact-epilogue variant, and early
// TMEM release — the latter three never existed for MLP4 in the donor
// tree (MLP3-only experiments) and are noted here only for the audit
// record. Stages and EpiChunkN remain free template parameters (no
// S5/E64 production hardcoding); callers pick tuned values at the
// trait-instantiation site (e.g. moe_bwd.cu).
//
// Cross-file invariants a caller/launcher MUST uphold (none of this is
// enforceable from inside this header-only file; documented for the
// next integrator of mlp_bwd.cuh / moe_bwd.cu / mlp4.cu):
//   * grid.x must be a MULTIPLE OF ClusterM (=2) before launch; the runtime
//     `__trap()` check in mlp4_fwd is only a defensive backstop.
//   * Compute != 100 must stay on the 1SM traits/path. Mlp4Traits2Sm's
//     TiledMma2Sm / ClusterShape / MainloopPipelineUmma2Sm /
//     AccumulatorPipeline2Sm members do not exist on Mlp4Traits, so mixing
//     an architecture with the wrong trait remains a compile-time error.
//   * Launch must use cudaLaunchKernelEx with a cluster dimension of
//     (ClusterM,1,1), and cudaFuncSetAttribute(...,
//     cudaFuncAttributeNonPortableClusterSizeAllowed, 1) must be set on
//     the kernel beforehand.
//   * The dynamic shared-memory allocation for Mlp4Smem<Mlp4Traits2Sm<...>>
//     must be sized/capped against the device's *dynamic* opt-in limit
//     (cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlockOptin,
//     ...)), not the static 48 KiB default, via
//     cudaFuncAttributeMaxDynamicSharedMemorySize.
//   * X / dU^T / dV^T TMA descriptors must be built with
//     make_tma_copy_A_sm100 / make_tma_copy_B_sm100 (SM100_TMA_2SM_LOAD
//     copy op) — the plain SM90 TMA copy-atom builders are not
//     pair-aware and would not multicast/replicate operands across the
//     two peer CTAs. dB / dC stay TMA REDUCE_ADD outputs.
//   * The producer/consumer specializations assume TMEM is already allocated
//     and smem.tmem_base populated. The standalone mlp4_fwd driver owns that
//     allocation and rendezvouses peer CTAs with cute::cluster_sync() (safe
//     because both CTAs run symmetric control flow). A fused caller sharing
//     one TMEM region across warp groups with asymmetric control flow must not
//     reuse a bare cluster_sync() for that rendezvous; it must pair
//     cute::TMEM::Allocator2Sm's
//     allocate()/free() with its own dedicated
//     cutlass::arch::ClusterBarrier rendezvous scoped to just the
//     participating warps, the way donor's mlp_bwd.cuh
//     MlpFusedBwdSmem::tmem_pair_barrier does. That smem field and its
//     init/arrive/wait call sites live in mlp_bwd.cuh, outside this
//     file's scope, and are not present in this target repo yet.

template <
	typename Element_,
	int TileM_     = 256,
	int TileN_     = 256,
	int TileK_     = 64,
	int Stages_    = 3,
	int EpiChunkN_ = 128,
	int AccStages_ = 2,
	int ClusterM_  = 2
>
struct Mlp4Traits2Sm {
	using Element      = Element_;
	using ElementAccum = float;

	static constexpr int TileM     = TileM_;
	static constexpr int TileN     = TileN_;
	static constexpr int TileK     = TileK_;
	static constexpr int Stages    = Stages_;
	static constexpr int EpiChunkN = EpiChunkN_;
	static constexpr int AccStages = AccStages_;
	static constexpr int ClusterM  = ClusterM_;
	static constexpr int NumPairs  = ClusterM / 2;
	static constexpr int CtaTileM     = TileM / 2;
	static constexpr int AtomTileM    = 64;
	static constexpr int WgTileN      = TileN / 2;
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;
	static constexpr int kAtomsPerCta = CtaTileM / AtomTileM;

	static_assert(TileM == 256, "MLP4 2SM requires joined TileM=256");
	static_assert(TileN == 256, "MLP4 2SM requires joined TileN=256");
	static_assert(TileK == 32 || TileK == 64,
		"MLP4 2SM supports TileK=32 or 64");
	static_assert(Stages >= 2 && Stages <= 12,
		"MLP4 2SM supports two to twelve mainloop stages");
	static_assert(CtaTileM == 128, "each peer CTA must own 128 M rows");
	static_assert(WgTileN % EpiChunkN == 0,
		"EpiChunkN must divide each warpgroup's N half");
	static_assert(kAtomsPerCta == 2,
		"each CTA must reduce-add two 64-row atoms");
	static_assert(AccStages == 2,
		"MLP4 2SM fixes accumulator double buffering");
	static_assert(AccStages * TileN <= 512,
		"MLP4 2SM accumulator stages must fit in 512 TMEM columns");
	static_assert(ClusterM == 2,
		"MLP4 2SM ports only the basic single paired-CTA cluster; "
		"ClusterM=4 pair-of-pairs multicast is an excluded experiment");

	using TileShape    = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
	using ClusterShape = Shape<Int<ClusterM>, _1, _1>;
	using AtomThrShape = Shape<_2, _1, _1>;

	using TiledMma2Sm = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element, Element, ElementAccum, TileM, TileN,
			UMMA::Major::MN, UMMA::Major::MN>{}));
	static_assert(size(typename TiledMma2Sm::AtomThrID{}) == 2,
		"MLP4 2SM requires a two-CTA MMA atom");

	using MmaShapeA_MK = decltype(partition_shape_A(
		TiledMma2Sm{}, make_shape(Int<TileM>{}, Int<TileK>{})));
	using MmaShapeB_NK = decltype(partition_shape_B(
		TiledMma2Sm{}, make_shape(Int<TileN>{}, Int<TileK>{})));

	using SmemLayoutAtom = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeA_MK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutX = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeB_NK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutA_1 = decltype(SmemLayoutA{}(_, _, _, Int<0>{}));
	using SmemLayoutX_1 = decltype(SmemLayoutX{}(_, _, _, Int<0>{}));

	static constexpr int TmaTransBytesA =
		static_cast<int>(cosize_v<SmemLayoutA_1> * sizeof(Element));
	static constexpr int TmaTransBytesX =
		static_cast<int>(cosize_v<SmemLayoutX_1> * sizeof(Element));
	// Both peer CTAs issue each 2SM load, and both contributions arrive on
	// the cluster-scoped full barrier.
	static constexpr int TmaTransBytes =
		2 * (TmaTransBytesA + TmaTransBytesX);

	using MainloopPipelineUmma2Sm = cutlass::PipelineTmaUmmaAsync<
		Stages, ClusterShape, AtomThrShape>;
	// Aliased to the names the generic Mlp4Smem<Traits>/Mlp4FusedSmem<Traits>
	// (and Mlp4MainloopPipelineFor<Traits, Compute>) look up.
	using MainloopPipelineUmma = MainloopPipelineUmma2Sm;
	using MainloopPipeline     = MainloopPipelineUmma2Sm;
	using PipelineState = typename MainloopPipelineUmma2Sm::PipelineState;
	using AccumulatorPipeline2Sm = cutlass::PipelineUmmaAsync<
		AccStages, AtomThrShape>;
	using AccumulatorPipeline = AccumulatorPipeline2Sm;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumConsumers    = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads      = 384;

	using SmemLayoutStoreSlot = Layout<
		Shape<Int<CtaTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

// Compatibility alias: Mlp4Traits2Sm's nested type names line up
// field-for-field with the generic Mlp4Smem<Traits> above (see the
// comment atop this section), so no bespoke 2SM smem struct is defined —
// this alias just lets callers/tooling written against the name
// "Mlp4Smem2Sm<Traits>" (e.g. bench_opt/bench_mlp4_2sm.cu) resolve to the
// same generic struct without any source changes on their end.
template <typename Traits>
using Mlp4Smem2Sm = Mlp4Smem<Traits>;

// SM100 2SM mainloop pipe. Mirrors mlp4_make_pipe_umma; num_consumers=1
// (one umma_arrive per stage releases the buffer). is_leader restricts the
// TMA-issuing thread to lane 0 of the pair's rank-0 CTA — both peer CTAs'
// smem buffers arrive on the shared cluster barrier via the 2SM TMA
// multicast, so only one CTA needs to issue the copy.
template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipelineUmma2Sm
mlp4_make_pipe_umma_2sm(
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
			threadIdx.x == 0 && cute::block_rank_in_cluster() % 2 == 0;
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
// 2SM producer/consumer — chunk-fixed, sequential dB/dC (same cell
// decode as mlp4_producer/mlp4_consumer above), ClusterM=2 only: the
// walk axis (num_m_tiles) is partitioned by `lane` alone (no cluster/pair
// multicast fan-out across multiple pairs).
// ═══════════════════════════════════════════════════════════════════

template <>
struct Mlp4ProducerImpl<100> {
template <
	typename Traits,
	typename Pipeline,
	typename TmaLoadX,
	typename TmaLoadA,
	typename SmemType>
static __device__ __forceinline__ void run(
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
		int ring_kb = 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	auto sA = make_tensor(
		make_smem_ptr(smem.A_data()), typename Traits::SmemLayoutA{});
	auto sX = make_tensor(
		make_smem_ptr(smem.X_data()), typename Traits::SmemLayoutX{});

	auto mX = tma_load_x.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(num_tokens)));
	auto mdUT = tma_load_dut.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));
	auto mdVT = tma_load_dvt.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));

	typename Traits::TiledMma2Sm tiled_mma;
	int cta_rank  = static_cast<int>(cute::block_rank_in_cluster());
	int pair_rank = cta_rank % 2;
	auto cta_mma  = tiled_mma.get_slice(pair_rank);
	// ClusterM=2 → the pair IS the whole cluster.
	auto pair_layout_vmnk = tiled_divide(
		make_layout(typename Traits::ClusterShape{}),
		make_tile(typename Traits::TiledMma2Sm::AtomThrID{}));
	auto pair_coord_vmnk = pair_layout_vmnk.get_flat_coord(pair_rank);
	uint16_t mcast_mask_a =
		create_tma_multicast_mask<2>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t mcast_mask_x =
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
		// Balanced split: multiply BEFORE dividing so any remainder
		// (num_m_tiles % outer_split != 0) is distributed across the
		// trailing lanes instead of being silently dropped — e.g.
		// num_m_tiles=3, outer_split=2 gives lane0=[0,1), lane1=[1,3),
		// covering all 3 tiles. `lane * (num_m_tiles/outer_split)` (a
		// naive floor-then-multiply) would leave the tail tile
		// unvisited by any lane. ClusterM=2 needs no further
		// valid_m_tile guard: walk_end for the last lane is always
		// exactly num_m_tiles.
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

		CUTE_UNROLL
		for (int phase = 0; phase < 2; ++phase) {
			auto mA = phase == 0 ? mdUT : mdVT;
			const auto& tma_a =
				phase == 0 ? tma_load_dut : tma_load_dvt;

			for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
				auto coord = make_coord(m_tile, n_tile, _);
				auto gA = local_tile(
					mA, typename Traits::TileShape{}, coord,
					Step<_1, X, _1>{});
				auto gX = local_tile(
					mX, typename Traits::TileShape{}, coord,
					Step<X, _1, _1>{});
				auto tCgA = cta_mma.partition_A(gA);
				auto tCgX = cta_mma.partition_B(gX);

				auto [tAgA, tAsA] = tma_partition(
					tma_a,
					get<2>(pair_coord_vmnk),
					make_layout(size<2>(pair_layout_vmnk)),
					group_modes<0, 3>(sA),
					group_modes<0, 3>(tCgA));
				auto [tXgX, tXsX] = tma_partition(
					tma_load_x,
					get<1>(pair_coord_vmnk),
					make_layout(size<1>(pair_layout_vmnk)),
					group_modes<0, 3>(sX),
					group_modes<0, 3>(tCgX));

				for (int kb = kb_lo; kb < kb_hi; ++kb) {
					int kb_g = ring_kb > 0 ? kb % ring_kb : kb;
					pipe.producer_acquire(state);
					if (cute::elect_one_sync()) {
						auto* barrier =
							pipe.producer_get_barrier(state);
						copy(
							tma_a.with(*barrier, mcast_mask_a),
							tAgA(_, kb_g),
							tAsA(_, state.index()));
						copy(
							tma_load_x.with(
								*barrier, mcast_mask_x),
							tXgX(_, kb_g),
							tXsX(_, state.index()));
					}
					++state;
				}
			}
		}
	}
	pipe.producer_tail(state);
#else
	__trap();
#endif
}
};  // Mlp4ProducerImpl<100>

template <>
struct Mlp4ConsumerImpl<100> {
template <
	typename Traits,
	bool Expert3D = false,
	typename Pipeline,
	typename TmaReduceAddDB,
	typename TmaReduceAddDC,
	typename SmemType>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDB const& tma_reduce_db,
		TmaReduceAddDC const& tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	(void)intermediate_dim;
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
	int tid_in_wg = is_epilogue
		? tid_in_epi % Traits::WarpGroupSize
		: 0;
	int wg_barrier_id = 1 + wg;
	bool is_wg_leader = is_epilogue && tid_in_wg == 0;

	typename Traits::TiledMma2Sm tiled_mma;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	auto sA = make_tensor(
		make_smem_ptr(smem.A_data()), typename Traits::SmemLayoutA{});
	auto sX = make_tensor(
		make_smem_ptr(smem.X_data()), typename Traits::SmemLayoutX{});
	auto tCrA = cta_mma.make_fragment_A(sA);
	auto tCrX = cta_mma.make_fragment_B(sX);

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

	auto mdB = tma_reduce_db.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_m_rows),
		static_cast<int64_t>(hidden_dim)));
	auto mdC = tma_reduce_dc.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_m_rows),
		static_cast<int64_t>(hidden_dim)));
	auto cta_tma_db = tma_reduce_db.get_slice(Int<0>{});
	auto cta_tma_dc = tma_reduce_dc.get_slice(Int<0>{});

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
		// Match the producer's balanced split: multiply-before-divide
		// avoids dropping the tail
		// M-tile(s) when num_m_tiles % outer_split != 0.
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

		CUTE_UNROLL
		for (int phase = 0; phase < 2; ++phase) {
			for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
				if (is_mma_warp && is_leader_cta) {
					acc_pipe.producer_acquire(acc_prod_state);
					int acc_stage = acc_prod_state.index();
					tCtAcc.data() =
						tmem_base + uint32_t(acc_stage * TileN);

					bool first_mma = true;
					for (int kb = kb_lo; kb < kb_hi; ++kb) {
						pipe.consumer_wait(state);
						CUTE_UNROLL
						for (int ks = 0;
						     ks < size<2>(tCrA);
						     ++ks) {
							tiled_mma.accumulate_ = first_mma
								? UMMA::ScaleOut::Zero
								: UMMA::ScaleOut::One;
							first_mma = false;
							gemm(
								tiled_mma,
								tCrA(_, _, ks, state.index()),
								tCrX(_, _, ks, state.index()),
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
								tid_in_wg / Traits::WarpSize ==
								subpart;
						}
						if (issue_tmem_load)
							copy(t2r, tAccChunk, tTR_rAcc);

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
							int out_m =
								e * num_m_tiles + m_tile;
							int n_chunk =
								n_tile *
									(TileN / EpiChunkN) +
								wg * NumEpiRounds +
								round;
							CUTE_UNROLL
							for (int atom = 0;
							     atom <
								 Traits::kAtomsPerCta;
							     ++atom) {
								int m_atom =
									(TileM /
									 Traits::AtomTileM) *
										out_m +
									Traits::kAtomsPerCta *
										pair_rank +
									atom;
								auto sStoreAtom =
									local_tile(
										sStore,
										make_tile(
											Int<
												Traits::
													AtomTileM>{},
											Int<
												EpiChunkN>{}),
										make_coord(
											atom, 0));
								if (phase == 0) {
									auto gdB = local_tile(
										mdB,
										make_tile(
											Int<
												Traits::
													AtomTileM>{},
											Int<
												EpiChunkN>{}),
										make_coord(
											m_atom,
											n_chunk));
									copy(
										tma_reduce_db,
										cta_tma_db.partition_S(
											sStoreAtom),
										cta_tma_db.partition_D(
											gdB));
								} else {
									auto gdC = local_tile(
										mdC,
										make_tile(
											Int<
												Traits::
													AtomTileM>{},
											Int<
												EpiChunkN>{}),
										make_coord(
											m_atom,
											n_chunk));
									copy(
										tma_reduce_dc,
										cta_tma_dc.partition_S(
											sStoreAtom),
										cta_tma_dc.partition_D(
											gdC));
								}
							}
							cute::tma_store_arrive();
						}
						store_in_flight = true;
					}

					cutlass::arch::NamedBarrier::sync(
						Traits::ConsumerThreads, 0);
					// One elected release from each CTA satisfies the
					// cluster-scoped accumulator pipeline's count of two.
					if (tid_in_epi == 0)
						acc_pipe.consumer_release(
							acc_cons_state);
					++acc_cons_state;
				}
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
};  // Mlp4ConsumerImpl<100>

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
	typename TmaLoadX,
	typename TmaLoadA,
	typename TmaReduceAddDB,
	typename TmaReduceAddDC>
__device__ __forceinline__ void mlp4_fwd(
		Mlp4Smem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadA const& tma_load_dut,
		TmaLoadA const& tma_load_dvt,
		TmaReduceAddDB const& tma_reduce_db,
		TmaReduceAddDC const& tma_reduce_dc,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
		int hidden_dim,
		int num_tokens,
		int total_m_rows,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int batch_kb_start = 0,
		int batch_kb_end = -1,
		int k_split = 1,
		int ring_kb = 0) {
	static_assert(Compute == 90 || Compute == 100,
		"mlp4_fwd supports Compute=90 or Compute=100");
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	constexpr int kFirstConsumerWarp = Compute == 100 ? 3 : 4;
	bool is_consumer =
		warp_id >= kFirstConsumerWarp && warp_id <= 11;

	cute::prefetch_tma_descriptor(
		tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_dut.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_dvt.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_reduce_db.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_reduce_dc.get_tma_descriptor());

	auto pipe = [&]() {
		if constexpr (Compute == 100)
			return mlp4_make_pipe_umma_2sm<Traits>(
				smem.pipe_storage);
		else
			return mlp4_make_pipe<Traits>(smem.pipe_storage);
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
		mlp4_producer<Traits, Compute>(
			pipe,
			producer_state,
			smem,
			tma_load_x,
			tma_load_dut,
			tma_load_dvt,
			expert_k_starts,
			expert_k_ends,
			num_experts,
			intermediate_dim,
			hidden_dim,
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
			mlp4_consumer<Traits, Compute, Expert3D>(
				pipe,
				consumer_state,
				smem,
				tma_reduce_db,
				tma_reduce_dc,
				expert_k_starts,
				expert_k_ends,
				num_experts,
				intermediate_dim,
				hidden_dim,
				total_m_rows,
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
			mlp4_consumer<Traits, Compute, Expert3D>(
				pipe,
				consumer_state,
				smem,
				tma_reduce_db,
				tma_reduce_dc,
				expert_k_starts,
				expert_k_ends,
				num_experts,
				intermediate_dim,
				hidden_dim,
				total_m_rows,
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
