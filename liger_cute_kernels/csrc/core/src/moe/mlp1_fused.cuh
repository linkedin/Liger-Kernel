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

// SM100 (Blackwell / UMMA) support for the Compute=100 consumer
// specialization. These headers are header-only and safe to include under
// an sm_90a build — the tcgen05 PTX they emit is guarded by
// CUTE_ARCH_TCGEN05_* and only the Compute=100 body (itself gated on
// __CUDA_ARCH__ >= 1000) ever instantiates it.
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>  // PipelineTmaUmmaAsync, PipelineUmmaAsync

namespace liger {

using namespace cute;

// ═══════════════════════════════════════════════════════════════════
// TMEM → register load-op selector (SM100 / Compute=100 epilogue)
// ═══════════════════════════════════════════════════════════════════
// tcgen05.ld.32x32b.xN loads N 32-bit words per datapath lane per instruction.
// The mlp1 UMMA epilogue maps one thread to one TMEM datapath row and reads an
// EpiChunkN-wide column strip per row, so the atom's per-thread register count
// (the "Nx" repeat, == RegNumDst) must equal EpiChunkN. Pick it at compile time
// from EpiChunkN so both the fused (EpiChunkN=64) and act (EpiChunkN=32) tiles —
// and any other divisor of WgTileN — get a matching load op.
template <int EpiChunkN> struct TmemLoadOpSelector;
template <> struct TmemLoadOpSelector<8>   { using Op = SM100_TMEM_LOAD_32dp32b8x;   };
template <> struct TmemLoadOpSelector<16>  { using Op = SM100_TMEM_LOAD_32dp32b16x;  };
template <> struct TmemLoadOpSelector<32>  { using Op = SM100_TMEM_LOAD_32dp32b32x;  };
template <> struct TmemLoadOpSelector<64>  { using Op = SM100_TMEM_LOAD_32dp32b64x;  };
template <> struct TmemLoadOpSelector<128> { using Op = SM100_TMEM_LOAD_32dp32b128x; };
template <int EpiChunkN>
using TmemLoadOp = typename TmemLoadOpSelector<EpiChunkN>::Op;

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

	// SM100 (Compute=100) pipelines — single CTA, no cluster. The mainloop
	// pipeline is the UMMA-aware TMA pipeline (its consumer_release issues the
	// UMMA-gated smem-buffer arrival); the accumulator pipeline hands the TMEM
	// accumulator from the MMA warp to the epilogue warps. See
	// Mlp1FusedConsumerImpl<100>. The Blackwell launcher must build the
	// mainloop pipe with num_consumers=1 (one umma_arrive per stage).
	using MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<
		Stages, cute::Shape<cute::_1, cute::_1, cute::_1>,
		cute::Shape<cute::_1, cute::_1, cute::_1>>;
	// TODO(perf): double-buffer the TMEM accumulator (AccStages=2) to overlap
	// epilogue(n) with MMA(n+1). Requires: 4·TileN ≤ 512 TMEM columns (so
	// TileN ≤ 128), a per-stage TMEM column offset on tCtAccU/V, and making the
	// MMA warp epilogue-free (epilogue on a separate warpgroup). Estimated win
	// is modest and shape-dependent — likely ~0-3% at production (HBM-bound)
	// shapes, more only when compute-bound / shallow-K. Profile the per-N-tile
	// MMA bubble before investing. See Mlp1FusedConsumerImpl<100> mainloop.
	static constexpr int AccStages = 1;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccStages>;

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

template <typename Traits, int Compute>
using Mlp1MainloopPipelineFor = cute::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

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

	// SM100 (Compute=100) only: landing slot for tcgen05.alloc's granted TMEM
	// base address, plus the accumulator pipeline's barrier storage (UMMA→
	// epilogue handoff). The Hopper (Compute=90) path never touches these, and
	// keeping them here leaves the consumer signature identical across Compute
	// values (TMEM/pipeline stay consumer-owned).
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

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

// SM100 (Compute=100) mainloop pipe. Mirrors mlp1_make_pipe but builds the
// UMMA-aware TMA pipeline (PipelineTmaUmmaAsync) whose consumer_release issues
// the UMMA-gated smem-buffer arrival. num_consumers=1: on Blackwell the buffer
// is released by a single umma_arrive per stage (the MMA warp), not per
// consumer thread. Params/PipelineState/SharedStorage alias the Hopper
// PipelineTmaAsync's, and the ctor arg pattern is identical, so the producer
// (mlp1_fused_producer) and the Compute=100 consumer drive it unchanged.
template <typename Traits>
__device__ __forceinline__ auto mlp1_make_pipe_umma(
		typename Traits::MainloopPipelineUmma::SharedStorage& storage) {

	using Pipeline = typename Traits::MainloopPipelineUmma;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

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
// Fused consumer — architecture-specialized on `int Compute`
// ═══════════════════════════════════════════════════════════════════
//
// Primary template is undefined; one full specialization per supported
// compute capability provides a `run(...)` member (a function-template on
// Traits/Pipeline/TmaStoreZ, since function templates can't be partially
// specialized). The free function `mlp1_fused_consumer` below forwards to
// the right specialization — call sites stay in the existing style.
//
//   Compute=90  → Hopper / WGMMA (cooperative 2-WG, dual register acc)
//   Compute=100 → Blackwell / UMMA (single-warp issue, TMEM accumulators)
//
// z_m is the M-tile index into the output Z buffer.

template <int Compute>
struct Mlp1FusedConsumerImpl;

// ───────────────────────────────────────────────────────────────────
// Compute=90 — Hopper. The two WGs split the (TileM, TileN) output along
// M: WG1 writes rows [0, AtomTileM) of tile z_m, WG2 writes [AtomTileM,
// TileM). (Verbatim from the original mlp1_fused_consumer.)
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp1FusedConsumerImpl<90> {
template <typename Traits, typename Pipeline, typename TmaStoreZ>
static __device__ __forceinline__ void run(
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
		int split_idx,
		int num_splits) {

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
};  // Mlp1FusedConsumerImpl<90>

// ───────────────────────────────────────────────────────────────────
// Compute=100 — Blackwell / UMMA.
//
// No cooperative-warpgroup machinery (UMMA has no warpgroup concept):
//   - One 1SM UMMA atom with M=TileM covers the whole tile; accumulators
//     U = X·B and V = X·C live in TMEM (no register/M-split).
//   - The first warp of WG1 (warp 4) issues both UMMAs and owns the
//     consumer-side TMEM allocation; all consumer threads drive the
//     shared TMA pipeline and help in the epilogue.
//   - Epilogue: the two consumer warpgroups split TileN; each reads its
//     N-half from TMEM→regs in EpiChunkN chunks, computes SiLU while the
//     previous TMA store drains, then stores 64-row (AtomTileM) tiles via
//     the existing reg→SMEM→TMA path (so the host TMA atom is unchanged).
//
// TileM is the template parameter as on Hopper (64 or 128 — both native
// 1SM UMMA M sizes). For TileM=128 the epilogue stores MSub=2 row-tiles.
// ───────────────────────────────────────────────────────────────────

template <>
struct Mlp1FusedConsumerImpl<100> {
template <typename Traits, typename Pipeline, typename TmaStoreZ>
static __device__ __forceinline__ void run(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedSmem<Traits>& smem,
		TmaStoreZ const& tma_store_z,
		int z_m,
		int intermediate_dim,
		int num_z_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx,
		int num_splits) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	constexpr int TileM     = Traits::TileM;
	constexpr int TileN     = Traits::TileN;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int AtomTileM = Traits::AtomTileM;     // 64-row TMA store tile
	static_assert(TileN % 2 == 0,
		"Blackwell consumer splits TileN across the two consumer warpgroups");
	constexpr int WgN         = TileN / 2;           // per-warpgroup N width
	static_assert(WgN % EpiChunkN == 0, "EpiChunkN must divide TileN/2");
	constexpr int NChunksHalf = WgN / EpiChunkN;     // n-chunks per warpgroup
	constexpr int MSub        = TileM / AtomTileM;   // 1 (TileM=64) or 2 (TileM=128)

	// ── Thread identity ─────────────────────────────────────
	// Consumer threads are warps 4..11 → tid_in_mma 0..255.
	const int  tid_in_mma   = threadIdx.x - Traits::WarpGroupSize;   // 0..255
	const int  wg           = tid_in_mma / Traits::WarpGroupSize;    // 0 or 1
	const int  tid_wg       = tid_in_mma % Traits::WarpGroupSize;    // 0..127
	const bool is_mma_warp  = (threadIdx.x / Traits::WarpSize) == 4; // first warp of WG1
	const bool is_wg_leader = (tid_wg == 0);
	const int  wg_barrier_id = 1 + wg;                               // 1 or 2

	// ── TiledMMA: single 1SM UMMA atom, M=TileM, N=TileN, K-major SS ──
	auto tiled_mma = make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, float, TileM, TileN,
		                     UMMA::Major::K, UMMA::Major::K>{});
	auto cta_mma = tiled_mma.get_slice(0);   // 1SM → single CTA, peer-coord 0

	auto sX  = make_tensor(make_smem_ptr(smem.X_data()),  typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	// ── Two TMEM accumulators: U = X·B, V = X·C ──
	auto cAccFull = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC     = cta_mma.partition_C(cAccFull);
	auto tCtAccU  = cta_mma.make_fragment_C(tCgC);
	auto tCtAccV  = cta_mma.make_fragment_C(tCgC);

	// ── TMEM allocation (consumer-owned): warp 4 allocs 2·TileN columns
	//    (U at base, V at base+TileN). ──
	cute::TMEM::Allocator1Sm tmem_alloc{};
	// tcgen05.alloc is warp-synchronous (.sync.aligned) and MUST be issued by a
	// single fully-active warp — NOT one elected thread (that deadlocks the
	// whole warp). See CUTLASS sm100 kernels: whole mma warp allocates, then
	// __syncwarp().
	if (is_mma_warp) {
		tmem_alloc.allocate(2 * TileN, &smem.tmem_base);
		__syncwarp();
	}

	// ── Accumulator pipeline: UMMA producer (warp 4) → epilogue consumers
	//    (all consumer warps). 1 stage — the U/V TMEM accumulators are reused
	//    each n-tile. producer_commit issues the UMMA-gated "accumulator ready"
	//    arrival; one consumer thread releases when the epilogue is done
	//    reading TMEM, gating the next tile's MMA. (Role asserts are compiled
	//    out under this build's NDEBUG, so warp 4 acting as both producer and
	//    an epilogue consumer is fine.) ──
	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp ? AccPipe::ThreadCategory::Producer
	                              : AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;          // one umma_arrive per commit
	acc_params.consumer_arv_count = 1;          // one elected releaser
	acc_params.initializing_warp  = 4;          // warp 4 inits the acc barriers
	AccPipe acc_pipe(smem.acc_pipe, acc_params,
		cute::Shape<cute::_1, cute::_1, cute::_1>{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
	const uint32_t tmem_base = smem.tmem_base;
	tCtAccU.data() = tmem_base;
	tCtAccV.data() = tmem_base + uint32_t(TileN);

	// ── Per-WG store slot (AtomTileM × EpiChunkN) — same smem as Hopper ──
	constexpr int store_slot_elems = AtomTileM * EpiChunkN;
	Element* my_store_ptr = smem.store_buf + wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	auto mZ = tma_store_z.get_tma_tensor(
		make_shape(static_cast<int64_t>(num_z_m_tiles) * TileM,
		           static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_z = tma_store_z.get_slice(Int<0>{});

	// ── Epilogue TMEM→reg copy plumbing (built once; sliced per WG-local
	//    tid). Each epi tile is the full TileM rows × EpiChunkN cols; the
	//    128-thread tcgen05.ld maps one thread per TMEM datapath row.
	//    flat_divide (not zipped_divide) keeps the epi tile as flat (M,N)
	//    modes, which is what make_tmem_copy's cotiled builder requires. ──
	auto epi_tile  = make_tile(Int<TileM>{}, Int<EpiChunkN>{});
	// Extract the flat (M,N) view from the UMMA C-fragment (MMA,MMA_M,MMA_N)
	// before tiling — matches CUTLASS's accumulators(make_coord(_,_),_0,_0).
	auto accU_mn   = tCtAccU(make_coord(_, _), _0{}, _0{});   // (TileM,TileN)
	auto accV_mn   = tCtAccV(make_coord(_, _), _0{}, _0{});
	auto tAccU_epi = flat_divide(accU_mn, epi_tile);   // (TileM,EpiChunkN,1,TileN/EpiChunkN)
	auto tAccV_epi = flat_divide(accV_mn, epi_tile);
	auto t2r       = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAccU_epi(_, _, _0{}, _0{}));
	auto thr_t2r   = t2r.get_slice(tid_wg);
	auto tTR_tAccU = thr_t2r.partition_S(tAccU_epi);     // (Cpy,Cpy_M,Cpy_N,1,nTiles)
	auto tTR_tAccV = thr_t2r.partition_S(tAccV_epi);
	// Within-chunk (m,n) coords for the reg → smem-slot scatter.
	auto cChunk    = make_identity_tensor(make_shape(Int<TileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);       // (Cpy,Cpy_M,Cpy_N)
	// Register fragments sized from the DEST (partition_D) shape — per-thread
	// (T2R,T2R_M,T2R_N). Sizing from partition_S(tmem) would wrongly include the
	// warp-collective datapath-lane dim (the TMEM_LOAD atom distributes those
	// lanes into per-thread registers internally). Matches CUTLASS's
	// make_tensor<ElementAccumulator>(shape(tTR_sD)).
	auto tTR_rU = make_tensor<float>(shape(tTR_cChunk));   // f32 regs (U, reused)
	auto tTR_rV = make_tensor<float>(shape(tTR_cChunk));   // f32 regs (V)

	int n_start  = (split_idx  >= 0) ? split_idx  : (int)blockIdx.y;
	int n_stride = (num_splits >= 0) ? num_splits : (int)gridDim.y;
	bool store_in_flight = false;

	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		// ── Mainloop (warp 4 only): bracket the k-loop with the accumulator
		//    pipeline. Each k-stage waits/releases the TMA→UMMA mainloop
		//    pipeline; consumer_release issues the UMMA-gated smem-buffer
		//    arrival (no hand-rolled mbarrier). (Conservative: no MMA/k overlap.)
		if (is_mma_warp) {
			// TODO(perf): with AccStages=1 this serializes MMA(n) against
			// epilogue(n-1) — the MMA warp stalls here through the prior
			// epilogue. AccStages=2 would overlap them (see Mlp1Traits::AccStages).
			acc_pipe.producer_acquire(acc_prod_state);   // TMEM acc free (prev epilogue done)
			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.consumer_wait(state);
				auto tCsX  = cta_mma.partition_A(sX (_, _, state.index()));
				auto tCsW1 = cta_mma.partition_B(sW1(_, _, state.index()));
				auto tCsW2 = cta_mma.partition_B(sW2(_, _, state.index()));
				auto tCrX  = cta_mma.make_fragment_A(tCsX);
				auto tCrW1 = cta_mma.make_fragment_B(tCsW1);
				auto tCrW2 = cta_mma.make_fragment_B(tCsW2);
				CUTE_UNROLL
				for (int kb = 0; kb < size<2>(tCrX); ++kb) {
					// First k-block of this n-tile clears TMEM; the rest accumulate.
					tiled_mma.accumulate_ = (k == 0 && kb == 0)
						? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
					gemm(tiled_mma, tCrX(_, _, kb), tCrW1(_, _, kb), tCtAccU);
					gemm(tiled_mma, tCrX(_, _, kb), tCrW2(_, _, kb), tCtAccV);
				}
				pipe.consumer_release(state);   // UMMA-gated smem-buffer release
				++state;
			}
			acc_pipe.producer_commit(acc_prod_state);    // signal U/V ready (umma_arrive)
			++acc_prod_state;
		}

		// ── Epilogue: wait for the accumulator, then this WG processes its
		//    n-chunks [wg·NChunksHalf, +NChunksHalf). ──
		acc_pipe.consumer_wait(acc_cons_state);

		CUTE_UNROLL
		for (int r = 0; r < NChunksHalf; ++r) {
			int chunk = wg * NChunksHalf + r;            // absolute n-chunk index

			// TMEM → registers (this chunk, full TileM rows).
			copy(t2r, tTR_tAccU(_, _, _, _0{}, chunk), tTR_rU);
			copy(t2r, tTR_tAccV(_, _, _, _0{}, chunk), tTR_rV);

			// SiLU(U)·V fused in place (overwrites the U regs) so it overlaps
			// the prior in-flight store. The cast to Element happens at the
			// smem write below — no extra register set.
			CUTE_UNROLL
			for (int i = 0; i < size(tTR_rU); ++i)
				tTR_rU(i) = fast_silu(tTR_rU(i)) * tTR_rV(i);

			// Store as MSub × (AtomTileM=64)-row TMA tiles (1 for TileM=64, 2 for 128).
			CUTE_UNROLL
			for (int ms = 0; ms < MSub; ++ms) {
				if (store_in_flight)
					cute::tma_store_wait<0>();

				cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);
				CUTE_UNROLL
				for (int i = 0; i < size(tTR_rU); ++i) {
					int m_row = get<0>(tTR_cChunk(i));   // 0..TileM
					int n_col = get<1>(tTR_cChunk(i));   // 0..EpiChunkN
					if (m_row >= ms * AtomTileM && m_row < (ms + 1) * AtomTileM)
						sStore(m_row - ms * AtomTileM, n_col) =
							static_cast<Element>(tTR_rU(i));
				}
				cutlass::arch::NamedBarrier::sync(Traits::WarpGroupSize, wg_barrier_id);

				if (is_wg_leader) {
					cute::tma_store_fence();
					int m_tile_idx = MSub * z_m + ms;
					int n_tile_idx = n * (TileN / EpiChunkN) + chunk;
					auto gZ = local_tile(mZ,
						make_tile(Int<AtomTileM>{}, Int<EpiChunkN>{}),
						make_coord(m_tile_idx, n_tile_idx));
					copy(tma_store_z, cta_tma_z.partition_S(sStore),
						cta_tma_z.partition_D(gZ));
					cute::tma_store_arrive();
				}
				store_in_flight = true;
			}
		}

		// All epilogue TMEM reads are done; release the accumulator so the next
		// n-tile's MMA may reuse it (one elected consumer thread arrives).
		cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
		if (tid_in_mma == Traits::WarpGroupSize)
			acc_pipe.consumer_release(acc_cons_state);
		++acc_cons_state;
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();

	// Release the alloc permit (so the next CTA can rasterize) then free TMEM.
	// tcgen05.relinquish_alloc_permit / tcgen05.dealloc are warp-synchronous —
	// issue from the whole mma warp, not one elected thread.
	cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
	if (is_mma_warp) {
		tmem_alloc.release_allocation_lock();
		tmem_alloc.free(tmem_base, 2 * TileN);
	}
#else
	// The Compute=100 specialization is never dispatched on non-SM100 targets
	// (host picks Compute=90 there). Trap if it is ever reached.
	__trap();
#endif
}
};  // Mlp1FusedConsumerImpl<100>

// ───────────────────────────────────────────────────────────────────
// Forwarder — keeps the existing call style. `Compute` defaults to 90 so
// current call sites (mlp1_fused_consumer<Traits>(...)) are unchanged;
// pass mlp1_fused_consumer<Traits, 100>(...) to select the Blackwell path.
// ───────────────────────────────────────────────────────────────────

template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStoreZ>
__device__ __forceinline__ void mlp1_fused_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedSmem<Traits>& smem,
		TmaStoreZ const& tma_store_z,
		int z_m,
		int intermediate_dim,
		int num_z_m_tiles,
		int num_n_tiles,
		int num_k_tiles,
		int split_idx = -1,
		int num_splits = -1) {
	Mlp1FusedConsumerImpl<Compute>::template run<Traits>(
		pipe, state, smem, tma_store_z, z_m, intermediate_dim,
		num_z_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
}

} // namespace liger
