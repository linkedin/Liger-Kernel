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
//   WG0 warps 1-2  : Idle / reserved for fused-kernel communication
//   WG0 warp 3     : Idle on SM90; dedicated UMMA producer on SM100
//   WG1 (warps 4-7)  : WGMMA consumer on SM90; epilogue WG0 on SM100
//   WG2 (warps 8-11) : WGMMA consumer on SM90; epilogue WG1 on SM100
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
#include "tmem_load_op.cuh"

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

#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>  // PipelineTmaUmmaAsync, PipelineUmmaAsync

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
	// Double-buffered SM100 TMEM accumulators: each stage owns U and V, each
	// TileN columns wide. This lets MMA(n+1) use the alternate stage while the
	// epilogue drains MMA(n)'s TMEM stage. The launcher allocates
	// AccStages·2·TileN columns; the dedicated epilogue warpgroup (warps 8..11)
	// drains the previous stage while the epilogue-free MMA warp runs ahead.
	static constexpr int AccStages = 2;
	static_assert(AccStages * (2 * TileN) <= 512,
		"SM100 MLP1 needs 2 accumulator stages x (U,V) to fit in 512 TMEM columns");
	static_assert(TileN <= 128,
		"SM100 double-buffered MLP1 accumulator requires TileN <= 128");
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
	// epilogue handoff). The Hopper (Compute=90) path never touches these;
	// SM100 launchers allocate/free TMEM outside the consumer and publish the
	// base here.
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
	bool is_consumer = (warp_id >= 3 && warp_id <= 11);  // warp 3 = UMMA consumer

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

template <typename Traits, bool Expert3D = false, typename Pipeline,
          typename TmaLoadX, typename TmaLoadW>
__device__ __forceinline__ void mlp1_fused_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp1FusedSmem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		int m,
		int expert_or_n_offset,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int num_experts,
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
	auto mB = [&]() {
		if constexpr (Expert3D) {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_b.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_n_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();
	auto mC = [&]() {
		if constexpr (Expert3D) {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(intermediate_dim),
				static_cast<int64_t>(hidden_dim),
				static_cast<int64_t>(num_experts)));
		} else {
			return tma_load_c.get_tma_tensor(make_shape(
				static_cast<int64_t>(total_n_rows),
				static_cast<int64_t>(hidden_dim)));
		}
	}();

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
		auto gB = [&]() {
			if constexpr (Expert3D) {
				return local_tile(mB,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _, expert_or_n_offset));
			} else {
				return local_tile(mB,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(expert_or_n_offset + n, _));
			}
		}();
		auto gC = [&]() {
			if constexpr (Expert3D) {
				return local_tile(mC,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(n, _, expert_or_n_offset));
			} else {
				return local_tile(mC,
					make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
					make_coord(expert_or_n_offset + n, _));
			}
		}();
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
//   - Warp 3 issues both UMMAs and stays epilogue-free so it can run ahead to
//     the alternate TMEM accumulator stage (AccStages=2).
//   - Epilogue: aligned WGs in warps 4..11 split TileN, read TMEM→regs,
//     compute SiLU while the previous TMA store drains, and store 64-row
//     (AtomTileM) tiles via the existing reg→SMEM→TMA path.
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
		"Blackwell consumer splits TileN across the two epilogue warpgroups");
	constexpr int WgN         = TileN / 2;           // per-WG N width
	static_assert(WgN % EpiChunkN == 0, "EpiChunkN must divide TileN/2");
	constexpr int NChunksHalf = WgN / EpiChunkN;     // n-chunks per epilogue WG
	constexpr int MSub        = TileM / AtomTileM;   // 1 (TileM=64) or 2 (TileM=128)

	// ── Thread identity (Blackwell warp-specialised layout) ──────────────
	// Warp 3 is the dedicated, epilogue-free UMMA producer (moved out of the
	// consumer warpgroups so it can run AccStages tiles ahead). Warps 4..11 are
	// the two aligned epilogue warpgroups (WG0 = warps 4..7, WG1 = warps 8..11);
	// each drains one half of TileN from the current accumulator stage while the
	// MMA warp fills the alternate TMEM stage. In the fused MoE CTA warp 0 = TMA
	// and warps 1..2 = NVSHMEM comm; warp 3 is the freed comm warp repurposed to
	// issue UMMA (see PR #19 + mlp.cuh). Total: 12 warps / 3 hardware warpgroups.
	const int  warp_id       = threadIdx.x / Traits::WarpSize;
	const bool is_mma_warp   = (warp_id == 3);                      // dedicated UMMA warp
	const bool is_epilogue   = (warp_id >= 4 && warp_id <= 11);     // two aligned WGs
	// Epilogue-local identity — valid only for warps 4..11. Warp 3 never enters
	// the epilogue block, so its (garbage) tid_in_epi/wg/tid_wg are unused.
	const int  tid_in_epi    = threadIdx.x - Traits::WarpGroupSize; // warps 4..11 -> 0..255
	const int  wg            = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;  // 0|1
	const int  tid_wg        = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;  // 0..127
	const int  tmem_copy_tid = tid_wg;
	const bool is_wg_leader  = is_epilogue && (tid_wg == 0);
	const int  wg_barrier_id = 1 + wg;                             // 1 (WG0) or 2 (WG1)
	// Warps 3..11 (MMA + both epilogue WGs) rendezvous for the acc-pipe init and
	// the launcher-safe TMEM free. ConsumerThreads (256) covers only warps 4..11.
	constexpr int kEpilogueThreads = Traits::ConsumerThreads;
	constexpr int kMmaEpiThreads = kEpilogueThreads + Traits::WarpSize;
	static_assert(Traits::WarpGroupSize == 4 * Traits::WarpSize);
	static_assert(kEpilogueThreads == 8 * Traits::WarpSize);
	static_assert(kMmaEpiThreads == 9 * Traits::WarpSize);

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

	// TMEM is allocated by the outer fused/standalone launcher once per CTA.

	// ── Accumulator pipeline: UMMA producer (warp 3) → epilogue consumers
	//    (warps 4..11, both WGs). AccStages TMEM stages let MMA(n+1) fill the
	//    alternate stage while both epilogue WGs drain MMA(n). producer_commit
	//    issues the UMMA-gated "accumulator ready" arrival; one elected epilogue
	//    thread releases the stage (after a 256-thread rendezvous) so the MMA
	//    warp may reuse it AccStages tiles later. ──
	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp ? AccPipe::ThreadCategory::Producer
	                              : AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;          // one umma_arrive per commit (warp 3)
	acc_params.consumer_arv_count = 1;          // one elected releaser per stage
	acc_params.initializing_warp  = 4;          // warp 4 inits the acc barriers
	AccPipe acc_pipe(smem.acc_pipe, acc_params,
		cute::Shape<cute::_1, cute::_1, cute::_1>{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	// Warps 3..11 rendezvous so the acc-pipe barriers (init'd by warp 4) are
	// visible before the MMA warp (3) and both epilogue WGs (4..11) use them.
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
	const uint32_t tmem_base = smem.tmem_base;
	tCtAccU.data() = tmem_base;
	tCtAccV.data() = tmem_base + uint32_t(TileN);

	// ── Per-WG store slot (AtomTileM × EpiChunkN) — WG0 uses store_buf[0..S),
	//    WG1 uses [S..2S). Same smem layout as Hopper. ──
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
	auto tAccU_epi = flat_divide(accU_mn, epi_tile);   // (TileM,EpiChunkN,1,TileN/EpiChunkN)
	auto t2r       = make_tmem_copy(TmemLoadOp<EpiChunkN>{}, tAccU_epi(_, _, _0{}, _0{}));
	auto thr_t2r   = t2r.get_slice(tmem_copy_tid);
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

		// ── Mainloop (MMA warp = warp 3 only): bracket the k-loop with the
		//    accumulator pipeline. producer_acquire selects the free TMEM stage
		//    (AccStages=2 → MMA(n+1) fills the alternate stage while both epilogue
		//    WGs drain MMA(n)). consumer_release issues the UMMA-gated smem-buffer
		//    arrival on the mainloop pipe (no hand-rolled mbarrier). ──
		if (is_mma_warp) {
			acc_pipe.producer_acquire(acc_prod_state);   // selected TMEM stage is free
			int acc_stage = acc_prod_state.index();
			uint32_t stage_base = tmem_base + uint32_t(acc_stage * (2 * TileN));
			tCtAccU.data() = stage_base;
			tCtAccV.data() = stage_base + uint32_t(TileN);
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

		// ── Epilogue (two aligned WGs, warps 4..11): wait for the accumulator
		//    stage, then THIS WG drains its half of TileN (chunks
		//    [wg·NChunksHalf, +NChunksHalf)) while the MMA warp fills the
		//    alternate stage. ──
		if (is_epilogue) {
			acc_pipe.consumer_wait(acc_cons_state);
			int acc_stage = acc_cons_state.index();
			uint32_t stage_base = tmem_base + uint32_t(acc_stage * (2 * TileN));
			tCtAccU.data() = stage_base;
			tCtAccV.data() = stage_base + uint32_t(TileN);
			auto accU_mn_stage   = tCtAccU(make_coord(_, _), _0{}, _0{});
			auto accV_mn_stage   = tCtAccV(make_coord(_, _), _0{}, _0{});
			auto tAccU_epi_stage = flat_divide(accU_mn_stage, epi_tile);
			auto tAccV_epi_stage = flat_divide(accV_mn_stage, epi_tile);
			auto tTR_tAccU       = thr_t2r.partition_S(tAccU_epi_stage);
			auto tTR_tAccV       = thr_t2r.partition_S(tAccV_epi_stage);

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

			// Both epilogue WGs done reading this TMEM stage → rendezvous, then
			// one elected thread releases the stage so the MMA warp may reuse it
			// when the AccStages accumulator pipeline wraps.
			cutlass::arch::NamedBarrier::sync(Traits::ConsumerThreads, /*id=*/0);
			if (tid_in_epi == 0)
				acc_pipe.consumer_release(acc_cons_state);
			++acc_cons_state;
		}
	}
	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();

	// Final rendezvous of the MMA warp (3) + both epilogue WGs (4..11) so the
	// outer launcher can safely free TMEM after this consumer returns.
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, /*id=*/3);
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

template <typename Traits, int Compute = 90,
          typename Pipeline, typename TmaStoreZ>
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
	Mlp1FusedConsumerImpl<Compute>::template run<Traits, Pipeline, TmaStoreZ>(
		pipe, state, smem, tma_store_z, z_m, intermediate_dim,
		num_z_m_tiles, num_n_tiles, num_k_tiles, split_idx, num_splits);
}

} // namespace liger
