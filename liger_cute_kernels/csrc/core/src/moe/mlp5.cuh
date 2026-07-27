#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 5:  dX = dU @ B + dV @ C  (input gradient, backward)
// NESTED-LOOP scheduler — the default.
// ═══════════════════════════════════════════════════════════════════
//
// Cooperative 2-WG consumer; split axis selected by TileM:
//   TileM=128 → Layout<_2,_1,_1>, split along M (each WG owns 64×TileN)
//   TileM=64  → Layout<_1,_2,_1>, split along N (each WG owns 64×TileN/2)
// ONE accumulator per WG covering (AtomTileM=64, AtomN). Each WG writes
// its strip independently — no cross-WG NamedBarrier sync in the epilogue.
//
// Fused mainloop runs a single continuous (2 × num_k_tiles) k-loop:
//   Phase 1 (k=0..num_k_tiles-1):    acc += dU @ B
//   Phase 2 (k=num_k_tiles..2K-1):   acc += dV @ C
//
// Scheduler:
//   - 2D grid (num_sms / NSplit, NSplit) — host-side launcher MUST
//     match (see mlp5.cu).
//   - nested `for (m = blockIdx.x; ...) for (n = blockIdx.y; ...)`
//   - amortizes expert lookup + dU/dV tile-coord setup once per m
//   - cross-CTA dU/dV multicast in L2 between (m, 0) and (m, 1) in the
//     same m-column
//
// Alternative flat work_idx scheduler is preserved in `mlp5_flat.cuh`
// (which includes this header for the shared building blocks).
//
// CUTLASS-canonical patterns:
//   - K_PIPE_MMAS = 1
//   - __launch_bounds__(NumThreads, 1) — applied by the caller kernel
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for dU/dV/B/C)
//   WG0 warps 1-2  : Idle / reserved for fused-kernel communication
//   WG0 warp 3     : Idle on SM90; dedicated UMMA producer on SM100
//   WG1 (warps 4-7)  : WGMMA consumer on SM90; epilogue WG0 on SM100
//   WG2 (warps 8-11) : WGMMA consumer on SM90; epilogue WG1 on SM100
//
// Per `expert_for_m_block`: one int per TileM=128 m-block.
//
// Smem at default config (TileM=128, TileN=256, TileK=64, Stages=4,
// EpiChunkN=64):
//   Z (TileM × TileK × S × 2)              =  64 KiB
//   W (TileN × TileK × S × 2)              = 128 KiB
//   store_buf (2 WGs × AtomTileM × EpiChunkN × 2) = 16 KiB
//   pipe + barriers                         :  ~1 KiB
//   ─────────────────────────────────────────────
//   Total                                   : ~209 KiB  (≤ 228 KiB cap)
// ═══════════════════════════════════════════════════════════════════

#include "models.cuh"
#include "tmem_load_op.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/barrier.h>

// SM100 (Blackwell / UMMA) support for the Compute=100 consumer
// specialization (mlp5_fused.cuh). These headers are header-only and safe to
// include under an sm_90a build — the tcgen05 PTX they emit is guarded by
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
	int TileN_      = 256,
	int TileK_      = 64,
	int Stages_     = 4,
	int EpiChunkN_  = 64
>
struct Mlp5Traits {
	using Element      = Element_;
	using ElementAccum = float;

	static constexpr int TileM      = TileM_;
	static constexpr int TileN      = TileN_;
	static constexpr int TileK      = TileK_;
	static constexpr int Stages     = Stages_;
	static constexpr int EpiChunkN  = EpiChunkN_;

	static_assert(TileM == 64 || TileM == 128, "TileM must be 64 or 128");

	// Hopper WGMMA atom has m=64. With TileM=128 the two WGs cooperate
	// along M (each owns 64 rows × TileN). With TileM=64 the WGs cooperate
	// along N (each owns 64 rows × TileN/2). AtomTileM is the per-WG M
	// extent — always 64. Mirrors mlp2_fused.cuh's kMSplit/N-split pattern.
	static constexpr bool kMSplit   = (TileM == 128);
	static constexpr int  AtomTileM = 64;
	static constexpr int  WgTileN   = kMSplit ? TileN : (TileN / 2);

	static_assert(TileN <= 256,             "TileN must fit TMA box limit");
	static_assert(WgTileN % EpiChunkN == 0, "EpiChunkN must divide WgTileN");
	static_assert(kMSplit || (TileN % 2 == 0),
		"TileN must be even when TileM=64 (cooperative N-split halves)");
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;

	// Cooperative 2-WG layout: M-split for TileM=128, N-split for TileM=64.
	// AtomN is the per-WG WGMMA atom N — combined across both WGs it must
	// cover exactly TileN cols (= sW's N extent).
	static constexpr int AtomN = kMSplit ? TileN : (TileN / 2);
	using GmmaAtom = typename GmmaSelectorKMN<Element, AtomN>::Atom;
	using WgLayout = cute::conditional_t<
		kMSplit,
		Layout<Shape<_2, _1, _1>>,
		Layout<Shape<_1, _2, _1>>>;
	using TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>;

	using SmemLayoutAtomZ = GMMA::Layout_K_SW128_Atom<Element>;
	using SmemLayoutAtomW = GMMA::Layout_MN_SW128_Atom<Element>;

	// One-stage layouts (used for TMA descriptors).
	using SmemLayoutZ_1 = decltype(tile_to_shape(SmemLayoutAtomZ{},
		Shape<Int<TileM>, Int<TileK>>{}));
	using SmemLayoutW_1 = decltype(tile_to_shape(SmemLayoutAtomW{},
		Shape<Int<TileN>, Int<TileK>>{}, Step<_2, _1>{}));

	// Multi-stage smem layouts.
	using SmemLayoutZ = decltype(tile_to_shape(SmemLayoutAtomZ{},
		Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
	using SmemLayoutW = decltype(tile_to_shape(SmemLayoutAtomW{},
		Shape<Int<TileN>, Int<TileK>, Int<Stages>>{}, Step<_2, _1, _3>{}));

	// Per-WG store buffer: (AtomTileM, EpiChunkN). One slot per WG.
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	// Alias for callers that build TMA descriptors using the single-slot
	// 2D layout (e.g., moe_bwd's mlp_bwd.cu/moe_bwd.cu).
	using SmemLayoutStore_1 = SmemLayoutStore;

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	// SM100 (Compute=100) pipelines — single CTA, no cluster. The mainloop
	// pipeline is the UMMA-aware TMA pipeline (its consumer_release issues the
	// UMMA-gated smem-buffer arrival); the accumulator pipeline hands the single
	// TMEM accumulator from the MMA warp to the epilogue warps. See
	// Mlp5FusedConsumerImpl<100> in mlp5_fused.cuh. The Blackwell launcher must
	// build the mainloop pipe with num_consumers=1 (one umma_arrive per stage).
	using MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<
		Stages, cute::Shape<cute::_1, cute::_1, cute::_1>,
		cute::Shape<cute::_1, cute::_1, cute::_1>>;
	// Double-buffered TMEM accumulators: the single accumulator per stage still
	// carries the cross-phase sum dU·B + dV·C across the whole 2·num_k_tiles
	// mainloop, while the alternate stage lets the next n-tile's MMA overlap
	// the previous n-tile's epilogue.
	static constexpr int AccStages = 2;
	static_assert(AccStages * TileN <= 512,
		"SM100 MLP5 accumulator stages must fit in 512 TMEM columns");
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccStages>;

	static constexpr int TmaTransBytesZ =
		static_cast<int>(size(SmemLayoutZ_1{}) * sizeof(Element));
	static constexpr int TmaTransBytesW =
		static_cast<int>(size(SmemLayoutW_1{}) * sizeof(Element));
	static constexpr int TmaTransBytes = TmaTransBytesZ + TmaTransBytesW;

	static constexpr int WarpSize        = 32;
	static constexpr int WarpGroupSize   = 128;
	static constexpr int NumConsumers    = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads      = 384;
};

template <typename Traits, int Compute>
using Mlp5MainloopPipelineFor = cute::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

// ═══════════════════════════════════════════════════════════════════
// Shared memory layout
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp5Smem {
	using Element = typename Traits::Element;

	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStore>;

	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element smem_W[smem_W_size];
	// Per-WG store buffer: WG0 uses store_buf[0..S-1], WG1 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	typename Traits::MainloopPipeline::SharedStorage pipe_storage;

	// SM100 (Compute=100) only: landing slot for tcgen05.alloc's granted TMEM
	// base address, plus the accumulator pipeline's barrier storage (UMMA→
	// epilogue handoff). The Hopper (Compute=90) path never touches these.
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
};

// ═══════════════════════════════════════════════════════════════════
// Pipeline construction
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
__device__ __forceinline__ auto mlp5_make_pipe(
		typename Traits::MainloopPipeline::SharedStorage& storage) {

	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	typename Pipeline::Params pp;
	pp.transaction_bytes = Traits::TmaTransBytes;
	pp.num_producers = 1;
	// Both consumer WGs (cooperative) consume each pipe slot.
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

// SM100 (Compute=100) mainloop pipe. Mirrors mlp5_make_pipe but builds the
// UMMA-aware TMA pipeline (PipelineTmaUmmaAsync) whose consumer_release issues
// the UMMA-gated smem-buffer arrival. num_consumers=1: on Blackwell the buffer
// is released by a single umma_arrive per stage (the MMA warp), not per
// consumer thread. Params/PipelineState/SharedStorage alias the Hopper
// PipelineTmaAsync's, and the ctor arg pattern is identical, so the producer
// (mlp5_fused_producer) and the Compute=100 consumer drive it unchanged.
template <typename Traits>
__device__ __forceinline__ auto mlp5_make_pipe_umma(
		typename Traits::MainloopPipelineUmma::SharedStorage& storage) {

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

// ── Producer (nested) ───────────────────────────────────────────────

template <typename Traits, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp5_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaLoadZ const& tma_load_du,
		TmaLoadZ const& tma_load_dv,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		const int* expert_for_m_block,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int total_k_cols,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles) {

	auto sZ = make_tensor(make_smem_ptr(smem.smem_Z), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.smem_W), typename Traits::SmemLayoutW{});

	auto mDU = tma_load_du.get_tma_tensor(make_shape(num_tokens, intermediate_dim));
	auto mDV = tma_load_dv.get_tma_tensor(make_shape(num_tokens, intermediate_dim));
	auto mB  = tma_load_b.get_tma_tensor(make_shape(hidden_dim, total_k_cols));
	auto mC  = tma_load_c.get_tma_tensor(make_shape(hidden_dim, total_k_cols));

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

	bool is_leader = (threadIdx.x == 0);

	int n_start  = (int)blockIdx.y;
	int n_stride = (int)gridDim.y;

	for (int m = (int)blockIdx.x; m < num_m_tiles; m += (int)gridDim.x) {
		int expert = expert_for_m_block[m];
		int expert_k_offset = expert * num_k_tiles;

		auto gDU = local_tile(mDU,
			make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
			make_coord(m, _));
		auto tZgDU = cta_tma_du.partition_S(gDU);
		auto gDV = local_tile(mDV,
			make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
			make_coord(m, _));
		auto tZgDV = cta_tma_du.partition_S(gDV);

		for (int n = n_start; n < num_n_tiles; n += n_stride) {
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
}

// ── Consumer (nested) ───────────────────────────────────────────────

template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStoreDX>
__device__ __forceinline__ void mlp5_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp5Smem<Traits>& smem,
		TmaStoreDX const& tma_store_dx,
		int hidden_dim,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles) {

	(void)Compute;
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

	auto mDX = tma_store_dx.get_tma_tensor(
		make_shape(num_m_tiles * Traits::TileM, hidden_dim));
	auto cta_tma_dx = tma_store_dx.get_slice(Int<0>{});

	int total_k     = 2 * num_k_tiles;
	constexpr int K_PIPE_MMAS = 1;
	bool store_in_flight = false;

	int n_start  = (int)blockIdx.y;
	int n_stride = (int)gridDim.y;

	for (int m = (int)blockIdx.x; m < num_m_tiles; m += (int)gridDim.x) {
	for (int n = n_start; n < num_n_tiles; n += n_stride) {

		clear(acc);
		auto state_release = state;
		int prologue_count = (total_k < K_PIPE_MMAS) ? total_k : K_PIPE_MMAS;

		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_wait(state);
			warpgroup_fence_operand(acc);
			warpgroup_arrive();
			gemm(tiled_mma, tCsZ(_, _, _, state.index()),
				tCsW(_, _, _, state.index()), acc);
			warpgroup_commit_batch();
			++state;
		}
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
		warpgroup_wait<0>();
		warpgroup_fence_operand(acc);
		for (int k = 0; k < prologue_count; ++k) {
			pipe.consumer_release(state_release);
			++state_release;
		}

		// TileM=128 (M-split): WG_w owns rows [w*64, (w+1)*64), full N.
		// TileM=64  (N-split): WG_w owns full M (=64), cols [w*WgTileN, ...).
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
					m_tile_idx = 2 * m + my_wg;
					n_tile_idx = n * Traits::NumEpiRounds + r;
				} else {
					// TileM=64: 1 row tile per logical M tile; WG_w handles
					// its N-half.
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
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

// ── Driver ──────────────────────────────────────────────────────────

template <typename Traits, typename TmaLoadZ, typename TmaLoadW, typename TmaStoreDX>
__device__ __forceinline__ void mlp5_fwd(
		Mlp5Smem<Traits>& smem,
		TmaLoadZ const& tma_load_du,
		TmaLoadZ const& tma_load_dv,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		TmaStoreDX const& tma_store_dx,
		const int* expert_for_m_block,
		int num_tokens,
		int intermediate_dim,
		int hidden_dim,
		int total_k_cols,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles) {

	using Pipeline = typename Traits::MainloopPipeline;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	auto pipe = mlp5_make_pipe<Traits>(smem.pipe_storage);

	__syncthreads();

	PipeState state = is_producer ?
		cutlass::make_producer_start_state<Pipeline>() : PipeState{};
	PipeState cons_state;

	if (is_producer) {
		mlp5_producer<Traits>(pipe, state, smem,
			tma_load_du, tma_load_dv, tma_load_b, tma_load_c,
			expert_for_m_block,
			num_tokens, hidden_dim, intermediate_dim, total_k_cols,
			num_m_tiles, num_n_tiles, num_k_tiles);
	}
	if (is_consumer) {
		mlp5_consumer<Traits>(pipe, cons_state, smem,
			tma_store_dx, hidden_dim,
			num_m_tiles, num_n_tiles, num_k_tiles);
	}

	__syncthreads();
}

// ═══════════════════════════════════════════════════════════════════
// Host launcher (raw-pointer; defined in mlp5.cu)
// ═══════════════════════════════════════════════════════════════════
//
// dX = dU @ B + dV @ C. Output dX is fully written; no zero-init needed.
//
//   dU                : [num_tokens, intermediate_dim]
//   dV                : [num_tokens, intermediate_dim]
//   B, C              : [num_experts, intermediate_dim, hidden_dim]
//   dX                : [num_tokens, hidden_dim] (output)
//   expert_for_m_block: [num_tokens / TileM] int32 (TileM is Traits::TileM)
void mlp5_fwd_bf16_launch(
		const cutlass::bfloat16_t* dU,
		const cutlass::bfloat16_t* dV,
		const cutlass::bfloat16_t* B,
		const cutlass::bfloat16_t* C,
		cutlass::bfloat16_t* dX,
		const int* expert_for_m_block,
		int num_tokens,
		int intermediate_dim,
		int hidden_dim,
		int num_experts,
		int device,
		cudaStream_t stream);

} // namespace liger
