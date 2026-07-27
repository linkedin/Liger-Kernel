#pragma once

// ═══════════════════════════════════════════════════════════════════
// MLP Phase 2 (transposed):  Y = Z @ A
// ═══════════════════════════════════════════════════════════════════
//
// Cooperative 2-WG consumer; split axis selected by TileM:
//   TileM=128 → Layout<_2,_1,_1>, split along M (each WG owns 64×TileN)
//   TileM=64  → Layout<_1,_2,_1>, split along N (each WG owns 64×TileN/2)
// One accumulator per WG covering (AtomTileM=64, AtomN). Each WG writes
// its strip independently — no cross-WG NamedBarrier sync in the epilogue.
//
// Difference from mlp2_fused.cuh: A is read through a COLUMN-MAJOR TMA view
// (the physical layout is the same per-expert [hidden_dim,
// intermediate_dim] row-major buffer, but the column-major view makes
// intermediate_dim the row axis so it lines up as the N axis of the
// GEMM). Output Y has shape [num_tokens, intermediate_dim].
//   - num_n_tiles  = intermediate_dim / TileN     (the N axis)
//   - num_k_tiles  = hidden_dim       / TileK     (the K axis)
//   - expert offset is along K: expert · num_k_tiles
//
// A single fused TMA pipeline carries Z + W per k-step; transaction =
// bytes(Z) + bytes(W). Producer issues two TMA copies against the same
// mbarrier per acquire.
//
// CUTLASS-canonical patterns:
//   - K_PIPE_MMAS = 1
//   - __launch_bounds__(NumThreads, 1) — applied by the caller kernel
//
// CTA layout (384 threads = 12 warps):
//   WG0 warp 0     : Producer (TMA loads for Z / A)
//   WG0 warps 1-3  : Idle on SM90; warp 3 issues UMMA on SM100
//   WG1 (warps 4-7)  : Consumer atom 0 — owns m=[0, AtomTileM)
//   WG2 (warps 8-11) : Consumer atom 1 — owns m=[AtomTileM, TileM)
//
// Per `expert_ids`: one int per TileM=128 m-block.
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
// specialization in mlp2_t_fused.cuh. These headers are header-only and safe
// to include under an sm_90a build — the tcgen05 PTX they emit is guarded by
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
struct Mlp2TTraits {
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

	// A is loaded through a column-major TMA view (stride=(1, I)), so the
	// W operand in shared memory must be MN-major to match. The wgmma atom
	// for operand B must accept an MN-major source (GmmaSelectorKMN — note
	// the trailing N in "KMN" denotes MN-major B). Z stays K-major (Z is
	// row-major in gmem with K=H contiguous-ish given the layout CUTE
	// produces for shape (M, K), stride (K, 1)), so its atom is the regular
	// K-major SW128 atom.
	//
	// AtomN is the per-WG WGMMA atom N — combined across both WGs it must
	// cover exactly TileN cols (= sW's N extent). With WgLayout=<1,2,1>
	// and AtomN=TileN, the TiledMMA would cover 2·TileN cols and WG1's
	// partition_B(sW) slice would read past sW's TileN-wide buffer → IMA.
	static constexpr int AtomN = kMSplit ? TileN : (TileN / 2);
	using GmmaAtom = typename GmmaSelectorKMN<Element, AtomN>::Atom;
	using WgLayout = cute::conditional_t<
		kMSplit,
		Layout<Shape<_2, _1, _1>>,
		Layout<Shape<_1, _2, _1>>>;
	using TiledMma = TiledMMA<MMA_Atom<GmmaAtom>, WgLayout>;

	using SmemLayoutAtomZ = GMMA::Layout_K_SW128_Atom<Element>;
	using SmemLayoutAtomW = GMMA::Layout_MN_SW128_Atom<Element>;

	using SmemLayoutZ_1 = decltype(tile_to_shape(SmemLayoutAtomZ{},
		Shape<Int<TileM>, Int<TileK>>{}));
	// Step<_2, _1>: tile_to_shape interprets the inner SmemLayoutAtomW as
	// MN-major (M is contiguous in smem), so passing the tile shape with
	// the swap step makes the resulting layout's M dim contiguous — which
	// is what the column-major A gmem feed expects.
	using SmemLayoutW_1 = decltype(tile_to_shape(SmemLayoutAtomW{},
		Shape<Int<TileN>, Int<TileK>>{},
		Step<_2, _1>{}));

	using SmemLayoutZ = decltype(tile_to_shape(SmemLayoutAtomZ{},
		Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
	using SmemLayoutW = decltype(tile_to_shape(SmemLayoutAtomW{},
		Shape<Int<TileN>, Int<TileK>, Int<Stages>>{},
		Step<_2, _1, _3>{}));

	// Per-WG store buffer slot: (AtomTileM, EpiChunkN).
	using SmemLayoutStoreSlot = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
	using SmemLayoutStore_1 = SmemLayoutStoreSlot;

	using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
	using PipelineState    = cutlass::PipelineState<Stages>;

	// SM100 (Compute=100) pipelines — single CTA, no cluster. The mainloop
	// pipeline is the UMMA-aware TMA pipeline (its consumer_release issues the
	// UMMA-gated smem-buffer arrival); the accumulator pipeline hands the single
	// TMEM accumulator from the MMA warp to the epilogue warps. See
	// Mlp2TFusedConsumerImpl<100> in mlp2_t_fused.cuh. The Blackwell launcher
	// must build the mainloop pipe with num_consumers=1 (one umma_arrive per
	// stage).
	using MainloopPipelineUmma = cutlass::PipelineTmaUmmaAsync<
		Stages, cute::Shape<cute::_1, cute::_1, cute::_1>,
		cute::Shape<cute::_1, cute::_1, cute::_1>>;
	// Double-buffered SM100 TMEM accumulators: each stage owns one TileN-wide
	// accumulator, allowing MMA(n+1) to fill the alternate stage while the
	// epilogue drains/TMA-stores MMA(n).
	static constexpr int AccStages = 2;
	static_assert(AccStages * TileN <= 512,
		"SM100 MLP2-T accumulator stages must fit in 512 TMEM columns");
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccStages>;

	// SM100 UMMA TiledMMA — one 1SM tcgen05 atom spanning the whole
	// (TileM, TileN) tile with the accumulator in TMEM. THE mlp2_t-specific
	// choice: operand A = Z is K-major (UMMA::Major::K), operand B = A (the
	// weight) is MN-major (UMMA::Major::MN) — mlp2_fused uses <K,K>. The B
	// operand descriptor is produced by make_umma_desc<Major::MN> reading
	// SmemLayoutW's (MN-contiguous, SW128) stride; SmemLayoutW stays built from
	// Layout_MN_SW128_Atom + Step<_2,_1>, which is exactly the canonical
	// Major-MN layout make_umma_desc<Major::MN> accepts (mma_traits_sm100.hpp).
	using TiledMmaUmma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_SS<Element, Element, ElementAccum,
		                     TileM, TileN,
		                     UMMA::Major::K, UMMA::Major::MN>{}));

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

template <typename Traits, int Compute>
using Mlp2TMainloopPipelineFor = cute::conditional_t<
	Compute == 100,
	typename Traits::MainloopPipelineUmma,
	typename Traits::MainloopPipeline>;

// ═══════════════════════════════════════════════════════════════════
// Shared Memory
// ═══════════════════════════════════════════════════════════════════

template <typename Traits>
struct Mlp2TSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_Z_size     = cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_W_size     = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size = cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element smem_W[smem_W_size];
	// Per-WG store buffer: WG1 uses store_buf[0..S-1], WG2 uses [S..2S-1].
	alignas(128) Element store_buf[2 * smem_store_size];

	typename Traits::MainloopPipeline::SharedStorage pipe_storage;

	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
	CUTE_DEVICE Element* W_data() { return &smem_W[0]; }
};

// ═══════════════════════════════════════════════════════════════════
// TMA Producer (Warp 0) — one acquire per k-step, issues Z + W
// against the same mbarrier.
//   - Z view: [num_tokens, hidden_dim] row-major (Z's K axis = H).
//   - A view: [intermediate_dim, total_k_cols] column-major, where
//     total_k_cols = num_experts · hidden_dim. Per-expert tile at
//     k = expert · num_k_tiles + k_local.
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename Pipeline,
          typename TmaLoadZ, typename TmaLoadW>
__device__ __forceinline__ void mlp2_t_producer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TSmem<Traits>& smem,
		TmaLoadZ const& tma_load_z,
		TmaLoadW const& tma_load_a,
		const int* expert_ids,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int total_k_cols,        // num_experts · hidden_dim
		int num_m_tiles,
		int num_n_tiles,         // intermediate_dim / TileN
		int num_k_tiles) {       // hidden_dim       / TileK

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	auto mZ = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	// Column-major view of A: shape (intermediate_dim, total_k_cols)
	// — same underlying buffer as the row-major [E·H, I] tensor in
	// mlp2_fused.cuh, but the box geometry makes intermediate_dim the row
	// axis so it lines up with the GEMM N dimension.
	auto mA = tma_load_a.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(total_k_cols)));

	auto cta_tma_z = tma_load_z.get_slice(Int<0>{});
	auto cta_tma_a = tma_load_a.get_slice(Int<0>{});

	auto tZsZ = cta_tma_z.partition_D(sZ);
	auto tWsW = cta_tma_a.partition_D(sW);

	int n_start  = blockIdx.y;
	int n_stride = gridDim.y;

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
		int expert = expert_ids[m];
		// In mlp2_t the expert is contiguous in the K direction
		// (= hidden_dim), so the offset is along k_tile, not n_tile.
		int expert_k_offset = expert * num_k_tiles;

		auto gZ = local_tile(mZ,
			make_tile(Int<Traits::TileM>{}, Int<Traits::TileK>{}),
			make_coord(m, _));
		auto tZgZ = cta_tma_z.partition_S(gZ);

		for (int n = n_start; n < num_n_tiles; n += n_stride) {
			// A tile: (TileN rows of I, TileK rows of K=H), positioned
			// at (n along I, expert_k_offset+k along total_k_cols).
			auto gW = local_tile(mA,
				make_tile(Int<Traits::TileN>{}, Int<Traits::TileK>{}),
				make_coord(n, _));
			auto tWgW = cta_tma_a.partition_S(gW);

			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.producer_acquire(state);
				if (threadIdx.x == 0) {
					auto* bar = pipe.producer_get_barrier(state);
					copy(tma_load_z.with(*bar, 0),
						tZgZ(_, _, _, k), tZsZ(_, _, _, state.index()));
					copy(tma_load_a.with(*bar, 0),
						tWgW(_, _, _, expert_k_offset + k),
						tWsW(_, _, _, state.index()));
				}
				++state;
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════
// Cooperative 2-WG Consumer — M-split, single acc per WG
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, int Compute = 90, typename Pipeline, typename TmaStoreY>
__device__ __forceinline__ void mlp2_t_consumer(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		Mlp2TSmem<Traits>& smem,
		TmaStoreY const& tma_store_y,
		int intermediate_dim,
		int num_m_tiles,
		int num_n_tiles,
		int num_k_tiles) {

	(void)Compute;
	using Element = typename Traits::Element;
	typename Traits::TiledMma tiled_mma;
	int tid_in_mma = threadIdx.x - Traits::WarpGroupSize;   // 0..255
	int tid_in_wg  = tid_in_mma % Traits::WarpGroupSize;    // 0..127
	auto thr_mma   = tiled_mma.get_slice(tid_in_mma);

	auto sZ = make_tensor(make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto sW = make_tensor(make_smem_ptr(smem.W_data()), typename Traits::SmemLayoutW{});

	auto tCsZ = thr_mma.partition_A(sZ);
	auto tCsW = thr_mma.partition_B(sW);

	auto acc = partition_fragment_C(tiled_mma,
		Shape<Int<Traits::TileM>, Int<Traits::TileN>>{});

	auto cC   = make_identity_tensor(make_shape(Int<Traits::TileM>{}, Int<Traits::TileN>{}));
	auto tCcC = thr_mma.partition_C(cC);

	const int my_wg = (threadIdx.x / Traits::WarpGroupSize) - 1;  // 0 or 1
	const int my_barrier_id = 1 + my_wg;                          // 1 or 2
	const bool is_my_wg_leader = (tid_in_wg == 0);

	constexpr int store_slot_elems = Traits::AtomTileM * Traits::EpiChunkN;
	Element* my_store_ptr = smem.store_buf + my_wg * store_slot_elems;
	auto sStore = make_tensor(make_smem_ptr(my_store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	// Y view: [num_tokens, intermediate_dim] row-major.
	auto mY = tma_store_y.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_m_tiles) * Traits::TileM,
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_y = tma_store_y.get_slice(Int<0>{});

	int n_start  = blockIdx.y;
	int n_stride = gridDim.y;
	bool store_in_flight = false;
	constexpr int K_PIPE_MMAS = 1;

	for (int m = blockIdx.x; m < num_m_tiles; m += gridDim.x) {
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
	}
	if (store_in_flight)
		cute::tma_store_wait<0>();
}

// ═══════════════════════════════════════════════════════════════════
// mlp2_t_fwd — device orchestrator
// ═══════════════════════════════════════════════════════════════════

template <typename Traits, typename TmaLoadZ, typename TmaLoadW, typename TmaStoreY>
__device__ __forceinline__ void mlp2_t_fwd(
		Mlp2TSmem<Traits>& smem,
		TmaLoadZ const& tma_load_z,
		TmaLoadW const& tma_load_a,
		TmaStoreY const& tma_store_y,
		const int* expert_ids,
		int num_tokens,
		int hidden_dim,
		int intermediate_dim,
		int total_k_cols,
		int num_m_tiles) {

	using Pipeline = typename Traits::MainloopPipeline;
	using PipeState = typename Traits::PipelineState;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int num_n_tiles = intermediate_dim / Traits::TileN;
	int num_k_tiles = hidden_dim       / Traits::TileK;

	bool is_producer = (warp_id == 0);
	bool is_consumer = (warp_id >= 4 && warp_id <= 11);

	cute::prefetch_tma_descriptor(tma_load_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_y.get_tma_descriptor());

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
	Pipeline pipe(smem.pipe_storage, pp, Shape<_1, _1, _1>{},
		cute::true_type{}, cute::true_type{});

	__syncthreads();

	PipeState state = is_producer ?
		cutlass::make_producer_start_state<Pipeline>() : PipeState{};
	PipeState cons_state;

	if (is_producer) {
		mlp2_t_producer<Traits>(
			pipe, state,
			smem, tma_load_z, tma_load_a,
			expert_ids,
			num_tokens, hidden_dim, intermediate_dim, total_k_cols,
			num_m_tiles, num_n_tiles, num_k_tiles);
	}

	if (is_consumer) {
		mlp2_t_consumer<Traits>(
			pipe, cons_state,
			smem, tma_store_y,
			intermediate_dim,
			num_m_tiles, num_n_tiles, num_k_tiles);
	}

	__syncthreads();
}

} // namespace liger
