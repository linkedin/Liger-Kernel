#pragma once

// Standalone SM100 tensor-parallel FSLCE dX GEMM.
//
// Computes BF16 [M,K] @ BF16 [K,N] with FP32 accumulation using one
// cluster-(2,1,1) 2SM UMMA per logical M256xN256 tile.  The output is not
// reduced or converted: each peer CTA writes its M128xN256 FP32 half into a
// compact tile-major staging buffer.  That layout is directly consumable by a
// later tile-local TP reduction without changing the GEMM epilogue.

#include "tmem_load_op_sm100.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

enum class DxRasterOrderSm100 : std::uint8_t {
	kNFast,
	kMFast,
};

template <int Stages_ = 4, int EpiChunkN_ = 32>
struct DxGemmTraitsSm100 {
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = 256;
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kStages = Stages_;
	static constexpr int kEpiChunkN = EpiChunkN_;
	static constexpr int kClusterM = 2;
	static constexpr int kAccumulatorStages = 2;
	static constexpr int kTmemColumns =
		kAccumulatorStages * kTileN;

	static constexpr int kWarpSize = 32;
	static constexpr int kNumThreads = 384;
	static constexpr int kTmaWarp = 0;
	static constexpr int kMmaWarp = 3;
	static constexpr int kFirstEpilogueWarp = 4;
	static constexpr int kLastEpilogueWarp = 11;
	static constexpr int kEpilogueWarps =
		kLastEpilogueWarp - kFirstEpilogueWarp + 1;
	static constexpr int kEpilogueThreads =
		kEpilogueWarps * kWarpSize;
	static constexpr int kWarpgroupSize = 4 * kWarpSize;
	static constexpr int kEpilogueWarpgroups =
		kEpilogueThreads / kWarpgroupSize;
	static constexpr int kWarpgroupTileN =
		kTileN / kEpilogueWarpgroups;
	static constexpr int kChunksPerWarpgroup =
		kWarpgroupTileN / kEpiChunkN;
	static constexpr int kMmaEpilogueThreads =
		(kLastEpilogueWarp - kMmaWarp + 1) * kWarpSize;

	static constexpr int kWarpgroup0BarrierId = 1;
	static constexpr int kWarpgroup1BarrierId = 2;
	static constexpr int kMmaEpilogueBarrierId = 3;
	static constexpr int kEpilogueBarrierId = 4;

	static_assert(kStages == 4, "the dX target fixes a four-stage TMA pipe");
	static_assert(kEpiChunkN == 32,
		"the compact FP32 epilogue uses N32 slots with four stages");
	static_assert(kTmemColumns == 512);
	static_assert(kChunksPerWarpgroup == 4);
	static_assert(kEpilogueWarpgroups == 2);

	using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
	using ClusterShape = Shape<Int<kClusterM>, _1, _1>;
	using AtomThrShape = Shape<Int<kClusterM>, _1, _1>;

	using TiledMma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			kTileM,
			kTileN,
			UMMA::Major::K,
			UMMA::Major::MN>{}));
	static_assert(
		size(typename TiledMma::AtomThrID{}) == kClusterM,
		"the dX MMA atom must span exactly the 2x1 CTA cluster");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma::AtomThrID{})));
	using MmaShapeA = decltype(partition_shape_A(
		TiledMma{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
	using MmaShapeB = decltype(partition_shape_B(
		TiledMma{}, make_shape(Int<kTileN>{}, Int<kTileK>{})));

	using SmemAtomA = UMMA::Layout_K_SW128_Atom<Element>;
	using SmemAtomB = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
		SmemAtomA{},
		append(MmaShapeA{}, Int<kStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
		SmemAtomB{},
		append(MmaShapeB{}, Int<kStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutA1 =
		decltype(SmemLayoutA{}(_, _, _, Int<0>{}));
	using SmemLayoutB1 =
		decltype(SmemLayoutB{}(_, _, _, Int<0>{}));

	static constexpr int kTmaTransBytesA =
		static_cast<int>(cosize_v<SmemLayoutA1> * sizeof(Element));
	static constexpr int kTmaTransBytesB =
		static_cast<int>(cosize_v<SmemLayoutB1> * sizeof(Element));
	static constexpr int kTmaTransBytes =
		kClusterM * (kTmaTransBytesA + kTmaTransBytesB);

	// Two N32 slots, one per epilogue warpgroup.  Together they occupy 32 KiB.
	using SmemStoreAtom =
		UMMA::Layout_K_SW128_Atom<ElementAccum>;
	using SmemLayoutStore = decltype(tile_to_shape(
		SmemStoreAtom{},
		Shape<Int<kCtaTileM>, Int<kEpiChunkN>>{}));
	static constexpr int kStoreElements =
		cosize_v<SmemLayoutStore>;

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		kAccumulatorStages,
		AtomThrShape>;
};

template <class Traits>
struct DxGemmSmemSm100 {
	using Element = typename Traits::Element;

	alignas(1024) Element operand_a[cosize_v<typename Traits::SmemLayoutA>];
	alignas(1024) Element operand_b[cosize_v<typename Traits::SmemLayoutB>];
	alignas(1024) float store[
		Traits::kEpilogueWarpgroups * Traits::kStoreElements];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	alignas(16) std::uint32_t tmem_base;

	CUTE_DEVICE Element* a_data() { return &operand_a[0]; }
	CUTE_DEVICE Element* b_data() { return &operand_b[0]; }
	CUTE_DEVICE float* store_data(int warpgroup) {
		return &store[warpgroup * Traits::kStoreElements];
	}
};

struct DxTileCoordSm100 {
	int m_tile;
	int n_tile;
};

template <DxRasterOrderSm100 Raster>
__host__ __device__ constexpr DxTileCoordSm100 dx_tile_coord_sm100(
		int item, int num_m_tiles, int num_n_tiles) {
	DxTileCoordSm100 coord;
	if constexpr (Raster == DxRasterOrderSm100::kNFast) {
		coord.m_tile = item / num_n_tiles;
		coord.n_tile = item - coord.m_tile * num_n_tiles;
	} else {
		coord.n_tile = item / num_m_tiles;
		coord.m_tile = item - coord.n_tile * num_m_tiles;
	}
	return coord;
}

template <class Traits>
CUTE_DEVICE typename Traits::MainloopPipeline dx_make_pipe_sm100(
		typename Traits::MainloopPipeline::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = static_cast<int>(threadIdx.x) / Traits::kWarpSize;
	typename Pipeline::Params params;
	params.transaction_bytes = Traits::kTmaTransBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = Traits::kTmaWarp;
	if (warp_id == Traits::kTmaWarp) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == 0 && cute::block_rank_in_cluster() == 0;
	} else if (
			warp_id >= Traits::kMmaWarp &&
			warp_id <= Traits::kLastEpilogueWarp) {
		params.role = Category::Consumer;
	} else {
		params.role = Category::NonParticipant;
	}
	return Pipeline(
		storage,
		params,
		typename Traits::ClusterShape{},
		cute::true_type{},
		cute::true_type{});
}

template <
	DxRasterOrderSm100 Raster,
	class Traits,
	class TmaLoadA,
	class TmaLoadB,
	class TmaStore>
__global__ __launch_bounds__(Traits::kNumThreads, 1) __cluster_dims__(2, 1, 1)
void dx_gemm_kernel_sm100(
		__grid_constant__ const TmaLoadA tma_a,
		__grid_constant__ const TmaLoadB tma_b,
		__grid_constant__ const TmaStore tma_store,
		int m,
		int n,
		int k,
		int num_m_tiles,
		int num_n_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	using AccPipe = typename Traits::AccumulatorPipeline;
	using Smem = DxGemmSmemSm100<Traits>;

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);

	int warp_id = static_cast<int>(threadIdx.x) / Traits::kWarpSize;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster_idx = static_cast<int>(blockIdx.x) / Traits::kClusterM;
	int num_clusters = static_cast<int>(gridDim.x) / Traits::kClusterM;
	int total_tiles = num_m_tiles * num_n_tiles;
	int num_k_tiles = (k + Traits::kTileK - 1) / Traits::kTileK;

	cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store.get_tma_descriptor());

	auto pipe = dx_make_pipe_sm100<Traits>(smem.pipeline);

	cute::TMEM::Allocator2Sm tmem_allocator;
	cute::cluster_sync();
	if (warp_id == Traits::kFirstEpilogueWarp) {
		tmem_allocator.allocate(Traits::kTmemColumns, &smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();
	std::uint32_t tmem_base = smem.tmem_base;

	typename Traits::PipelineState mainloop_state;
	if (warp_id == Traits::kTmaWarp) {
		mainloop_state =
			cutlass::make_producer_start_state<
				typename Traits::MainloopPipeline>();
	}

	bool is_mma_warp = warp_id == Traits::kMmaWarp;
	bool is_leader_cta = cluster_rank == 0;
	bool is_epilogue =
		warp_id >= Traits::kFirstEpilogueWarp &&
		warp_id <= Traits::kLastEpilogueWarp;

	typename Traits::TiledMma tiled_mma;
	auto cta_mma = tiled_mma.get_slice(cluster_rank);

	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp && is_leader_cta
		? AccPipe::ThreadCategory::Producer
		: (warp_id >= Traits::kMmaWarp
			? AccPipe::ThreadCategory::Consumer
			: AccPipe::ThreadCategory::NonParticipant);
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = Traits::kClusterM;
	acc_params.initializing_warp = Traits::kFirstEpilogueWarp;
	AccPipe acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});
	auto acc_prod_state =
		cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;
	if (warp_id >= Traits::kMmaWarp) {
		cutlass::arch::NamedBarrier::sync(
			Traits::kMmaEpilogueThreads,
			Traits::kMmaEpilogueBarrierId);
	}

	auto sA = make_tensor(
		make_smem_ptr(smem.a_data()), typename Traits::SmemLayoutA{});
	auto sB = make_tensor(
		make_smem_ptr(smem.b_data()), typename Traits::SmemLayoutB{});
	auto tCrA = cta_mma.make_fragment_A(sA);
	auto tCrB = cta_mma.make_fragment_B(sB);
	auto cAccFull = make_identity_tensor(
		make_shape(Int<Traits::kTileM>{}, Int<Traits::kTileN>{}));
	auto tCtAcc = cta_mma.make_fragment_C(
		cta_mma.partition_C(cAccFull));

	if (warp_id == Traits::kTmaWarp) {
		auto mA = tma_a.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(m),
			static_cast<std::int64_t>(k)));
		auto mB = tma_b.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(n),
			static_cast<std::int64_t>(k)));
		typename Traits::ClusterLayoutVMNK cluster_layout_vmnk;
		auto cluster_coord =
			cluster_layout_vmnk.get_flat_coord(cluster_rank);
		std::uint16_t mcast_mask_a =
			create_tma_multicast_mask<2>(
				cluster_layout_vmnk, cluster_coord);
		std::uint16_t mcast_mask_b =
			create_tma_multicast_mask<1>(
				cluster_layout_vmnk, cluster_coord);

		for (int item = cluster_idx;
				item < total_tiles;
				item += num_clusters) {
			DxTileCoordSm100 coord =
				dx_tile_coord_sm100<Raster>(
					item, num_m_tiles, num_n_tiles);
			auto tile_coord =
				make_coord(coord.m_tile, coord.n_tile, _);
			auto gA = local_tile(
				mA,
				typename Traits::TileShape{},
				tile_coord,
				Step<_1, X, _1>{});
			auto gB = local_tile(
				mB,
				typename Traits::TileShape{},
				tile_coord,
				Step<X, _1, _1>{});
			auto tCgA = cta_mma.partition_A(gA);
			auto tCgB = cta_mma.partition_B(gB);
			auto [tAgA, tAsA] = tma_partition(
				tma_a,
				get<2>(cluster_coord),
				make_layout(size<2>(cluster_layout_vmnk)),
				group_modes<0, 3>(sA),
				group_modes<0, 3>(tCgA));
			auto [tBgB, tBsB] = tma_partition(
				tma_b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout_vmnk)),
				group_modes<0, 3>(sB),
				group_modes<0, 3>(tCgB));

			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.producer_acquire(mainloop_state);
				if (cute::elect_one_sync()) {
					auto* barrier =
						pipe.producer_get_barrier(mainloop_state);
					copy(
						tma_a.with(*barrier, mcast_mask_a),
						tAgA(_, k_tile),
						tAsA(_, mainloop_state.index()));
					copy(
						tma_b.with(*barrier, mcast_mask_b),
						tBgB(_, k_tile),
						tBsB(_, mainloop_state.index()));
				}
				++mainloop_state;
			}
		}
		pipe.producer_tail(mainloop_state);
	}

	if (is_mma_warp && is_leader_cta) {
		for (int item = cluster_idx;
				item < total_tiles;
				item += num_clusters) {
			(void)item;
			acc_pipe.producer_acquire(acc_prod_state);
			tCtAcc.data() = tmem_base +
				static_cast<std::uint32_t>(
					acc_prod_state.index() * Traits::kTileN);
			bool first = true;
			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.consumer_wait(mainloop_state);
				CUTE_UNROLL
				for (int k_block = 0;
						k_block < size<2>(tCrA);
						++k_block) {
					tiled_mma.accumulate_ = first
						? UMMA::ScaleOut::Zero
						: UMMA::ScaleOut::One;
					first = false;
					gemm(
						tiled_mma,
						tCrA(
							_, _, k_block,
							mainloop_state.index()),
						tCrB(
							_, _, k_block,
							mainloop_state.index()),
						tCtAcc);
				}
				pipe.consumer_release(mainloop_state);
				++mainloop_state;
			}
			acc_pipe.producer_commit(acc_prod_state);
			++acc_prod_state;
		}
	} else if (is_epilogue) {
		int tid_in_epi =
			static_cast<int>(threadIdx.x) -
			Traits::kFirstEpilogueWarp * Traits::kWarpSize;
		int warpgroup = tid_in_epi / Traits::kWarpgroupSize;
		int tid_in_warpgroup = tid_in_epi % Traits::kWarpgroupSize;
		int warpgroup_barrier =
			Traits::kWarpgroup0BarrierId + warpgroup;

		tCtAcc.data() = tmem_base;
		auto epi_tile = make_tile(
			Int<Traits::kCtaTileM>{},
			Int<Traits::kEpiChunkN>{});
		auto acc_mn =
			tCtAcc(make_coord(_, _), _0{}, _0{});
		auto tAccEpi = flat_divide(acc_mn, epi_tile);
		auto t2r = make_tmem_copy(
			::liger::TmemLoadOp<Traits::kEpiChunkN>{},
			tAccEpi(_, _, _0{}, _0{}));
		auto thr_t2r = t2r.get_slice(tid_in_warpgroup);
		auto cChunk = make_identity_tensor(make_shape(
			Int<Traits::kCtaTileM>{},
			Int<Traits::kEpiChunkN>{}));
		auto tTR_cChunk = thr_t2r.partition_D(cChunk);
		auto tTR_rAcc =
			make_tensor<float>(shape(tTR_cChunk));
		Layout tmem_warp_layout =
			typename decltype(make_tmem_warp_partitioner(
				tAccEpi(_, _, _0{}, _0{})))::TiledLayout_TV{};
		constexpr bool kPredicateTmemLoad =
			size(tmem_warp_layout) != cosize(tmem_warp_layout);

		auto sStore = make_tensor(
			make_smem_ptr(smem.store_data(warpgroup)),
			typename Traits::SmemLayoutStore{});
		int staging_rows =
			total_tiles * Traits::kClusterM * Traits::kCtaTileM;
		auto mStore = tma_store.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(staging_rows),
			static_cast<std::int64_t>(Traits::kTileN)));
		auto cta_store = tma_store.get_slice(Int<0>{});
		auto tSsStore = cta_store.partition_S(sStore);
		auto tSgStore = cta_store.partition_D(local_tile(
			mStore,
			make_tile(
				Int<Traits::kCtaTileM>{},
				Int<Traits::kEpiChunkN>{}),
			make_coord(_, _)));

		for (int item = cluster_idx;
				item < total_tiles;
				item += num_clusters) {
			DxTileCoordSm100 coord =
				dx_tile_coord_sm100<Raster>(
					item, num_m_tiles, num_n_tiles);
			int tile_linear =
				coord.m_tile * num_n_tiles + coord.n_tile;
			int staging_m_tile =
				tile_linear * Traits::kClusterM + cluster_rank;

			acc_pipe.consumer_wait(acc_cons_state);
			tCtAcc.data() = tmem_base +
				static_cast<std::uint32_t>(
					acc_cons_state.index() * Traits::kTileN);
			auto acc_mn_stage =
				tCtAcc(make_coord(_, _), _0{}, _0{});
			auto tAccEpiStage =
				flat_divide(acc_mn_stage, epi_tile);
			auto tTR_tAcc =
				thr_t2r.partition_S(tAccEpiStage);

			CUTE_UNROLL
			for (int round = 0;
					round < Traits::kChunksPerWarpgroup;
					++round) {
				int chunk =
					warpgroup * Traits::kChunksPerWarpgroup +
					round;
				auto tAccChunk =
					tTR_tAcc(_, _, _, _0{}, chunk);
				bool issue_tmem_load = true;
				if constexpr (kPredicateTmemLoad) {
					int subpart =
						(tAccChunk.data().dp_ / 32) % 4;
					issue_tmem_load =
						tid_in_warpgroup / Traits::kWarpSize ==
						subpart;
				}
				if (issue_tmem_load) {
					copy(t2r, tAccChunk, tTR_rAcc);
					cutlass::arch::fence_view_async_tmem_load();
				}
				if (tid_in_warpgroup == 0) {
					cute::tma_store_wait<0>();
				}
				cutlass::arch::NamedBarrier::sync(
					Traits::kWarpgroupSize,
					warpgroup_barrier);
				if (issue_tmem_load) {
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rAcc); ++i) {
						sStore(
							get<0>(tTR_cChunk(i)),
							get<1>(tTR_cChunk(i))) =
							tTR_rAcc(i);
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Traits::kWarpgroupSize,
					warpgroup_barrier);
				if (tid_in_warpgroup == 0) {
					cute::tma_store_fence();
					copy(
						tma_store,
						tSsStore,
						tSgStore(
							_, _, _,
							staging_m_tile,
							chunk));
					cute::tma_store_arrive();
				}
			}
			if (tid_in_warpgroup == 0) {
				cute::tma_store_wait<0>();
			}
			cutlass::arch::NamedBarrier::sync(
				Traits::kEpilogueThreads,
				Traits::kEpilogueBarrierId);
			if (tid_in_epi == 0) {
				acc_pipe.consumer_release(acc_cons_state);
			}
			++acc_cons_state;
		}
	}

	if (warp_id >= Traits::kMmaWarp) {
		cutlass::arch::NamedBarrier::sync(
			Traits::kMmaEpilogueThreads,
			Traits::kMmaEpilogueBarrierId);
	}
	// The standalone kernel has symmetric control flow, so use the canonical
	// MoE 2SM teardown: both complete CTAs rendezvous before releasing the
	// shared Allocator2Sm reservation.  This avoids carrying fused-kernel
	// early-exit assumptions into repeated GEMM-only launches.
	__syncthreads();
	cute::cluster_sync();
	if (warp_id == Traits::kFirstEpilogueWarp) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(smem.tmem_base, Traits::kTmemColumns);
	}
#else
	__trap();
#endif
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
