#pragma once

#include "tmem_load_op_sm100.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

struct BackwardDxCustom2x1ResultSm100 {
	static constexpr int kClusterM = 2;
	static constexpr int kClusterN = 1;
	static constexpr int kUmmaTileM = 256;
	static constexpr int kUmmaTileN = 256;
	static constexpr int kLogicalTileN = 512;
	static constexpr int kTileK = 64;
	static constexpr int kStages = 4;
	static constexpr int kResidentClusters = 74;
	static constexpr int kResidentCtas = 148;
	static constexpr int kRegisters = 89;
	static constexpr int kDynamicSmemBytes = 229504;
	static constexpr float kMilliseconds = 1.13502f;
	static constexpr float kTflops = 1937.43f;
	static constexpr float kCublasTflops = 1963.36f;
	static constexpr float kCublasRatio = 0.986792f;
	static constexpr float kSmActivePercent = 86.26f;
	static constexpr float kTensorActivePercent = 83.82f;
	static constexpr float kTensorPerActiveSmPercent = 97.17f;
	static constexpr float kDramReadPercent = 19.38f;

	static constexpr float kNaiveN256Milliseconds = 1.26059f;
	static constexpr float kNaiveN256Tflops = 1744.43f;
	static constexpr float kNaiveN256TensorActivePercent = 75.48f;
	static constexpr float kNaiveN256DramReadPercent = 28.79f;
};

struct BackwardDxWideTraitsSm100 {
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = 256;
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileN = 512;
	static constexpr int kUmmaTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kStages = 4;
	static constexpr int kClusterM = 2;
	static constexpr int kThreads = 256;
	static constexpr int kEpilogueThreads = 128;
	static constexpr int kTmaWarp = 4;
	static constexpr int kMmaWarp = 5;
	static constexpr int kTmemColumns = 512;

	using TileShape = Shape<_256, _512, _64>;
	using ClusterShape = Shape<_2, _1, _1>;
	using AtomThrShape = Shape<_2, _1, _1>;
	using TiledMma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			kTileM,
			kUmmaTileN,
			UMMA::Major::K,
			UMMA::Major::MN>{}));
	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma::AtomThrID{})));
	using MmaShapeA = decltype(partition_shape_A(
		TiledMma{}, make_shape(_256{}, _64{})));
	using MmaShapeB = decltype(partition_shape_B(
		TiledMma{}, make_shape(_512{}, _64{})));
	using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
		UMMA::Layout_K_SW128_Atom<Element>{},
		append(MmaShapeA{}, _4{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
		UMMA::Layout_MN_SW128_Atom<Element>{},
		append(MmaShapeB{}, _4{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutA1 =
		decltype(SmemLayoutA{}(_, _, _, _0{}));
	using SmemLayoutB1 =
		decltype(SmemLayoutB{}(_, _, _, _0{}));

	static constexpr int kTmaTransBytesA =
		cosize_v<SmemLayoutA1> * sizeof(Element);
	static constexpr int kTmaTransBytesB =
		cosize_v<SmemLayoutB1> * sizeof(Element);
	static constexpr int kTmaTransBytes =
		kClusterM * (kTmaTransBytesA + kTmaTransBytesB);
	using SmemLayoutStore =
		Layout<Shape<_128, _64>, Stride<_64, _1>>;
	using SmemLayoutStoreTile =
		Layout<Shape<_16, _64>, Stride<_64, _1>>;
	static constexpr int kStoreElements =
		cosize_v<SmemLayoutStore>;

	static_assert(kTmaTransBytesA == 16 * 1024);
	static_assert(kTmaTransBytesB == 32 * 1024);
	static_assert(kTmaTransBytes == 96 * 1024);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<1, AtomThrShape>;

	static_assert(
		kStages == BackwardDxCustom2x1ResultSm100::kStages);
	static_assert(
		kTileN == BackwardDxCustom2x1ResultSm100::kLogicalTileN);
};

struct BackwardDxWideParamsSm100 {
	float* output;
	int m;
	int n;
	int k;
};

template <class TmaA, class TmaB, class TmaD>
struct BackwardDxWideTmaBundleSm100 {
	TmaA a;
	TmaB b;
	TmaD d;
};

struct BackwardDxWideSmemSm100 {
	using Traits = BackwardDxWideTraitsSm100;
	alignas(128) Traits::Element a[cosize_v<Traits::SmemLayoutA>];
	alignas(128) Traits::Element b[cosize_v<Traits::SmemLayoutB>];
	alignas(128) float store[Traits::kStoreElements];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	alignas(16) std::uint32_t tmem_base;
};

template <class TmaA, class TmaB, class TmaD>
__global__ __launch_bounds__(BackwardDxWideTraitsSm100::kThreads, 1)
__cluster_dims__(2, 1, 1) void backward_dx_wide_kernel_sm100(
		__grid_constant__ const BackwardDxWideTmaBundleSm100<
			TmaA, TmaB, TmaD> tma,
		__grid_constant__ const BackwardDxWideParamsSm100 params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Traits = BackwardDxWideTraitsSm100;
	using Element = Traits::Element;

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<BackwardDxWideSmemSm100*>(raw_smem);
	int warp_id = static_cast<int>(threadIdx.x) / 32;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster_idx = static_cast<int>(blockIdx.z);
	int num_clusters = static_cast<int>(gridDim.z);
	int m_pairs = (params.m + 255) / 256;
	int n_wide_tiles = (params.n + 511) / 512;
	int total_items = m_pairs * n_wide_tiles;

	cute::prefetch_tma_descriptor(tma.a.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma.b.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma.d.get_tma_descriptor());

	typename Traits::MainloopPipeline::Params pipe_params;
	pipe_params.transaction_bytes = Traits::kTmaTransBytes;
	pipe_params.num_producers = 1;
	pipe_params.num_consumers = 1;
	pipe_params.initializing_warp = Traits::kTmaWarp;
	if (warp_id == Traits::kTmaWarp) {
		pipe_params.role =
			Traits::MainloopPipeline::ThreadCategory::Producer;
		pipe_params.is_leader =
			(threadIdx.x == Traits::kTmaWarp * 32) &&
			cluster_rank == 0;
	} else if (warp_id == Traits::kMmaWarp || warp_id < 4) {
		pipe_params.role =
			Traits::MainloopPipeline::ThreadCategory::Consumer;
	} else {
		pipe_params.role =
			Traits::MainloopPipeline::ThreadCategory::NonParticipant;
	}
	typename Traits::MainloopPipeline pipe(
		smem.pipeline,
		pipe_params,
		typename Traits::ClusterShape{},
		cute::true_type{},
		cute::true_type{});

	bool is_mma_warp = warp_id == Traits::kMmaWarp;
	bool is_mma_leader = is_mma_warp && cluster_rank == 0;
	typename Traits::AccumulatorPipeline::Params acc_params;
	acc_params.role = is_mma_leader
		? Traits::AccumulatorPipeline::ThreadCategory::Producer
		: (warp_id < 4
			? Traits::AccumulatorPipeline::ThreadCategory::Consumer
			: Traits::AccumulatorPipeline::ThreadCategory::NonParticipant);
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = Traits::kClusterM;
	acc_params.initializing_warp = 0;
	typename Traits::AccumulatorPipeline acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});

	cutlass::arch::fence_barrier_init();
	cute::cluster_sync();
	cute::TMEM::Allocator2Sm tmem_allocator;
	if (warp_id == 0) {
		tmem_allocator.allocate(
			Traits::kTmemColumns, &smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();

	typename Traits::PipelineState mainloop_state;
	if (warp_id == Traits::kTmaWarp) {
		mainloop_state =
			cutlass::make_producer_start_state<
				typename Traits::MainloopPipeline>();
		auto sA = make_tensor(
			make_smem_ptr(&smem.a[0]), typename Traits::SmemLayoutA{});
		auto sB = make_tensor(
			make_smem_ptr(&smem.b[0]), typename Traits::SmemLayoutB{});
		auto mA = tma.a.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.m),
			static_cast<std::int64_t>(params.k)));
		auto mB = tma.b.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.n),
			static_cast<std::int64_t>(params.k)));
		typename Traits::TiledMma tiled_mma;
		auto cta_mma = tiled_mma.get_slice(cluster_rank);
		typename Traits::ClusterLayoutVMNK cluster_layout;
		auto cluster_coord = cluster_layout.get_flat_coord(cluster_rank);
		std::uint16_t mask_a = create_tma_multicast_mask<2>(
			cluster_layout, cluster_coord);
		std::uint16_t mask_b = create_tma_multicast_mask<1>(
			cluster_layout, cluster_coord);
		int k_tiles = (params.k + 63) / 64;
		for (int item = cluster_idx; item < total_items;
				item += num_clusters) {
			int m_pair = item % m_pairs;
			int n_wide = item / m_pairs;
			auto coord = make_coord(m_pair, n_wide, _);
			auto gA = local_tile(
				mA,
				typename Traits::TileShape{},
				coord,
				Step<_1, X, _1>{});
			auto gB = local_tile(
				mB,
				typename Traits::TileShape{},
				coord,
				Step<X, _1, _1>{});
			auto tCgA = cta_mma.partition_A(gA);
			auto tCgB = cta_mma.partition_B(gB);
			auto [tAgA, tAsA] = tma_partition(
				tma.a,
				get<2>(cluster_coord),
				make_layout(size<2>(cluster_layout)),
				group_modes<0, 3>(sA),
				group_modes<0, 3>(tCgA));
			auto [tBgB, tBsB] = tma_partition(
				tma.b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout)),
				group_modes<0, 3>(sB),
				group_modes<0, 3>(tCgB));
			for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
				pipe.producer_acquire(mainloop_state);
				if (cute::elect_one_sync()) {
					auto* barrier =
						pipe.producer_get_barrier(mainloop_state);
					copy(
						tma.a.with(*barrier, mask_a),
						tAgA(_, k_tile),
						tAsA(_, mainloop_state.index()));
					copy(
						tma.b.with(*barrier, mask_b),
						tBgB(_, k_tile),
						tBsB(_, mainloop_state.index()));
				}
				++mainloop_state;
			}
		}
		pipe.producer_tail(mainloop_state);
	} else if (is_mma_leader) {
		typename Traits::TiledMma tiled_mma;
		auto cta_mma = tiled_mma.get_slice(cluster_rank);
		auto sA = make_tensor(
			make_smem_ptr(&smem.a[0]), typename Traits::SmemLayoutA{});
		auto sB = make_tensor(
			make_smem_ptr(&smem.b[0]), typename Traits::SmemLayoutB{});
		auto tCrA = cta_mma.make_fragment_A(sA);
		auto tCrB = cta_mma.make_fragment_B(sB);
		auto cAcc = make_identity_tensor(
			make_shape(_256{}, _512{}));
		auto tCtAcc =
			cta_mma.make_fragment_C(cta_mma.partition_C(cAcc));
		auto acc_state =
			cutlass::make_producer_start_state<
				typename Traits::AccumulatorPipeline>();
		int k_tiles = (params.k + 63) / 64;
		for (int item = cluster_idx; item < total_items;
				item += num_clusters) {
			acc_pipe.producer_acquire(acc_state);
			tCtAcc.data() = smem.tmem_base;
			bool first = true;
			for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
				pipe.consumer_wait(mainloop_state);
				CUTE_UNROLL
				for (int k_block = 0; k_block < size<2>(tCrA);
						++k_block) {
					tiled_mma.accumulate_ = first
						? UMMA::ScaleOut::Zero
						: UMMA::ScaleOut::One;
					first = false;
					gemm(
						tiled_mma,
						tCrA(_, _, k_block, mainloop_state.index()),
						tCrB(_, _, k_block, mainloop_state.index()),
						tCtAcc);
				}
				pipe.consumer_release(mainloop_state);
				++mainloop_state;
			}
			acc_pipe.producer_commit(acc_state);
			++acc_state;
		}
	} else if (warp_id < 4) {
		typename Traits::TiledMma tiled_mma;
		auto cta_mma = tiled_mma.get_slice(cluster_rank);
		auto cAcc = make_identity_tensor(
			make_shape(_256{}, _512{}));
		auto tCtAcc =
			cta_mma.make_fragment_C(cta_mma.partition_C(cAcc));
		tCtAcc.data() = smem.tmem_base;
		auto acc_state =
			typename Traits::AccumulatorPipeline::PipelineState{};

		auto epi_tile = make_tile(_128{}, _64{});
		auto acc_mn = tCtAcc(make_coord(_, _), _0{}, _0{});
		auto tAccEpi = flat_divide(acc_mn, epi_tile);
		auto t2r = make_tmem_copy(
			::liger::TmemLoadOp<64>{},
			tAccEpi(_, _, _0{}, _0{}));
		int tid = static_cast<int>(threadIdx.x);
		auto thr_t2r = t2r.get_slice(tid);
		auto cSlab = make_identity_tensor(
			make_shape(_128{}, _64{}));
		auto tTR_cSlab = thr_t2r.partition_D(cSlab);
		auto tTR_rAcc = make_tensor<float>(shape(tTR_cSlab));
		auto tTR_tAcc = thr_t2r.partition_S(tAccEpi);
		Layout tmem_warp_layout =
			typename decltype(make_tmem_warp_partitioner(
				tAccEpi(_, _, _0{}, _0{})))::TiledLayout_TV{};
		constexpr bool kPredicateTmemLoad =
			size(tmem_warp_layout) != cosize(tmem_warp_layout);
		auto mD = tma.d.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.m),
			static_cast<std::int64_t>(params.n)));
		auto cta_store = tma.d.get_slice(_0{});
		auto tSgD = cta_store.partition_D(local_tile(
			mD, make_tile(_16{}, _64{}), make_coord(_, _)));
		auto sD = make_tensor(
			make_smem_ptr(&smem.store[0]),
			typename Traits::SmemLayoutStore{});
		for (int item = cluster_idx; item < total_items;
				item += num_clusters) {
			int m_pair = item % m_pairs;
			int n_wide = item / m_pairs;
			acc_pipe.consumer_wait(acc_state);
			int m_tile =
				m_pair * Traits::kClusterM + cluster_rank;
			for (int chunk = 0; chunk < 8; ++chunk) {
				if (chunk > 0) {
					if (tid == 0) cute::tma_store_wait<0>();
					cutlass::arch::NamedBarrier::sync(128, 1);
				}
				auto tAccSlab =
					tTR_tAcc(_, _, _, _0{}, chunk);
				bool issue = true;
				if constexpr (kPredicateTmemLoad) {
					int subpart =
						(tAccSlab.data().dp_ / 32) % 4;
					issue = tid / 32 == subpart;
				}
				if (issue) {
					copy(t2r, tAccSlab, tTR_rAcc);
					cutlass::arch::fence_view_async_tmem_load();
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rAcc); ++i) {
						sD(
							get<0>(tTR_cSlab(i)),
							get<1>(tTR_cSlab(i))) =
							tTR_rAcc(i);
					}
				}
				cutlass::arch::NamedBarrier::sync(128, 1);
				if (tid == 0) {
					cute::tma_store_fence();
					CUTE_UNROLL
					for (int row_group = 0; row_group < 8;
							++row_group) {
						auto sTile = make_tensor(
							make_smem_ptr(
								&smem.store[
									row_group * 16 * 64]),
							typename Traits::
								SmemLayoutStoreTile{});
						auto tSsD =
							cta_store.partition_S(sTile);
						copy(
							tma.d,
							tSsD,
							tSgD(
								_, _, _,
								m_tile * 8 + row_group,
								n_wide * 8 + chunk));
						cute::tma_store_arrive();
					}
				}
			}
			if (tid == 0) cute::tma_store_wait<0>();
			cutlass::arch::NamedBarrier::sync(128, 1);
			if (tid == 0) {
				acc_pipe.consumer_release(acc_state);
			}
			++acc_state;
		}
	}

	__syncthreads();
	cute::cluster_sync();
	if (warp_id == 0) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(
			smem.tmem_base, Traits::kTmemColumns);
	}
#else
	__trap();
#endif
}

template <class TmaA, class TmaB, class TmaD>
inline cudaError_t launch_backward_dx_wide_sm100(
		const BackwardDxWideTmaBundleSm100<TmaA, TmaB, TmaD>& tma,
		const BackwardDxWideParamsSm100& params,
		cudaStream_t stream = nullptr) {
	using Traits = BackwardDxWideTraitsSm100;
	auto kernel = &backward_dx_wide_kernel_sm100<TmaA, TmaB, TmaD>;
	int smem_bytes = static_cast<int>(sizeof(BackwardDxWideSmemSm100));
	cudaError_t error = cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		smem_bytes);
	if (error != cudaSuccess) return error;
	error = cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
	if (error != cudaSuccess) return error;
	error = cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributePreferredSharedMemoryCarveout,
		cudaSharedmemCarveoutMaxShared);
	if (error != cudaSuccess) return error;
	error = cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeClusterSchedulingPolicyPreference,
		cudaClusterSchedulingPolicySpread);
	if (error != cudaSuccess) return error;

	int m_pairs = cute::ceil_div(params.m, Traits::kTileM);
	int n_tiles = cute::ceil_div(params.n, Traits::kTileN);
	int total_clusters = m_pairs * n_tiles;
	static int resident_clusters = 0;
	if (resident_clusters == 0) {
		int device = 0;
		cudaDeviceProp properties{};
		error = cudaGetDevice(&device);
		if (error != cudaSuccess) return error;
		error = cudaGetDeviceProperties(&properties, device);
		if (error != cudaSuccess) return error;
		resident_clusters = properties.multiProcessorCount / 2;
	}
	int launch_clusters =
		total_clusters < resident_clusters
		? total_clusters
		: resident_clusters;
	dim3 grid(2, 1, static_cast<unsigned>(launch_clusters));
	cudaLaunchAttribute cluster_attribute{};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = 2;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	cudaLaunchConfig_t config{};
	config.gridDim = grid;
	config.blockDim = dim3(Traits::kThreads, 1, 1);
	config.dynamicSmemBytes = smem_bytes;
	config.stream = stream;
	config.attrs = &cluster_attribute;
	config.numAttrs = 1;
	return cudaLaunchKernelEx(&config, kernel, tma, params);
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
