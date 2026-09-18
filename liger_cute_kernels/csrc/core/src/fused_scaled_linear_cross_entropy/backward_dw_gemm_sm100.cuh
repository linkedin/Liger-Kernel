#pragma once

// Standalone SM100 TP-FSLCE dW GEMM.
//
// Computes dW = dZ^T @ X without communication, dZ, dX, or a device-side
// wave loop.  A launch handles exactly one K extent:
//   * K=4096 with a plain TMA store for the single-pass path.
//   * K=1024 with a plain TMA store for wave 0 or TMA reduce-add afterwards.
//
// The schedule matches the B300 cuBLAS winner: a 2x1 cluster, 2SM UMMA,
// M256xN256xK64 joined tile (M128xN256 per CTA), six TMA stages, vertical
// M raster within pairs of adjacent N tiles, FP32 TMEM accumulation, BF16
// packing through a swizzled two-stage shared-memory ring, and 8 KiB TMA
// stores.

#include "tmem_load_op_sm100.cuh"

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

struct DwGemmConfigSm100 {
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = 256;
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kClusterM = 2;
	static constexpr int kMainloopStages = 6;
	static constexpr int kAccumulatorStages = 2;
	static constexpr int kEpilogueN = 32;
	static constexpr int kOutputStages = 2;

	static constexpr int kFirstEpilogueWarp = 4;
	static constexpr int kEpilogueWarps = 4;
	static constexpr int kEpilogueWarpgroups = 1;
	static constexpr int kWarpgroupSize = 128;
	static constexpr int kMmaWarp = 2;
	static constexpr int kTmaWarp = 1;
	static constexpr int kNumThreads = 256;
	static constexpr int kWarpSize = 32;
	static constexpr int kEpilogueThreads =
		kEpilogueWarps * kWarpSize;
	static constexpr int kMmaEpilogueThreads =
		(kEpilogueWarps + 1) * kWarpSize;

	static constexpr int kWarpgroupBarrierBase = 1;
	static constexpr int kEpilogueBarrierId = 2;
	static constexpr int kMmaEpilogueBarrierId = 3;

	static_assert(kCtaTileM * kClusterM == kTileM);
	static_assert(kEpilogueN * sizeof(Element) * kCtaTileM == 8 * 1024);
	static_assert(kAccumulatorStages * kTileN == 512);
};

struct DwPairCoordSm100 {
	int m_pair;
	int n_tile_begin;
};

// A work item owns two adjacent N256 tiles for one M256 cluster tile. Work
// items advance through M first (vertical raster), then to the next N pair.
CUTE_HOST_DEVICE DwPairCoordSm100 dw_pair_coord_sm100(
		int item, int m_pairs) {
	return {item % m_pairs, 2 * (item / m_pairs)};
}

template <class TmaA, class TmaB, class TmaOutput>
struct DwTmaBundleSm100 {
	using TmaAType = TmaA;
	using TmaBType = TmaB;
	using TmaOutputType = TmaOutput;

	TmaA a;
	TmaB b;
	TmaOutput output;
};

template <class TmaA, class TmaB, class TmaOutput>
struct DwGemmTraitsSm100 {
	using Config = DwGemmConfigSm100;
	using Element = typename Config::Element;
	using ElementAccum = typename Config::ElementAccum;

	using TileShape = Shape<
		Int<Config::kTileM>,
		Int<Config::kTileN>,
		Int<Config::kTileK>>;
	using ClusterShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using AtomThrShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using TiledMma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			Config::kTileM,
			Config::kTileN,
			UMMA::Major::MN,
			UMMA::Major::MN>{}));

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma::AtomThrID{})));
	using MmaShapeA = decltype(partition_shape_A(
		TiledMma{},
		make_shape(Int<Config::kTileM>{}, Int<Config::kTileK>{})));
	using MmaShapeB = decltype(partition_shape_B(
		TiledMma{},
		make_shape(Int<Config::kTileN>{}, Int<Config::kTileK>{})));

	using OperandAtom = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
		OperandAtom{},
		append(MmaShapeA{}, Int<Config::kMainloopStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
		OperandAtom{},
		append(MmaShapeB{}, Int<Config::kMainloopStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutA1 =
		decltype(SmemLayoutA{}(_, _, _, Int<0>{}));
	using SmemLayoutB1 =
		decltype(SmemLayoutB{}(_, _, _, Int<0>{}));

	using OutputAtom = UMMA::Layout_K_SW64_Atom<Element>;
	using SmemLayoutOutput1 = decltype(tile_to_shape(
		OutputAtom{},
		Shape<Int<Config::kCtaTileM>, Int<Config::kEpilogueN>>{},
		Step<_2, _1>{}));

	static constexpr int kTmaTransactionBytesA =
		cosize_v<SmemLayoutA1> * sizeof(Element);
	static constexpr int kTmaTransactionBytesB =
		cosize_v<SmemLayoutB1> * sizeof(Element);
	static constexpr int kTmaTransactionBytes =
		Config::kClusterM *
		(kTmaTransactionBytesA + kTmaTransactionBytesB);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		Config::kMainloopStages,
		ClusterShape,
		AtomThrShape>;
	using MainloopState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		Config::kAccumulatorStages,
		AtomThrShape>;

	using TmaBundle = DwTmaBundleSm100<TmaA, TmaB, TmaOutput>;
};

template <class Traits>
struct DwGemmSharedStorageSm100 {
	using Config = typename Traits::Config;
	using Element = typename Traits::Element;

	alignas(1024) Element operand_a[cosize_v<typename Traits::SmemLayoutA>];
	alignas(1024) Element operand_b[cosize_v<typename Traits::SmemLayoutB>];
	alignas(1024) Element output[
		Config::kEpilogueWarpgroups *
		Config::kOutputStages *
		cosize_v<typename Traits::SmemLayoutOutput1>];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage mainloop;
	alignas(16)
		typename Traits::AccumulatorPipeline::SharedStorage accumulator;
	alignas(16) std::uint32_t tmem_base;
};

template <class Traits>
CUTE_DEVICE typename Traits::MainloopPipeline make_dw_mainloop_pipeline_sm100(
		typename Traits::MainloopPipeline::SharedStorage& storage) {
	using Config = typename Traits::Config;
	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp = static_cast<int>(threadIdx.x) / Config::kWarpSize;
	typename Pipeline::Params params;
	params.transaction_bytes = Traits::kTmaTransactionBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = Config::kTmaWarp;
	if (warp == Config::kTmaWarp) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == Config::kTmaWarp * Config::kWarpSize &&
			cute::block_rank_in_cluster() == 0;
	} else if (warp == Config::kMmaWarp) {
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

template <int KTiles, class TmaA, class TmaB, class TmaOutput>
__global__ __launch_bounds__(DwGemmConfigSm100::kNumThreads, 1)
	__cluster_dims__(2, 1, 1)
void backward_dw_gemm_kernel_sm100(
		__grid_constant__ const DwTmaBundleSm100<TmaA, TmaB, TmaOutput>
			tma,
		int m,
		int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	static_assert(KTiles == 16 || KTiles == 64);
	using Traits = DwGemmTraitsSm100<TmaA, TmaB, TmaOutput>;
	using Config = typename Traits::Config;
	using Smem = DwGemmSharedStorageSm100<Traits>;
	using Element = typename Traits::Element;

	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Smem*>(raw_smem);

	int warp = static_cast<int>(threadIdx.x) / Config::kWarpSize;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster = static_cast<int>(blockIdx.x) / Config::kClusterM;
	int cluster_count =
		static_cast<int>(gridDim.x) / Config::kClusterM;
	bool leader_cta = cluster_rank == 0;
	bool epilogue_warp =
		warp >= Config::kFirstEpilogueWarp &&
		warp < Config::kFirstEpilogueWarp + Config::kEpilogueWarps;
	int epilogue_tid =
		static_cast<int>(threadIdx.x) -
		Config::kFirstEpilogueWarp * Config::kWarpSize;
	int epilogue_warpgroup =
		epilogue_warp ? epilogue_tid / Config::kWarpgroupSize : 0;
	int tid_in_warpgroup =
		epilogue_warp ? epilogue_tid % Config::kWarpgroupSize : 0;
	int warpgroup_barrier =
		Config::kWarpgroupBarrierBase + epilogue_warpgroup;

	int m_pairs = (m + Config::kTileM - 1) / Config::kTileM;
	int n_tiles = (n + Config::kTileN - 1) / Config::kTileN;
	int n_pairs = (n_tiles + 1) / 2;
	int work_items = m_pairs * n_pairs;
	int first_item = static_cast<int>(
		static_cast<std::int64_t>(work_items) * cluster /
		cluster_count);
	int last_item = static_cast<int>(
		static_cast<std::int64_t>(work_items) * (cluster + 1) /
		cluster_count);
	constexpr int k_tiles = KTiles;
	constexpr int k = KTiles * Config::kTileK;

	if (warp == Config::kTmaWarp) {
		cute::prefetch_tma_descriptor(tma.a.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.b.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.output.get_tma_descriptor());
	}

	auto mainloop =
		make_dw_mainloop_pipeline_sm100<Traits>(smem.mainloop);

	cute::TMEM::Allocator2Sm tmem_allocator;
	cute::cluster_sync();
	if (warp == Config::kFirstEpilogueWarp) {
		tmem_allocator.allocate(
			Config::kAccumulatorStages * Config::kTileN,
			&smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();

	typename Traits::TiledMma tiled_mma;
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	auto sA = make_tensor(
		make_smem_ptr(&smem.operand_a[0]),
		typename Traits::SmemLayoutA{});
	auto sB = make_tensor(
		make_smem_ptr(&smem.operand_b[0]),
		typename Traits::SmemLayoutB{});

	auto mA = tma.a.get_tma_tensor(make_shape(
		static_cast<std::int64_t>(m),
		static_cast<std::int64_t>(k)));
	auto mB = tma.b.get_tma_tensor(make_shape(
		static_cast<std::int64_t>(n),
		static_cast<std::int64_t>(k)));
	auto mOutput = tma.output.get_tma_tensor(make_shape(
		static_cast<std::int64_t>(m),
		static_cast<std::int64_t>(n)));

	typename Traits::MainloopState mainloop_state;
	if (warp == Config::kTmaWarp) {
		mainloop_state =
			cutlass::make_producer_start_state<
				typename Traits::MainloopPipeline>();
	}

	using AccumulatorPipeline = typename Traits::AccumulatorPipeline;
	typename AccumulatorPipeline::Params accumulator_params;
	accumulator_params.role =
		warp == Config::kMmaWarp && leader_cta
		? AccumulatorPipeline::ThreadCategory::Producer
		: (epilogue_warp
			? AccumulatorPipeline::ThreadCategory::Consumer
			: AccumulatorPipeline::ThreadCategory::NonParticipant);
	accumulator_params.producer_arv_count = 1;
	accumulator_params.consumer_arv_count = Config::kClusterM;
	accumulator_params.initializing_warp =
		Config::kFirstEpilogueWarp;
	AccumulatorPipeline accumulator_pipeline(
		smem.accumulator,
		accumulator_params,
		typename Traits::ClusterShape{});
	auto accumulator_producer_state =
		cutlass::make_producer_start_state<AccumulatorPipeline>();
	typename AccumulatorPipeline::PipelineState accumulator_consumer_state;

	if (warp == Config::kMmaWarp || epilogue_warp) {
		cutlass::arch::NamedBarrier::sync(
			Config::kMmaEpilogueThreads,
			Config::kMmaEpilogueBarrierId);
	}

	if (warp == Config::kTmaWarp) {
		typename Traits::ClusterLayoutVMNK cluster_layout;
		auto cluster_coord =
			cluster_layout.get_flat_coord(cluster_rank);
		std::uint16_t multicast_a =
			create_tma_multicast_mask<2>(cluster_layout, cluster_coord);
		std::uint16_t multicast_b =
			create_tma_multicast_mask<1>(cluster_layout, cluster_coord);

		for (int item = first_item; item < last_item; ++item) {
			DwPairCoordSm100 coord =
				dw_pair_coord_sm100(item, m_pairs);
			int n_tile0 = coord.n_tile_begin;
			int n_tile1 =
				n_tile0 + 1 < n_tiles ? n_tile0 + 1 : n_tile0;
			bool has_second = n_tile0 + 1 < n_tiles;
			auto tile_coord0 =
				make_coord(coord.m_pair, n_tile0, _);
			auto tile_coord1 =
				make_coord(coord.m_pair, n_tile1, _);
			auto gA = local_tile(
				mA,
				typename Traits::TileShape{},
				tile_coord0,
				Step<_1, X, _1>{});
			auto gB0 = local_tile(
				mB,
				typename Traits::TileShape{},
				tile_coord0,
				Step<X, _1, _1>{});
			auto gB1 = local_tile(
				mB,
				typename Traits::TileShape{},
				tile_coord1,
				Step<X, _1, _1>{});
			auto tCgA = cta_mma.partition_A(gA);
			auto tCgB0 = cta_mma.partition_B(gB0);
			auto tCgB1 = cta_mma.partition_B(gB1);
			auto [tAgA, tAsA] = tma_partition(
				tma.a,
				get<2>(cluster_coord),
				make_layout(size<2>(cluster_layout)),
				group_modes<0, 3>(sA),
				group_modes<0, 3>(tCgA));
			auto [tBgB0, tBsB0] = tma_partition(
				tma.b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout)),
				group_modes<0, 3>(sB),
				group_modes<0, 3>(tCgB0));
			auto [tBgB1, tBsB1] = tma_partition(
				tma.b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout)),
				group_modes<0, 3>(sB),
				group_modes<0, 3>(tCgB1));

			CUTE_NO_UNROLL
			for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
				mainloop.producer_acquire(mainloop_state);
				if (cute::elect_one_sync()) {
					auto* barrier =
						mainloop.producer_get_barrier(mainloop_state);
					copy(
						tma.a.with(*barrier, multicast_a),
						tAgA(_, k_tile),
						tAsA(_, mainloop_state.index()));
					copy(
						tma.b.with(*barrier, multicast_b),
						tBgB0(_, k_tile),
						tBsB0(_, mainloop_state.index()));
				}
				++mainloop_state;
				if (has_second) {
					mainloop.producer_acquire(mainloop_state);
					if (cute::elect_one_sync()) {
						auto* barrier =
							mainloop.producer_get_barrier(mainloop_state);
						copy(
							tma.a.with(*barrier, multicast_a),
							tAgA(_, k_tile),
							tAsA(_, mainloop_state.index()));
						copy(
							tma.b.with(*barrier, multicast_b),
							tBgB1(_, k_tile),
							tBsB1(_, mainloop_state.index()));
					}
					++mainloop_state;
				}
			}
		}
		mainloop.producer_tail(mainloop_state);
	}

	if (warp == Config::kMmaWarp && leader_cta) {
		auto tCrA = cta_mma.make_fragment_A(sA);
		auto tCrB = cta_mma.make_fragment_B(sB);
		auto cAccumulator = make_identity_tensor(make_shape(
			Int<Config::kTileM>{},
			Int<Config::kTileN>{}));
		auto tCtAccumulator =
			cta_mma.make_fragment_C(cta_mma.partition_C(cAccumulator));

		for (int item = first_item; item < last_item; ++item) {
			DwPairCoordSm100 coord =
				dw_pair_coord_sm100(item, m_pairs);
			bool has_second = coord.n_tile_begin + 1 < n_tiles;

			accumulator_pipeline.producer_acquire(
				accumulator_producer_state);
			auto accumulator_state0 = accumulator_producer_state;
			++accumulator_producer_state;
			typename AccumulatorPipeline::PipelineState
				accumulator_state1;
			if (has_second) {
				accumulator_pipeline.producer_acquire(
					accumulator_producer_state);
				accumulator_state1 = accumulator_producer_state;
				++accumulator_producer_state;
			}

			CUTE_NO_UNROLL
			for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
				auto accumulate_tile =
					[&](const auto& accumulator_state) {
						mainloop.consumer_wait(mainloop_state);
						tCtAccumulator.data() =
							smem.tmem_base +
							static_cast<std::uint32_t>(
								accumulator_state.index() *
								Config::kTileN);
						CUTE_UNROLL
						for (int k_block = 0;
								k_block < size<2>(tCrA);
								++k_block) {
							tiled_mma.accumulate_ =
								k_tile == 0 && k_block == 0
								? UMMA::ScaleOut::Zero
								: UMMA::ScaleOut::One;
							gemm(
								tiled_mma,
								tCrA(
									_,
									_,
									k_block,
									mainloop_state.index()),
								tCrB(
									_,
									_,
									k_block,
									mainloop_state.index()),
								tCtAccumulator);
						}
						mainloop.consumer_release(mainloop_state);
						++mainloop_state;
						if (k_tile + 1 == k_tiles) {
							accumulator_pipeline.producer_commit(
								accumulator_state);
						}
					};
				accumulate_tile(accumulator_state0);
				if (has_second) {
					accumulate_tile(accumulator_state1);
				}
			}
		}
	}

	if (epilogue_warp) {
		auto cAccumulator = make_identity_tensor(make_shape(
			Int<Config::kTileM>{},
			Int<Config::kTileN>{}));
		auto tCtAccumulator =
			cta_mma.make_fragment_C(cta_mma.partition_C(cAccumulator));
		tCtAccumulator.data() = smem.tmem_base;

		auto epilogue_tile = make_tile(
			Int<Config::kCtaTileM>{},
			Int<Config::kEpilogueN>{});
		auto accumulator_mn =
			tCtAccumulator(make_coord(_, _), _0{}, _0{});
		auto accumulator_epilogue =
			flat_divide(accumulator_mn, epilogue_tile);
		auto tmem_to_register = make_tmem_copy(
			::liger::TmemLoadOp<Config::kEpilogueN>{},
			accumulator_epilogue(_, _, _0{}, _0{}));
		auto thread_tmem_to_register =
			tmem_to_register.get_slice(tid_in_warpgroup);
		auto identity_chunk = make_identity_tensor(make_shape(
			Int<Config::kCtaTileM>{},
			Int<Config::kEpilogueN>{}));
		auto thread_register_coord =
			thread_tmem_to_register.partition_D(identity_chunk);
		auto register_accumulator =
			make_tensor<float>(shape(thread_register_coord));

		constexpr int kOutputSlotElements =
			cosize_v<typename Traits::SmemLayoutOutput1>;
		Element* output_base =
			&smem.output[
				epilogue_warpgroup *
				Config::kOutputStages *
				kOutputSlotElements];
		auto sOutput0 = cute::as_position_independent_swizzle_tensor(
			make_tensor(
				make_smem_ptr(output_base),
				typename Traits::SmemLayoutOutput1{}));
		auto sOutput1 = cute::as_position_independent_swizzle_tensor(
			make_tensor(
				make_smem_ptr(
					output_base + kOutputSlotElements),
				typename Traits::SmemLayoutOutput1{}));
		auto cta_output = tma.output.get_slice(Int<0>{});
		auto tSsOutput0 = cta_output.partition_S(sOutput0);
		auto tSsOutput1 = cta_output.partition_S(sOutput1);
		auto tSgOutput = cta_output.partition_D(local_tile(
			mOutput,
			epilogue_tile,
			make_coord(_, _)));

		for (int item = first_item; item < last_item; ++item) {
			DwPairCoordSm100 coord =
				dw_pair_coord_sm100(item, m_pairs);
			CUTE_UNROLL
			for (int n_in_pair = 0; n_in_pair < 2; ++n_in_pair) {
				int n_tile = coord.n_tile_begin + n_in_pair;
				if (n_tile >= n_tiles) continue;

				accumulator_pipeline.consumer_wait(
					accumulator_consumer_state);
				tCtAccumulator.data() =
					smem.tmem_base +
					static_cast<std::uint32_t>(
						accumulator_consumer_state.index() *
						Config::kTileN);
				auto accumulator_mn_stage =
					tCtAccumulator(make_coord(_, _), _0{}, _0{});
				auto accumulator_epilogue_stage =
					flat_divide(accumulator_mn_stage, epilogue_tile);
				auto thread_tmem_accumulator =
					thread_tmem_to_register.partition_S(
						accumulator_epilogue_stage);

				CUTE_UNROLL
				for (int local_fragment = 0;
						local_fragment <
							Config::kTileN /
							Config::kEpilogueN /
							Config::kEpilogueWarpgroups;
						++local_fragment) {
					int fragment =
						epilogue_warpgroup *
							(Config::kTileN /
								Config::kEpilogueN /
								Config::kEpilogueWarpgroups) +
						local_fragment;
					auto accumulator_fragment =
						thread_tmem_accumulator(
							_,
							_,
							_,
							_0{},
							fragment);
					copy(
						tmem_to_register,
						accumulator_fragment,
						register_accumulator);
					cutlass::arch::fence_view_async_tmem_load();

					int output_stage =
						fragment % Config::kOutputStages;
					if (tid_in_warpgroup == 0) {
						cute::tma_store_wait<
							Config::kOutputStages - 1>();
					}
					cutlass::arch::NamedBarrier::sync(
						Config::kWarpgroupSize,
						warpgroup_barrier);

					auto& sOutput =
						output_stage == 0 ? sOutput0 : sOutput1;
					CUTE_UNROLL
					for (int element = 0;
							element < size(register_accumulator);
							++element) {
						sOutput(
							get<0>(thread_register_coord(element)),
							get<1>(thread_register_coord(element))) =
							static_cast<Element>(
								register_accumulator(element));
					}
					cutlass::arch::NamedBarrier::sync(
						Config::kWarpgroupSize,
						warpgroup_barrier);
					if (tid_in_warpgroup == 0) {
						cutlass::arch::fence_view_async_shared();
						int m_tile =
							coord.m_pair * Config::kClusterM +
							cluster_rank;
						int n_chunk =
							n_tile *
								(Config::kTileN /
									Config::kEpilogueN) +
							fragment;
						copy(
							tma.output,
							output_stage == 0
								? tSsOutput0
								: tSsOutput1,
							tSgOutput(
								_,
								_,
								_,
								m_tile,
								n_chunk));
						cute::tma_store_arrive();
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kEpilogueThreads,
					Config::kEpilogueBarrierId);
				if (epilogue_tid == 0) {
					accumulator_pipeline.consumer_release(
						accumulator_consumer_state);
				}
				++accumulator_consumer_state;
			}
		}
		if (tid_in_warpgroup == 0) {
			cute::tma_store_wait<0>();
		}
	}

	__syncthreads();
	cute::cluster_sync();
	if (warp == Config::kFirstEpilogueWarp) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(
			smem.tmem_base,
			Config::kAccumulatorStages * Config::kTileN);
	}
#else
	(void)tma;
	(void)m;
	(void)n;
#endif
}

template <bool Add>
auto make_backward_dw_tma_bundle_sm100(
		const cutlass::bfloat16_t* dz_km,
		const cutlass::bfloat16_t* x_kn,
		cutlass::bfloat16_t* dw_mn,
		int m,
		int n,
		int k) {
	using LayoutTraits = DwGemmTraitsSm100<int, int, int>;
	auto tensor_a = make_tensor(
		make_gmem_ptr(dz_km),
		make_shape(
			static_cast<std::int64_t>(m),
			static_cast<std::int64_t>(k)),
		make_stride(Int<1>{}, static_cast<std::int64_t>(m)));
	auto tensor_b = make_tensor(
		make_gmem_ptr(x_kn),
		make_shape(
			static_cast<std::int64_t>(n),
			static_cast<std::int64_t>(k)),
		make_stride(Int<1>{}, static_cast<std::int64_t>(n)));
	auto tensor_output = make_tensor(
		make_gmem_ptr(dw_mn),
		make_shape(
			static_cast<std::int64_t>(m),
			static_cast<std::int64_t>(n)),
		make_stride(static_cast<std::int64_t>(n), Int<1>{}));

	auto tma_a = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_a,
		typename LayoutTraits::SmemLayoutA1{},
		typename LayoutTraits::TileShape{},
		typename LayoutTraits::TiledMma{});
	auto tma_b = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_b,
		typename LayoutTraits::SmemLayoutB1{},
		typename LayoutTraits::TileShape{},
		typename LayoutTraits::TiledMma{});
	if constexpr (Add) {
		auto tma_output = make_tma_copy(
			SM90_TMA_REDUCE_ADD{},
			tensor_output,
			typename LayoutTraits::SmemLayoutOutput1{});
		return DwTmaBundleSm100<
			decltype(tma_a),
			decltype(tma_b),
			decltype(tma_output)>{tma_a, tma_b, tma_output};
	} else {
		auto tma_output = make_tma_copy(
			SM90_TMA_STORE{},
			tensor_output,
			typename LayoutTraits::SmemLayoutOutput1{});
		return DwTmaBundleSm100<
			decltype(tma_a),
			decltype(tma_b),
			decltype(tma_output)>{tma_a, tma_b, tma_output};
	}
}

template <int KTiles, class TmaA, class TmaB, class TmaOutput>
cudaError_t prepare_backward_dw_gemm_sm100() {
	using Traits = DwGemmTraitsSm100<TmaA, TmaB, TmaOutput>;
	using Smem = DwGemmSharedStorageSm100<Traits>;
	auto kernel =
		&backward_dw_gemm_kernel_sm100<
			KTiles,
			TmaA,
			TmaB,
			TmaOutput>;
	cudaError_t error = cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeNonPortableClusterSizeAllowed,
		1);
	if (error != cudaSuccess) return error;
	error = cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		static_cast<int>(sizeof(Smem)));
	return error;
}

template <int KTiles, class TmaA, class TmaB, class TmaOutput>
cudaError_t launch_backward_dw_gemm_sm100(
		const DwTmaBundleSm100<TmaA, TmaB, TmaOutput>& tma,
		int m,
		int n,
		int cluster_pairs,
		cudaStream_t stream = nullptr) {
	using Traits = DwGemmTraitsSm100<TmaA, TmaB, TmaOutput>;
	using Smem = DwGemmSharedStorageSm100<Traits>;
	auto kernel =
		&backward_dw_gemm_kernel_sm100<
			KTiles,
			TmaA,
			TmaB,
			TmaOutput>;

	cudaLaunchAttribute attributes[2] = {};
	attributes[0].id = cudaLaunchAttributeClusterDimension;
	attributes[0].val.clusterDim.x = DwGemmConfigSm100::kClusterM;
	attributes[0].val.clusterDim.y = 1;
	attributes[0].val.clusterDim.z = 1;
	attributes[1].id =
		cudaLaunchAttributeClusterSchedulingPolicyPreference;
	attributes[1].val.clusterSchedulingPolicyPreference =
		cudaClusterSchedulingPolicySpread;
	cudaLaunchConfig_t config = {};
	config.gridDim = dim3(
		static_cast<unsigned>(
			DwGemmConfigSm100::kClusterM * cluster_pairs),
		1,
		1);
	config.blockDim = dim3(
		DwGemmConfigSm100::kNumThreads,
		1,
		1);
	config.dynamicSmemBytes = sizeof(Smem);
	config.stream = stream;
	config.attrs = attributes;
	config.numAttrs = 2;
	return cudaLaunchKernelEx(
		&config,
		kernel,
		tma,
		m,
		n);
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
