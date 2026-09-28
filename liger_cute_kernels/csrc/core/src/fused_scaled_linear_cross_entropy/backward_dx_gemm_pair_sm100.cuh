#pragma once

// Paired two-tile schedules for the standalone SM100 TP-FSLCE dX GEMM.
//
// A 2x1 CTA cluster computes two logical M256xN256 tiles sharing either their
// M coordinate (N-pair / horizontal) or N coordinate (M-pair / vertical).
// For each K64 step the common operand is loaded once and reused by two 2SM
// UMMA instructions.  The two FP32 accumulators occupy the full 512-column
// Allocator2Sm allocation and drain to the same compact
// M128xN256-per-CTA staging layout as the single-tile kernel.

#include "backward_dx_gemm_sm100.cuh"

namespace liger {
namespace fused_scaled_linear_cross_entropy {

enum class DxPairAxisSm100 : std::uint8_t {
	kN,
	kM,
};

template <
	DxPairAxisSm100 PairAxis_,
	int Stages_ = 4,
	int EpiChunkN_ = 32>
struct DxGemmPairTraitsSm100 {
	using Base = DxGemmTraitsSm100<Stages_, EpiChunkN_>;
	using Element = typename Base::Element;
	using ElementAccum = typename Base::ElementAccum;
	static constexpr DxPairAxisSm100 kPairAxis = PairAxis_;

	static constexpr int kTileM = Base::kTileM;
	static constexpr int kCtaTileM = Base::kCtaTileM;
	static constexpr int kTileN = Base::kTileN;
	static constexpr int kTileK = Base::kTileK;
	static constexpr int kStages = Base::kStages;
	static constexpr int kEpiChunkN = Base::kEpiChunkN;
	static constexpr int kClusterM = Base::kClusterM;
	static constexpr int kTmemColumns = 2 * kTileN;

	static constexpr int kWarpSize = Base::kWarpSize;
	static constexpr int kNumThreads = Base::kNumThreads;
	static constexpr int kTmaWarp = Base::kTmaWarp;
	static constexpr int kMmaWarp = Base::kMmaWarp;
	static constexpr int kFirstEpilogueWarp =
		Base::kFirstEpilogueWarp;
	static constexpr int kLastEpilogueWarp =
		Base::kLastEpilogueWarp;
	static constexpr int kEpilogueThreads =
		Base::kEpilogueThreads;
	static constexpr int kWarpgroupSize =
		Base::kWarpgroupSize;
	static constexpr int kEpilogueWarpgroups =
		Base::kEpilogueWarpgroups;
	static constexpr int kChunksPerWarpgroup =
		Base::kChunksPerWarpgroup;
	static constexpr int kMmaEpilogueThreads =
		Base::kMmaEpilogueThreads;
	static constexpr int kWarpgroup0BarrierId =
		Base::kWarpgroup0BarrierId;
	static constexpr int kMmaEpilogueBarrierId =
		Base::kMmaEpilogueBarrierId;
	static constexpr int kEpilogueBarrierId =
		Base::kEpilogueBarrierId;

	using TileShape = typename Base::TileShape;
	using ClusterShape = typename Base::ClusterShape;
	using AtomThrShape = typename Base::AtomThrShape;
	using TiledMma = typename Base::TiledMma;
	using ClusterLayoutVMNK =
		typename Base::ClusterLayoutVMNK;
	using SmemLayoutA = typename Base::SmemLayoutA;
	using SmemLayoutB = typename Base::SmemLayoutB;
	using SmemLayoutA1 = typename Base::SmemLayoutA1;
	using SmemLayoutB1 = typename Base::SmemLayoutB1;
	using SmemLayoutStore = typename Base::SmemLayoutStore;

	static constexpr int kStoreElements =
		Base::kStoreElements;
	static constexpr int kTmaTransBytesA =
		Base::kTmaTransBytesA;
	static constexpr int kTmaTransBytesB =
		Base::kTmaTransBytesB;
	static constexpr int kTmaTransBytes =
		kClusterM *
		(kPairAxis == DxPairAxisSm100::kN
			? kTmaTransBytesA + 2 * kTmaTransBytesB
			: 2 * kTmaTransBytesA + kTmaTransBytesB);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	// Both logical outputs are one handoff unit.  The epilogue releases this
	// single stage only after both N256 accumulators have drained.
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		1,
		AtomThrShape>;
};

template <int Stages_ = 4, int EpiChunkN_ = 32>
using DxGemmNPairTraitsSm100 = DxGemmPairTraitsSm100<
	DxPairAxisSm100::kN, Stages_, EpiChunkN_>;

template <int Stages_ = 4, int EpiChunkN_ = 32>
using DxGemmMPairTraitsSm100 = DxGemmPairTraitsSm100<
	DxPairAxisSm100::kM, Stages_, EpiChunkN_>;

template <class Traits>
struct DxGemmPairSmemSm100 {
	using Element = typename Traits::Element;
	static constexpr int kACopies =
		Traits::kPairAxis == DxPairAxisSm100::kM ? 2 : 1;
	static constexpr int kBCopies =
		Traits::kPairAxis == DxPairAxisSm100::kN ? 2 : 1;

	alignas(1024) Element operand_a[
		kACopies * cosize_v<typename Traits::SmemLayoutA>];
	alignas(1024) Element operand_b[
		kBCopies * cosize_v<typename Traits::SmemLayoutB>];
	alignas(1024) float store[
		Traits::kEpilogueWarpgroups * Traits::kStoreElements];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	alignas(16) std::uint32_t tmem_base;

	CUTE_DEVICE Element* a_data(int index) {
		return &operand_a[
			index * cosize_v<typename Traits::SmemLayoutA>];
	}
	CUTE_DEVICE Element* b_data(int index) {
		return &operand_b[
			index * cosize_v<typename Traits::SmemLayoutB>];
	}
	CUTE_DEVICE float* store_data(int warpgroup) {
		return &store[warpgroup * Traits::kStoreElements];
	}
};

template <class Traits>
CUTE_DEVICE typename Traits::MainloopPipeline dx_pair_make_pipe_sm100(
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
	bool FixedRepresentative,
	class Traits,
	class TmaLoadA,
	class TmaLoadB,
	class TmaStore>
__global__ __launch_bounds__(Traits::kNumThreads, 1) __cluster_dims__(2, 1, 1)
void dx_gemm_pair_kernel_sm100(
		__grid_constant__ const TmaLoadA tma_a,
		__grid_constant__ const TmaLoadB tma_b,
		__grid_constant__ const TmaStore tma_store,
		int m,
		int n,
		int k,
		int num_m_tiles,
		int num_n_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using AccPipe = typename Traits::AccumulatorPipeline;
	using Smem = DxGemmPairSmemSm100<Traits>;

	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);

	int warp_id = static_cast<int>(threadIdx.x) / Traits::kWarpSize;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster_idx = static_cast<int>(blockIdx.x) / Traits::kClusterM;
	int num_clusters;
	int pair_count;
	int unpaired_tiles;
	int total_groups;
	int num_k_tiles;
	if constexpr (FixedRepresentative) {
		num_clusters = 64;
		pair_count = 8;
		unpaired_tiles = 16;
		total_groups = 128;
		num_k_tiles = 1024;
	} else {
		num_clusters =
			static_cast<int>(gridDim.x) / Traits::kClusterM;
		int paired_tiles =
			Traits::kPairAxis == DxPairAxisSm100::kN
			? num_n_tiles
			: num_m_tiles;
		pair_count = (paired_tiles + 1) / 2;
		unpaired_tiles =
			Traits::kPairAxis == DxPairAxisSm100::kN
			? num_m_tiles
			: num_n_tiles;
		total_groups = unpaired_tiles * pair_count;
		num_k_tiles =
			(k + Traits::kTileK - 1) / Traits::kTileK;
	}

	cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store.get_tma_descriptor());

	auto pipe = dx_pair_make_pipe_sm100<Traits>(smem.pipeline);

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

	auto sA0 = make_tensor(
		make_smem_ptr(smem.a_data(0)), typename Traits::SmemLayoutA{});
	auto sA1 = make_tensor(
		make_smem_ptr(smem.a_data(
			Traits::kPairAxis == DxPairAxisSm100::kM ? 1 : 0)),
		typename Traits::SmemLayoutA{});
	auto sB0 = make_tensor(
		make_smem_ptr(smem.b_data(0)), typename Traits::SmemLayoutB{});
	auto sB1 = make_tensor(
		make_smem_ptr(smem.b_data(
			Traits::kPairAxis == DxPairAxisSm100::kN ? 1 : 0)),
		typename Traits::SmemLayoutB{});
	auto tCrA0 = cta_mma.make_fragment_A(sA0);
	auto tCrA1 = cta_mma.make_fragment_A(sA1);
	auto tCrB0 = cta_mma.make_fragment_B(sB0);
	auto tCrB1 = cta_mma.make_fragment_B(sB1);
	auto cAccFull = make_identity_tensor(
		make_shape(Int<Traits::kTileM>{}, Int<Traits::kTileN>{}));
	auto tCtAcc0 = cta_mma.make_fragment_C(
		cta_mma.partition_C(cAccFull));
	auto tCtAcc1 = cta_mma.make_fragment_C(
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

		for (int group = cluster_idx;
				group < total_groups;
				group += num_clusters) {
			int outer = group / pair_count;
			int pair_index = group - outer * pair_count;
			int paired0 = 2 * pair_index;
			int m0 = Traits::kPairAxis == DxPairAxisSm100::kN
				? outer
				: paired0;
			int m1 = Traits::kPairAxis == DxPairAxisSm100::kN
				? m0
				: m0 + 1;
			int n0 = Traits::kPairAxis == DxPairAxisSm100::kN
				? paired0
				: outer;
			int n1 = Traits::kPairAxis == DxPairAxisSm100::kN
				? n0 + 1
				: n0;
			auto gA0 = local_tile(
				mA,
				typename Traits::TileShape{},
				make_coord(m0, n0, _),
				Step<_1, X, _1>{});
			auto gA1 = local_tile(
				mA,
				typename Traits::TileShape{},
				make_coord(m1, n1, _),
				Step<_1, X, _1>{});
			auto gB0 = local_tile(
				mB,
				typename Traits::TileShape{},
				make_coord(m0, n0, _),
				Step<X, _1, _1>{});
			auto gB1 = local_tile(
				mB,
				typename Traits::TileShape{},
				make_coord(m1, n1, _),
				Step<X, _1, _1>{});
			auto tCgA0 = cta_mma.partition_A(gA0);
			auto tCgA1 = cta_mma.partition_A(gA1);
			auto tCgB0 = cta_mma.partition_B(gB0);
			auto tCgB1 = cta_mma.partition_B(gB1);
			auto [tA0gA, tA0sA] = tma_partition(
				tma_a,
				get<2>(cluster_coord),
				make_layout(size<2>(cluster_layout_vmnk)),
				group_modes<0, 3>(sA0),
				group_modes<0, 3>(tCgA0));
			auto [tA1gA, tA1sA] = tma_partition(
				tma_a,
				get<2>(cluster_coord),
				make_layout(size<2>(cluster_layout_vmnk)),
				group_modes<0, 3>(sA1),
				group_modes<0, 3>(tCgA1));
			auto [tB0gB, tB0sB] = tma_partition(
				tma_b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout_vmnk)),
				group_modes<0, 3>(sB0),
				group_modes<0, 3>(tCgB0));
			auto [tB1gB, tB1sB] = tma_partition(
				tma_b,
				get<1>(cluster_coord),
				make_layout(size<1>(cluster_layout_vmnk)),
				group_modes<0, 3>(sB1),
				group_modes<0, 3>(tCgB1));

			#pragma unroll 4
			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.producer_acquire(mainloop_state);
				if (cute::elect_one_sync()) {
					auto* barrier =
						pipe.producer_get_barrier(mainloop_state);
					copy(
						tma_a.with(*barrier, mcast_mask_a),
						tA0gA(_, k_tile),
						tA0sA(_, mainloop_state.index()));
					if constexpr (
						Traits::kPairAxis ==
						DxPairAxisSm100::kM) {
						copy(
							tma_a.with(*barrier, mcast_mask_a),
							tA1gA(_, k_tile),
							tA1sA(_, mainloop_state.index()));
					}
					copy(
						tma_b.with(*barrier, mcast_mask_b),
						tB0gB(_, k_tile),
						tB0sB(_, mainloop_state.index()));
					if constexpr (
						Traits::kPairAxis ==
						DxPairAxisSm100::kN) {
						copy(
							tma_b.with(*barrier, mcast_mask_b),
							tB1gB(_, k_tile),
							tB1sB(_, mainloop_state.index()));
					}
				}
				++mainloop_state;
			}
		}
		pipe.producer_tail(mainloop_state);
	}

	if (is_mma_warp && is_leader_cta) {
		for (int group = cluster_idx;
				group < total_groups;
				group += num_clusters) {
			(void)group;
			acc_pipe.producer_acquire(acc_prod_state);
			tCtAcc0.data() = tmem_base;
			tCtAcc1.data() =
				tmem_base + static_cast<std::uint32_t>(Traits::kTileN);
			pipe.consumer_wait(mainloop_state);
			bool first_block = true;
			CUTE_UNROLL
			for (int k_block = 0;
					k_block < size<2>(tCrA0);
					++k_block) {
				tiled_mma.accumulate_ = first_block
					? UMMA::ScaleOut::Zero
					: UMMA::ScaleOut::One;
				gemm(
					tiled_mma,
					tCrA0(
						_, _, k_block,
						mainloop_state.index()),
					tCrB0(
						_, _, k_block,
						mainloop_state.index()),
					tCtAcc0);
				gemm(
					tiled_mma,
					tCrA1(
						_, _, k_block,
						mainloop_state.index()),
					(Traits::kPairAxis ==
							DxPairAxisSm100::kN
						? tCrB1
						: tCrB0)(
						_, _, k_block,
						mainloop_state.index()),
					tCtAcc1);
				first_block = false;
			}
			pipe.consumer_release(mainloop_state);
			++mainloop_state;

			#pragma unroll 4
			for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
				pipe.consumer_wait(mainloop_state);
				CUTE_UNROLL
				for (int k_block = 0;
						k_block < size<2>(tCrA0);
						++k_block) {
					tiled_mma.accumulate_ = UMMA::ScaleOut::One;
					gemm(
						tiled_mma,
						tCrA0(
							_, _, k_block,
							mainloop_state.index()),
						tCrB0(
							_, _, k_block,
							mainloop_state.index()),
						tCtAcc0);
					gemm(
						tiled_mma,
						tCrA1(
							_, _, k_block,
							mainloop_state.index()),
						(Traits::kPairAxis ==
								DxPairAxisSm100::kN
							? tCrB1
							: tCrB0)(
							_, _, k_block,
							mainloop_state.index()),
						tCtAcc1);
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

		auto epi_tile = make_tile(
			Int<Traits::kCtaTileM>{},
			Int<Traits::kEpiChunkN>{});
		tCtAcc0.data() = tmem_base;
		auto acc_mn =
			tCtAcc0(make_coord(_, _), _0{}, _0{});
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
			num_m_tiles * num_n_tiles *
			Traits::kClusterM * Traits::kCtaTileM;
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

		for (int group = cluster_idx;
				group < total_groups;
				group += num_clusters) {
			int outer = group / pair_count;
			int pair_index = group - outer * pair_count;
			int paired0 = 2 * pair_index;

			acc_pipe.consumer_wait(acc_cons_state);
			CUTE_UNROLL
			for (int output = 0; output < 2; ++output) {
				int m_tile =
					Traits::kPairAxis == DxPairAxisSm100::kN
					? outer
					: paired0 + output;
				int n_tile =
					Traits::kPairAxis == DxPairAxisSm100::kN
					? paired0 + output
					: outer;
				if (
					m_tile >= num_m_tiles ||
					n_tile >= num_n_tiles) {
					continue;
				}
				tCtAcc0.data() = tmem_base +
					static_cast<std::uint32_t>(
						output * Traits::kTileN);
				auto acc_mn_output =
					tCtAcc0(make_coord(_, _), _0{}, _0{});
				auto tAccEpiOutput =
					flat_divide(acc_mn_output, epi_tile);
				auto tTR_tAcc =
					thr_t2r.partition_S(tAccEpiOutput);
				int tile_linear =
					m_tile * num_n_tiles + n_tile;
				int staging_m_tile =
					tile_linear * Traits::kClusterM +
					cluster_rank;

				CUTE_UNROLL
				for (int round = 0;
						round < Traits::kChunksPerWarpgroup;
						++round) {
					int chunk =
						warpgroup *
							Traits::kChunksPerWarpgroup +
						round;
					auto tAccChunk =
						tTR_tAcc(_, _, _, _0{}, chunk);
					bool issue_tmem_load = true;
					if constexpr (kPredicateTmemLoad) {
						int subpart =
							(tAccChunk.data().dp_ / 32) % 4;
						issue_tmem_load =
							tid_in_warpgroup /
									Traits::kWarpSize ==
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
						for (int i = 0;
								i < size(tTR_rAcc);
								++i) {
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
