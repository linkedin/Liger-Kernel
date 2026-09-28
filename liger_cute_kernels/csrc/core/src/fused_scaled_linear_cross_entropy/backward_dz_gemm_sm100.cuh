#pragma once

// Isolated SM100 dZ GEMM used to develop and benchmark the backward projection
// without pulling in dX, dW, communication, or the fused token-wave loop.
//
// Geometry and pipelines intentionally match the proven SM100 forward path:
//   * 2x1 clusters and one 2SM UMMA issuer
//   * joined M256 x N256 x K64 tiles (M128 owned by each CTA)
//   * five paired TMA stages
//   * FP32 accumulation in TMEM
//   * two epilogue warpgroups and compact N32 TMA stores
//
// A resident cluster persistently walks output tiles. The five-stage operand
// arena and a compact N32 TMA-store arena are disjoint so the next GEMM tile
// can overlap the current tile's gradient epilogue.

#include "tmem_load_op_sm100.cuh"

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
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace dz_sm100 {

using namespace cute;

#ifndef LIGER_CUTE_FSLCE_SM100_DZ_EPILOGUE_CHUNK_N
#define LIGER_CUTE_FSLCE_SM100_DZ_EPILOGUE_CHUNK_N 32
#endif

#ifndef LIGER_CUTE_FSLCE_SM100_DZ_STORE_BUFFERS
#define LIGER_CUTE_FSLCE_SM100_DZ_STORE_BUFFERS 1
#endif

enum class EpilogueMode : int {
	kRawGemm = 0,
	kSoftmaxGradient = 1,
	kSoftmaxGradientEntropy = 2,
};

struct Config {
	static constexpr int kCtaTileM = 128;
	static constexpr int kTileM = 256;
	static constexpr int kTileN = 256;
	static constexpr int kTileK = 64;
	static constexpr int kClusterM = 2;
	static constexpr int kMainloopStages = 5;
	static constexpr int kAccumulatorStages = 2;
	static constexpr int kTmemStageColumns = kTileN;
	static constexpr int kTmemColumns =
		kAccumulatorStages * kTmemStageColumns;

	static constexpr int kWarpSize = 32;
	static constexpr int kNumWarps = 12;
	static constexpr int kNumThreads = kNumWarps * kWarpSize;
	static constexpr int kTmaWarp = 2;
	static constexpr int kUmmaWarp = 3;
	static constexpr int kFirstEpilogueWarp = 4;
	static constexpr int kLastEpilogueWarp = 11;
	static constexpr int kEpilogueWarps =
		kLastEpilogueWarp - kFirstEpilogueWarp + 1;
	static constexpr int kEpilogueThreads = kEpilogueWarps * kWarpSize;
	static constexpr int kWarpgroupSize = 4 * kWarpSize;
	static constexpr int kEpilogueWarpgroups =
		kEpilogueThreads / kWarpgroupSize;
	static constexpr int kEpilogueChunkN =
		LIGER_CUTE_FSLCE_SM100_DZ_EPILOGUE_CHUNK_N;
	static constexpr int kStoreBuffers =
		LIGER_CUTE_FSLCE_SM100_DZ_STORE_BUFFERS;
	static constexpr int kWarpgroupTileN =
		kTileN / kEpilogueWarpgroups;
	static constexpr int kChunksPerWarpgroup =
		kWarpgroupTileN / kEpilogueChunkN;

	static constexpr int kWarpgroup0BarrierId = 1;
	static constexpr int kWarpgroup1BarrierId = 2;
	static constexpr int kMmaEpilogueBarrierId = 3;
	static constexpr int kEpilogueBarrierId = 4;
	static constexpr int kArenaHandoffBarrierId = 5;
	static constexpr int kTmemFreeBarrierId = 6;
	static constexpr int kMmaEpilogueThreads =
		(kLastEpilogueWarp - kUmmaWarp + 1) * kWarpSize;
	static constexpr int kComputeThreads =
		(kLastEpilogueWarp - kTmaWarp + 1) * kWarpSize;

	static_assert(kCtaTileM * kClusterM == kTileM);
	static_assert(kMainloopStages == 5);
	static_assert(kTmemColumns == 512);
	static_assert(kEpilogueWarpgroups == 2);
	static_assert(
		kEpilogueChunkN == 32 || kEpilogueChunkN == 64,
		"the dZ TMA epilogue supports N32 or N64 chunks");
	static_assert(
		kStoreBuffers == 1 || kStoreBuffers == 2,
		"the dZ TMA epilogue supports one or two store buffers");
	static_assert(kNumThreads == 384);
	static_assert(kMmaEpilogueThreads == 288);
	static_assert(kComputeThreads == 320);
};

struct Params {
	const void* x = nullptr;       // BF16 [tokens, hidden]
	const void* weight = nullptr;  // BF16 [local_vocab, hidden]
	const std::int64_t* target = nullptr;
	const float* grad_output = nullptr;
	const float* lse = nullptr;
	const float* entropy = nullptr;
	const float* entropy_grad = nullptr;
	void* output = nullptr;  // BF16 [padded_tokens, padded_vocab]

	int tokens = 0;
	int hidden = 0;
	int local_vocab = 0;
	int padded_tokens = 0;
	int padded_vocab = 0;
	std::int64_t vocab_start = 0;
	std::int64_t ignore_index = -100;
	float inverse_temperature = 1.0f;
	// Zero selects the maximum resident cluster-pair count. A positive value
	// is a benchmark/tuning override and is clamped to resident capacity.
	int cluster_pairs = 0;
};

inline constexpr float kLog2E = 1.4426950408889634f;

CUTE_HOST_DEVICE int ceil_div_int(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

CUTE_DEVICE float fast_exp2(float value) {
#if defined(__CUDA_ARCH__)
	float result;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
	return result;
#else
	return 0.0f;
#endif
}

struct Traits {
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	using TileShape =
		Shape<Int<Config::kTileM>, Int<Config::kTileN>, Int<Config::kTileK>>;
	using ClusterShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using AtomThrShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using TiledMma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			Config::kTileM,
			Config::kTileN,
			UMMA::Major::K,
			UMMA::Major::K>{}));

	static_assert(
		size(typename TiledMma::AtomThrID{}) == Config::kClusterM,
		"the 2SM UMMA atom must match the two-CTA cluster");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma::AtomThrID{})));
	using MmaShapeX = decltype(partition_shape_A(
		TiledMma{},
		make_shape(Int<Config::kTileM>{}, Int<Config::kTileK>{})));
	using MmaShapeW = decltype(partition_shape_B(
		TiledMma{},
		make_shape(Int<Config::kTileN>{}, Int<Config::kTileK>{})));
	using AtomK = UMMA::Layout_K_SW128_Atom<Element>;
	using SmemLayoutX = decltype(UMMA::tile_to_mma_shape(
		AtomK{},
		append(MmaShapeX{}, Int<Config::kMainloopStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutW = decltype(UMMA::tile_to_mma_shape(
		AtomK{},
		append(MmaShapeW{}, Int<Config::kMainloopStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutX1 = decltype(SmemLayoutX{}(_, _, _, Int<0>{}));
	using SmemLayoutW1 = decltype(SmemLayoutW{}(_, _, _, Int<0>{}));

	static constexpr int kSmemXElements = cosize_v<SmemLayoutX>;
	static constexpr int kSmemWElements = cosize_v<SmemLayoutW>;
	static constexpr int kTmaTransBytesX =
		static_cast<int>(cosize_v<SmemLayoutX1> * sizeof(Element));
	static constexpr int kTmaTransBytesW =
		static_cast<int>(cosize_v<SmemLayoutW1> * sizeof(Element));
	static constexpr int kTmaTransBytes =
		Config::kClusterM * (kTmaTransBytesX + kTmaTransBytesW);

	using StoreAtom = cute::conditional_t<
		Config::kEpilogueChunkN == 32,
		UMMA::Layout_K_SW64_Atom<Element>,
		UMMA::Layout_K_SW128_Atom<Element>>;
	using SmemLayoutStoreSlot = decltype(tile_to_shape(
		StoreAtom{},
		Shape<Int<Config::kCtaTileM>, Int<Config::kEpilogueChunkN>>{}));
	static constexpr int kStoreSlotElements =
		cosize_v<SmemLayoutStoreSlot>;

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		Config::kMainloopStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		Config::kAccumulatorStages,
		AtomThrShape>;
};

template <EpilogueMode Mode>
struct SharedStorage {
	using Element = typename Traits::Element;
	static constexpr bool kGradient =
		Mode != EpilogueMode::kRawGemm;
	static constexpr bool kEntropy =
		Mode == EpilogueMode::kSoftmaxGradientEntropy;

	static constexpr std::size_t align_up(
			std::size_t value, std::size_t alignment) {
		return (value + alignment - 1) / alignment * alignment;
	}

	static constexpr std::size_t kXBytes =
		static_cast<std::size_t>(Traits::kSmemXElements) * sizeof(Element);
	static constexpr std::size_t kWOffset = align_up(kXBytes, 1024);
	static constexpr std::size_t kWBytes =
		static_cast<std::size_t>(Traits::kSmemWElements) * sizeof(Element);
	static constexpr std::size_t kOperandBytes =
		kWOffset + kWBytes;
	static constexpr std::size_t kStoreBytes =
		static_cast<std::size_t>(Config::kEpilogueWarpgroups) *
		static_cast<std::size_t>(Config::kStoreBuffers) *
		static_cast<std::size_t>(Traits::kStoreSlotElements) *
		sizeof(Element);
	static constexpr std::size_t kStoreOffset =
		align_up(kOperandBytes, 1024);
	static constexpr std::size_t kArenaBytes =
		kStoreOffset + kStoreBytes;

	alignas(1024) char arena[kArenaBytes];
	alignas(16) float row_scale[kGradient ? Config::kCtaTileM : 1];
	alignas(16) float row_exp_bias[kGradient ? Config::kCtaTileM : 1];
	alignas(16) float row_entropy_bias[kEntropy ? Config::kCtaTileM : 1];
	alignas(16) float row_entropy_slope[kEntropy ? Config::kCtaTileM : 1];
	alignas(16) int row_target[kGradient ? Config::kCtaTileM : 1];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	alignas(16) std::uint32_t tmem_base;
	alignas(16) cutlass::arch::ClusterBarrier tmem_pair_barrier;

	CUTE_DEVICE Element* x_data() {
		return reinterpret_cast<Element*>(&arena[0]);
	}
	CUTE_DEVICE Element* w_data() {
		return reinterpret_cast<Element*>(&arena[kWOffset]);
	}
	CUTE_DEVICE Element* store_data(int warpgroup, int buffer) {
		std::size_t slot =
			static_cast<std::size_t>(
				warpgroup * Config::kStoreBuffers + buffer);
		return reinterpret_cast<Element*>(&arena[kStoreOffset]) +
			slot * Traits::kStoreSlotElements;
	}
};

template <class TmaX, class TmaW, class TmaStore>
struct TmaBundle {
	TmaX x;
	TmaW weight;
	TmaStore output;
};

CUTE_DEVICE Traits::MainloopPipeline make_mainloop_pipeline(
		typename Traits::MainloopPipeline::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = static_cast<int>(threadIdx.x) / Config::kWarpSize;
	typename Pipeline::Params params;
	params.transaction_bytes = Traits::kTmaTransBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = Config::kTmaWarp;
	if (warp_id == Config::kTmaWarp) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == Config::kTmaWarp * Config::kWarpSize &&
			cute::block_rank_in_cluster() == 0;
	} else if (
			warp_id >= Config::kUmmaWarp &&
			warp_id <= Config::kLastEpilogueWarp) {
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

CUTE_DEVICE void tmem_pair_barrier_sync(
		const cutlass::arch::ClusterBarrier& barrier,
		int warp_id) {
	if (warp_id == Config::kFirstEpilogueWarp) {
		if (cute::elect_one_sync()) {
			std::uint32_t rank = cute::block_rank_in_cluster();
			std::uint32_t peer = rank ^ 1u;
			bool leader = (rank & 1u) == 0u;
			barrier.arrive(peer, !leader);
			barrier.wait(0);
			barrier.arrive(peer, leader);
		}
		__syncwarp();
	}
}

template <
	EpilogueMode Mode,
	class TmaX,
	class TmaW,
	class TmaOutput>
__device__ __forceinline__ void kernel_body(
		const TmaX& tma_x,
		const TmaW& tma_weight,
		const TmaOutput& tma_output,
		const Params& params,
		SharedStorage<Mode>& smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	using AccPipe = typename Traits::AccumulatorPipeline;

	int warp_id = static_cast<int>(threadIdx.x) / Config::kWarpSize;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster_index = static_cast<int>(blockIdx.z);
	int num_clusters = static_cast<int>(gridDim.z);
	int num_m_pairs = ceil_div_int(params.tokens, Config::kTileM);
	int num_n_tiles =
		ceil_div_int(params.local_vocab, Config::kTileN);
	int total_items = num_m_pairs * num_n_tiles;
	int num_k_tiles = ceil_div_int(params.hidden, Config::kTileK);

	cute::prefetch_tma_descriptor(tma_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_weight.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_output.get_tma_descriptor());

	auto pipe = make_mainloop_pipeline(smem.pipeline);
	cutlass::arch::fence_barrier_init();

	cute::TMEM::Allocator2Sm tmem_allocator;
	cute::cluster_sync();
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.allocate(Config::kTmemColumns, &smem.tmem_base);
		if (cute::elect_one_sync()) {
			smem.tmem_pair_barrier.init(1);
		}
		cutlass::arch::fence_barrier_init();
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();
	std::uint32_t tmem_base = smem.tmem_base;

	typename Traits::PipelineState state;
	if (warp_id == Config::kTmaWarp) {
		state = cutlass::make_producer_start_state<
			typename Traits::MainloopPipeline>();

		auto sX = make_tensor(
			make_smem_ptr(smem.x_data()),
			typename Traits::SmemLayoutX{});
		auto sW = make_tensor(
			make_smem_ptr(smem.w_data()),
			typename Traits::SmemLayoutW{});
		auto mX = tma_x.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.tokens),
			static_cast<std::int64_t>(params.hidden)));
		auto mW = tma_weight.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.local_vocab),
			static_cast<std::int64_t>(params.hidden)));

		typename Traits::TiledMma tiled_mma;
		auto cta_mma = tiled_mma.get_slice(cluster_rank);
		typename Traits::ClusterLayoutVMNK pair_layout_vmnk;
		auto pair_coord_vmnk =
			pair_layout_vmnk.get_flat_coord(cluster_rank);
		std::uint16_t mcast_mask_x =
			create_tma_multicast_mask<2>(
				pair_layout_vmnk, pair_coord_vmnk);
		std::uint16_t mcast_mask_w =
			create_tma_multicast_mask<1>(
				pair_layout_vmnk, pair_coord_vmnk);

		for (int item = cluster_index;
				item < total_items;
				item += num_clusters) {
			int m_pair = item % num_m_pairs;
			int n_tile = item / num_m_pairs;
			auto coord = make_coord(m_pair, n_tile, _);
			auto gX = local_tile(
				mX,
				typename Traits::TileShape{},
				coord,
				Step<_1, X, _1>{});
			auto gW = local_tile(
				mW,
				typename Traits::TileShape{},
				coord,
				Step<X, _1, _1>{});
			auto tCgX = cta_mma.partition_A(gX);
			auto tCgW = cta_mma.partition_B(gW);
			auto [tXgX, tXsX] = tma_partition(
				tma_x,
				get<2>(pair_coord_vmnk),
				make_layout(size<2>(pair_layout_vmnk)),
				group_modes<0, 3>(sX),
				group_modes<0, 3>(tCgX));
			auto [tWgW, tWsW] = tma_partition(
				tma_weight,
				get<1>(pair_coord_vmnk),
				make_layout(size<1>(pair_layout_vmnk)),
				group_modes<0, 3>(sW),
				group_modes<0, 3>(tCgW));

			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.producer_acquire(state);
				if (cute::elect_one_sync()) {
					auto* barrier = pipe.producer_get_barrier(state);
					copy(
						tma_x.with(*barrier, mcast_mask_x),
						tXgX(_, k_tile),
						tXsX(_, state.index()));
					copy(
						tma_weight.with(*barrier, mcast_mask_w),
						tWgW(_, k_tile),
						tWsW(_, state.index()));
				}
				++state;
			}
		}
		pipe.producer_tail(state);
		cutlass::arch::NamedBarrier::sync(
			Config::kComputeThreads,
			Config::kArenaHandoffBarrierId);
		return;
	}

	if (warp_id < Config::kUmmaWarp) {
		return;
	}

	bool is_mma_warp = warp_id == Config::kUmmaWarp;
	bool is_leader_cta = cluster_rank == 0;
	bool is_epilogue =
		warp_id >= Config::kFirstEpilogueWarp &&
		warp_id <= Config::kLastEpilogueWarp;
	int tid_in_epi = static_cast<int>(threadIdx.x) -
		Config::kFirstEpilogueWarp * Config::kWarpSize;
	int warpgroup = is_epilogue
		? tid_in_epi / Config::kWarpgroupSize
		: 0;
	int tid_in_warpgroup = is_epilogue
		? tid_in_epi % Config::kWarpgroupSize
		: 0;
	int warpgroup_barrier =
		Config::kWarpgroup0BarrierId + warpgroup;

	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp && is_leader_cta
		? AccPipe::ThreadCategory::Producer
		: AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = Config::kClusterM;
	acc_params.initializing_warp = Config::kFirstEpilogueWarp;
	AccPipe acc_pipe(
		smem.acc_pipe,
		acc_params,
		typename Traits::ClusterShape{});
	auto acc_prod_state =
		cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;
	cutlass::arch::NamedBarrier::sync(
		Config::kMmaEpilogueThreads,
		Config::kMmaEpilogueBarrierId);

	typename Traits::TiledMma tiled_mma;
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	auto sX = make_tensor(
		make_smem_ptr(smem.x_data()),
		typename Traits::SmemLayoutX{});
	auto sW = make_tensor(
		make_smem_ptr(smem.w_data()),
		typename Traits::SmemLayoutW{});
	auto tCrX = cta_mma.make_fragment_A(sX);
	auto tCrW = cta_mma.make_fragment_B(sW);
	auto cAccFull = make_identity_tensor(
		make_shape(Int<Config::kTileM>{}, Int<Config::kTileN>{}));
	auto tCtAcc =
		cta_mma.make_fragment_C(cta_mma.partition_C(cAccFull));

	if (is_mma_warp && is_leader_cta) {
		for (int item = cluster_index;
				item < total_items;
				item += num_clusters) {
			acc_pipe.producer_acquire(acc_prod_state);
			tCtAcc.data() = tmem_base +
				static_cast<std::uint32_t>(
					acc_prod_state.index() *
					Config::kTmemStageColumns);
			bool first = true;
			for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
				pipe.consumer_wait(state);
				CUTE_UNROLL
				for (int k_block = 0; k_block < size<2>(tCrX); ++k_block) {
					tiled_mma.accumulate_ = first
						? UMMA::ScaleOut::Zero
						: UMMA::ScaleOut::One;
					first = false;
					gemm(
						tiled_mma,
						tCrX(_, _, k_block, state.index()),
						tCrW(_, _, k_block, state.index()),
						tCtAcc);
				}
				pipe.consumer_release(state);
				++state;
			}
			acc_pipe.producer_commit(acc_prod_state);
			++acc_prod_state;
		}
	}

	if (is_epilogue) {
		auto sStore0 = make_tensor(
			make_smem_ptr(smem.store_data(warpgroup, 0)),
			typename Traits::SmemLayoutStoreSlot{});
		auto sStore1 = make_tensor(
			make_smem_ptr(smem.store_data(
				warpgroup,
				Config::kStoreBuffers == 2 ? 1 : 0)),
			typename Traits::SmemLayoutStoreSlot{});
		auto mOutput = tma_output.get_tma_tensor(make_shape(
			static_cast<std::int64_t>(params.padded_tokens),
			static_cast<std::int64_t>(params.padded_vocab)));
		auto cta_store = tma_output.get_slice(Int<0>{});
		auto tSsS0 = cta_store.partition_S(sStore0);
		auto tSsS1 = cta_store.partition_S(sStore1);
		auto tSgS = cta_store.partition_D(local_tile(
			mOutput,
			make_tile(
				Int<Config::kCtaTileM>{},
				Int<Config::kEpilogueChunkN>{}),
			make_coord(_, _)));
		for (int item = cluster_index;
				item < total_items;
				item += num_clusters) {
			int m_pair = item % num_m_pairs;
			int n_tile = item / num_m_pairs;
			int m_tile =
				m_pair * Config::kClusterM + cluster_rank;

			if constexpr (Mode != EpilogueMode::kRawGemm) {
				for (int row = tid_in_epi; row < Config::kCtaTileM;
						row += Config::kEpilogueThreads) {
					int global_row =
						m_tile * Config::kCtaTileM + row;
					float scale = 0.0f;
					float exp_bias = 0.0f;
					float entropy_bias = 0.0f;
					float entropy_slope = 0.0f;
					int target_local = -1;
					if (global_row < params.tokens) {
						std::int64_t target_id =
							params.target[global_row];
						if (target_id != params.ignore_index) {
							float lse = params.lse[global_row];
							scale = params.grad_output[global_row];
							exp_bias = -lse * kLog2E;
							if constexpr (
								Mode ==
								EpilogueMode::
									kSoftmaxGradientEntropy) {
								float entropy =
									params.entropy[global_row];
								float entropy_scale =
									params.entropy_grad[global_row];
								entropy_bias = fmaf(
									lse - entropy,
									entropy_scale,
									scale);
								entropy_slope =
									-params.inverse_temperature *
									entropy_scale;
							}
							std::int64_t local =
								target_id - params.vocab_start;
							if (
								local >= 0 &&
								local < params.local_vocab) {
								target_local =
									static_cast<int>(local);
							}
						}
					}
					smem.row_scale[row] = scale;
					smem.row_exp_bias[row] = exp_bias;
					smem.row_target[row] = target_local;
					if constexpr (
						Mode ==
						EpilogueMode::
							kSoftmaxGradientEntropy) {
						smem.row_entropy_bias[row] = entropy_bias;
						smem.row_entropy_slope[row] = entropy_slope;
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kEpilogueThreads,
					Config::kEpilogueBarrierId);
			}

			acc_pipe.consumer_wait(acc_cons_state);
			tCtAcc.data() = tmem_base +
				static_cast<std::uint32_t>(
					acc_cons_state.index() *
					Config::kTmemStageColumns);
			auto epi_tile = make_tile(
				Int<Config::kCtaTileM>{},
				Int<Config::kEpilogueChunkN>{});
			auto acc_mn =
				tCtAcc(make_coord(_, _), _0{}, _0{});
			auto tAccEpi = flat_divide(acc_mn, epi_tile);
			auto t2r = make_tmem_copy(
				::liger::TmemLoadOp<Config::kEpilogueChunkN>{},
				tAccEpi(_, _, _0{}, _0{}));
			auto thr_t2r = t2r.get_slice(tid_in_warpgroup);
			auto cChunk = make_identity_tensor(make_shape(
				Int<Config::kCtaTileM>{},
				Int<Config::kEpilogueChunkN>{}));
			auto tTR_cChunk = thr_t2r.partition_D(cChunk);
			auto tTR_rAcc =
				make_tensor<float>(shape(tTR_cChunk));
			Layout tmem_warp_layout =
				typename decltype(make_tmem_warp_partitioner(
					tAccEpi(_, _, _0{}, _0{})))::TiledLayout_TV{};
			constexpr bool kPredicateTmemLoad =
				size(tmem_warp_layout) != cosize(tmem_warp_layout);

			float exp_scale =
				params.inverse_temperature * kLog2E;
			CUTE_UNROLL
			for (int round = 0;
					round < Config::kChunksPerWarpgroup;
					++round) {
				int chunk =
					warpgroup *
						Config::kChunksPerWarpgroup +
					round;
				auto tAccChunk =
					thr_t2r.partition_S(tAccEpi)(
						_, _, _, _0{}, chunk);
				bool issue_tmem_load = true;
				if constexpr (kPredicateTmemLoad) {
					int subpart =
						(tAccChunk.data().dp_ / 32) % 4;
					issue_tmem_load =
						tid_in_warpgroup /
							Config::kWarpSize ==
						subpart;
				}
				int buffer =
					Config::kStoreBuffers == 2 ? (round & 1) : 0;
				if (issue_tmem_load) {
					copy(t2r, tAccChunk, tTR_rAcc);
					cutlass::arch::
						fence_view_async_tmem_load();
				}
				if (tid_in_warpgroup == 0) {
					if constexpr (Config::kStoreBuffers == 2) {
						cute::tma_store_wait<1>();
					} else {
						cute::tma_store_wait<0>();
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kWarpgroupSize,
					warpgroup_barrier);

				int vocab_base =
					n_tile * Config::kTileN +
					chunk * Config::kEpilogueChunkN;
				if (issue_tmem_load) {
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rAcc); ++i) {
						int row = get<0>(tTR_cChunk(i));
						int column = get<1>(tTR_cChunk(i));
						int global_row =
							m_tile * Config::kCtaTileM +
							row;
						int global_column =
							vocab_base + column;
						float logit = tTR_rAcc(i);
						float value = 0.0f;
						if (
							global_row < params.tokens &&
							global_column <
								params.local_vocab) {
							if constexpr (
								Mode ==
								EpilogueMode::kRawGemm) {
								value = logit;
							} else {
								float scale =
									smem.row_scale[row];
								float probability = 0.0f;
								if constexpr (
									Mode ==
									EpilogueMode::
										kSoftmaxGradientEntropy) {
									// dZ = invT * (p * (g + (lse-H)gH
									// - invT*z*gH) - g*one_hot).
									float slope =
										smem.row_entropy_slope[row];
									if (
										scale != 0.0f ||
										slope != 0.0f) {
										probability = fast_exp2(fmaf(
											logit,
											exp_scale,
											smem.row_exp_bias[row]));
										value = probability * fmaf(
											logit,
											slope,
											smem.row_entropy_bias[row]);
									}
								} else if (scale != 0.0f) {
									probability = fast_exp2(fmaf(
										logit,
										exp_scale,
										smem.row_exp_bias[row]));
									value = probability * scale;
								}
								if (
									global_column ==
									smem.row_target[row]) {
									value -= scale;
								}
								value *=
									params.inverse_temperature;
							}
						}
						(buffer == 0 ? sStore0 : sStore1)(
							row,
							column) =
							static_cast<Element>(value);
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kWarpgroupSize,
					warpgroup_barrier);
				if (tid_in_warpgroup == 0) {
					cute::tma_store_fence();
					copy(
						tma_output,
						buffer == 0 ? tSsS0 : tSsS1,
						tSgS(
							_,
							_,
							_,
							m_tile,
							n_tile *
									(Config::kTileN /
										Config::kEpilogueChunkN) +
								chunk));
					cute::tma_store_arrive();
				}
			}
			cutlass::arch::NamedBarrier::sync(
				Config::kEpilogueThreads,
				Config::kEpilogueBarrierId);
			if (tid_in_epi == 0) {
				acc_pipe.consumer_release(acc_cons_state);
			}
			++acc_cons_state;
		}
		if (tid_in_warpgroup == 0) {
			cute::tma_store_wait<0>();
		}
		cutlass::arch::NamedBarrier::sync(
			Config::kEpilogueThreads,
			Config::kEpilogueBarrierId);
		if (tid_in_epi == 0) {
			asm volatile("fence.proxy.async.global;" ::: "memory");
		}
	}
	cutlass::arch::NamedBarrier::sync(
		Config::kComputeThreads,
		Config::kArenaHandoffBarrierId);

	cutlass::arch::NamedBarrier::sync(
		Config::kMmaEpilogueThreads,
		Config::kMmaEpilogueBarrierId);
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.release_allocation_lock();
		tmem_pair_barrier_sync(smem.tmem_pair_barrier, warp_id);
		tmem_allocator.free(smem.tmem_base, Config::kTmemColumns);
	}
	cutlass::arch::NamedBarrier::sync(
		Config::kMmaEpilogueThreads,
		Config::kTmemFreeBarrierId);
#else
	__trap();
#endif
}

template <EpilogueMode Mode, class Bundle>
__global__ __launch_bounds__(Config::kNumThreads, 1) __cluster_dims__(2, 1, 1)
void kernel(
		__grid_constant__ const Bundle tma,
		__grid_constant__ const Params params) {
	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<SharedStorage<Mode>*>(raw_smem);
	kernel_body<Mode>(
		tma.x,
		tma.weight,
		tma.output,
		params,
		smem);
}

inline auto make_tma_bundle(
		const Traits::Element* x,
		const Traits::Element* weight,
		Traits::Element* output,
		int tokens,
		int hidden,
		int local_vocab,
		int padded_tokens,
		int padded_vocab) {
	auto tensor_x = make_tensor(
		make_gmem_ptr(x),
		make_shape(
			static_cast<std::int64_t>(tokens),
			static_cast<std::int64_t>(hidden)),
		make_stride(static_cast<std::int64_t>(hidden), Int<1>{}));
	auto tensor_w = make_tensor(
		make_gmem_ptr(weight),
		make_shape(
			static_cast<std::int64_t>(local_vocab),
			static_cast<std::int64_t>(hidden)),
		make_stride(static_cast<std::int64_t>(hidden), Int<1>{}));
	auto tma_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_x,
		typename Traits::SmemLayoutX1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});
	auto tma_w = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_w,
		typename Traits::SmemLayoutW1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma{});
	auto tensor_output = make_tensor(
		make_gmem_ptr(output),
		make_shape(
			static_cast<std::int64_t>(padded_tokens),
			static_cast<std::int64_t>(padded_vocab)),
		make_stride(static_cast<std::int64_t>(padded_vocab), Int<1>{}));
	auto tma_output = make_tma_copy(
		SM90_TMA_STORE{},
		tensor_output,
		typename Traits::SmemLayoutStoreSlot{});
	return TmaBundle<
		decltype(tma_x),
		decltype(tma_w),
		decltype(tma_output)>{tma_x, tma_w, tma_output};
}

struct KernelResources {
	int tile_m = Config::kTileM;
	int tile_n = Config::kTileN;
	int tile_k = Config::kTileK;
	int cluster_m = Config::kClusterM;
	int mainloop_stages = Config::kMainloopStages;
	int tmem_columns = Config::kTmemColumns;
	int threads = Config::kNumThreads;
	int dynamic_smem_bytes = 0;
	int registers_per_thread = 0;
	std::size_t local_bytes_per_thread = 0;
	int max_active_cluster_pairs = 0;
	int launched_cluster_pairs = 0;
};

template <EpilogueMode Mode>
inline cudaError_t launch(
		const Params& params,
		cudaStream_t stream,
		KernelResources* resources = nullptr) {
	using Smem = SharedStorage<Mode>;
	using Element = typename Traits::Element;

	if (
		params.x == nullptr ||
		params.weight == nullptr ||
		params.output == nullptr ||
		params.tokens <= 0 ||
		params.hidden <= 0 ||
		params.hidden % 8 != 0 ||
		params.local_vocab <= 0 ||
		params.padded_tokens <
			ceil_div_int(params.tokens, Config::kTileM) *
				Config::kTileM ||
		params.padded_vocab <
			ceil_div_int(params.local_vocab, Config::kTileN) *
				Config::kTileN) {
		return cudaErrorInvalidValue;
	}
	auto aligned_16 = [](const void* pointer) {
		return reinterpret_cast<std::uintptr_t>(pointer) % 16u == 0u;
	};
	if (
		!aligned_16(params.x) ||
		!aligned_16(params.weight) ||
		!aligned_16(params.output)) {
		return cudaErrorInvalidValue;
	}
	if constexpr (Mode != EpilogueMode::kRawGemm) {
		if (
			params.target == nullptr ||
			params.grad_output == nullptr ||
			params.lse == nullptr ||
			params.inverse_temperature <= 0.0f) {
			return cudaErrorInvalidValue;
		}
	}
	if constexpr (Mode == EpilogueMode::kSoftmaxGradientEntropy) {
		if (params.entropy == nullptr || params.entropy_grad == nullptr) {
			return cudaErrorInvalidValue;
		}
	}

	auto bundle = make_tma_bundle(
		static_cast<const Element*>(params.x),
		static_cast<const Element*>(params.weight),
		static_cast<Element*>(params.output),
		params.tokens,
		params.hidden,
		params.local_vocab,
		params.padded_tokens,
		params.padded_vocab);
	auto* kernel_ptr = &kernel<Mode, decltype(bundle)>;
	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));

	cudaError_t status = cudaFuncSetAttribute(
		kernel_ptr,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		kSmemBytes);
	if (status != cudaSuccess) return status;

	int num_m_pairs = ceil_div_int(params.tokens, Config::kTileM);
	int num_n_tiles = ceil_div_int(params.local_vocab, Config::kTileN);
	int total_cluster_items = num_m_pairs * num_n_tiles;

	cudaLaunchAttribute cluster_attribute = {};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = Config::kClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;

	cudaLaunchConfig_t occupancy_config = {};
	occupancy_config.gridDim =
		dim3(static_cast<unsigned>(Config::kClusterM), 1u, 1u);
	occupancy_config.blockDim =
		dim3(static_cast<unsigned>(Config::kNumThreads), 1u, 1u);
	occupancy_config.dynamicSmemBytes = kSmemBytes;
	occupancy_config.stream = stream;
	occupancy_config.attrs = &cluster_attribute;
	occupancy_config.numAttrs = 1;
	int max_clusters = 0;
	status = cudaOccupancyMaxActiveClusters(
		&max_clusters,
		kernel_ptr,
		&occupancy_config);
	if (status != cudaSuccess) return status;
	if (max_clusters <= 0) return cudaErrorLaunchOutOfResources;
	int resident_clusters =
		total_cluster_items < max_clusters
		? total_cluster_items
		: max_clusters;
	if (
		params.cluster_pairs > 0 &&
		params.cluster_pairs < resident_clusters) {
		resident_clusters = params.cluster_pairs;
	}

	cudaLaunchConfig_t launch_config = occupancy_config;
	launch_config.gridDim = dim3(
		static_cast<unsigned>(Config::kClusterM),
		1u,
		static_cast<unsigned>(resident_clusters));

	if (resources != nullptr) {
		cudaFuncAttributes attributes = {};
		status = cudaFuncGetAttributes(&attributes, kernel_ptr);
		if (status != cudaSuccess) return status;
		resources->dynamic_smem_bytes = kSmemBytes;
		resources->registers_per_thread = attributes.numRegs;
		resources->local_bytes_per_thread = attributes.localSizeBytes;

		resources->max_active_cluster_pairs = max_clusters;
		resources->launched_cluster_pairs = resident_clusters;
	}

	return cudaLaunchKernelEx(
		&launch_config,
		kernel_ptr,
		bundle,
		params);
}

}  // namespace dz_sm100
}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
