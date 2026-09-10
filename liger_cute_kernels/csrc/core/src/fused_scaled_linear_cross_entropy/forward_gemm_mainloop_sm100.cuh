#pragma once

#include "forward_gemm_sm100.cuh"
#include "dx_reduce.cuh"
#include "liger_cute/detail/local_reduce.cuh"
#include "tmem_load_op_sm100.cuh"
#if defined(LIGER_CUTE_FSLCE_SM100_ENABLE_NVSHMEM)
#include "forward_remote_reduce.cuh"
#endif

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

#include <cstddef>
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

template <int Compute = 100>
struct ForwardGemmTraitsSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	using Config = ForwardGemmConfigSm100<Compute>;
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;
	using ElementLogit = cutlass::half_t;

	static constexpr int kTileM = Config::kTileM;
	static constexpr int kCtaTileM = Config::kCtaTileM;
	static constexpr int kTileN = Config::kUmmaTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStages = Config::kMainloopStages;
	static constexpr int kAccumulatorStages = Config::kAccumulatorStages;
	static constexpr int kAccumulatorPanels = Config::kAccumulatorPanels;
	static constexpr int kTmemStageColumns = Config::kTmemStageColumns;

	using TileShape =
		Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
	using ClusterShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using AtomThrShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using TiledMma = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			kTileM,
			kTileN,
			UMMA::Major::K,
			UMMA::Major::K>{}));
	static_assert(
		size(typename TiledMma::AtomThrID{}) == Config::kClusterM,
		"SM100 forward MMA atom size must match its CTA group");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma::AtomThrID{})));
	using MmaShapeX = decltype(partition_shape_A(
		TiledMma{},
		make_shape(Int<kTileM>{}, Int<kTileK>{})));
	using MmaShapeW = decltype(partition_shape_B(
		TiledMma{},
		make_shape(Int<kTileN>{}, Int<kTileK>{})));
	using SmemLayoutX = decltype(UMMA::tile_to_mma_shape(
		UMMA::Layout_K_SW128_Atom<Element>{},
		append(MmaShapeX{}, Int<kStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutW = decltype(UMMA::tile_to_mma_shape(
		UMMA::Layout_K_SW128_Atom<Element>{},
		append(MmaShapeW{}, Int<kStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutX1 =
		decltype(SmemLayoutX{}(_, _, _, Int<0>{}));
	using SmemLayoutW1 =
		decltype(SmemLayoutW{}(_, _, _, Int<0>{}));

	static constexpr int kTmaTransBytesX =
		static_cast<int>(cosize_v<SmemLayoutX1> * sizeof(Element));
	static constexpr int kTmaTransBytesW =
		static_cast<int>(cosize_v<SmemLayoutW1> * sizeof(Element));
	// The paired pipeline accounts for both CTAs' complete X and W boxes.
	static constexpr int kTmaTransBytes =
		Config::kClusterM *
		(kTmaTransBytesX + kTmaTransBytesW);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		kAccumulatorStages,
		AtomThrShape>;
};

template <int Compute, bool ReturnEntropy>
struct ForwardGemmSmemSm100 {
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Element = typename Traits::Element;
	using ElementLogit = typename Traits::ElementLogit;

	static constexpr int kSmemX =
		cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int kSmemW =
		cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int kLogitElements =
		Config::kEpilogueWarpgroups *
		Config::kCtaTileM *
		Config::kEpilogueChunkN;
	static constexpr int kStateElements =
		Config::kEpilogueWarpgroups * Config::kCtaTileM;
	static constexpr int kStateFields = ReturnEntropy ? 5 : 4;
	static constexpr int kMaxField = 0;
	static constexpr int kSumField = 1;
	static constexpr int kTargetField = 2;
	static constexpr int kWeightedField = 3;
	static constexpr int kHasTargetField = ReturnEntropy ? 4 : 3;
	static_assert(
		kStateFields * kStateElements * sizeof(float) <=
			kLogitElements * sizeof(ElementLogit),
		"forward epilogue state must fit in the reusable logit scratch");

	alignas(128) Element smem_x[kSmemX];
	alignas(128) Element smem_w[kSmemW];
	alignas(128) ElementLogit logits[kLogitElements];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	int finalizer;

	CUTE_DEVICE Element* x_data() { return &smem_x[0]; }
	CUTE_DEVICE Element* w_data() { return &smem_w[0]; }
	CUTE_DEVICE ElementLogit* logit_row(int warpgroup, int row) {
		return &logits[
			(warpgroup * Config::kCtaTileM + row) *
			Config::kEpilogueChunkN];
	}
	CUTE_DEVICE int state_index(int warpgroup, int row) const {
		return warpgroup * Config::kCtaTileM + row;
	}
	CUTE_DEVICE float& state_float(int field, int index) {
		return reinterpret_cast<float*>(&logits[0])[
			field * kStateElements + index];
	}
	CUTE_DEVICE int& state_int(int field, int index) {
		return reinterpret_cast<int*>(&logits[0])[
			field * kStateElements + index];
	}
};

template <int Compute = 100>
struct ForwardGemmWorkSm100 {
	int m_pair;
	int split_id;
	int split_count;
	int raw_pid_m;
	int output_work;
	bool store_enabled;
};

template <int Compute = 100>
__host__ __device__ constexpr ForwardGemmWorkSm100<Compute>
forward_gemm_assign_work_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int cluster_work,
		int cluster_rank) {
	using Config = ForwardGemmConfigSm100<Compute>;
	int m_pair;
	int split_id;
	int split_count;
	int base_work = split.base_split_n * split.num_m_pairs;
	if (cluster_work < base_work) {
		split_id = cluster_work / split.num_m_pairs;
		m_pair = cluster_work % split.num_m_pairs;
		split_count = split.split_count_for_pair(m_pair);
	} else {
		split_id = split.base_split_n;
		m_pair = cluster_work - base_work;
		split_count = split.split_n;
	}

	ForwardGemmWorkSm100<Compute> work;
	work.m_pair = m_pair;
	work.split_id = split_id;
	work.split_count = split_count;
	work.raw_pid_m =
		m_pair * Config::kClusterM + cluster_rank;
	work.output_work =
		work.raw_pid_m * split.split_n + split_id;
	work.store_enabled =
		work.raw_pid_m < split.num_m_tiles;
	return work;
}

template <int Compute = 100>
__host__ __device__ constexpr int forward_wave_split_first_tile_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int wave,
		int split_id,
		int split_count) {
	int begin = forward_wave_begin_tile_sm100(wave);
	int end = forward_wave_end_tile_sm100(
		wave, split.num_logical_n_tiles);
	int remainder = begin % split_count;
	int delta = split_id - remainder;
	if (delta < 0) delta += split_count;
	int first = begin + delta;
	return first < end ? first : end;
}

template <int Compute = 100>
__host__ __device__ constexpr int forward_wave_split_tile_count_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int wave,
		int split_id,
		int split_count) {
	int first = forward_wave_split_first_tile_sm100(
		split, wave, split_id, split_count);
	int end = forward_wave_end_tile_sm100(
		wave, split.num_logical_n_tiles);
	return first < end
		? 1 + (end - 1 - first) / split_count
		: 0;
}

template <int Compute = 100>
__host__ __device__ constexpr std::size_t
forward_wave_partial_row_offset_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int slot,
		int m_tile,
		int split_id) {
	using Config = ForwardGemmConfigSm100<Compute>;
	return (
		(static_cast<std::size_t>(slot) *
				static_cast<std::size_t>(split.num_m_tiles) +
			static_cast<std::size_t>(m_tile)) *
				static_cast<std::size_t>(split.split_n) +
		static_cast<std::size_t>(split_id)) *
		static_cast<std::size_t>(Config::kCtaTileM);
}

template <int Compute = 100>
__host__ __device__ constexpr std::size_t
forward_wave_partial_ready_offset_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int slot,
		int m_tile,
		int split_id) {
	return (
		static_cast<std::size_t>(slot) *
				static_cast<std::size_t>(split.num_m_tiles) +
			static_cast<std::size_t>(m_tile)) *
			static_cast<std::size_t>(split.split_n) +
		static_cast<std::size_t>(split_id);
}

template <int Compute = 100>
__host__ __device__ constexpr std::size_t
forward_wave_tile_ready_offset_sm100(
		const ForwardGemmSplitSm100<Compute>& split,
		int slot,
		int m_tile) {
	return static_cast<std::size_t>(slot) *
			static_cast<std::size_t>(split.num_m_tiles) +
		static_cast<std::size_t>(m_tile);
}

CUTE_DEVICE std::uint64_t forward_load_acquire_system_sm100(
		const std::uint64_t* address) {
	std::uint64_t value = 0;
#if defined(__CUDA_ARCH__)
	asm volatile(
		"ld.acquire.sys.global.u64 %0, [%1];"
		: "=l"(value)
		: "l"(address)
		: "memory");
#endif
	return value;
}

CUTE_DEVICE void forward_store_release_system_sm100(
		std::uint64_t* address, std::uint64_t value) {
#if defined(__CUDA_ARCH__)
	asm volatile(
		"st.release.sys.global.u64 [%0], %1;"
		:
		: "l"(address), "l"(value)
		: "memory");
#else
	(void)address;
	(void)value;
#endif
}

CUTE_DEVICE std::uint64_t forward_globaltimer_sm100() {
	std::uint64_t value = 0;
#if defined(__CUDA_ARCH__)
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
#endif
	return value;
}

CUTE_DEVICE void forward_diagnostic_min_sm100(
		std::uint64_t* diagnostics, int index, std::uint64_t value) {
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicMin(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				static_cast<unsigned long long>(value));
		}
	}
}

CUTE_DEVICE void forward_diagnostic_max_sm100(
		std::uint64_t* diagnostics, int index, std::uint64_t value) {
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicMax(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				static_cast<unsigned long long>(value));
		}
	}
}

CUTE_DEVICE void forward_diagnostic_add_sm100(
		std::uint64_t* diagnostics, int index, std::uint64_t value) {
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicAdd(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				static_cast<unsigned long long>(value));
		}
	}
}

__host__ __device__ constexpr int forward_diagnostic_wave_start_sm100(
		int wave) {
	return kForwardDiagnosticWaveBaseSm100 +
		wave * kForwardDiagnosticWaveStrideSm100;
}

__host__ __device__ constexpr int forward_diagnostic_wave_end_sm100(
		int wave) {
	return forward_diagnostic_wave_start_sm100(wave) + 1;
}

CUTE_DEVICE void forward_wait_epoch_warp_sm100(
		const std::uint64_t* address, std::uint64_t epoch) {
	unsigned int active = __activemask();
	for (;;) {
		bool ready =
			forward_load_acquire_system_sm100(address) == epoch;
		if (__all_sync(active, ready)) break;
#if defined(__CUDA_ARCH__)
		__nanosleep(64);
#endif
	}
	__syncwarp(active);
}

template <int Compute = 100>
CUTE_DEVICE void forward_wait_partial_epochs_warp_sm100(
		const ForwardGemmPartialsSm100<Compute>& partials,
		const ForwardGemmSplitSm100<Compute>& split,
		int slot,
		int m_tile,
		int split_count,
		std::uint64_t epoch) {
	int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	unsigned int active = __activemask();
	for (;;) {
		bool ready = true;
		for (int split_id = lane;
				split_id < split_count;
				split_id += kWarpSize) {
			std::size_t offset =
				forward_wave_partial_ready_offset_sm100(
					split, slot, m_tile, split_id);
			ready =
				forward_load_acquire_system_sm100(
					partials.ready + offset) == epoch;
		}
		if (__all_sync(active, ready)) break;
#if defined(__CUDA_ARCH__)
		__nanosleep(64);
#endif
	}
	__syncwarp(active);
}

template <int Compute = 100>
CUTE_DEVICE void forward_wait_tile_epochs_warp_sm100(
		const ForwardWaveWorkspaceSm100<Compute>& wave_workspace,
		const ForwardGemmSplitSm100<Compute>& split,
		int slot,
		std::uint64_t epoch) {
	int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	unsigned int active = __activemask();
	for (;;) {
		bool ready = true;
		for (int m_tile = lane;
				m_tile < split.num_m_tiles;
				m_tile += kWarpSize) {
			std::size_t offset =
				forward_wave_tile_ready_offset_sm100(
					split, slot, m_tile);
			ready =
				forward_load_acquire_system_sm100(
					wave_workspace.tile_ready + offset) ==
				epoch;
		}
		if (__all_sync(active, ready)) break;
#if defined(__CUDA_ARCH__)
		__nanosleep(64);
#endif
	}
	__syncwarp(active);
}

template <int Compute = 100>
CUTE_DEVICE void forward_wait_source_slot_sm100(
		const ForwardWaveWorkspaceSm100<Compute>& wave_workspace,
		const std::uint64_t* launch_epoch,
		int wave) {
	int reused_wave = forward_wave_reused_wave_sm100(wave);
	if (reused_wave < 0) return;
	int slot = forward_wave_slot_sm100(wave);
	forward_wait_epoch_warp_sm100(
		wave_workspace.slot_released + slot,
		forward_wave_epoch_sm100(*launch_epoch, reused_wave));
}

template <int Compute = 100>
CUTE_DEVICE typename ForwardGemmTraitsSm100<
	Compute>::MainloopPipeline
forward_make_pipe_sm100(
		typename ForwardGemmTraitsSm100<
			Compute>::MainloopPipeline::SharedStorage& storage) {
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
	typename Pipeline::Params params;
	params.transaction_bytes = Traits::kTmaTransBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = Config::kTmaWarp;
	if (warp_id == Config::kTmaWarp) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == Config::kTmaWarp * kWarpSize &&
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

template <int Compute = 100>
struct ForwardGemmProducerSm100 {
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;

	template <
		bool ReturnEntropy,
		bool WavePipeline,
		class TmaLoadX,
		class TmaLoadW>
	CUTE_DEVICE static void run(
			typename Traits::MainloopPipeline& pipe,
			typename Traits::PipelineState& state,
			ForwardGemmSmemSm100<
				Compute,
				ReturnEntropy>& smem,
			const TmaLoadX& tma_load_x,
			const TmaLoadW& tma_load_w,
			const ForwardGemmParamsSm100<Compute>& params,
			const ForwardGemmWorkSm100<
				Compute>& work,
			const ForwardGemmSplitSm100<
				Compute>& split,
			const ForwardWaveWorkspaceSm100<
				Compute>& wave_workspace,
			int num_k_tiles) {
		run_impl<WavePipeline>(
			pipe,
			state,
			smem,
			tma_load_x,
			tma_load_w,
			params,
			work,
			split,
			wave_workspace,
			num_k_tiles);
	}

private:
	template <
		bool WavePipeline,
		class Smem,
		class TmaLoadX,
		class TmaLoadW>
	CUTE_DEVICE static void run_impl(
			typename Traits::MainloopPipeline& pipe,
			typename Traits::PipelineState& state,
			Smem& smem,
			const TmaLoadX& tma_load_x,
			const TmaLoadW& tma_load_w,
			const ForwardGemmParamsSm100<Compute>& params,
			const ForwardGemmWorkSm100<
				Compute>& work,
			const ForwardGemmSplitSm100<
				Compute>& split,
			const ForwardWaveWorkspaceSm100<
				Compute>& wave_workspace,
			int num_k_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
		auto sX = make_tensor(
			make_smem_ptr(smem.x_data()),
			typename Traits::SmemLayoutX{});
		auto sW = make_tensor(
			make_smem_ptr(smem.w_data()),
			typename Traits::SmemLayoutW{});
		auto mX = tma_load_x.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.tokens),
			static_cast<int64_t>(params.hidden)));
		auto mW = tma_load_w.get_tma_tensor(make_shape(
			static_cast<int64_t>(params.local_vocab),
			static_cast<int64_t>(params.hidden)));

		typename Traits::TiledMma tiled_mma;
		int cluster_rank =
			static_cast<int>(cute::block_rank_in_cluster());
		auto cta_mma = tiled_mma.get_slice(cluster_rank);
		typename Traits::ClusterLayoutVMNK pair_layout_vmnk;
		auto pair_coord_vmnk =
			pair_layout_vmnk.get_flat_coord(cluster_rank);
		uint16_t mcast_mask_x =
			create_tma_multicast_mask<2>(
				pair_layout_vmnk,
				pair_coord_vmnk);
		uint16_t mcast_mask_w =
			create_tma_multicast_mask<1>(
				pair_layout_vmnk,
				pair_coord_vmnk);

		[[maybe_unused]] std::uint64_t producer_wait_total = 0;
		int wave_count = WavePipeline ? split.num_waves : 1;
		for (int wave = 0; wave < wave_count; ++wave) {
			if constexpr (WavePipeline) {
				[[maybe_unused]] std::uint64_t wait_begin = 0;
				if constexpr (kForwardDiagnosticTimestampsSm100) {
					if (
						(threadIdx.x & (kWarpSize - 1)) == 0 &&
						forward_wave_reused_wave_sm100(wave) >= 0) {
						wait_begin = forward_globaltimer_sm100();
					}
				}
				forward_wait_source_slot_sm100(
					wave_workspace,
					wave_workspace.launch_epoch,
					wave);
				if constexpr (kForwardDiagnosticTimestampsSm100) {
					if (
						(threadIdx.x & (kWarpSize - 1)) == 0 &&
						forward_wave_reused_wave_sm100(wave) >= 0) {
						producer_wait_total +=
							forward_globaltimer_sm100() -
							wait_begin;
					}
				}
			}
			int first_logical_n = WavePipeline
				? forward_wave_split_first_tile_sm100(
					split,
					wave,
					work.split_id,
					work.split_count)
				: work.split_id;
			int num_split_tiles = WavePipeline
				? forward_wave_split_tile_count_sm100(
					split,
					wave,
					work.split_id,
					work.split_count)
				: ceil_div(
					split.num_logical_n_tiles -
						work.split_id,
					work.split_count);
			for (int local_n = 0;
					local_n < num_split_tiles;
					++local_n) {
				int logical_n =
					first_logical_n +
					local_n * work.split_count;
			for (int panel = 0;
					panel < Config::kAccumulatorPanels;
					++panel) {
				int n_tile =
					logical_n * Config::kAccumulatorPanels +
					panel;
				auto coord =
					make_coord(work.m_pair, n_tile, _);
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
					tma_load_x,
					get<2>(pair_coord_vmnk),
					make_layout(size<2>(pair_layout_vmnk)),
					group_modes<0, 3>(sX),
					group_modes<0, 3>(tCgX));
				auto [tWgW, tWsW] = tma_partition(
					tma_load_w,
					get<1>(pair_coord_vmnk),
					make_layout(size<1>(pair_layout_vmnk)),
					group_modes<0, 3>(sW),
					group_modes<0, 3>(tCgW));
				for (int k_tile = 0;
						k_tile < num_k_tiles;
						++k_tile) {
					pipe.producer_acquire(state);
					if (cute::elect_one_sync()) {
						auto* barrier =
							pipe.producer_get_barrier(state);
						copy(
							tma_load_x.with(
								*barrier,
								mcast_mask_x),
							tXgX(_, k_tile),
							tXsX(_, state.index()));
						copy(
							tma_load_w.with(
								*barrier,
								mcast_mask_w),
							tWgW(_, k_tile),
							tWsW(_, state.index()));
					}
					++state;
				}
			}
		}
		}
		if constexpr (
			WavePipeline &&
			kForwardDiagnosticTimestampsSm100) {
			if ((threadIdx.x & (kWarpSize - 1)) == 0) {
				forward_diagnostic_max_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticProducerWaitSm100,
					producer_wait_total);
			}
		}
		pipe.producer_tail(state);
#else
		__trap();
#endif
	}
};

template <
	bool ReturnEntropy,
	bool WavePipeline,
	int Compute = 100>
struct ForwardGemmConsumerSm100 {
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Smem =
		ForwardGemmSmemSm100<
			Compute,
			ReturnEntropy>;
	using Epilogue = ForwardGemmEpilogueSm100<Compute>;

	CUTE_DEVICE static void run(
			typename Traits::MainloopPipeline& pipe,
			typename Traits::PipelineState& state,
			Smem& smem,
			const ForwardGemmParamsSm100<Compute>& params,
			const ForwardGemmPartialsSm100<
				Compute>& partials,
			const ForwardGemmWorkSm100<
				Compute>& work,
			const ForwardGemmSplitSm100<
				Compute>& split,
			const ForwardWaveWorkspaceSm100<
				Compute>& wave_workspace,
			int num_k_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
		int cta_rank =
			static_cast<int>(cute::block_rank_in_cluster());
		bool is_leader_cta = cta_rank == 0;
		int warp_id =
			static_cast<int>(threadIdx.x) / kWarpSize;
		bool is_mma_warp =
			warp_id == Config::kUmmaWarp;
		bool is_epilogue =
			warp_id >= Config::kFirstEpilogueWarp &&
			warp_id <= Config::kLastEpilogueWarp;
		int tid_in_epi = static_cast<int>(threadIdx.x) -
			Config::kFirstEpilogueWarp * kWarpSize;
		int warpgroup = is_epilogue
			? tid_in_epi / Config::kWarpgroupSize
			: 0;
		int tid_in_warpgroup = is_epilogue
			? tid_in_epi % Config::kWarpgroupSize
			: 0;
		int warpgroup_barrier =
			Config::kWarpgroup0BarrierId + warpgroup;
		constexpr int kMmaEpilogueThreads =
			Config::kEpilogueThreads + kWarpSize;

		typename Traits::TiledMma tiled_mma;
		auto cta_mma = tiled_mma.get_slice(cta_rank);
		auto sX = make_tensor(
			make_smem_ptr(smem.x_data()),
			typename Traits::SmemLayoutX{});
		auto sW = make_tensor(
			make_smem_ptr(smem.w_data()),
			typename Traits::SmemLayoutW{});
		auto tCrX = cta_mma.make_fragment_A(sX);
		auto tCrW = cta_mma.make_fragment_B(sW);
		auto cAccFull = make_identity_tensor(
			make_shape(
				Int<Config::kTileM>{},
				Int<Config::kUmmaTileN>{}));
		auto tCgC = cta_mma.partition_C(cAccFull);
		auto tCtAcc = cta_mma.make_fragment_C(tCgC);

		using AccPipe = typename Traits::AccumulatorPipeline;
		typename AccPipe::Params acc_params;
		acc_params.role =
			is_mma_warp && is_leader_cta
			? AccPipe::ThreadCategory::Producer
			: AccPipe::ThreadCategory::Consumer;
		acc_params.producer_arv_count = 1;
		acc_params.consumer_arv_count = Config::kClusterM;
		acc_params.initializing_warp =
			Config::kFirstEpilogueWarp;
		AccPipe acc_pipe(
			smem.acc_pipe,
			acc_params,
			typename Traits::ClusterShape{});
		auto acc_prod_state =
			cutlass::make_producer_start_state<AccPipe>();
		typename AccPipe::PipelineState acc_cons_state;

		if (
			warp_id >= Config::kUmmaWarp &&
			warp_id <= Config::kLastEpilogueWarp) {
			cutlass::arch::NamedBarrier::sync(
				kMmaEpilogueThreads,
				Config::kMmaEpilogueBarrierId);
		}
		uint32_t tmem_base = smem.tmem_base;
		tCtAcc.data() = tmem_base;

		auto epi_tile = make_tile(
			Int<Config::kCtaTileM>{},
			Int<Config::kEpilogueChunkN>{});
		auto acc_mn =
			tCtAcc(make_coord(_, _), _0{}, _0{});
		auto tAccEpi = flat_divide(acc_mn, epi_tile);
		auto t2r = make_tmem_copy(
			::liger::TmemLoadOp<
				Config::kEpilogueChunkN>{},
			tAccEpi(_, _, _0{}, _0{}));
		auto thr_t2r =
			t2r.get_slice(tid_in_warpgroup);
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
			size(tmem_warp_layout) !=
			cosize(tmem_warp_layout);

		int wave_count = WavePipeline ? split.num_waves : 1;
		for (int wave = 0; wave < wave_count; ++wave) {
			if constexpr (WavePipeline) {
				forward_wait_source_slot_sm100(
					wave_workspace,
					wave_workspace.launch_epoch,
					wave);
			}
			int first_logical_n = WavePipeline
				? forward_wave_split_first_tile_sm100(
					split,
					wave,
					work.split_id,
					work.split_count)
				: work.split_id;
			int num_split_tiles = WavePipeline
				? forward_wave_split_tile_count_sm100(
					split,
					wave,
					work.split_id,
					work.split_count)
				: ceil_div(
					split.num_logical_n_tiles -
						work.split_id,
					work.split_count);
			OnlineSoftmaxState softmax;
			softmax.max_value = kForwardNegInfSm100;

			for (int local_n = 0;
					local_n < num_split_tiles;
					++local_n) {
				int logical_n =
					first_logical_n +
					local_n * work.split_count;
				if (is_mma_warp && is_leader_cta) {
				acc_pipe.producer_acquire(acc_prod_state);
				int acc_stage = acc_prod_state.index();
				uint32_t stage_base =
					tmem_base +
					static_cast<uint32_t>(
						acc_stage *
						Config::kTmemStageColumns);
				for (int panel = 0;
						panel < Config::kAccumulatorPanels;
						++panel) {
					tCtAcc.data() =
						stage_base +
						static_cast<uint32_t>(
							panel *
							Config::kUmmaTileN);
					bool first = true;
					for (int k_tile = 0;
							k_tile < num_k_tiles;
							++k_tile) {
						pipe.consumer_wait(state);
						CUTE_UNROLL
						for (int k_block = 0;
								k_block <
									size<2>(tCrX);
								++k_block) {
							tiled_mma.accumulate_ =
								first
								? UMMA::ScaleOut::Zero
								: UMMA::ScaleOut::One;
							first = false;
							gemm(
								tiled_mma,
								tCrX(
									_,
									_,
									k_block,
									state.index()),
								tCrW(
									_,
									_,
									k_block,
									state.index()),
								tCtAcc);
						}
						pipe.consumer_release(state);
						++state;
					}
				}
				acc_pipe.producer_commit(acc_prod_state);
				++acc_prod_state;
				}

				if (is_epilogue) {
				acc_pipe.consumer_wait(acc_cons_state);
				int acc_stage = acc_cons_state.index();
				uint32_t stage_base =
					tmem_base +
					static_cast<uint32_t>(
						acc_stage *
						Config::kTmemStageColumns);
				for (int panel = 0;
						panel < Config::kAccumulatorPanels;
						++panel) {
					tCtAcc.data() =
						stage_base +
						static_cast<uint32_t>(
							panel *
							Config::kUmmaTileN);
					auto acc_mn_stage =
						tCtAcc(
							make_coord(_, _),
							_0{},
							_0{});
					auto tAccEpiStage =
						flat_divide(
							acc_mn_stage,
							epi_tile);
					auto tTR_tAcc =
						thr_t2r.partition_S(
							tAccEpiStage);
					CUTE_UNROLL
					for (int round = 0;
							round <
								Config::
									kChunksPerWarpgroup;
							++round) {
						int chunk =
							warpgroup *
								Config::
									kChunksPerWarpgroup +
							round;
						auto tAccChunk =
							tTR_tAcc(
								_,
								_,
								_,
								_0{},
								chunk);
						bool issue_tmem_load = true;
						if constexpr (kPredicateTmemLoad) {
							int subpart =
								(tAccChunk.data().dp_ /
									32) %
								4;
							issue_tmem_load =
								tid_in_warpgroup /
									kWarpSize ==
								subpart;
						}
						if (issue_tmem_load) {
							copy(
								t2r,
								tAccChunk,
								tTR_rAcc);
							cutlass::arch::
								fence_view_async_tmem_load();
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize,
							warpgroup_barrier);
						if (issue_tmem_load) {
							CUTE_UNROLL
							for (int i = 0;
									i <
										size(
											tTR_rAcc);
									++i) {
								int row =
									get<0>(
										tTR_cChunk(
											i));
								int column =
									get<1>(
										tTR_cChunk(
											i));
								smem.logit_row(
									warpgroup,
									row)[column] =
									static_cast<
										typename Traits::
											ElementLogit>(
										tTR_rAcc(i));
							}
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize,
							warpgroup_barrier);

						int row = tid_in_warpgroup;
						int global_row =
							work.raw_pid_m *
								Config::kCtaTileM +
							row;
						int vocab_base =
							logical_n *
								Config::kLogicalTileN +
							panel *
								Config::kUmmaTileN +
							chunk *
								Config::kEpilogueChunkN;
						int valid_cols =
							params.local_vocab -
							vocab_base;
						if (valid_cols < 0)
							valid_cols = 0;
						if (valid_cols >
							Config::kEpilogueChunkN) {
							valid_cols =
								Config::
									kEpilogueChunkN;
						}
						int target_offset = -1;
						if (
							work.store_enabled &&
							global_row < params.tokens) {
							std::int64_t target_id =
								params.target[
									global_row];
							std::int64_t local_target =
								target_id -
								params.vocab_start;
							if (
								target_id !=
									params.ignore_index &&
								local_target >=
									vocab_base &&
								local_target <
									vocab_base +
										valid_cols) {
								target_offset =
									static_cast<int>(
										local_target) -
									vocab_base;
							}
						}
						Epilogue::template fold_chunk<
							ReturnEntropy,
							Config::kEpilogueChunkN>(
								softmax,
								smem.logit_row(
									warpgroup,
									row),
								valid_cols,
								target_offset,
								params.
									inverse_temperature);
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize,
							warpgroup_barrier);
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kEpilogueThreads,
					Config::kEpilogueBarrierId);
				if (tid_in_epi == 0) {
					acc_pipe.consumer_release(
						acc_cons_state);
				}
				++acc_cons_state;
				}
			}

			if (is_epilogue) {
			int row = tid_in_warpgroup;
			int index =
				smem.state_index(warpgroup, row);
			smem.state_float(Smem::kMaxField, index) =
				softmax.max_value;
			smem.state_float(Smem::kSumField, index) =
				softmax.exp_sum;
			smem.state_float(Smem::kTargetField, index) =
				softmax.target_logit;
			smem.state_int(Smem::kHasTargetField, index) =
				softmax.has_target;
			if constexpr (ReturnEntropy) {
				smem.state_float(
					Smem::kWeightedField,
					index) =
					softmax.exp_weighted_sum;
			}
			cutlass::arch::NamedBarrier::sync(
				Config::kEpilogueThreads,
				Config::kEpilogueBarrierId);
			if (warpgroup == 0 && work.store_enabled) {
				int other =
					smem.state_index(1, row);
				OnlineSoftmaxState rhs;
				rhs.max_value =
					smem.state_float(
						Smem::kMaxField,
						other);
				rhs.exp_sum =
					smem.state_float(
						Smem::kSumField,
						other);
				rhs.target_logit =
					smem.state_float(
						Smem::kTargetField,
						other);
				rhs.has_target =
					smem.state_int(
						Smem::kHasTargetField,
						other);
				if constexpr (ReturnEntropy) {
					rhs.exp_weighted_sum =
						smem.state_float(
							Smem::kWeightedField,
							other);
				}
				OnlineSoftmaxState combined =
					Epilogue::template combine_scaled<
						ReturnEntropy>(
							softmax,
							rhs);
				Epilogue::template store_partial<
					ReturnEntropy>(
						combined,
						partials,
						(WavePipeline
								? forward_wave_partial_row_offset_sm100(
									split,
									forward_wave_slot_sm100(wave),
									work.raw_pid_m,
									work.split_id)
								: static_cast<std::size_t>(
									work.output_work) *
									Config::kCtaTileM) +
							static_cast<std::size_t>(row));
				if constexpr (WavePipeline) {
					__threadfence_system();
				}
			}
			if constexpr (WavePipeline) {
				cutlass::arch::NamedBarrier::sync(
					Config::kEpilogueThreads,
					Config::kEpilogueBarrierId);
				if (
					tid_in_epi == 0 &&
					work.store_enabled) {
					std::size_t ready_offset =
						forward_wave_partial_ready_offset_sm100(
							split,
							forward_wave_slot_sm100(wave),
							work.raw_pid_m,
							work.split_id);
					forward_store_release_system_sm100(
						partials.ready + ready_offset,
						forward_wave_epoch_sm100(
							*wave_workspace.launch_epoch,
							wave));
				}
			}
			}
		}

		cutlass::arch::NamedBarrier::sync(
			kMmaEpilogueThreads,
			Config::kMmaEpilogueBarrierId);
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (is_epilogue && tid_in_epi == 0 && work.store_enabled) {
				forward_diagnostic_max_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticFinalPublishSm100,
					forward_globaltimer_sm100());
			}
		}
#else
		__trap();
#endif
	}
};

template <
	liger_cute::detail::LocalReduceBackend Backend,
	liger_cute::detail::ReduceOp Op,
	class Mapping>
CUTE_DEVICE void forward_local_reduce_warp_epoch_sm100(
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		std::size_t data_offset,
		std::size_t count,
		std::size_t ready_offset,
		std::uint64_t epoch) {
	unsigned int lane =
		liger_cute::detail::nvls_lane_id();
	if (mapping.size == 1) {
		for (std::size_t index = lane;
				index < count;
				index += kWarpSize) {
			comm.reduced[data_offset + index] =
				comm.partial[data_offset + index];
		}
		__syncwarp();
		return;
	}

	std::size_t complete_offset =
		ready_offset +
		static_cast<std::size_t>(mapping.size);

	if constexpr (
		Backend ==
		liger_cute::detail::LocalReduceBackend::kNvls) {
		liger_cute::detail::LocalReduceContext<
			Backend,
			float> context{
				comm.partial + data_offset,
				comm.reduced + data_offset,
				comm.sync + ready_offset,
				mapping.multicast_sync + ready_offset,
				comm.sync + complete_offset,
				mapping.multicast_sync +
					complete_offset,
				mapping.rank,
				mapping.size};
		liger_cute::detail::local_all_reduce<
			Backend,
			Op>(
				context,
				mapping.multicast_reduced +
					data_offset,
				mapping.multicast_partial +
					data_offset,
				count,
				epoch);
	} else {
		liger_cute::detail::LocalReduceContext<
			Backend,
			float> context{
				mapping.peer_partial,
				data_offset,
				comm.sync + ready_offset,
				mapping.peer_sync,
				ready_offset,
				comm.sync + complete_offset,
				complete_offset,
				mapping.rank,
				mapping.size};
		liger_cute::detail::local_all_reduce<
			Backend,
			Op>(
				context,
				comm.reduced + data_offset,
				comm.partial + data_offset,
				count,
				epoch);
	}
}

template <
	liger_cute::detail::LocalReduceBackend Backend,
	liger_cute::detail::ReduceOp Op,
	class Mapping>
CUTE_DEVICE void forward_local_reduce_warp_sm100(
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		std::size_t data_offset,
		std::size_t count,
		int m_tile,
		int phase) {
	std::size_t ready_offset =
		static_cast<std::size_t>(m_tile * 4 + phase * 2) *
		static_cast<std::size_t>(mapping.size);
	std::uint64_t epoch =
		dx_epoch_base(comm) |
		kForwardLocalReduceEpochSuffixSm100 |
		(static_cast<std::uint64_t>(m_tile) << 4) |
		static_cast<std::uint64_t>(phase + 1);
	forward_local_reduce_warp_epoch_sm100<Backend, Op>(
		comm,
		mapping,
		data_offset,
		count,
		ready_offset,
		epoch);
}

template <
	liger_cute::detail::LocalReduceBackend Backend,
	liger_cute::detail::ReduceOp Op,
	class Mapping,
	int Compute>
CUTE_DEVICE void forward_wave_local_reduce_warp_sm100(
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		const ForwardGemmSplitSm100<Compute>& split,
		std::size_t data_offset,
		std::size_t count,
		int slot,
		int m_tile,
		int wave,
		int phase) {
	std::size_t sync_group =
		(static_cast<std::size_t>(slot) *
				static_cast<std::size_t>(split.num_m_tiles) +
			static_cast<std::size_t>(m_tile)) *
			4u +
		static_cast<std::size_t>(phase * 2);
	std::size_t ready_offset =
		sync_group * static_cast<std::size_t>(mapping.size);
	std::uint64_t epoch =
		forward_wave_epoch_sm100(dx_epoch_base(comm), wave) |
		kForwardLocalReduceEpochSuffixSm100 |
		static_cast<std::uint64_t>(phase + 1);
	forward_local_reduce_warp_epoch_sm100<Backend, Op>(
		comm,
		mapping,
		data_offset,
		count,
		ready_offset,
		epoch);
}

template <
	bool ReturnEntropy,
	int Compute,
	liger_cute::detail::LocalReduceBackend Backend,
	class Mapping>
CUTE_DEVICE void forward_finalize_splits_and_reduce_local_sm100(
		ForwardGemmSmemSm100<
			Compute,
			ReturnEntropy>& smem,
		const ForwardGemmParamsSm100<Compute>& params,
		const ForwardGemmPartialsSm100<
			Compute>& partials,
		const ForwardGemmSplitSm100<
			Compute>& split,
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		int* split_ready,
		float* global_max,
		float* reduced,
		std::uint64_t* diagnostics,
		const ForwardFinalOutputsSm100& outputs) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	using Config = ForwardGemmConfigSm100<Compute>;
	using Epilogue = ForwardGemmEpilogueSm100<Compute>;

	ForwardGemmWorkSm100<Compute> work =
		forward_gemm_assign_work_sm100<Compute>(
			split,
			static_cast<int>(blockIdx.z),
			static_cast<int>(
				cute::block_rank_in_cluster()));
	int tid = static_cast<int>(threadIdx.x);
	bool wrote_partial =
		tid >= Config::kFirstEpilogueWarp * kWarpSize &&
		tid <
			Config::kFirstEpilogueWarp * kWarpSize +
				Config::kCtaTileM;
	if (wrote_partial && work.store_enabled) {
		__threadfence();
	}
	__syncthreads();

	if (tid == 0) {
		smem.finalizer = 0;
		if (work.store_enabled) {
			int completed = atomicAdd(
				split_ready + work.raw_pid_m,
				1) + 1;
			smem.finalizer =
				completed ==
				split.split_count_for_pair(
					work.m_pair);
		}
	}
	__syncthreads();

	if (smem.finalizer == 0) return;

	if (tid < Config::kCtaTileM) {
		int row =
			work.raw_pid_m * Config::kCtaTileM + tid;
		int split_count =
			split.split_count_for_pair(work.m_pair);
		float row_max = kForwardNegInfSm100;
		float row_target = 0.0f;
		for (int split_id = 0;
				split_id < split_count;
				++split_id) {
			int index =
				(work.raw_pid_m * split.split_n +
					split_id) *
					Config::kCtaTileM +
				tid;
			row_max = fmaxf(
				row_max,
				partials.partial_max[index]);
			row_target +=
				partials.partial_target[index];
		}

		float row_sum = 0.0f;
		float row_weighted = 0.0f;
		for (int split_id = 0;
				split_id < split_count;
				++split_id) {
			int index =
				(work.raw_pid_m * split.split_n +
					split_id) *
					Config::kCtaTileM +
				tid;
			float scale = forward_exp2_sm100(
				(partials.partial_max[index] - row_max) *
				kForwardLog2ESm100);
			row_sum +=
				partials.partial_sum[index] * scale;
			if constexpr (ReturnEntropy) {
				row_weighted +=
					partials.partial_weighted[index] *
					scale;
			}
		}

		if (row < params.tokens) {
			OnlineSoftmaxState state;
			state.max_value = row_max;
			state.exp_sum = row_sum;
			state.target_logit = row_target;
			std::int64_t target_id =
				params.target[row];
			std::int64_t local =
				target_id - params.vocab_start;
			state.has_target =
				target_id != params.ignore_index &&
				local >= 0 &&
				local < params.local_vocab;
			if constexpr (ReturnEntropy) {
				state.exp_weighted_sum =
					row_weighted;
			}
			Epilogue::template store_row<
				ReturnEntropy>(
					state,
					params.output,
					row);
		}
	}
	__syncthreads();

	int warp = tid / kWarpSize;
	int lane = tid % kWarpSize;
	if (warp == Config::kLocalReduceWarp) {
		constexpr std::size_t kRows =
			Config::kCtaTileM;
		constexpr std::size_t kStride =
			kForwardReducedFields * kRows;
		std::size_t base =
			static_cast<std::size_t>(work.raw_pid_m) *
			kStride;
		for (int row_in_tile = lane;
				row_in_tile < Config::kCtaTileM;
				row_in_tile += kWarpSize) {
			int row =
				work.raw_pid_m * Config::kCtaTileM +
				row_in_tile;
			comm.partial[base + row_in_tile] =
				row < params.tokens
				? params.output.local_max[row]
				: kForwardNegInfSm100;
		}
		__syncwarp();
		forward_local_reduce_warp_sm100<
			Backend,
			liger_cute::detail::ReduceOp::kMax>(
				comm,
				mapping,
				base,
				kRows,
				work.raw_pid_m,
				0);

		constexpr std::size_t kFields =
			forward_reduced_fields<ReturnEntropy>();
		for (int row_in_tile = lane;
				row_in_tile < Config::kCtaTileM;
				row_in_tile += kWarpSize) {
			int row =
				work.raw_pid_m * Config::kCtaTileM +
				row_in_tile;
			float local_sum =
				row < params.tokens
				? params.output.local_sum[row]
				: 0.0f;
			float local_max =
				row < params.tokens
				? params.output.local_max[row]
				: kForwardNegInfSm100;
			float node_max =
				comm.reduced[base + row_in_tile];
			if (row < params.tokens) {
				params.output.local_max[row] =
					node_max;
				global_max[row] = node_max;
			}
			float correction =
				local_sum == 0.0f
				? 0.0f
				: forward_exp2_sm100(
					(local_max - node_max) *
					kForwardLog2ESm100);
			float local_target =
				row < params.tokens
				? params.output.local_target[row]
				: 0.0f;
			float local_weighted = 0.0f;
			if constexpr (ReturnEntropy) {
				local_weighted =
					row < params.tokens
					? params.output.
						local_weighted_sum[row]
					: 0.0f;
			}
			comm.partial[
				base +
				kForwardReducedSumField *
					kRows +
				row_in_tile] =
				local_sum * correction;
			comm.partial[
				base +
				kForwardReducedTargetField *
					kRows +
				row_in_tile] =
				local_target;
			if constexpr (ReturnEntropy) {
				comm.partial[
					base +
					kForwardReducedWeightedField *
							kRows +
						row_in_tile] =
							local_weighted * correction;
			}
		}
		__syncwarp();
		liger_cute::detail::
			publish_local_reduce_source();
		forward_local_reduce_warp_sm100<
			Backend,
			liger_cute::detail::ReduceOp::kSum>(
				comm,
				mapping,
				base,
				kFields * kRows,
				work.raw_pid_m,
				1);
		for (int row_in_tile = lane;
				row_in_tile < Config::kCtaTileM;
				row_in_tile += kWarpSize) {
			int row =
				work.raw_pid_m * Config::kCtaTileM +
				row_in_tile;
			if (row < params.tokens) {
				CUTE_UNROLL
				for (int field = 0;
						field <
							static_cast<int>(
								kFields);
						++field) {
					reduced[
						field * params.tokens +
						row] =
						comm.reduced[
							base +
							static_cast<
								std::size_t>(
									field) *
								kRows +
							row_in_tile];
				}
				float global_weighted = 0.0f;
				if constexpr (ReturnEntropy) {
					global_weighted =
						comm.reduced[
							base +
							kForwardReducedWeightedField *
							kRows +
							row_in_tile];
				}
				FinalizedSoftmax result =
					finalize_softmax<ReturnEntropy>(
						global_max[row],
						comm.reduced[
							base +
							kForwardReducedSumField *
							kRows +
							row_in_tile],
						comm.reduced[
							base +
							kForwardReducedTargetField *
							kRows +
							row_in_tile],
						global_weighted,
						params.target[row] ==
							params.ignore_index);
				outputs.nll[row] = result.nll;
				outputs.lse[row] = result.lse;
				if constexpr (ReturnEntropy) {
					outputs.entropy[row] =
						result.entropy;
				}
			}
		}
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				std::uint64_t complete =
					forward_globaltimer_sm100();
				forward_diagnostic_max_sm100(
					diagnostics,
					kForwardDiagnosticLocalReduceCompleteSm100,
					complete);
				forward_diagnostic_max_sm100(
					diagnostics,
					kForwardDiagnosticOutputCompleteSm100,
					complete);
			}
		}
	}
	__syncthreads();
}

template <
	bool ReturnEntropy,
	int Compute,
	liger_cute::detail::LocalReduceBackend Backend,
	class Mapping>
CUTE_DEVICE void forward_reduce_waves_sm100(
		const ForwardGemmParamsSm100<Compute>& params,
		const ForwardGemmPartialsSm100<Compute>& partials,
		const ForwardGemmSplitSm100<Compute>& split,
		const DxReduceWorkspace<float>& comm,
		const Mapping& mapping,
		const ForwardWaveWorkspaceSm100<Compute>& wave_workspace,
		const ForwardGemmWorkSm100<Compute>& work) {
	static_assert(
		Backend == liger_cute::detail::LocalReduceBackend::kNvls,
		"the remote wave pipeline requires node-local NVLS");
	using Config = ForwardGemmConfigSm100<Compute>;
	if (work.split_id != 0 || !work.store_enabled) return;

	int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	constexpr std::size_t kRows = Config::kCtaTileM;
	constexpr std::size_t kFields =
		forward_reduced_fields<ReturnEntropy>();
	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	constexpr std::size_t kTileStride =
		kForwardReducedFields * kRows;
	int split_count =
		split.split_count_for_pair(work.m_pair);
	int owned_row_begin =
		mapping.rank * wave_workspace.rows_per_rank;
	int owned_row_end =
		owned_row_begin + wave_workspace.rows_per_rank;

	for (int wave = 0; wave < split.num_waves; ++wave) {
		forward_wait_source_slot_sm100(
			wave_workspace,
			wave_workspace.launch_epoch,
			wave);
		int slot = forward_wave_slot_sm100(wave);
		std::uint64_t epoch = forward_wave_epoch_sm100(
			*wave_workspace.launch_epoch, wave);
		[[maybe_unused]] std::uint64_t wait_begin = 0;
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				wait_begin = forward_globaltimer_sm100();
			}
		}
		forward_wait_partial_epochs_warp_sm100(
			partials,
			split,
			slot,
			work.raw_pid_m,
			split_count,
			epoch);
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				forward_diagnostic_add_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticWarp0WaitSm100,
					forward_globaltimer_sm100() - wait_begin);
			}
		}

		std::size_t base =
			(static_cast<std::size_t>(slot) *
					static_cast<std::size_t>(
						split.num_m_tiles) +
				static_cast<std::size_t>(
					work.raw_pid_m)) *
			kTileStride;
		for (int row_in_tile = lane;
				row_in_tile < Config::kCtaTileM;
				row_in_tile += kWarpSize) {
			int row =
				work.raw_pid_m * Config::kCtaTileM +
				row_in_tile;
			float row_max = kForwardNegInfSm100;
			if (row < params.tokens) {
				for (int split_id = 0;
						split_id < split_count;
						++split_id) {
					std::size_t index =
						forward_wave_partial_row_offset_sm100(
							split,
							slot,
							work.raw_pid_m,
							split_id) +
						static_cast<std::size_t>(
							row_in_tile);
					row_max = fmaxf(
						row_max,
						partials.partial_max[index]);
				}
			}
			comm.partial[base + row_in_tile] = row_max;
		}
		__syncwarp();
		liger_cute::detail::publish_local_reduce_source();
		forward_wave_local_reduce_warp_sm100<
			Backend,
			liger_cute::detail::ReduceOp::kMax>(
				comm,
				mapping,
				split,
				base,
				kRows,
				slot,
				work.raw_pid_m,
				wave,
				0);

		float* source =
			wave_workspace.source_slots +
			static_cast<std::size_t>(slot) *
				wave_workspace.source_slot_elements;
		for (int row_in_tile = lane;
				row_in_tile < Config::kCtaTileM;
				row_in_tile += kWarpSize) {
			int row =
				work.raw_pid_m * Config::kCtaTileM +
				row_in_tile;
			float node_max =
				comm.reduced[base + row_in_tile];
			float row_sum = 0.0f;
			float row_target = 0.0f;
			float row_weighted = 0.0f;
			if (row < params.tokens) {
				for (int split_id = 0;
						split_id < split_count;
						++split_id) {
					std::size_t index =
						forward_wave_partial_row_offset_sm100(
							split,
							slot,
							work.raw_pid_m,
							split_id) +
						static_cast<std::size_t>(
							row_in_tile);
					float split_sum =
						partials.partial_sum[index];
					float correction =
						split_sum == 0.0f
						? 0.0f
						: forward_exp2_sm100(
							(partials.partial_max[index] -
								node_max) *
							kForwardLog2ESm100);
					row_sum += split_sum * correction;
					row_target +=
						partials.partial_target[index];
					if constexpr (ReturnEntropy) {
						row_weighted +=
							partials.partial_weighted[index] *
							correction;
					}
				}
			}
			comm.partial[
				base +
				kForwardReducedSumField * kRows +
				row_in_tile] = row_sum;
			comm.partial[
				base +
				kForwardReducedTargetField * kRows +
				row_in_tile] = row_target;
			if constexpr (ReturnEntropy) {
				comm.partial[
					base +
						kForwardReducedWeightedField *
							kRows +
						row_in_tile] =
					row_weighted;
			}
			if (
				row >= owned_row_begin &&
				row < owned_row_end) {
				std::size_t owned =
					static_cast<std::size_t>(
						row - owned_row_begin);
				source[owned * kStateFields] =
					row < params.tokens
					? node_max
					: kForwardNegInfSm100;
			}
		}
		__syncwarp();
		liger_cute::detail::publish_local_reduce_source();
		forward_wave_local_reduce_warp_sm100<
			Backend,
			liger_cute::detail::ReduceOp::kSum>(
				comm,
				mapping,
				split,
				base,
				kFields * kRows,
				slot,
				work.raw_pid_m,
				wave,
				1);

		int tile_row_begin =
			work.raw_pid_m * Config::kCtaTileM;
		int tile_row_end =
			tile_row_begin + Config::kCtaTileM;
		if (work.raw_pid_m + 1 == split.num_m_tiles) {
			tile_row_end = wave_workspace.padded_tokens;
		}
		for (int row = tile_row_begin + lane;
				row < tile_row_end;
				row += kWarpSize) {
			if (
				row < owned_row_begin ||
				row >= owned_row_end) {
				continue;
			}
			std::size_t owned =
				static_cast<std::size_t>(
					row - owned_row_begin);
			float* state =
				source + owned * kStateFields;
			if (row >= params.tokens) {
				store_forward_reduced_state<ReturnEntropy>(
					state,
					ReducedSoftmaxState{
						kForwardNegInfSm100,
						0.0f,
						0.0f,
						0.0f});
				continue;
			}
			int row_in_tile = row - tile_row_begin;
			for (int field = 0;
					field < static_cast<int>(kFields);
					++field) {
				state[1 + field] =
					comm.reduced[
						base +
						static_cast<std::size_t>(field) *
							kRows +
						static_cast<std::size_t>(
							row_in_tile)];
			}
		}
		__syncwarp();
		liger_cute::detail::publish_local_reduce_source();
		if (lane == 0) {
			std::size_t ready_offset =
				forward_wave_tile_ready_offset_sm100(
					split,
					slot,
					work.raw_pid_m);
			forward_store_release_system_sm100(
				wave_workspace.tile_ready + ready_offset,
				epoch);
			if (
				wave + 1 == split.num_waves) {
				forward_diagnostic_max_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticLocalReduceCompleteSm100,
					forward_globaltimer_sm100());
			}
		}
		__syncwarp();
	}
}

#if defined(LIGER_CUTE_FSLCE_SM100_ENABLE_NVSHMEM)
template <bool ReturnEntropy, int Compute>
CUTE_DEVICE void forward_communicate_waves_sm100(
		const ForwardGemmParamsSm100<Compute>& params,
		const ForwardGemmSplitSm100<Compute>& split,
		const DxReduceWorkspace<float>& comm,
		const liger_cute::detail::NvlsReduceView& local,
		const liger_cute::detail::RemoteReduceView& remote,
		const ForwardWaveWorkspaceSm100<Compute>& wave_workspace,
		const ForwardFinalOutputsSm100& outputs) {
	static_assert(
		!liger_cute::detail::remote_ring_uses_grid_sync<1>(),
		"the fused SM100 remote path must remain warp-only");
	using Config = ForwardGemmConfigSm100<Compute>;
	int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	constexpr std::size_t kStateFields =
		forward_reduced_state_fields<ReturnEntropy>();
	for (int owned = lane;
			owned < wave_workspace.rows_per_rank;
			owned += kWarpSize) {
		store_forward_reduced_state<ReturnEntropy>(
			wave_workspace.running_state +
				static_cast<std::size_t>(owned) *
					kStateFields,
			ReducedSoftmaxState{
				kForwardNegInfSm100,
				0.0f,
				0.0f,
				0.0f});
	}
	__syncwarp();

	for (int wave = 0; wave < split.num_waves; ++wave) {
		int slot = forward_wave_slot_sm100(wave);
		std::uint64_t epoch = forward_wave_epoch_sm100(
			*wave_workspace.launch_epoch, wave);
		[[maybe_unused]] std::uint64_t ready_wait_begin = 0;
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				ready_wait_begin = forward_globaltimer_sm100();
			}
		}
		forward_wait_tile_epochs_warp_sm100(
			wave_workspace,
			split,
			slot,
			epoch);
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				forward_diagnostic_add_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticWarp1WaitSm100,
					forward_globaltimer_sm100() -
						ready_wait_begin);
			}
		}
		float* source =
			wave_workspace.source_slots +
			static_cast<std::size_t>(slot) *
				wave_workspace.source_slot_elements;
		[[maybe_unused]] std::uint64_t ring_begin = 0;
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				ring_begin = forward_globaltimer_sm100();
				if (wave < kForwardDiagnosticMaxWavesSm100) {
					forward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						forward_diagnostic_wave_start_sm100(
							wave),
						ring_begin);
				}
			}
		}
		if constexpr (!kForwardDiagnosticDisableRemoteSm100) {
			if constexpr (kForwardUseWarpTeamCollectivesSm100) {
				reduce_forward_state_team_warp<ReturnEntropy>(
					remote,
					source,
					static_cast<std::size_t>(
						wave_workspace.rows_per_rank));
			} else {
				std::uint64_t operation_suffix =
					forward_wave_operation_suffix_sm100(
						liger_cute::detail::
							kForwardRemoteEpochSuffix,
						wave);
				std::uint64_t previous_operation_suffix =
					kForwardPipelinedRingWavesSm100 &&
						wave > 0
					? forward_wave_operation_suffix_sm100(
						liger_cute::detail::
							kForwardRemoteEpochSuffix,
						wave - 1)
					: 0;
				reduce_forward_state_ring<ReturnEntropy, 1>(
					remote,
					wave_workspace.launch_epoch,
					source,
					static_cast<std::size_t>(
						wave_workspace.rows_per_rank),
					operation_suffix,
					wave,
					previous_operation_suffix,
					!kForwardPipelinedRingWavesSm100 ||
						wave + 1 == split.num_waves);
			}
		}
		if constexpr (kForwardDiagnosticTimestampsSm100) {
			if (lane == 0) {
				std::uint64_t ring_end =
					forward_globaltimer_sm100();
				forward_diagnostic_add_sm100(
					wave_workspace.diagnostics,
					kForwardDiagnosticRingTicksSm100,
					ring_end - ring_begin);
				if (wave < kForwardDiagnosticMaxWavesSm100) {
					forward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						forward_diagnostic_wave_end_sm100(
							wave),
						ring_end);
				}
				if (wave + 1 == split.num_waves) {
					forward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kForwardDiagnosticRingCompleteSm100,
						ring_end);
				}
			}
		}
		for (int owned = lane;
				owned < wave_workspace.rows_per_rank;
				owned += kWarpSize) {
			std::size_t offset =
				static_cast<std::size_t>(owned) *
				kStateFields;
			ReducedSoftmaxState running =
				load_forward_reduced_state<ReturnEntropy>(
					wave_workspace.running_state +
						offset);
			ReducedSoftmaxState contribution =
				load_forward_reduced_state<ReturnEntropy>(
					source + offset);
			store_forward_reduced_state<ReturnEntropy>(
				wave_workspace.running_state + offset,
				merge_reduced_softmax<ReturnEntropy>(
					running, contribution));
		}
		__syncwarp();
		if (lane == 0) {
			forward_store_release_system_sm100(
				wave_workspace.slot_released + slot,
				epoch);
		}
		__syncwarp();
	}

	int final_wave = split.num_waves - 1;
	liger_cute::detail::publish_local_reduce_source();
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (lane == 0) {
			forward_diagnostic_max_sm100(
				wave_workspace.diagnostics,
				kForwardDiagnosticAllgatherStartSm100,
				forward_globaltimer_sm100());
		}
	}
	allgather_forward_reduced_state_warp<ReturnEntropy>(
		local,
		comm,
		wave_workspace.launch_epoch,
		wave_workspace.running_state,
		wave_workspace.padded_tokens,
		Config::kRemoteCommunicationWarp,
		Config::kRemoteCommunicationWarp,
		forward_wave_operation_suffix_sm100(
			kForwardLocalAllgatherEpochSuffix,
			final_wave));
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (lane == 0) {
			forward_diagnostic_max_sm100(
				wave_workspace.diagnostics,
				kForwardDiagnosticAllgatherCompleteSm100,
				forward_globaltimer_sm100());
		}
	}

	for (int row = lane;
			row < params.tokens;
			row += kWarpSize) {
		std::size_t offset =
			static_cast<std::size_t>(row) *
			kStateFields;
		ReducedSoftmaxState state =
			load_forward_reduced_state<ReturnEntropy>(
				comm.reduced + offset);
		FinalizedSoftmax result =
			finalize_softmax<ReturnEntropy>(
				state.max_value,
				state.exp_sum,
				state.target_logit,
				state.exp_weighted_sum,
				params.target[row] ==
					params.ignore_index);
		outputs.nll[row] = result.nll;
		outputs.lse[row] = result.lse;
		if constexpr (ReturnEntropy) {
			outputs.entropy[row] = result.entropy;
		}
	}
	__syncwarp();
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (lane == 0) {
			forward_diagnostic_max_sm100(
				wave_workspace.diagnostics,
				kForwardDiagnosticOutputCompleteSm100,
				forward_globaltimer_sm100());
		}
	}
}
#endif

template <
	bool ReturnEntropy,
	int Compute,
	bool RequiresRemote,
	liger_cute::detail::LocalReduceBackend Backend,
	class TmaLoadX,
	class TmaLoadW,
	class Mapping>
__global__ __launch_bounds__(
	ForwardGemmConfigSm100<
		Compute>::kNumThreads,
	1) __cluster_dims__(2, 1, 1)
void forward_gemm_tp_kernel_sm100(
		__grid_constant__ const TmaLoadX tma_load_x,
		__grid_constant__ const TmaLoadW tma_load_w,
		__grid_constant__ const ForwardGemmParamsSm100<
			Compute> params,
		__grid_constant__ const ForwardGemmPartialsSm100<
			Compute> partials,
		__grid_constant__ const ForwardGemmSplitSm100<
			Compute> split,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const Mapping mapping,
		int* split_ready,
		float* global_max,
		float* reduced,
		__grid_constant__ const ForwardWaveWorkspaceSm100<
			Compute> wave_workspace,
		__grid_constant__ const liger_cute::detail::
			RemoteReduceView remote,
		__grid_constant__ const ForwardFinalOutputsSm100 outputs) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	static_assert(
		!RequiresRemote ||
			Backend ==
				liger_cute::detail::
					LocalReduceBackend::kNvls,
		"the fused SM100 remote path requires node-local NVLS");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Traits = ForwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Smem = ForwardGemmSmemSm100<
		Compute,
		ReturnEntropy>;
	extern __shared__ char raw_smem[];
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);

	cute::prefetch_tma_descriptor(
		tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_w.get_tma_descriptor());
	auto pipe =
		forward_make_pipe_sm100<Compute>(
			smem.pipeline);

	int warp_id =
		static_cast<int>(threadIdx.x) / kWarpSize;
	cute::TMEM::Allocator2Sm tmem_allocator;
	// Startup-only rendezvous: all roles enter before any wave reduction or
	// remote communication begins. No whole-CTA barrier is used after this
	// point by the RequiresRemote specialization.
	cute::cluster_sync();
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.allocate(
			Config::kTmemColumns,
			&smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();

	ForwardGemmWorkSm100<Compute> work =
		forward_gemm_assign_work_sm100<Compute>(
				split,
				static_cast<int>(blockIdx.z),
				static_cast<int>(
					cute::block_rank_in_cluster()));
	int num_k_tiles =
		ForwardGemmLaunchSm100<
			Compute>::num_k_tiles(params.hidden);
	if constexpr (kForwardDiagnosticTimestampsSm100) {
		if (
			warp_id == Config::kTmaWarp &&
			(threadIdx.x & (kWarpSize - 1)) == 0) {
			forward_diagnostic_min_sm100(
				wave_workspace.diagnostics,
				kForwardDiagnosticKernelStartSm100,
				forward_globaltimer_sm100());
		}
	}

	typename Traits::PipelineState state;
	if (warp_id == Config::kTmaWarp) {
		state =
			cutlass::make_producer_start_state<
				typename Traits::MainloopPipeline>();
		ForwardGemmProducerSm100<
			Compute>::template run<
				ReturnEntropy,
				RequiresRemote>(
				pipe,
				state,
				smem,
				tma_load_x,
				tma_load_w,
				params,
				work,
				split,
				wave_workspace,
				num_k_tiles);
	} else if (
		warp_id >= Config::kUmmaWarp &&
		warp_id <= Config::kLastEpilogueWarp) {
		ForwardGemmConsumerSm100<
			ReturnEntropy,
			RequiresRemote,
			Compute>::run(
				pipe,
				state,
				smem,
				params,
				partials,
				work,
				split,
				wave_workspace,
				num_k_tiles);
	}

	if constexpr (RequiresRemote) {
		if (warp_id == Config::kLocalReduceWarp) {
			forward_reduce_waves_sm100<
				ReturnEntropy,
				Compute,
				Backend>(
					params,
					partials,
					split,
					comm,
					mapping,
					wave_workspace,
					work);
		} else if (
			warp_id ==
				Config::kRemoteCommunicationWarp &&
			blockIdx.x == 0 &&
			blockIdx.y == 0 &&
			blockIdx.z == 0 &&
			cute::block_rank_in_cluster() == 0) {
#if defined(LIGER_CUTE_FSLCE_SM100_ENABLE_NVSHMEM)
			forward_communicate_waves_sm100<
				ReturnEntropy,
				Compute>(
					params,
					split,
					comm,
					mapping,
					remote,
					wave_workspace,
					outputs);
#else
			__trap();
#endif
		}

		if (
			warp_id >= Config::kUmmaWarp &&
			warp_id <= Config::kLastEpilogueWarp) {
			// Compute-only TMEM teardown. Warps 0 and 1 never enter this
			// barrier, and warp 2 drains its TMA producer loop independently.
			constexpr int kUmmaEpilogueThreads =
				(Config::kLastEpilogueWarp -
					Config::kUmmaWarp + 1) *
				kWarpSize;
			static_assert(kUmmaEpilogueThreads == 288);
			static_assert(
				!(Config::kTmaWarp >= Config::kUmmaWarp &&
					Config::kTmaWarp <=
						Config::kLastEpilogueWarp));
			static_assert(
				!(Config::kLocalReduceWarp >=
						Config::kUmmaWarp &&
					Config::kLocalReduceWarp <=
						Config::kLastEpilogueWarp));
			static_assert(
				!(Config::kRemoteCommunicationWarp >=
						Config::kUmmaWarp &&
					Config::kRemoteCommunicationWarp <=
						Config::kLastEpilogueWarp));
			cutlass::arch::NamedBarrier::sync(
				kUmmaEpilogueThreads,
				Config::kComputeDoneBarrierId);
			if (
				warp_id ==
				Config::kFirstEpilogueWarp) {
				tmem_allocator.release_allocation_lock();
				tmem_allocator.free(
					smem.tmem_base,
					Config::kTmemColumns);
			}
		}
		return;
	} else {
		__syncthreads();
		cute::cluster_sync();
		if (warp_id == Config::kFirstEpilogueWarp) {
			tmem_allocator.release_allocation_lock();
			tmem_allocator.free(
				smem.tmem_base,
				Config::kTmemColumns);
		}
		__syncthreads();

		forward_finalize_splits_and_reduce_local_sm100<
			ReturnEntropy,
			Compute,
			Backend>(
				smem,
				params,
				partials,
				split,
				comm,
				mapping,
				split_ready,
				global_max,
				reduced,
				wave_workspace.diagnostics,
				outputs);
		cute::cluster_sync();
	}
#else
	__trap();
#endif
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
