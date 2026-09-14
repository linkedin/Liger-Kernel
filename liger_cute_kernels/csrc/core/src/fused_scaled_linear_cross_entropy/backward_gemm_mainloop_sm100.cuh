#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Executable SM100 fused backward: one persistent cluster-2 kernel that runs
// dZ, dX and dW plus the token-wave loop. See backward_gemm_sm100.cuh for the
// contract, the warp plan and the epoch layout.
// ═══════════════════════════════════════════════════════════════════════════

#include "backward_gemm_sm100.cuh"
#include "backward_dz_gemm_sm100.cuh"
#include "backward_remote_reduce.cuh"
#include "backward_dx_gemm_sm100.cuh"
#include "backward_dx_wide_sm100.cuh"
#include "dx_reduce.cuh"
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
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {

using namespace cute;

// ───────────────────────────────────────────────────────────────────────────
// Global tensor views
// ───────────────────────────────────────────────────────────────────────────

// int64 extents throughout: tokens * hidden and vocab * hidden both exceed
// INT_MAX at production shapes and would wrap CuTe's layout algebra.
template <class Element>
CUTE_HOST_DEVICE auto backward_row_major_tensor_sm100(
		Element* pointer, int64_t rows, int64_t cols) {
	return make_tensor(
		make_gmem_ptr(pointer),
		make_layout(make_shape(rows, cols), make_stride(cols, _1{})));
}

// A transposed *view* of a row-major [cols, rows] matrix: shape (rows, cols)
// with the first mode contiguous. This is how the MN-major dX / dW operands
// are addressed without ever materialising a transpose.
template <class Element>
CUTE_HOST_DEVICE auto backward_col_major_tensor_sm100(
		Element* pointer, int64_t rows, int64_t cols, int64_t leading) {
	return make_tensor(
		make_gmem_ptr(pointer),
		make_layout(make_shape(rows, cols), make_stride(_1{}, leading)));
}

// ───────────────────────────────────────────────────────────────────────────
// Traits
//
// All three phases use the joined M256 x N256 x K64 tile and one aliased TMEM
// allocation. The validated phase-specific pipeline depths are retained:
// dZ=5, dX=4, dW=6.
//
//   dZ   A = X       (tokens, hidden)          K-major
//        B = W       (vocab,  hidden)          K-major
//   dX   A = dZ      (waveRows, paddedVocab)   K-major
//        B = W^T     (hidden, vocab)           MN-major
//   dW   A = dZ^T    (paddedVocab, waveRows)   MN-major
//        B = X^T     (hidden, tokens)          MN-major
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 100>
struct BackwardGemmTraitsSm100 {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");

	using Config = BackwardGemmConfigSm100<Compute>;
	using Element = cutlass::bfloat16_t;
	using ElementAccum = float;

	static constexpr int kTileM = Config::kTileM;
	static constexpr int kCtaTileM = Config::kCtaTileM;
	static constexpr int kTileN = Config::kTileN;
	static constexpr int kTileK = Config::kTileK;
	static constexpr int kStages = Config::kDzMainloopStages;
	static constexpr int kAccumulatorStages = Config::kAccumulatorStages;

	using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
	using ClusterShape = Shape<Int<Config::kClusterM>, _1, _1>;
	using AtomThrShape = Shape<Int<Config::kClusterM>, _1, _1>;

	template <UMMA::Major MajorA, UMMA::Major MajorB>
	using TiledMmaFor = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element,
			Element,
			ElementAccum,
			kTileM,
			kTileN,
			MajorA,
			MajorB>{}));

	using TiledMmaDz = TiledMmaFor<UMMA::Major::K, UMMA::Major::K>;
	using TiledMmaDx = TiledMmaFor<UMMA::Major::K, UMMA::Major::MN>;
	using TiledMmaDw = TiledMmaFor<UMMA::Major::MN, UMMA::Major::MN>;

	static_assert(
		size(typename TiledMmaDz::AtomThrID{}) == Config::kClusterM,
		"SM100 backward MMA atoms must match their CTA group");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMmaDz::AtomThrID{})));

	using MmaShapeA = decltype(partition_shape_A(
		TiledMmaDz{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
	using MmaShapeB = decltype(partition_shape_B(
		TiledMmaDz{}, make_shape(Int<kTileN>{}, Int<kTileK>{})));

	using AtomK = UMMA::Layout_K_SW128_Atom<Element>;
	using AtomMn = UMMA::Layout_MN_SW128_Atom<Element>;

	using SmemLayoutAK = decltype(UMMA::tile_to_mma_shape(
		AtomK{}, append(MmaShapeA{}, Int<kStages>{}), Step<_2, _1, _3>{}));
	using SmemLayoutBK = decltype(UMMA::tile_to_mma_shape(
		AtomK{}, append(MmaShapeB{}, Int<kStages>{}), Step<_2, _1, _3>{}));
	using SmemLayoutAMn = decltype(UMMA::tile_to_mma_shape(
		AtomMn{}, append(MmaShapeA{}, Int<kStages>{}), Step<_2, _1, _3>{}));
	using SmemLayoutBMn = decltype(UMMA::tile_to_mma_shape(
		AtomMn{}, append(MmaShapeB{}, Int<kStages>{}), Step<_2, _1, _3>{}));

	using SmemLayoutAK1 = decltype(SmemLayoutAK{}(_, _, _, Int<0>{}));
	using SmemLayoutBK1 = decltype(SmemLayoutBK{}(_, _, _, Int<0>{}));
	using SmemLayoutAMn1 = decltype(SmemLayoutAMn{}(_, _, _, Int<0>{}));
	using SmemLayoutBMn1 = decltype(SmemLayoutBMn{}(_, _, _, Int<0>{}));

	static constexpr int kSmemAElements = cosize_v<SmemLayoutAK>;
	static constexpr int kSmemBElements = cosize_v<SmemLayoutBK>;
	static constexpr int kTmaTransBytesA =
		static_cast<int>(cosize_v<SmemLayoutAK1> * sizeof(Element));
	static constexpr int kTmaTransBytesB =
		static_cast<int>(cosize_v<SmemLayoutBK1> * sizeof(Element));
	// The paired pipeline accounts for both CTAs' complete A and B boxes.
	static constexpr int kTmaTransBytes =
		Config::kClusterM * (kTmaTransBytesA + kTmaTransBytesB);
	// BF16 epilogue staging for the dZ and dW TMA stores, one (128, 32) slot
	// per epilogue warpgroup.
	using StoreAtom = UMMA::Layout_K_SW64_Atom<Element>;
	using SmemLayoutStoreSlot = decltype(tile_to_shape(
		StoreAtom{},
		Shape<Int<Config::kCtaTileM>, Int<Config::kEpilogueChunkN>>{}));
	static constexpr int kStoreSlotElements =
		cosize_v<SmemLayoutStoreSlot>;

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline =
		cutlass::PipelineUmmaAsync<kAccumulatorStages, AtomThrShape>;
};

using BackwardDxTraitsSm100 = BackwardDxWideTraitsSm100;

template <int Compute = 100>
struct BackwardDwTraitsSm100 {
	using Base = BackwardGemmTraitsSm100<Compute>;
	using Config = typename Base::Config;
	using Element = typename Base::Element;
	using ElementAccum = typename Base::ElementAccum;

	static constexpr int kStages = Config::kDwMainloopStages;
	static constexpr int kAccumulatorStages = Config::kAccumulatorStages;
	static constexpr int kClusterM = Config::kClusterM;

	using TileShape = typename Base::TileShape;
	using ClusterShape = typename Base::ClusterShape;
	using AtomThrShape = typename Base::AtomThrShape;
	using TiledMma = typename Base::TiledMmaDw;
	using ClusterLayoutVMNK = typename Base::ClusterLayoutVMNK;
	using MmaShapeA = typename Base::MmaShapeA;
	using MmaShapeB = typename Base::MmaShapeB;

	using OperandAtom = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
		OperandAtom{},
		append(MmaShapeA{}, Int<kStages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
		OperandAtom{},
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
		Config::kClusterM * (kTmaTransBytesA + kTmaTransBytesB);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		kStages,
		ClusterShape,
		AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		kAccumulatorStages,
		AtomThrShape>;
};

// ───────────────────────────────────────────────────────────────────────────
// Shared memory
//
// The operand arenas are the phase-serial union: dZ, dX and dW alias the same
// `operand_a` / `operand_b` bytes and are handed over on the compute-only
// named barrier. Pipelines, mbarriers, the TMEM handle and the epilogue's row
// metadata live outside it.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute, bool ReturnEntropy>
struct BackwardGemmSmemSm100 {
	using Traits = BackwardGemmTraitsSm100<Compute>;
	using DxTraits = BackwardDxTraitsSm100;
	using DwTraits = BackwardDwTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Element = typename Traits::Element;

	static constexpr int kRows = Config::kCtaTileM;

	struct DzStorage {
		alignas(1024) Element operand_a[Traits::kSmemAElements];
		alignas(1024) Element operand_b[Traits::kSmemBElements];
		alignas(1024) Element store[
			Config::kEpilogueWarpgroups * Traits::kStoreSlotElements];
		alignas(16) float row_scale[kRows];
		alignas(16) float row_exp_bias[kRows];
		alignas(16) float row_entropy_bias[ReturnEntropy ? kRows : 1];
		alignas(16) float row_entropy_slope[ReturnEntropy ? kRows : 1];
		alignas(16) int row_target[kRows];
	};

	struct DxStorage {
		alignas(1024) Element operand_a[
			cosize_v<typename DxTraits::SmemLayoutA>];
		alignas(1024) Element operand_b[
			cosize_v<typename DxTraits::SmemLayoutB>];
		alignas(1024) float store[DxTraits::kStoreElements];
	};

	struct DwStorage {
		alignas(1024) Element operand_a[
			cosize_v<typename DwTraits::SmemLayoutA>];
		alignas(1024) Element operand_b[
			cosize_v<typename DwTraits::SmemLayoutB>];
		alignas(1024) Element store[
			Config::kEpilogueWarpgroups * Traits::kStoreSlotElements];
	};

	union alignas(1024) PhaseStorage {
		DzStorage dz;
		DxStorage dx;
		DwStorage dw;
	} phase;

	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipeline;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;
	alignas(16) typename DxTraits::MainloopPipeline::SharedStorage dx_pipeline;
	alignas(16) typename DxTraits::AccumulatorPipeline::SharedStorage
		dx_acc_pipe;
	alignas(16) typename DwTraits::MainloopPipeline::SharedStorage dw_pipeline;
	alignas(16) typename DwTraits::AccumulatorPipeline::SharedStorage
		dw_acc_pipe;
	alignas(16) uint32_t tmem_base;
	alignas(16) cutlass::arch::ClusterBarrier tmem_pair_barrier;
	alignas(16) cutlass::arch::ClusterBarrier wave_pair_barrier;
	alignas(8) std::uint64_t dx_ready[kDxRingStages];
	alignas(8) std::uint64_t dx_consumed[kDxRingStages];
	alignas(8) std::uint64_t dz_phase_ready;
	alignas(8) std::uint64_t dx_phase_free;

	CUTE_DEVICE Element* dz_a_data() { return &phase.dz.operand_a[0]; }
	CUTE_DEVICE Element* dz_b_data() { return &phase.dz.operand_b[0]; }
	CUTE_DEVICE Element* dz_store_data(int warpgroup) {
		return &phase.dz.store[warpgroup * Traits::kStoreSlotElements];
	}
	CUTE_DEVICE Element* dx_a_data() { return &phase.dx.operand_a[0]; }
	CUTE_DEVICE Element* dx_b_data() { return &phase.dx.operand_b[0]; }
	CUTE_DEVICE float* dx_store_data(int = 0) {
		return &phase.dx.store[0];
	}
	CUTE_DEVICE Element* dw_a_data() { return &phase.dw.operand_a[0]; }
	CUTE_DEVICE Element* dw_b_data() { return &phase.dw.operand_b[0]; }
	CUTE_DEVICE Element* dw_store_data(int warpgroup) {
		return &phase.dw.store[warpgroup * Traits::kStoreSlotElements];
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Schedule
// ───────────────────────────────────────────────────────────────────────────

struct BackwardPairCoordSm100 {
	int m_pair;
	int n_tile;
};

// N-fast raster used by the validated dX kernel.
__host__ __device__ inline BackwardPairCoordSm100
backward_pair_n_fast_sm100(int item, int num_n_tiles) {
	BackwardPairCoordSm100 coord;
	coord.m_pair = item / num_n_tiles;
	coord.n_tile = item - coord.m_pair * num_n_tiles;
	return coord;
}

// M-fastest raster: cluster pairs sweep the M dimension first so the shared
// B panel stays hot across consecutive work items.
__host__ __device__ inline BackwardPairCoordSm100
backward_pair_m_fast_sm100(int item, int num_m_pairs) {
	BackwardPairCoordSm100 coord;
	coord.m_pair = item % num_m_pairs;
	coord.n_tile = item / num_m_pairs;
	return coord;
}

struct BackwardDwPairCoordSm100 {
	int m_pair;
	int n_tile_begin;
};

// Vertical dW raster: walk vocabulary-M first and pair adjacent hidden-N
// tiles, matching the accepted six-stage dW schedule.
__host__ __device__ inline BackwardDwPairCoordSm100
backward_dw_pair_coord_sm100(int item, int num_m_pairs) {
	return {item % num_m_pairs, 2 * (item / num_m_pairs)};
}

__host__ __device__ inline int backward_items_for_cluster_sm100(
		int total_items, int cluster, int num_clusters) {
	return total_items > cluster
		? (total_items - cluster + num_clusters - 1) / num_clusters
		: 0;
}

// ───────────────────────────────────────────────────────────────────────────
// Full-grid software barrier
//
// Only the compute warps (2..11) take part; warps 0 and 1 keep running their
// communication loops. The counter is monotonic for the whole launch, so no
// reset or sense reversal is needed and a stalled CTA can never alias a later
// generation. Correctness requires the strict full-residency launch invariant
// enforced by the launcher: every cluster of the grid must be co-resident.
// ───────────────────────────────────────────────────────────────────────────

CUTE_DEVICE void backward_grid_barrier_arrive_sm100(
		std::uint64_t* counter) {
	__threadfence();
	atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ull);
}

CUTE_DEVICE std::uint64_t backward_grid_barrier_load_sm100(
		const std::uint64_t* counter) {
	std::uint64_t value;
	asm volatile(
		"ld.acquire.gpu.global.u64 %0, [%1];"
		: "=l"(value)
		: "l"(counter)
		: "memory");
	return value;
}

CUTE_DEVICE std::uint32_t backward_dz_tile_ready_load_sm100(
		const std::uint32_t* address) {
	std::uint32_t value;
	asm volatile(
		"ld.acquire.gpu.global.u32 %0, [%1];"
		: "=r"(value)
		: "l"(address)
		: "memory");
	return value;
}

CUTE_DEVICE void backward_dz_tile_wait_sm100(
		const std::uint32_t* address, std::uint32_t target) {
	unsigned int active = __activemask();
	for (;;) {
		bool ready = backward_dz_tile_ready_load_sm100(address) >= target;
		if (__all_sync(active, ready)) break;
		__nanosleep(64);
	}
	__syncwarp(active);
}

CUTE_DEVICE void backward_dz_tile_publish_sm100(std::uint32_t* address) {
	liger_cute::detail::publish_local_reduce_source();
	atomicAdd(address, 1u);
}

CUTE_DEVICE std::uint64_t backward_globaltimer_sm100() {
	std::uint64_t value;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
	return value;
}

CUTE_DEVICE void backward_diagnostic_first_sm100(
		std::uint64_t* diagnostics, int index) {
	if constexpr (kBackwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicCAS(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				0ull,
				static_cast<unsigned long long>(
					backward_globaltimer_sm100()));
		}
	}
}

CUTE_DEVICE void backward_diagnostic_max_sm100(
		std::uint64_t* diagnostics, int index) {
	if constexpr (kBackwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicMax(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				static_cast<unsigned long long>(
					backward_globaltimer_sm100()));
		}
	}
}

CUTE_DEVICE void backward_diagnostic_duration_max_sm100(
		std::uint64_t* diagnostics,
		int index,
		std::uint64_t begin) {
	if constexpr (kBackwardDiagnosticTimestampsSm100) {
		if (diagnostics != nullptr) {
			atomicMax(
				reinterpret_cast<unsigned long long*>(
					diagnostics + index),
				static_cast<unsigned long long>(
					backward_globaltimer_sm100() - begin));
		}
	}
}

CUTE_DEVICE void backward_grid_wait_warp_sm100(
		const std::uint64_t* counter, unsigned long long target) {
	unsigned int active = __activemask();
	for (;;) {
		bool ready = backward_grid_barrier_load_sm100(counter) >= target;
		if (__all_sync(active, ready)) break;
		__nanosleep(64);
	}
	__syncwarp(active);
}

template <int Compute>
CUTE_DEVICE void backward_grid_barrier_sm100(
		std::uint64_t* counter, unsigned long long target, int warp_id) {
	using Config = BackwardGemmConfigSm100<Compute>;
	cutlass::arch::NamedBarrier::sync(
		Config::kComputeThreads, Config::kComputeBarrierId);
	if (warp_id == Config::kTmaWarp) {
		if ((static_cast<int>(threadIdx.x) & (kWarpSize - 1)) == 0) {
			backward_grid_barrier_arrive_sm100(counter);
		}
		__syncwarp();
		backward_grid_wait_warp_sm100(counter, target);
	}
	cutlass::arch::NamedBarrier::sync(
		Config::kComputeThreads, Config::kComputeBarrierId);
}

template <int Compute>
CUTE_DEVICE void backward_grid_barrier_pipeline_sm100(
		std::uint64_t* counter,
		unsigned long long target,
		int warp_id,
		std::uint64_t& dz_phase_ready,
		int wave) {
	using Config = BackwardGemmConfigSm100<Compute>;
	int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	if (warp_id == Config::kTmaWarp) {
		if (lane == 0) {
			cute::wait_barrier(dz_phase_ready, wave & 1);
			backward_grid_barrier_arrive_sm100(counter);
		}
		__syncwarp();
	}
	backward_grid_wait_warp_sm100(counter, target);
}

// Compute-only phase handoff for the aliased operand and store arenas.
template <int Compute>
CUTE_DEVICE void backward_compute_barrier_sm100() {
	using Config = BackwardGemmConfigSm100<Compute>;
	cutlass::arch::NamedBarrier::sync(
		Config::kComputeThreads, Config::kComputeBarrierId);
}

template <int Compute>
CUTE_DEVICE void backward_tmem_pair_barrier_sm100(
		const cutlass::arch::ClusterBarrier& barrier, int warp_id) {
	using Config = BackwardGemmConfigSm100<Compute>;
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

template <int Compute>
CUTE_DEVICE void backward_wave_pair_barrier_sm100(
		const cutlass::arch::ClusterBarrier& barrier, int warp_id) {
	backward_compute_barrier_sm100<Compute>();
	backward_tmem_pair_barrier_sm100<Compute>(barrier, warp_id);
	backward_compute_barrier_sm100<Compute>();
}

// ───────────────────────────────────────────────────────────────────────────
// Pipelines
// ───────────────────────────────────────────────────────────────────────────

template <int Compute>
CUTE_DEVICE typename BackwardGemmTraitsSm100<Compute>::MainloopPipeline
backward_make_pipe_sm100(
		typename BackwardGemmTraitsSm100<
			Compute>::MainloopPipeline::SharedStorage& storage) {
	using Traits = BackwardGemmTraitsSm100<Compute>;
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
	} else if (warp_id == Config::kUmmaWarp) {
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

template <int Compute>
CUTE_DEVICE typename BackwardDxTraitsSm100::MainloopPipeline
backward_make_dx_pipe_sm100(
		typename BackwardDxTraitsSm100::MainloopPipeline::SharedStorage&
			storage) {
	using Traits = BackwardDxTraitsSm100;
	using Config = BackwardGemmConfigSm100<Compute>;
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
			warp_id < Config::kFirstEpilogueWarp + 4) {
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

template <int Compute>
CUTE_DEVICE typename BackwardDwTraitsSm100<Compute>::MainloopPipeline
backward_make_dw_pipe_sm100(
		typename BackwardDwTraitsSm100<
			Compute>::MainloopPipeline::SharedStorage& storage) {
	using Traits = BackwardDwTraitsSm100<Compute>;
	using Config = BackwardGemmConfigSm100<Compute>;
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

// ───────────────────────────────────────────────────────────────────────────
// Warp 2 — the single TMA producer for every phase
// ───────────────────────────────────────────────────────────────────────────

// Shared forced-inline producer/MMA loops for the single-output dZ and dX
// phases. Their tile mapping and epilogues remain template policies at the
// call site, so the generated loop is identical to the standalone phase.
// dW intentionally retains the paired-N loop below: it acquires two
// accumulator states and interleaves two N tiles per K step, which is a
// materially different instruction schedule rather than an epilogue choice.
//
// `AGlobal` / `BGlobal` are already the phase's (M, K) / (N, K) TMA tensors
// and `a_tile_of` / `b_tile_of` map a work item to its tile index in each.
template <
	int Compute,
	class PhaseTraits,
	class SmemLayoutA,
	class SmemLayoutB,
	class TiledMma,
	class TmaA,
	class TmaB,
	class MA,
	class MB,
	class ATileFn,
	class BTileFn,
	class KOffsetFn,
	class KReadyFn>
CUTE_DEVICE void backward_produce_phase_sm100(
		typename PhaseTraits::MainloopPipeline& pipe,
		typename PhaseTraits::PipelineState& state,
		typename PhaseTraits::Element* a_pointer,
		typename PhaseTraits::Element* b_pointer,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const MA& mA,
		const MB& mB,
		int num_items,
		int first_item,
		int item_stride,
		int num_k_tiles,
		ATileFn a_tile_of,
		BTileFn b_tile_of,
		KOffsetFn k_offset_of,
		KReadyFn k_ready_of) {
	using Traits = PhaseTraits;

	auto sA = make_tensor(make_smem_ptr(a_pointer), SmemLayoutA{});
	auto sB = make_tensor(make_smem_ptr(b_pointer), SmemLayoutB{});

	TiledMma tiled_mma;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	typename Traits::ClusterLayoutVMNK pair_layout_vmnk;
	auto pair_coord_vmnk = pair_layout_vmnk.get_flat_coord(cluster_rank);
	uint16_t mcast_mask_a =
		create_tma_multicast_mask<2>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t mcast_mask_b =
		create_tma_multicast_mask<1>(pair_layout_vmnk, pair_coord_vmnk);

	for (int index = 0; index < num_items; ++index) {
		int item = first_item + index * item_stride;
		auto coord = make_coord(a_tile_of(item), b_tile_of(item), _);
		auto gA = local_tile(
			mA, typename Traits::TileShape{}, coord, Step<_1, X, _1>{});
		auto gB = local_tile(
			mB, typename Traits::TileShape{}, coord, Step<X, _1, _1>{});
		auto tCgA = cta_mma.partition_A(gA);
		auto tCgB = cta_mma.partition_B(gB);
		auto [tAgA, tAsA] = tma_partition(
			tma_a,
			get<2>(pair_coord_vmnk),
			make_layout(size<2>(pair_layout_vmnk)),
			group_modes<0, 3>(sA),
			group_modes<0, 3>(tCgA));
		auto [tBgB, tBsB] = tma_partition(
			tma_b,
			get<1>(pair_coord_vmnk),
			make_layout(size<1>(pair_layout_vmnk)),
			group_modes<0, 3>(sB),
			group_modes<0, 3>(tCgB));

		int k_base_a = k_offset_of(item, 0);
		int k_base_b = k_offset_of(item, 1);
		for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
			k_ready_of(item, k_tile);
			pipe.producer_acquire(state);
			if (cute::elect_one_sync()) {
				auto* barrier = pipe.producer_get_barrier(state);
				copy(
					tma_a.with(*barrier, mcast_mask_a),
					tAgA(_, k_base_a + k_tile),
					tAsA(_, state.index()));
				copy(
					tma_b.with(*barrier, mcast_mask_b),
					tBgB(_, k_base_b + k_tile),
					tBsB(_, state.index()));
			}
			++state;
		}
	}
}

// ───────────────────────────────────────────────────────────────────────────
// Warp 3 — the single UMMA issuer for every phase
// ───────────────────────────────────────────────────────────────────────────

template <
	int Compute,
	class PhaseTraits,
	class SmemLayoutA,
	class SmemLayoutB,
	class TiledMma,
	class AccPipe,
	class AccState>
CUTE_DEVICE void backward_mma_phase_sm100(
		typename PhaseTraits::MainloopPipeline& pipe,
		typename PhaseTraits::PipelineState& state,
		AccPipe& acc_pipe,
		AccState& acc_state,
		typename PhaseTraits::Element* a_pointer,
		typename PhaseTraits::Element* b_pointer,
		uint32_t tmem_base,
		int num_items,
		int num_k_tiles,
		std::uint64_t* diagnostics,
		int first_ready_index) {
	using Config = BackwardGemmConfigSm100<Compute>;

	TiledMma tiled_mma;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	auto sA = make_tensor(make_smem_ptr(a_pointer), SmemLayoutA{});
	auto sB = make_tensor(make_smem_ptr(b_pointer), SmemLayoutB{});
	auto tCrA = cta_mma.make_fragment_A(sA);
	auto tCrB = cta_mma.make_fragment_B(sB);
	auto cAccFull = make_identity_tensor(
		make_shape(
			Int<Config::kTileM>{}, Int<PhaseTraits::kTileN>{}));
	auto tCtAcc = cta_mma.make_fragment_C(cta_mma.partition_C(cAccFull));

	for (int index = 0; index < num_items; ++index) {
		acc_pipe.producer_acquire(acc_state);
		tCtAcc.data() = tmem_base +
			static_cast<uint32_t>(
				acc_state.index() * PhaseTraits::kTileN);
		bool first = true;
		for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
			pipe.consumer_wait(state);
			if constexpr (kBackwardDiagnosticTimestampsSm100) {
				if (
					index == 0 && k_tile == 0 &&
					(static_cast<int>(threadIdx.x) &
						(kWarpSize - 1)) == 0) {
					backward_diagnostic_first_sm100(
						diagnostics, first_ready_index);
				}
			}
			CUTE_UNROLL
			for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
				tiled_mma.accumulate_ = first
					? UMMA::ScaleOut::Zero
					: UMMA::ScaleOut::One;
				first = false;
				gemm(
					tiled_mma,
					tCrA(_, _, k_block, state.index()),
					tCrB(_, _, k_block, state.index()),
					tCtAcc);
			}
			pipe.consumer_release(state);
			++state;
		}
		acc_pipe.producer_commit(acc_state);
		++acc_state;
	}
}

template <
	int Compute,
	int KTiles,
	bool WaitDzReady,
	class TmaA,
	class TmaB,
	class MA,
	class MB>
CUTE_DEVICE void backward_produce_dw_pairs_sm100(
		typename BackwardDwTraitsSm100<Compute>::MainloopPipeline& pipe,
		typename BackwardDwTraitsSm100<Compute>::PipelineState& state,
		typename BackwardDwTraitsSm100<Compute>::Element* a_pointer,
		typename BackwardDwTraitsSm100<Compute>::Element* b_pointer,
		const TmaA& tma_a,
		const TmaB& tma_b,
		const MA& mA,
		const MB& mB,
		int num_items,
		int first_item,
		int item_stride,
		int num_m_pairs,
		int num_n_tiles,
		int a_k_offset,
		int b_k_offset,
		const std::uint32_t* dz_tile_ready,
		int wave,
		int dz_items,
		int dz_n_tiles) {
	using Traits = BackwardDwTraitsSm100<Compute>;
		using Config = typename Traits::Config;

	auto sA = make_tensor(
		make_smem_ptr(a_pointer), typename Traits::SmemLayoutA{});
	auto sB = make_tensor(
		make_smem_ptr(b_pointer), typename Traits::SmemLayoutB{});
	typename Traits::TiledMma tiled_mma;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	typename Traits::ClusterLayoutVMNK cluster_layout;
	auto cluster_coord = cluster_layout.get_flat_coord(cluster_rank);
	std::uint16_t multicast_a =
		create_tma_multicast_mask<2>(cluster_layout, cluster_coord);
	std::uint16_t multicast_b =
		create_tma_multicast_mask<1>(cluster_layout, cluster_coord);

	for (int index = 0; index < num_items; ++index) {
		int item = first_item + index * item_stride;
		BackwardDwPairCoordSm100 coord =
			backward_dw_pair_coord_sm100(item, num_m_pairs);
		int n_tile0 = coord.n_tile_begin;
		int n_tile1 =
			n_tile0 + 1 < num_n_tiles ? n_tile0 + 1 : n_tile0;
		bool has_second = n_tile0 + 1 < num_n_tiles;
		auto tile_coord0 = make_coord(coord.m_pair, n_tile0, _);
		auto tile_coord1 = make_coord(coord.m_pair, n_tile1, _);
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
			tma_a,
			get<2>(cluster_coord),
			make_layout(size<2>(cluster_layout)),
			group_modes<0, 3>(sA),
			group_modes<0, 3>(tCgA));
		auto [tBgB0, tBsB0] = tma_partition(
			tma_b,
			get<1>(cluster_coord),
			make_layout(size<1>(cluster_layout)),
			group_modes<0, 3>(sB),
			group_modes<0, 3>(tCgB0));
		auto [tBgB1, tBsB1] = tma_partition(
			tma_b,
			get<1>(cluster_coord),
			make_layout(size<1>(cluster_layout)),
			group_modes<0, 3>(sB),
			group_modes<0, 3>(tCgB1));

		CUTE_NO_UNROLL
		for (int k_tile = 0; k_tile < KTiles; ++k_tile) {
			if constexpr (WaitDzReady) {
				if ((k_tile & 3) == 0) {
					int token_m_pair = k_tile / 4;
					std::size_t ready_index =
						static_cast<std::size_t>(wave) *
							static_cast<std::size_t>(dz_items) +
						static_cast<std::size_t>(token_m_pair) *
							static_cast<std::size_t>(dz_n_tiles) +
						static_cast<std::size_t>(coord.m_pair);
					backward_dz_tile_wait_sm100(
						dz_tile_ready + ready_index,
						Config::kClusterM);
				}
			}
			pipe.producer_acquire(state);
			if (cute::elect_one_sync()) {
				auto* barrier = pipe.producer_get_barrier(state);
				copy(
					tma_a.with(*barrier, multicast_a),
					tAgA(_, a_k_offset + k_tile),
					tAsA(_, state.index()));
				copy(
					tma_b.with(*barrier, multicast_b),
					tBgB0(_, b_k_offset + k_tile),
					tBsB0(_, state.index()));
			}
			++state;
			if (has_second) {
				pipe.producer_acquire(state);
				if (cute::elect_one_sync()) {
					auto* barrier = pipe.producer_get_barrier(state);
					copy(
						tma_a.with(*barrier, multicast_a),
						tAgA(_, a_k_offset + k_tile),
						tAsA(_, state.index()));
					copy(
						tma_b.with(*barrier, multicast_b),
						tBgB1(_, b_k_offset + k_tile),
						tBsB1(_, state.index()));
				}
				++state;
			}
		}
	}
}

template <int Compute, int KTiles, class AccPipe, class AccState>
CUTE_DEVICE void backward_mma_dw_pairs_sm100(
		typename BackwardDwTraitsSm100<Compute>::MainloopPipeline& pipe,
		typename BackwardDwTraitsSm100<Compute>::PipelineState& state,
		AccPipe& acc_pipe,
		AccState& acc_state,
		typename BackwardDwTraitsSm100<Compute>::Element* a_pointer,
		typename BackwardDwTraitsSm100<Compute>::Element* b_pointer,
		std::uint32_t tmem_base,
		int num_items,
		int first_item,
		int item_stride,
		int num_m_pairs,
		int num_n_tiles,
		std::uint64_t* diagnostics,
		int first_ready_index) {
	using Traits = BackwardDwTraitsSm100<Compute>;
	using Config = typename Traits::Config;

	typename Traits::TiledMma tiled_mma;
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	auto cta_mma = tiled_mma.get_slice(cluster_rank);
	auto sA = make_tensor(
		make_smem_ptr(a_pointer), typename Traits::SmemLayoutA{});
	auto sB = make_tensor(
		make_smem_ptr(b_pointer), typename Traits::SmemLayoutB{});
	auto tCrA = cta_mma.make_fragment_A(sA);
	auto tCrB = cta_mma.make_fragment_B(sB);
	auto cAccumulator = make_identity_tensor(make_shape(
		Int<Config::kTileM>{}, Int<Config::kTileN>{}));
	auto tCtAccumulator =
		cta_mma.make_fragment_C(cta_mma.partition_C(cAccumulator));

	for (int index = 0; index < num_items; ++index) {
		int item = first_item + index * item_stride;
		BackwardDwPairCoordSm100 coord =
			backward_dw_pair_coord_sm100(item, num_m_pairs);
		bool has_second = coord.n_tile_begin + 1 < num_n_tiles;

		acc_pipe.producer_acquire(acc_state);
		auto accumulator_state0 = acc_state;
		++acc_state;
		typename AccPipe::PipelineState accumulator_state1;
		if (has_second) {
			acc_pipe.producer_acquire(acc_state);
			accumulator_state1 = acc_state;
			++acc_state;
		}

		CUTE_NO_UNROLL
		for (int k_tile = 0; k_tile < KTiles; ++k_tile) {
			auto accumulate_tile =
				[&](const auto& accumulator_state, bool diagnose) {
					pipe.consumer_wait(state);
					if constexpr (kBackwardDiagnosticTimestampsSm100) {
						if (
							diagnose && index == 0 && k_tile == 0 &&
							(static_cast<int>(threadIdx.x) &
								(kWarpSize - 1)) == 0) {
							backward_diagnostic_first_sm100(
								diagnostics, first_ready_index);
						}
					}
					tCtAccumulator.data() =
						tmem_base +
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
							tCrA(_, _, k_block, state.index()),
							tCrB(_, _, k_block, state.index()),
							tCtAccumulator);
					}
					pipe.consumer_release(state);
					++state;
					if (k_tile + 1 == KTiles) {
						acc_pipe.producer_commit(
							accumulator_state);
					}
				};
			accumulate_tile(accumulator_state0, true);
			if (has_second) {
				accumulate_tile(accumulator_state1, false);
			}
		}
	}
}

// ───────────────────────────────────────────────────────────────────────────
// Warps 4..11 — epilogues
// ───────────────────────────────────────────────────────────────────────────

// ───────────────────────────────────────────────────────────────────────────
// The fused kernel
// ───────────────────────────────────────────────────────────────────────────

template <
	class TmaX,
	class TmaW,
	class TmaDzStore,
	class TmaDz,
	class TmaWt,
	class TmaDxStore,
	class TmaDzt,
	class TmaXt,
	class TmaDwStore,
	class TmaDwAdd>
struct BackwardTmaBundleSm100 {
	TmaX x;
	TmaW w;
	TmaDzStore dz_store;
	TmaDz dz;
	TmaWt wt;
	TmaDxStore dx_store;
	TmaDzt dzt;
	TmaXt xt;
	TmaDwStore dw_store;
	TmaDwAdd dw_add;
};

template <
	bool ReturnEntropy,
	int Compute,
	bool RequiresRemote,
	bool EnableLocalReduce,
	class CommConfig,
	class Bundle,
	int PhaseMask = kBackwardPhaseAll>
__global__ __launch_bounds__(
	BackwardGemmConfigSm100<Compute>::kNumThreads,
	1) __cluster_dims__(2, 1, 1)
void backward_gemm_tp_kernel_sm100(
		__grid_constant__ const Bundle tma,
		__grid_constant__ const BackwardGemmParamsSm100<Compute> params,
		__grid_constant__ const DxReduceWorkspace<float> comm,
		__grid_constant__ const liger_cute::detail::NvlsReduceView mapping,
		__grid_constant__ const liger_cute::detail::RemoteReduceView remote,
		__grid_constant__ const BackwardWaveWorkspaceSm100<Compute>
			wave_workspace) {
	static_assert(
		Compute == 100,
		"SM100 fused scaled linear cross entropy requires Compute=100");
	static_assert(
		CommConfig::kCompute == 100,
		"the fused SM100 backward consumes the SM100 staging contract");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Traits = BackwardGemmTraitsSm100<Compute>;
	using Config = typename Traits::Config;
	using Launch = BackwardGemmLaunchSm100<Compute>;
	using Smem = BackwardGemmSmemSm100<Compute, ReturnEntropy>;
	using Element = typename Traits::Element;
	constexpr int kPhaseBits = PhaseMask & kBackwardPhaseAll;
	constexpr bool kRunDz = (kPhaseBits & kBackwardPhaseDz) != 0;
	constexpr bool kRunDx = (kPhaseBits & kBackwardPhaseDx) != 0;
	constexpr bool kRunDw = (kPhaseBits & kBackwardPhaseDw) != 0;
	constexpr bool kAuditSkipDxDwBarrier =
		(PhaseMask & kBackwardPhaseAuditSkipDxDwBarrier) != 0;
	constexpr bool kAuditSkipDzGridBarrier =
		(PhaseMask & kBackwardPhaseAuditSkipDzGridBarrier) != 0;
	constexpr bool kAuditSkipDzDrain =
		(PhaseMask & kBackwardPhaseAuditSkipDzDrain) != 0;
	constexpr bool kAuditForceDzGridBarrier =
		(PhaseMask & kBackwardPhaseAuditForceDzGridBarrier) != 0;
	constexpr bool kAuditForceDxDwBarrier =
		(PhaseMask & kBackwardPhaseAuditForceDxDwBarrier) != 0;
	constexpr bool kNeedGridBarrier =
		kRunDz &&
		(kRunDx || kRunDw || kAuditForceDzGridBarrier) &&
		!kAuditSkipDzGridBarrier &&
		kBackwardSyncVariantSm100 != 2;
	static_assert(
		PhaseMask == kBackwardPhaseAll ||
			backward_phase_is_isolated_sm100(PhaseMask),
		"only the fused kernel or a single isolated benchmark phase are "
		"supported");

	extern __shared__ char raw_smem[];
	if constexpr (
			PhaseMask == kBackwardPhaseDz &&
			!RequiresRemote &&
			!EnableLocalReduce) {
		constexpr auto kDzMode = ReturnEntropy
			? dz_sm100::EpilogueMode::kSoftmaxGradientEntropy
			: dz_sm100::EpilogueMode::kSoftmaxGradient;
		dz_sm100::Params dz_params;
		dz_params.x = params.x;
		dz_params.weight = params.weight;
		dz_params.target = params.target;
		dz_params.grad_output = params.grad_output;
		dz_params.lse = params.lse;
		dz_params.entropy = params.entropy;
		dz_params.entropy_grad = params.entropy_grad;
		dz_params.output = params.dz_workspace;
		dz_params.tokens = params.tokens;
		dz_params.hidden = params.hidden;
		dz_params.local_vocab = params.local_vocab;
		dz_params.padded_tokens = Config::kWaveRows;
		dz_params.padded_vocab =
			Launch::padded_vocab(params.local_vocab);
		dz_params.vocab_start = params.vocab_start;
		dz_params.ignore_index = params.ignore_index;
		dz_params.inverse_temperature = params.inverse_temperature;
		dz_params.cluster_pairs = static_cast<int>(gridDim.z);
		auto& dz_smem =
			*reinterpret_cast<dz_sm100::SharedStorage<kDzMode>*>(
				raw_smem);
		dz_sm100::kernel_body<kDzMode>(
			tma.x,
			tma.w,
			tma.dz_store,
			dz_params,
			dz_smem);
		return;
	}
	Smem& smem = *reinterpret_cast<Smem*>(raw_smem);

	int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
	[[maybe_unused]] int lane =
		static_cast<int>(threadIdx.x) & (kWarpSize - 1);
	int cluster_rank = static_cast<int>(cute::block_rank_in_cluster());
	int cluster_idx = static_cast<int>(blockIdx.z);
	int num_clusters = static_cast<int>(gridDim.z);
	[[maybe_unused]] int cta =
		cluster_idx * Config::kClusterM + cluster_rank;
	int grid_ctas = num_clusters * Config::kClusterM;
	if (threadIdx.x == 0) {
		backward_diagnostic_first_sm100(
			wave_workspace.diagnostics,
			kBackwardDiagnosticKernelStart);
	}

	int padded_vocab = Launch::padded_vocab(params.local_vocab);
	int num_waves = Launch::num_waves(params.tokens);
	int dz_n_tiles = Launch::num_dz_n_tiles(params.local_vocab);
	int dz_k_tiles = Launch::num_dz_k_tiles(params.hidden);
	int dx_n_tiles = Launch::num_dx_n_tiles(params.hidden);
	int dx_wide_n_tiles =
		Launch::num_dx_wide_n_tiles(params.hidden);
	int dx_k_tiles = Launch::num_dx_k_tiles(params.local_vocab);
	int dw_m_pairs = Launch::num_dw_m_pairs(params.local_vocab);
	int dw_n_tiles = Launch::num_dw_n_tiles(params.hidden);
	constexpr int kDwKTiles = Config::kDwKTiles;

	int dz_items = Config::kMPairsPerWave * dz_n_tiles;
	int dx_items = Config::kMPairsPerWave * dx_wide_n_tiles;
	int dw_items = dw_m_pairs * ceil_div(dw_n_tiles, 2);
	int dz_local = backward_items_for_cluster_sm100(
		dz_items, cluster_idx, num_clusters);
	int dx_local = backward_items_for_cluster_sm100(
		dx_items, cluster_idx, num_clusters);
	int dw_first = static_cast<int>(
		static_cast<std::int64_t>(dw_items) * cluster_idx /
		num_clusters);
	int dw_last = static_cast<int>(
		static_cast<std::int64_t>(dw_items) * (cluster_idx + 1) /
		num_clusters);
	int dw_local = dw_last - dw_first;
	[[maybe_unused]] int dx_tiles_per_wave =
		Launch::dx_tiles_per_wave(params.hidden);

	if constexpr (kRunDz) {
		cute::prefetch_tma_descriptor(tma.x.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.w.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.dz_store.get_tma_descriptor());
	}
	if constexpr (kRunDx) {
		cute::prefetch_tma_descriptor(tma.dz.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.wt.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.dx_store.get_tma_descriptor());
	}
	if constexpr (kRunDw) {
		cute::prefetch_tma_descriptor(tma.dzt.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.xt.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.dw_store.get_tma_descriptor());
		cute::prefetch_tma_descriptor(tma.dw_add.get_tma_descriptor());
	}

	auto pipe = backward_make_pipe_sm100<Compute>(smem.pipeline);
	struct UnusedPipeline {};
	auto dx_pipe = [&]() {
		if constexpr (kRunDx) {
			return backward_make_dx_pipe_sm100<Compute>(smem.dx_pipeline);
		} else {
			return UnusedPipeline{};
		}
	}();
	auto dw_pipe = [&]() {
		if constexpr (kRunDw) {
			return backward_make_dw_pipe_sm100<Compute>(smem.dw_pipeline);
		} else {
			return UnusedPipeline{};
		}
	}();
	if (threadIdx.x == 0) {
		if constexpr (kRunDx && EnableLocalReduce) {
			CUTE_UNROLL
			for (int stage = 0; stage < kDxRingStages; ++stage) {
				cute::initialize_barrier(smem.dx_ready[stage], 1);
				cute::initialize_barrier(smem.dx_consumed[stage], 1);
			}
		}
		if constexpr (
				kRunDz && (kRunDx || kRunDw) &&
				(kBackwardSyncVariantSm100 == 1 ||
					kBackwardSyncVariantSm100 == 2)) {
			cute::initialize_barrier(smem.dz_phase_ready, 1);
		}
		if constexpr (kRunDx && kRunDw) {
			cute::initialize_barrier(smem.dx_phase_free, 1);
		}
	}
	cute::TMEM::Allocator2Sm tmem_allocator;
	// Startup-only rendezvous. Every role enters exactly once, before any
	// wave, dX reduction or remote communication begins; after this point the
	// communication warps never touch a CTA-wide barrier again.
	cute::cluster_sync();
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.allocate(Config::kTmemColumns, &smem.tmem_base);
		if (cute::elect_one_sync()) {
			smem.tmem_pair_barrier.init(1);
			smem.wave_pair_barrier.init(1);
		}
		cutlass::arch::fence_barrier_init();
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();
	uint32_t tmem_base = smem.tmem_base;

	if constexpr (
		RequiresRemote || kBackwardDiagnosticChunkPipelineSm100) {
	constexpr bool kEnableComm = EnableLocalReduce;
	// The full-grid software barriers exist only to publish and then recycle
	// the dZ wave workspace between phases. An isolated benchmark phase has no
	// cross-phase producer/consumer relationship on that workspace, so the
	// barriers carry no meaning there and are compiled out; the measurement is
	// then GEMM plus its required epilogue, with no cross-phase scaffolding.
	// The CTA-local staging-ring handshake is part of the dX epilogue, not of
	// the reduction, so the benchmark keeps it and only drops the NVLS / IB
	// traffic. Warp 0 then still retires slots and the epilogue still takes
	// the production barrier sequence.
	constexpr bool kDxRingHandshake = EnableLocalReduce && kRunDx;

	// The dX schedule. The chunk-granular, one-chunk-deferred pipeline is the
	// production schedule whenever an inter-host ring is configured; the
	// diagnostic knob forces the same state machine on a single host so it can
	// be validated without a second host.
	constexpr bool kChunkPipeline = true;

	// ── Communication warps ────────────────────────────────────────────────
	//
	// Warp 0 runs a three-stage chunk pipeline. Stage R is tile granular and
	// releases each staging slot the instant its NVLS reduce-scatter retires.
	// Stage P is chunk granular: it publishes this CTA's contribution to the
	// collectively packed chunk so warp 1 can start the inter-host message.
	// Stage F is chunk granular *and deferred by one chunk*, so chunk k's
	// all-gather/scatter overlaps chunk k+1's dZ GEMM instead of gating it.
	if (warp_id == Config::kDxLocalReduceWarp) {
		if constexpr (!kEnableComm) {
			if constexpr (!kDxRingHandshake) return;
			// Benchmark: retire the staging ring with no reduction traffic.
			int drain_index = 0;
			for (int chunk = 0; chunk < num_waves; ++chunk) {
				for (int index = 0; index < dx_local;
						++index, ++drain_index) {
					int stage = drain_index % kDxRingStages;
					int pass = drain_index / kDxRingStages;
					if (lane == 0) {
						cute::wait_barrier(
							smem.dx_ready[stage], pass & 1);
					}
					__syncwarp();
					if (lane == 0) {
						cute::arrive_barrier(smem.dx_consumed[stage]);
					}
					__syncwarp();
				}
			}
			return;
		}
		const std::size_t packed_tile_elements =
			static_cast<std::size_t>(CommConfig::kTileElements) /
			static_cast<std::size_t>(mapping.size);

		// This CTA's deterministic tile sequence inside one chunk. Every PE
		// and every CTA derives it from the shape alone, so the packed chunk
		// is the same disjoint partition everywhere.
		auto chunk_wide_tile = [&](int index) {
			return backward_pair_m_fast_sm100(
				cluster_idx + index * num_clusters,
				Config::kMPairsPerWave);
		};
		auto chunk_durable_tile = [&](int chunk,
				const BackwardPairCoordSm100& coord,
				int panel) {
			int n_tile = coord.n_tile * 2 + panel;
			return backward_dx_durable_tile_sm100(
				chunk,
				dx_tiles_per_wave,
				coord.m_pair * Config::kClusterM + cluster_rank,
				n_tile,
				dx_n_tiles);
		};
		// Stage F. Replays the chunk's own tile sequence, so the NVLS signal
		// slot and pass of each all-gather match the reduce-scatter that
		// produced it and every PE issues the same collective order.
		auto finalize_chunk = [&](int chunk, int first_local) {
			int replay = first_local;
			for (int index = 0; index < dx_local; ++index) {
				BackwardPairCoordSm100 coord =
					chunk_wide_tile(index);
				for (int panel = 0; panel < 2; ++panel) {
					int n_tile = coord.n_tile * 2 + panel;
					if (n_tile >= dx_n_tiles) continue;
					std::size_t durable_tile =
						chunk_durable_tile(chunk, coord, panel);
					backward_dx_allgather_scatter_warp_sm100<
						CommConfig, Compute>(
							params,
							comm,
							mapping,
							wave_workspace.packed_shard +
								durable_tile * packed_tile_elements,
							durable_tile,
							cta,
							replay % kDxRingStages,
							chunk,
							replay / kDxRingStages,
							coord.m_pair * Config::kClusterM +
								cluster_rank,
							n_tile);
					++replay;
				}
			}
			if (lane == 0) {
				backward_diagnostic_max_sm100(
					wave_workspace.diagnostics,
					kBackwardDiagnosticLocalStoreEnd);
			}
		};

		int local_index = 0;
		[[maybe_unused]] int pending_chunk = -1;
		[[maybe_unused]] int pending_first = 0;

		for (int chunk = 0; chunk < num_waves; ++chunk) {
			if constexpr (kChunkPipeline) {
				// Stage F for the previous chunk. This runs while the compute
				// warps are still in this chunk's dZ GEMM: dX(k) is finalized
				// before its slots could be reused, never before compute
				// advances.
				if (pending_chunk >= 0) {
					backward_wait_ge_warp_sm100(
						wave_workspace.dx_remote_ready,
						backward_wave_epoch_sm100(
							*wave_workspace.launch_epoch,
							kBackwardDxRemoteEpochSuffixSm100,
							pending_chunk));
					finalize_chunk(pending_chunk, pending_first);
					pending_chunk = -1;
				}
			}

			// Stage R. Tile granular for both dispatches.
			[[maybe_unused]] int first_local = local_index;
			for (int index = 0; index < dx_local; ++index) {
				BackwardPairCoordSm100 coord =
					chunk_wide_tile(index);
				for (int panel = 0; panel < 2; ++panel) {
					int n_tile = coord.n_tile * 2 + panel;
					if (n_tile >= dx_n_tiles) continue;
					int stage = local_index % kDxRingStages;
					int pass = local_index / kDxRingStages;

					if (lane == 0) {
						cute::wait_barrier(
							smem.dx_ready[stage], pass & 1);
					}
					__syncwarp();
					if (
						chunk == 0 && index == 0 && panel == 0 &&
						lane == 0) {
						backward_diagnostic_first_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticLocalFirstReady);
					}

					std::size_t durable_tile =
						chunk_durable_tile(chunk, coord, panel);
					float* packed = wave_workspace.packed_shard +
						durable_tile * packed_tile_elements;

					backward_dx_reduce_scatter_warp_sm100<
						CommConfig, Compute>(
							comm,
							mapping,
							packed,
							cta,
							stage,
							chunk,
							pass);

					__syncwarp();
					if (lane == 0) {
						cute::arrive_barrier(
							smem.dx_consumed[stage]);
					}
					__syncwarp();
					++local_index;
				}
			}

			if constexpr (kChunkPipeline) {
				// Stage P. The chunk is now completely reduce-scattered by
				// this CTA; warp 1 starts the IB message once every CTA has
				// arrived.
				__syncwarp();
				if (lane == 0) {
					liger_cute::detail::publish_local_reduce_source();
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticLocalReduceEnd);
					backward_grid_barrier_arrive_sm100(
						wave_workspace.dx_scatter_ready);
				}
				__syncwarp();
				pending_chunk = chunk;
				pending_first = first_local;
			}
		}

		if constexpr (kChunkPipeline) {
			// Drain the final chunk; nothing else is left to overlap with.
			if (pending_chunk >= 0) {
				backward_wait_ge_warp_sm100(
					wave_workspace.dx_remote_ready,
					backward_wave_epoch_sm100(
						*wave_workspace.launch_epoch,
						kBackwardDxRemoteEpochSuffixSm100,
						pending_chunk));
				finalize_chunk(pending_chunk, pending_first);
			}
		}
		return;
	}

	if (warp_id == Config::kRemoteCommunicationWarp) {
		if constexpr (!kEnableComm) return;
		if constexpr (kChunkPipeline) {
			// CTA 0 owns the verified matching-rank ring transport. Once the
			// peer contribution is visible, warp 1 from every resident CTA
			// merges a global-strided slice and contributes one HBM atomic
			// arrival. CTA 0 acknowledges the ring only after all CTAs retire.
			std::size_t chunk_elements =
				static_cast<std::size_t>(dx_tiles_per_wave) *
				static_cast<std::size_t>(
					CommConfig::kTileElements) /
				static_cast<std::size_t>(mapping.size);
			for (int chunk = 0; chunk < num_waves; ++chunk) {
				std::uint64_t remote_epoch =
					backward_wave_epoch_sm100(
						*wave_workspace.launch_epoch,
						kBackwardDxRemoteEpochSuffixSm100,
						chunk);
				float* chunk_shard =
					wave_workspace.packed_shard +
					static_cast<std::size_t>(
						backward_dx_chunk_slot_sm100(chunk)) *
						chunk_elements;
				if (cta == 0) {
					// Chunk-granular gate: every CTA of the grid has retired
					// all of its chunk-k tiles.
					backward_wait_ge_warp_sm100(
						wave_workspace.dx_scatter_ready,
						static_cast<unsigned long long>(chunk + 1) *
							static_cast<unsigned long long>(grid_ctas));
					if (chunk == 0 && lane == 0) {
						backward_diagnostic_first_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticRemoteStart);
					}
					if constexpr (RequiresRemote) {
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)
						backward_dx_ring_transport_warp_sm100(
							remote,
							wave_workspace.launch_epoch,
							chunk_shard,
							chunk_elements,
							chunk);
#else
						__trap();
#endif
					} else {
						// Diagnostic schedule: no transport, the node-local
						// reduce-scatter already left the complete shard here.
						(void)chunk_shard;
						(void)chunk_elements;
					}
					if (lane == 0) {
						backward_diagnostic_max_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticRemoteTransportEnd);
					}
					__syncwarp();
					if (lane == 0) {
						liger_cute::detail::publish_local_reduce_source();
						backward_store_release_system_sm100(
							wave_workspace.dx_remote_received,
							remote_epoch);
					}
					__syncwarp();
				}

				backward_wait_epoch_warp_sm100(
					wave_workspace.dx_remote_received,
					remote_epoch);
				if constexpr (RequiresRemote) {
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)
					const float* contribution =
						liger_cute::detail::remote_ring_inbox_slot(
							remote, 0, chunk);
					backward_dx_remote_merge_workers_sm100(
						chunk_shard,
						contribution,
						chunk_elements,
						backward_remote_merge_worker_sm100(
							cta, lane),
						backward_remote_merge_workers_sm100(
							grid_ctas));
#else
					__trap();
#endif
				}
				__threadfence();
				__syncwarp();
				if (lane == 0) {
					backward_grid_barrier_arrive_sm100(
						wave_workspace.dx_remote_merge_arrived);
				}
				__syncwarp();

				if (cta == 0) {
					backward_grid_wait_warp_sm100(
						wave_workspace.dx_remote_merge_arrived,
						backward_remote_merge_target_sm100(
							chunk, grid_ctas));
					if constexpr (RequiresRemote) {
#if defined(LIGER_CUTE_FSLCE_SM100_BACKWARD_ENABLE_NVSHMEM)
						backward_dx_ring_finish_warp_sm100(
							remote,
							wave_workspace.launch_epoch,
							chunk,
							chunk + 1 == num_waves);
#else
						__trap();
#endif
					}
					if (lane == 0) {
						backward_diagnostic_max_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticRemoteEnd);
						liger_cute::detail::publish_local_reduce_source();
						backward_store_release_system_sm100(
							wave_workspace.dx_remote_ready,
							remote_epoch);
					}
					__syncwarp();
				}
			}
		}
		return;
	}
	} else {
	if (warp_id == Config::kDxLocalReduceWarp) {
		if constexpr (!EnableLocalReduce) {
			return;
		}
		if constexpr (EnableLocalReduce) {
		static_assert(
			PhaseMask == kBackwardPhaseAll,
			"node-local dX reduction is only enabled for the fused kernel");
		const std::size_t packed_tile_elements =
			static_cast<std::size_t>(CommConfig::kTileElements) /
			static_cast<std::size_t>(mapping.size);
		int local_index = 0;
		for (int wave = 0; wave < num_waves; ++wave) {
			for (int index = 0; index < dx_local; ++index) {
				int item = cluster_idx + index * num_clusters;
				BackwardPairCoordSm100 coord =
					backward_pair_m_fast_sm100(
						item, Config::kMPairsPerWave);
				int m_tile =
					coord.m_pair * Config::kClusterM + cluster_rank;
				for (int panel = 0; panel < 2; ++panel) {
					int n_tile = coord.n_tile * 2 + panel;
					if (n_tile >= dx_n_tiles) continue;
					int stage = local_index % kDxRingStages;
					int pass = local_index / kDxRingStages;

					if (lane == 0) {
						cute::wait_barrier(
							smem.dx_ready[stage], pass & 1);
					}
					__syncwarp();
					if (
						wave == 0 && index == 0 && panel == 0 &&
						lane == 0) {
						backward_diagnostic_first_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticLocalFirstReady);
					}

					std::size_t durable_tile =
						backward_dx_durable_tile_sm100(
							wave,
							dx_tiles_per_wave,
							m_tile,
							n_tile,
							dx_n_tiles);
					float* packed =
						wave_workspace.packed_shard +
						durable_tile * packed_tile_elements;
					const float* final_source = packed;
					if (mapping.size == 1) {
						final_source =
							comm.partial +
							dx_slot_offset<CommConfig>(
								cta, 0, stage);
					} else {
						backward_dx_reduce_scatter_warp_sm100<
							CommConfig, Compute>(
								comm,
								mapping,
								packed,
								cta,
								stage,
								wave,
								pass);
					}
					if (lane == 0) {
						backward_diagnostic_max_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticLocalReduceEnd);
					}

					backward_dx_allgather_scatter_warp_sm100<
						CommConfig, Compute>(
							params,
							comm,
							mapping,
							final_source,
							durable_tile,
							cta,
							stage,
							wave,
							pass,
							m_tile,
							n_tile);
					if (lane == 0) {
						backward_diagnostic_max_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticLocalStoreEnd);
					}
					__syncwarp();
					if (lane == 0) {
						cute::arrive_barrier(
							smem.dx_consumed[stage]);
					}
					__syncwarp();
					++local_index;
				}
			}
		}
		}
		return;
	}
	if (warp_id == Config::kRemoteCommunicationWarp) {
		return;
	}
	}

	// ── Compute warps (2..11) ─────────────────────────────────────────────
	bool is_mma_warp = warp_id == Config::kUmmaWarp;
	bool is_leader_cta = cluster_rank == 0;
	bool is_epilogue = warp_id >= Config::kFirstEpilogueWarp &&
		warp_id <= Config::kLastEpilogueWarp;
	bool is_dx_epilogue =
		warp_id >= Config::kFirstEpilogueWarp &&
		warp_id < Config::kFirstEpilogueWarp + 4;
	bool is_dw_epilogue =
		warp_id >= Config::kFirstEpilogueWarp &&
		warp_id < Config::kFirstEpilogueWarp + 4;
	int tid_in_epi = static_cast<int>(threadIdx.x) -
		Config::kFirstEpilogueWarp * kWarpSize;
	int warpgroup = is_epilogue ? tid_in_epi / Config::kWarpgroupSize : 0;
	int tid_in_warpgroup =
		is_epilogue ? tid_in_epi % Config::kWarpgroupSize : 0;
	int warpgroup_barrier = Config::kWarpgroup0BarrierId + warpgroup;

	using AccPipe = typename Traits::AccumulatorPipeline;
	typename AccPipe::Params acc_params;
	// Warp 2 owns the TMA producer only; it must stay out of the accumulator
	// pipeline and out of every warp-3..11 named barrier, otherwise the
	// 288-thread rendezvous silently gains 32 arrivals.
	acc_params.role = is_mma_warp && is_leader_cta
		? AccPipe::ThreadCategory::Producer
		: (warp_id >= Config::kUmmaWarp
			? AccPipe::ThreadCategory::Consumer
			: AccPipe::ThreadCategory::NonParticipant);
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = Config::kClusterM;
	acc_params.initializing_warp = Config::kFirstEpilogueWarp;
	AccPipe acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	using DxTraits = BackwardDxTraitsSm100;
	using DxAccPipe = typename DxTraits::AccumulatorPipeline;
	typename DxAccPipe::Params dx_acc_params;
	dx_acc_params.role = is_mma_warp && is_leader_cta
		? DxAccPipe::ThreadCategory::Producer
		: (is_dx_epilogue
			? DxAccPipe::ThreadCategory::Consumer
			: DxAccPipe::ThreadCategory::NonParticipant);
	dx_acc_params.producer_arv_count = 1;
	dx_acc_params.consumer_arv_count = Config::kClusterM;
	dx_acc_params.initializing_warp = Config::kFirstEpilogueWarp;
	DxAccPipe dx_acc_pipe(
		smem.dx_acc_pipe,
		dx_acc_params,
		typename DxTraits::ClusterShape{});
	auto dx_acc_prod_state =
		cutlass::make_producer_start_state<DxAccPipe>();
	typename DxAccPipe::PipelineState dx_acc_cons_state;

	using DwTraits = BackwardDwTraitsSm100<Compute>;
	using DwAccPipe = typename DwTraits::AccumulatorPipeline;
	typename DwAccPipe::Params dw_acc_params;
	dw_acc_params.role = is_mma_warp && is_leader_cta
		? DwAccPipe::ThreadCategory::Producer
		: (is_dw_epilogue
			? DwAccPipe::ThreadCategory::Consumer
			: DwAccPipe::ThreadCategory::NonParticipant);
	dw_acc_params.producer_arv_count = 1;
	dw_acc_params.consumer_arv_count = Config::kClusterM;
	dw_acc_params.initializing_warp = Config::kFirstEpilogueWarp;
	DwAccPipe dw_acc_pipe(
		smem.dw_acc_pipe,
		dw_acc_params,
		typename DwTraits::ClusterShape{});
	auto dw_acc_prod_state =
		cutlass::make_producer_start_state<DwAccPipe>();
	typename DwAccPipe::PipelineState dw_acc_cons_state;
	if (warp_id >= Config::kUmmaWarp) {
		cutlass::arch::NamedBarrier::sync(
			Config::kMmaEpilogueThreads, Config::kMmaEpilogueBarrierId);
	}
	if (is_mma_warp || is_dx_epilogue) {
		cutlass::arch::NamedBarrier::sync(
			Config::kDxMmaEpilogueThreads,
			Config::kDxMmaEpilogueBarrierId);
	}
	if (is_mma_warp || is_dw_epilogue) {
		cutlass::arch::NamedBarrier::sync(
			Config::kDwMmaEpilogueThreads,
			Config::kDwMmaEpilogueBarrierId);
	}

	typename Traits::PipelineState state;
	typename DxTraits::PipelineState dx_state;
	typename DwTraits::PipelineState dw_state;
	if (warp_id == Config::kTmaWarp) {
		state = cutlass::make_producer_start_state<
			typename Traits::MainloopPipeline>();
		dx_state = cutlass::make_producer_start_state<
			typename DxTraits::MainloopPipeline>();
		dw_state = cutlass::make_producer_start_state<
			typename DwTraits::MainloopPipeline>();
	}

	float exp_scale = params.inverse_temperature * kBackwardLog2ESm100;
	unsigned long long barrier_generation = 0;
	int dx_local_index = 0;
	for (int wave = 0; wave < num_waves; ++wave) {
		int dz_slot =
			kBackwardSyncVariantSm100 == 3
			? wave % Config::kDzWorkspaceSlots
			: 0;
		if constexpr (kRunDz) {  // compiled out by the isolated dX / dW benchmarks
			// ── phase dZ ──────────────────────────────────────────────────────
			if (warp_id == Config::kTmaWarp) {
				auto mX = tma.x.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.tokens),
					static_cast<int64_t>(params.hidden)));
				auto mW = tma.w.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.local_vocab),
					static_cast<int64_t>(params.hidden)));
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzTmaStart);
				}
				backward_produce_phase_sm100<
					Compute,
					Traits,
					typename Traits::SmemLayoutAK,
					typename Traits::SmemLayoutBK,
					typename Traits::TiledMmaDz>(
						pipe,
						state,
						smem.dz_a_data(),
						smem.dz_b_data(),
						tma.x,
						tma.w,
						mX,
						mW,
						dz_local,
						cluster_idx,
						num_clusters,
						dz_k_tiles,
						[&](int item) {
							return wave * Config::kMPairsPerWave +
								backward_pair_m_fast_sm100(
									item, Config::kMPairsPerWave).m_pair;
						},
						[&](int item) {
							return backward_pair_m_fast_sm100(
								item, Config::kMPairsPerWave).n_tile;
						},
						[](int, int) { return 0; },
						[](int, int) {});
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzTmaEnd);
				}
			} else if (is_mma_warp && is_leader_cta) {
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzMmaStart);
				}
				backward_mma_phase_sm100<
					Compute,
					Traits,
					typename Traits::SmemLayoutAK,
					typename Traits::SmemLayoutBK,
					typename Traits::TiledMmaDz>(
						pipe,
						state,
						acc_pipe,
						acc_prod_state,
						smem.dz_a_data(),
						smem.dz_b_data(),
						tmem_base,
						dz_local,
						dz_k_tiles,
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzMmaFirstReady);
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzMmaEnd);
				}
			} else if (is_epilogue) {
				if (tid_in_epi == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzEpiStart);
				}
				typename Traits::TiledMmaDz epilogue_mma;
				auto cta_mma_epi =
					epilogue_mma.get_slice(cluster_rank);
				auto cAccFull = make_identity_tensor(
					make_shape(
						Int<Config::kTileM>{},
						Int<Config::kTileN>{}));
				auto tCtAcc =
					cta_mma_epi.make_fragment_C(
						cta_mma_epi.partition_C(cAccFull));
				tCtAcc.data() = tmem_base;
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
					size(tmem_warp_layout) !=
					cosize(tmem_warp_layout);
				auto sDzStore = make_tensor(
					make_smem_ptr(smem.dz_store_data(warpgroup)),
					typename Traits::SmemLayoutStoreSlot{});
				auto mDzStore =
					tma.dz_store.get_tma_tensor(make_shape(
						static_cast<int64_t>(
							Config::kWaveRows *
							Config::kDzWorkspaceSlots),
						static_cast<int64_t>(padded_vocab)));
				auto cta_dz_store =
					tma.dz_store.get_slice(Int<0>{});
				auto tSsS_dz =
					cta_dz_store.partition_S(sDzStore);
				auto tSgS_dz =
					cta_dz_store.partition_D(local_tile(
						mDzStore,
						make_tile(
							Int<Config::kCtaTileM>{},
							Int<Config::kEpilogueChunkN>{}),
						make_coord(_, _)));
				for (int index = 0; index < dz_local; ++index) {
					int item = cluster_idx + index * num_clusters;
					BackwardPairCoordSm100 coord = backward_pair_m_fast_sm100(
						item, Config::kMPairsPerWave);
					int m_tile = coord.m_pair * Config::kClusterM + cluster_rank;
					int row_base =
						(wave * Config::kMTilesPerWave + m_tile) *
						Config::kCtaTileM;

					// Per-row softmax / entropy gradient metadata for this CTA's
					// 128 token rows, recomputed once per work item.
					for (int row = tid_in_epi; row < Config::kCtaTileM;
							row += Config::kEpilogueThreads) {
						int token = row_base + row;
						float scale = 0.0f;
						float exp_bias = 0.0f;
						float entropy_bias = 0.0f;
						float entropy_slope = 0.0f;
						int target_local = -1;
						if (token < params.tokens) {
							std::int64_t target_id = params.target[token];
							if (target_id != params.ignore_index) {
								float lse = params.lse[token];
								scale = params.grad_output[token];
								exp_bias = -lse * kBackwardLog2ESm100;
								if constexpr (ReturnEntropy) {
									float entropy = params.entropy[token];
									float entropy_scale =
										params.entropy_grad[token];
									entropy_bias = fmaf(
										lse - entropy, entropy_scale, scale);
									entropy_slope =
										-params.inverse_temperature *
										entropy_scale;
								}
								std::int64_t local =
									target_id - params.vocab_start;
								if (local >= 0 &&
									local < static_cast<std::int64_t>(
										params.local_vocab)) {
									target_local = static_cast<int>(local);
								}
							}
						}
						smem.phase.dz.row_scale[row] = scale;
						smem.phase.dz.row_exp_bias[row] = exp_bias;
						smem.phase.dz.row_target[row] = target_local;
						if constexpr (ReturnEntropy) {
							smem.phase.dz.row_entropy_bias[row] = entropy_bias;
							smem.phase.dz.row_entropy_slope[row] = entropy_slope;
						}
					}
					cutlass::arch::NamedBarrier::sync(
						Config::kEpilogueThreads, Config::kEpilogueBarrierId);

					acc_pipe.consumer_wait(acc_cons_state);
					if (index == 0 && tid_in_epi == 0) {
						backward_diagnostic_first_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticDzEpiFirstReady);
					}
					tCtAcc.data() = tmem_base +
						static_cast<uint32_t>(
							acc_cons_state.index() * Config::kTmemStageColumns);
					auto acc_mn_stage = tCtAcc(make_coord(_, _), _0{}, _0{});
					auto tAccEpiStage = flat_divide(acc_mn_stage, epi_tile);
					auto tTR_tAcc = thr_t2r.partition_S(tAccEpiStage);

					CUTE_UNROLL
					for (int round = 0; round < Config::kChunksPerWarpgroup;
							++round) {
						int chunk =
							warpgroup * Config::kChunksPerWarpgroup + round;
						auto tAccChunk = tTR_tAcc(_, _, _, _0{}, chunk);
						bool issue_tmem_load = true;
						if constexpr (kPredicateTmemLoad) {
							int subpart = (tAccChunk.data().dp_ / 32) % 4;
							issue_tmem_load =
								tid_in_warpgroup / kWarpSize == subpart;
						}
						int buf =
							(index * Config::kChunksPerWarpgroup + round) & 1;
						if (issue_tmem_load) {
							copy(t2r, tAccChunk, tTR_rAcc);
							cutlass::arch::fence_view_async_tmem_load();
						}
						// Double buffered: only the store from two chunks back
						// must have drained, so one TMA store stays in flight
						// while the other slot is refilled. Blocking on
						// tma_store_wait<0> here serialised the whole epilogue
						// against the store and cost dZ a large part of its
						// achievable rate.
						if (tid_in_warpgroup == 0) {
							cute::tma_store_wait<0>();
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize, warpgroup_barrier);

						int vocab_base = coord.n_tile * Config::kTileN +
							chunk * Config::kEpilogueChunkN;
						if (issue_tmem_load) {
							CUTE_UNROLL
							for (int i = 0; i < size(tTR_rAcc); ++i) {
								int row = get<0>(tTR_cChunk(i));
								int column = get<1>(tTR_cChunk(i));
								int global_column = vocab_base + column;
								float value = 0.0f;
								if (global_column < params.local_vocab) {
									float scale =
										smem.phase.dz.row_scale[row];
									float exp_bias =
										smem.phase.dz.row_exp_bias[row];
									int target =
										smem.phase.dz.row_target[row];
									float logit = tTR_rAcc(i);
									if constexpr (ReturnEntropy) {
										float entropy_bias =
											smem.phase.dz.row_entropy_bias[row];
										float entropy_slope =
											smem.phase.dz.row_entropy_slope[row];
										if (scale != 0.0f ||
											entropy_slope != 0.0f) {
											float probability =
												backward_exp2_sm100(fmaf(
													logit, exp_scale, exp_bias));
											value = probability * fmaf(
												logit,
												entropy_slope,
												entropy_bias);
										}
									} else if (scale != 0.0f) {
										value = backward_exp2_sm100(fmaf(
											logit, exp_scale, exp_bias)) * scale;
									}
									if (global_column == target) value -= scale;
									value *= params.inverse_temperature;
								}
								sDzStore(row, column) =
									static_cast<Element>(value);
							}
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize, warpgroup_barrier);
						if (tid_in_warpgroup == 0) {
							cute::tma_store_fence();
							copy(
								tma.dz_store,
								tSsS_dz,
								tSgS_dz(
									_, _, _,
									dz_slot *
											Config::kMTilesPerWave +
										m_tile,
									coord.n_tile *
										(Config::kTileN /
											Config::kEpilogueChunkN) +
										chunk));
							cute::tma_store_arrive();
						}
					}
					if constexpr (kBackwardSyncVariantSm100 == 2) {
						if (tid_in_warpgroup == 0) {
							cute::tma_store_wait<0>();
						}
					}
					cutlass::arch::NamedBarrier::sync(
						Config::kEpilogueThreads, Config::kEpilogueBarrierId);
					if (tid_in_epi == 0) {
						if constexpr (kBackwardSyncVariantSm100 == 2) {
							std::size_t ready_index =
								static_cast<std::size_t>(wave) *
									static_cast<std::size_t>(dz_items) +
								static_cast<std::size_t>(coord.m_pair) *
									static_cast<std::size_t>(dz_n_tiles) +
								static_cast<std::size_t>(coord.n_tile);
							backward_dz_tile_publish_sm100(
								wave_workspace.dz_tile_ready +
									ready_index);
						}
						acc_pipe.consumer_release(acc_cons_state);
					}
					++acc_cons_state;
				}
				// Drain every dZ TMA store before the grid barrier publishes the
				// workspace.
				if constexpr (!kAuditSkipDzDrain) {
					if (tid_in_warpgroup == 0) {
						cute::tma_store_wait<0>();
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kEpilogueThreads, Config::kEpilogueBarrierId);
				if (tid_in_epi == 0) {
					asm volatile("fence.proxy.async.global;" ::: "memory");
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzEpiEnd);
					if constexpr (
						kBackwardSyncVariantSm100 == 1 ||
						kBackwardSyncVariantSm100 == 2) {
						cute::arrive_barrier(smem.dz_phase_ready);
					}
				}
			}
		}

		// dZ publication.
		if constexpr (kNeedGridBarrier) {
			[[maybe_unused]] std::uint64_t grid_wait_begin = 0;
			if constexpr (kBackwardDiagnosticTimestampsSm100) {
				if (lane == 0) {
					grid_wait_begin = backward_globaltimer_sm100();
				}
			}
			++barrier_generation;
			if constexpr (
				kBackwardSyncVariantSm100 == 1 ||
				kBackwardSyncVariantSm100 == 2) {
				backward_grid_barrier_pipeline_sm100<Compute>(
					wave_workspace.grid_barrier,
					barrier_generation *
						static_cast<unsigned long long>(grid_ctas),
					warp_id,
					smem.dz_phase_ready,
					wave);
			} else {
				backward_grid_barrier_sm100<Compute>(
					wave_workspace.grid_barrier,
					barrier_generation *
						static_cast<unsigned long long>(grid_ctas),
					warp_id);
			}
			if constexpr (kBackwardDiagnosticTimestampsSm100) {
				if (lane == 0) {
					backward_diagnostic_duration_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDzGridWaitMax,
						grid_wait_begin);
				}
			}
		}
		if constexpr (
			kBackwardSyncVariantSm100 == 2 &&
			kRunDz && (kRunDx || kRunDw)) {
			if (warp_id == Config::kTmaWarp) {
				if (lane == 0) {
					cute::wait_barrier(
						smem.dz_phase_ready, wave & 1);
				}
				__syncwarp();
			}
		}

		if constexpr (kRunDx) {  // compiled out by the isolated dZ / dW benchmarks
			// ── phase dX ──────────────────────────────────────────────────────
			if (warp_id == Config::kTmaWarp) {
				auto mDz = tma.dz.get_tma_tensor(make_shape(
					static_cast<int64_t>(
						Config::kWaveRows *
						Config::kDzWorkspaceSlots),
					static_cast<int64_t>(padded_vocab)));
				auto mWt = tma.wt.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.hidden),
					static_cast<int64_t>(params.local_vocab)));
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxTmaStart);
				}
				backward_produce_phase_sm100<
					Compute,
					DxTraits,
					typename DxTraits::SmemLayoutA,
					typename DxTraits::SmemLayoutB,
					typename DxTraits::TiledMma>(
						dx_pipe,
						dx_state,
						smem.dx_a_data(),
						smem.dx_b_data(),
						tma.dz,
						tma.wt,
						mDz,
						mWt,
						dx_local,
						cluster_idx,
						num_clusters,
						dx_k_tiles,
						[&](int item) {
							return dz_slot *
									Config::kMPairsPerWave +
								backward_pair_m_fast_sm100(
									item,
									Config::kMPairsPerWave).m_pair;
						},
						[&](int item) {
							return backward_pair_m_fast_sm100(
								item,
								Config::kMPairsPerWave).n_tile;
						},
						[](int, int) { return 0; },
						[&](int item, int k_tile) {
							if constexpr (
								kBackwardSyncVariantSm100 == 2 &&
								kRunDz) {
								if ((k_tile & 3) == 0) {
									BackwardPairCoordSm100 ready_coord =
										backward_pair_m_fast_sm100(
											item,
											Config::kMPairsPerWave);
									int dz_n_tile = k_tile / 4;
									std::size_t ready_index =
										static_cast<std::size_t>(wave) *
											static_cast<std::size_t>(
												dz_items) +
										static_cast<std::size_t>(
											ready_coord.m_pair) *
											static_cast<std::size_t>(
												dz_n_tiles) +
										static_cast<std::size_t>(
											dz_n_tile);
									backward_dz_tile_wait_sm100(
										wave_workspace.dz_tile_ready +
											ready_index,
										Config::kClusterM);
								}
							}
						});
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxTmaEnd);
				}
			} else if (is_mma_warp && is_leader_cta) {
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxMmaStart);
				}
				backward_mma_phase_sm100<
					Compute,
					DxTraits,
					typename DxTraits::SmemLayoutA,
					typename DxTraits::SmemLayoutB,
					typename DxTraits::TiledMma>(
						dx_pipe,
						dx_state,
						dx_acc_pipe,
						dx_acc_prod_state,
						smem.dx_a_data(),
						smem.dx_b_data(),
						tmem_base,
						dx_local,
						dx_k_tiles,
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxMmaFirstReady);
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxMmaEnd);
				}
			} else if (is_dx_epilogue) {
				if (tid_in_epi == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxEpiStart);
				}
				typename DxTraits::TiledMma dx_epilogue_mma;
				auto dx_cta_mma_epi =
					dx_epilogue_mma.get_slice(cluster_rank);
				auto dx_cAccFull = make_identity_tensor(
					make_shape(_256{}, _512{}));
				auto dx_tCtAcc =
					dx_cta_mma_epi.make_fragment_C(
						dx_cta_mma_epi.partition_C(dx_cAccFull));
				dx_tCtAcc.data() = tmem_base;
				auto dx_epi_tile = make_tile(_128{}, _64{});
				auto dx_acc_mn =
					dx_tCtAcc(make_coord(_, _), _0{}, _0{});
				auto dx_tAccEpi =
					flat_divide(dx_acc_mn, dx_epi_tile);
				auto dx_t2r = make_tmem_copy(
					::liger::TmemLoadOp<64>{},
					dx_tAccEpi(_, _, _0{}, _0{}));
				auto dx_thr_t2r =
					dx_t2r.get_slice(tid_in_warpgroup);
				auto dx_cSlab =
					make_identity_tensor(make_shape(_128{}, _64{}));
				auto dx_tTR_cSlab =
					dx_thr_t2r.partition_D(dx_cSlab);
				auto dx_tTR_rAcc =
					make_tensor<float>(shape(dx_tTR_cSlab));
				Layout dx_tmem_warp_layout =
					typename decltype(make_tmem_warp_partitioner(
						dx_tAccEpi(_, _, _0{}, _0{})))::TiledLayout_TV{};
				constexpr bool kPredicateDxTmemLoad =
					size(dx_tmem_warp_layout) !=
					cosize(dx_tmem_warp_layout);
				auto sDxStore = make_tensor(
					make_smem_ptr(smem.dx_store_data()),
					typename DxTraits::SmemLayoutStore{});
				auto mDxStore =
					tma.dx_store.get_tma_tensor(make_shape(
						static_cast<int64_t>(
							wave_workspace.staging_rows),
						static_cast<int64_t>(
							CommConfig::kTileN)));
				auto cta_dx_store =
					tma.dx_store.get_slice(Int<0>{});
				auto tSgS_dx =
					cta_dx_store.partition_D(local_tile(
						mDxStore,
						make_tile(_16{}, _64{}),
						make_coord(_, _)));
				int local_index = dx_local_index;
				for (int index = 0; index < dx_local; ++index) {
					int item = cluster_idx + index * num_clusters;
					BackwardPairCoordSm100 coord =
						backward_pair_m_fast_sm100(
							item, Config::kMPairsPerWave);
					int m_tile =
						coord.m_pair * Config::kClusterM + cluster_rank;
					if constexpr (EnableLocalReduce) {
						if (tid_in_epi == 0) {
							for (int panel = 0; panel < 2; ++panel) {
								int n_tile =
									coord.n_tile * 2 + panel;
								if (n_tile >= dx_n_tiles) continue;
								int panel_index =
									local_index + panel;
								int stage =
									panel_index % kDxRingStages;
								int pass =
									panel_index / kDxRingStages;
								if (pass > 0) {
									cute::wait_barrier(
										smem.dx_consumed[stage],
										(pass - 1) & 1);
								}
							}
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kDxEpilogueThreads,
							Config::kEpilogueBarrierId);
					}

					dx_acc_pipe.consumer_wait(dx_acc_cons_state);
					if (index == 0 && tid_in_epi == 0) {
						backward_diagnostic_first_sm100(
							wave_workspace.diagnostics,
							kBackwardDiagnosticDxEpiFirstReady);
					}
					dx_tCtAcc.data() = tmem_base;
					auto dx_acc_stage =
						dx_tCtAcc(make_coord(_, _), _0{}, _0{});
					auto dx_tAccStage =
						flat_divide(dx_acc_stage, dx_epi_tile);
					auto dx_tTR_tAcc =
						dx_thr_t2r.partition_S(dx_tAccStage);

					for (int chunk = 0; chunk < 8; ++chunk) {
						int panel = chunk / 4;
						int n_tile = coord.n_tile * 2 + panel;
						bool valid_panel = n_tile < dx_n_tiles;
						auto tAccChunk =
							dx_tTR_tAcc(_, _, _, _0{}, chunk);
						bool issue_tmem_load = true;
						if constexpr (kPredicateDxTmemLoad) {
							int subpart = (tAccChunk.data().dp_ / 32) % 4;
							issue_tmem_load =
								tid_in_warpgroup / kWarpSize == subpart;
						}
						if (issue_tmem_load && valid_panel) {
							copy(dx_t2r, tAccChunk, dx_tTR_rAcc);
							cutlass::arch::fence_view_async_tmem_load();
						}
						if (chunk > 0 && tid_in_warpgroup == 0) {
							cute::tma_store_wait<0>();
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize, warpgroup_barrier);
						if (issue_tmem_load && valid_panel) {
							CUTE_UNROLL
							for (int i = 0; i < size(dx_tTR_rAcc); ++i) {
								sDxStore(
									get<0>(dx_tTR_cSlab(i)),
									get<1>(dx_tTR_cSlab(i))) =
									dx_tTR_rAcc(i);
							}
						}
						cutlass::arch::NamedBarrier::sync(
							Config::kWarpgroupSize, warpgroup_barrier);

						if (tid_in_warpgroup == 0 && valid_panel) {
							cute::tma_store_fence();
							int panel_index = local_index + panel;
							int stage = panel_index % kDxRingStages;
							int slot_tile = static_cast<int>(
								dx_slot_offset<CommConfig>(
									cta, 0, stage) /
								CommConfig::kTileElements);
							CUTE_UNROLL
							for (int row_group = 0; row_group < 8;
									++row_group) {
								auto sTile = make_tensor(
									make_smem_ptr(
										smem.dx_store_data() +
										row_group * 16 * 64),
									typename DxTraits::
										SmemLayoutStoreTile{});
								auto tSsTile =
									cta_dx_store.partition_S(sTile);
								copy(
									tma.dx_store,
									tSsTile,
									tSgS_dx(
										_,
										_,
										_,
										slot_tile * 8 + row_group,
										chunk % 4));
								cute::tma_store_arrive();
							}
						}
					}
					if (tid_in_warpgroup == 0) {
						cute::tma_store_wait<0>();
					}
					cutlass::arch::NamedBarrier::sync(
						Config::kDxEpilogueThreads,
						Config::kEpilogueBarrierId);
					if (tid_in_epi == 0) {
						dx_acc_pipe.consumer_release(dx_acc_cons_state);
						if constexpr (EnableLocalReduce) {
							liger_cute::detail::
								publish_local_reduce_source();
							for (int panel = 0; panel < 2; ++panel) {
								int n_tile =
									coord.n_tile * 2 + panel;
								if (n_tile >= dx_n_tiles) continue;
								int stage =
									(local_index + panel) %
									kDxRingStages;
								cute::arrive_barrier(
									smem.dx_ready[stage]);
							}
						}
					}
					++dx_acc_cons_state;
					local_index +=
						coord.n_tile * 2 + 1 < dx_n_tiles ? 2 : 1;
				}
				dx_local_index = local_index;
				if (tid_in_epi == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxEpiEnd);
				}
			}
		}

		if constexpr (
			((kRunDx && kRunDw) || kAuditForceDxDwBarrier) &&
			!kAuditSkipDxDwBarrier) {
			[[maybe_unused]] std::uint64_t dx_dw_wait_begin = 0;
			if constexpr (kBackwardDiagnosticTimestampsSm100) {
				if (lane == 0) {
					dx_dw_wait_begin = backward_globaltimer_sm100();
				}
			}
			if constexpr (
				kBackwardSyncVariantSm100 == 1 ||
				kBackwardSyncVariantSm100 == 2) {
				if (is_epilogue && tid_in_epi == 0) {
					cute::arrive_barrier(smem.dx_phase_free);
				}
				if (
					warp_id == Config::kTmaWarp ||
					warp_id == Config::kUmmaWarp) {
					if (lane == 0) {
						cute::wait_barrier(
							smem.dx_phase_free, wave & 1);
					}
					__syncwarp();
				}
			} else {
				backward_compute_barrier_sm100<Compute>();
			}
			if constexpr (kBackwardDiagnosticTimestampsSm100) {
				if (lane == 0) {
					backward_diagnostic_duration_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDxDwWaitMax,
						dx_dw_wait_begin);
				}
			}
		}

		if constexpr (kRunDw) {  // compiled out by the isolated dZ / dX benchmarks
			// ── phase dW ──────────────────────────────────────────────────────
			if (warp_id == Config::kTmaWarp) {
				auto mDzt = tma.dzt.get_tma_tensor(make_shape(
					static_cast<int64_t>(padded_vocab),
					static_cast<int64_t>(
						Config::kWaveRows *
						Config::kDzWorkspaceSlots)));
				auto mXt = tma.xt.get_tma_tensor(make_shape(
					static_cast<int64_t>(params.hidden),
					static_cast<int64_t>(params.tokens)));
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwTmaStart);
				}
				backward_produce_dw_pairs_sm100<
					Compute,
					kDwKTiles,
					kBackwardSyncVariantSm100 == 2 && kRunDz>(
					dw_pipe,
					dw_state,
					smem.dw_a_data(),
					smem.dw_b_data(),
					tma.dzt,
					tma.xt,
					mDzt,
					mXt,
					dw_local,
					dw_first,
					1,
					dw_m_pairs,
					dw_n_tiles,
					dz_slot * kDwKTiles,
					wave * kDwKTiles,
					wave_workspace.dz_tile_ready,
					wave,
					dz_items,
					dz_n_tiles);
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwTmaEnd);
				}
			} else if (is_mma_warp && is_leader_cta) {
				if (lane == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwMmaStart);
				}
				backward_mma_dw_pairs_sm100<Compute, kDwKTiles>(
					dw_pipe,
					dw_state,
					dw_acc_pipe,
					dw_acc_prod_state,
					smem.dw_a_data(),
					smem.dw_b_data(),
					tmem_base,
					dw_local,
					dw_first,
					1,
					dw_m_pairs,
					dw_n_tiles,
					wave_workspace.diagnostics,
					kBackwardDiagnosticDwMmaFirstReady);
				if (lane == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwMmaEnd);
				}
			} else if (is_dw_epilogue) {
				if (tid_in_epi == 0) {
					backward_diagnostic_first_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwEpiStart);
				}
				typename DwTraits::TiledMma epilogue_mma;
				auto cta_mma_epi =
					epilogue_mma.get_slice(cluster_rank);
				auto cAccFull = make_identity_tensor(
					make_shape(
						Int<Config::kTileM>{},
						Int<Config::kTileN>{}));
				auto tCtAcc =
					cta_mma_epi.make_fragment_C(
						cta_mma_epi.partition_C(cAccFull));
				tCtAcc.data() = tmem_base;
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
					size(tmem_warp_layout) !=
					cosize(tmem_warp_layout);
				auto sDwStore0 =
					cute::as_position_independent_swizzle_tensor(
						make_tensor(
							make_smem_ptr(smem.dw_store_data(0)),
							typename Traits::SmemLayoutStoreSlot{}));
				auto sDwStore1 =
					cute::as_position_independent_swizzle_tensor(
						make_tensor(
							make_smem_ptr(smem.dw_store_data(1)),
							typename Traits::SmemLayoutStoreSlot{}));
				auto tTR_sDw0 = thr_t2r.partition_D(sDwStore0);
				auto tTR_sDw1 = thr_t2r.partition_D(sDwStore1);
				auto tTR_rDw =
					make_tensor<Element>(shape(tTR_sDw0));
				constexpr int kFragmentSize = 32;
				auto tTR_rAccFragments =
					recast<cutlass::Array<float, kFragmentSize>>(
						coalesce(tTR_rAcc));
				auto tTR_rDwFragments =
					recast<cutlass::Array<Element, kFragmentSize>>(
						coalesce(tTR_rDw));
				auto shape_dw = make_shape(
					static_cast<int64_t>(params.local_vocab),
					static_cast<int64_t>(params.hidden));
				auto mDwStore =
					tma.dw_store.get_tma_tensor(shape_dw);
				auto mDwAdd =
					tma.dw_add.get_tma_tensor(shape_dw);
				auto cta_dw_store =
					tma.dw_store.get_slice(Int<0>{});
				auto cta_dw_add =
					tma.dw_add.get_slice(Int<0>{});
				auto tSsS_dw0 =
					cta_dw_store.partition_S(sDwStore0);
				auto tSsS_dw1 =
					cta_dw_store.partition_S(sDwStore1);
				auto tSgS_dw =
					cta_dw_store.partition_D(local_tile(
						mDwStore,
						make_tile(
							Int<Config::kCtaTileM>{},
							Int<Config::kEpilogueChunkN>{}),
						make_coord(_, _)));
				auto tAsS_dw0 =
					cta_dw_add.partition_S(sDwStore0);
				auto tAsS_dw1 =
					cta_dw_add.partition_S(sDwStore1);
				auto tAgA_dw =
					cta_dw_add.partition_D(local_tile(
						mDwAdd,
						make_tile(
							Int<Config::kCtaTileM>{},
							Int<Config::kEpilogueChunkN>{}),
						make_coord(_, _)));
				for (int index = 0; index < dw_local; ++index) {
					int item = dw_first + index;
					BackwardDwPairCoordSm100 coord =
						backward_dw_pair_coord_sm100(item, dw_m_pairs);
					int m_tile = coord.m_pair * Config::kClusterM + cluster_rank;
					CUTE_UNROLL
					for (int n_in_pair = 0; n_in_pair < 2; ++n_in_pair) {
						int n_tile = coord.n_tile_begin + n_in_pair;
						if (n_tile >= dw_n_tiles) continue;

						dw_acc_pipe.consumer_wait(dw_acc_cons_state);
						if (
							index == 0 && n_in_pair == 0 &&
							tid_in_epi == 0) {
							backward_diagnostic_first_sm100(
								wave_workspace.diagnostics,
								kBackwardDiagnosticDwEpiFirstReady);
						}
						tCtAcc.data() = tmem_base +
							static_cast<uint32_t>(
								dw_acc_cons_state.index() *
									Config::kTmemStageColumns);
						auto acc_mn_stage =
							tCtAcc(make_coord(_, _), _0{}, _0{});
						auto tAccEpiStage =
							flat_divide(acc_mn_stage, epi_tile);
						auto tTR_tAcc =
							thr_t2r.partition_S(tAccEpiStage);

						CUTE_UNROLL
						for (int round = 0;
								round <
									Config::kTileN /
										Config::kEpilogueChunkN;
								++round) {
							int chunk = round;
							int buffer = round & 1;
							auto tAccChunk =
								tTR_tAcc(_, _, _, _0{}, chunk);
							bool issue_tmem_load = true;
							if constexpr (kPredicateTmemLoad) {
								int subpart =
									(tAccChunk.data().dp_ / 32) % 4;
								issue_tmem_load =
									tid_in_warpgroup / kWarpSize ==
									subpart;
							}
							if (issue_tmem_load) {
								copy(t2r, tAccChunk, tTR_rAcc);
							}
							if (
								round + 1 ==
								Config::kTileN /
									Config::kEpilogueChunkN) {
								cutlass::arch::fence_view_async_tmem_load();
							}
							if (tid_in_warpgroup == 0) {
								cute::tma_store_wait<1>();
							}
							cutlass::arch::NamedBarrier::sync(
								Config::kWarpgroupSize,
								warpgroup_barrier);
							if (
								round + 1 ==
									Config::kTileN /
										Config::kEpilogueChunkN &&
								tid_in_epi == 0) {
								dw_acc_pipe.consumer_release(
									dw_acc_cons_state);
							}
							if (issue_tmem_load) {
								CUTE_UNROLL
								for (int i = 0;
										i < size(tTR_rAccFragments);
										++i) {
									tTR_rDwFragments(i) =
										cutlass::NumericArrayConverter<
											Element,
											float,
											kFragmentSize>{}(
											tTR_rAccFragments(i));
								}
								if (buffer == 0) {
									copy(
										AutoVectorizingCopyWithAssumedAlignment<
											128>{},
										tTR_rDw,
										tTR_sDw0);
								} else {
									copy(
										AutoVectorizingCopyWithAssumedAlignment<
											128>{},
										tTR_rDw,
										tTR_sDw1);
								}
							}
							cutlass::arch::NamedBarrier::sync(
								Config::kWarpgroupSize,
								warpgroup_barrier);
							if (tid_in_warpgroup == 0) {
								cutlass::arch::
									fence_view_async_shared();
								int n_chunk =
									n_tile *
										(Config::kTileN /
											Config::kEpilogueChunkN) +
									chunk;
								if (wave == 0) {
									copy(
										tma.dw_store,
										buffer == 0
											? tSsS_dw0
											: tSsS_dw1,
										tSgS_dw(
											_, _, _, m_tile, n_chunk));
								} else {
									copy(
										tma.dw_add,
										buffer == 0
											? tAsS_dw0
											: tAsS_dw1,
										tAgA_dw(
											_, _, _, m_tile, n_chunk));
								}
								cute::tma_store_arrive();
							}
						}
						++dw_acc_cons_state;
					}
				}
				if (tid_in_warpgroup == 0) {
					cute::tma_store_wait<0>();
				}
				cutlass::arch::NamedBarrier::sync(
					Config::kDwEpilogueThreads,
					Config::kEpilogueBarrierId);
				if (tid_in_epi == 0) {
					backward_diagnostic_max_sm100(
						wave_workspace.diagnostics,
						kBackwardDiagnosticDwEpiEnd);
				}
			}
		}

		// dZ workspace reuse only. dX(wave) is deliberately NOT finalized
		// here: warp 0's stage F and warp 1's inter-host ring are still in
		// flight and are allowed to run straight through the next chunk's dZ
		// GEMM. The final wave needs no barrier: nothing overwrites the
		// workspace afterwards.
		if constexpr (kNeedGridBarrier) {
			bool needs_workspace_reuse =
				wave + 1 < num_waves &&
				(kBackwardSyncVariantSm100 != 3 ||
					wave + 1 >= Config::kDzWorkspaceSlots);
			if (needs_workspace_reuse) {
				++barrier_generation;
				backward_grid_barrier_sm100<Compute>(
					wave_workspace.grid_barrier,
					barrier_generation *
						static_cast<unsigned long long>(grid_ctas),
					warp_id);
			} else if constexpr (kBackwardSyncVariantSm100 == 3) {
				if (wave + 1 < num_waves) {
					backward_wave_pair_barrier_sm100<Compute>(
						smem.wave_pair_barrier, warp_id);
				}
			}
		}
	}

	if (warp_id == Config::kTmaWarp) {
		if constexpr (kRunDz) pipe.producer_tail(state);
		if constexpr (kRunDx) dx_pipe.producer_tail(dx_state);
		if constexpr (kRunDw) dw_pipe.producer_tail(dw_state);
		if (lane == 0) {
			backward_diagnostic_max_sm100(
				wave_workspace.diagnostics,
				kBackwardDiagnosticKernelEnd);
		}
		return;
	}
	// Compute-only TMEM teardown over warps 3..11; warps 0, 1 and 2 never
	// reach this barrier.
	cutlass::arch::NamedBarrier::sync(
		Config::kMmaEpilogueThreads, Config::kMmaEpilogueBarrierId);
	if (warp_id == Config::kFirstEpilogueWarp) {
		tmem_allocator.release_allocation_lock();
		backward_tmem_pair_barrier_sm100<Compute>(
			smem.tmem_pair_barrier, warp_id);
		tmem_allocator.free(smem.tmem_base, Config::kTmemColumns);
	}
	if (lane == 0) {
		backward_diagnostic_max_sm100(
			wave_workspace.diagnostics,
			kBackwardDiagnosticKernelEnd);
	}
#else
	__trap();
#endif
}

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
