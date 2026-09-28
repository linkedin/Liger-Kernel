#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Reusable SM90 (Hopper) warp-specialized GEMM mechanics.
//
// Architecture-specific building blocks shared by the fused scaled linear
// cross entropy forward path and the (upcoming) backward dZ/dX/dW paths, so
// neither has to copy/paste TMA descriptor construction, SMEM swizzle atom
// selection, WGMMA atom selection, mbarrier pipeline wiring, warp-group
// register reconfiguration or the cluster launch dance.
//
// Design rules:
//   * Zero-overhead: every entity is a template or a CUTE_DEVICE
//     (__device__ __forceinline__) function, and the mainloop drivers take
//     the caller's issue/fence steps as inlined functors. The generated SASS
//     is bit-identical to the hand-written loops these replaced — see the
//     cuobjdump diff in the module's validation notes.
//   * Nothing here decides *ordering policy*. MainloopKLoopSm90 encodes the
//     one-stage release lag the CuTe-DSL kernels rely on and nothing else;
//     what to issue per stage stays with the caller.
//   * `int Compute` on every GEMM-related template, defaulted to 90, with a
//     static_assert so a future SM100 port is a new header, not a silent
//     mis-instantiation of this one.
// ═══════════════════════════════════════════════════════════════════════════

// The extended SM90 MMA shapes (e.g. the N160 WGMMA atom) are opt-in in CuTe
// and must be enabled before the first CuTe include of the translation unit.
// Including this header first is what guarantees that.
#if !defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
#endif

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace liger {
namespace sm90 {

using namespace cute;

// ───────────────────────────────────────────────────────────────────────────
// Architecture constants
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ArchSm90 {
	static_assert(Compute == 90, "ArchSm90 requires Compute=90");

	static constexpr int kCompute = Compute;
	static constexpr int kWarpSize = 32;
	static constexpr int kWarpsPerWarpGroup = 4;
	static constexpr int kWarpGroupSize = kWarpSize * kWarpsPerWarpGroup;
	// Hopper's opt-in dynamic shared-memory ceiling.
	static constexpr int kMaxDynamicSmemBytes = 227 * 1024;
	// A CTA may own at most 64K registers; ptxas therefore caps a 384-thread
	// warp-specialized CTA at 168 registers per thread.
	static constexpr int kRegisterFilePerCta = 65536;
};

// ───────────────────────────────────────────────────────────────────────────
// WGMMA atom / TiledMMA selection
// ───────────────────────────────────────────────────────────────────────────

// The M64 SS WGMMA atom for the requested N. Wraps CuTe's selector so callers
// do not have to know which shapes live in the extended set.
template <
	class ElementA,
	class ElementB,
	class ElementAccum,
	int AtomTileN,
	GMMA::Major MajorA = GMMA::Major::K,
	GMMA::Major MajorB = GMMA::Major::K,
	int Compute = 90>
struct GmmaAtomSm90 {
	static_assert(Compute == 90, "GmmaAtomSm90 requires Compute=90");

	static constexpr int kCompute = Compute;
	static constexpr int kAtomTileM = 64;
	static constexpr int kAtomTileN = AtomTileN;
	static constexpr int kAtomTileK = 16;

	using Atom = decltype(cute::GMMA::ss_op_selector<
		ElementA,
		ElementB,
		ElementAccum,
		Shape<Int<kAtomTileM>, Int<AtomTileN>, Int<kAtomTileK>>,
		MajorA,
		MajorB>());
};

// Cooperative TiledMMA: NumWarpGroups warp groups split the tile along M, so
// each warp group owns an (AtomTileM, AtomTileN) accumulator.
template <
	class ElementA,
	class ElementB,
	class ElementAccum,
	int AtomTileN,
	int NumWarpGroups,
	GMMA::Major MajorA = GMMA::Major::K,
	GMMA::Major MajorB = GMMA::Major::K,
	int Compute = 90>
struct TiledMmaMSplitSm90 {
	static_assert(Compute == 90, "TiledMmaMSplitSm90 requires Compute=90");

	static constexpr int kCompute = Compute;
	static constexpr int kNumWarpGroups = NumWarpGroups;

	using AtomTraits = GmmaAtomSm90<
		ElementA, ElementB, ElementAccum, AtomTileN, MajorA, MajorB, Compute>;
	using Atom = typename AtomTraits::Atom;
	using Type = TiledMMA<
		MMA_Atom<Atom>, Layout<Shape<Int<NumWarpGroups>, _1, _1>>>;

	static constexpr int kTileM = AtomTraits::kAtomTileM * NumWarpGroups;
	static constexpr int kTileN = AtomTileN;
};

// ───────────────────────────────────────────────────────────────────────────
// SMEM layouts
// ───────────────────────────────────────────────────────────────────────────

// K-major operand staging with the 128 B swizzle atom, which is what a K tile
// of exactly 128 bytes requires.
template <class Element, int Rows, int Cols, int Stages, int Compute = 90>
struct SmemLayoutKMajorSm90 {
	static_assert(Compute == 90, "SmemLayoutKMajorSm90 requires Compute=90");
	static_assert(Cols * static_cast<int>(sizeof(Element)) % 128 == 0,
		"the SW128 K-major atom needs 128 B rows");

	static constexpr int kCompute = Compute;

	using Atom = GMMA::Layout_K_SW128_Atom<Element>;
	using Single = decltype(tile_to_shape(
		Atom{}, Shape<Int<Rows>, Int<Cols>>{}));
	using Staged = decltype(tile_to_shape(
		Atom{}, Shape<Int<Rows>, Int<Cols>, Int<Stages>>{}));

	static constexpr int kStageBytes = static_cast<int>(
		cute::size(Single{}) * sizeof(Element));
	static constexpr int kStagedElements = cosize_v<Staged>;
};

// ───────────────────────────────────────────────────────────────────────────
// TMA descriptors
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct TmaSm90 {
	static_assert(Compute == 90, "TmaSm90 requires Compute=90");

	static constexpr int kCompute = Compute;

	// int64 extents: rows * cols exceeds INT_MAX at production shapes and
	// would otherwise wrap CuTe's layout algebra into a negative byte offset.
	template <class Element>
	static auto row_major_tensor(const Element* ptr, int rows, int cols) {
		return make_tensor(
			make_gmem_ptr(ptr),
			make_layout(
				make_shape(
					static_cast<int64_t>(rows), static_cast<int64_t>(cols)),
				make_stride(static_cast<int64_t>(cols), _1{})));
	}

	template <class SmemLayout, int TileRows, int TileCols, class Element>
	static auto make_load(const Element* ptr, int rows, int cols) {
		return make_tma_copy(
			SM90_TMA_LOAD{},
			row_major_tensor(ptr, rows, cols),
			SmemLayout{},
			make_shape(Int<TileRows>{}, Int<TileCols>{}),
			_1{});
	}

	// Multicast splits the box across `Multicast` CTAs of the cluster; each
	// peer issues its own slice and every peer receives the whole box, so the
	// stage's transaction byte count stays the full tile.
	template <
		class SmemLayout, int TileRows, int TileCols, int Multicast,
		class Element>
	static auto make_load_multicast(const Element* ptr, int rows, int cols) {
		return make_tma_copy(
			SM90_TMA_LOAD_MULTICAST{},
			row_major_tensor(ptr, rows, cols),
			SmemLayout{},
			make_shape(Int<TileRows>{}, Int<TileCols>{}),
			Int<Multicast>{});
	}
};

// Multicast mask over the cluster's M mode: every CTA sharing this N tile.
template <class ClusterShape, int Compute = 90>
CUTE_DEVICE std::uint16_t cluster_multicast_mask_m_sm90() {
	static_assert(Compute == 90,
		"cluster_multicast_mask_m_sm90 requires Compute=90");
	std::uint16_t mask = 0;
	auto block_layout = Layout<ClusterShape>{};
	CUTE_UNROLL
	for (int m = 0; m < size<0>(block_layout); ++m) {
		mask |= std::uint16_t(1) << block_layout(m, Int<0>{}, Int<0>{});
	}
	return mask;
}

// ───────────────────────────────────────────────────────────────────────────
// Mainloop pipeline
// ───────────────────────────────────────────────────────────────────────────

enum class PipelineRoleSm90 : std::uint8_t {
	kProducer,
	kConsumer,
	kNonParticipant,
};

template <int Stages, int Compute = 90>
struct MainloopPipelineSm90 {
	static_assert(Compute == 90, "MainloopPipelineSm90 requires Compute=90");

	static constexpr int kCompute = Compute;
	static constexpr int kStages = Stages;

	using Type = cutlass::PipelineTmaAsync<Stages>;
	using State = cutlass::PipelineState<Stages>;
	using SharedStorage = typename Type::SharedStorage;
	using Category = typename Type::ThreadCategory;

	// Role-specific construction. Barrier initialisation is split out of the
	// constructors so a warp-specialized kernel can (a) reconfigure registers
	// before any pipeline object exists and (b) keep the producer's and the
	// consumer's pipeline objects in disjoint scopes — neither is ever live on
	// the other's path, and idle warps never materialise one at all.

	// All threads call this; only `initializing_warp` touches the mbarriers,
	// but every thread must issue the init fence. Must be followed by a
	// cluster-wide handshake before any barrier is used.
	template <class ClusterShape>
	CUTE_DEVICE static void init_barriers(
			SharedStorage& storage,
			int transaction_bytes,
			int num_consumers,
			ClusterShape cluster_shape) {
		typename Type::Params params;
		params.transaction_bytes = transaction_bytes;
		params.num_producers = 1;
		params.num_consumers = num_consumers;
		Type::init_barriers(storage, params, cluster_shape);
	}

	template <class ClusterShape>
	CUTE_DEVICE static Type make_producer(
			SharedStorage& storage,
			int transaction_bytes,
			int num_consumers,
			bool is_leader,
			ClusterShape cluster_shape) {
		typename Type::Params params;
		params.transaction_bytes = transaction_bytes;
		params.num_producers = 1;
		params.num_consumers = num_consumers;
		params.is_leader = is_leader;
		params.role = Category::Producer;
		// Barriers are already live; only the per-thread masks are built here.
		return Type(storage, params, cluster_shape,
			cute::false_type{}, cute::true_type{});
	}

	template <class ClusterShape>
	CUTE_DEVICE static Type make_consumer(
			SharedStorage& storage,
			int transaction_bytes,
			int num_consumers,
			ClusterShape cluster_shape) {
		typename Type::Params params;
		params.transaction_bytes = transaction_bytes;
		params.num_producers = 1;
		params.num_consumers = num_consumers;
		params.is_leader = 0;
		params.role = Category::Consumer;
		return Type(storage, params, cluster_shape,
			cute::false_type{}, cute::true_type{});
	}

	CUTE_DEVICE static State producer_start_state() {
		return cutlass::make_producer_start_state<Type>();
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Warp-group register reconfiguration and cluster synchronisation
// ───────────────────────────────────────────────────────────────────────────

template <int ProducerRegisters, int ConsumerRegisters, int Compute = 90>
struct WarpSpecializedRegistersSm90 {
	static_assert(Compute == 90,
		"WarpSpecializedRegistersSm90 requires Compute=90");
	static_assert(ProducerRegisters % 8 == 0 && ConsumerRegisters % 8 == 0,
		"setmaxnreg operands must be multiples of 8");

	static constexpr int kCompute = Compute;
	static constexpr int kProducerRegisters = ProducerRegisters;
	static constexpr int kConsumerRegisters = ConsumerRegisters;

	// Must be executed by every warp of the warp group, including idle ones.
	CUTE_DEVICE static void producer() {
		cutlass::arch::warpgroup_reg_dealloc<ProducerRegisters>();
	}

	CUTE_DEVICE static void consumer() {
		cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();
	}
};

// pipeline_init_arrive / pipeline_init_wait: the cluster-wide handshake that
// must complete before any mbarrier in the cluster is touched.
template <int Compute = 90>
CUTE_DEVICE void cluster_pipeline_init_sm90() {
	static_assert(Compute == 90,
		"cluster_pipeline_init_sm90 requires Compute=90");
	cute::cluster_arrive_relaxed();
	cute::cluster_wait();
}

// No CTA may retire while a cluster peer can still address its SMEM.
template <int Compute = 90>
CUTE_DEVICE void cluster_exit_sm90() {
	static_assert(Compute == 90, "cluster_exit_sm90 requires Compute=90");
	cute::cluster_sync();
}

template <int BarrierId, int NumThreads, int Compute = 90>
CUTE_DEVICE void named_barrier_sync_sm90() {
	static_assert(Compute == 90, "named_barrier_sync_sm90 requires Compute=90");
	static_assert(BarrierId >= 0 && BarrierId < 16,
		"hardware named barrier ids are 0..15");
	static_assert(NumThreads % 32 == 0,
		"named barrier thread counts must be warp aligned");
	asm volatile("barrier.sync %0, %1;"
		:
		: "n"(BarrierId), "n"(NumThreads));
}

// ───────────────────────────────────────────────────────────────────────────
// Mainloop drivers
//
// These encode the CuTe-DSL pipeline *ordering* and nothing else. The caller
// supplies what to issue per stage; both functors are inlined, so the emitted
// instruction stream is identical to writing the loop out by hand.
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct MainloopKLoopSm90 {
	static_assert(Compute == 90, "MainloopKLoopSm90 requires Compute=90");

	static constexpr int kCompute = Compute;

	// Producer: acquire the stage, issue every TMA copy for it, advance.
	// `issue(barrier, stage, k_tile)` performs the copies.
	template <class Pipeline, class State, class IssueCopies>
	CUTE_DEVICE static void produce(
			Pipeline& pipe,
			State& state,
			int num_k_tiles,
			IssueCopies issue) {
		for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
			pipe.producer_acquire(state);
			issue(pipe.producer_get_barrier(state), state.index(), k_tile);
			++state;
		}
	}

	// Consumer with the deliberate one-stage release lag.
	//
	// Stage zero is issued without a release because no earlier WGMMA group
	// has retired. Every later stage is committed, `wait_group(1)` retires the
	// previous group, and only then is that previous AB stage handed back to
	// the producer. The tail uses `wait_group(0)` before the final release.
	// Without the lag, TMA would overwrite operands still feeding a live
	// WGMMA group.
	//
	// `issue_mma(stage)` fences the accumulators, arrives, issues the GEMMs
	// and commits the batch. `fence_accumulators()` re-fences them after a
	// wait_group so the accumulator registers are readable again.
	template <class Pipeline, class State, class IssueMma, class FenceAccumulators>
	CUTE_DEVICE static void consume(
			Pipeline& pipe,
			State& read_state,
			State& release_state,
			int num_k_tiles,
			IssueMma issue_mma,
			FenceAccumulators fence_accumulators) {
		pipe.consumer_wait(read_state);
		issue_mma(read_state.index());
		++read_state;

		for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
			pipe.consumer_wait(read_state);
			issue_mma(read_state.index());

			warpgroup_wait<1>();
			fence_accumulators();
			pipe.consumer_release(release_state);
			++release_state;
			++read_state;
		}

		warpgroup_wait<0>();
		fence_accumulators();
		pipe.consumer_release(release_state);
		++release_state;
	}
};

// ───────────────────────────────────────────────────────────────────────────
// Host-side cluster launch
// ───────────────────────────────────────────────────────────────────────────

template <int Compute = 90>
struct ClusterLaunchSm90 {
	static_assert(Compute == 90, "ClusterLaunchSm90 requires Compute=90");

	static constexpr int kCompute = Compute;

	// Opt into the Hopper dynamic-SMEM ceiling and non-portable cluster sizes.
	template <class Kernel>
	static cudaError_t prepare(Kernel kernel, int smem_bytes) {
		cudaError_t error = cudaFuncSetAttribute(
			kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
		if (error != cudaSuccess) return error;
		return cudaFuncSetAttribute(
			kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
	}

	static cudaLaunchConfig_t make_config(
			dim3 grid,
			int threads,
			int smem_bytes,
			int cluster_m,
			cudaLaunchAttribute& attribute,
			cudaStream_t stream) {
		attribute = {};
		attribute.id = cudaLaunchAttributeClusterDimension;
		attribute.val.clusterDim.x = static_cast<unsigned>(cluster_m);
		attribute.val.clusterDim.y = 1;
		attribute.val.clusterDim.z = 1;

		cudaLaunchConfig_t config = {};
		config.gridDim = grid;
		config.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
		config.dynamicSmemBytes = smem_bytes;
		config.stream = stream;
		config.attrs = &attribute;
		config.numAttrs = 1;
		return config;
	}

	// Concurrent clusters this device can hold, i.e. the occupancy target a
	// persistent/split scheduler should aim at. Returns 0 if unavailable.
	template <class Kernel>
	static int max_active_clusters(
			Kernel kernel, int threads, int smem_bytes, int cluster_m) {
		cudaLaunchAttribute attribute = {};
		cudaLaunchConfig_t config = make_config(
			dim3(static_cast<unsigned>(cluster_m), 1, 1),
			threads,
			smem_bytes,
			cluster_m,
			attribute,
			nullptr);
		int clusters = 0;
		if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) !=
			cudaSuccess) {
			cudaGetLastError();
			return 0;
		}
		return clusters;
	}

	template <class Kernel, class... Args>
	static cudaError_t launch(
			Kernel kernel,
			dim3 grid,
			int threads,
			int smem_bytes,
			int cluster_m,
			cudaStream_t stream,
			Args const&... args) {
		cudaLaunchAttribute attribute = {};
		cudaLaunchConfig_t config = make_config(
			grid, threads, smem_bytes, cluster_m, attribute, stream);
		return cudaLaunchKernelEx(&config, kernel, args...);
	}
};

}  // namespace sm90
}  // namespace liger
