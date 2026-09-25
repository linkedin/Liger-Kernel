// ═══════════════════════════════════════════════════════════════════════════
// SM90 fused scaled linear cross entropy backward host dispatch. dZ always
// selects the zero-spill cluster-2 handoff. NVLS and hierarchical paths use
// the four-stage combined dX+dW cluster kernel; direct-peer uses its fallback.
// ═══════════════════════════════════════════════════════════════════════════

// Must precede every other CUTLASS/CuTe include: it opts the TU into CuTe's
// extended SM90 MMA shapes.
#include "backward_gemm_mainloop_sm90.cuh"

#include "backward_gemm_sm90.cuh"
#include "backward_reduce_sm90.cuh"
#include "gemm_sm90.cuh"
#include "liger_cute/check.h"
#include "liger_cute/detail/tp_reduce.cuh"
#include "workspace.cuh"

#include <cstddef>
#include <cstdint>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace {

using namespace cute;

using HostConfig = BackwardGemmConfigSm90<90>;
using HostLaunch = BackwardGemmLaunchSm90<90>;
using HostTraits = BackwardGemmTraitsSm90<90>;
using HostCombinedDwTraits = BackwardGemmTraitsSm90<90, 4, 64, 2>;
using HostDxClusterTraits = BackwardDxCluster2TraitsSm90<90>;
using HostDzHandoffTraits = BackwardDzHandoffTraitsSm90<90>;
using HostElement = HostTraits::Element;

// The staging ring depth comes from dx_reduce.cuh so the kernel template,
// symmetric capacity and CTA-local mbarrier arrays cannot drift apart.

void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy backward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

template <class Layout, class Element>
auto make_row_major_load(const Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_LOAD{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, int TileRows, int TileCols, class Element>
auto make_row_major_load_tile(
		const Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_LOAD{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{},
		make_shape(Int<TileRows>{}, Int<TileCols>{}),
		_1{});
}

template <class Layout, class Element>
auto make_row_major_store(Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_STORE{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, class Element>
auto make_row_major_reduce_add(Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_REDUCE_ADD{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{});
}

template <class Layout, int TileRows, int TileCols, class Element>
auto make_row_major_store_tile(Element* ptr, int64_t rows, int64_t cols) {
	return make_tma_copy(
		SM90_TMA_STORE{},
		backward_row_major_tensor(ptr, rows, cols),
		Layout{},
		make_shape(Int<TileRows>{}, Int<TileCols>{}),
		_1{});
}

template <class Layout, class Element>
auto make_col_major_load(
		const Element* ptr, int64_t rows, int64_t cols, int64_t leading) {
	return make_tma_copy(
		SM90_TMA_LOAD{},
		backward_col_major_tensor(ptr, rows, cols, leading),
		Layout{});
}

template <
	class Layout,
	int TileRows,
	int TileCols,
	int Multicast,
	class Element>
auto make_col_major_load_multicast(
		const Element* ptr, int64_t rows, int64_t cols, int64_t leading) {
	return make_tma_copy(
		SM90_TMA_LOAD_MULTICAST{},
		backward_col_major_tensor(ptr, rows, cols, leading),
		Layout{},
		make_shape(Int<TileRows>{}, Int<TileCols>{}),
		Int<Multicast>{});
}

void validate(const BackwardTpParamsSm90<90>& params) {
	const auto& gemm = params.gemm;
	LIGER_CHECK(gemm.tokens >= 0, "tokens must be non-negative");
	LIGER_CHECK(gemm.hidden > 0, "hidden must be positive");
	LIGER_CHECK(gemm.local_vocab > 0, "local_vocab must be positive");
	LIGER_CHECK(
		gemm.x != nullptr && gemm.weight != nullptr &&
			gemm.target != nullptr && gemm.grad_output != nullptr &&
			gemm.lse != nullptr,
		"backward inputs x / weight / target / grad_output / lse must be "
		"non-null");
	LIGER_CHECK(
		gemm.grad_input != nullptr && gemm.grad_weight != nullptr,
		"backward outputs grad_input and grad_weight must be non-null");
	LIGER_CHECK(
		gemm.dz_workspace != nullptr,
		"the dZ wave workspace must be reserved");
	LIGER_CHECK(
		params.tiles_per_reduce == 1 || params.tiles_per_reduce == 2 ||
			params.tiles_per_reduce == 4,
		"TilesPerReduce must be 1, 2 or 4, got ",
		params.tiles_per_reduce);
	LIGER_CHECK(
		params.num_comm_channels >= 1,
		"num_comm_channels must be positive (legacy compatibility argument)");

	// TMA needs 16 B aligned base addresses and row strides for its BF16
	// operands. dX uses generic 8 B vector stores.
	auto aligned = [](const void* pointer) {
		return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
	};
	LIGER_CHECK(
		aligned(gemm.x) && aligned(gemm.weight) &&
			aligned(gemm.grad_weight) && aligned(gemm.dz_workspace),
		"backward TMA BF16 operands must be 16 B aligned");
	LIGER_CHECK(
		reinterpret_cast<std::uintptr_t>(gemm.grad_input) % 8 == 0,
		"grad_input must be 8 B aligned for vectorized BF16 stores");
	LIGER_CHECK(
		(static_cast<std::size_t>(gemm.hidden) * sizeof(HostElement)) % 16 == 0,
		"hidden * sizeof(bfloat16) must be a multiple of 16 B for TMA");
	validate_backward_tp_shape(
		gemm.tokens, gemm.hidden, gemm.local_vocab);

	std::size_t needed = HostLaunch::dz_workspace_bytes(gemm.local_vocab);
	LIGER_CHECK(
		gemm.dz_workspace_bytes >= needed,
		"the dZ wave workspace needs ",
		needed,
		" B, got ",
		gemm.dz_workspace_bytes);
}

template <
	int TilesPerReduce,
	bool ReturnEntropy,
	liger_cute::detail::LocalReduceBackend Backend,
	bool RequiresRemote>
void launch_instance(
		const BackwardTpParamsSm90<90>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		cudaStream_t stream) {
	static constexpr bool kUseNvls =
		Backend == liger_cute::detail::LocalReduceBackend::kNvls;
	static_assert(
		!RequiresRemote || kUseNvls,
		"the remote follow-up requires a local NVLS reduce-scatter");
	using CommConfig =
		DxCommConfig<HostConfig, kDxRingStages, TilesPerReduce, 90>;
	using DzSmem = BackwardDzHandoffSmemSm90<90, ReturnEntropy>;
	static constexpr bool kUseCluster = kUseNvls;

	const auto& gemm = params.gemm;
	int padded_vocab = HostLaunch::padded_vocab(gemm.local_vocab);

	auto* x = static_cast<const HostElement*>(gemm.x);
	auto* w = static_cast<const HostElement*>(gemm.weight);
	auto* dz = static_cast<HostElement*>(gemm.dz_workspace);
	auto* dw_out = static_cast<HostElement*>(gemm.grad_weight);
	DxReduceWorkspace<float> comm = {};

	int resident_ctas = backward_dx_resident_cta_capacity();
	constexpr int kClusterStoreRows =
		kMaxDxResidentCtas * kDxCommWarpsPerChannel *
		kDxRingStages * TilesPerReduce * HostDxClusterTraits::kTileM;
	int cluster_store_rows =
		resident_ctas * kDxCommWarpsPerChannel *
		kDxRingStages * TilesPerReduce * HostDxClusterTraits::kTileM;
	using ClusterStoreTma = decltype(
		make_row_major_store<
			typename HostDxClusterTraits::SmemStore::Single>(
				static_cast<float*>(nullptr),
				kClusterStoreRows,
				HostDxClusterTraits::kTileN));
	ClusterStoreTma cluster_tma_store = {};
	if constexpr (kUseCluster) {
		LIGER_CHECK(
			reduce.backend ==
				liger_cute::detail::LocalReduceBackend::kNvls,
			"fused_scaled_linear_cross_entropy backward: clustered dX "
			"requires a node-local NVLS team");
		LIGER_CHECK(
			reduce.remote.enabled() == RequiresRemote,
			"fused_scaled_linear_cross_entropy backward: remote reduction "
			"dispatch does not match the configured plan");
		comm = reserve_dx_reduce_workspace(
			TilesPerReduce,
			kDxRingStages,
			resident_ctas);
		cluster_tma_store =
			make_row_major_store<
				typename HostDxClusterTraits::SmemStore::Single>(
					comm.partial,
					cluster_store_rows,
					HostDxClusterTraits::kTileN);
	}
	auto cluster_tma_dz =
		make_row_major_load<typename HostDxClusterTraits::SmemA::Single>(
			dz,
			HostConfig::kWaveRows,
			padded_vocab);
	auto cluster_tma_wt = make_col_major_load_multicast<
		typename HostDxClusterTraits::SmemB::Single,
		HostDxClusterTraits::kFragmentTileN,
		HostDxClusterTraits::kTileK,
		HostDxClusterTraits::kClusterM>(
			w,
			gemm.hidden,
			gemm.local_vocab,
			gemm.hidden);
	auto cluster_tma_dzt = make_col_major_load_multicast<
		typename HostCombinedDwTraits::SmemAMnMajor::Single,
		HostCombinedDwTraits::kTileM,
		HostCombinedDwTraits::kTileK,
		HostDxClusterTraits::kClusterM>(
			dz,
			padded_vocab,
			HostConfig::kWaveRows,
			padded_vocab);
	auto cluster_tma_xt = make_col_major_load<
		typename HostCombinedDwTraits::SmemBMnMajor::Single>(
			x, gemm.hidden, gemm.tokens, gemm.hidden);
	auto cluster_tma_dw_store = make_row_major_store<
		typename HostCombinedDwTraits::DwSmemStore1::Single>(
			dw_out, gemm.local_vocab, gemm.hidden);
	auto cluster_tma_dw_add = make_row_major_reduce_add<
		typename HostCombinedDwTraits::DwSmemStore1::Single>(
			dw_out, gemm.local_vocab, gemm.hidden);
	using ClusterBundle = BackwardClusterCombinedTmaBundleSm90<
		decltype(cluster_tma_dz),
		decltype(cluster_tma_wt),
		ClusterStoreTma,
		decltype(cluster_tma_dzt),
		decltype(cluster_tma_xt),
		decltype(cluster_tma_dw_store),
		decltype(cluster_tma_dw_add)>;
	ClusterBundle cluster_bundle{
		cluster_tma_dz,
		cluster_tma_wt,
		cluster_tma_store,
		cluster_tma_dzt,
		cluster_tma_xt,
		cluster_tma_dw_store,
		cluster_tma_dw_add};
	const bool use_split_k2 =
		backward_dx_split_k(gemm.hidden, gemm.local_vocab) == 2;
	auto* cluster_kernel = use_split_k2
		? &backward_dx_cluster2_gemm_wave_kernel_sm90<
			2, 90, CommConfig, ClusterBundle>
		: &backward_dx_cluster2_gemm_wave_kernel_sm90<
			1, 90, CommConfig, ClusterBundle>;

	auto tma_x =
		make_row_major_load_tile<
			typename HostDzHandoffTraits::SmemTraitsA::Single,
			HostDzHandoffTraits::kTileM,
			HostDzHandoffTraits::kTileK>(
				x, gemm.tokens, gemm.hidden);
	auto tma_w =
		make_row_major_load_tile<
			typename HostDzHandoffTraits::SmemTraitsB::Single,
			HostDzHandoffTraits::kTileN,
			HostDzHandoffTraits::kTileK>(
				w, gemm.local_vocab, gemm.hidden);
	auto tma_dz_store =
		make_row_major_store_tile<
			typename HostDzHandoffTraits::SmemStoreSingle,
			HostDzHandoffTraits::kTileM,
			HostDzHandoffTraits::kStoreTileN>(
				dz, HostConfig::kWaveRows, padded_vocab);
	auto* dz_kernel_fn = &backward_dz_handoff_wave_kernel_sm90<
		ReturnEntropy,
		90,
		decltype(tma_x),
		decltype(tma_w),
		decltype(tma_dz_store)>;
	constexpr int kDzSmemBytes = static_cast<int>(sizeof(DzSmem));
	static_assert(kDzSmemBytes <= kBackwardMaxSmemBytes,
		"SM90 backward dZ shared-memory footprint exceeds the Hopper limit");
	using DzCluster = sm90::ClusterLaunchSm90<90>;
	check_cuda(
		DzCluster::prepare(dz_kernel_fn, kDzSmemBytes),
		"cudaFuncSetAttribute(dZ MaxDynamicSharedMemorySize / "
		"NonPortableClusterSizeAllowed)");
	using DxCluster = sm90::ClusterLaunchSm90<90>;
	constexpr int kClusterSmemBytes =
		static_cast<int>(sizeof(BackwardDxCluster2SmemSm90<90>));
	static_assert(kClusterSmemBytes == 230400,
		"production combined dX+dW shared memory must remain 230400 B");
	static_assert(kClusterSmemBytes <= kBackwardMaxSmemBytes,
		"SM90 dX cluster2 shared-memory footprint exceeds Hopper limit");
	static_assert(HostConfig::kMTilesPerWave %
		HostDxClusterTraits::kClusterM == 0);
	int dx_cluster_count = 0;
	if constexpr (kUseCluster) {
		check_cuda(
			DxCluster::prepare(cluster_kernel, kClusterSmemBytes),
			"cudaFuncSetAttribute(dX cluster2 MaxDynamicSharedMemorySize / "
			"NonPortableClusterSizeAllowed)");
		int max_active_clusters = DxCluster::max_active_clusters(
			cluster_kernel,
			HostDxClusterTraits::kNumThreads,
			kClusterSmemBytes,
			HostDxClusterTraits::kClusterM);
		LIGER_CHECK(
			max_active_clusters > 0,
			"fused_scaled_linear_cross_entropy backward: "
			"no active dX cluster2 fits");
		int total_cluster_tiles =
			(HostConfig::kMTilesPerWave /
				HostDxClusterTraits::kClusterM) *
			HostLaunch::num_dx_n_tiles(gemm.hidden) *
			(use_split_k2 ? 2 : 1);
		int dw_tiles = HostLaunch::num_dw_m_tiles(gemm.local_vocab) *
			HostLaunch::num_dw_n_tiles(gemm.hidden);
		int combined_tiles = total_cluster_tiles > ceil_div(dw_tiles, 2)
			? total_cluster_tiles
			: ceil_div(dw_tiles, 2);
		dx_cluster_count =
			combined_tiles < max_active_clusters
				? combined_tiles
				: max_active_clusters;
		LIGER_CHECK(
			dx_cluster_count * HostDxClusterTraits::kClusterM <=
				resident_ctas,
			"fused_scaled_linear_cross_entropy backward: dX cluster grid "
			"exceeds the configured resident CTA capacity");
	}

	BackwardGemmParamsSm90<90> device_params = gemm;

	int dz_work = HostConfig::kMTilesPerWave * ceil_div(
		padded_vocab,
		HostDzHandoffTraits::kClusterM *
			HostDzHandoffTraits::kTileN);
	int max_dz_clusters = DzCluster::max_active_clusters(
		dz_kernel_fn,
		HostConfig::kNumThreads,
		kDzSmemBytes,
		HostDzHandoffTraits::kClusterM);
	LIGER_CHECK(
		max_dz_clusters > 0,
		"fused_scaled_linear_cross_entropy backward: "
		"no active dZ handoff clusters fit");
	int dz_clusters =
		dz_work < max_dz_clusters ? dz_work : max_dz_clusters;

	int dx_groups_per_wave = HostConfig::kMTilesPerWave *
		ceil_div(HostLaunch::num_dx_n_tiles(gemm.hidden), TilesPerReduce);
	int dw_total = HostLaunch::num_dw_m_tiles(gemm.local_vocab) *
		HostLaunch::num_dw_n_tiles(gemm.hidden);
	int work = dx_groups_per_wave;
	if (dw_total > work) work = dw_total;
	int grid = work < resident_ctas ? work : resident_ctas;
	int num_waves = HostLaunch::num_waves(gemm.tokens);
	if constexpr (!kUseCluster) {
		comm = reserve_dx_reduce_workspace(
			TilesPerReduce, kDxRingStages, grid);
	}
	liger_cute::detail::begin_tp_reduce(comm.launch_epoch, stream);
	if constexpr (kUseCluster) {
		if (use_split_k2) {
			std::size_t durable_elements =
				static_cast<std::size_t>(num_waves) *
				HostConfig::kWaveRows *
				static_cast<std::size_t>(
					HostLaunch::num_dx_n_tiles(gemm.hidden)) *
				HostDxClusterTraits::kTileN /
				static_cast<std::size_t>(
					reduce.nvls.size);
			check_cuda(
				cudaMemsetAsync(
					reduce.nvls.reduced_shard,
					0,
					durable_elements * sizeof(float),
					stream),
				"clear split-K dX durable output");
		}
	}
	for (int wave = 0; wave < num_waves; ++wave) {
		check_cuda(
			DzCluster::launch(
				dz_kernel_fn,
				dim3(
					HostDzHandoffTraits::kClusterM,
					1,
					static_cast<unsigned>(dz_clusters)),
				HostConfig::kNumThreads,
				kDzSmemBytes,
				HostDzHandoffTraits::kClusterM,
				stream,
				tma_x,
				tma_w,
				tma_dz_store,
				device_params,
				wave),
			"cudaLaunchKernelEx(backward_dz_handoff_wave_kernel_sm90)");
		if constexpr (kUseCluster) {
			check_cuda(
				DxCluster::launch(
					cluster_kernel,
					dim3(
						HostDxClusterTraits::kClusterM,
						1,
						static_cast<unsigned>(dx_cluster_count)),
					HostDxClusterTraits::kNumThreads,
					kClusterSmemBytes,
					HostDxClusterTraits::kClusterM,
					stream,
					cluster_bundle,
					comm,
					reduce.nvls,
					device_params,
					wave),
				"cudaLaunchKernelEx("
				"backward_dx_cluster2_gemm_wave_kernel_sm90)");
		} else {
			launch_backward_dx_dw_direct_peer_wave_sm90(
				ReturnEntropy,
				TilesPerReduce,
				device_params,
				comm,
				reduce.direct,
				grid,
				stream,
				wave);
		}
	}
	if constexpr (kUseCluster) {
		int cluster_grid =
			dx_cluster_count * HostDxClusterTraits::kClusterM;
		finalize_backward_dx_cluster_sm90<RequiresRemote>(
			device_params,
			comm,
			reduce.nvls,
			reduce.remote,
			cluster_grid,
			num_waves,
			stream);
	}
	liger_cute::detail::end_tp_reduce(stream);
}

template <int TilesPerReduce, bool ReturnEntropy>
void dispatch_instance(
		const BackwardTpParamsSm90<90>& params,
		const liger_cute::detail::TpReducePlan& reduce,
		cudaStream_t stream) {
	if (reduce.backend ==
		liger_cute::detail::LocalReduceBackend::kNvls) {
		if (reduce.remote.enabled()) {
			launch_instance<
				TilesPerReduce,
				ReturnEntropy,
				liger_cute::detail::LocalReduceBackend::kNvls,
				true>(params, reduce, stream);
		} else {
			launch_instance<
				TilesPerReduce,
				ReturnEntropy,
				liger_cute::detail::LocalReduceBackend::kNvls,
				false>(params, reduce, stream);
		}
	} else {
		launch_instance<
			TilesPerReduce,
			ReturnEntropy,
			liger_cute::detail::LocalReduceBackend::kDirectPeer,
			false>(params, reduce, stream);
	}
}

}  // namespace

template <bool ReturnEntropy, int Compute>
void fused_linear_scaled_cross_entropy_backward(
		const BackwardTpParamsSm90<Compute>& params, cudaStream_t stream) {
	static_assert(Compute == 90,
		"SM90 fused scaled linear cross entropy requires Compute=90");
	if (params.gemm.tokens == 0) return;
	validate(params);

	liger_cute::detail::TpReducePlan reduce =
		liger_cute::detail::tp_reduce_plan();
	switch (params.tiles_per_reduce) {
		case 1:
			dispatch_instance<1, ReturnEntropy>(params, reduce, stream);
			break;
		case 2:
			dispatch_instance<2, ReturnEntropy>(params, reduce, stream);
			break;
		default:
			dispatch_instance<4, ReturnEntropy>(params, reduce, stream);
			break;
	}
}

template void fused_linear_scaled_cross_entropy_backward<false, 90>(
	const BackwardTpParamsSm90<90>&,
	cudaStream_t);
template void fused_linear_scaled_cross_entropy_backward<true, 90>(
	const BackwardTpParamsSm90<90>&,
	cudaStream_t);

}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
