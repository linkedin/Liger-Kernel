#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Shared SM100 backward launch machinery.
//
// The production launcher and the benchmark-only per-phase launcher build the
// *same* nine TMA descriptors from the same helper, so an isolated phase
// benchmark cannot accidentally measure a different operand layout, tile
// shape or box size than production executes.
// ═══════════════════════════════════════════════════════════════════════════

#include "backward_gemm_mainloop_sm100.cuh"
#include "liger_cute/check.h"
#include "workspace.cuh"

#include <cuda_runtime.h>
#include <cute/atom/copy_traits_sm100_tma.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace liger {
namespace fused_scaled_linear_cross_entropy {
namespace backward_sm100 {

using namespace cute;

using HostConfig = BackwardGemmConfigSm100<100>;
using HostLaunch = BackwardGemmLaunchSm100<100>;
using HostTraits = BackwardGemmTraitsSm100<100>;
using HostDxTraits = BackwardDxTraitsSm100;
using HostDwTraits = BackwardDwTraitsSm100<100>;
using HostElement = HostTraits::Element;
// The fused SM100 dX message is exactly one CTA-owned M128xN256 FP32 tile, so
// the in-kernel staging stride is the TilesPerReduce=1 layout. The pooled
// arena is reserved for the configured maximum, which is a superset.
using HostCommConfig = DxCommConfig<HostConfig, kDxRingStages, 1, 100>;

inline void check_cuda(cudaError_t error, const char* what) {
	LIGER_CHECK(
		error == cudaSuccess,
		"fused_scaled_linear_cross_entropy SM100 backward: ",
		what,
		" failed: ",
		cudaGetErrorString(error));
}

struct ClusterLaunch {
	template <class Kernel>
	static void prepare(Kernel kernel, int smem_bytes) {
		int device = 0;
		int optin = 0;
		check_cuda(cudaGetDevice(&device), "cudaGetDevice");
		check_cuda(
			cudaDeviceGetAttribute(
				&optin,
				cudaDevAttrMaxSharedMemoryPerBlockOptin,
				device),
			"cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");
		LIGER_CHECK(
			smem_bytes <= optin,
			"SM100 backward requires ",
			smem_bytes,
			" B dynamic shared memory, but the device supports ",
			optin,
			" B");
		check_cuda(
			cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared),
			"cudaFuncSetCacheConfig(PreferShared)");
		check_cuda(
			cudaFuncSetAttribute(
				kernel,
				cudaFuncAttributePreferredSharedMemoryCarveout,
				cudaSharedmemCarveoutMaxShared),
			"cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
		check_cuda(
			cudaFuncSetAttribute(
				kernel,
				cudaFuncAttributeClusterSchedulingPolicyPreference,
				cudaClusterSchedulingPolicySpread),
			"cudaFuncSetAttribute(ClusterSchedulingPolicySpread)");
		check_cuda(
			cudaFuncSetAttribute(
				kernel,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				smem_bytes),
			"cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
	}

	static cudaLaunchConfig_t config(
			dim3 grid,
			int threads,
			int smem_bytes,
			cudaStream_t stream,
			cudaLaunchAttribute& cluster_attribute) {
		cluster_attribute = {};
		cluster_attribute.id = cudaLaunchAttributeClusterDimension;
		cluster_attribute.val.clusterDim.x = HostConfig::kClusterM;
		cluster_attribute.val.clusterDim.y = 1;
		cluster_attribute.val.clusterDim.z = 1;
		cudaLaunchConfig_t launch = {};
		launch.gridDim = grid;
		launch.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
		launch.dynamicSmemBytes = smem_bytes;
		launch.stream = stream;
		launch.attrs = &cluster_attribute;
		launch.numAttrs = 1;
		return launch;
	}

	template <class Kernel>
	static int max_active_clusters(
			Kernel kernel, int threads, int smem_bytes) {
		cudaLaunchAttribute cluster_attribute = {};
		cudaLaunchConfig_t launch = config(
			dim3(HostConfig::kClusterM, 1, 1),
			threads,
			smem_bytes,
			nullptr,
			cluster_attribute);
		int clusters = 0;
		if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch) !=
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
			cudaStream_t stream,
			const Args&... args) {
		cudaLaunchAttribute cluster_attribute = {};
		cudaLaunchConfig_t launch = config(
			grid, threads, smem_bytes, stream, cluster_attribute);
		return cudaLaunchKernelEx(&launch, kernel, args...);
	}
};

// ───────────────────────────────────────────────────────────────────────────
// TMA descriptors
//
//   dZ   A = X    (tokens, hidden)        K-major
//        B = W    (vocab,  hidden)        K-major        store: dZ workspace
//   dX   A = dZ   (waveRows, padded)      K-major
//        B = W^T  (hidden, vocab)         MN-major
//   dW   A = dZ^T (padded, waveRows)      MN-major
//        B = X^T  (hidden, tokens)        MN-major       store/add: grad_weight
// ───────────────────────────────────────────────────────────────────────────

struct TmaOperands {
	// CTA-owned FP32 dX staging arena, viewed as [rows, kDxTileN].
	float* dx_staging = nullptr;
	int dx_staging_rows = 0;
	const HostElement* x = nullptr;
	const HostElement* weight = nullptr;
	HostElement* dz = nullptr;
	HostElement* grad_weight = nullptr;
	std::int64_t tokens = 0;
	std::int64_t hidden = 0;
	std::int64_t local_vocab = 0;
	std::int64_t padded_vocab = 0;
};

inline auto make_tma_bundle(const TmaOperands& op) {
	const std::int64_t wave_rows =
		HostConfig::kWaveRows * HostConfig::kDzWorkspaceSlots;

	auto tensor_x = make_tensor(
		make_gmem_ptr(op.x),
		make_shape(op.tokens, op.hidden),
		make_stride(op.hidden, Int<1>{}));
	auto tensor_w = make_tensor(
		make_gmem_ptr(op.weight),
		make_shape(op.local_vocab, op.hidden),
		make_stride(op.hidden, Int<1>{}));
	auto tma_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_x,
		typename HostTraits::SmemLayoutAK1{},
		typename HostTraits::TileShape{},
		typename HostTraits::TiledMmaDz{});
	auto tma_w = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_w,
		typename HostTraits::SmemLayoutBK1{},
		typename HostTraits::TileShape{},
		typename HostTraits::TiledMmaDz{});

	auto tensor_dz_store = make_tensor(
		make_gmem_ptr(op.dz),
		make_shape(wave_rows, op.padded_vocab),
		make_stride(op.padded_vocab, Int<1>{}));
	auto tma_dz_store = make_tma_copy(
		SM90_TMA_STORE{},
		tensor_dz_store,
		typename HostTraits::SmemLayoutStoreSlot{});

	auto tensor_dz = make_tensor(
		make_gmem_ptr(static_cast<const HostElement*>(op.dz)),
		make_shape(wave_rows, op.padded_vocab),
		make_stride(op.padded_vocab, Int<1>{}));
	auto tensor_wt = make_tensor(
		make_gmem_ptr(op.weight),
		make_shape(op.hidden, op.local_vocab),
		make_stride(Int<1>{}, op.hidden));
	auto tma_dz = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_dz,
		typename HostDxTraits::SmemLayoutA1{},
		typename HostDxTraits::TileShape{},
		typename HostDxTraits::TiledMma{});
	auto tma_wt = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_wt,
		typename HostDxTraits::SmemLayoutB1{},
		typename HostDxTraits::TileShape{},
		typename HostDxTraits::TiledMma{});

	auto tensor_dzt = make_tensor(
		make_gmem_ptr(static_cast<const HostElement*>(op.dz)),
		make_shape(op.padded_vocab, wave_rows),
		make_stride(Int<1>{}, op.padded_vocab));
	auto tensor_xt = make_tensor(
		make_gmem_ptr(op.x),
		make_shape(op.hidden, op.tokens),
		make_stride(Int<1>{}, op.hidden));
	auto tma_dzt = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_dzt,
		typename HostDwTraits::SmemLayoutA1{},
		typename HostDwTraits::TileShape{},
		typename HostDwTraits::TiledMma{});
	auto tma_xt = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{},
		tensor_xt,
		typename HostDwTraits::SmemLayoutB1{},
		typename HostDwTraits::TileShape{},
		typename HostDwTraits::TiledMma{});

	auto tensor_dx_store = make_tensor(
		make_gmem_ptr(op.dx_staging),
		make_shape(
			static_cast<std::int64_t>(op.dx_staging_rows),
			static_cast<std::int64_t>(HostCommConfig::kTileN)),
		make_stride(
			static_cast<std::int64_t>(HostCommConfig::kTileN), Int<1>{}));
	auto tma_dx_store = make_tma_copy(
		SM90_TMA_STORE{},
		tensor_dx_store,
		typename HostDxTraits::SmemLayoutStoreTile{});

	auto tensor_dw = make_tensor(
		make_gmem_ptr(op.grad_weight),
		make_shape(op.local_vocab, op.hidden),
		make_stride(op.hidden, Int<1>{}));
	auto tma_dw_store = make_tma_copy(
		SM90_TMA_STORE{},
		tensor_dw,
		typename HostTraits::SmemLayoutStoreSlot{});
	auto tma_dw_add = make_tma_copy(
		SM90_TMA_REDUCE_ADD{},
		tensor_dw,
		typename HostTraits::SmemLayoutStoreSlot{});

	using Bundle = BackwardTmaBundleSm100<
		decltype(tma_x),
		decltype(tma_w),
		decltype(tma_dz_store),
		decltype(tma_dz),
		decltype(tma_wt),
		decltype(tma_dx_store),
		decltype(tma_dzt),
		decltype(tma_xt),
		decltype(tma_dw_store),
		decltype(tma_dw_add)>;
	return Bundle{
		tma_x,
		tma_w,
		tma_dz_store,
		tma_dz,
		tma_wt,
		tma_dx_store,
		tma_dzt,
		tma_xt,
		tma_dw_store,
		tma_dw_add};
}

using Bundle = decltype(make_tma_bundle(std::declval<const TmaOperands&>()));

// Row extent of the pooled staging arena at the configured resident capacity.
inline int dx_staging_rows() {
	return backward_dx_resident_cta_capacity() * kDxCommWarpsPerChannel *
		kDxRingStages * HostCommConfig::kTilesPerReduce *
		HostCommConfig::kTileM;
}

inline TmaOperands operands_of(
		const BackwardGemmParamsSm100<100>& gemm) {
	TmaOperands op;
	op.x = static_cast<const HostElement*>(gemm.x);
	op.weight = static_cast<const HostElement*>(gemm.weight);
	op.dz = static_cast<HostElement*>(gemm.dz_workspace);
	op.grad_weight = static_cast<HostElement*>(gemm.grad_weight);
	op.tokens = gemm.tokens;
	op.hidden = gemm.hidden;
	op.local_vocab = gemm.local_vocab;
	op.padded_vocab = HostLaunch::padded_vocab(gemm.local_vocab);
	return op;
}

// Cluster-pair work count of the widest phase; the grid is shared by all three.
inline int work_cluster_pairs(
		const BackwardGemmParamsSm100<100>& gemm, int phase_mask) {
	int pairs = 0;
	if (phase_mask & kBackwardPhaseDz) {
		int dz = HostLaunch::num_dz_cluster_pairs(gemm.local_vocab);
		if (dz > pairs) pairs = dz;
	}
	if (phase_mask & kBackwardPhaseDx) {
		int dx = HostLaunch::num_dx_cluster_pairs_split(
			gemm.hidden, gemm.local_vocab);
		if (dx > pairs) pairs = dx;
	}
	if (phase_mask & kBackwardPhaseDw) {
		int dw = HostLaunch::num_dw_cluster_pairs(
			gemm.hidden, gemm.local_vocab);
		if (dw > pairs) pairs = dw;
	}
	return pairs;
}

}  // namespace backward_sm100
}  // namespace fused_scaled_linear_cross_entropy
}  // namespace liger
