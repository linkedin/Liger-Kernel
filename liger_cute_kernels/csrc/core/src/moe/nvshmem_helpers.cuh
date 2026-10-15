// Device-side NVSHMEM leaf helpers for the MoE kernels.
#pragma once

#include "liger_cute/detail/comm_schedule.cuh"

namespace liger {

#ifdef __CUDACC__
#if defined(LIGER_CUTE_SM90_NONRDC_SPLIT) && \
	!defined(LIGER_CUTE_SM90_NONRDC_LOCAL_TRANSPORT) && \
	!defined(__CUDACC_RDC__)
#define LIGER_MOE_NVSHMEM_LEAF static __device__ __noinline__
#else
#define LIGER_MOE_NVSHMEM_LEAF static __device__ __forceinline__
#endif

LIGER_MOE_NVSHMEM_LEAF void moe_nvshmem_getmem_warp(
		void* dest, const void* source, size_t bytes, int pe) {
	nvshmemx_getmem_warp(dest, source, bytes, pe);
}

LIGER_MOE_NVSHMEM_LEAF void moe_nvshmem_putmem_warp(
		void* dest, const void* source, size_t bytes, int pe) {
	nvshmemx_putmem_warp(dest, source, bytes, pe);
}

LIGER_MOE_NVSHMEM_LEAF void moe_nvshmem_putmem_nbi_warp(
		void* dest, const void* source, size_t bytes, int pe) {
	nvshmemx_putmem_nbi_warp(dest, source, bytes, pe);
}

LIGER_MOE_NVSHMEM_LEAF void moe_nvshmem_quiet() {
	nvshmem_quiet();
}

#undef LIGER_MOE_NVSHMEM_LEAF
#endif

}  // namespace liger
