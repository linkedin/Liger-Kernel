#pragma once

// Whole-program NVSHMEM module state. The generated PTX declarations are made
// visible by export_nonrdc_nvshmem_ptx.py before ptxas assembles the cubin.

#define NVSHMEM_ENABLE_ALL_DEVICE_INLINING

#include <device_host/nvshmem_common.cuh>
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <non_abi/nvshmem_version.h>

#include "liger_cute/detail/comm_schedule.cuh"

#ifndef LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT
#error "LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT must be defined"
#endif

extern "C" {
__device__ __constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
__device__ __constant__ nvshmemi_ibgda_device_state_t
	nvshmemi_ibgda_device_state_d;
__device__ __constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
	NVSHMEM_VENDOR_MAJOR_VERSION,
	NVSHMEM_VENDOR_MINOR_VERSION,
	NVSHMEM_VENDOR_PATCH_VERSION};
__device__ __constant__ unsigned long long
	liger_cute_sm90_nonrdc_build_fingerprint =
		LIGER_CUTE_SM90_NONRDC_BUILD_FINGERPRINT;
__device__ __constant__ int liger_cute_sm90_nonrdc_transport_mode =
	3;
}

#include <device/nvshmem_defines.h>
#include <device/nvshmem_coll_defines.cuh>
#include <device/nvshmemx_defines.h>
#include <device/nvshmemx_coll_defines.cuh>
#include <non_abi/device/pt-to-pt/transfer_device.cuh>

namespace liger_cute {
namespace detail {

__device__ __constant__ int g_dest_table[kMaxPEs];
__device__ __constant__ int g_rank_table[kMaxPEs];

} // namespace detail
} // namespace liger_cute

extern "C" __global__ __launch_bounds__(384, 1)
void liger_cute_sm90_nonrdc_setmaxnreg_probe() {
	const int warp_group = static_cast<int>(threadIdx.x) / 128;
	if (warp_group == 0)
		asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
	else
		asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
}
