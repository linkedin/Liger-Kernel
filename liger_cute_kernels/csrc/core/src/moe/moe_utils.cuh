#pragma once

#include <cuda_runtime.h>

#include "liger_cute/check.h"
#include "liger_cute/detail/status.h"

namespace liger {

inline int moe_compute_dispatch_key_from_capability(int major, int minor) {
	if (major == 9) return 90;
	if (major == 10) return 100;
	LIGER_CHECK(false,
		"unsupported CUDA compute capability sm_", major, minor,
		". Supported compute capability families are sm_9x (Hopper) "
		"and sm_10x (Blackwell).");
	return 0;
}

inline int moe_detect_compute_dispatch_key(int device, const char* context) {
	int major = 0;
	int minor = 0;
	if (cudaError_t e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); e != cudaSuccess)
		LIGER_FAIL_CUDA(context, ": cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor) failed: ",
		                cudaGetErrorString(e));
	if (cudaError_t e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); e != cudaSuccess)
		LIGER_FAIL_CUDA(context, ": cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMinor) failed: ",
		                cudaGetErrorString(e));
	return moe_compute_dispatch_key_from_capability(major, minor);
}

inline int moe_detect_compute_dispatch_key_noexcept(int device) {
	int major = 0;
	int minor = 0;
	if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
	    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
		return 0;
	}
	try {
		return moe_compute_dispatch_key_from_capability(major, minor);
	} catch (...) {
		return -(major * 10 + minor);
	}
}

} // namespace liger
