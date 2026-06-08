// liger_cute.cu — placeholder translation unit for liger_cute_kernels.so.
//
// Exists so the harness produces a real, loadable shared library before any
// functional kernel is ported. Also the canonical example of the boundary
// contract: a .cu TU (where CUTLASS/CuTe and NVSHMEM device code will
// eventually live) exposing ONLY flat `extern "C"` symbols tagged
// LIGER_CUTE_API.
#define LIGER_CUTE_BUILDING 1

#include "liger_cute/liger_cute.h"

extern "C" {

const char* liger_cute_status_string(liger_cute_status_t status) {
  switch (status) {
    case LIGER_CUTE_OK:
      return "ok";
    case LIGER_CUTE_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case LIGER_CUTE_ERR_UNSUPPORTED:
      return "unsupported";
    case LIGER_CUTE_ERR_CUDA:
      return "cuda error";
    case LIGER_CUTE_ERR_NVSHMEM:
      return "nvshmem error";
    case LIGER_CUTE_ERR_INTERNAL:
      return "internal error";
    default:
      return "unknown status";
  }
}

}  // extern "C"
