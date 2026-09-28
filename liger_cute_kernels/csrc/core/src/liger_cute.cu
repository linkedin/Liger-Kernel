// liger_cute.cu — base translation unit for liger_cute_kernels.so.
//
// Home of the shared, family-agnostic ABI: the status-string table and the
// thread-local last-error accessor that every other entry-point family (MoE,
// NVSHMEM) reports through. Also the canonical example of the boundary
// contract: a .cu TU (where CUTLASS/CuTe and NVSHMEM device code will
// eventually live) exposing ONLY flat `extern "C"` symbols tagged
// LIGER_CUTE_API.
#define LIGER_CUTE_BUILDING 1

#include "liger_cute/liger_cute.h"

#include "liger_cute/detail/status.h"

extern "C" {

const char* liger_cute_last_error_string(void) {
  return liger_cute::detail::tls_error_buf();  // empty string when no error; never null
}

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
