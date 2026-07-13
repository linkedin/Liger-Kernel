// liger_cute.h — public ABI of liger_cute_kernels.so.
//
// This is the stable C control surface. Tensor-carrying entry points are exposed
// through TVM FFI. The C boundary stays a flat `extern "C"` interface: raw
// pointers, fixed-width ints, enums, POD structs, and no exceptions, which is
// what makes the .so ABI-agnostic across torch wheels (see export.h).
// CUTLASS/CuTe usage lives entirely behind it in the .cu/.cpp translation units.
//
// NOTE: harness stage — the status type + shared error string live here; the
// MoE entry points are in moe.h and the NVSHMEM bootstrap/team/comm-schedule
// entry points in nvshmem.h, both behind this same boundary.
#pragma once

#include "liger_cute/export.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Status / error reporting (no exceptions across the ABI) ──────────────────
typedef enum liger_cute_status_t {
  LIGER_CUTE_OK = 0,
  LIGER_CUTE_ERR_INVALID_ARGUMENT = 1,
  LIGER_CUTE_ERR_UNSUPPORTED = 2,
  LIGER_CUTE_ERR_CUDA = 3,
  LIGER_CUTE_ERR_NVSHMEM = 4,
  LIGER_CUTE_ERR_INTERNAL = 5,
} liger_cute_status_t;

// Translate a status code to a static, human-readable string.
LIGER_CUTE_API const char* liger_cute_status_string(liger_cute_status_t status);

// Most recent error message on the calling thread, set whenever ANY liger_cute_*
// entry point returns non-OK. Static thread-local storage; never null (empty
// string when there is no error). Cleared at the start of each entry point.
// Shared across all entry-point families (MoE, NVSHMEM, ...) so the binding's
// status check reads one place regardless of which call failed.
LIGER_CUTE_API const char* liger_cute_last_error_string(void);

#ifdef __cplusplus
}  // extern "C"
#endif
