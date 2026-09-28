// moe.h — flat MoE control ABI for liger_cute_kernels.so.
//
// The tensor-carrying MoE entry points are exported through TVM FFI
// (liger_cute_kernels/tvm_ffi_bindings.cpp), which validates tvm::ffi::TensorView
// arguments and lowers them directly into the internal liger::MoeFwdArgs /
// liger::MoeBwdArgs launch bundles. This header intentionally contains only the
// flat control/configuration C ABI that is shared by TVM FFI and native callers.
#pragma once

#include <stdint.h>

#include "liger_cute/export.h"
#include "liger_cute/liger_cute.h"  // liger_cute_status_t, liger_cute_last_error_string

#ifdef __cplusplus

extern "C" {

// POD snapshot of the symmetric configuration (filled by
// liger_cute_moe_get_symm_config). Flat — safe to read from the binding so it
// can size caller-owned output tensors before a forward call.
typedef struct liger_cute_moe_symm_config_t {
  int32_t max_total_slots;
  int32_t max_num_experts;
  int32_t hidden_dim;
  int32_t num_pes;
  int32_t experts_per_pe;
  int32_t max_top_k;
  int32_t initialized;  // 0 until liger_cute_moe_configure_symmetric succeeds
} liger_cute_moe_symm_config_t;

LIGER_CUTE_API liger_cute_status_t
liger_cute_moe_get_symm_config(liger_cute_moe_symm_config_t* out);

// Size symmetric buffers + populate the comm schedule. Call once per process
// after NVSHMEM init and before any forward. num_hosts*gpus_per_host must equal
// num_pes.
LIGER_CUTE_API liger_cute_status_t liger_cute_moe_configure_symmetric(
    const int max_tokens, const int hidden_dim, const int max_num_experts, const int max_top_k,
    const int num_pes, const int num_hosts, const int gpus_per_host);

// Release the most recent forward's symmetric tensors (x_sorted / y_buf /
// all_expert_offsets) back to the free stack. Collective; LIFO vs. forward.
// Skip when the matching backward will consume them.
LIGER_CUTE_API liger_cute_status_t liger_cute_moe_pop_fwd(void);

}  // extern "C"

#endif  // __cplusplus
