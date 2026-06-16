// moe.h — fused MoE fwd/bwd host-wrapper boundary (C++-only, extern "C" linkage).
//
// Unlike liger_cute.h, this header is C++-ONLY: its entry points take POD
// liger_cute::TensorView<N> directly (inputs by value, outputs by pointer)
// instead of a hand-rolled flat tensor descriptor. TensorView<N> is
// trivially-copyable POD (no std::, no torch), so its layout is fixed by the
// platform C++ ABI and is independent of _GLIBCXX_USE_CXX11_ABI — it crosses the
// core<->binding boundary as safely as a C struct would, while keeping a single
// tensor representation. Inputs pass by value (a prvalue argument is constructed
// in place — guaranteed copy elision); outputs are pointers the core writes
// through.
//
// The functions keep `extern "C"` linkage so they export as plain liger_cute_*
// symbols (caught by the version script) and so the rule "no exceptions cross
// the boundary" is explicit: each entry point catches internally and reports
// via liger_cute_status_t + liger_cute_last_error_string() (declared in
// liger_cute.h — shared across all entry-point families). The torch::Tensor
// <-> TensorView marshalling lives in the binding (_C); the core never sees
// torch.
#pragma once

#include <stdint.h>

#include "liger_cute/export.h"
#include "liger_cute/liger_cute.h"  // liger_cute_status_t, liger_cute_last_error_string
#include "liger_cute/tensor_view.h"

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

// Auto-tuned fused MoE forward (bf16).
//
// Inputs are validated; the symmetric X/Y/offset buffers are allocated from the
// global stack and the *_out_symm views are filled to alias them (keep alive
// until pop / the matching backward). The caller-owned outputs
// (Y / token_expert_slots / tile_expert_ids) are allocated by the binding and
// passed in; the kernel writes them.
//
// HARNESS STAGE: validation + symmetric memory management only — no kernel is
// launched yet (see TODO in moe.cu). *chosen_tile_m_out is the WGMMA default
// (128) until the autotuner is ported.
LIGER_CUTE_API liger_cute_status_t liger_cute_moe_fused_fwd_bf16_auto(
    const liger_cute::TensorView<2> X,
    const liger_cute::TensorView<2> expert_indices,
    const liger_cute::TensorView<2> expert_weights,
    const liger_cute::TensorView<3> all_B,
    const liger_cute::TensorView<3> all_C,
    const liger_cute::TensorView<3> all_A,
    const int num_experts,
    const int top_k,
    const int64_t team_handle,
    liger_cute::TensorView<2>* Y_out,
    liger_cute::TensorView<2>* token_expert_slots_out,
    liger_cute::TensorView<1>* tile_expert_ids_out,
    liger_cute::TensorView<2>* x_sorted_out_symm,
    liger_cute::TensorView<2>* y_buf_out_symm,
    liger_cute::TensorView<2>* all_expert_offsets_out_symm,
    int* chosen_tile_m_out);

// Auto-tuned fused MoE backward (bf16). Reads the forward's intermediates
// (including the symmetric x_sorted / expert_offsets) plus dY. fwd_tile_m must
// equal the TileM returned by the matching forward. Caller-owned gradient
// outputs are allocated by the binding and passed in.
//
// This call is functional w.r.t. the symmetric stack: it does NOT pop. The
// caller releases the forward's symmetric buffers explicitly via
// liger_cute_moe_pop_fwd() (the autograd layer calls it after backward).
//
// HARNESS STAGE: validation only; no kernel is launched.
LIGER_CUTE_API liger_cute_status_t liger_cute_moe_fused_bwd_bf16_auto(
    const liger_cute::TensorView<2> dY,
    const liger_cute::TensorView<2> Y_fwd,
    const liger_cute::TensorView<2> x_sorted,
    const liger_cute::TensorView<2> token_expert_slots,
    const liger_cute::TensorView<1> tile_expert_ids,
    const liger_cute::TensorView<2> expert_offsets,
    const liger_cute::TensorView<2> expert_indices,
    const liger_cute::TensorView<2> expert_weights,
    const liger_cute::TensorView<3> all_B,
    const liger_cute::TensorView<3> all_C,
    const liger_cute::TensorView<3> all_A,
    const int num_experts,
    const int top_k,
    const int64_t team_handle,
    const int fwd_tile_m,
    liger_cute::TensorView<2>* dX_out,
    liger_cute::TensorView<3>* dB_out,
    liger_cute::TensorView<3>* dC_out,
    liger_cute::TensorView<3>* dA_out,
    liger_cute::TensorView<2>* dW_out);

}  // extern "C"

#endif  // __cplusplus
