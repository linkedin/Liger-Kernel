// moe.cu — fused MoE fwd/bwd host wrappers (HARNESS STAGE).
//
// Implements the extern "C" boundary declared in moe.h: input validation
// (LIGER_CHECK) and symmetric memory management (global stack + pool) only. No
// kernels are launched yet — each launch site is marked TODO(port). Exceptions
// thrown by LIGER_CHECK / the allocators are caught here and converted to a
// liger_cute_status_t; nothing unwinds across the .so boundary.
#define LIGER_CUTE_BUILDING 1

#include "liger_cute/moe.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

#include "liger_cute/check.h"
#include "liger_cute/detail/symmetric_memory.h"

namespace {

// Symmetric-stack keys for the buffers that persist fwd -> bwd. Lexicographic
// order here is irrelevant; the stack keys on these exact strings.
constexpr const char* kSymmKeyXSorted = "moe.x_sorted";
constexpr const char* kSymmKeyYBuf = "moe.y_buf";
constexpr const char* kSymmKeyAllExpertOffsets = "moe.all_expert_offsets";

constexpr int kTileM = 128;        // fixed by WGMMA
constexpr std::size_t kElemBytes = 2;  // sizeof(__nv_bfloat16)

// Process-wide symmetric configuration captured by configure_symmetric.
struct SymmConfig {
  int max_total_slots = 0;
  int max_num_experts = 0;
  int hidden_dim = 0;
  int num_pes = 0;
  int experts_per_pe = 0;
  int max_top_k = 0;
  bool initialized = false;
};

SymmConfig& symm_config() {
  static SymmConfig cfg;
  return cfg;
}

// Per-thread last-error message backing liger_cute_last_error_string(). A fixed
// buffer (not std::string) so recording an error never allocates — the report
// survives even when the exception being handled is std::bad_alloc. snprintf
// truncates gracefully; 1024 comfortably fits a LIGER_CHECK message.
constexpr int kErrorBufLen = 1024;

char* tls_error_buf() {
  static thread_local char buf[kErrorBufLen] = {0};
  return buf;
}

void set_tls_error(const char* msg) {
  std::snprintf(tls_error_buf(), kErrorBufLen, "%s", msg != nullptr ? msg : "");
}

// Run `body`, translating any escaping exception into a status code + a
// last-error message. This is the ONLY place exceptions are allowed to surface;
// every extern "C" entry point routes through it so nothing crosses the ABI.
template <typename Body>
liger_cute_status_t guarded(Body&& body) {
  try {
    tls_error_buf()[0] = '\0';
    return body();
  } catch (const liger_cute::Error& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    set_tls_error(e.what());
    return LIGER_CUTE_ERR_INTERNAL;
  } catch (...) {
    set_tls_error("liger_cute: unknown error");
    return LIGER_CUTE_ERR_INTERNAL;
  }
}

// Fill an output view to alias `data` with the given shape/dtype.
void set_view(liger_cute::TensorView<2>* v, void* data, int64_t d0, int64_t d1,
              liger_cute_dtype_t dtype) {
  v->data = data;
  v->sizes[0] = d0;
  v->sizes[1] = d1;
  v->dtype = dtype;
}

// Validate that an input view carries the expected dtype. Templated on rank so
// it works for any TensorView<N> (inputs are passed by value — POD, no null).
template <int N>
void require_dtype(const liger_cute::TensorView<N>& v, const char* name,
                   liger_cute_dtype_t want) {
  LIGER_CHECK(v.dtype == want, "moe: input '", name, "' must be ",
              liger_cute_dtype_string(want), ", got ", liger_cute_dtype_string(v.dtype));
}

}  // namespace

extern "C" {

const char* liger_cute_last_error_string(void) {
  return tls_error_buf();  // empty string when no error; never null
}

liger_cute_status_t liger_cute_moe_get_symm_config(liger_cute_moe_symm_config_t* out) {
  return guarded([&]() -> liger_cute_status_t {
    LIGER_CHECK(out != nullptr, "moe_get_symm_config: out is null");
    const SymmConfig& cfg = symm_config();
    out->max_total_slots = cfg.max_total_slots;
    out->max_num_experts = cfg.max_num_experts;
    out->hidden_dim = cfg.hidden_dim;
    out->num_pes = cfg.num_pes;
    out->experts_per_pe = cfg.experts_per_pe;
    out->max_top_k = cfg.max_top_k;
    out->initialized = cfg.initialized ? 1 : 0;
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_moe_configure_symmetric(
    const int max_tokens, const int hidden_dim, const int max_num_experts, const int max_top_k,
    const int num_pes, const int num_hosts, const int gpus_per_host) {
  return guarded([&]() -> liger_cute_status_t {
    // num_hosts × gpus_per_host must equal the team size so the comm schedule
    // addresses exactly the PEs in the team.
    LIGER_CHECK(num_hosts * gpus_per_host == num_pes,
                "moe_configure_symmetric: num_hosts (", num_hosts, ") * gpus_per_host (",
                gpus_per_host, ") = ", num_hosts * gpus_per_host,
                " must equal num_pes (", num_pes, ")");
    LIGER_CHECK(num_pes > 0, "moe_configure_symmetric: num_pes must be positive");
    LIGER_CHECK(max_num_experts % num_pes == 0,
                "moe_configure_symmetric: max_num_experts (", max_num_experts,
                ") must be divisible by num_pes (", num_pes, ")");

    // TODO(port): init_comm_schedule(num_hosts, gpus_per_host) — populate the
    // g_dest_table / g_rank_table device constant memory.

    SymmConfig& cfg = symm_config();
    cfg.max_total_slots = max_tokens * max_top_k + max_num_experts * kTileM;
    cfg.max_num_experts = max_num_experts;
    cfg.hidden_dim = hidden_dim;
    cfg.num_pes = num_pes;
    cfg.experts_per_pe = max_num_experts / num_pes;
    cfg.max_top_k = max_top_k;
    cfg.initialized = true;

    // Preset sizes for the symmetric stack holding X / Y / offsets across
    // fwd -> bwd (bf16 elements; kept in sync with the fwd allocation below).
    const std::size_t symm_slot_bytes =
        static_cast<std::size_t>(cfg.max_total_slots) * cfg.hidden_dim * kElemBytes;
    const std::size_t all_off_bytes =
        static_cast<std::size_t>(cfg.num_pes) * (cfg.max_num_experts + 1) * sizeof(int);

    auto& stack = liger_cute::detail::global_symmetric_stack();
    stack.set_size(kSymmKeyXSorted, symm_slot_bytes);
    stack.set_size(kSymmKeyYBuf, symm_slot_bytes);
    stack.set_size(kSymmKeyAllExpertOffsets, all_off_bytes);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_moe_pop_fwd(void) {
  return guarded([&]() -> liger_cute_status_t {
    auto& stack = liger_cute::detail::global_symmetric_stack();
    // LIFO order: reverse of the put order in the forward.
    stack.pop(kSymmKeyAllExpertOffsets);
    stack.pop(kSymmKeyYBuf);
    stack.pop(kSymmKeyXSorted);
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_moe_fused_fwd_bf16_auto(
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
    int* chosen_tile_m_out) {
  return guarded([&]() -> liger_cute_status_t {
    (void)team_handle;  // unused until the kernel/comm path is ported

    const SymmConfig& cfg = symm_config();
    LIGER_CHECK(cfg.initialized,
                "moe_fused_fwd_bf16_auto: call liger_cute_moe_configure_symmetric first");

    // ── Input validation ────────────────────────────────────────────────
    require_dtype(X, "X", LIGER_CUTE_DTYPE_BFLOAT16);
    require_dtype(expert_indices, "expert_indices", LIGER_CUTE_DTYPE_INT32);
    require_dtype(expert_weights, "expert_weights", LIGER_CUTE_DTYPE_BFLOAT16);
    LIGER_CHECK(all_B.dtype == LIGER_CUTE_DTYPE_BFLOAT16 &&
                    all_C.dtype == LIGER_CUTE_DTYPE_BFLOAT16 &&
                    all_A.dtype == LIGER_CUTE_DTYPE_BFLOAT16,
                "moe_fused_fwd_bf16_auto: expert weights all_B/all_C/all_A must be bf16");

    const int64_t num_tokens = X.sizes[0];
    const int64_t hidden_dim = X.sizes[1];
    LIGER_CHECK(hidden_dim == cfg.hidden_dim,
                "moe_fused_fwd_bf16_auto: X hidden_dim (", hidden_dim,
                ") != configured hidden_dim (", cfg.hidden_dim, ")");
    LIGER_CHECK(num_experts == cfg.max_num_experts,
                "moe_fused_fwd_bf16_auto: num_experts (", num_experts,
                ") != configured max_num_experts (", cfg.max_num_experts, ")");
    LIGER_CHECK(top_k >= 1 && top_k <= cfg.max_top_k,
                "moe_fused_fwd_bf16_auto: top_k (", top_k, ") out of range [1, ",
                cfg.max_top_k, "]");
    LIGER_CHECK(expert_indices.sizes[0] == num_tokens && expert_indices.sizes[1] == top_k,
                "moe_fused_fwd_bf16_auto: expert_indices must be [num_tokens, top_k]");
    LIGER_CHECK(num_tokens * top_k <= cfg.max_total_slots,
                "moe_fused_fwd_bf16_auto: num_tokens*top_k (", num_tokens * top_k,
                ") exceeds configured capacity (", cfg.max_total_slots, ")");

    LIGER_CHECK(Y_out != nullptr && token_expert_slots_out != nullptr &&
                    tile_expert_ids_out != nullptr,
                "moe_fused_fwd_bf16_auto: caller-owned output views must be non-null");
    LIGER_CHECK(x_sorted_out_symm != nullptr && y_buf_out_symm != nullptr &&
                    all_expert_offsets_out_symm != nullptr && chosen_tile_m_out != nullptr,
                "moe_fused_fwd_bf16_auto: symmetric output views must be non-null");

    // ── Symmetric memory management ──────────────────────────────────────
    // Push X / Y / offsets onto the symmetric stack (bottom-to-top order
    // mirrored by moe_pop_fwd). These persist until pop or the matching bwd.
    auto& stack = liger_cute::detail::global_symmetric_stack();
    void* x_sorted = stack.put(kSymmKeyXSorted);
    void* y_buf = stack.put(kSymmKeyYBuf);
    void* all_expert_offsets = stack.put(kSymmKeyAllExpertOffsets);

    set_view(x_sorted_out_symm, x_sorted, cfg.max_total_slots, hidden_dim,
             LIGER_CUTE_DTYPE_BFLOAT16);
    set_view(y_buf_out_symm, y_buf, cfg.max_total_slots, hidden_dim,
             LIGER_CUTE_DTYPE_BFLOAT16);
    set_view(all_expert_offsets_out_symm, all_expert_offsets, cfg.num_pes,
             cfg.max_num_experts + 1, LIGER_CUTE_DTYPE_INT32);

    // TODO(port): autotune (NSplit, TileN, TileM) from shape and launch the
    // fused MoE forward, writing Y_out / token_expert_slots_out /
    // tile_expert_ids_out and the symmetric x_sorted / y_buf / offsets.
    *chosen_tile_m_out = kTileM;
    return LIGER_CUTE_OK;
  });
}

liger_cute_status_t liger_cute_moe_fused_bwd_bf16_auto(
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
    liger_cute::TensorView<2>* dW_out) {
  return guarded([&]() -> liger_cute_status_t {
    (void)Y_fwd;
    (void)token_expert_slots;
    (void)tile_expert_ids;
    (void)expert_offsets;
    (void)expert_indices;
    (void)all_B;
    (void)all_C;
    (void)all_A;
    (void)num_experts;
    (void)team_handle;

    const SymmConfig& cfg = symm_config();
    LIGER_CHECK(cfg.initialized,
                "moe_fused_bwd_bf16_auto: call liger_cute_moe_configure_symmetric first");
    LIGER_CHECK(fwd_tile_m == 64 || fwd_tile_m == 128,
                "moe_fused_bwd_bf16_auto: fwd_tile_m must be 64 or 128, got ", fwd_tile_m);

    require_dtype(dY, "dY", LIGER_CUTE_DTYPE_BFLOAT16);
    require_dtype(x_sorted, "x_sorted", LIGER_CUTE_DTYPE_BFLOAT16);
    require_dtype(expert_weights, "expert_weights", LIGER_CUTE_DTYPE_BFLOAT16);
    LIGER_CHECK(top_k >= 1 && top_k <= cfg.max_top_k,
                "moe_fused_bwd_bf16_auto: top_k (", top_k, ") out of range [1, ",
                cfg.max_top_k, "]");
    LIGER_CHECK(dX_out != nullptr && dB_out != nullptr && dC_out != nullptr &&
                    dA_out != nullptr && dW_out != nullptr,
                "moe_fused_bwd_bf16_auto: gradient output views must be non-null");

    // TODO(port): launch the fused MoE backward, writing dX / dB / dC / dA / dW
    // by reading the symmetric x_sorted / expert_offsets buffers (without
    // consuming them).
    //
    // NOTE: the backward does NOT pop the symmetric stack — it is purely
    // functional w.r.t. the global stack. The caller releases the forward's
    // symmetric buffers explicitly via liger_cute_moe_pop_fwd() (the autograd
    // layer calls it after backward), so there is no hidden state mutation here.
    return LIGER_CUTE_OK;
  });
}

}  // extern "C"
