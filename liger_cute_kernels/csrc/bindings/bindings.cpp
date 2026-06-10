// bindings.cpp — torch + pybind11 shim for liger_cute_kernels.
//
// This TU is where torch::Tensor is unwrapped into liger_cute::TensorView<N>
// and handed to the ABI-agnostic core (moe.h). It is the ONLY place <torch/...>
// is included; the core never sees it. Recompiled once per torch version.
//
// The torch::Tensor <-> TensorView marshalling helpers live in
// tensor_view_conversion.h (C++-only, not exposed to Python). Symmetric and
// device buffers are owned by the core; the views returned for them alias core
// storage and stay valid until pop / the matching backward.
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tuple>

#include "liger_cute/moe.h"
#include "tensor_view_conversion.h"

namespace py = pybind11;

namespace {

// Throw a Python-visible error carrying the core's last-error message when a
// liger_cute_* entry point returns non-OK.
void check_status(liger_cute_status_t status, const char* what) {
  TORCH_CHECK(status == LIGER_CUTE_OK, "liger_cute: ", what, " failed (",
              liger_cute_status_string(status), "): ", liger_cute_last_error_string());
}

// Empty CUDA tensor with the input's device, used for caller-owned outputs the
// (future) kernel writes into.
torch::Tensor empty_like_device(const torch::Tensor& ref, c10::IntArrayRef sizes,
                                c10::ScalarType dtype) {
  return torch::empty(sizes, ref.options().dtype(dtype));
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "LigerCute: torch bindings over the ABI-agnostic CUTLASS/NVSHMEM core";

  // ── Symmetric configuration ────────────────────────────────────────────
  m.def(
      "moe_configure_symmetric",
      [](int max_tokens, int hidden_dim, int max_num_experts, int max_top_k, int num_pes,
         int num_hosts, int gpus_per_host) {
        check_status(liger_cute_moe_configure_symmetric(max_tokens, hidden_dim, max_num_experts,
                                                        max_top_k, num_pes, num_hosts,
                                                        gpus_per_host),
                     "moe_configure_symmetric");
      },
      py::arg("max_tokens"), py::arg("hidden_dim"), py::arg("max_num_experts"),
      py::arg("max_top_k"), py::arg("num_pes"), py::arg("num_hosts"), py::arg("gpus_per_host"),
      "Size symmetric buffers for the fused MoE forward + populate the comm schedule. "
      "num_hosts * gpus_per_host must equal num_pes. Call once after nvshmem init.");

  m.def(
      "moe_pop_fwd", []() { check_status(liger_cute_moe_pop_fwd(), "moe_pop_fwd"); },
      "Release the most recent fwd's symmetric tensors back to the free stack. Call "
      "collectively on every PE after fwd, unless bwd will consume them.");

  // ── Fused forward (bf16, auto-tuned) ───────────────────────────────────
  m.def(
      "moe_fused_fwd_bf16",
      [](const torch::Tensor& X, const torch::Tensor& expert_indices,
         const torch::Tensor& expert_weights, const torch::Tensor& all_B,
         const torch::Tensor& all_C, const torch::Tensor& all_A, int num_experts, int top_k,
         int64_t team_handle) {
        // Capacity needed to size caller-owned outputs before the call.
        liger_cute_moe_symm_config_t cfg;
        check_status(liger_cute_moe_get_symm_config(&cfg), "moe_get_symm_config");
        TORCH_CHECK(cfg.initialized,
                    "liger_cute: call moe_configure_symmetric before moe_fused_fwd_bf16");

        const int64_t num_tokens = X.size(0);
        const int64_t hidden_dim = X.size(1);
        const int64_t max_m_tiles = (cfg.max_total_slots + 127) / 128;

        // Caller-owned outputs (the kernel writes them; binding owns lifetime).
        torch::Tensor Y = empty_like_device(X, {num_tokens, hidden_dim}, torch::kBFloat16);
        torch::Tensor token_expert_slots =
            empty_like_device(X, {num_tokens, top_k}, torch::kInt32);
        torch::Tensor tile_expert_ids = empty_like_device(X, {max_m_tiles}, torch::kInt32);

        // Output views the core fills / writes into (must be addressable lvalues).
        auto Y_v = liger_cute::to_tensor_view<2>(Y);
        auto slots_v = liger_cute::to_tensor_view<2>(token_expert_slots);
        auto tiles_v = liger_cute::to_tensor_view<1>(tile_expert_ids);
        liger_cute::TensorView<2> x_sorted_v{};  // filled to alias symmetric buffers
        liger_cute::TensorView<2> y_buf_v{};
        liger_cute::TensorView<2> offsets_v{};
        int chosen_tile_m = 0;

        // Inputs are passed by value: each to_tensor_view prvalue is constructed
        // directly into the parameter (guaranteed copy elision in C++17).
        check_status(liger_cute_moe_fused_fwd_bf16_auto(
                         liger_cute::to_tensor_view<2>(X),
                         liger_cute::to_tensor_view<2>(expert_indices),
                         liger_cute::to_tensor_view<2>(expert_weights),
                         liger_cute::to_tensor_view<3>(all_B),
                         liger_cute::to_tensor_view<3>(all_C),
                         liger_cute::to_tensor_view<3>(all_A), num_experts, top_k, team_handle,
                         &Y_v, &slots_v, &tiles_v, &x_sorted_v, &y_buf_v, &offsets_v,
                         &chosen_tile_m),
                     "moe_fused_fwd_bf16");

        // Wrap the symmetric buffers as (non-owning) views on X's device.
        torch::Tensor x_sorted = liger_cute::from_tensor_view<2>(x_sorted_v, X.device());
        torch::Tensor y_buf = liger_cute::from_tensor_view<2>(y_buf_v, X.device());
        torch::Tensor all_expert_offsets = liger_cute::from_tensor_view<2>(offsets_v, X.device());

        return std::make_tuple(Y, x_sorted, y_buf, all_expert_offsets, token_expert_slots,
                               tile_expert_ids, chosen_tile_m);
      },
      py::arg("X"), py::arg("expert_indices"), py::arg("expert_weights"), py::arg("all_B"),
      py::arg("all_C"), py::arg("all_A"), py::arg("num_experts"), py::arg("top_k"),
      py::arg("team_handle"),
      "Auto-tuned fused MoE forward (bf16). Returns (Y, x_sorted, y_buf, all_expert_offsets, "
      "token_expert_slots, tile_expert_ids, chosen_tile_m). x_sorted / y_buf / "
      "all_expert_offsets alias symmetric buffers — keep alive until moe_pop_fwd or the "
      "matching backward. HARNESS STAGE: no kernel launched yet.");

  // ── Fused backward (bf16, auto-tuned) ──────────────────────────────────
  m.def(
      "moe_fused_bwd_bf16",
      [](const torch::Tensor& dY, const torch::Tensor& Y_fwd, const torch::Tensor& x_sorted,
         const torch::Tensor& token_expert_slots, const torch::Tensor& tile_expert_ids,
         const torch::Tensor& expert_offsets, const torch::Tensor& expert_indices,
         const torch::Tensor& expert_weights, const torch::Tensor& all_B,
         const torch::Tensor& all_C, const torch::Tensor& all_A, int num_experts, int top_k,
         int64_t team_handle, int fwd_tile_m) {
        // Gradient outputs mirror the corresponding input shapes.
        torch::Tensor dX = torch::empty_like(Y_fwd);
        torch::Tensor dB = torch::empty_like(all_B);
        torch::Tensor dC = torch::empty_like(all_C);
        torch::Tensor dA = torch::empty_like(all_A);
        torch::Tensor dW = torch::empty_like(expert_weights);

        // Gradient output views the core writes into (must be addressable lvalues).
        auto dX_v = liger_cute::to_tensor_view<2>(dX);
        auto dB_v = liger_cute::to_tensor_view<3>(dB);
        auto dC_v = liger_cute::to_tensor_view<3>(dC);
        auto dA_v = liger_cute::to_tensor_view<3>(dA);
        auto dW_v = liger_cute::to_tensor_view<2>(dW);

        // Inputs by value: prvalue constructed directly into the parameter.
        check_status(liger_cute_moe_fused_bwd_bf16_auto(
                         liger_cute::to_tensor_view<2>(dY),
                         liger_cute::to_tensor_view<2>(Y_fwd),
                         liger_cute::to_tensor_view<2>(x_sorted),
                         liger_cute::to_tensor_view<2>(token_expert_slots),
                         liger_cute::to_tensor_view<1>(tile_expert_ids),
                         liger_cute::to_tensor_view<2>(expert_offsets),
                         liger_cute::to_tensor_view<2>(expert_indices),
                         liger_cute::to_tensor_view<2>(expert_weights),
                         liger_cute::to_tensor_view<3>(all_B),
                         liger_cute::to_tensor_view<3>(all_C),
                         liger_cute::to_tensor_view<3>(all_A), num_experts, top_k, team_handle,
                         fwd_tile_m, &dX_v, &dB_v, &dC_v, &dA_v, &dW_v),
                     "moe_fused_bwd_bf16");

        return std::make_tuple(dX, dB, dC, dA, dW);
      },
      py::arg("dY"), py::arg("Y_fwd"), py::arg("x_sorted"), py::arg("token_expert_slots"),
      py::arg("tile_expert_ids"), py::arg("expert_offsets"), py::arg("expert_indices"),
      py::arg("expert_weights"), py::arg("all_B"), py::arg("all_C"), py::arg("all_A"),
      py::arg("num_experts"), py::arg("top_k"), py::arg("team_handle"), py::arg("fwd_tile_m"),
      "Auto-tuned fused MoE backward (bf16). fwd_tile_m must equal the TileM returned by the "
      "matching forward. Reads (does NOT release) the forward's symmetric buffers — call "
      "moe_pop_fwd() afterwards to release them. Returns (dX, dB, dC, dA, dW). HARNESS STAGE: "
      "no kernel launched yet.");
}
