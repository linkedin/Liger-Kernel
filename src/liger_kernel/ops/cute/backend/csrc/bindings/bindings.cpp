// bindings.cpp — torch + pybind11 shim for liger_cute_kernels.
//
// This TU is where torch::Tensor is unwrapped into the flat POD types and
// handed to the ABI-agnostic core. It is the ONLY place <torch/...> is
// included; the core never sees it. Recompiled once per torch version.
//
// The torch::Tensor <-> liger_cute::TensorView<N> marshalling helpers live in
// tensor_view_conversion.h; they are C++-only glue used when binding ops and
// are intentionally NOT exposed to Python.
//
// NOTE: harness stage — the module is intentionally empty. The real MoE /
// NVSHMEM ops get bound here as they are ported.
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "liger_cute/liger_cute.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "LigerCute: torch bindings over the ABI-agnostic CUTLASS/NVSHMEM core";
}
