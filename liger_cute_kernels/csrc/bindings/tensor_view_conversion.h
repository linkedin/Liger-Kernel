// tensor_view_conversion.h — torch::Tensor <-> liger_cute::TensorView<N> glue.
//
// Binding-only (includes <torch/...>): this is where torch::Tensor is unwrapped
// into the flat TensorView the core's host wrappers consume, and where a
// TensorView is rewrapped into a (non-owning) torch::Tensor via from_blob. The
// core never sees any of this.
#pragma once

#include <torch/extension.h>

#include "liger_cute/tensor_view.h"

namespace liger_cute {

// torch scalar type -> liger_cute dtype. Throws (TORCH_CHECK) on an unsupported
// type rather than returning INVALID, so callers fail loudly at the boundary.
inline liger_cute_dtype_t dtype_from_torch(c10::ScalarType st) {
  switch (st) {
    case torch::kFloat32:
      return LIGER_CUTE_DTYPE_FLOAT32;
    case torch::kFloat16:
      return LIGER_CUTE_DTYPE_FLOAT16;
    case torch::kBFloat16:
      return LIGER_CUTE_DTYPE_BFLOAT16;
    case torch::kFloat64:
      return LIGER_CUTE_DTYPE_FLOAT64;
    case torch::kInt8:
      return LIGER_CUTE_DTYPE_INT8;
    case torch::kInt16:
      return LIGER_CUTE_DTYPE_INT16;
    case torch::kInt32:
      return LIGER_CUTE_DTYPE_INT32;
    case torch::kInt64:
      return LIGER_CUTE_DTYPE_INT64;
    case torch::kUInt8:
      return LIGER_CUTE_DTYPE_UINT8;
    case torch::kBool:
      return LIGER_CUTE_DTYPE_BOOL;
    default:
      TORCH_CHECK(false, "liger_cute: unsupported torch dtype ", st);
  }
}

// liger_cute dtype -> torch scalar type.
inline c10::ScalarType dtype_to_torch(liger_cute_dtype_t dt) {
  switch (dt) {
    case LIGER_CUTE_DTYPE_FLOAT32:
      return torch::kFloat32;
    case LIGER_CUTE_DTYPE_FLOAT16:
      return torch::kFloat16;
    case LIGER_CUTE_DTYPE_BFLOAT16:
      return torch::kBFloat16;
    case LIGER_CUTE_DTYPE_FLOAT64:
      return torch::kFloat64;
    case LIGER_CUTE_DTYPE_INT8:
      return torch::kInt8;
    case LIGER_CUTE_DTYPE_INT16:
      return torch::kInt16;
    case LIGER_CUTE_DTYPE_INT32:
      return torch::kInt32;
    case LIGER_CUTE_DTYPE_INT64:
      return torch::kInt64;
    case LIGER_CUTE_DTYPE_UINT8:
      return torch::kUInt8;
    case LIGER_CUTE_DTYPE_BOOL:
      return torch::kBool;
    case LIGER_CUTE_DTYPE_INVALID:
    default:
      TORCH_CHECK(false, "liger_cute: invalid dtype ", liger_cute_dtype_string(dt));
  }
}

// torch::Tensor -> TensorView<N>. The view aliases the tensor's storage (no
// copy), so it must not outlive `t`. Requires a contiguous, rank-N tensor.
template <int N>
TensorView<N> to_tensor_view(const torch::Tensor& t) {
  TORCH_CHECK(t.dim() == N, "liger_cute: expected a rank-", N, " tensor, got rank ", t.dim());
  TORCH_CHECK(t.is_contiguous(), "liger_cute: TensorView requires a contiguous tensor");

  TensorView<N> view;
  view.data = t.data_ptr();
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    view.sizes[i] = t.size(i);
  }
  view.dtype = dtype_from_torch(t.scalar_type());
  return view;
}

// TensorView<N> -> torch::Tensor via from_blob. The result is NON-OWNING: it
// aliases `view.data` and must not outlive the storage the view points at.
// `device` supplies the placement the view itself does not carry (e.g.
// torch::kCUDA, or the source tensor's device on a round trip).
template <int N>
torch::Tensor from_tensor_view(const TensorView<N>& view, c10::Device device) {
  auto options = torch::TensorOptions().dtype(dtype_to_torch(view.dtype)).device(device);
  return torch::from_blob(view.data, c10::IntArrayRef(view.sizes, N), options);
}

}  // namespace liger_cute
