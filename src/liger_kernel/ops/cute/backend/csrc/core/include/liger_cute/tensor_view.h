// tensor_view.h — rank-templated POD view passed to the lck host wrappers.
//
// A TensorView<N> is the intermediate representation the binding layer marshals
// a torch::Tensor into before calling a host wrapper: a non-owning window onto
// element storage, described by a base pointer, N extents, and an element
// dtype. It is header-only and POD (trivially copyable), so it can be used on
// either side of the build without dragging torch or std:: into the core.
//
// The element-dtype enum is a flat C enum (it can cross the extern "C" ABI in
// liger_cute.h). The TensorView<N> template itself is C++-only and is never
// named across the flat boundary — exported `extern "C"` entry points take the
// unpacked PODs (pointer / sizes / dtype) and reconstruct the view internally.
#pragma once

#include <stdint.h>

#include "liger_cute/export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Element data type of a TensorView. Covers the subset of torch scalar types
// the MoE kernels consume; extend as kernels need more.
typedef enum liger_cute_dtype_t {
  LIGER_CUTE_DTYPE_INVALID = 0,
  LIGER_CUTE_DTYPE_FLOAT32 = 1,
  LIGER_CUTE_DTYPE_FLOAT16 = 2,
  LIGER_CUTE_DTYPE_BFLOAT16 = 3,
  LIGER_CUTE_DTYPE_FLOAT64 = 4,
  LIGER_CUTE_DTYPE_INT8 = 5,
  LIGER_CUTE_DTYPE_INT16 = 6,
  LIGER_CUTE_DTYPE_INT32 = 7,
  LIGER_CUTE_DTYPE_INT64 = 8,
  LIGER_CUTE_DTYPE_UINT8 = 9,
  LIGER_CUTE_DTYPE_BOOL = 10,
} liger_cute_dtype_t;

// Human-readable name for a dtype (static storage; never null).
LIGER_CUTE_API const char* liger_cute_dtype_string(liger_cute_dtype_t dtype);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus

namespace liger_cute {

// Non-owning, rank-N view over contiguous element storage.
//
//   * data  — base pointer to the first element (device or host memory).
//   * sizes — extent along each dimension, outermost first.
//   * dtype — element type held at `data`.
//
// Strides are intentionally absent: a TensorView describes contiguous,
// row-major storage. The binding layer enforces contiguity when constructing
// one from a torch::Tensor.
template <int N>
struct TensorView {
  static_assert(N > 0, "TensorView rank must be positive");

  void* data;
  int64_t sizes[N];
  liger_cute_dtype_t dtype;

  static constexpr int rank = N;
};

}  // namespace liger_cute

#endif  // __cplusplus
