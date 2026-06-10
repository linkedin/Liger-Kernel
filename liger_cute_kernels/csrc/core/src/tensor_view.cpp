// tensor_view.cpp — flat ABI helpers for the TensorView dtype enum.
//
// Lives in the torch-free core (like liger_cute.cu) and exposes only
// LIGER_CUTE_API `extern "C"` symbols. The TensorView<N> template itself is
// header-only (tensor_view.h) and needs no translation unit.
#define LIGER_CUTE_BUILDING 1

#include "liger_cute/tensor_view.h"

extern "C" {

const char* liger_cute_dtype_string(liger_cute_dtype_t dtype) {
  switch (dtype) {
    case LIGER_CUTE_DTYPE_INVALID:
      return "invalid";
    case LIGER_CUTE_DTYPE_FLOAT32:
      return "float32";
    case LIGER_CUTE_DTYPE_FLOAT16:
      return "float16";
    case LIGER_CUTE_DTYPE_BFLOAT16:
      return "bfloat16";
    case LIGER_CUTE_DTYPE_FLOAT64:
      return "float64";
    case LIGER_CUTE_DTYPE_INT8:
      return "int8";
    case LIGER_CUTE_DTYPE_INT16:
      return "int16";
    case LIGER_CUTE_DTYPE_INT32:
      return "int32";
    case LIGER_CUTE_DTYPE_INT64:
      return "int64";
    case LIGER_CUTE_DTYPE_UINT8:
      return "uint8";
    case LIGER_CUTE_DTYPE_BOOL:
      return "bool";
    default:
      return "unknown dtype";
  }
}

}  // extern "C"
