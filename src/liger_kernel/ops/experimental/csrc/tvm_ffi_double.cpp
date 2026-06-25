#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace {
namespace ffi = tvm::ffi;

void TvmFfiDouble(ffi::TensorView input, ffi::TensorView output) {
    TVM_FFI_ICHECK_EQ(input.device().device_type, kDLCPU);
    TVM_FFI_ICHECK_EQ(output.device().device_type, kDLCPU);
    TVM_FFI_ICHECK_EQ(input.dtype(), (DLDataType{kDLFloat, 32, 1}));
    TVM_FFI_ICHECK_EQ(output.dtype(), (DLDataType{kDLFloat, 32, 1}));
    TVM_FFI_ICHECK_EQ(input.ndim(), output.ndim());
    TVM_FFI_ICHECK(input.IsContiguous());
    TVM_FFI_ICHECK(output.IsContiguous());

    int64_t numel = 1;
    for (int i = 0; i < input.ndim(); ++i) {
        TVM_FFI_ICHECK_EQ(input.size(i), output.size(i));
        numel *= input.size(i);
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    for (int64_t i = 0; i < numel; ++i) {
        output_data[i] = input_data[i] * 2.0f;
    }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tvm_ffi_double, TvmFfiDouble);
