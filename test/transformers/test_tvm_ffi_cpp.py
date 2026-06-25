import torch

from liger_kernel.ops.experimental import tvm_ffi_double


def test_tvm_ffi_double_cpu():
    x = torch.arange(8, dtype=torch.float32)
    y = tvm_ffi_double(x)
    torch.testing.assert_close(y, x * 2)
