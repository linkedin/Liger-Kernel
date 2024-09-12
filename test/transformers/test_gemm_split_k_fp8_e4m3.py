from test.utils import assert_verbose_allclose

import pytest
import torch

from liger_kernel.ops.experimental.gemm_split_k_fp8_e4m3 import (
    LigerFP8GemmSplitKFunction,
)

compute_capability = torch.cuda.get_device_capability(0)


def to_float8_e4m3(x):
    scale = x.abs().amax(1) / torch.finfo(torch.float8_e4m3fn).max
    x = x.float() / scale.view(-1, 1)
    return x.to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize(
    "m, k, n",
    [
        (64, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (64, 128, 64),
        (256, 512, 256),
        (512, 1024, 512),
        (1024, 2048, 1024),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-1, 2e-1),
    ],
)
@pytest.mark.skipif(
    compute_capability not in [(8, 9), (9, 0)],
    reason="FP8 GEMM is only supported on SM_89 and higher",
)
def test_gemm_split_k(m, k, n, dtype, atol, rtol):
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)

    c_liger = LigerFP8GemmSplitKFunction.apply(a_fp8, b_fp8)
    c_torch = torch.mm(a_fp8.to(torch.float16), b_fp8.to(torch.float16))

    assert_verbose_allclose(c_liger.to(torch.bfloat16), c_torch, atol=atol, rtol=rtol)

    a_autograd = a_fp8.float().requires_grad_()
    b_autograd = b_fp8.float().requires_grad_()

    c_autograd = torch.matmul(a_autograd, b_autograd)
    c_autograd.backward(torch.ones_like(c_autograd))

    dc = torch.ones_like(c_liger).float()
    da_liger = LigerFP8GemmSplitKFunction.apply(
        dc, b_fp8.t()
    )  # contiguous is already ensured
    db_liger = LigerFP8GemmSplitKFunction.apply(
        a_fp8.t(), dc
    )  # contiguous is already ensured

    assert_verbose_allclose(da_liger.float(), a_autograd.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(db_liger.float(), b_autograd.grad, atol=atol, rtol=rtol)
