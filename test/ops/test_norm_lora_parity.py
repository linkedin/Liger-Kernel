from __future__ import annotations

import pytest
import torch

from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction


def _torch_rms_norm_ref(x, w, eps=1e-6, offset=0.0):
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normed = (x.float() / rms).to(x.dtype)
    if w is not None:
        return x_normed * (w + offset)
    return x_normed


def _torch_layer_norm_ref(x, w, b, eps=1e-6):
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    x_hat = ((x.float() - mean) * rstd).to(x.dtype)
    return x_hat * w + b


def _torch_fused_add_rms_norm_ref(x, r, w, eps=1e-6, offset=0.0):
    s = x + r
    y = _torch_rms_norm_ref(s, w, eps=eps, offset=offset)
    return y, s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_requires_grad", [False, True])
@pytest.mark.parametrize("shape", [(16, 256), (32, 4096)])
def test_rms_norm_lora_parity(shape, dtype, w_requires_grad):
    torch.manual_seed(42)
    device = "cuda"
    B_T, H = shape

    x_liger = torch.randn(B_T, H, dtype=dtype, device=device, requires_grad=True)
    x_ref = x_liger.detach().clone().requires_grad_(True)

    w_liger = torch.randn(H, dtype=dtype, device=device, requires_grad=w_requires_grad)
    w_ref = w_liger.detach().clone().requires_grad_(w_requires_grad)

    eps = 1e-6

    # Forward
    y_liger = LigerRMSNormFunction.apply(x_liger, w_liger, eps, 0.0, "llama", False, None)
    y_ref = _torch_rms_norm_ref(x_ref, w_ref, eps, 0.0)

    # Backward
    dy = torch.randn_like(y_liger)
    y_liger.backward(dy)
    y_ref.backward(dy)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(x_liger.grad, x_ref.grad, atol=atol, rtol=rtol)

    if w_requires_grad:
        assert w_liger.grad is not None
        assert w_ref.grad is not None
        torch.testing.assert_close(w_liger.grad, w_ref.grad, atol=atol, rtol=rtol)
    else:
        assert w_liger.grad is None
        assert w_ref.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("params_require_grad", [False, True])
@pytest.mark.parametrize("shape", [(16, 256), (32, 4096)])
def test_layer_norm_lora_parity(shape, dtype, params_require_grad):
    torch.manual_seed(42)
    device = "cuda"
    B_T, H = shape

    x_liger = torch.randn(B_T, H, dtype=dtype, device=device, requires_grad=True)
    x_ref = x_liger.detach().clone().requires_grad_(True)

    w_liger = torch.randn(H, dtype=dtype, device=device, requires_grad=params_require_grad)
    w_ref = w_liger.detach().clone().requires_grad_(params_require_grad)

    b_liger = torch.randn(H, dtype=dtype, device=device, requires_grad=params_require_grad)
    b_ref = b_liger.detach().clone().requires_grad_(params_require_grad)

    eps = 1e-6

    # Forward
    y_liger = LigerLayerNormFunction.apply(x_liger, w_liger, b_liger, eps)
    y_ref = _torch_layer_norm_ref(x_ref, w_ref, b_ref, eps)

    # Backward
    dy = torch.randn_like(y_liger)
    y_liger.backward(dy)
    y_ref.backward(dy)

    atol = 1e-1 if dtype == torch.bfloat16 else 1e-4
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(x_liger.grad, x_ref.grad, atol=atol, rtol=rtol)

    if params_require_grad:
        assert w_liger.grad is not None
        assert b_liger.grad is not None
        torch.testing.assert_close(w_liger.grad, w_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(b_liger.grad, b_ref.grad, atol=atol, rtol=rtol)
    else:
        assert w_liger.grad is None
        assert b_liger.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_requires_grad", [False, True])
@pytest.mark.parametrize("shape", [(16, 256), (32, 4096)])
def test_fused_add_rms_norm_lora_parity(shape, dtype, w_requires_grad):
    torch.manual_seed(42)
    device = "cuda"
    B_T, H = shape

    x_liger = torch.randn(B_T, H, dtype=dtype, device=device, requires_grad=True)
    x_ref = x_liger.detach().clone().requires_grad_(True)

    r_liger = torch.randn(B_T, H, dtype=dtype, device=device, requires_grad=True)
    r_ref = r_liger.detach().clone().requires_grad_(True)

    w_liger = torch.randn(H, dtype=dtype, device=device, requires_grad=w_requires_grad)
    w_ref = w_liger.detach().clone().requires_grad_(w_requires_grad)

    eps = 1e-6

    # Forward
    y_liger, s_liger = LigerFusedAddRMSNormFunction.apply(x_liger, r_liger, w_liger, eps, 0.0, "llama", False)
    y_ref, s_ref = _torch_fused_add_rms_norm_ref(x_ref, r_ref, w_ref, eps, 0.0)

    # Backward
    dy = torch.randn_like(y_liger)
    ds = torch.randn_like(s_liger)
    torch.autograd.backward((y_liger, s_liger), (dy, ds))
    torch.autograd.backward((y_ref, s_ref), (dy.clone(), ds.clone()))

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    torch.testing.assert_close(x_liger.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(r_liger.grad, r_ref.grad, atol=atol, rtol=rtol)

    if w_requires_grad:
        assert w_liger.grad is not None
        torch.testing.assert_close(w_liger.grad, w_ref.grad, atol=atol, rtol=rtol)
    else:
        assert w_liger.grad is None
