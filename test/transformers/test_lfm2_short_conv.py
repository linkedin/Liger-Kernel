import pytest
import torch
import torch.nn.functional as F

import liger_kernel.ops.lfm2_short_conv as short_conv_ops

from liger_kernel.ops import LigerLfm2ShortConvFunction
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    ("arch", "batch_tokens", "expected"),
    [
        ("hopper", 16383, (256, None, None)),
        ("hopper", 16384, (128, 4, 2)),
        ("hopper", 32767, (128, 4, 2)),
        ("hopper", 32768, (64, 2, 2)),
        ("cdna3", 65536, (256, None, None)),
        ("ampere_ada", 65536, (256, None, None)),
        ("blackwell", 65536, (256, None, None)),
    ],
)
def test_lfm2_short_conv_weight_backward_dispatch(monkeypatch, arch, batch_tokens, expected):
    monkeypatch.setattr(short_conv_ops, "infer_device_arch", lambda: arch)
    assert short_conv_ops._short_conv_weight_backward_config(batch_tokens) == expected


def _reference(bcx, weight, bias):
    gate_b, gate_c, value = bcx.chunk(3, dim=-1)
    product = (gate_b * value).transpose(1, 2)
    conv = F.conv1d(product, weight, bias=bias, padding=weight.shape[-1] - 1, groups=weight.shape[0])
    conv = conv[..., : bcx.shape[1]].transpose(1, 2)
    return gate_c * conv


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_enabled", [False, True])
@pytest.mark.parametrize("shape", [(2, 17, 32, 3), (1, 128, 64, 4)])
def test_lfm2_short_conv_forward_backward(dtype, bias_enabled, shape):
    batch, seq_len, hidden_size, kernel_size = shape
    torch.manual_seed(42)
    bcx = torch.randn(batch, seq_len, 3 * hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, 1, kernel_size, device=device, dtype=dtype) * 0.02
    bias = torch.randn(hidden_size, device=device, dtype=dtype) * 0.02 if bias_enabled else None
    grad = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)

    bcx_ref = bcx.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True) if bias_enabled else None
    output_ref = _reference(bcx_ref, weight_ref, bias_ref)
    output_ref.backward(grad)

    bcx_liger = bcx.detach().clone().requires_grad_(True)
    weight_liger = weight.detach().clone().requires_grad_(True)
    bias_liger = bias.detach().clone().requires_grad_(True) if bias_enabled else None
    output_liger = LigerLfm2ShortConvFunction.apply(bcx_liger, weight_liger, bias_liger)
    output_liger.backward(grad)

    atol = 1e-5 if dtype == torch.float32 else 5e-2
    rtol = 1e-5 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(output_liger, output_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(bcx_liger.grad, bcx_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(weight_liger.grad, weight_ref.grad, atol=atol, rtol=rtol)
    if bias_enabled:
        torch.testing.assert_close(bias_liger.grad, bias_ref.grad, atol=atol, rtol=rtol)


def test_lfm2_short_conv_long_sequence_backward(monkeypatch):
    """Exercise the Hopper-only long-sequence weight-reduction configuration."""
    torch.manual_seed(123)
    shape = (1, 16384, 32, 3)
    batch, seq_len, hidden_size, kernel_size = shape
    bcx = torch.randn(batch, seq_len, 3 * hidden_size, device=device, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, 1, kernel_size, device=device, dtype=torch.bfloat16) * 0.02
    grad = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    bcx_ref = bcx.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    output_ref = _reference(bcx_ref, weight_ref, None)
    output_ref.backward(grad)

    monkeypatch.setattr(short_conv_ops, "infer_device_arch", lambda: "ampere_ada")
    bcx_portable = bcx.detach().clone().requires_grad_(True)
    weight_portable = weight.detach().clone().requires_grad_(True)
    output_portable = LigerLfm2ShortConvFunction.apply(bcx_portable, weight_portable, None)
    output_portable.backward(grad)

    monkeypatch.setattr(short_conv_ops, "infer_device_arch", lambda: "hopper")
    bcx_liger = bcx.detach().clone().requires_grad_(True)
    weight_liger = weight.detach().clone().requires_grad_(True)
    output_liger = LigerLfm2ShortConvFunction.apply(bcx_liger, weight_liger, None)
    output_liger.backward(grad)

    torch.testing.assert_close(output_liger, output_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(bcx_liger.grad, bcx_ref.grad, atol=5e-2, rtol=5e-2)
    # Long BF16 reductions have the same rounding error on the portable and
    # Hopper paths. Require the tuned launch to match the portable result
    # exactly, then compare that common result with a reduction-aware tolerance.
    torch.testing.assert_close(weight_liger.grad, weight_portable.grad, atol=0, rtol=0)
    torch.testing.assert_close(weight_liger.grad, weight_ref.grad, atol=5e-1, rtol=8e-2)
