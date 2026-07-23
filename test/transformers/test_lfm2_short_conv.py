import pytest
import torch
import torch.nn.functional as F

from liger_kernel.ops import LigerLfm2ShortConvFunction
from liger_kernel.utils import infer_device

device = infer_device()


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
