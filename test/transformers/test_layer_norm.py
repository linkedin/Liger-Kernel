import pytest
import torch

from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.transformers.functional import liger_layer_norm
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 8, 64),
        (4, 16, 128),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_layer_norm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Test basic layer norm functionality against PyTorch implementation."""
    torch.manual_seed(0)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    liger_x = x.clone().requires_grad_(True)
    torch_x = x.clone().requires_grad_(True)

    liger_ln = LigerLayerNorm(hidden_size, eps=1e-6).to(dtype).to(device)
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6).to(dtype).to(device)

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(liger_x)
    torch_output = torch_ln(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)

    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 8, 64),
        (4, 16, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_layer_norm_functional(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Test functional layer norm interface against autograd function."""
    torch.manual_seed(0)

    input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    x1 = input.clone().requires_grad_(True)
    x2 = input.clone().requires_grad_(True)

    w = torch.randn(hidden_size, device=device, dtype=dtype)
    w1 = w.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)

    b = torch.randn(hidden_size, device=device, dtype=dtype)
    b1 = b.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)

    y1 = liger_layer_norm(X=x1, W=w1, B=b1, eps=1e-6)
    y2 = LigerLayerNormFunction.apply(x2, w2, b2, 1e-6)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y2)
    y1.backward(grad_output, retain_graph=True)
    y2.backward(grad_output, retain_graph=True)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w1.grad, w2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)
