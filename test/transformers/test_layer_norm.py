import pytest
import torch

from liger_kernel.ops import LigerLayerNormFunction
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
        (1, 1, 1500),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),  # Relaxed tolerance for bfloat16 due to lower precision + atomic limitations
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
        (3, 512, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),  # Relaxed tolerance for bfloat16 due to lower precision + atomic limitations
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


@pytest.mark.parametrize(
    "n_rows, hidden_size",
    [
        (4096, 16),
        (4103, 61),  # ragged row blocks + odd hidden size
        (32771, 123),
        (65537, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_liger_layer_norm_block_row(
    n_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Exercise the block-row LayerNorm kernels (BLOCK_SIZE <= 256 and n_rows >= 4096)."""
    torch.manual_seed(0)

    x = torch.randn(n_rows, hidden_size, dtype=dtype, device=device)
    dy = torch.randn_like(x)

    x_blk = x.clone().requires_grad_(True)
    x_ref = x.detach().to(torch.float32).clone().requires_grad_(True)

    w = torch.randn(hidden_size, dtype=dtype, device=device)
    w_blk = w.clone().requires_grad_(True)
    w_ref = w.detach().to(torch.float32).clone().requires_grad_(True)

    b = torch.randn(hidden_size, dtype=dtype, device=device)
    b_blk = b.clone().requires_grad_(True)
    b_ref = b.detach().to(torch.float32).clone().requires_grad_(True)

    y_blk = liger_layer_norm(x_blk, w_blk, b_blk, eps=1e-6, row_mode=None)
    y_ref = torch.nn.functional.layer_norm(x_ref, (hidden_size,), w_ref, b_ref, eps=1e-6)

    assert torch.allclose(y_blk, y_ref.to(dtype), atol=atol, rtol=rtol)

    y_blk.backward(dy)
    y_ref.backward(dy.to(torch.float32))

    param_atol = 5e-2 if dtype == torch.bfloat16 else 5e-3
    param_rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4

    assert torch.allclose(x_blk.grad, x_ref.grad.to(dtype), atol=atol, rtol=rtol)
    assert torch.allclose(w_blk.grad, w_ref.grad.to(dtype), atol=param_atol, rtol=param_rtol)
    assert torch.allclose(b_blk.grad, b_ref.grad.to(dtype), atol=param_atol, rtol=param_rtol)
