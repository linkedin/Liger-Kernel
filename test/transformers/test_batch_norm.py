import pytest
import torch

from liger_kernel.ops.batch_norm import LigerBatchNormFunction
from liger_kernel.transformers.batch_norm import LigerBatchNorm
from liger_kernel.transformers.functional import liger_batch_norm
from liger_kernel.utils import infer_device

device = infer_device()


# Test for LigerBatchNorm
@pytest.mark.parametrize(
    "batch_size, hidden_size",
    [
        (3, 96),
        (4, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-1, 1e-1),
    ],
)
def test_liger_batch_norm(batch_size, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)

    # Modify the input shape to (N, C)
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    liger_x = x.clone().requires_grad_(True)
    torch_x = x.clone().requires_grad_(True)

    liger_bn = LigerBatchNorm(hidden_size, eps=1e-6).to(dtype).to(device)
    torch_bn = torch.nn.BatchNorm1d(hidden_size, eps=1e-6).to(dtype).to(device)

    with torch.no_grad():
        torch_bn.weight.copy_(liger_bn.weight)
        torch_bn.bias.copy_(liger_bn.bias)

    liger_output = liger_bn(liger_x)
    torch_output = torch_bn(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)

    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_bn.weight.grad, torch_bn.weight.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_bn.bias.grad, torch_bn.bias.grad, atol=atol, rtol=rtol)


# Test for LigerBatchNormFunction
@pytest.mark.parametrize(
    "batch_size, hidden_size",
    [
        (3, 96),
        (4, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_batch_norm_functional(hidden_size, batch_size, dtype, atol, rtol):
    torch.manual_seed(0)

    # Modify the input shape to (N, C)
    input = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    x1 = input.clone().requires_grad_(True)
    x2 = input.clone().requires_grad_(True)

    w = torch.randn(hidden_size, device=device, dtype=dtype)

    w1 = w.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)

    b = torch.randn(hidden_size, device=device, dtype=dtype)

    b1 = b.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)

    # Using LigerBatchNorm function
    y1 = liger_batch_norm(X=x1, gamma=w1, beta=b1, eps=1e-6)
    # Using LigerBatchNormFunction directly
    y2 = LigerBatchNormFunction.apply(x2, w2, b2, 1e-6)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y2)

    y1.backward(grad_output, retain_graph=True)
    y2.backward(grad_output, retain_graph=True)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w1.grad, w2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)
