import pytest
import torch

from liger_kernel.transformers.functional import liger_softmax
from liger_kernel.transformers.softmax import LigerKernelSoftmax
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),  # Large single row
        (3, 7, 256),  # 3D input
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 5e-2, 5e-2)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        else pytest.param(torch.bfloat16, 0, 0, marks=pytest.mark.skip(reason="bfloat16 not supported")),
    ],
)
def test_liger_softmax(shape, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    torch_softmax = torch.nn.Softmax(dim=-1)
    ref_out = torch_softmax(x1)
    liger_softmax = LigerKernelSoftmax().to(device).to(dtype)
    liger_out = liger_softmax(x2)

    assert torch.allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (4, 16),
        (1, 1023),
        (3, 7, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 5e-2, 5e-2)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        else pytest.param(torch.bfloat16, 0, 0, marks=pytest.mark.skip(reason="bfloat16 not supported")),
    ],
)
def test_liger_softmax_functional(shape, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    ref_out = torch.nn.functional.softmax(x1, dim=-1)
    liger_out = liger_softmax(x2)

    assert torch.allclose(ref_out, liger_out, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output, retain_graph=True)
    liger_out.backward(grad_output, retain_graph=True)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
