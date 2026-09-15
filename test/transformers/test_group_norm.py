import pytest
import torch

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()


def _test_liger_group_norm(shape, num_channels, num_groups, dtype, atol, rtol):
    torch.manual_seed(0)

    _tensor = torch.randn(*shape, dtype=dtype, device=device)

    liger_x = _tensor.clone().detach().requires_grad_(True)
    torch_x = _tensor.clone().detach().requires_grad_(True)

    liger_ln = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).to(device)
    torch_ln = torch.nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, eps=1e-6).to(dtype).to(device)

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(
        liger_x,
    )
    torch_output = torch_ln(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)
    grad_output = torch.randn_like(torch_x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)
    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol), "Bias grads different"
    assert torch.allclose(liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol), "Weight grads different"


@pytest.mark.parametrize(
    "batch_size, num_channels, num_groups, hidden_size",
    [
        (1, 1, 1, 3),  # minimal
        (1, 32, 32, 4),  # group == channel
        (16, 32, 1, 4096),  # single group
        (2, 63, 21, 2163),  # non-aligned hidden
        (16, 48, 12, 8192),  # large hidden
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_liger_group_norm(batch_size, num_channels, num_groups, hidden_size, dtype, atol, rtol):
    _test_liger_group_norm((batch_size, num_channels, hidden_size), num_channels, num_groups, dtype, atol, rtol)


@pytest.mark.parametrize(
    "shape, num_channels, num_groups",
    [
        ((2, 6, 4, 8), 6, 3),  # convolutional features
        ((4, 32, 16, 16), 32, 8),  # larger spatial plane
        ((2, 8, 2, 4, 4), 8, 4),  # volumetric features
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_liger_group_norm_spatial_dims(shape, num_channels, num_groups, dtype, atol, rtol):
    _test_liger_group_norm(shape, num_channels, num_groups, dtype, atol, rtol)
