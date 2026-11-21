import pytest
import torch

from test.utils import supports_bfloat16

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


@pytest.mark.parametrize(
    "shape, num_groups, offset, dtype, atol, rtol",
    [
        # fp16: sum(x * x) over a 65536-element group exceeds the fp16 max (65504)
        ((1, 512, 64, 64), 32, 0.0, torch.float16, 2e-3, 2e-3),
        ((1, 512, 64, 64), 32, 1.0, torch.float16, 2e-3, 2e-3),
        # bf16: E[x^2] - E[x]^2 cancels when the mean is large relative to the spread
        pytest.param(
            (1, 64, 8),
            8,
            16.0,
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_group_norm_low_precision_statistics(shape, num_groups, offset, dtype, atol, rtol):
    # the group mean and variance must be accumulated in fp32 even for fp16/bf16 inputs
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=device) + offset).to(dtype)
    num_channels = shape[1]

    liger_gn = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).to(device)
    with torch.no_grad():
        liger_output = liger_gn(x)
        torch_output = torch.nn.functional.group_norm(
            x.float(), num_groups, liger_gn.weight.float(), liger_gn.bias.float(), eps=1e-6
        )

    assert torch.isfinite(liger_output).all()
    assert torch.allclose(liger_output.float(), torch_output, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_liger_group_norm_int32_batch_offset_overflow():
    # The kernels move to sample `batch_idx` with `batch_idx * stride(0)`. With an
    # int32 program id this wraps once the sample offset reaches 2**31 elements, even
    # though each sample is far smaller than that.
    if torch.cuda.mem_get_info()[0] < 48 * 2**30:
        pytest.skip("needs about 48 GiB of free GPU memory to cross the 2**31 offset")
    batch_size, num_channels, num_groups, spatial = 17, 128, 32, (1024, 1024)
    assert (batch_size - 1) * num_channels * spatial[0] * spatial[1] >= 2**31
    torch.manual_seed(0)
    x = torch.randn(batch_size, num_channels, *spatial, device="cuda")
    liger_ln = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to("cuda")
    x.requires_grad_(True)
    y = liger_ln(x)
    dy = torch.randn_like(y)
    y.backward(dy)
    torch.cuda.synchronize()

    # GroupNorm is independent per sample: compare the first sample and the last one
    # (past the 2**31 offset) with a reference on that sample alone.
    weight = liger_ln.weight.detach()
    bias = liger_ln.bias.detach()
    for s in (0, batch_size - 1):
        xs = x.detach()[s : s + 1].clone().requires_grad_(True)
        ys = torch.nn.functional.group_norm(xs, num_groups, weight, bias, 1e-6)
        ys.backward(dy[s : s + 1])
        assert (y[s] - ys[0]).abs().max().item() < 1e-3, f"output of sample {s}"
        assert (x.grad[s] - xs.grad[0]).abs().max().item() < 1e-3, f"input grad of sample {s}"
