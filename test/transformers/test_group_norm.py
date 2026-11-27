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


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 2e-3, 2e-3),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_group_norm_constant_groups_no_nan(dtype, atol, rtol):
    """Regression test for #1646: constant groups must not produce negative variance / NaN."""
    torch.manual_seed(0)
    x1 = torch.full((2, 4, 25, 10), 5.3, device=device, dtype=dtype)
    liger_gn1 = LigerGroupNorm(4, 2, eps=1e-6, bias=True).to(dtype).to(device)
    torch_gn1 = torch.nn.GroupNorm(2, 4, eps=1e-6).to(dtype).to(device)
    with torch.no_grad():
        torch_gn1.weight.copy_(liger_gn1.weight)
        torch_gn1.bias.copy_(liger_gn1.bias)
        y1 = liger_gn1(x1)
        ref1 = torch_gn1(x1)
    assert torch.isfinite(y1).all()
    assert torch.allclose(y1, ref1, atol=atol, rtol=rtol)

    x2_base = (torch.randn(8, 16, 1, 1, device=device, dtype=dtype) * 10).expand(8, 16, 8, 8).contiguous()
    x2_lig = x2_base.clone().requires_grad_(True)
    x2_ref = x2_base.clone().float().requires_grad_(True)
    liger_gn2 = LigerGroupNorm(16, 16, eps=1e-6, bias=True).to(dtype).to(device)
    torch_gn2 = torch.nn.GroupNorm(16, 16, eps=1e-6).to(torch.float32).to(device)
    with torch.no_grad():
        torch_gn2.weight.copy_(liger_gn2.weight.float())
        torch_gn2.bias.copy_(liger_gn2.bias.float())

    y2 = liger_gn2(x2_lig)
    ref2 = torch_gn2(x2_ref)
    assert torch.isfinite(y2).all()
    assert torch.allclose(y2.float(), ref2, atol=atol, rtol=rtol)

    # Note: low-precision Mean/RSTD storage and backward reductions are tracked separately in #1606 / PR #1608.
    if dtype == torch.float32:
        dy = torch.randn_like(ref2)
        y2.backward(dy)
        ref2.backward(dy)
        assert torch.isfinite(x2_lig.grad).all()
        assert torch.allclose(x2_lig.grad, x2_ref.grad, atol=atol, rtol=rtol)


def test_liger_group_norm_large_mean_shift_stability():
    """Regression test for #1646: |mean| >> std across multi-block groups must stay accurate in fwd and bwd."""
    torch.manual_seed(0)
    # hidden_size per group = (16 // 4) * 64 * 64 = 16384 > MAX_FUSED_SIZE_FORWARD (8192),
    # exercising multi-block Welford merge under a large mean offset.
    base = torch.randn(2, 16, 64, 64, device=device, dtype=torch.float32) + 1000.0
    x_lig = base.clone().requires_grad_(True)
    x_ref = base.clone().requires_grad_(True)

    liger_gn = LigerGroupNorm(16, 4, eps=1e-6, bias=True).to(device)
    torch_gn = torch.nn.GroupNorm(4, 16, eps=1e-6).to(device)
    with torch.no_grad():
        torch_gn.weight.copy_(liger_gn.weight)
        torch_gn.bias.copy_(liger_gn.bias)

    y_lig = liger_gn(x_lig)
    y_ref = torch_gn(x_ref)
    dy = torch.randn_like(y_ref)
    y_lig.backward(dy)
    y_ref.backward(dy)

    assert torch.allclose(y_lig, y_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x_lig.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(liger_gn.weight.grad, torch_gn.weight.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(liger_gn.bias.grad, torch_gn.bias.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_group_norm_torch_compile_matches_eager(dtype):
    """Ensure tl.cast keeps scalar block counts torch.compile-clean (#1544)."""
    import gc

    torch.manual_seed(0)
    torch._dynamo.reset()
    try:
        base = torch.full((2, 8, 15, 17), 5.3, device=device, dtype=dtype)
        dy = torch.randn_like(base)

        eager_gn = LigerGroupNorm(8, 4, eps=1e-6, bias=True).to(dtype).to(device)
        compiled_gn = LigerGroupNorm(8, 4, eps=1e-6, bias=True).to(dtype).to(device)
        with torch.no_grad():
            compiled_gn.weight.copy_(eager_gn.weight)
            compiled_gn.bias.copy_(eager_gn.bias)

        x_eager = base.clone().requires_grad_(True)
        x_comp = base.clone().requires_grad_(True)

        y_eager = eager_gn(x_eager)
        y_comp = torch.compile(compiled_gn)(x_comp)
        y_eager.backward(dy)
        y_comp.backward(dy)

        assert torch.isfinite(y_comp).all()
        torch.testing.assert_close(y_comp, y_eager, atol=0, rtol=0)
        torch.testing.assert_close(x_comp.grad, x_eager.grad, atol=0, rtol=0)
    finally:
        torch._dynamo.reset()
        gc.collect()
