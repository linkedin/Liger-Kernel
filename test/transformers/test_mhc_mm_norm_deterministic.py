"""CUDA regressions for the deterministic mHC matrix/RMS backward reduction."""

import pytest
import torch

import liger_kernel.ops.mhc as mhc_ops

from liger_kernel.ops.mhc import _mhc_mm_norm_bwd_num_groups
from liger_kernel.ops.mhc import mhc_mm_norm_bwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="mHC deterministic Triton backward requires NVIDIA CUDA",
)


@pytest.fixture
def deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _inputs(n, k, m, x_dtype, phi_dtype):
    generator = torch.Generator(device="cuda").manual_seed(20261124)
    x = torch.randn((n, k), device="cuda", dtype=x_dtype, generator=generator) * 0.2
    phi = torch.randn((k, m), device="cuda", dtype=phi_dtype, generator=generator) * 0.2
    grad_mix = torch.randn((n, m), device="cuda", dtype=torch.float32, generator=generator)
    invr = torch.rsqrt(x.float().square().mean(-1) + 1e-6)
    mix = (x.float() @ phi.float()) * invr[:, None]
    return x, phi, mix, invr, grad_mix


def _reference(x, phi, mix, invr, grad_mix):
    # The tensor-core path rounds grad_z to the common low dtype before dot.
    grad_z = grad_mix * invr[:, None]
    if x.dtype == phi.dtype and x.dtype in (torch.float16, torch.bfloat16):
        grad_z = grad_z.to(phi.dtype)
    grad_z = grad_z.float()
    g = (grad_mix * mix).sum(-1) / invr
    factor = -(g * invr.square() * invr) / x.shape[1]
    if x.dtype == phi.dtype and x.dtype in (torch.float16, torch.bfloat16):
        grad_x = (grad_z[:, :, None] * phi.float().T[None, :, :]).sum(1) + x.float() * factor[:, None]
        grad_phi = (x.float()[:, :, None] * grad_z[:, None, :]).sum(0)
    else:
        # CUDA FP32 matmul may itself use TF32; double products give an IEEE oracle.
        grad_x = (grad_z.double()[:, :, None] * phi.double().T[None, :, :]).sum(1).float()
        grad_x = grad_x + x.float() * factor[:, None]
        grad_phi = (x.double()[:, :, None] * grad_z.double()[:, None, :]).sum(0).float()
    return grad_x.to(x.dtype), grad_phi


def _uncast_reference_x(x, phi, mix, invr, grad_mix):
    grad_z = _grad_z(x, phi, invr, grad_mix)
    g = (grad_mix * mix).sum(-1) / invr
    factor = -(g * invr.square() * invr) / x.shape[1]
    dot = (grad_z.double()[:, :, None] * phi.double().T[None, :, :]).sum(1).float()
    return dot + x.float() * factor[:, None]


def _grad_z(x, phi, invr, grad_mix):
    grad_z = grad_mix * invr[:, None]
    if x.dtype == phi.dtype and x.dtype in (torch.float16, torch.bfloat16):
        grad_z = grad_z.to(phi.dtype)
    return grad_z.float()


def _output_rounding(dtype, actual, expected):
    if dtype == torch.float32:
        return torch.zeros_like(expected, dtype=torch.float64)
    info = torch.finfo(dtype)
    return info.eps * torch.maximum(actual.double().abs(), expected.double().abs()) + info.tiny


def _assert_reference_gradients(x, phi, mix, invr, grad_mix, actual_x, actual_phi, *, atol, rtol, phi_initial=0):
    expected_x, expected_phi = _reference(x, phi, mix, invr, grad_mix)
    if x.dtype == phi.dtype and x.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(actual_x.float(), expected_x.float(), atol=atol, rtol=rtol)
        expected_accumulated = (expected_phi + phi_initial).to(actual_phi.dtype)
        torch.testing.assert_close(actual_phi.float(), expected_accumulated.float(), atol=atol, rtol=rtol)
        return

    grad_z = _grad_z(x, phi, invr, grad_mix)
    # tl.dot on the CAST_FP32 path may round both operands to TF32 before
    # multiplying. This bounds that error plus FP32 accumulation, elementwise.
    u = 2**-10
    eps = 2**-24

    def dot_allowance(reduction, scale):
        gamma = reduction * eps / (1 - reduction * eps)
        return (2 * u + u * u + 2 * gamma * (1 + u) ** 2) * scale

    phi_scale = (x.double().abs()[:, :, None] * grad_z.double().abs()[:, None, :]).sum(0)
    x_scale = (grad_z.double().abs()[:, :, None] * phi.double().abs().T[None, :, :]).sum(1)
    phi_expected = expected_phi.double() + phi_initial
    phi_bound = (
        atol
        + rtol * phi_expected.abs()
        + dot_allowance(x.shape[0], phi_scale)
        + _output_rounding(actual_phi.dtype, actual_phi, phi_expected)
    )
    uncast_x = _uncast_reference_x(x, phi, mix, invr, grad_mix)
    x_bound = (
        atol
        + rtol * uncast_x.double().abs()
        + dot_allowance(phi.shape[1], x_scale)
        + _output_rounding(actual_x.dtype, actual_x, uncast_x)
    )
    assert bool(torch.all((actual_phi.double() - phi_expected).abs() <= phi_bound))
    assert bool(torch.all((actual_x.double() - uncast_x.double()).abs() <= x_bound))


def _baseline_phi_bound(x, phi, invr, grad_mix, actual, baseline):
    # Both kernels use the same dot precision; only reduction order differs.
    eps = 2**-24
    gamma = x.shape[0] * eps / (1 - x.shape[0] * eps)
    scale = (x.double().abs()[:, :, None] * _grad_z(x, phi, invr, grad_mix).double().abs()[:, None, :]).sum(0)
    info = torch.finfo(actual.dtype)
    one_ulp = info.eps * torch.maximum(actual.double().abs(), baseline.double().abs()) + info.tiny
    return 4 * gamma * scale + one_ulp


def _assert_baseline_gradients(x, phi, invr, grad_mix, actual_x, actual_phi, baseline_x, baseline_phi):
    phi_bound = _baseline_phi_bound(x, phi, invr, grad_mix, actual_phi, baseline_phi)
    eps = 2**-24
    gamma = phi.shape[1] * eps / (1 - phi.shape[1] * eps)
    x_scale = (_grad_z(x, phi, invr, grad_mix).double().abs()[:, :, None] * phi.double().abs().T[None, :, :]).sum(1)
    info = torch.finfo(actual_x.dtype)
    one_ulp = info.eps * torch.maximum(actual_x.double().abs(), baseline_x.double().abs()) + info.tiny
    x_bound = 4 * gamma * x_scale + one_ulp
    assert bool(torch.all((actual_phi.double() - baseline_phi.double()).abs() <= phi_bound))
    assert bool(torch.all((actual_x.double() - baseline_x.double()).abs() <= x_bound))


@pytest.mark.parametrize(
    "n,k,m,x_dtype,phi_dtype,atol,rtol",
    [
        (1, 32, 16, torch.float32, torch.float32, 2e-3, 2e-3),
        (17, 37, 19, torch.float32, torch.float32, 3e-3, 3e-3),
        (65, 37, 19, torch.float16, torch.float16, 2e-2, 2e-2),
        (65, 37, 19, torch.bfloat16, torch.bfloat16, 6e-2, 6e-2),
        (65, 37, 19, torch.float16, torch.float32, 4e-3, 4e-3),
    ],
)
def test_backward_reference_and_repeatability(n, k, m, x_dtype, phi_dtype, atol, rtol, deterministic_algorithms):
    if x_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this CUDA device")

    x, phi, mix, invr, grad_mix = _inputs(n, k, m, x_dtype, phi_dtype)
    kwargs = dict(block_n=16, block_k=32, block_m=16)

    torch.use_deterministic_algorithms(False)
    default_x, default_phi = mhc_mm_norm_bwd(x, phi, mix, invr, grad_mix, **kwargs)
    torch.use_deterministic_algorithms(True)
    first_x, first_phi = mhc_mm_norm_bwd(x, phi, mix, invr, grad_mix, **kwargs)
    second_x, second_phi = mhc_mm_norm_bwd(x, phi, mix, invr, grad_mix, **kwargs)

    assert first_x.dtype == x.dtype
    assert first_phi.dtype == phi.dtype
    assert torch.equal(first_x, second_x)
    assert torch.equal(first_phi, second_phi)
    _assert_reference_gradients(x, phi, mix, invr, grad_mix, first_x, first_phi, atol=atol, rtol=rtol)
    _assert_baseline_gradients(x, phi, invr, grad_mix, first_x, first_phi, default_x, default_phi)

    # The old/new comparator must reject a real missing-token contribution.
    if n == 17 and x_dtype == torch.float32:
        dropped_last = default_phi.float() - x[-1, :, None].float() * _grad_z(x, phi, invr, grad_mix)[-1]
        bound = _baseline_phi_bound(x, phi, invr, grad_mix, dropped_last, default_phi)
        assert not bool(torch.all((dropped_last.double() - default_phi.double()).abs() <= bound))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nonzero_strided_outputs_accumulate_once(dtype, deterministic_algorithms):
    x, phi, mix, invr, grad_mix = _inputs(49, 37, 19, dtype, dtype)
    torch.use_deterministic_algorithms(False)
    baseline_phi = torch.full((37, 19), 0.25, device="cuda", dtype=torch.float32)
    default_x, _ = mhc_mm_norm_bwd(
        x, phi, mix, invr, grad_mix, out_grad_phi=baseline_phi, block_n=16, block_k=32, block_m=16
    )
    grad_x_storage = torch.empty((49, 74), device="cuda", dtype=dtype)
    grad_phi_storage = torch.full((37, 38), 0.25, device="cuda", dtype=dtype)
    out_grad_x = grad_x_storage[:, ::2]
    out_grad_phi = grad_phi_storage[:, ::2]
    initial_phi = out_grad_phi.clone()

    torch.use_deterministic_algorithms(True)
    result_x, result_phi = mhc_mm_norm_bwd(
        x,
        phi,
        mix,
        invr,
        grad_mix,
        out_grad_x=out_grad_x,
        out_grad_phi=out_grad_phi,
        block_n=16,
        block_k=32,
        block_m=16,
    )

    assert result_x is out_grad_x
    assert result_phi is out_grad_phi
    assert result_x.stride() == (74, 2)
    assert result_phi.stride() == (38, 2)
    _assert_reference_gradients(
        x, phi, mix, invr, grad_mix, result_x, result_phi, atol=2e-2, rtol=2e-2, phi_initial=initial_phi
    )
    _assert_baseline_gradients(x, phi, invr, grad_mix, result_x, result_phi, default_x, baseline_phi)
    assert torch.equal(grad_phi_storage[:, 1::2], torch.full_like(grad_phi_storage[:, 1::2], 0.25))


def test_fp32_output_buffer_with_low_precision_phi(deterministic_algorithms):
    x, phi, mix, invr, grad_mix = _inputs(49, 37, 19, torch.float16, torch.float16)
    _, expected_phi = _reference(x, phi, mix, invr, grad_mix)
    out_grad_phi = torch.full((37, 19), 0.125, device="cuda", dtype=torch.float32)
    initial_phi = out_grad_phi.clone()
    baseline_buffer = initial_phi.clone()
    torch.use_deterministic_algorithms(False)
    baseline_x, _ = mhc_mm_norm_bwd(
        x, phi, mix, invr, grad_mix, out_grad_phi=baseline_buffer, block_n=16, block_k=32, block_m=16
    )

    torch.use_deterministic_algorithms(True)
    result_x, returned_phi = mhc_mm_norm_bwd(
        x, phi, mix, invr, grad_mix, out_grad_phi=out_grad_phi, block_n=16, block_k=32, block_m=16
    )

    assert returned_phi is not out_grad_phi
    assert returned_phi.dtype == phi.dtype
    assert out_grad_phi.dtype == torch.float32
    torch.testing.assert_close(out_grad_phi, initial_phi + expected_phi, atol=2e-2, rtol=2e-2)
    assert torch.equal(returned_phi, out_grad_phi.to(phi.dtype))
    _assert_baseline_gradients(x, phi, invr, grad_mix, result_x, out_grad_phi, baseline_x, baseline_buffer)


def test_single_group_direct_path_spans_multiple_token_blocks(monkeypatch, deterministic_algorithms):
    x, phi, mix, invr, grad_mix = _inputs(49, 37, 19, torch.float32, torch.float32)
    out_grad_phi = torch.full_like(phi, 0.125)
    torch.use_deterministic_algorithms(False)
    default_x, default_phi = mhc_mm_norm_bwd(x, phi, mix, invr, grad_mix, block_n=16, block_k=32, block_m=16)
    monkeypatch.setattr(mhc_ops, "_MHC_GRAD_PHI_WORKSPACE_BYTES", 0)
    assert _mhc_mm_norm_bwd_num_groups(49, 37, 19, 16) == 1

    torch.use_deterministic_algorithms(True)
    result_x, result_phi = mhc_mm_norm_bwd(
        x, phi, mix, invr, grad_mix, out_grad_phi=out_grad_phi, block_n=16, block_k=32, block_m=16
    )

    assert result_phi is out_grad_phi
    _assert_reference_gradients(
        x, phi, mix, invr, grad_mix, result_x, result_phi, atol=3e-3, rtol=3e-3, phi_initial=0.125
    )
    _assert_baseline_gradients(x, phi, invr, grad_mix, result_x, result_phi, default_x, default_phi + 0.125)


def test_x_can_alias_grad_x_output(deterministic_algorithms):
    x, phi, mix, invr, grad_mix = _inputs(49, 37, 19, torch.float32, torch.float32)
    original_x = x.clone()
    torch.use_deterministic_algorithms(False)
    default_x, default_phi = mhc_mm_norm_bwd(x, phi, mix, invr, grad_mix, block_n=16, block_k=32, block_m=16)

    torch.use_deterministic_algorithms(True)
    result_x, result_phi = mhc_mm_norm_bwd(
        x, phi, mix, invr, grad_mix, out_grad_x=x, block_n=16, block_k=32, block_m=16
    )

    assert result_x is x
    _assert_reference_gradients(original_x, phi, mix, invr, grad_mix, result_x, result_phi, atol=3e-3, rtol=3e-3)
    _assert_baseline_gradients(original_x, phi, invr, grad_mix, result_x, result_phi, default_x, default_phi)


def test_public_coeffs_one_optimizer_update(deterministic_algorithms):
    from test.transformers.test_mhc import mhc_coeffs_ref

    from liger_kernel.transformers.functional import liger_mhc_coeffs

    generator = torch.Generator(device="cuda").manual_seed(20261124)
    hc, c = 2, 32
    m = hc * hc + 2 * hc
    original = [
        torch.randn((2, 4, hc, c), device="cuda", generator=generator),
        torch.randn((hc * c, m), device="cuda", generator=generator) * 0.02,
        torch.zeros(m, device="cuda"),
        torch.tensor(1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
    ]
    ref_params = [p.detach().clone().requires_grad_(True) for p in original]
    liger_params = [p.detach().clone().requires_grad_(True) for p in original]
    upstream = [
        torch.randn((2, 4, hc), device="cuda", generator=generator),
        torch.randn((2, 4, hc), device="cuda", generator=generator),
        torch.randn((2, 4, hc, hc), device="cuda", generator=generator),
    ]
    cfg = dict(tmax=8, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)

    ref_outputs = mhc_coeffs_ref(*ref_params, **cfg)
    ref_loss = sum((output * weight).sum() for output, weight in zip(ref_outputs, upstream))
    ref_loss.backward()
    expected_grads = [p.grad.detach().clone() for p in ref_params]
    ref_optimizer = torch.optim.SGD(ref_params, lr=0.1)
    ref_optimizer.step()

    torch.use_deterministic_algorithms(True)
    liger_outputs = liger_mhc_coeffs(*liger_params, allow_fp32=True, **cfg)
    liger_loss = sum((output * weight).sum() for output, weight in zip(liger_outputs, upstream))
    liger_loss.backward()
    liger_optimizer = torch.optim.SGD(liger_params, lr=0.1)
    liger_optimizer.step()

    for actual, expected, expected_grad in zip(liger_params, ref_params, expected_grads):
        torch.testing.assert_close(actual.grad, expected_grad, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "n,k,m,block_n",
    [
        (0, 64, 16, 32),
        (1, 64, 16, 32),
        (33, 64, 16, 32),
        (1_000_000, 64, 16, 32),
        (1_000_000, 4096, 1024, 32),
        (1_000_000, 4096, 2048, 32),
    ],
)
def test_group_planning_bounds_scratch(n, k, m, block_n):
    groups = _mhc_mm_norm_bwd_num_groups(n, k, m, block_n)
    assert 1 <= groups <= 32
    assert groups <= max(1, (n + block_n - 1) // block_n)
    if groups > 1:
        assert groups * k * m * 4 <= 16 * 1024 * 1024


def test_group_planning_uses_parallel_groups_when_possible():
    assert _mhc_mm_norm_bwd_num_groups(33, 64, 16, 32) > 1
    assert _mhc_mm_norm_bwd_num_groups(1_000_000, 64, 16, 32) == 32
    assert _mhc_mm_norm_bwd_num_groups(1_000_000, 4096, 2048, 32) == 1
