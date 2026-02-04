import pytest
import torch

from test.utils import mhc_coeffs_ref
from test.utils import supports_bfloat16

from liger_kernel.ops.mhc import liger_mhc_coeffs
from liger_kernel.ops.mhc import liger_mhc_post_res
from liger_kernel.ops.mhc import liger_mhc_pre


def _mhc_tols(dtype: torch.dtype) -> tuple[float, float, float]:
    # Tolerances aligned with other kernel tests (fp32 tightest, then fp16, then bf16).
    if dtype == torch.float16:
        return 8e-3, 1.5e-2, 2e-2
    if dtype == torch.bfloat16:
        return 1.5e-2, 2.5e-2, 5e-2
    if dtype == torch.float32:
        return 5e-4, 1e-3, 2e-3
    raise AssertionError(f"Unsupported dtype: {dtype}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phi_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "x_dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16, marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported")
        ),
    ],
)
def test_mhc_coeffs_forward_backward(phi_dtype, x_dtype):
    torch.manual_seed(0)
    device = "cuda"

    B, T, HC, C = 2, 4, 4, 64
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=x_dtype, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=phi_dtype) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=8, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)
    pre_post_tol, res_tol, grad_tol = _mhc_tols(x_dtype)

    # Triton path
    h_pre, h_post, h_res = liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res, **cfg)

    loss = h_pre.square().mean() + h_post.square().mean() + h_res.square().mean()
    loss.backward()

    grads_triton = (
        x.grad.detach().float().clone(),
        phi.grad.detach().float().clone(),
        b.grad.detach().float().clone(),
        alpha_pre.grad.detach().float().clone(),
        alpha_post.grad.detach().float().clone(),
        alpha_res.grad.detach().float().clone(),
    )

    # Reference path
    x2 = x.detach().clone().requires_grad_(True)
    phi2 = phi.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    ap2 = alpha_pre.detach().clone().requires_grad_(True)
    apo2 = alpha_post.detach().clone().requires_grad_(True)
    ar2 = alpha_res.detach().clone().requires_grad_(True)

    rh_pre, rh_post, rh_res = mhc_coeffs_ref(x2, phi2, b2, ap2, apo2, ar2, **cfg)
    rloss = rh_pre.square().mean() + rh_post.square().mean() + rh_res.square().mean()
    rloss.backward()

    grads_ref = (
        x2.grad.detach().float(),
        phi2.grad.detach().float(),
        b2.grad.detach().float(),
        ap2.grad.detach().float(),
        apo2.grad.detach().float(),
        ar2.grad.detach().float(),
    )

    # Forward compare
    assert torch.allclose(h_pre.float(), rh_pre.float(), rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(h_post.float(), rh_post.float(), rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(h_res.float(), rh_res.float(), rtol=res_tol, atol=res_tol)

    # Backward compare (looser)
    for gt, gr in zip(grads_triton, grads_ref):
        assert torch.allclose(gt, gr, rtol=grad_tol, atol=grad_tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mhc_coeffs_allow_fp32():
    torch.manual_seed(0)
    device = "cuda"

    B, T, HC, C = 2, 4, 4, 64
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.float32, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=8, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)
    pre_post_tol, res_tol, grad_tol = _mhc_tols(torch.float32)

    h_pre, h_post, h_res = liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res, allow_fp32=True, **cfg)

    loss = h_pre.square().mean() + h_post.square().mean() + h_res.square().mean()
    loss.backward()

    grads_triton = (
        x.grad.detach().float().clone(),
        phi.grad.detach().float().clone(),
        b.grad.detach().float().clone(),
        alpha_pre.grad.detach().float().clone(),
        alpha_post.grad.detach().float().clone(),
        alpha_res.grad.detach().float().clone(),
    )

    x2 = x.detach().clone().requires_grad_(True)
    phi2 = phi.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    ap2 = alpha_pre.detach().clone().requires_grad_(True)
    apo2 = alpha_post.detach().clone().requires_grad_(True)
    ar2 = alpha_res.detach().clone().requires_grad_(True)

    rh_pre, rh_post, rh_res = mhc_coeffs_ref(x2, phi2, b2, ap2, apo2, ar2, **cfg)
    rloss = rh_pre.square().mean() + rh_post.square().mean() + rh_res.square().mean()
    rloss.backward()

    grads_ref = (
        x2.grad.detach().float(),
        phi2.grad.detach().float(),
        b2.grad.detach().float(),
        ap2.grad.detach().float(),
        apo2.grad.detach().float(),
        ar2.grad.detach().float(),
    )

    assert torch.allclose(h_pre.float(), rh_pre.float(), rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(h_post.float(), rh_post.float(), rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(h_res.float(), rh_res.float(), rtol=res_tol, atol=res_tol)

    for gt, gr in zip(grads_triton, grads_ref):
        assert torch.allclose(gt, gr, rtol=grad_tol, atol=grad_tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mhc_coeffs_disallow_fp32():
    device = "cuda"
    B, T, HC, C = 1, 2, 2, 8
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.float32)
    phi = torch.randn(K, M, device=device, dtype=torch.float32)
    b = torch.zeros(M, device=device, dtype=torch.float32)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32)

    with pytest.raises(AssertionError):
        _ = liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "use_pre,use_post,use_res",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_mhc_coeffs_backward_allows_unused_outputs(use_pre, use_post, use_res):
    torch.manual_seed(0)
    device = "cuda"

    B, T, HC, C = 2, 4, 2, 32
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=torch.float16, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=torch.float16) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=4, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)

    h_pre, h_post, h_res = liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res, **cfg)

    loss = torch.zeros((), device=device)
    if use_pre:
        loss = loss + h_pre.square().mean()
    if use_post:
        loss = loss + h_post.square().mean()
    if use_res:
        loss = loss + h_res.square().mean()
    loss.backward()

    for tensor in (x, phi, b, alpha_pre, alpha_post, alpha_res):
        assert tensor.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "x_dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16, marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported")
        ),
    ],
)
def test_mhc_pre_and_post_res_match_reference(x_dtype):
    torch.manual_seed(0)
    device = "cuda"
    B, T, HC, C = 2, 4, 4, 64

    x = torch.randn(B, T, HC, C, device=device, dtype=x_dtype, requires_grad=True)
    h_pre = torch.rand(B, T, HC, device=device, dtype=torch.float32, requires_grad=True)
    h_post = torch.rand(B, T, HC, device=device, dtype=torch.float32, requires_grad=True)
    h_res = torch.rand(B, T, HC, HC, device=device, dtype=torch.float32, requires_grad=True)
    pre_post_tol, res_tol, _ = _mhc_tols(x_dtype)

    # Forward
    x_in = liger_mhc_pre(x, h_pre)
    f_out = torch.randn(B, T, C, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_out = liger_mhc_post_res(x, f_out, h_post, h_res)

    # Reference
    x_in_ref = (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)
    x_out_ref = torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.float().unsqueeze(
        -2
    )

    assert torch.allclose(x_in.float(), x_in_ref, rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(x_out.float(), x_out_ref, rtol=res_tol, atol=res_tol)
