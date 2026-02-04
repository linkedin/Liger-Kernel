import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import mhc_coeffs_ref
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.functional import liger_mhc_coeffs
from liger_kernel.transformers.functional import liger_mhc_post_res
from liger_kernel.transformers.functional import liger_mhc_pre
from liger_kernel.transformers.mhc import LigerMHC


def _mhc_tols(dtype: torch.dtype) -> tuple[float, float, float]:
    # Tolerances aligned with other kernel tests (fp16 < bf16).
    if dtype == torch.float16:
        return 8e-3, 1.5e-2, 2e-2
    if dtype == torch.bfloat16:
        return 1.5e-2, 2.5e-2, 5e-2
    raise AssertionError(f"Unsupported dtype: {dtype}")


device = infer_device()


@pytest.mark.skipif(device != "cuda", reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16, marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported")
        ),
    ],
)
def test_liger_mhc_functional(dtype):
    set_seed(0)
    B, T, HC, C = 2, 4, 2, 32
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=4, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)
    pre_post_tol, res_tol, grad_tol = _mhc_tols(dtype)

    h_pre, h_post, h_res = liger_mhc_coeffs(x, phi, b, alpha_pre, alpha_post, alpha_res, **cfg)
    rh_pre, rh_post, rh_res = mhc_coeffs_ref(x, phi, b, alpha_pre, alpha_post, alpha_res, **cfg)

    assert_verbose_allclose(h_pre.float(), rh_pre.float(), rtol=pre_post_tol, atol=pre_post_tol, extra_info="[h_pre]")
    assert_verbose_allclose(
        h_post.float(), rh_post.float(), rtol=pre_post_tol, atol=pre_post_tol, extra_info="[h_post]"
    )
    assert_verbose_allclose(h_res.float(), rh_res.float(), rtol=res_tol, atol=res_tol, extra_info="[h_res]")

    loss = h_pre.square().mean() + h_post.square().mean() + h_res.square().mean()
    loss.backward()

    x2 = x.detach().clone().requires_grad_(True)
    phi2 = phi.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    ap2 = alpha_pre.detach().clone().requires_grad_(True)
    apo2 = alpha_post.detach().clone().requires_grad_(True)
    ar2 = alpha_res.detach().clone().requires_grad_(True)
    rh_pre2, rh_post2, rh_res2 = mhc_coeffs_ref(x2, phi2, b2, ap2, apo2, ar2, **cfg)
    rloss = rh_pre2.square().mean() + rh_post2.square().mean() + rh_res2.square().mean()
    rloss.backward()

    assert_verbose_allclose(x.grad.float(), x2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[x.grad]")
    assert_verbose_allclose(phi.grad.float(), phi2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[phi.grad]")
    assert_verbose_allclose(b.grad.float(), b2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[b.grad]")
    assert_verbose_allclose(
        alpha_pre.grad.float(), ap2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[alpha_pre]"
    )
    assert_verbose_allclose(
        alpha_post.grad.float(), apo2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[alpha_post]"
    )
    assert_verbose_allclose(
        alpha_res.grad.float(), ar2.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[alpha_res]"
    )

    x3 = x.detach().clone().requires_grad_(True)
    h_pre3 = h_pre.detach().clone().requires_grad_(True)
    h_post3 = h_post.detach().clone().requires_grad_(True)
    h_res3 = h_res.detach().clone().requires_grad_(True)
    f_out = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)

    x_in = liger_mhc_pre(x3, h_pre3)
    x_out = liger_mhc_post_res(x3, f_out, h_post3, h_res3)

    x_in_ref = (x3.float() * h_pre3.unsqueeze(-1)).sum(dim=-2)
    x_out_ref = torch.einsum("...oi,...ic->...oc", h_res3, x3.float()) + h_post3.unsqueeze(
        -1
    ) * f_out.float().unsqueeze(-2)

    assert_verbose_allclose(x_in.float(), x_in_ref, rtol=pre_post_tol, atol=pre_post_tol, extra_info="[x_in]")
    assert_verbose_allclose(x_out.float(), x_out_ref, rtol=res_tol, atol=res_tol, extra_info="[x_out]")


@pytest.mark.skipif(device != "cuda", reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16, marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported")
        ),
    ],
)
def test_liger_mhc_module(dtype):
    set_seed(0)
    B, T, HC, C = 2, 4, 2, 32

    layer = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    model = LigerMHC(
        layer,
        hc=HC,
        c=C,
        tmax=4,
        rms_eps=1e-6,
        pre_eps=1e-4,
        sinkhorn_eps=1e-6,
        post_mult=2.0,
        phi_dtype=dtype,
    ).to(device)

    x_fast = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    out_fast = model(x_fast)

    x_ref = x_fast.detach().clone().requires_grad_(True)
    phi_ref = model.phi.detach().clone().requires_grad_(True)
    b_ref = model.b.detach().clone().requires_grad_(True)
    ap_ref = model.alpha_pre.detach().clone().requires_grad_(True)
    apo_ref = model.alpha_post.detach().clone().requires_grad_(True)
    ar_ref = model.alpha_res.detach().clone().requires_grad_(True)

    layer_ref = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    layer_ref.weight.data.copy_(model.layer.weight.data)

    h_pre, h_post, h_res = mhc_coeffs_ref(
        x_ref,
        phi_ref,
        b_ref,
        ap_ref,
        apo_ref,
        ar_ref,
        tmax=4,
        rms_eps=1e-6,
        pre_eps=1e-4,
        sinkhorn_eps=1e-6,
        post_mult=2.0,
    )
    x_in_ref = (x_ref.float() * h_pre.unsqueeze(-1)).sum(dim=-2).to(dtype)
    f_out_ref = layer_ref(x_in_ref)
    out_ref = torch.einsum("...oi,...ic->...oc", h_res, x_ref.float()) + h_post.unsqueeze(
        -1
    ) * f_out_ref.float().unsqueeze(-2)

    pre_post_tol, res_tol, grad_tol = _mhc_tols(dtype)
    assert_verbose_allclose(out_fast.float(), out_ref.float(), rtol=res_tol, atol=res_tol, extra_info="[output]")

    grad = torch.randn_like(out_fast, dtype=torch.float32)
    out_fast.backward(grad.to(out_fast.dtype))
    out_ref.backward(grad)

    assert_verbose_allclose(
        x_fast.grad.float(), x_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[x.grad]"
    )
    assert_verbose_allclose(
        model.phi.grad.float(), phi_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[phi.grad]"
    )
    assert_verbose_allclose(
        model.b.grad.float(), b_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[b.grad]"
    )
    assert_verbose_allclose(
        model.alpha_pre.grad.float(), ap_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[alpha_pre.grad]"
    )
    assert_verbose_allclose(
        model.alpha_post.grad.float(),
        apo_ref.grad.float(),
        rtol=grad_tol,
        atol=grad_tol,
        extra_info="[alpha_post.grad]",
    )
    assert_verbose_allclose(
        model.alpha_res.grad.float(), ar_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[alpha_res.grad]"
    )
    assert_verbose_allclose(
        model.layer.weight.grad.float(),
        layer_ref.weight.grad.float(),
        rtol=grad_tol,
        atol=grad_tol,
        extra_info="[layer.weight.grad]",
    )
