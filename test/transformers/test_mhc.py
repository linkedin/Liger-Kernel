import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import infer_device
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.functional import liger_mhc_coeffs
from liger_kernel.transformers.functional import liger_mhc_post_res
from liger_kernel.transformers.functional import liger_mhc_pre
from liger_kernel.transformers.mhc import LigerMHC

device = infer_device()

MHC_SHAPES = [
    (2, 4, 2, 32),
    (1, 8, 4, 64),
]

MHC_DTYPE_TOLS = [
    (torch.float16, 8e-3, 1.5e-2, 2e-2),
    pytest.param(
        torch.bfloat16,
        1.5e-2,
        2.5e-2,
        5e-2,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
    ),
]


class TorchMHCCoeffs(nn.Module):
    def __init__(
        self,
        *,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
    ):
        super().__init__()
        self.tmax = int(tmax)
        self.rms_eps = float(rms_eps)
        self.pre_eps = float(pre_eps)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.post_mult = float(post_mult)

    def forward(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        b: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_coeffs_ref(
            x,
            phi,
            b,
            alpha_pre,
            alpha_post,
            alpha_res,
            tmax=self.tmax,
            rms_eps=self.rms_eps,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            post_mult=self.post_mult,
        )


def mhc_sinkhorn_ref(logits: torch.Tensor, *, tmax: int, eps: float) -> torch.Tensor:
    """
    logits: [N, HC, HC]
    """
    mat = torch.softmax(logits, dim=-1) + eps
    mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
    for _ in range(tmax - 1):
        mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
        mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
    return mat


def mhc_coeffs_ref(
    x: torch.Tensor,
    phi: torch.Tensor,
    b: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    *,
    tmax: int,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
):
    x_flat = x.contiguous().view(-1, x.shape[-2], x.shape[-1]).float()
    n, hc, c = x_flat.shape
    k = hc * c
    x_mat = x_flat.view(n, k)
    invr = torch.rsqrt(x_mat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    mix = (x_mat @ phi.float()) * invr

    pre_logits = mix[:, :hc] * alpha_pre + b[:hc]
    post_logits = mix[:, hc : 2 * hc] * alpha_post + b[hc : 2 * hc]
    res_logits = mix[:, 2 * hc :].view(n, hc, hc) * alpha_res + b[2 * hc :].view(hc, hc)

    h_pre = torch.sigmoid(pre_logits) + pre_eps
    h_post = torch.sigmoid(post_logits) * post_mult
    h_res = mhc_sinkhorn_ref(res_logits, tmax=tmax, eps=sinkhorn_eps)

    outer = x.shape[:-2]
    return (
        h_pre.view(*outer, hc),
        h_post.view(*outer, hc),
        h_res.view(*outer, hc, hc),
    )


@pytest.mark.parametrize("B, T, HC, C", MHC_SHAPES)
@pytest.mark.parametrize("phi_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dtype, pre_post_tol, res_tol, grad_tol", MHC_DTYPE_TOLS)
def test_mhc_coeffs_forward_backward(B, T, HC, C, phi_dtype, dtype, pre_post_tol, res_tol, grad_tol):
    set_seed(0)
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=phi_dtype) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=8, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)

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


@pytest.mark.parametrize("B, T, HC, C", [MHC_SHAPES[0]])
@pytest.mark.parametrize("dtype, pre_post_tol, res_tol, grad_tol", [(torch.float32, 5e-4, 1e-3, 2e-3)])
def test_mhc_coeffs_allow_fp32(B, T, HC, C, dtype, pre_post_tol, res_tol, grad_tol):
    set_seed(0)
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=8, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)

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


def test_mhc_coeffs_disallow_fp32():
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


@pytest.mark.parametrize("B, T, HC, C", MHC_SHAPES)
@pytest.mark.parametrize(
    "use_pre,use_post,use_res",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_mhc_coeffs_backward_allows_unused_outputs(B, T, HC, C, use_pre, use_post, use_res):
    set_seed(0)
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


@pytest.mark.parametrize("B, T, HC, C", MHC_SHAPES)
@pytest.mark.parametrize("dtype, pre_post_tol, res_tol, _grad_tol", MHC_DTYPE_TOLS)
def test_mhc_pre_and_post_res_match_reference(B, T, HC, C, dtype, pre_post_tol, res_tol, _grad_tol):
    set_seed(0)

    x = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    h_pre = torch.rand(B, T, HC, device=device, dtype=torch.float32, requires_grad=True)
    h_post = torch.rand(B, T, HC, device=device, dtype=torch.float32, requires_grad=True)
    h_res = torch.rand(B, T, HC, HC, device=device, dtype=torch.float32, requires_grad=True)

    x_in = liger_mhc_pre(x, h_pre)
    f_out = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)
    x_out = liger_mhc_post_res(x, f_out, h_post, h_res)

    x_in_ref = (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)
    x_out_ref = torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.float().unsqueeze(
        -2
    )

    assert torch.allclose(x_in.float(), x_in_ref, rtol=pre_post_tol, atol=pre_post_tol)
    assert torch.allclose(x_out.float(), x_out_ref, rtol=res_tol, atol=res_tol)


@pytest.mark.parametrize("B, T, HC, C", MHC_SHAPES)
@pytest.mark.parametrize("dtype, pre_post_tol, res_tol, grad_tol", MHC_DTYPE_TOLS)
def test_liger_mhc_functional(B, T, HC, C, dtype, pre_post_tol, res_tol, grad_tol):
    set_seed(0)
    K = HC * C
    M = HC * HC + 2 * HC

    x = torch.randn(B, T, HC, C, device=device, dtype=dtype, requires_grad=True)
    phi = (torch.randn(K, M, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    b = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    alpha_pre = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_post = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
    alpha_res = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

    cfg = dict(tmax=4, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0)

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


@pytest.mark.parametrize("B, T, HC, C", MHC_SHAPES)
@pytest.mark.parametrize("dtype, _pre_post_tol, res_tol, grad_tol", MHC_DTYPE_TOLS)
def test_liger_mhc_module(B, T, HC, C, dtype, _pre_post_tol, res_tol, grad_tol):
    set_seed(0)

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

    assert_verbose_allclose(out_fast.float(), out_ref.float(), rtol=res_tol, atol=res_tol, extra_info="[output]")

    grad = torch.randn_like(out_fast, dtype=torch.float32)
    out_fast.backward(grad.to(out_fast.dtype))
    out_ref.backward(grad)

    assert_verbose_allclose(
        x_fast.grad.float(), x_ref.grad.float(), rtol=grad_tol, atol=grad_tol, extra_info="[x.grad]"
    )
    phi_grad_tol = grad_tol * 4 if dtype == torch.bfloat16 else grad_tol
    assert_verbose_allclose(
        model.phi.grad.float(),
        phi_ref.grad.float(),
        rtol=phi_grad_tol,
        atol=phi_grad_tol,
        extra_info="[phi.grad]",
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


class MiniMHCLM(nn.Module):
    """Tiny language model using mHC for end-to-end correctness testing."""

    def __init__(self, *, vocab_size, hc, c, tmax, rms_eps, pre_eps, sinkhorn_eps, post_mult, use_fast, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.hc = hc
        self.c = c
        self.tmax = tmax
        self.rms_eps = rms_eps
        self.pre_eps = pre_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult = post_mult
        self.use_fast = use_fast
        self.act_dtype = torch.bfloat16

        self.embed = nn.Embedding(vocab_size, hc * c, device=device)
        self.inner = nn.Linear(c, c, bias=False, device=device)
        self.head = nn.Linear(hc * c, vocab_size, bias=False, device=device)

        m = hc * hc + 2 * hc
        k = hc * c
        self.phi = nn.Parameter(torch.randn(k, m, device=device, dtype=self.act_dtype) * 0.02)
        self.b = nn.Parameter(torch.zeros(m, device=device, dtype=torch.float32))
        self.alpha_pre = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))

    def forward(self, input_ids):
        x = self.embed(input_ids).to(self.act_dtype)
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.hc, self.c)

        cfg = dict(
            tmax=self.tmax,
            rms_eps=self.rms_eps,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            post_mult=self.post_mult,
        )
        if self.use_fast:
            h_pre, h_post, h_res = liger_mhc_coeffs(
                x, self.phi, self.b, self.alpha_pre, self.alpha_post, self.alpha_res, **cfg
            )
            x_in = liger_mhc_pre(x, h_pre)
            f_out = self.inner(x_in.float())
            x_out = liger_mhc_post_res(x, f_out, h_post, h_res)
        else:
            h_pre, h_post, h_res = mhc_coeffs_ref(
                x, self.phi, self.b, self.alpha_pre, self.alpha_post, self.alpha_res, **cfg
            )
            x_in = (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)
            f_out = self.inner(x_in)
            x_out = torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)

        x_merge = x_out.float().view(bsz, seq_len, self.hc * self.c)
        return self.head(x_merge)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize(
    "vocab_size, hc, c, tmax",
    [
        (32, 2, 16, 4),
        (64, 4, 32, 8),
    ],
)
def test_mhc_mini_lm_output_match(vocab_size, hc, c, tmax):
    set_seed(0)

    model_cfg = dict(
        vocab_size=vocab_size, hc=hc, c=c, tmax=tmax, rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-6, post_mult=2.0
    )

    model_fast = MiniMHCLM(**model_cfg, use_fast=True, device=device)
    model_ref = MiniMHCLM(**model_cfg, use_fast=False, device=device)
    model_ref.load_state_dict(model_fast.state_dict())

    input_ids = torch.randint(0, vocab_size, (2, 8), device=device)
    labels = torch.randint(0, vocab_size, (2, 8), device=device)

    logits_fast = model_fast(input_ids)
    logits_ref = model_ref(input_ids)

    assert_verbose_allclose(logits_fast.float(), logits_ref.float(), atol=5e-3, rtol=2e-2, extra_info="[logits]")

    loss_fast = F.cross_entropy(logits_fast.view(-1, vocab_size), labels.view(-1))
    loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), labels.view(-1))

    loss_fast.backward()
    loss_ref.backward()

    for name in ["phi", "b", "alpha_pre", "alpha_post", "alpha_res"]:
        g_fast = getattr(model_fast, name).grad.float()
        g_ref = getattr(model_ref, name).grad.float()
        assert_verbose_allclose(g_fast, g_ref, atol=5e-2, rtol=5e-2, extra_info=f"[{name}.grad]")

    assert_verbose_allclose(
        model_fast.inner.weight.grad.float(),
        model_ref.inner.weight.grad.float(),
        atol=5e-2,
        rtol=5e-2,
        extra_info="[inner.weight.grad]",
    )
    assert_verbose_allclose(
        model_fast.head.weight.grad.float(),
        model_ref.head.weight.grad.float(),
        atol=5e-2,
        rtol=5e-2,
        extra_info="[head.weight.grad]",
    )
