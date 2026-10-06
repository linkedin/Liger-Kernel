"""Canonical dispatcher tests for the CuTe DSL fused_add_rms_norm op.

Oracles are the Triton implementation and an independent PyTorch autograd
reference. Checked: both returned tensors (normed output and updated residual)
plus ``dX``/``dR``/``dW``, across dtypes and 2-D/3-D input shapes.

The 3-D case is the regression test for a bug where the CuTe DSL backward
crashed with ``ValueError: Mismatched Tensor ... expected ndim=2`` because the
saved residual sum kept the original 3-D shape while the shared backward
kernel is compiled for 2-D rows (fixed by flattening around the backward).

Note on ``in_place=True``: the backward reuses the upstream dY storage for dX
(documented Liger convention), i.e. the caller's grad tensor is consumed by
design. Tests therefore pass a fresh clone of the shared grads to every
backward call.

All shapes below must run natively: strict dispatch rejects fallback. The suite
requires CUDA, ``nvidia-cutlass-dsl``, and the advertised sm_90+ capability;
passing collection alone does not establish correctness on Hopper or Blackwell.
"""

import pytest
import torch

import liger_kernel.functional  # noqa: F401

from test.cutedsl.test_gap_ops import _assert_nonzero_close

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl fused_add_rms_norm requires CUDA")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cutedsl_available() -> bool:
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


skip_no_cutedsl = pytest.mark.skipif(
    not _cutedsl_available(), reason="cutedsl backend unavailable (cutlass.cute or sm_90+ missing)"
)


@pytest.fixture(autouse=True)
def _require_native_dispatch(monkeypatch):
    monkeypatch.setenv("LIGER_KERNEL_STRICT", "1")


# (atol_fwd, atol_bwd, rtol_bwd) per dtype: fp32 is near-exact. bf16 must stay
# loose: the Triton side of the comparison (llama casting keeps dW/dX partials
# in bf16) deviates from an fp64 reference by up to ~6e-2, while CuTe DSL is
# ~8e-3, so cross-impl parity needs room for the noisier side. dW is a row-sum
# (magnitude ~50 for 512-777 rows), where 1 bf16 ulp is already 0.25 — an
# rtol is mandatory there.
_TOL = {
    torch.float32: (2e-4, 2e-3, 1e-4),
    torch.bfloat16: (1e-1, 2e-1, 2e-2),
    torch.float16: (1e-2, 1e-1, 1e-2),
}


def _clone_args(X, R, W):
    out = []
    for t in (X, R, W):
        c = t.detach().clone()
        c.requires_grad_(True)
        out.append(c)
    return out


def _pytorch_fused_add_rms_norm(X, R, W, eps):
    S = X + R
    S_fp32 = S.float()
    normalized = S_fp32 * torch.rsqrt(S_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(X.dtype) * W, S


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize(
    "rows_shape",
    [
        (512, 512),
        (777, 1024),  # irregular, non-power-of-2
        (4, 128, 512),  # 3-D decoder-layer shape (regression: bwd used to crash)
        (2, 333, 1024),  # 3-D irregular
    ],
    ids=["2d", "2d-irregular", "3d", "3d-irregular"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("in_place", [False, True])
def test_cutedsl_fused_add_rms_norm_matches_triton(rows_shape, dtype, in_place):
    from liger_kernel.backends import dispatch

    set_seed()
    *batch, N = rows_shape
    X = torch.randn(*batch, N, device="cuda", dtype=dtype)
    R = torch.randn(*batch, N, device="cuda", dtype=dtype)
    W = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6
    atol_fwd, atol_bwd, rtol_bwd = _TOL[dtype]

    Xc, Rc, Wc = _clone_args(X, R, W)
    Xt, Rt, Wt = _clone_args(X, R, W)
    Xp, Rp, Wp = _clone_args(X, R, W)

    Yp, Sp = _pytorch_fused_add_rms_norm(Xp, Rp, Wp, eps)
    Yc, Sc = dispatch("fused_add_rms_norm", Xc, Rc, Wc, eps, in_place=in_place, impl="nvidia-cutedsl")
    Yt, St = dispatch("fused_add_rms_norm", Xt, Rt, Wt, eps, in_place=in_place, impl="nvidia-triton")

    for Y, S in ((Yc, Sc), (Yt, St)):
        assert Y.shape == S.shape == X.shape
        assert Y.dtype == S.dtype == dtype
        assert Y.requires_grad and S.requires_grad
        torch.testing.assert_close(Y.float(), Yp.float(), atol=atol_fwd, rtol=0)
        torch.testing.assert_close(S, Sp, atol=0, rtol=0)
    torch.testing.assert_close(Yc.float(), Yt.float(), atol=atol_fwd, rtol=0)
    torch.testing.assert_close(Sc.float(), St.float(), atol=atol_fwd, rtol=0)

    # Backward through BOTH outputs — the residual branch is exactly the path
    # that used to crash (and must contribute dS to dX/dR).
    #
    # Each call gets a PRISTINE clone of the upstream grads: with
    # ``in_place=True`` the backward reuses the dY storage for dX (documented
    # Liger convention — "in_place determines whether to modify dY in-place
    # to store dX"), so the first backward mutates the caller's grad tensor
    # by design. Sharing one gY buffer across the two impls corrupts the
    # second call's inputs (this is a test artifact, not a kernel bug — the
    # in-place kernels are bitwise identical to out-of-place on cloned dY).
    gY = torch.randn_like(Yc)
    gS = torch.randn_like(Sc)
    torch.autograd.backward((Yc, Sc), grad_tensors=(gY.clone(), gS.clone()))
    torch.autograd.backward((Yt, St), grad_tensors=(gY.clone(), gS.clone()))
    torch.autograd.backward((Yp, Sp), grad_tensors=(gY.clone(), gS.clone()))

    for actual, reference in ((Xc, Xp), (Rc, Rp), (Wc, Wp), (Xt, Xp), (Rt, Rp), (Wt, Wp)):
        assert actual.grad is not None and actual.grad.shape == actual.shape
        _assert_nonzero_close(actual.grad, reference.grad, atol_bwd, rtol_bwd)
    torch.testing.assert_close(Xc.grad.float(), Xt.grad.float(), atol=atol_bwd, rtol=rtol_bwd)
    torch.testing.assert_close(Rc.grad.float(), Rt.grad.float(), atol=atol_bwd, rtol=rtol_bwd)
    torch.testing.assert_close(Wc.grad.float(), Wt.grad.float(), atol=atol_bwd, rtol=rtol_bwd)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("output_branch", ["both", "norm-only", "residual-only"])
def test_cutedsl_fused_add_rms_norm_3d_backward_does_not_raise(dtype, in_place, output_branch):
    """Focused regression: a 3-D [batch, seq, hidden] fwd+bwd must NOT raise.

    Before the flatten-around-launch fix, the ``ds=`` epilogue fold path handed
    the saved 3-D ``S`` straight to the 2-D compiled backward kernel, raising
    ``ValueError: ... expected ndim=2``.
    """
    from liger_kernel.backends import dispatch

    set_seed()
    B, T, N = 4, 128, 512
    X = torch.randn(B, T, N, device="cuda", dtype=dtype, requires_grad=True)
    R = torch.randn(B, T, N, device="cuda", dtype=dtype, requires_grad=True)
    W = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    Xp, Rp, Wp = _clone_args(X, R, W)
    Yp, Sp = _pytorch_fused_add_rms_norm(Xp, Rp, Wp, 1e-6)
    atol_fwd, atol_bwd, rtol_bwd = _TOL[dtype]

    Y, S = dispatch("fused_add_rms_norm", X, R, W, 1e-6, in_place=in_place, impl="nvidia-cutedsl")
    assert Y.shape == (B, T, N)
    assert S.shape == (B, T, N)
    assert Y.dtype == S.dtype == dtype
    torch.testing.assert_close(Y.float(), Yp.float(), atol=atol_fwd, rtol=0)
    torch.testing.assert_close(S, Sp, atol=0, rtol=0)

    gY = torch.randn_like(Y)
    gS = torch.randn_like(S)
    if output_branch == "both":
        torch.autograd.backward((Y, S), grad_tensors=(gY.clone(), gS.clone()))
        torch.autograd.backward((Yp, Sp), grad_tensors=(gY.clone(), gS.clone()))
    elif output_branch == "norm-only":
        Y.backward(gY.clone())
        Yp.backward(gY.clone())
    else:
        S.backward(gS.clone())
        Sp.backward(gS.clone())

    assert X.grad is not None and X.grad.shape == (B, T, N)
    assert R.grad is not None and R.grad.shape == (B, T, N)
    assert W.grad is not None and W.grad.shape == (N,)
    _assert_nonzero_close(X.grad, Xp.grad, atol_bwd, rtol_bwd)
    _assert_nonzero_close(R.grad, Rp.grad, atol_bwd, rtol_bwd)
    if output_branch == "residual-only":
        assert Wp.grad is None  # The independent graph never visits the normalization branch.
        torch.testing.assert_close(W.grad, torch.zeros_like(W), atol=0, rtol=0)
        torch.testing.assert_close(X.grad, gS, atol=0, rtol=0)
        torch.testing.assert_close(R.grad, gS, atol=0, rtol=0)
    else:
        _assert_nonzero_close(W.grad, Wp.grad, atol_bwd, rtol_bwd)
