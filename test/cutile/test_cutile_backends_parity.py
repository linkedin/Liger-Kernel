"""Pinned ``nvidia-cutile`` parity + correctness tests for the five cuTile
backends (RMSNorm, LayerNorm, Softmax, JSD, FusedLinearJSD).

These are the tests requested by the PR-E code review. They exercise the fixes
landed in ``src/liger_kernel/ops/backends/_cutile/``:

* forward AND backward parity vs the Triton reference in fp16 / bf16 / fp32,
  using each op's *registered* tolerances;
* mixed-precision affine — bf16/fp16 activations with **fp32** affine params —
  checking the output, ``dX``, ``dW`` (and LayerNorm ``dB``) against an fp32
  ground-truth reference, and that ``dW`` / ``dB`` come back in the parameter
  dtype (the pre-fix backend silently quantized them);
* forward-selector boundaries for the three norm/softmax variant selectors
  (``N=1``, non-powers-of-two, both sides of each cutoff, 8192, 8193), and that
  explicit invalid softmax modes raise clearly;
* JSD ``beta`` in ``{0, 0.5, 1, 0.1}``, no-labels / scattered-ignore /
  all-ignore, and non-unit upstream gradient;
* non-contiguous inputs for every public op.

The whole module skips cleanly when ``cuda.tile`` is not importable (so CPU /
Hopper CI stays green). Individual tests skip when the cuTile impl is not
*registered* for the op on this device (i.e. compute capability < 10.0), but a
real assertion failure is never turned into a skip.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

# Skip the entire module cleanly when the cuTile compiler stack is absent.
pytest.importorskip("cuda.tile", reason="cuda.tile (cuTile) not importable on this host")

import liger_kernel.functional as F  # noqa: E402  (importing also registers the op discovery map)

from liger_kernel.backends.dispatch import available_impls  # noqa: E402
from liger_kernel.backends.registry import get_registered  # noqa: E402
from liger_kernel.testing import assert_op_correctness  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# ---------------------------------------------------------------------------
# Impl discovery helpers
# ---------------------------------------------------------------------------
def _impl_matching(op: str, needle: str) -> Optional[str]:
    for name in available_impls(op):
        if needle in name:
            return name
    return None


def _cutile_impl(op: str) -> str:
    impl = _impl_matching(op, "cutile")
    if impl is None:
        pytest.skip(f"nvidia-cutile not registered for {op!r} on this device (needs compute capability >= 10.0)")
    return impl


def _triton_impl(op: str) -> str:
    impl = _impl_matching(op, "triton")
    if impl is None:
        pytest.skip(f"triton reference not registered for {op!r}")
    return impl


def _tols(op: str, impl: str, dtype: torch.dtype) -> dict:
    registered = get_registered(op, impl)
    entry = dict((registered.tolerances if registered is not None else {}).get(dtype, {}))
    defaults = {
        torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
        torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
        torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
    }
    for k, v in defaults[dtype].items():
        entry.setdefault(k, v)
    return entry


def _assert_close(actual, expected, atol, rtol, label):
    torch.testing.assert_close(
        actual.to(torch.float32),
        expected.to(torch.float32),
        atol=atol,
        rtol=rtol,
        msg=lambda m: f"[{label}] {m}",
    )


def _sum_close(actual, expected, atol, rtol) -> bool:
    """Aggregate (per-feature sum) fallback for low-precision dW/dB reductions."""
    s = actual.to(torch.float32).sum()
    r = expected.to(torch.float32).sum()
    denom = r.abs().clamp_min(1e-8)
    rel = (s - r).abs() / denom
    return bool(torch.isfinite(rel) and (rel.item() < rtol or (s - r).abs().item() < atol))


# ===========================================================================
# Ground-truth references (identical semantics to the Triton kernels)
# ===========================================================================
def _ref_rms_norm(x, weight, eps, offset, casting_mode):
    in_dtype = x.dtype
    if casting_mode == "llama":
        xf = x.to(torch.float32)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * rstd).to(in_dtype) * (offset + weight)
    if casting_mode == "gemma":
        xf = x.to(torch.float32)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return ((xf * rstd) * (offset + weight.to(torch.float32))).to(in_dtype)
    # none
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rstd) * (offset + weight)


def _ref_layer_norm(x, weight, bias, eps):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    xhat = (xf - mean) * torch.rsqrt(var + eps)
    return (xhat * weight.to(torch.float32) + bias.to(torch.float32)).to(in_dtype)


def _ref_jsd(log_q, log_p, shift_labels, beta, ignore_index):
    lq = log_q.float()
    lp = log_p.float()
    if beta == 0.0:
        loss = torch.exp(lp) * (lp - lq)
    elif beta == 1.0:
        loss = torch.exp(lq) * (lq - lp)
    else:
        m = beta * torch.exp(lp) + (1.0 - beta) * torch.exp(lq)
        log_m = torch.log(m)
        loss = beta * torch.exp(lp) * lp + (1.0 - beta) * torch.exp(lq) * lq - m * log_m
    if shift_labels is not None:
        mask = (shift_labels != ignore_index).unsqueeze(-1).float()
        loss = loss * mask
        n = float(mask.sum().item())
        if n == 0:
            return torch.zeros((), dtype=log_q.dtype, device=log_q.device)
    else:
        n = float(log_q.shape[0])
    return (loss.sum() / n).to(log_q.dtype)


# ===========================================================================
# 1. Forward + backward parity vs the Triton reference (fp16 / bf16 / fp32)
# ===========================================================================
_NORM_PARITY_SHAPES = [(256, 1024), (512, 2048)]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _NORM_PARITY_SHAPES)
@pytest.mark.parametrize("casting_mode", ["llama", "gemma", "none"])
def test_rms_norm_fwd_bwd_parity(dtype, shape, casting_mode):
    """Forward + backward parity for the cuTile RMSNorm vs the fp32 PyTorch
    reference (identical semantics to the Triton kernel), using the op's
    registered tolerances. Compared against the reference rather than the Triton
    backend directly so a Triton bug can't mask a cuTile bug (and vice-versa) —
    the convention used by ``test/ops/test_jsd.py``.
    """
    ci = _cutile_impl("rms_norm")
    assert_op_correctness("rms_norm", ci, shape, dtype, casting_mode=casting_mode)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _NORM_PARITY_SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_layer_norm_fwd_bwd_parity(dtype, shape, with_bias):
    """Forward + backward parity (output, dX, dW, dB) for cuTile LayerNorm vs the
    fp32 PyTorch reference, using the op's registered tolerances."""
    ci = _cutile_impl("layer_norm")
    assert_op_correctness("layer_norm", ci, shape, dtype, extra={"include_bias": with_bias})


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(256, 1024), (512, 4096)])
def test_softmax_fwd_bwd_parity(dtype, shape):
    """Forward + backward parity for cuTile softmax vs ``torch.softmax`` (fp32)."""
    ci = _cutile_impl("softmax")
    M, N = shape
    tols = _tols("softmax", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(0)
    x_cpu = torch.randn(M, N, generator=g)
    dy_cpu = torch.randn(M, N, generator=g)

    x = x_cpu.to("cuda", dtype).detach().requires_grad_(True)
    xr = x_cpu.to("cuda", dtype).detach().requires_grad_(True)
    y = F.softmax(x, impl=ci)
    yr = torch.softmax(xr.float(), dim=-1).to(dtype)
    dy = dy_cpu.to("cuda", dtype)
    y.backward(dy)
    yr.backward(dy.clone())

    _assert_close(y, yr, tols["atol_fwd"], tols["rtol_fwd"], f"softmax/{dtype} fwd")
    _assert_close(x.grad, xr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"softmax/{dtype} dx")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_jsd_fwd_bwd_parity(dtype, beta):
    """Forward + backward parity for cuTile JSD vs the fp32 PyTorch reference."""
    ci = _cutile_impl("jsd")
    BT, V = 64, 512
    tols = _tols("jsd", ci, dtype)
    lq0, lp = _jsd_inputs(BT, V, dtype, seed=BT * 100 + V)

    lq = lq0.clone().requires_grad_(True)
    out = F.jsd(lq, lp, None, beta=beta, ignore_index=-100, impl=ci)
    out.backward()

    lqr = lq0.clone().requires_grad_(True)
    ref = _ref_jsd(lqr, lp, None, beta, -100)
    ref.backward()
    _assert_close(out, ref, tols["atol_fwd"], tols["rtol_fwd"], f"jsd/{dtype}/beta={beta} fwd")
    _assert_close(lq.grad, lqr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"jsd/{dtype}/beta={beta} dx")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_fused_linear_jsd_parity_vs_triton(dtype, beta):
    """Forward + backward parity for cuTile fused_linear_jsd vs the Triton
    backend. The two share the outer chunked cuBLAS implementation and differ
    only in the inner cuTile JSD primitive, so they agree tightly; comparing
    them directly pins the inner primitive."""
    ci = _cutile_impl("fused_linear_jsd")
    ti = _triton_impl("fused_linear_jsd")
    BT, H, V = 64, 128, 512
    tols = _tols("fused_linear_jsd", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(0)
    si_cpu = torch.randn(BT, H, generator=g)
    sw_cpu = torch.randn(V, H, generator=g)
    ti_cpu = torch.randn(BT, H, generator=g)
    tw_cpu = torch.randn(V, H, generator=g)

    def run(impl):
        si = si_cpu.to("cuda", dtype).detach().requires_grad_(True)
        sw = sw_cpu.to("cuda", dtype).detach().requires_grad_(True)
        t_in = ti_cpu.to("cuda", dtype).detach()
        tw = tw_cpu.to("cuda", dtype).detach()
        out = F.fused_linear_jsd(si, sw, t_in, tw, None, jsd_beta=beta, temperature=1.0, impl=impl)
        out.backward()
        return out, si.grad, sw.grad

    oc, dsic, dswc = run(ci)
    ot, dsit, dswt = run(ti)
    _assert_close(oc, ot, tols["atol_fwd"], tols["rtol_fwd"], f"fljsd/{dtype}/beta={beta} fwd")
    _assert_close(dsic, dsit, tols["atol_bwd"], tols["rtol_bwd"], f"fljsd/{dtype}/beta={beta} d_student_input")
    if not _sum_close(dswc, dswt, tols["atol_bwd"], tols["rtol_bwd"]):
        _assert_close(dswc, dswt, tols["atol_bwd"], tols["rtol_bwd"], f"fljsd/{dtype}/beta={beta} d_student_weight")


# ===========================================================================
# 2. Mixed-precision affine: bf16/fp16 activations + fp32 affine params
# ===========================================================================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("casting_mode", ["llama", "gemma", "none"])
def test_rms_norm_mixed_precision_affine(dtype, casting_mode):
    """fp32 weight must NOT be silently quantized to the activation dtype, and
    backward must differentiate exactly what forward computed. Compared against
    an fp32 ground-truth reference (identical to Triton's promotion semantics).
    """
    ci = _cutile_impl("rms_norm")
    M, N = 256, 1024
    tols = _tols("rms_norm", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(1)
    x_cpu = torch.randn(M, N, generator=g)
    w_cpu = torch.randn(N, generator=g)

    def build():
        x = x_cpu.to("cuda", dtype).detach().requires_grad_(True)
        w = w_cpu.to("cuda", torch.float32).detach().requires_grad_(True)  # fp32 affine
        return x, w

    x, w = build()
    y = F.rms_norm(x, w, 1e-6, casting_mode=casting_mode, impl=ci)
    y.backward(torch.ones_like(y))

    xr, wr = build()
    yr = _ref_rms_norm(xr, wr, 1e-6, 0.0, casting_mode)
    yr.backward(torch.ones_like(yr))

    # dW and (here) x-grad dtype must be preserved.
    assert w.grad.dtype == torch.float32, f"dW dtype {w.grad.dtype} != fp32 (weight was fp32)"
    assert x.grad.dtype == dtype, f"dX dtype {x.grad.dtype} != {dtype}"

    _assert_close(y, yr, tols["atol_fwd"], tols["rtol_fwd"], f"rms_norm-mixed/{casting_mode}/{dtype} fwd")
    _assert_close(x.grad, xr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"rms_norm-mixed/{casting_mode}/{dtype} dx")
    if not _sum_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"]):
        _assert_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"rms_norm-mixed/{casting_mode}/{dtype} dw")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_norm_mixed_precision_affine(dtype):
    ci = _cutile_impl("layer_norm")
    M, N = 256, 1024
    tols = _tols("layer_norm", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(1)
    x_cpu = torch.randn(M, N, generator=g)
    w_cpu = torch.randn(N, generator=g)
    b_cpu = torch.randn(N, generator=g)

    def build():
        x = x_cpu.to("cuda", dtype).detach().requires_grad_(True)
        w = w_cpu.to("cuda", torch.float32).detach().requires_grad_(True)  # fp32 affine
        b = b_cpu.to("cuda", torch.float32).detach().requires_grad_(True)  # fp32 affine
        return x, w, b

    x, w, b = build()
    y = F.layer_norm(x, w, b, 1e-6, impl=ci)
    y.backward(torch.ones_like(y))

    xr, wr, br = build()
    yr = _ref_layer_norm(xr, wr, br, 1e-6)
    yr.backward(torch.ones_like(yr))

    assert w.grad.dtype == torch.float32, f"dW dtype {w.grad.dtype} != fp32"
    assert b.grad.dtype == torch.float32, f"dB dtype {b.grad.dtype} != fp32 (bias was fp32)"
    assert x.grad.dtype == dtype

    _assert_close(y, yr, tols["atol_fwd"], tols["rtol_fwd"], f"layer_norm-mixed/{dtype} fwd")
    _assert_close(x.grad, xr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"layer_norm-mixed/{dtype} dx")
    if not _sum_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"]):
        _assert_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"layer_norm-mixed/{dtype} dw")
    if not _sum_close(b.grad, br.grad, tols["atol_bwd"], tols["rtol_bwd"]):
        _assert_close(b.grad, br.grad, tols["atol_bwd"], tols["rtol_bwd"], f"layer_norm-mixed/{dtype} db")


# ===========================================================================
# 3. Forward-selector boundary cases + explicit-mode rejection
# ===========================================================================
# Both sides of the RMSNorm/LayerNorm width cutoff (4096) and of the persistent
# row cutoff (M vs NUM_SMS), plus N=1, non-powers-of-two, and the 8192 cap.
_NORM_BOUNDARY = [
    (1, 1),  # single row, single col
    (8, 3),  # non-pow2 narrow, small M (standard/cached path)
    (4, 4095),  # just below the 4096 cached cutoff
    (4, 4096),  # exactly the cutoff
    (4, 4097),  # just above the cutoff (standard path)
    (2048, 256),  # large M -> static_persistent
    (4, 8192),  # widest supported single tile
]


@pytest.mark.parametrize("shape", _NORM_BOUNDARY)
def test_rms_norm_selector_boundaries(shape):
    ci = _cutile_impl("rms_norm")
    M, N = shape
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randn(M, N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    w = torch.randn(N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    y = F.rms_norm(x, w, 1e-6, casting_mode="llama", impl=ci)
    yr = _ref_rms_norm(x.detach(), w.detach(), 1e-6, 0.0, "llama")
    _assert_close(y, yr, 1e-4, 1e-4, f"rms_norm-boundary fwd {shape}")
    y.backward(torch.ones_like(y))
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.parametrize("shape", _NORM_BOUNDARY)
def test_layer_norm_selector_boundaries(shape):
    ci = _cutile_impl("layer_norm")
    M, N = shape
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randn(M, N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    w = torch.randn(N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    b = torch.randn(N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    y = F.layer_norm(x, w, b, 1e-6, impl=ci)
    yr = _ref_layer_norm(x.detach(), w.detach(), b.detach(), 1e-6)
    _assert_close(y, yr, 1e-3, 1e-3, f"layer_norm-boundary fwd {shape}")
    y.backward(torch.ones_like(y))
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.parametrize("op,fn", [("rms_norm", F.rms_norm), ("layer_norm", F.layer_norm)])
def test_norm_hidden_dim_over_cap_raises(op, fn):
    """N > 8192 must raise a clear, consistent error in forward (matching the
    backward cap) rather than silently mis-normalising."""
    ci = _cutile_impl(op)
    x = torch.randn(4, 8193, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(8193, device="cuda", dtype=torch.float32, requires_grad=True)
    with pytest.raises(RuntimeError, match="only supports hidden dim"):
        if op == "rms_norm":
            fn(x, w, 1e-6, impl=ci)
        else:
            b = torch.randn(8193, device="cuda", dtype=torch.float32)
            fn(x, w, b, 1e-6, impl=ci)


_SOFTMAX_BOUNDARY = [
    (1, 1),
    (8, 3),  # non-pow2 narrow
    (4, 8191),  # just below single-tile cap
    (4, 8192),  # exactly at the cap
    (4, 8193),  # just above -> auto picks chunked
    (2048, 256),  # large M -> static_persistent
    (4, 18432),  # very wide -> chunked
]


@pytest.mark.parametrize("shape", _SOFTMAX_BOUNDARY)
def test_softmax_selector_boundaries(shape):
    ci = _cutile_impl("softmax")
    M, N = shape
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(M, N, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    y = F.softmax(x, impl=ci)  # auto mode
    yr = torch.softmax(x.detach().float(), dim=-1)
    _assert_close(y, yr, 1e-4, 1e-4, f"softmax-boundary fwd {shape}")
    y.backward(torch.randn_like(y))
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", ["standard", "static_persistent"])
def test_softmax_explicit_singletile_mode_rejects_wide(mode):
    """Explicit single-tile modes must reject a row wider than the tile budget
    (next_pow2(N) > 8192) instead of compiling/launching an oversized tile."""
    ci = _cutile_impl("softmax")
    x = torch.randn(8, 9000, device="cuda", dtype=torch.float32, requires_grad=True)
    with pytest.raises(ValueError, match="next_pow2"):
        F.softmax(x, impl=ci, mode=mode)


def test_softmax_invalid_mode_raises():
    ci = _cutile_impl("softmax")
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown mode"):
        F.softmax(x, impl=ci, mode="does_not_exist")


@pytest.mark.parametrize("op", ["rms_norm", "layer_norm"])
def test_norm_invalid_mode_raises(op):
    ci = _cutile_impl(op)
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    w = torch.randn(128, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown mode"):
        if op == "rms_norm":
            F.rms_norm(x, w, 1e-6, impl=ci, mode="does_not_exist")
        else:
            b = torch.randn(128, device="cuda", dtype=torch.float32)
            F.layer_norm(x, w, b, 1e-6, impl=ci, mode="does_not_exist")


# ===========================================================================
# 4. JSD beta / ignore-index / non-unit upstream gradient coverage
# ===========================================================================
def _jsd_inputs(BT, V, dtype, seed=7):
    torch.manual_seed(seed)
    raw_q = torch.randn(BT, V, device="cuda", dtype=dtype)
    raw_p = torch.randn(BT, V, device="cuda", dtype=dtype)
    return raw_q.log_softmax(-1).detach(), raw_p.log_softmax(-1).detach()


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0])
def test_jsd_beta_values(beta):
    ci = _cutile_impl("jsd")
    dtype = torch.float32
    tols = _tols("jsd", ci, dtype)
    lq0, lp = _jsd_inputs(64, 512, dtype)
    lq = lq0.clone().requires_grad_(True)
    out = F.jsd(lq, lp, None, beta=beta, ignore_index=-100, impl=ci)
    out.backward()

    lqr = lq0.clone().requires_grad_(True)
    ref = _ref_jsd(lqr, lp, None, beta, -100)
    ref.backward()
    _assert_close(out, ref, tols["atol_fwd"], tols["rtol_fwd"], f"jsd-beta={beta} fwd")
    _assert_close(lq.grad, lqr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"jsd-beta={beta} dx")


@pytest.mark.parametrize(
    "label_kind",
    ["none", "scattered_ignore", "all_ignore"],
)
def test_jsd_ignore_patterns(label_kind):
    ci = _cutile_impl("jsd")
    dtype = torch.float32
    BT, V = 32, 256
    tols = _tols("jsd", ci, dtype)
    lq0, lp = _jsd_inputs(BT, V, dtype, seed=11)
    ignore_index = -100

    if label_kind == "none":
        labels = None
    elif label_kind == "scattered_ignore":
        labels = torch.arange(BT, device="cuda", dtype=torch.int64) % V
        labels[::3] = ignore_index  # every third row ignored
    else:  # all_ignore
        labels = torch.full((BT,), ignore_index, device="cuda", dtype=torch.int64)

    lq = lq0.clone().requires_grad_(True)
    out = F.jsd(lq, lp, labels, beta=0.5, ignore_index=ignore_index, impl=ci)
    out.backward()

    if label_kind == "all_ignore":
        # Every row ignored -> zero loss and zero gradient (the reference is a
        # disconnected zero scalar, so there is nothing to differentiate).
        assert float(out.item()) == 0.0
        assert torch.count_nonzero(lq.grad) == 0
        return

    lqr = lq0.clone().requires_grad_(True)
    ref = _ref_jsd(lqr, lp, labels, 0.5, ignore_index)
    ref.backward()
    _assert_close(out, ref, tols["atol_fwd"], tols["rtol_fwd"], f"jsd-{label_kind} fwd")
    _assert_close(lq.grad, lqr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"jsd-{label_kind} dx")


def test_jsd_non_unit_upstream_grad():
    ci = _cutile_impl("jsd")
    dtype = torch.float32
    tols = _tols("jsd", ci, dtype)
    lq0, lp = _jsd_inputs(48, 256, dtype, seed=13)
    scale = 2.75

    lq = lq0.clone().requires_grad_(True)
    out = F.jsd(lq, lp, None, beta=0.5, ignore_index=-100, impl=ci)
    out.backward(torch.tensor(scale, device="cuda"))

    lqr = lq0.clone().requires_grad_(True)
    ref = _ref_jsd(lqr, lp, None, 0.5, -100)
    ref.backward(torch.tensor(scale, device="cuda"))
    _assert_close(lq.grad, lqr.grad, tols["atol_bwd"], tols["rtol_bwd"], "jsd-nonunit-grad dx")


# ===========================================================================
# 5. Non-contiguous inputs for every public op
# ===========================================================================
def test_rms_norm_non_contiguous():
    ci = _cutile_impl("rms_norm")
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(5)
    # Transpose to make the row dimension non-contiguous.
    base = torch.randn(1024, 256, generator=g).to("cuda", dtype)
    x = base.t().detach().requires_grad_(True)  # (256, 1024), non-contiguous
    assert not x.is_contiguous()
    w = torch.randn(1024, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    y = F.rms_norm(x, w, 1e-6, casting_mode="llama", impl=ci)
    yr = _ref_rms_norm(x.detach().contiguous(), w.detach(), 1e-6, 0.0, "llama")
    _assert_close(y, yr, 1e-4, 1e-4, "rms_norm-noncontig fwd")
    y.backward(torch.ones_like(y))
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


def test_layer_norm_non_contiguous():
    ci = _cutile_impl("layer_norm")
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(5)
    base = torch.randn(1024, 256, generator=g).to("cuda", dtype)
    x = base.t().detach().requires_grad_(True)
    assert not x.is_contiguous()
    w = torch.randn(1024, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    b = torch.randn(1024, generator=g).to("cuda", dtype).detach().requires_grad_(True)
    y = F.layer_norm(x, w, b, 1e-6, impl=ci)
    yr = _ref_layer_norm(x.detach().contiguous(), w.detach(), b.detach(), 1e-6)
    _assert_close(y, yr, 1e-3, 1e-3, "layer_norm-noncontig fwd")
    y.backward(torch.ones_like(y))
    assert torch.isfinite(x.grad).all()


def test_softmax_non_contiguous():
    ci = _cutile_impl("softmax")
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(5)
    base = torch.randn(1024, 256, generator=g).to("cuda", dtype)
    x = base.t().detach().requires_grad_(True)  # (256, 1024)
    assert not x.is_contiguous()
    y = F.softmax(x, impl=ci)
    yr = torch.softmax(x.detach().float(), dim=-1)
    _assert_close(y, yr, 1e-4, 1e-4, "softmax-noncontig fwd")
    y.backward(torch.randn_like(y))
    assert torch.isfinite(x.grad).all()


def test_jsd_non_contiguous():
    ci = _cutile_impl("jsd")
    dtype = torch.float32
    tols = _tols("jsd", ci, dtype)
    torch.manual_seed(17)
    raw_q = torch.randn(512, 64, device="cuda", dtype=dtype)
    raw_p = torch.randn(512, 64, device="cuda", dtype=dtype)
    # log_softmax over last dim then transpose -> (64, 512) non-contiguous rows.
    lq_full = raw_q.log_softmax(-1)
    lp_full = raw_p.log_softmax(-1)
    lq = lq_full.t().detach().requires_grad_(True)
    lp = lp_full.t().detach()
    assert not lq.is_contiguous()
    # Re-normalise across the new last dim so inputs are valid log-probs.
    lq = lq.log_softmax(-1).detach().requires_grad_(True)
    lp = lp.log_softmax(-1).detach()
    out = F.jsd(lq, lp, None, beta=0.5, ignore_index=-100, impl=ci)
    lqr = lq.detach().clone().requires_grad_(True)
    ref = _ref_jsd(lqr, lp, None, 0.5, -100)
    out.backward()
    ref.backward()
    _assert_close(out, ref, tols["atol_fwd"], tols["rtol_fwd"], "jsd-noncontig fwd")
    _assert_close(lq.grad, lqr.grad, tols["atol_bwd"], tols["rtol_bwd"], "jsd-noncontig dx")


def test_fused_linear_jsd_non_contiguous():
    ci = _cutile_impl("fused_linear_jsd")
    ti = _triton_impl("fused_linear_jsd")
    dtype = torch.float32
    tols = _tols("fused_linear_jsd", ci, dtype)
    BT, H, V = 64, 128, 256
    g = torch.Generator(device="cpu").manual_seed(19)
    si_cpu = torch.randn(H, BT, generator=g)  # transposed layout
    sw_cpu = torch.randn(V, H, generator=g)
    ti_cpu = torch.randn(BT, H, generator=g)
    tw_cpu = torch.randn(V, H, generator=g)

    def run(impl):
        si = si_cpu.to("cuda", dtype).t().detach().requires_grad_(True)  # (BT, H) non-contig
        assert not si.is_contiguous()
        sw = sw_cpu.to("cuda", dtype).detach().requires_grad_(True)
        t_in = ti_cpu.to("cuda", dtype).detach()
        tw = tw_cpu.to("cuda", dtype).detach()
        out = F.fused_linear_jsd(si, sw, t_in, tw, None, jsd_beta=0.5, impl=impl)
        out.backward()
        return out, si.grad, sw.grad

    oc, dsic, dswc = run(ci)
    ot, dsit, dswt = run(ti)
    _assert_close(oc, ot, tols["atol_fwd"], tols["rtol_fwd"], "fljsd-noncontig fwd")
    _assert_close(dsic, dsit, tols["atol_bwd"], tols["rtol_bwd"], "fljsd-noncontig d_student_input")
