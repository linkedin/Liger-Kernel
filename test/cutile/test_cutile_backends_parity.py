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
from liger_kernel.backends.registry import get_discovery_failures  # noqa: E402
from liger_kernel.backends.registry import get_registered  # noqa: E402
from liger_kernel.ops._nvidia_shared import cutile_compiler_available  # noqa: E402
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


def _cutile_supported() -> tuple[bool, str]:
    """Is this host *genuinely* capable of running the cuTile impls?

    ``cuda.tile`` importability is already guaranteed by the module-level
    ``importorskip``. Here we check the remaining hard requirements the
    ``Capability`` gate enforces: a CUDA device, compute capability >= (10, 0),
    and a reachable ``tileiras`` compiler. When all three hold, absence of the
    registered impl is a real import/registration regression — NOT an
    "unsupported machine" — so callers must fail instead of skip.
    """
    if not torch.cuda.is_available():
        return False, "CUDA device not available"
    cc = torch.cuda.get_device_capability()
    if cc < (10, 0):
        return False, f"device sm_{cc[0]}{cc[1]} < required sm_100 (cuTile needs compute capability >= 10.0)"
    if not cutile_compiler_available():
        return False, "cuTile (tileiras) compiler not reachable"
    return True, ""


def _cutile_impl(op: str) -> str:
    impl = _impl_matching(op, "cutile")
    if impl is not None:
        return impl

    supported, reason = _cutile_supported()
    if not supported:
        # Genuinely unsupported host (CPU / cc < 10.0 / missing compiler): skip.
        pytest.skip(f"nvidia-cutile not available for {op!r}: {reason}")

    # Compatible host (CUDA + sm>=100 + cuTile compiler) but the impl is still
    # absent -> a real backend import/registration regression that discovery
    # swallowed. Surface it and FAIL rather than mask it as a skip.
    failures = get_discovery_failures()
    detail = (
        "; ".join(f"{path} -> {etype.__name__}: {msg}" for path, (etype, msg) in failures.items())
        if failures
        else "none recorded"
    )
    registered = get_registered(op, "nvidia-cutile")
    if registered is None:
        pytest.fail(
            f"nvidia-cutile impl for {op!r} did NOT register on a cuTile-capable host "
            f"(CUDA + sm>=100 + tileiras present). This is an import/registration regression, "
            f"not an unsupported machine. Recorded discovery failures: {detail}"
        )
    # Registered but filtered out by the capability gate on a host we consider
    # compatible -> a capability-evaluation mismatch worth surfacing.
    pytest.fail(
        f"nvidia-cutile impl for {op!r} is registered but reported unavailable on a cuTile-capable host. "
        f"Capability reason: {registered.capability.reason_if_unsatisfied()!r}. "
        f"Recorded discovery failures: {detail}"
    )


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
    # none — cuTile keeps input-dtype math when W matches the activation but
    # promotes to fp32 when W is a wider dtype (its backward reduces dW as
    # ``x.float() * rstd.float()``). Mirror that fp32-faithful normalization so
    # the elementwise dW/dX comparison reflects the gradient the kernel really
    # computes (the mixed-precision test always uses a wider fp32 weight). The
    # kernel forms ``mean(x*x)`` from input-dtype products (``ct.mul(x, x)``)
    # then normalizes/reduces in fp32, so square in the input dtype first.
    mean_sq = x.pow(2).to(torch.float32).mean(-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    return ((x.to(torch.float32) * rstd) * (offset + weight)).to(in_dtype)


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
    _assert_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"layer_norm-mixed/{dtype} dw")
    _assert_close(b.grad, br.grad, tols["atol_bwd"], tols["rtol_bwd"], f"layer_norm-mixed/{dtype} db")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("w_dtype,b_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_layer_norm_affine_dtype_mismatch(dtype, w_dtype, b_dtype):
    """W and B with DIFFERENT dtypes: dW must come back in W.dtype and dB in
    B.dtype *independently* (the pre-fix bug cast dB to W.dtype), and each must
    be elementwise-correct vs an fp32 ground-truth reference. A same-dtype test
    can never catch dB being reduced/cast to W.dtype instead of B.dtype."""
    assert w_dtype != b_dtype
    ci = _cutile_impl("layer_norm")
    M, N = 256, 1024
    tols = _tols("layer_norm", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(1)
    x_cpu = torch.randn(M, N, generator=g)
    w_cpu = torch.randn(N, generator=g)
    b_cpu = torch.randn(N, generator=g)

    def build():
        x = x_cpu.to("cuda", dtype).detach().requires_grad_(True)
        w = w_cpu.to("cuda", w_dtype).detach().requires_grad_(True)
        b = b_cpu.to("cuda", b_dtype).detach().requires_grad_(True)
        return x, w, b

    x, w, b = build()
    y = F.layer_norm(x, w, b, 1e-6, impl=ci)
    y.backward(torch.ones_like(y))

    xr, wr, br = build()
    yr = _ref_layer_norm(xr, wr, br, 1e-6)
    yr.backward(torch.ones_like(yr))

    # Each param gradient must independently preserve its OWN dtype.
    assert w.grad.dtype == w_dtype, f"dW dtype {w.grad.dtype} != W.dtype {w_dtype}"
    assert b.grad.dtype == b_dtype, f"dB dtype {b.grad.dtype} != B.dtype {b_dtype}"
    assert x.grad.dtype == dtype

    _assert_close(y, yr, tols["atol_fwd"], tols["rtol_fwd"], f"ln-wbmismatch/{dtype}/W{w_dtype}/B{b_dtype} fwd")
    _assert_close(x.grad, xr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"ln-wbmismatch/{dtype} dx")
    _assert_close(w.grad, wr.grad, tols["atol_bwd"], tols["rtol_bwd"], f"ln-wbmismatch/{dtype} dw")
    _assert_close(b.grad, br.grad, tols["atol_bwd"], tols["rtol_bwd"], f"ln-wbmismatch/{dtype} db")


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
    BT, V = 256, 64
    # Build VALID log-probs first (row-wise log_softmax over the V last dim),
    # then take a strided row-slice to obtain a genuinely NON-contiguous VIEW
    # whose rows are still valid log-probs (each retained row is untouched).
    # The previous version re-ran log_softmax after transposing, which
    # materialised a fresh contiguous tensor and never exercised a
    # non-contiguous JSD input at all.
    lq_full = torch.randn(2 * BT, V, device="cuda", dtype=dtype).log_softmax(-1)
    lp_full = torch.randn(2 * BT, V, device="cuda", dtype=dtype).log_softmax(-1)
    lq = lq_full[::2].detach().requires_grad_(True)  # (BT, V) rows stride 2V -> non-contig
    lp = lp_full[::2].detach()
    # Assert the exact layout that reaches the kernel, immediately before dispatch.
    assert not lq.is_contiguous() and not lp.is_contiguous()
    assert lq.shape == (BT, V)

    out = F.jsd(lq, lp, None, beta=0.5, ignore_index=-100, impl=ci)
    lqr = lq.detach().clone().requires_grad_(True)  # contiguous copy of the same values
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
        # Assert the exact layout that reaches the fused kernel, before dispatch.
        assert not si.is_contiguous() and si.shape == (BT, H)
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
    _assert_close(dswc, dswt, tols["atol_bwd"], tols["rtol_bwd"], "fljsd-noncontig d_student_weight")


# ===========================================================================
# 6. Fused-linear JSD semantics vs an INDEPENDENT torch reference
# ===========================================================================
# The parity-vs-Triton tests above only pin the inner cuTile primitive against
# Triton (both share the outer chunked cuBLAS loop, so a shared bug would hide).
# These cases build the whole fused_linear_jsd forward+backward from plain torch
# ops in fp32 and differentiate through it, then compare the cuTile op against
# that ground truth across: ignored labels, non-unit upstream gradient,
# non-default temperature, asymmetric beta, and a multi-outer-chunk shape.
def _ref_fused_linear_jsd(si_cpu, sw_cpu, t_in_cpu, tw_cpu, labels, beta, ignore_index, temperature, upstream):
    """fp32 autograd ground truth for fused_linear_jsd -> (loss, d_si, d_sw)."""
    si = si_cpu.to("cuda", torch.float32).detach().requires_grad_(True)
    sw = sw_cpu.to("cuda", torch.float32).detach().requires_grad_(True)
    t_in = t_in_cpu.to("cuda", torch.float32).detach()
    tw = tw_cpu.to("cuda", torch.float32).detach()
    student_logits = (si @ sw.t()) / temperature
    teacher_logits = (t_in @ tw.t()) / temperature
    log_q = student_logits.log_softmax(-1)
    log_p = teacher_logits.log_softmax(-1)
    loss = _ref_jsd(log_q, log_p, labels, beta, ignore_index)
    loss.backward(upstream)
    return loss.detach(), si.grad, sw.grad


def _run_cutile_fused(si_cpu, sw_cpu, t_in_cpu, tw_cpu, labels, beta, ignore_index, temperature, upstream, impl):
    si = si_cpu.to("cuda", torch.float32).detach().requires_grad_(True)
    sw = sw_cpu.to("cuda", torch.float32).detach().requires_grad_(True)
    t_in = t_in_cpu.to("cuda", torch.float32).detach()
    tw = tw_cpu.to("cuda", torch.float32).detach()
    out = F.fused_linear_jsd(
        si, sw, t_in, tw, labels, jsd_beta=beta, ignore_index=ignore_index, temperature=temperature, impl=impl
    )
    out.backward(upstream)
    return out.detach(), si.grad, sw.grad


# (name, BT, H, V, beta, temperature, upstream_scale, label_kind)
# multi_chunk uses V >> chunk_mem_const*H so the outer loop runs >1 chunk on
# BOTH B200 (const 8) and B300 (const 16): V=4096 > 16*64.
_FLJSD_SEMANTIC_CASES = [
    ("ignored_labels", 32, 128, 512, 0.5, 1.0, 1.0, "scattered_ignore"),
    ("non_unit_grad", 32, 128, 512, 0.5, 1.0, 2.75, "none"),
    ("non_default_temp", 32, 128, 512, 0.5, 2.0, 1.0, "none"),
    ("asymmetric_beta_low", 32, 128, 512, 0.1, 1.0, 1.0, "none"),
    ("asymmetric_beta_high", 32, 128, 512, 0.9, 1.0, 1.0, "none"),
    ("multi_chunk", 256, 64, 4096, 0.5, 1.0, 1.0, "none"),
]


@pytest.mark.parametrize("case", _FLJSD_SEMANTIC_CASES, ids=[c[0] for c in _FLJSD_SEMANTIC_CASES])
def test_fused_linear_jsd_semantics_vs_torch(case):
    name, BT, H, V, beta, temperature, scale, label_kind = case
    ci = _cutile_impl("fused_linear_jsd")
    dtype = torch.float32
    tols = _tols("fused_linear_jsd", ci, dtype)
    g = torch.Generator(device="cpu").manual_seed(hash(name) & 0xFFFF)
    si_cpu = torch.randn(BT, H, generator=g)
    sw_cpu = torch.randn(V, H, generator=g)
    ti_cpu = torch.randn(BT, H, generator=g)
    tw_cpu = torch.randn(V, H, generator=g)

    if label_kind == "scattered_ignore":
        labels = torch.arange(BT, device="cuda", dtype=torch.int64) % V
        labels[::3] = -100
    else:
        labels = None
    upstream = torch.tensor(scale, device="cuda", dtype=torch.float32)

    oc, dsic, dswc = _run_cutile_fused(si_cpu, sw_cpu, ti_cpu, tw_cpu, labels, beta, -100, temperature, upstream, ci)
    orf, dsir, dswr = _ref_fused_linear_jsd(si_cpu, sw_cpu, ti_cpu, tw_cpu, labels, beta, -100, temperature, upstream)
    _assert_close(oc, orf, tols["atol_fwd"], tols["rtol_fwd"], f"fljsd-sem/{name} fwd")
    _assert_close(dsic, dsir, tols["atol_bwd"], tols["rtol_bwd"], f"fljsd-sem/{name} d_student_input")
    _assert_close(dswc, dswr, tols["atol_bwd"], tols["rtol_bwd"], f"fljsd-sem/{name} d_student_weight")
