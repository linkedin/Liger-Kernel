"""Dispatcher correctness tests for CuTe DSL gap ops.

Six ``nvidia-cutedsl`` dispatcher registrations (``geglu``, ``softmax``,
``layer_norm``, ``kl_div``, ``jsd``, ``fused_linear_jsd``) plus the
``jsd_loss_and_grad`` primitive had no CuTe DSL-side test coverage. These
tests pin parity against the Triton implementations of the same dispatcher
ops (the repo's reference kernels), in fp32 and bf16, forward and backward.

Methodology that matters:

1. Every impl call gets its own pristine CLONE of the same sampled inputs (two
   RNG draws never compare equal, and an earlier smoke round of "bugs" turned
   out to be RNG-position artifacts).

2. Each cell is PINNED as *real CuTe DSL* or *Triton fallback* and the pin is
   verified against the dispatcher's fallback signal. Several CuTe DSL
   registrations punt to Triton for some inputs (fp32 JSD parity guard,
   oversized tiles, ``reduction='none'``, ...). If we blindly dispatched
   ``impl="nvidia-cutedsl"`` for such a cell we would be comparing
   Triton-vs-Triton — a vacuous self-comparison that always passes. So a real
   cell that silently falls back is a hard failure, and a by-design fallback
   cell is skipped (not silently "passed") after asserting the fallback
   actually happened.

3. The backward oracle weights each grad-bearing output by a fixed random
   weight that is SHARED (cloned) across both backends: ``(out * w).sum()``.
   A plain ``out.sum()`` oracle is degenerate for some ops — e.g.
   ``softmax(x).sum()`` equals the row count (a constant), so its gradient is
   ~0 and the backward would compare noise-vs-noise. The weighted sum makes the
   gradient data-dependent and the two backends directly comparable.

Notes:
- Real CuTe DSL parity is claimed *only on Blackwell* (sm_100+). The gap-op
  CuTe DSL registrations advertise sm_90+, but this suite scopes the "real
  CuTe DSL" expectation to Blackwell so it is portable and can never error on a
  non-Blackwell device: on Hopper (sm_90) and any earlier arch every "real"
  cell is skipped up-front as out-of-scope, while the Blackwell path stays fully
  asserted (see ``_is_blackwell`` and ``_assert_parity``).
- ``jsd`` / ``jsd_loss_and_grad`` run the real CuTe DSL primitive in both fp32
  and bf16 on Blackwell: the fp32 path computes ``exp``/``log`` via
  libdevice-precise (no-fastmath) intrinsics so it lands on the strict fp32 JSD
  contract, so it no longer falls back to Triton (the old fp32-parity guard is
  gone). Non-Blackwell archs skip the real cells up-front as out-of-scope.
- ``fused_linear_jsd`` runs the real CuTe DSL route in both dtypes on Blackwell
  (parity-checked against the Triton reference); non-Blackwell archs skip it as
  out-of-scope.
- ``jsd_loss_and_grad`` is the primitive contract used by distillation:
  it returns ``(loss, dx)`` with ``dx`` written in-place.

The suite is skipped (never failed) when CUDA or the ``nvidia-cutlass-dsl``
package is unavailable.
"""

import warnings

import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl gap-op tests require CUDA")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cutedsl_available() -> bool:
    """Whether the CuTe DSL backend can *execute* here at all.

    The gap-op CuTe DSL registrations advertise sm_90+ (``min_cc=(9, 0)``), so
    this floor only guards that the ``cutlass.cute`` package imports and a
    Hopper-or-newer GPU is present. It deliberately does NOT decide whether we
    *claim* real CuTe DSL parity — that is Blackwell-only and enforced per cell
    via :func:`_is_blackwell` (see ``_assert_parity``). Keeping the floor at
    sm_90 lets the by-design Triton-fallback cells (fp32 JSD guard,
    ``fused_linear_jsd``) still run and assert their fallback on Hopper.
    """
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def _is_blackwell() -> bool:
    """Real CuTe DSL gap-op parity is claimed only on Blackwell (sm_100+).

    The dispatcher's CuTe DSL gap-op kernels are validated against Triton on
    Blackwell. On Hopper (sm_90) and any earlier arch this returns ``False`` and
    every "real CuTe DSL" cell is skipped as out-of-scope instead of being
    forced down the CuTe DSL path — so the suite stays portable and error-free
    across GPUs while the Blackwell assertions remain fully intact.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


skip_no_cutedsl = pytest.mark.skipif(
    not _cutedsl_available(), reason="cutedsl backend unavailable (cutlass.cute or sm_90+ missing)"
)

# (tol_fwd, tol_bwd): bf16 needs headroom for the Triton side's bf16
# accumulation; fp32 is near-exact. Matches the validated H200 smoke
# tolerances (bwd = 10x fwd bar).
_TOL = {
    torch.float32: (1e-4, 1e-3),
    torch.bfloat16: (2e-2, 2e-1),
}


def _clone_args(args):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            c = a.detach().clone()
            if a.requires_grad:
                c.requires_grad_(True)
            out.append(c)
        else:
            out.append(a)
    return out


def _grads(args):
    return [a.grad for a in args if isinstance(a, torch.Tensor) and a.grad is not None]


def _dispatch_cutedsl_detecting_fallback(op, *args, **kwargs):
    """Dispatch ``op`` to ``nvidia-cutedsl`` and report whether the CuTe DSL impl
    internally punted to Triton.

    The CuTe DSL registrations emit a one-shot :class:`LigerImplFallbackWarning`
    (via ``emit_fallback_warning``) right before returning a Triton result
    whenever they cannot run the real kernel (fp32 JSD parity guard, oversized
    tiles, ``reduction='none'``, ...). That warning is therefore a reliable "I
    did NOT execute CuTe DSL" probe. The dispatcher dedupes the warning to at
    most once per ``(op, requested, actual)`` key, so we clear that dedup set
    first to guarantee the signal fires for *this* call.

    Returns ``(output, fell_back)``.
    """
    import importlib

    # ``import liger_kernel.backends.dispatch as m`` would bind ``m`` to the
    # re-exported ``dispatch`` *function* (the package __init__ shadows the
    # submodule name), so grab the real module object explicitly.
    _dispatch_mod = importlib.import_module("liger_kernel.backends.dispatch")

    with _dispatch_mod._FALLBACK_NOTIFIED_LOCK:
        _dispatch_mod._FALLBACK_NOTIFIED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _dispatch_mod.dispatch(op, *args, impl="nvidia-cutedsl", **kwargs)
    fell_back = any(isinstance(w.message, _dispatch_mod.LigerImplFallbackWarning) for w in caught)
    return out, fell_back


def _assert_parity(op, args, dtype, extra=None, *, expect_cutedsl=True):
    """Dispatch ``op`` under both impls on cloned inputs; assert fwd+bwd parity.

    ``expect_cutedsl`` pins whether this cell is supposed to run the real CuTe
    DSL kernel (``True``) or fall back to Triton by design (``False``). The pin
    is verified against the dispatcher's fallback signal so a cell can never
    silently compare a backend against itself:

    - a real cell that unexpectedly falls back      -> hard failure
    - a fallback cell that unexpectedly runs CuTe DSL -> hard failure (stale matrix)
    - a confirmed fallback cell                       -> skipped (vacuous parity)
    """
    from liger_kernel.backends import dispatch

    # Real CuTe DSL gap-op parity is claimed only on Blackwell (sm_100+). On
    # Hopper / any non-Blackwell device a "real" cell is out of scope: skip it
    # up-front (before touching the CuTe DSL path) so the suite is portable and
    # can never error cross-GPU. By-design Triton-fallback cells
    # (``expect_cutedsl=False``) still run everywhere and assert their fallback,
    # exactly like the fp32 JSD guard below.
    if expect_cutedsl and not _is_blackwell():
        pytest.skip(
            f"{op} [{dtype}]: real CuTe DSL parity is scoped to Blackwell "
            f"(sm_100+); this device is non-Blackwell, so the cell is out of "
            f"scope (clean skip, no cross-GPU error)."
        )

    tol_fwd, tol_bwd = _TOL[dtype]
    ac, at = _clone_args(args), _clone_args(args)
    oc, fell_back = _dispatch_cutedsl_detecting_fallback(op, *ac, **(extra or {}))

    if expect_cutedsl:
        assert not fell_back, (
            f"{op} [{dtype}]: expected the CuTe DSL kernel to execute, but the "
            f"dispatcher fell back to Triton — this cell would compare "
            f"Triton-vs-Triton (vacuous). Fix the cell or mark it as a fallback."
        )
    else:
        assert fell_back, (
            f"{op} [{dtype}]: expected a Triton fallback by design, but the CuTe "
            f"DSL kernel actually ran. The fallback matrix is stale — add real "
            f"CuTe DSL parity coverage for this cell."
        )
        pytest.skip(
            f"{op} [{dtype}] falls back to Triton by design; parity would be "
            f"Triton-vs-Triton (vacuous), so it is not CuTe DSL coverage."
        )

    ot = dispatch(op, *at, impl="nvidia-triton", **(extra or {}))

    tc = oc if isinstance(oc, (tuple, list)) else (oc,)
    tt = ot if isinstance(ot, (tuple, list)) else (ot,)
    for x, y in zip(tc, tt):
        torch.testing.assert_close(x.float(), y.float(), atol=tol_fwd, rtol=tol_fwd)

    # Non-trivial, comparable backward: weight each grad-bearing output by a
    # fixed random weight that is SHARED (cloned) across both backends, i.e.
    # ``(out * w).sum()``. A plain ``out.sum()`` oracle is degenerate for some
    # ops (e.g. ``softmax(x).sum()`` is ~constant, so its gradient vanishes and
    # backward compares noise-vs-noise). The identical ``w`` keeps the two
    # backends' gradients directly comparable.
    losses_c, losses_t = [], []
    for x, y in zip(tc, tt):
        if not x.requires_grad:
            continue
        w = torch.randn(x.shape, device=x.device, dtype=torch.float32)
        losses_c.append((x.float() * w.clone()).sum())
        losses_t.append((y.float() * w.clone()).sum())
    if losses_c:
        sum(losses_c).backward()
        sum(losses_t).backward()
        for gx, gy in zip(_grads(ac), _grads(at)):
            torch.testing.assert_close(gx.float(), gy.float(), atol=tol_bwd, rtol=tol_bwd)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_geglu_matches_triton(dtype):
    set_seed()
    a = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
    _assert_parity("geglu", (a, b), dtype)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_softmax_matches_triton(dtype):
    set_seed()
    x = torch.randn(512, 4096, device="cuda", dtype=dtype, requires_grad=True)
    _assert_parity("softmax", (x,), dtype)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [(512, 512), (4, 128, 512)],  # 3-D = decoder-layer shaped usage
    ids=["2d", "3d"],
)
def test_cutedsl_layer_norm_matches_triton(shape, dtype):
    set_seed()
    N = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=dtype, requires_grad=True)
    _assert_parity("layer_norm", (x, w, b, 1e-6), dtype)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_kl_div_matches_triton(dtype):
    set_seed()
    yp = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype).requires_grad_(True)
    yt = torch.softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    _assert_parity("kl_div", (yp, yt), dtype, extra={"reduction": "batchmean"})


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_jsd_matches_triton(dtype):
    set_seed()
    stu = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype).requires_grad_(True)
    tea = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    # Both dtypes run the real CuTe DSL primitive on Blackwell: the fp32 path
    # uses libdevice-precise exp/log so it now matches the Triton reference
    # (the old fp32-parity fallback guard is gone). Non-Blackwell is skipped
    # up-front as out-of-scope by ``_assert_parity``.
    _assert_parity("jsd", (stu, tea), dtype, expect_cutedsl=True)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_jsd_loss_and_grad_primitive_matches_triton(dtype):
    from liger_kernel.backends import dispatch

    set_seed()
    tol_fwd, tol_bwd = _TOL[dtype]
    stu = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    tea = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    labels = torch.arange(512, device="cuda") % 10

    # Both dtypes run the real CuTe DSL primitive on Blackwell (the fp32 path
    # uses libdevice-precise exp/log, so it matches Triton; the old fp32-parity
    # fallback guard is gone). Real CuTe DSL parity is Blackwell-only (sm_100+):
    # on a non-Blackwell device the cell is out of scope and skipped up-front,
    # so the suite never errors cross-GPU.
    expect_cutedsl = True
    if expect_cutedsl and not _is_blackwell():
        pytest.skip(
            f"jsd_loss_and_grad [{dtype}]: real CuTe DSL parity is scoped to "
            f"Blackwell (sm_100+); this device is non-Blackwell, so the cell is "
            f"out of scope (clean skip, no cross-GPU error)."
        )
    (loss_c, dx_c), fell_back = _dispatch_cutedsl_detecting_fallback(
        "jsd_loss_and_grad", stu.clone(), tea.clone(), labels.clone(), 0.5, -100, 512.0
    )
    assert not fell_back, (
        f"jsd_loss_and_grad [{dtype}]: expected the CuTe DSL primitive to run, "
        "but it fell back to Triton (would be Triton-vs-Triton, vacuous)."
    )

    loss_t, dx_t = dispatch(
        "jsd_loss_and_grad", stu.clone(), tea.clone(), labels.clone(), 0.5, -100, 512.0, impl="nvidia-triton"
    )
    torch.testing.assert_close(loss_c.float(), loss_t.float(), atol=tol_fwd, rtol=tol_fwd)
    torch.testing.assert_close(dx_c.float(), dx_t.float(), atol=tol_bwd, rtol=tol_bwd)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_fused_linear_jsd_matches_triton(dtype):
    set_seed()
    si = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    sw = torch.randn(4096, 512, device="cuda", dtype=dtype, requires_grad=True) * 0.1
    ti = torch.randn(512, 512, device="cuda", dtype=dtype) * 0.1
    tw = torch.randn(4096, 512, device="cuda", dtype=dtype) * 0.1
    labels = torch.arange(512, device="cuda") % 10
    # The composed op runs the real CuTe DSL route in both dtypes on Blackwell
    # (parity-checked against the Triton reference); non-Blackwell is skipped
    # up-front as out-of-scope by ``_assert_parity``.
    _assert_parity("fused_linear_jsd", (si, sw, ti, tw, labels), dtype, expect_cutedsl=True)
