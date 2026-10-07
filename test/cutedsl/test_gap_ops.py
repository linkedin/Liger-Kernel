"""Native CuTe DSL gap-op correctness against Triton AND independent PyTorch.

Inputs and upstream gradients are cloned per implementation because some
backwards consume their storage. Output arity and every expected input gradient
are checked, including a relative-norm check for small loss gradients.

The native cells use aligned, supported widths on the advertised sm_90+ path.
Hopper is not excluded merely because validation is pending; launch or numerical
failures on an advertised device must surface. Pre-sm_90 devices and missing
CUDA/CuTe DSL are unsupported and skipped. Fallback coverage lives separately in
test_shape_fallbacks.py. For fused_linear_jsd, "native" refers to the inner JSD
primitive, not its shared PyTorch matmuls or Triton gradient-scaling kernel.
"""

import warnings

import pytest
import torch

import liger_kernel.functional  # noqa: F401

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="cutedsl gap-op tests require CUDA")


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

# Existing (tol_fwd, tol_bwd) parity tolerances, unchanged.
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


def _assert_outputs_close(actual, expected, tol):
    assert type(actual) is type(expected), "output container mismatch"
    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    assert len(actual) == len(expected) > 0, "output arity mismatch"
    for x, y in zip(actual, expected):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert x.requires_grad == y.requires_grad, "output detached from autograd"
        _assert_nonzero_close(x, y, tol, tol)


def _assert_nonzero_close(actual, expected, atol, rtol):
    assert actual is not None and expected is not None, "missing expected gradient/output"
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.float(), expected.float()
    assert torch.count_nonzero(actual) > 0, "all-zero actual gradient/output"
    assert torch.count_nonzero(expected) > 0, "degenerate reference gradient/output"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    # Elementwise atol alone can exceed every entry of a batch-averaged loss gradient.
    error = torch.linalg.vector_norm(actual - expected)
    scale = torch.linalg.vector_norm(expected)
    assert error <= rtol * scale, f"relative L2 error {error / scale} exceeds {rtol}"


def _assert_grads_close(actual, expected, tol):
    assert len(actual) == len(expected), "input arity mismatch"
    for i, (x, y) in enumerate(zip(actual, expected)):
        if not isinstance(x, torch.Tensor):
            continue
        assert x.requires_grad == y.requires_grad
        if x.requires_grad:
            assert x.grad is not None and y.grad is not None, f"missing gradient for input {i}"
            assert x.grad.shape == x.shape and y.grad.shape == y.shape
            _assert_nonzero_close(x.grad, y.grad, tol, tol)
        else:
            assert x.grad is None and y.grad is None


def _pytorch_jsd_loss(log_q, log_p, labels=None, beta=0.5, ignore_index=-100, n_non_ignore=None):
    """Per-element generalized JSD; autograd supplies the independent d(log Q)."""
    x, y = log_q.float(), log_p.float()
    q, p = x.exp(), y.exp()
    if beta == 0.0:
        loss = p * (y - x)
    elif beta == 1.0:
        loss = q * (x - y)
    else:
        mixture = beta * p + (1.0 - beta) * q
        loss = beta * p * (y - mixture.log()) + (1.0 - beta) * q * (x - mixture.log())
    if labels is not None:
        keep = labels != ignore_index
        loss = loss * keep.unsqueeze(-1)
    if n_non_ignore is None:
        n_non_ignore = log_q.shape[0] if labels is None else int(keep.sum())
    return loss / max(n_non_ignore, 1)


def _pytorch_fused_linear_jsd(si, sw, ti, tw, labels, jsd_beta=0.5, ignore_index=-100, temperature=1.0):
    # Match the public projection dtype, but differentiate the unfused PyTorch graph.
    log_q = torch.log_softmax((si @ sw.T).float() / temperature, dim=-1)
    log_p = torch.log_softmax((ti @ tw.T).float() / temperature, dim=-1)
    return _pytorch_jsd_loss(log_q, log_p, labels, jsd_beta, ignore_index).sum()


def _dispatch_cutedsl_detecting_fallback(op, *args, fallback_reason=None, **kwargs):
    """Dispatch ``op`` to ``nvidia-cutedsl`` and report whether the CuTe DSL impl
    internally punted to Triton.

    The CuTe DSL registrations emit a one-shot :class:`LigerImplFallbackWarning`
    (via ``emit_fallback_warning``) right before returning a Triton result
    whenever they cannot run the real kernel (architecture-specific JSD guard, oversized
    tiles, ``reduction='none'``, ...). That warning is therefore a reliable "I
    did NOT execute CuTe DSL" probe. The dispatcher dedupes the warning to at
    most once per ``(op, requested, actual)`` key, so we clear that dedup set
    for this call, restoring the original set afterwards.

    Returns ``(output, fell_back)``.
    """
    import importlib

    # ``import liger_kernel.backends.dispatch as m`` would bind ``m`` to the
    # re-exported ``dispatch`` *function* (the package __init__ shadows the
    # submodule name), so grab the real module object explicitly.
    _dispatch_mod = importlib.import_module("liger_kernel.backends.dispatch")
    inner_impls = []

    def track_inner_dispatch(op_name, *inner_args, **inner_kwargs):
        if op_name == "jsd_loss_and_grad":
            inner_impls.append(inner_kwargs.get("impl"))
        return _dispatch_mod.dispatch(op_name, *inner_args, **inner_kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_dispatch_mod, "_FALLBACK_NOTIFIED", set())
        patch.setenv("LIGER_KERNEL_STRICT", "0")
        if op in ("jsd", "fused_linear_jsd"):
            composed_module = importlib.import_module(f"liger_kernel.ops.{op}")
            patch.setattr(composed_module, "dispatch", track_inner_dispatch)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", _dispatch_mod.LigerImplFallbackWarning)
            out = _dispatch_mod.dispatch(op, *args, impl="nvidia-cutedsl", **kwargs)
    if op in ("jsd", "fused_linear_jsd"):
        assert inner_impls and set(inner_impls) == {"nvidia-cutedsl"}, f"{op}: inner JSD route was {inner_impls}"
    fallbacks = [w for w in caught if isinstance(w.message, _dispatch_mod.LigerImplFallbackWarning)]
    for warning in caught:
        if warning not in fallbacks:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    if fallback_reason is not None:
        assert any(
            f"Liger {op}:" in str(w.message)
            and "falling back to 'nvidia-triton'" in str(w.message)
            and fallback_reason in str(w.message)
            for w in fallbacks
        ), f"{op}: expected Triton fallback reason {fallback_reason!r}; got {fallbacks}"
    return out, bool(fallbacks)


def _assert_parity(op, args, dtype, reference, extra=None):
    """Compare native CuTe DSL, Triton and PyTorch, including every input gradient."""
    from liger_kernel.backends import dispatch

    tol_fwd, tol_bwd = _TOL[dtype]
    ac, at, ap = _clone_args(args), _clone_args(args), _clone_args(args)
    op_ref = reference(*ap, **(extra or {}))
    oc, fell_back = _dispatch_cutedsl_detecting_fallback(op, *ac, **(extra or {}))
    assert not fell_back, f"{op} [{dtype}]: expected native CuTe DSL, not Triton-vs-Triton"
    ot = dispatch(op, *at, impl="nvidia-triton", **(extra or {}))

    _assert_outputs_close(oc, ot, tol_fwd)
    _assert_outputs_close(oc, op_ref, tol_fwd)
    _assert_outputs_close(ot, op_ref, tol_fwd)
    tc = oc if isinstance(oc, (tuple, list)) else (oc,)
    tt = ot if isinstance(ot, (tuple, list)) else (ot,)
    tp = op_ref if isinstance(op_ref, (tuple, list)) else (op_ref,)

    # Non-trivial, comparable backward: weight each grad-bearing output by a
    # fixed random weight that is SHARED (cloned) across both backends, i.e.
    # ``(out * w).sum()``. A plain ``out.sum()`` oracle is degenerate for some
    # ops (e.g. ``softmax(x).sum()`` is ~constant, so its gradient vanishes and
    # backward compares noise-vs-noise). The identical ``w`` keeps the two
    # backends' gradients directly comparable.
    losses_c, losses_t, losses_p = [], [], []
    assert len(tc) == len(tt) == len(tp)
    for x, y, z in zip(tc, tt, tp):
        if not x.requires_grad:
            continue
        w = torch.randn(x.shape, device=x.device, dtype=torch.float32)
        losses_c.append((x.float() * w.clone()).sum())
        losses_t.append((y.float() * w.clone()).sum())
        losses_p.append((z.float() * w.clone()).sum())
    assert losses_c, f"{op}: no differentiable output"
    sum(losses_c).backward()
    sum(losses_t).backward()
    sum(losses_p).backward()
    _assert_grads_close(ac, at, tol_bwd)
    _assert_grads_close(ac, ap, tol_bwd)
    _assert_grads_close(at, ap, tol_bwd)


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_geglu_matches_triton(dtype):
    set_seed()
    a = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
    _assert_parity(
        "geglu", (a, b), dtype, lambda a, b: torch.nn.functional.gelu(a.float(), approximate="tanh").to(a.dtype) * b
    )


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_softmax_matches_triton(dtype):
    set_seed()
    x = torch.randn(512, 4096, device="cuda", dtype=dtype, requires_grad=True)
    _assert_parity("softmax", (x,), dtype, lambda x: torch.softmax(x.float(), dim=-1).to(x.dtype))


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
    _assert_parity(
        "layer_norm",
        (x, w, b, 1e-6),
        dtype,
        lambda x, w, b, eps: torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), eps).to(
            x.dtype
        ),
    )


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_kl_div_matches_triton(dtype):
    set_seed()
    yp = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype).requires_grad_(True)
    yt = torch.softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    _assert_parity(
        "kl_div",
        (yp, yt),
        dtype,
        lambda yp, yt, reduction: torch.nn.functional.kl_div(yp.float(), yt.float(), reduction=reduction),
        extra={"reduction": "batchmean"},
    )


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_jsd_matches_triton(dtype):
    set_seed()
    stu = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype).requires_grad_(True)
    tea = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    _assert_parity("jsd", (stu, tea), dtype, lambda stu, tea: _pytorch_jsd_loss(stu, tea).sum().to(stu.dtype))


@cuda_required
@skip_no_cutedsl
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("label_mode", ["packed", "ignored", "no-labels"])
def test_cutedsl_jsd_loss_and_grad_primitive_matches_triton(dtype, beta, label_mode):
    from liger_kernel.backends import dispatch

    set_seed()
    tol_fwd, tol_bwd = _TOL[dtype]
    stu = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    tea = torch.nn.functional.log_softmax(torch.randn(512, 4096, device="cuda"), -1).to(dtype)
    labels = torch.arange(512, device="cuda") % 10
    if label_mode == "ignored":
        labels[::7] = -100
    elif label_mode == "no-labels":
        labels = None
    # The denominator may be GLOBAL while the primitive sees just one chunk.
    n_non_ignore = 1024.0 if label_mode == "ignored" else 512.0
    sc, tc, lc = _clone_args((stu, tea, labels))
    st, tt, lt = _clone_args((stu, tea, labels))
    sp = stu.detach().clone().requires_grad_(True)
    loss_p = _pytorch_jsd_loss(sp, tea, labels, beta, -100, n_non_ignore)
    loss_p.sum().backward()
    (loss_c, dx_c), fell_back = _dispatch_cutedsl_detecting_fallback(
        "jsd_loss_and_grad", sc, tc, lc, beta, -100, n_non_ignore
    )
    assert not fell_back, (
        f"jsd_loss_and_grad [{dtype}]: expected the CuTe DSL primitive to run, "
        "but it fell back to Triton (would be Triton-vs-Triton, vacuous)."
    )

    loss_t, dx_t = dispatch("jsd_loss_and_grad", st, tt, lt, beta, -100, n_non_ignore, impl="nvidia-triton")
    assert dx_c.data_ptr() == sc.data_ptr() and dx_t.data_ptr() == st.data_ptr()
    assert loss_c.shape == loss_t.shape == stu.shape
    assert loss_c.dtype == loss_t.dtype == torch.float32
    assert dx_c.dtype == dx_t.dtype == dtype
    torch.testing.assert_close(tc, tea, atol=0, rtol=0)
    torch.testing.assert_close(tt, tea, atol=0, rtol=0)
    if labels is not None:
        torch.testing.assert_close(lc, labels, atol=0, rtol=0)
        torch.testing.assert_close(lt, labels, atol=0, rtol=0)
    torch.testing.assert_close(loss_c.float(), loss_t.float(), atol=tol_fwd, rtol=tol_fwd)
    torch.testing.assert_close(dx_c.float(), dx_t.float(), atol=tol_bwd, rtol=tol_bwd)
    for loss, dx in ((loss_c, dx_c), (loss_t, dx_t)):
        _assert_nonzero_close(loss, loss_p.detach(), tol_fwd, tol_fwd)
        _assert_nonzero_close(dx, sp.grad, tol_bwd, tol_bwd)
        if label_mode == "ignored":
            assert torch.count_nonzero(loss[labels == -100]) == 0
            assert torch.count_nonzero(dx[labels == -100]) == 0


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
    _assert_parity("fused_linear_jsd", (si, sw, ti, tw, labels), dtype, _pytorch_fused_linear_jsd)


def test_parity_helper_rejects_missing_gradient():
    actual = [torch.ones(4, requires_grad=True) for _ in range(2)]
    expected = _clone_args(actual)
    for tensor in (actual[0], *expected):
        tensor.sum().backward()
    with pytest.raises(AssertionError, match="missing gradient for input 1"):
        _assert_grads_close(actual, expected, 1e-3)


def test_parity_helper_rejects_truncated_outputs():
    with pytest.raises(AssertionError, match="output arity mismatch"):
        _assert_outputs_close((torch.ones(4),), (torch.ones(4), torch.ones(4)), 1e-4)


def test_parity_helper_rejects_detached_output():
    with pytest.raises(AssertionError, match="output detached from autograd"):
        _assert_outputs_close(torch.ones(4), torch.ones(4, requires_grad=True), 1e-4)


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_parity_helper_rejects_small_incorrect_gradients(scale):
    expected = torch.full((16,), 1e-7)
    with pytest.raises(AssertionError):
        _assert_nonzero_close(expected * scale, expected, 1e-3, 1e-3)


def test_clone_args_isolates_storage_and_preserves_grad_requirements():
    args = (torch.ones(4, requires_grad=True), torch.ones(4), None, 0.5)
    first, second = _clone_args(args), _clone_args(args)
    with torch.no_grad():
        first[0].zero_()
        first[1].zero_()
    assert len(first) == len(second) == len(args)
    for i, original in enumerate(args[:2]):
        clone = second[i]
        torch.testing.assert_close(clone, original, atol=0, rtol=0)
        assert clone.requires_grad == original.requires_grad and clone.is_leaf
    assert first[2:] == second[2:] == [None, 0.5]


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_pytorch_jsd_reference_mask_and_global_denominator(beta):
    q = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    p = torch.tensor([[0.5, 0.5], [0.125, 0.875]])
    log_q = q.log().requires_grad_(True)
    labels = torch.tensor([0, -100])
    loss = _pytorch_jsd_loss(log_q, p.log(), labels, beta, n_non_ignore=8)
    loss.sum().backward()
    if beta == 0.0:
        expected_loss = torch.nn.functional.kl_div(q.log(), p, reduction="none")
        expected_grad = -p
    elif beta == 1.0:
        expected_loss = torch.nn.functional.kl_div(p.log(), q, reduction="none")
        expected_grad = q * (q.log() - p.log() + 1)
    else:
        mixture = beta * p + (1 - beta) * q
        expected_loss = beta * torch.nn.functional.kl_div(mixture.log(), p, reduction="none")
        expected_loss += (1 - beta) * torch.nn.functional.kl_div(mixture.log(), q, reduction="none")
        expected_grad = (1 - beta) * q * (q.log() - mixture.log())
    torch.testing.assert_close(loss[0], expected_loss[0] / 8)
    torch.testing.assert_close(log_q.grad[0], expected_grad[0] / 8)
    assert torch.count_nonzero(loss[1]) == 0
    assert torch.count_nonzero(log_q.grad[1]) == 0
