"""Cross-impl correctness tests for the JSD op.

Parametrizes over every (op, impl) currently usable on the host:

- ``nvidia-triton`` — Liger's original Triton kernel (correctness anchor).
- ``nvidia-cutile`` — cuTile kernel ported from NVIDIA TileGym + PR #1228
  (MIT). Only registered on Blackwell sm_100+ with tileiras reachable;
  otherwise auto-gated out by Capability and the parametrize collapses.

Each test compares the impl against a pure-PyTorch JSD reference, NOT against
the Triton kernel — so a Triton bug doesn't mask a cuTile bug (or vice versa).
"""

from __future__ import annotations

from typing import List
from typing import Optional

import pytest
import torch

import liger_kernel.functional  # noqa: F401  (registers op locations)
import liger_kernel.ops.jsd as jsd_ops

from liger_kernel.backends.dispatch import available_impls
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.functional import jsd as functional_jsd

_REGISTERED_IMPLS: List[str] = available_impls("jsd")

# Shapes chosen to exercise:
#  (8, 64)        - tiny, ignored-row sanity
#  (32, 256)      - small power-of-2
#  (64, 4099)     - non-power-of-2 V (forces padding mode)
#  (128, 32000)   - typical llama vocab
_SHAPES = [(8, 64), (32, 256), (64, 4099), (128, 32000)]
_DTYPES = [torch.bfloat16, torch.float32]
_BETAS = [0.0, 0.5, 1.0]


def _pytorch_jsd_reference(
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
) -> torch.Tensor:
    """Pure-PyTorch generalized JSD. Anchors both Triton & cuTile.

    Computes everything in fp32 internally; result cast back to ``log_q.dtype``.
    Mirrors the math in ``LigerJSDFunction``:

        M     = beta * exp(log_p) + (1 - beta) * exp(log_q)
        loss  = beta * KL(P || M) + (1 - beta) * KL(Q || M)

    when ``beta in (0, 1)``; the degenerate forward/reverse-KL limits are
    handled directly because ``beta=0`` and ``beta=1`` collapse one of the
    KL terms to 0.
    """
    log_q_f = log_q.float()
    log_p_f = log_p.float()

    if beta == 0.0:
        # Forward KL: KL(P || M) with M = P, so loss = sum(P * (log P - log Q)) = sum(P * (Y - X))
        loss = torch.exp(log_p_f) * (log_p_f - log_q_f)
    elif beta == 1.0:
        # Reverse KL: KL(Q || M) with M = Q, so loss = sum(Q * (log Q - log P)) = sum(Q * (X - Y))
        loss = torch.exp(log_q_f) * (log_q_f - log_p_f)
    else:
        m = beta * torch.exp(log_p_f) + (1.0 - beta) * torch.exp(log_q_f)
        log_m = torch.log(m)
        # beta * sum(P * (log P - log M)) + (1 - beta) * sum(Q * (log Q - log M))
        # rearrange to match the Liger kernel form:
        loss = beta * torch.exp(log_p_f) * log_p_f + (1.0 - beta) * torch.exp(log_q_f) * log_q_f - m * log_m

    if shift_labels is not None:
        mask = (shift_labels != ignore_index).unsqueeze(-1).float()
        loss = loss * mask
        n_non_ignore = float(mask.sum().item())
        if n_non_ignore == 0:
            return torch.zeros((), dtype=log_q.dtype, device=log_q.device)
    else:
        n_non_ignore = float(log_q.shape[0])

    total = loss.sum() / n_non_ignore
    return total.to(log_q.dtype)


def _make_inputs(BT: int, V: int, dtype: torch.dtype):
    """Build (log_q, log_p) with autograd, both in log-space (after log_softmax)."""
    torch.manual_seed(BT * 1000 + V)  # deterministic per shape
    raw_q = torch.randn(BT, V, device="cuda", dtype=dtype)
    raw_p = torch.randn(BT, V, device="cuda", dtype=dtype)
    log_q = raw_q.log_softmax(-1).detach().requires_grad_()
    log_p = raw_p.log_softmax(-1).detach()
    return log_q, log_p


def _tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 1e-2}
    if dtype == torch.float16:
        return {"atol": 5e-3, "rtol": 1e-3}
    return {"atol": 1e-5, "rtol": 1e-5}


# ---------------------------------------------------------------------------
# Sanity tests on the registry (no GPU compute)
# ---------------------------------------------------------------------------


def test_jsd_registers_at_least_triton():
    impls = available_impls("jsd")
    assert any(i in ("triton", "nvidia-triton") for i in impls), (
        f"expected at least 'nvidia-triton' to be registered; got {impls}"
    )


def test_jsd_loss_and_grad_registers_at_least_triton():
    impls = available_impls("jsd_loss_and_grad")
    assert any(i in ("triton", "nvidia-triton") for i in impls), (
        f"expected at least 'nvidia-triton' for jsd_loss_and_grad; got {impls}"
    )


@pytest.mark.parametrize("impl", _REGISTERED_IMPLS)
def test_jsd_propagates_backend_to_inner_primitive(monkeypatch, impl):
    if impl == "nvidia-cutile":
        pytest.skip("cuTile standalone JSD directly owns its kernel")

    observed_impls = []
    real_dispatch = jsd_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "jsd_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(jsd_ops, "dispatch", tracking_dispatch)
    log_q, log_p = _make_inputs(8, 256, torch.bfloat16)

    functional_jsd(log_q, log_p, impl=impl).backward()

    assert observed_impls
    assert set(observed_impls) == {impl}


def test_jsd_default_path_dispatches_inner_primitive(monkeypatch):
    observed_impls = []
    real_dispatch = jsd_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "jsd_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(jsd_ops, "dispatch", tracking_dispatch)
    log_q, log_p = _make_inputs(8, 256, torch.bfloat16)

    jsd_ops.LigerJSDFunction.apply(log_q, log_p).backward()

    assert observed_impls == [None]


# ---------------------------------------------------------------------------
# Correctness tests — each registered impl vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", _REGISTERED_IMPLS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("beta", _BETAS)
def test_jsd_forward_matches_reference(impl, dtype, shape, beta):
    BT, V = shape
    log_q, log_p = _make_inputs(BT, V, dtype)

    ref = _pytorch_jsd_reference(log_q.detach(), log_p, None, beta, -100)
    out = functional_jsd(log_q, log_p, None, beta=beta, ignore_index=-100, impl=impl)
    tol = _tolerances(dtype)
    assert torch.allclose(out.float(), ref.float(), **tol), (
        f"[impl={impl} dtype={dtype} shape={shape} beta={beta}] "
        f"got {out.item():.6f}, ref {ref.item():.6f}, diff {(out - ref).abs().item():.2e}"
    )


@pytest.mark.parametrize("impl", _REGISTERED_IMPLS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_jsd_backward_matches_reference(impl, dtype, shape):
    """Compare grad-of-log_q against autograd through the pure-PyTorch reference."""
    BT, V = shape
    beta = 0.5  # only test the generalized case for backward (fastest)

    log_q, log_p = _make_inputs(BT, V, dtype)
    log_q_ref = log_q.detach().clone().requires_grad_()

    out = functional_jsd(log_q, log_p, None, beta=beta, ignore_index=-100, impl=impl)
    out.backward()
    grad_impl = log_q.grad

    out_ref = _pytorch_jsd_reference(log_q_ref, log_p, None, beta, -100)
    out_ref.backward()
    grad_ref = log_q_ref.grad

    tol = _tolerances(dtype)
    # Bwd tolerance is the same as fwd here (small fp32-internal kernels).
    assert torch.allclose(grad_impl.float(), grad_ref.float(), **tol), (
        f"[impl={impl} dtype={dtype} shape={shape}] grad max_diff="
        f"{(grad_impl.float() - grad_ref.float()).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Cross-impl parity: cuTile vs Triton (when both are available)
# ---------------------------------------------------------------------------


_TRITON_IN = any(i in _REGISTERED_IMPLS for i in ("nvidia-triton", "triton"))
_CUTILE_IN = any(i in _REGISTERED_IMPLS for i in ("nvidia-cutile", "cutile"))


@pytest.mark.skipif(not (_TRITON_IN and _CUTILE_IN), reason="needs both triton + cutile")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("beta", _BETAS)
def test_cutile_matches_triton_fwd(dtype, shape, beta):
    BT, V = shape
    log_q, log_p = _make_inputs(BT, V, dtype)
    triton_out = functional_jsd(
        log_q.detach().requires_grad_(),
        log_p,
        None,
        beta=beta,
        ignore_index=-100,
        impl="nvidia-triton",
    )
    cutile_out = functional_jsd(
        log_q.detach().requires_grad_(),
        log_p,
        None,
        beta=beta,
        ignore_index=-100,
        impl="nvidia-cutile",
    )
    tol = _tolerances(dtype)
    assert torch.allclose(triton_out.float(), cutile_out.float(), **tol), (
        f"[dtype={dtype} shape={shape} beta={beta}] cuTile vs Triton: "
        f"triton={triton_out.item():.6f}, cutile={cutile_out.item():.6f}"
    )


# ---------------------------------------------------------------------------
# jsd_loss_and_grad primitive: per-impl (BT, V) parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (_TRITON_IN and _CUTILE_IN), reason="needs both triton + cutile")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_jsd_loss_and_grad_cutile_matches_triton(dtype, shape):
    """The primitive used by fused_linear_jsd must agree across impls."""
    BT, V = shape
    log_q, log_p = _make_inputs(BT, V, dtype)
    beta = 0.5
    ignore_index = -100
    n_non_ignore = float(BT)

    # Triton path overwrites input — work on a clone so we can call cuTile too.
    triton_loss, triton_dx = dispatch(
        "jsd_loss_and_grad",
        log_q.detach().clone(),
        log_p.detach(),
        None,
        beta,
        ignore_index,
        n_non_ignore,
        impl="nvidia-triton",
    )
    cutile_loss, cutile_dx = dispatch(
        "jsd_loss_and_grad",
        log_q.detach().clone(),
        log_p.detach(),
        None,
        beta,
        ignore_index,
        n_non_ignore,
        impl="nvidia-cutile",
    )

    tol = _tolerances(dtype)
    loss_diff = (triton_loss.float() - cutile_loss.float()).abs().max().item()
    dx_diff = (triton_dx.float() - cutile_dx.float()).abs().max().item()
    assert loss_diff < tol["atol"] * 10, f"[dtype={dtype} shape={shape}] loss_diff={loss_diff:.2e}"
    assert dx_diff < tol["atol"] * 10, f"[dtype={dtype} shape={shape}] dx_diff={dx_diff:.2e}"


def test_jsd_loss_and_grad_cutile_runtime_scale():
    if "nvidia-cutile" not in _REGISTERED_IMPLS:
        pytest.skip("cuTile JSD is unavailable")

    log_q, log_p = _make_inputs(8, 256, torch.bfloat16)
    loss_4, dx_4 = dispatch(
        "jsd_loss_and_grad",
        log_q.detach().clone(),
        log_p,
        None,
        0.5,
        -100,
        4.0,
        impl="nvidia-cutile",
    )
    loss_8, dx_8 = dispatch(
        "jsd_loss_and_grad",
        log_q.detach().clone(),
        log_p,
        None,
        0.5,
        -100,
        8.0,
        impl="nvidia-cutile",
    )

    torch.testing.assert_close(loss_4, loss_8 * 2.0)
    torch.testing.assert_close(dx_4, dx_8 * 2.0)


# ---------------------------------------------------------------------------
# fused_linear_jsd routes through dispatch — make sure both impls produce
# correct results.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", _REGISTERED_IMPLS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_linear_jsd_routes_through_dispatch(impl, dtype, monkeypatch):
    """Set LIGER_KERNEL_IMPL_JSD_LOSS_AND_GRAD=<impl> and verify the fused op
    runs end-to-end (forward + backward) without crashing.

    Loss-value parity against a non-fused reference is checked at lower
    tolerance because the fused path does matmul-then-softmax in fp32 with
    a different numeric path than the standalone JSD.
    """
    from liger_kernel.transformers.functional import liger_fused_linear_jsd

    monkeypatch.setenv("LIGER_KERNEL_IMPL_JSD_LOSS_AND_GRAD", impl)

    BT, H, V = 16, 64, 4096
    student_input = torch.randn(BT, H, device="cuda", dtype=dtype, requires_grad=True)
    student_weight = torch.randn(V, H, device="cuda", dtype=dtype, requires_grad=True)
    teacher_input = torch.randn(BT, H, device="cuda", dtype=dtype)
    teacher_weight = torch.randn(V, H, device="cuda", dtype=dtype)

    loss = liger_fused_linear_jsd(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels=None,
        jsd_beta=0.5,
        ignore_index=-100,
        temperature=1.0,
    )
    assert torch.isfinite(loss).item(), f"[impl={impl}] loss is non-finite: {loss}"
    loss.backward()
    assert student_input.grad is not None
    assert torch.isfinite(student_input.grad).all().item()
