"""Cross-backend correctness tests for ``fused_linear_jsd``.

Each backend registered for ``fused_linear_jsd`` is exercised with the same
shapes and dtypes and compared (forward + backward) against a PyTorch reference.
Tolerances come from each ``OpImpl``'s registered ``tolerances`` table.

Mirrors ``test/ops/test_swiglu.py`` in structure: collects cleanly on a
CPU-only box (the conftest ``autouse`` fixture skips when CUDA is unavailable).
"""

from __future__ import annotations

import pytest
import torch

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401
import liger_kernel.ops.fused_linear_jsd as fljsd_ops

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered

from .conftest import get_available_backends_for_op

FLJSD_TEST_SHAPES = [
    (32, 256, 1024),  # (BT, V, H)
    (64, 512, 768),
    (128, 256, 1024),
]
FLJSD_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("fused_linear_jsd")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 5e-3, "rtol_fwd": 1e-3, "atol_bwd": 5e-2, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 1e-2, "atol_bwd": 1e-1, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("fused_linear_jsd", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _fused_linear_jsd_ref(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """PyTorch reference: compute logits, log_softmax, then JSD loss."""
    student_logits = student_input.float() @ student_weight.float().T
    teacher_logits = teacher_input.float() @ teacher_weight.float().T

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    # Pure FKL/RKL (beta in {0, 1}) reduce to a plain KL and are computed that
    # way directly: the generic probability-space mixture below can underflow
    # to exactly 0 for peaked distributions at large logit magnitudes (0 *
    # log(0) -> nan) -- a separate bug from the one under test here (see
    # linkedin/Liger-Kernel#1432's "out of scope" section). This mirrors
    # test/transformers/test_jsd.py's JSD.forward reference.
    if beta == 0.0:  # JSD(0) == KL(teacher || student)
        loss = torch.nn.functional.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(
            dim=-1
        )
    elif beta == 1.0:  # JSD(1) == KL(student || teacher)
        loss = torch.nn.functional.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(
            dim=-1
        )
    else:
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()
        m = beta * student_probs + (1.0 - beta) * teacher_probs
        log_m = m.log()
        # JSD = beta * KL(student || m) + (1 - beta) * KL(teacher || m)
        loss = beta * (student_probs * (student_log_probs - log_m)).sum(dim=-1)
        loss += (1.0 - beta) * (teacher_probs * (teacher_log_probs - log_m)).sum(dim=-1)
    return loss.mean()


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FLJSD_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FLJSD_TEST_DTYPES)
def test_fused_linear_jsd_correctness(backend, shape, dtype):
    """Forward + backward parity against the PyTorch fused_linear_jsd reference."""
    if backend == "__none__":
        pytest.skip("No fused_linear_jsd backends registered in this environment")

    BT, V, H = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)

    si_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    sw_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02
    ti_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    tw_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02

    si = si_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    sw = sw_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    ti = ti_cpu.to(device=device, dtype=dtype).detach()
    tw = tw_cpu.to(device=device, dtype=dtype).detach()

    si_ref = si_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    sw_ref = sw_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    ti_ref = ti_cpu.to(device=device, dtype=dtype).detach()
    tw_ref = tw_cpu.to(device=device, dtype=dtype).detach()

    tols = _tolerances_for(backend, dtype)

    # Forward.
    y = dispatch(
        "fused_linear_jsd",
        si,
        sw,
        ti,
        tw,
        None,
        0.5,
        -100,
        1.0,
        backend=backend,
    )
    y_ref = _fused_linear_jsd_ref(si_ref, sw_ref, ti_ref, tw_ref, beta=0.5)

    torch.testing.assert_close(
        y.to(torch.float32),
        y_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[fused_linear_jsd/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )

    # Backward — scalar loss, so backward uses implicit grad=1.
    y.backward()
    y_ref.backward()

    torch.testing.assert_close(
        si.grad.to(torch.float32),
        si_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_jsd/{backend} shape={shape} dtype={dtype}] d_student_input: {m}",
    )
    torch.testing.assert_close(
        sw.grad.to(torch.float32),
        sw_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_jsd/{backend} shape={shape} dtype={dtype}] d_student_weight: {m}",
    )


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [0.0, 1.0])  # FKL / RKL; see note below on 0 < beta < 1
def test_fused_linear_jsd_correctness_realistic_logit_scale(backend, dtype, beta):
    """Regression test for linkedin/Liger-Kernel#1432.

    ``FLJSD_TEST_SHAPES``/``* 0.02`` weight scaling above keeps logits small
    enough (std ~ a few) that casting a rounded bf16/fp16 matmul result to
    FP32 -- instead of projecting in FP32 -- doesn't move the gradient beyond
    the existing (loose) bwd tolerances. At a realistic logit spread (std
    ~30, as in real LM heads) the same bug produces 4-23% gradient error, so
    this test uses a wider weight scale and a tolerance tight enough to catch
    a reintroduction of the "cast after matmul" bug.

    Only pure FKL/RKL (beta in {0, 1}) are exercised here: 0 < beta < 1 at
    this logit scale hits a separate, independent bug (the probability-space
    mixture underflows to exactly 0 for peaked distributions, giving
    log(0) -> nan) tracked in issue #1432's "out of scope" section, and is
    not this fix's concern.
    """
    if backend == "__none__":
        pytest.skip("No fused_linear_jsd backends registered in this environment")

    BT, V, H = 256, 32000, 1024
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)

    logit_std = 30.0
    si_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    sw_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * (logit_std / H**0.5)
    ti_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    tw_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * (logit_std / H**0.5)

    si = si_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    sw = sw_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    ti = ti_cpu.to(device=device, dtype=dtype).detach()
    tw = tw_cpu.to(device=device, dtype=dtype).detach()

    si_ref = si_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    sw_ref = sw_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    ti_ref = ti_cpu.to(device=device, dtype=dtype).detach()
    tw_ref = tw_cpu.to(device=device, dtype=dtype).detach()

    # The bug (cast-after-matmul) produces 4-23% relative gradient error at
    # this scale; the fix (FP32 projection) brings it back under ~1%.
    rtol_bwd = 3e-2

    y = dispatch("fused_linear_jsd", si, sw, ti, tw, None, beta, -100, 1.0, backend=backend)
    y_ref = _fused_linear_jsd_ref(si_ref, sw_ref, ti_ref, tw_ref, beta=beta)

    y.backward()
    y_ref.backward()

    torch.testing.assert_close(
        si.grad.to(torch.float32),
        si_ref.grad.to(torch.float32),
        atol=1e-2,
        rtol=rtol_bwd,
        msg=lambda m: f"[fused_linear_jsd/{backend} dtype={dtype} beta={beta}] d_student_input: {m}",
    )
    torch.testing.assert_close(
        sw.grad.to(torch.float32),
        sw_ref.grad.to(torch.float32),
        atol=1e-2,
        rtol=rtol_bwd,
        msg=lambda m: f"[fused_linear_jsd/{backend} dtype={dtype} beta={beta}] d_student_weight: {m}",
    )


def test_fused_linear_jsd_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("fused_linear_jsd")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_fused_linear_jsd_propagates_backend_to_inner_jsd(monkeypatch, backend):
    if backend == "__none__":
        pytest.skip("No fused_linear_jsd backends registered in this environment")

    observed_impls = []
    real_dispatch = fljsd_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "jsd_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(fljsd_ops, "dispatch", tracking_dispatch)

    student_input = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    student_weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher_input = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    teacher_weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)

    loss = dispatch(
        "fused_linear_jsd",
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        backend=backend,
    )
    loss.backward()

    assert observed_impls
    assert set(observed_impls) == {backend}


def test_fused_linear_jsd_default_path_dispatches_inner_jsd(monkeypatch):
    observed_impls = []
    real_dispatch = fljsd_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "jsd_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(fljsd_ops, "dispatch", tracking_dispatch)

    student_input = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    student_weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher_input = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    teacher_weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)

    fljsd_ops.LigerFusedLinearJSDFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
    ).backward()

    assert observed_impls
    assert set(observed_impls) == {None}
