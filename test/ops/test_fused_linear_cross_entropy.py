"""Cross-backend correctness tests for ``fused_linear_cross_entropy``.

Each backend registered for ``fused_linear_cross_entropy`` is exercised with
the same shapes and dtypes and compared (forward + backward) against a PyTorch
reference. Tolerances come from each ``OpImpl``'s registered ``tolerances``
table.

Mirrors ``test/ops/test_fused_linear_jsd.py`` in structure: collects cleanly
on a CPU-only box (the conftest ``autouse`` fixture skips when CUDA is
unavailable).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401
import liger_kernel.ops.fused_linear_cross_entropy as flce_ops

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered

from .conftest import get_available_backends_for_op

# (BT, V, H) — V must be a multiple of 8 for CuTe DSL 128-bit vectorized loads.
FLCE_TEST_SHAPES = [
    (32, 256, 1024),
    (64, 512, 768),
    (128, 256, 1024),
]
FLCE_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("fused_linear_cross_entropy")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 5e-3, "rtol_fwd": 1e-3, "atol_bwd": 5e-2, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 1e-2, "atol_bwd": 1e-1, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("fused_linear_cross_entropy", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _flce_ref(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch reference: compute logits then cross-entropy loss."""
    logits = _input.float() @ weight.float().T
    if bias is not None:
        logits = logits + bias.float()
    loss = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    return loss


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FLCE_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FLCE_TEST_DTYPES)
def test_fused_linear_cross_entropy_correctness(backend, shape, dtype):
    """Forward + backward parity against the PyTorch FLCE reference."""
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    BT, V, H = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(42)

    inp_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02
    target_cpu = torch.randint(0, V, (BT,), generator=g)

    inp = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target = target_cpu.to(device=device)

    inp_ref = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w_ref = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target_ref = target_cpu.to(device=device)

    tols = _tolerances_for(backend, dtype)

    # Forward — dispatch returns (loss, z_loss, token_accuracy, predicted_tokens).
    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        w,
        target,
        None,  # bias
        None,  # ce_weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        None,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        backend=backend,
    )
    loss_ref = _flce_ref(inp_ref, w_ref, target_ref)

    torch.testing.assert_close(
        loss.to(torch.float32),
        loss_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )

    # Backward — scalar loss, so backward uses implicit grad=1.
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(
        inp.grad.to(torch.float32),
        inp_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] d_input: {m}",
    )
    torch.testing.assert_close(
        w.grad.to(torch.float32),
        w_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] d_weight: {m}",
    )


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FLCE_TEST_SHAPES[:1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_linear_cross_entropy_with_bias(backend, shape, dtype):
    """FLCE with bias term."""
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    BT, V, H = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(42)

    inp_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02
    b_cpu = torch.randn(V, dtype=torch.float32, generator=g) * 0.01
    target_cpu = torch.randint(0, V, (BT,), generator=g)

    inp = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target = target_cpu.to(device=device)

    inp_ref = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w_ref = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b_ref = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target_ref = target_cpu.to(device=device)

    tols = _tolerances_for(backend, dtype)

    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        w,
        target,
        b,
        backend=backend,
    )
    loss_ref = _flce_ref(inp_ref, w_ref, target_ref, bias=b_ref)

    torch.testing.assert_close(
        loss.to(torch.float32),
        loss_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[FLCE+bias/{backend} shape={shape}] forward: {m}",
    )

    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(
        b.grad.to(torch.float32),
        b_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[FLCE+bias/{backend} shape={shape}] d_bias: {m}",
    )


def test_fused_linear_cross_entropy_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("fused_linear_cross_entropy")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_fused_linear_cross_entropy_propagates_backend_to_inner_ce(monkeypatch, backend):
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    observed_impls = []
    real_dispatch = flce_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "cross_entropy_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(flce_ops, "dispatch", tracking_dispatch)

    inp = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, 256, (8,), device="cuda")

    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        weight,
        target,
        backend=backend,
    )
    loss.backward()

    assert observed_impls
    assert set(observed_impls) == {backend}
