"""Cross-backend correctness tests for ``kl_div``.

Each backend registered for ``kl_div`` is exercised with the same shapes and
dtypes and compared (forward + backward) against a PyTorch reference. Tolerances
come from each ``OpImpl``'s registered ``tolerances`` table.

Mirrors ``test/ops/test_swiglu.py`` in structure: collects cleanly on a
CPU-only box (the conftest ``autouse`` fixture skips when CUDA is unavailable).
"""

from __future__ import annotations

import pytest
import torch

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered

from .conftest import get_available_backends_for_op

KLDIV_TEST_SHAPES = [
    (32, 256),
    (128, 1024),
    (256, 4096),
    (8192, 768),
]
KLDIV_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("kl_div")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("kl_div", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _kl_div_ref(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str, log_target: bool) -> torch.Tensor:
    """PyTorch reference using torch.nn.functional.kl_div."""
    return torch.nn.functional.kl_div(y_pred.float(), y_true.float(), reduction=reduction, log_target=log_target)


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", KLDIV_TEST_SHAPES)
@pytest.mark.parametrize("dtype", KLDIV_TEST_DTYPES)
@pytest.mark.parametrize("log_target", [False, True])
def test_kl_div_correctness(backend, shape, dtype, log_target):
    """Forward + backward parity against the PyTorch KL-div reference."""
    if backend == "__none__":
        pytest.skip("No kl_div backends registered in this environment")

    M, N = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)

    # Generate probabilities for the target.
    if log_target:
        y_true_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    else:
        probs = torch.softmax(torch.randn(M, N, dtype=torch.float32, generator=g), dim=-1)
        y_true_cpu = probs

    y_pred_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)

    y_pred = y_pred_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    y_true = y_true_cpu.to(device=device, dtype=dtype).detach()
    y_pred_ref = y_pred_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    y_true_ref = y_true_cpu.to(device=device, dtype=dtype).detach()

    tols = _tolerances_for(backend, dtype)

    # Forward.
    y = dispatch("kl_div", y_pred, y_true, "batchmean", log_target, 1e-10, backend=backend)
    y_ref = _kl_div_ref(y_pred_ref, y_true_ref, "batchmean", log_target)

    torch.testing.assert_close(
        y.to(torch.float32),
        y_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[kl_div/{backend} shape={shape} dtype={dtype} log_target={log_target}] forward: {m}",
    )

    # Backward — deterministic upstream gradient (scalar loss → unit grad).
    y.backward()
    y_ref.backward()

    torch.testing.assert_close(
        y_pred.grad.to(torch.float32),
        y_pred_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[kl_div/{backend} shape={shape} dtype={dtype} log_target={log_target}] dy_pred: {m}",
    )


def test_kl_div_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("kl_div")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


def test_kl_div_reduction_none_cutedsl_matches_triton():
    if "nvidia-cutedsl" not in _REGISTERED_BACKENDS:
        pytest.skip("CuTe DSL kl_div is unavailable")

    y_pred = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y_true = torch.softmax(torch.randn_like(y_pred), dim=-1)
    y_pred_ref = y_pred.detach().clone().requires_grad_(True)

    actual = dispatch(
        "kl_div",
        y_pred,
        y_true,
        "none",
        False,
        1e-10,
        backend="nvidia-cutedsl",
    )
    expected = dispatch(
        "kl_div",
        y_pred_ref,
        y_true,
        "none",
        False,
        1e-10,
        backend="nvidia-triton",
    )

    assert actual.shape == y_pred.shape
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(y_pred.grad, y_pred_ref.grad)


@pytest.mark.parametrize("eps", [1e-2, 5e-2])
def test_kl_div_cutedsl_honors_nondefault_eps(eps):
    if "nvidia-cutedsl" not in _REGISTERED_BACKENDS:
        pytest.skip("CuTe DSL kl_div is unavailable")

    y_pred = torch.log_softmax(
        torch.randn(8, 256, device="cuda", dtype=torch.bfloat16),
        dim=-1,
    ).requires_grad_(True)
    y_true = torch.full_like(y_pred, 1e-3)
    y_true[:, ::16] = 1e-1
    y_pred_ref = y_pred.detach().clone().requires_grad_(True)

    actual = dispatch(
        "kl_div",
        y_pred,
        y_true,
        "batchmean",
        False,
        eps,
        backend="nvidia-cutedsl",
    )
    expected = dispatch(
        "kl_div",
        y_pred_ref,
        y_true,
        "batchmean",
        False,
        eps,
        backend="nvidia-triton",
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(y_pred.grad, y_pred_ref.grad, atol=2e-2, rtol=2e-2)
