"""Cross-backend correctness tests for ``geglu``.

Each backend registered for ``geglu`` is exercised with the same shapes and
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

GEGLU_TEST_SHAPES = [
    (32, 256),
    (128, 1024),
    (4096, 4096),
    (8192, 768),
    (256, 14336),
    (1024, 8192),
]
GEGLU_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("geglu")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("geglu", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _geglu_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: gelu_tanh(a) * b."""
    return (torch.nn.functional.gelu(a.float(), approximate="tanh") * b.float()).to(a.dtype)


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", GEGLU_TEST_SHAPES)
@pytest.mark.parametrize("dtype", GEGLU_TEST_DTYPES)
def test_geglu_correctness(backend, shape, dtype):
    """Forward + backward parity against the PyTorch GeGLU reference."""
    if backend == "__none__":
        pytest.skip("No geglu backends registered in this environment")

    M, N = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    a_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    b_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    a = a_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    a_ref = a_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b_ref = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)

    tols = _tolerances_for(backend, dtype)

    # Forward.
    y = dispatch("geglu", a, b, backend=backend)
    y_ref = _geglu_ref(a_ref, b_ref)

    torch.testing.assert_close(
        y.to(torch.float32),
        y_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[geglu/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )

    # Backward — deterministic upstream gradient.
    g2 = torch.Generator(device="cpu").manual_seed(1)
    dy = torch.randn(M, N, dtype=torch.float32, generator=g2).to(device=device, dtype=dtype)
    y.backward(dy)
    y_ref.backward(dy.clone())

    torch.testing.assert_close(
        a.grad.to(torch.float32),
        a_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[geglu/{backend} shape={shape} dtype={dtype}] da: {m}",
    )
    torch.testing.assert_close(
        b.grad.to(torch.float32),
        b_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[geglu/{backend} shape={shape} dtype={dtype}] db: {m}",
    )


def test_geglu_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("geglu")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"
