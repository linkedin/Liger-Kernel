"""Cross-backend correctness tests for ``fused_add_rms_norm``.

Each backend registered for ``fused_add_rms_norm`` is exercised with the same
shapes and dtypes and compared (forward + backward) against a PyTorch reference.
Tolerances come from each ``OpImpl``'s registered ``tolerances`` table.

Mirrors ``test/ops/test_swiglu.py`` in structure: collects cleanly on a
CPU-only box (the conftest ``autouse`` fixture skips when CUDA is unavailable).
"""

from __future__ import annotations

import math

import pytest
import torch

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered

from .conftest import get_available_backends_for_op

FARN_TEST_SHAPES = [
    (32, 256),
    (128, 1024),
    (4096, 4096),
    (8192, 768),
    (256, 14336),
    (1024, 8192),
]
FARN_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("fused_add_rms_norm")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("fused_add_rms_norm", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _fused_add_rms_norm_ref(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference: Y = RMSNorm(X + R), S = X + R."""
    s = x + r
    s_fp = s.float()
    ms = s_fp.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    y = (s_fp * rstd).to(x.dtype) * w
    return y, s


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FARN_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FARN_TEST_DTYPES)
def test_fused_add_rms_norm_correctness(backend, shape, dtype):
    """Forward + backward parity against the PyTorch reference."""
    if backend == "__none__":
        pytest.skip("No fused_add_rms_norm backends registered in this environment")

    M, N = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    x_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    r_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(N, dtype=torch.float32, generator=g)

    x = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    r = r_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    x_ref = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    r_ref = r_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w_ref = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)

    tols = _tolerances_for(backend, dtype)

    # Forward.
    y, s = dispatch("fused_add_rms_norm", x, r, w, 1e-6, 0.0, "llama", False, backend=backend)
    y_ref, s_ref = _fused_add_rms_norm_ref(x_ref, r_ref, w_ref, 1e-6)

    torch.testing.assert_close(
        y.to(torch.float32),
        y_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[fused_add_rms_norm/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )
    torch.testing.assert_close(
        s.to(torch.float32),
        s_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[fused_add_rms_norm/{backend} shape={shape} dtype={dtype}] residual: {m}",
    )

    # Backward — deterministic upstream gradient.
    g2 = torch.Generator(device="cpu").manual_seed(1)
    dy = torch.randn(M, N, dtype=torch.float32, generator=g2).to(device=device, dtype=dtype)
    ds = torch.randn(M, N, dtype=torch.float32, generator=g2).to(device=device, dtype=dtype)
    torch.autograd.backward((y, s), (dy, ds))
    torch.autograd.backward((y_ref, s_ref), (dy.clone(), ds.clone()))

    torch.testing.assert_close(
        x.grad.to(torch.float32),
        x_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_add_rms_norm/{backend} shape={shape} dtype={dtype}] dx: {m}",
    )
    torch.testing.assert_close(
        r.grad.to(torch.float32),
        r_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_add_rms_norm/{backend} shape={shape} dtype={dtype}] dr: {m}",
    )
    dw_atol = tols["atol_bwd"]
    if dtype == torch.bfloat16:
        # dW is a reduction over M rows; independent bf16 rounding grows
        # approximately with sqrt(M). Keep a bounded per-feature check.
        dw_atol *= max(1.0, math.sqrt(M / 128))
    torch.testing.assert_close(
        w.grad.to(torch.float32),
        w_ref.grad.to(torch.float32),
        atol=dw_atol,
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_add_rms_norm/{backend} shape={shape} dtype={dtype}] dw: {m}",
    )


def test_fused_add_rms_norm_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("fused_add_rms_norm")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"
