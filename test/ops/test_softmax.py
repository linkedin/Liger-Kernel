"""Cross-backend correctness tests for ``softmax``.

Each backend registered for ``softmax`` is exercised with the same shapes and
dtypes and compared (forward + backward) against ``torch.softmax``. Tolerances
come from each ``OpImpl``'s registered ``tolerances`` table.

Softmax has a simpler signature than rms_norm / layer_norm (a single input
tensor, no weight), so this file does not use the ``assert_op_correctness``
driver (which is wired for the norm signatures); it does the fwd/bwd comparison
inline. The collection / skip structure mirrors ``test_rms_norm.py``: it
collects cleanly on a CPU-only box (the conftest ``autouse`` fixture skips when
CUDA is unavailable, and the placeholder ``"__none__"`` keeps parametrization
valid when no backends are registered).
"""

from __future__ import annotations

import os

import pytest
import torch

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered

from .conftest import get_available_backends_for_op

SOFTMAX_TEST_SHAPES = [
    (32, 256),  # tiny — cuTile auto-picks standard
    (128, 1024),  # small
    (4096, 4096),  # large M -> static_persistent
    (8192, 768),  # non-pow2 hidden, large M
    (256, 18432),  # very wide hidden -> chunked (cuTile) / cuTeDSL range-capped
    (1024, 8192),  # mid
]
SOFTMAX_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("softmax")

# Default tolerance fallbacks if an impl didn't register a dtype entry.
_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("softmax", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", SOFTMAX_TEST_SHAPES)
@pytest.mark.parametrize("dtype", SOFTMAX_TEST_DTYPES)
def test_softmax_correctness(backend, shape, dtype):
    """Forward + backward parity against ``torch.softmax``.

    Auto-skips when CUDA is unavailable (conftest fixture) and when no backends
    are registered (the placeholder ``"__none__"`` parameter keeps collection
    valid in that edge case). Backends that document a hidden-dim range limit
    raise ``RuntimeError`` for wide rows; we treat that as a clean skip.
    """
    if backend == "__none__":
        pytest.skip("No softmax backends registered in this environment")

    M, N = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    x_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    x = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    x_ref = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)

    tols = _tolerances_for(backend, dtype)

    # Forward.
    try:
        y = dispatch("softmax", x, backend=backend)
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend {backend} documents range limit: {e}")
        raise

    y_ref = torch.softmax(x_ref.to(torch.float32), dim=-1).to(dtype)

    torch.testing.assert_close(
        y.to(torch.float32),
        y_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[softmax/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )

    # Backward — deterministic upstream gradient.
    g2 = torch.Generator(device="cpu").manual_seed(1)
    dy = torch.randn(M, N, dtype=torch.float32, generator=g2).to(device=device, dtype=dtype)
    try:
        y.backward(dy)
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend {backend} documents range limit: {e}")
        raise
    y_ref.backward(dy.clone())

    torch.testing.assert_close(
        x.grad.to(torch.float32),
        x_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[softmax/{backend} shape={shape} dtype={dtype}] dx: {m}",
    )


def test_softmax_available_backends_includes_triton():
    """Sanity: in any normal install the Triton implementation should be available.

    Accepts both the new hyphenated ``"nvidia-triton"`` and the legacy bare
    ``"triton"`` form (the dispatcher accepts either).
    """
    impls = available_backends("softmax")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_softmax_global_set_backend(backend):
    """``liger_kernel.set_backend(name)`` pins dispatch to ``name`` when no
    explicit backend is passed; ``set_backend(None)`` restores auto."""
    if backend == "__none__":
        pytest.skip("No softmax backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)

    try:
        liger_kernel.set_backend(backend)
        y_pinned = dispatch("softmax", x)
        y_explicit = dispatch("softmax", x, backend=backend)
        torch.testing.assert_close(y_pinned, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        liger_kernel.set_backend(None)

    assert liger_kernel.get_backend() is None


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_softmax_env_per_op(backend):
    """``LIGER_KERNEL_BACKEND_SOFTMAX=<backend>`` overrides auto-select."""
    if backend == "__none__":
        pytest.skip("No softmax backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)

    saved = os.environ.get("LIGER_KERNEL_BACKEND_SOFTMAX")
    try:
        os.environ["LIGER_KERNEL_BACKEND_SOFTMAX"] = backend
        y_env = dispatch("softmax", x)
        y_explicit = dispatch("softmax", x, backend=backend)
        torch.testing.assert_close(y_env, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        if saved is None:
            os.environ.pop("LIGER_KERNEL_BACKEND_SOFTMAX", None)
        else:
            os.environ["LIGER_KERNEL_BACKEND_SOFTMAX"] = saved


# ---------------------------------------------------------------------------
# Exercise every explicit mode advertised by each backend (cuTile has three).
# ---------------------------------------------------------------------------
_MODE_BY_BACKEND = {
    "nvidia-triton": ["default"],
    "nvidia-cutile": ["standard", "static_persistent", "chunked"],
    "nvidia-cutedsl": ["default"],
    "triton": ["default"],
    "cutile": ["standard", "static_persistent", "chunked"],
    "cutedsl": ["default"],
}
_MODE_SHAPES = [(32, 256), (256, 4096), (4096, 4096)]


@pytest.mark.parametrize(
    "backend,mode",
    [(b, m) for b in _REGISTERED_BACKENDS or ["__none__"] for m in _MODE_BY_BACKEND.get(b, ["default"])],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
@pytest.mark.parametrize("shape", _MODE_SHAPES, ids=lambda v: "x".join(str(d) for d in v))
def test_softmax_explicit_mode(backend, mode, shape):
    """Every explicit mode advertised by a backend must produce correct output."""
    if backend == "__none__":
        pytest.skip("No softmax backends registered")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    M, N = shape
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    try:
        y = liger_kernel.functional.softmax(x, backend=backend, mode=mode)
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend documents range limit: {e}")
        raise

    y_ref = torch.softmax(x_ref.to(torch.float32), dim=-1).to(torch.bfloat16)
    assert torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), atol=2e-2, rtol=2e-2), (
        f"[{backend}/{mode} shape={shape}] forward max diff "
        f"{(y.to(torch.float32) - y_ref.to(torch.float32)).abs().max().item():.4f}"
    )

    # Backward sanity — verify no exception and dx is finite.
    dy = torch.randn_like(y)
    try:
        y.backward(dy)
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend documents range limit: {e}")
        raise
    assert torch.isfinite(x.grad).all(), f"non-finite dx on {backend}/{mode}/{shape}"
