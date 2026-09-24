"""Cross-backend correctness tests for ``layer_norm``.

Each backend registered for ``layer_norm`` is exercised with the same set of
shapes, dtypes, and bias modes. Tolerances come from each ``OpImpl``'s
registered ``tolerances`` table — adding a new backend means adding entries
there, not editing this file.
"""

from __future__ import annotations

import os

from unittest import mock

import pytest
import torch

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends.capability import Capability
from liger_kernel.backends.dispatch import BackendNotAvailableError
from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import register_op
from liger_kernel.testing import assert_op_correctness

from .conftest import get_available_backends_for_op

LAYER_NORM_TEST_SHAPES = [
    (32, 256),  # tiny
    (128, 1024),  # small
    (4096, 4096),  # Llama-7B-ish hidden, seq=4096 batch=1
    (8192, 768),  # non-pow2 hidden, large M
    (256, 18432),  # very wide hidden (cuTile bwd will skip on >8192)
    (1024, 8192),  # mid
]
LAYER_NORM_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


_REGISTERED_BACKENDS = get_available_backends_for_op("layer_norm")


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", LAYER_NORM_TEST_SHAPES)
@pytest.mark.parametrize("dtype", LAYER_NORM_TEST_DTYPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_layer_norm_correctness(backend, shape, dtype, with_bias):
    """Forward + backward parity against the PyTorch reference.

    Auto-skips when CUDA is unavailable (conftest fixture). Auto-skips when
    no backends are registered at collection time (the placeholder
    ``"__none__"`` parameter is only used to keep collection valid in that
    edge case).
    """
    if backend == "__none__":
        pytest.skip("No layer_norm backends registered in this environment")
    try:
        assert_op_correctness(
            "layer_norm",
            backend,
            shape,
            dtype,
            extra={"include_bias": with_bias},
        )
    except RuntimeError as e:
        # See test_rms_norm.py: honour a backend's documented hidden-dim range
        # limit (cuTile caps forward and backward at 8192) as a clean skip.
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend {backend} documents range limit: {e}")
        raise


def test_layer_norm_explicit_backend_unavailable():
    """Requesting a backend whose capability is unsatisfied must raise
    ``BackendNotAvailableError`` whose message names the unsatisfied gate.
    """
    fake_backend = "test_fake_cutile_unavailable_layer_norm"

    @register_op(
        "layer_norm",
        backend=fake_backend,
        capability=Capability(min_cc=(10, 0)),  # Blackwell-only
        preference_rank=1,
    )
    def _fake(x, *args, **kwargs):  # pragma: no cover — never invoked
        return x

    try:
        # Pretend we're on H100/H200 (sm_90).
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with pytest.raises(BackendNotAvailableError) as ei:
                dispatch(
                    "layer_norm",
                    torch.zeros(4, 8, device="cuda"),
                    torch.zeros(8, device="cuda"),
                    torch.zeros(8, device="cuda"),
                    1e-6,
                    backend=fake_backend,
                )
            msg = str(ei.value)
            assert fake_backend in msg
            assert "sm_90" in msg
            assert "sm_100" in msg
    finally:
        # Clean up the test-only registration.
        from liger_kernel.backends.registry import _REGISTRY

        _REGISTRY.pop(("layer_norm", fake_backend), None)


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_layer_norm_global_set_backend(backend):
    """``liger_kernel.set_backend(name)`` pins dispatch to ``name`` when no
    explicit backend is passed; ``set_backend(None)`` restores auto."""
    if backend == "__none__":
        pytest.skip("No layer_norm backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.ones(N, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    try:
        liger_kernel.set_backend(backend)
        # No explicit backend kwarg — must use the pinned one.
        y_pinned = dispatch("layer_norm", x, w, b, 1e-6)
        y_explicit = dispatch(
            "layer_norm",
            x,
            w,
            b,
            1e-6,
            backend=backend,
        )
        assert torch.allclose(y_pinned, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        liger_kernel.set_backend(None)

    assert liger_kernel.get_backend() is None


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_layer_norm_env_per_op(backend):
    """``LIGER_KERNEL_BACKEND_LAYER_NORM=<backend>`` overrides auto-select."""
    if backend == "__none__":
        pytest.skip("No layer_norm backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.ones(N, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    saved = os.environ.get("LIGER_KERNEL_BACKEND_LAYER_NORM")
    try:
        os.environ["LIGER_KERNEL_BACKEND_LAYER_NORM"] = backend
        y_env = dispatch("layer_norm", x, w, b, 1e-6)
        y_explicit = dispatch(
            "layer_norm",
            x,
            w,
            b,
            1e-6,
            backend=backend,
        )
        assert torch.allclose(y_env, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        if saved is None:
            os.environ.pop("LIGER_KERNEL_BACKEND_LAYER_NORM", None)
        else:
            os.environ["LIGER_KERNEL_BACKEND_LAYER_NORM"] = saved


def test_layer_norm_available_backends_includes_triton():
    """Sanity: in any normal install the Triton implementation should be available.

    The current dispatcher exposes the impl as ``"nvidia-triton"``; the test
    accepts the bare ``"triton"`` legacy alias too.
    """
    impls = available_backends("layer_norm")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


# ---------------------------------------------------------------------------
# Explicit-mode coverage for the cuTile LayerNorm kernel. LayerNorm has only
# one casting model (fp32 reduction); cuTile registers 3 modes:
#   standard         — one row per program, no persistence
#   static_persistent — single-row persistent (the multi-row variant was
#                       removed: pathological tileiras compile time)
#   multi_wave_cached — register-cached weight, narrow rows
# Verify each produces correct output across a handful of shapes.
# ---------------------------------------------------------------------------


def _layer_norm_modes_by_backend(backend):
    # Accept both the canonical hyphenated impl name and the bare legacy alias.
    if backend in ("nvidia-cutile", "cutile"):
        # Matches the cuTile impl's registered modes — see
        # ``ops/backends/_cutile/layer_norm.py`` ``modes=`` declaration.
        return ["standard", "static_persistent", "multi_wave_cached"]
    return ["default"]


_LN_MODE_SHAPES = [(32, 256), (256, 4096), (4096, 4096), (127, 4096)]
# (127, 4096) deliberately not divisible by TILE_SIZE_M=16; tests the host
# launcher's automatic multi-row -> singlerow downgrade path.


@pytest.mark.parametrize(
    "backend,mode",
    [(b, m) for b in _REGISTERED_BACKENDS or ["__none__"] for m in _layer_norm_modes_by_backend(b)],
)
@pytest.mark.parametrize("shape", _LN_MODE_SHAPES)
def test_layer_norm_explicit_mode(backend, mode, shape):
    """Every explicit mode advertised by a backend must produce correct
    output, including on shapes that exercise the multi-row → singlerow
    fallback path in the host launcher."""
    if backend == "__none__":
        pytest.skip("No layer_norm backends registered")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    M, N = shape
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)
    w = torch.randn(N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)
    b = torch.randn(N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)

    try:
        y = liger_kernel.functional.layer_norm(
            x,
            w,
            b,
            1e-6,
            backend=backend,
            mode=mode,
        )
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend documents range limit: {e}")
        raise
    except ValueError as e:
        # Some explicit modes may not exist on every backend (e.g. cutedsl
        # doesn't advertise multi_wave_cached). Skip those gracefully.
        if "unknown mode" in str(e).lower() or "valid modes" in str(e).lower():
            pytest.skip(f"mode={mode!r} not supported by backend={backend!r}: {e}")
        raise

    # PyTorch reference in fp32.
    x_ref = x.detach().clone().to(torch.float32)
    w_ref = w.detach().clone().to(torch.float32)
    b_ref = b.detach().clone().to(torch.float32)
    y_ref = (torch.nn.functional.layer_norm(x_ref, (N,), w_ref, b_ref, 1e-6)).to(torch.bfloat16)

    max_diff = (y - y_ref).abs().max().item()
    # Tolerance relaxed slightly vs the auto-select test because explicit
    # mode may hit a path with different rounding (multi-row vs singlerow).
    assert max_diff < 5e-2, f"[{backend}/{mode} shape={shape}] forward max_diff={max_diff:.4f}"
