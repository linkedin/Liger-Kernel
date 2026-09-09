"""Cross-backend correctness tests for ``rms_norm``.

Each backend registered for ``rms_norm`` is exercised with the same set of
shapes, dtypes, and casting modes. Tolerances come from each ``OpImpl``'s
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
from liger_kernel.testing import pytorch_reference_rms_norm

from .conftest import get_available_backends_for_op

RMS_NORM_TEST_SHAPES = [
    (32, 256),  # tiny
    (128, 1024),  # small
    (4096, 4096),  # Llama-7B-ish hidden, seq=4096 batch=1
    (8192, 768),  # non-pow2 hidden, large M
    (256, 18432),  # very wide hidden (Mixtral-ish)
    (1024, 8192),  # mid
]
RMS_NORM_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
RMS_NORM_TEST_CASTING_MODES = ["llama", "gemma", "none"]


_REGISTERED_BACKENDS = get_available_backends_for_op("rms_norm")


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", RMS_NORM_TEST_SHAPES)
@pytest.mark.parametrize("dtype", RMS_NORM_TEST_DTYPES)
@pytest.mark.parametrize("casting_mode", RMS_NORM_TEST_CASTING_MODES)
def test_rms_norm_correctness(backend, shape, dtype, casting_mode):
    """Forward + backward parity against the PyTorch reference.

    Auto-skips when CUDA is unavailable (conftest fixture). Auto-skips when
    no backends are registered at collection time (the placeholder
    ``"__none__"`` parameter is only used to keep collection valid in that
    edge case).
    """
    if backend == "__none__":
        pytest.skip("No rms_norm backends registered in this environment")
    assert_op_correctness(
        "rms_norm",
        backend,
        shape,
        dtype,
        casting_mode=casting_mode,
    )


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_rms_norm_offset_gemma(backend):
    """Verify ``offset=1.0`` (Gemma's ``(1 + w)`` trick) is applied correctly.

    We compare two reference computations:
      - Gemma-style: ``offset=1.0``, weight values around 0.
      - Llama-style: ``offset=0.0``, weight = original_weight + 1.

    Both should produce the same output. Then we check the backend dispatches
    to the same answer as the Gemma reference.
    """
    if backend == "__none__":
        pytest.skip("No rms_norm backends registered")
    torch.manual_seed(0)
    M, N = 64, 1024
    dtype = torch.float32
    device = "cuda"
    x = torch.randn(M, N, dtype=dtype, device=device, requires_grad=True)
    w_small = torch.randn(N, dtype=dtype, device=device, requires_grad=False) * 0.01

    y_kernel = dispatch(
        "rms_norm",
        x,
        w_small.clone().requires_grad_(True),
        1e-6,
        1.0,
        "gemma",
        False,
        None,
        backend=backend,
    )
    y_ref = pytorch_reference_rms_norm(x, w_small, 1e-6, offset=1.0, casting_mode="gemma")
    assert torch.allclose(y_kernel, y_ref, atol=1e-4, rtol=1e-4), f"backend {backend} failed gemma offset test"

    # Sanity: (offset=1, w) ≡ (offset=0, 1+w) under "gemma" mode.
    y_alt = pytorch_reference_rms_norm(x, w_small + 1.0, 1e-6, offset=0.0, casting_mode="gemma")
    assert torch.allclose(y_ref, y_alt, atol=1e-4, rtol=1e-4), (
        "reference inconsistency: offset=1 should equal weight+1 under gemma"
    )


def test_rms_norm_explicit_backend_unavailable():
    """Requesting a backend whose capability is unsatisfied must raise
    ``BackendNotAvailableError`` whose message names the unsatisfied gate.
    """
    fake_backend = "test_fake_cutile_unavailable"

    @register_op(
        "rms_norm",
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
                    "rms_norm",
                    torch.zeros(4, 8, device="cuda"),
                    torch.zeros(8, device="cuda"),
                    1e-6,
                    0.0,
                    "llama",
                    False,
                    None,
                    backend=fake_backend,
                )
            msg = str(ei.value)
            assert fake_backend in msg
            assert "sm_90" in msg
            assert "sm_100" in msg
    finally:
        # Clean up the test-only registration.
        from liger_kernel.backends.registry import _REGISTRY

        _REGISTRY.pop(("rms_norm", fake_backend), None)


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_rms_norm_global_set_backend(backend):
    """``liger_kernel.set_backend(name)`` pins dispatch to ``name`` when no
    explicit backend is passed; ``set_backend(None)`` restores auto."""
    if backend == "__none__":
        pytest.skip("No rms_norm backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    try:
        liger_kernel.set_backend(backend)
        # No explicit backend kwarg — must use the pinned one.
        y_pinned = dispatch("rms_norm", x, w, 1e-6, 0.0, "llama", False, None)
        y_explicit = dispatch(
            "rms_norm",
            x,
            w,
            1e-6,
            0.0,
            "llama",
            False,
            None,
            backend=backend,
        )
        assert torch.allclose(y_pinned, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        liger_kernel.set_backend(None)

    assert liger_kernel.get_backend() is None


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_rms_norm_env_per_op(backend):
    """``LIGER_KERNEL_BACKEND_RMS_NORM=<backend>`` overrides auto-select."""
    if backend == "__none__":
        pytest.skip("No rms_norm backends registered")

    M, N = 32, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    saved = os.environ.get("LIGER_KERNEL_BACKEND_RMS_NORM")
    try:
        os.environ["LIGER_KERNEL_BACKEND_RMS_NORM"] = backend
        y_env = dispatch("rms_norm", x, w, 1e-6, 0.0, "llama", False, None)
        y_explicit = dispatch(
            "rms_norm",
            x,
            w,
            1e-6,
            0.0,
            "llama",
            False,
            None,
            backend=backend,
        )
        assert torch.allclose(y_env, y_explicit, atol=1e-6, rtol=1e-6)
    finally:
        if saved is None:
            os.environ.pop("LIGER_KERNEL_BACKEND_RMS_NORM", None)
        else:
            os.environ["LIGER_KERNEL_BACKEND_RMS_NORM"] = saved


def test_rms_norm_available_backends_includes_triton():
    """Sanity: in any normal install the Triton implementation should be available.

    The current dispatcher exposes the impl as ``"nvidia-triton"``; the test
    accepts the bare ``"triton"`` legacy alias too (still resolved via
    :data:`_LEGACY_ALIASES`).
    """
    impls = available_backends("rms_norm")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


# ---------------------------------------------------------------------------
# Exercise EVERY explicit mode per backend, not just the auto-select path.
# The multi-row static_persistent kernel only kicks in for M >= NUM_SMS*8;
# a singlerow fallback exists. Cover both.
# ---------------------------------------------------------------------------


_MODE_BY_BACKEND = {
    # Canonical hyphenated names
    "nvidia-triton": ["default"],
    "nvidia-cutile": ["standard", "static_persistent", "multi_wave_cached"],
    "nvidia-cutedsl": ["default"],
    # Legacy back-compat aliases (the dispatcher still accepts these)
    "triton": ["default"],
    "cutile": ["standard", "static_persistent", "multi_wave_cached"],
    "cutedsl": ["default"],
}

# Shapes chosen to exercise the perf-push code paths:
#   (32, 256)    — auto-picks multi_wave_cached; tests its forward
#   (256, 4096)  — auto-picks standard
#   (4096, 4096) — M >= NUM_SMS*8 on B200 -> multi-row static_persistent
#   (256, 4096)  — also exercises explicit static_persistent (singlerow fallback)
_MODE_SHAPES = [(32, 256), (256, 4096), (4096, 4096)]


def _mode_param_id(val):
    return val if isinstance(val, str) else "-".join(str(x) for x in val)


@pytest.mark.parametrize(
    "backend,mode",
    [(b, m) for b in _REGISTERED_BACKENDS or ["__none__"] for m in _MODE_BY_BACKEND.get(b, ["default"])],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
@pytest.mark.parametrize("shape", _MODE_SHAPES, ids=_mode_param_id)
def test_rms_norm_explicit_mode(backend, mode, shape):
    """Every explicit mode advertised by a backend must produce correct
    output. Auto-select tests already cover the heuristic; this catches
    regressions in the rarely-picked paths (e.g. single-row static_persistent
    on a shape where multi-row would normally win)."""
    if backend == "__none__":
        pytest.skip("No rms_norm backends registered")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    # Run a llama-mode bf16 forward+backward with the explicit mode pinned.
    M, N = shape
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)
    w = torch.randn(N, dtype=torch.bfloat16, generator=g).to("cuda").requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    try:
        y = liger_kernel.functional.rms_norm(
            x,
            w,
            1e-6,
            casting_mode="llama",
            backend=backend,
            mode=mode,
        )
    except RuntimeError as e:
        # Backends can refuse out-of-range configs (e.g. cuTile bwd > 8K hidden).
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend documents range limit: {e}")
        raise

    y_ref = pytorch_reference_rms_norm(x_ref, w_ref, 1e-6, casting_mode="llama")
    # Tolerances mirror the impl's declared bfloat16 entry. Slightly relaxed
    # for the explicit-mode path because non-default modes may have larger
    # rounding (e.g. multi_wave_cached vs static_persistent diff is ~1 ULP).
    assert torch.allclose(y, y_ref, atol=2e-2, rtol=2e-2), (
        f"[{backend}/{mode} shape={shape}] forward max diff {(y - y_ref).abs().max().item():.4f}"
    )

    # Backward sanity — just verify no exception, dx is finite.
    dy = torch.randn_like(y)
    try:
        y.backward(dy)
    except RuntimeError as e:
        if "only supports hidden dim" in str(e) or "out of range" in str(e).lower():
            pytest.skip(f"backend documents range limit: {e}")
        raise
    assert torch.isfinite(x.grad).all(), f"non-finite dx on {backend}/{mode}/{shape}"
    assert torch.isfinite(w.grad).all(), f"non-finite dw on {backend}/{mode}/{shape}"
