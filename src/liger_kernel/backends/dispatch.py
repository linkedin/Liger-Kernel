"""Implementation resolution and dispatch.

Resolution order (first match wins):

    1. explicit ``impl=`` argument on the call (legacy ``backend=`` accepted)
    2. per-op env var: ``LIGER_KERNEL_IMPL_<OP_UPPER>``
       (legacy ``LIGER_KERNEL_BACKEND_<OP_UPPER>`` accepted)
    3. global env var: ``LIGER_KERNEL_IMPL``
       (legacy ``LIGER_KERNEL_BACKEND`` accepted)
    4. global state set by :func:`set_impl` (legacy :func:`set_backend`)
    5. auto-select: the available impl with the lowest ``preference_rank``

An *implementation* is a ``(backend, DSL)`` pair encoded as a hyphenated
string: ``nvidia-triton``, ``nvidia-cutile``, ``nvidia-cutedsl``,
``ascend-triton``, ``ascend-tilelang``. Legacy callers passing the bare DSL
name (``cutile``, ``cutedsl``, ``triton``) continue to work via the
:func:`_resolve_legacy_alias` translation below.

Strict mode (``LIGER_KERNEL_STRICT=1``) makes the error message reference the
*registered-but-unavailable* impls rather than the *not-installed* impls
("install one of …"). In both modes, a missing impl raises
:class:`NoImplAvailableError` — the dispatcher does not silently fall back to
PyTorch reference code. Callers that want a fallback should catch the error
explicitly and choose what to call instead.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

from liger_kernel.backends.registry import OpImpl
from liger_kernel.backends.registry import get_registered
from liger_kernel.backends.registry import list_registered

logger = logging.getLogger(__name__)


class ImplNotAvailableError(RuntimeError):
    """Raised when an explicitly-requested implementation cannot run here."""


class NoImplAvailableError(RuntimeError):
    """Raised when no registered impl can satisfy the op (in any mode).

    Strict mode (``LIGER_KERNEL_STRICT=1``) only changes the error message
    to enumerate registered-but-unavailable impls, not whether the error
    is raised — there is no silent PyTorch fallback at the dispatcher level.
    """


class LigerImplFallbackWarning(UserWarning):
    """Emitted (once per (op, impl)) when auto-select falls back."""


# Legacy back-compat aliases — old names point at the new exception classes
# so existing ``except BackendNotAvailableError`` blocks still catch.
BackendNotAvailableError = ImplNotAvailableError
NoBackendAvailableError = NoImplAvailableError
LigerBackendFallbackWarning = LigerImplFallbackWarning


# Global override set via set_impl().
_GLOBAL_IMPL: Optional[str] = None
_GLOBAL_LOCK = threading.Lock()
# Back-compat alias — legacy callers touched ``_GLOBAL_BACKEND`` directly.
_GLOBAL_BACKEND = _GLOBAL_IMPL  # type: ignore[assignment]  # mutated below

# Track fallback warnings we've already emitted, so we don't spam stdout.
_FALLBACK_NOTIFIED: set[Tuple[str, str, str]] = set()
_FALLBACK_NOTIFIED_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Legacy alias translation
# ---------------------------------------------------------------------------
# Earlier internal versions registered impls under bare DSL names (``triton``, ``cutile``,
# ``cutedsl``). The new convention is ``<backend>-<dsl>``, so we accept both
# and translate at the call boundary.

_LEGACY_ALIASES = {
    "triton": "nvidia-triton",
    "cutile": "nvidia-cutile",
    "cutedsl": "nvidia-cutedsl",
}


def _resolve_legacy_alias(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _LEGACY_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Public configuration API — new (impl) and legacy back-compat (backend)
# ---------------------------------------------------------------------------


def set_impl(name: Optional[str]) -> None:
    """Pin an implementation process-wide. Pass ``None`` to clear the pin."""
    global _GLOBAL_IMPL, _GLOBAL_BACKEND
    name = _resolve_legacy_alias(name)
    with _GLOBAL_LOCK:
        _GLOBAL_IMPL = name
        _GLOBAL_BACKEND = name  # mirror for back-compat readers


def get_impl() -> Optional[str]:
    """Return the process-wide implementation pin (if any), else ``None``."""
    return _GLOBAL_IMPL


# Memoized auto-select lists: (op_name, cc, cuda_ok) -> sorted available impl
# names. Valid while the (op, impl) registration set and the device are
# static — modules register only at import (once per process, via
# ``_discover``), so the evaluation is deterministic between registration/
# discovery events, both of which call :func:`clear_available_cache`.
# Measured on B200 sm_100 (2026-08-09): in-process auto resolution
# 7.61us -> 5.36us/call with the memo (the residual hit-path cost is the
# (cc, cuda_ok) key probe); end-to-end enqueue wall per call
# functional.rms_norm 28.3us -> 21.3us, functional.cross_entropy
# 137.1us -> 130.3us. Unmemoized, every dispatch paid ``list_registered``
# lock + re-sort + one ``Capability`` re-evaluation per impl (incl. the
# cuTile compiler-on-PATH probe). Explicit-impl resolution never caches.
_AVAILABLE_MEMO: Dict[Tuple[str, Optional[object], Optional[Tuple[int, int]], bool], List[str]] = {}
_AVAILABLE_MEMO_LOCK = threading.Lock()


def _device_memo_key(device) -> Optional[object]:
    """Normalize ``device`` to a hashable identity for the auto-select memo.

    ``None`` (current process device) maps to ``None`` so the legacy memo key
    shape/behavior is preserved. A concrete device maps to its integer index
    when one is available (so ``cuda:0`` and ``cuda:1`` get distinct entries),
    otherwise to a stable string form.
    """
    if device is None:
        return None
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        return device.index if device.index is not None else str(device)
    try:
        dev = torch.device(device)
        return dev.index if dev.index is not None else str(dev)
    except (TypeError, RuntimeError, ValueError):
        return str(device)


def _infer_device(args) -> Optional["torch.device"]:
    """Return the device of the first CUDA ``torch.Tensor`` found in ``args``.

    Used by :func:`dispatch` so implementation selection follows the device the
    input tensors actually live on (heterogeneous multi-GPU in one process),
    rather than the current process device. Returns ``None`` when no CUDA
    tensor is present, which preserves the legacy current-device behavior.
    """
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_cuda:
            return a.device
    return None


def clear_available_cache() -> None:
    """Invalidate the memoized ``available_impls`` auto-select lists.

    Called automatically by ``register_impl`` and ``_discover``. Test
    fixtures that snapshot/restore ``_REGISTRY`` directly must call it too
    (``test/backends/test_registry.py``'s fixture does).
    """
    with _AVAILABLE_MEMO_LOCK:
        _AVAILABLE_MEMO.clear()


def available_impls(op_name: str, device=None) -> List[str]:
    """Return registered implementation names for ``op_name`` whose capability
    is currently satisfied on the given device/process.

    ``device`` (an int index, ``str``, ``torch.device``, or ``None``) selects
    which GPU's compute capability gates are evaluated against. ``None``
    preserves the legacy behavior of querying the current process device.

    Order: sorted by ``preference_rank`` (best first).

    Memoized per ``(op_name, device, device cc, cuda-visible)``: that tuple
    fully determines every built-in ``Capability`` gate (min/max cc +
    requires_cuda key on the same ``torch.cuda`` calls; module gates are
    ``sys.modules`` hits after first import; the shipped predicates are pure
    host-info checks — see ``Capability``'s own design note that evaluation is
    negligible-cost host work). Including the resolved device in the key keeps
    heterogeneous multi-GPU selections from aliasing to one device's cc.
    ``_discover`` still runs via ``list_registered`` on a memo miss, so
    first-call discovery semantics are unchanged; explicit-``impl`` resolution
    never uses the memo.
    """
    cc: Optional[Tuple[int, int]] = None
    cuda_ok = False
    try:
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            cc = torch.cuda.get_device_capability(device)
    except (RuntimeError, AssertionError) as exc:  # pragma: no cover - broken CUDA driver edge
        # Expected when the CUDA driver/runtime is unavailable or mismatched;
        # fall back to CPU routing. Unexpected errors propagate.
        logger.debug("CUDA capability probe failed (%s); routing as CUDA-unavailable.", exc)
        cuda_ok, cc = False, None
    memo_key = (op_name, _device_memo_key(device), cc, cuda_ok)
    with _AVAILABLE_MEMO_LOCK:
        cached = _AVAILABLE_MEMO.get(memo_key)
    if cached is not None:
        return list(cached)

    impls = list_registered(op_name)
    usable = [(impl.preference_rank, impl.impl_name) for impl in impls if impl.capability.is_satisfied(device)]
    usable.sort()
    names = [b for _, b in usable]
    with _AVAILABLE_MEMO_LOCK:
        _AVAILABLE_MEMO[memo_key] = list(names)
    return names


def all_impls(op_name: str) -> List[str]:
    """All registered implementations for ``op_name``, regardless of availability."""
    impls = list_registered(op_name)
    return sorted({impl.impl_name for impl in impls})


# Legacy back-compat aliases — exact same behaviour, different names.
set_backend = set_impl
get_backend = get_impl
available_backends = available_impls
all_backends = all_impls


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _per_op_env(op_name: str) -> Optional[str]:
    """Read ``LIGER_KERNEL_IMPL_<OP>`` (preferred) or legacy
    ``LIGER_KERNEL_BACKEND_<OP>``."""
    op_up = op_name.upper()
    return os.environ.get(f"LIGER_KERNEL_IMPL_{op_up}") or os.environ.get(f"LIGER_KERNEL_BACKEND_{op_up}")


def _global_env() -> Optional[str]:
    """Read ``LIGER_KERNEL_IMPL`` (preferred) or legacy ``LIGER_KERNEL_BACKEND``."""
    return os.environ.get("LIGER_KERNEL_IMPL") or os.environ.get("LIGER_KERNEL_BACKEND")


def _strict() -> bool:
    return os.environ.get("LIGER_KERNEL_STRICT", "0") not in ("0", "", "false", "False")


def resolve_impl(op_name: str, requested: Optional[str] = None, device=None) -> OpImpl:
    """Pick the OpImpl that will run.

    Args:
        op_name: registered op (e.g., ``"rms_norm"``).
        requested: optional explicit implementation name (e.g. ``"nvidia-cutile"``
            or legacy ``"cutile"``); honored as long as it is registered. A
            capability mismatch with an explicit request raises
            :class:`ImplNotAvailableError` with a precise reason.
        device: optional device (int index, ``str``, ``torch.device``, or
            ``None``) whose compute capability the capability gates are checked
            against. ``None`` preserves the legacy current-device behavior.

    Returns:
        The selected :class:`OpImpl`.
    """
    # Honor both _GLOBAL_IMPL (canonical) and _GLOBAL_BACKEND (legacy direct-mutation
    # surface). set_impl() keeps them mirrored, but external code that touches the
    # legacy name directly should still affect resolution.
    explicit = _resolve_legacy_alias(
        requested or _per_op_env(op_name) or _global_env() or _GLOBAL_IMPL or _GLOBAL_BACKEND
    )

    if explicit:
        impl = get_registered(op_name, explicit)
        if impl is None:
            registered = sorted({i.impl_name for i in list_registered(op_name)})
            raise ImplNotAvailableError(
                f"No '{explicit}' implementation registered for op '{op_name}'. Registered: {registered or '<none>'}."
            )
        reason = impl.capability.reason_if_unsatisfied(device)
        if reason is not None:
            raise ImplNotAvailableError(
                f"Implementation '{explicit}' for op '{op_name}' is not available: {reason}. "
                f"Available here: {available_impls(op_name, device) or '<none>'}."
            )
        return impl

    # Auto-select.
    auto_order = available_impls(op_name, device)
    if auto_order:
        chosen = auto_order[0]
        impl = get_registered(op_name, chosen)
        assert impl is not None  # available_impls only returns registered ones
        return impl

    if _strict():
        registered = sorted({i.impl_name for i in list_registered(op_name)})
        raise NoImplAvailableError(
            f"No usable implementation for op '{op_name}' in strict mode. "
            f"Registered (but unavailable): {registered or '<none>'}."
        )

    # Outside strict mode but still no usable impl: raise with a clear install
    # hint rather than returning None and letting the caller hit an AttributeError
    # downstream.
    raise NoImplAvailableError(
        f"No implementation available for op '{op_name}'. "
        f"Install one of: {sorted({i.impl_name for i in list_registered(op_name)}) or '<none>'}."
    )


# Legacy back-compat alias.
resolve_backend = resolve_impl


@torch.compiler.disable
def dispatch(
    op_name: str,
    *args,
    impl: Optional[str] = None,
    backend: Optional[str] = None,  # back-compat alias for impl
    **kwargs,
):
    """Resolve the right implementation for ``op_name`` and invoke it.

    ``mode`` (if passed in ``kwargs``) is forwarded to the impl, which may use
    it to pick a kernel variant. Unknown modes are rejected by the impl.

    ``impl=`` is the new canonical kwarg; ``backend=`` remains as a legacy
    alias and forwards to the same field.

    Auto-select resolves the impl against the device of the first CUDA input
    tensor in ``args`` (via :func:`_infer_device`), so heterogeneous multi-GPU
    processes select based on where the inputs actually live rather than the
    current process device. With no CUDA input tensor, the current device is
    used (legacy behavior).
    """
    if impl is None and backend is not None:
        impl = backend
    device = _infer_device(args)
    selected = resolve_impl(op_name, requested=impl, device=device)
    return selected.call(*args, **kwargs)


def emit_fallback_warning(op_name: str, requested: str, actual: str, reason: str) -> None:
    """Helper for impls that internally fall back to a sibling impl.

    Emits at most one warning per (op, requested, actual) tuple.
    """
    message = (
        f"Liger {op_name}: requested '{requested}' implementation not usable ({reason}); falling back to '{actual}'."
    )
    if _strict():
        raise ImplNotAvailableError(message)

    key = (op_name, requested, actual)
    with _FALLBACK_NOTIFIED_LOCK:
        if key in _FALLBACK_NOTIFIED:
            return
        _FALLBACK_NOTIFIED.add(key)
    warnings.warn(
        f"{message} Set LIGER_KERNEL_STRICT=1 to raise instead.",
        LigerImplFallbackWarning,
        stacklevel=3,
    )
