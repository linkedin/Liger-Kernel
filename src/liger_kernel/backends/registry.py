"""Per-(op, backend) registry of kernel implementations.

Each impl file registers itself via :func:`register_op` at import time. The
registry stores :class:`OpImpl` records keyed by ``(op_name, backend)``.

Impl modules are **not** imported at ``liger_kernel`` import time — that would
force every consumer to have triton, cuda.tile, etc. installed. Instead, the
dispatcher imports them on demand using :func:`discover_op` below.

Registration patterns
---------------------

A backend impl can register either a ``torch.autograd.Function`` (preferred for
ops with custom backward) or a plain forward callable.

Plain forward registration (recommended — supports kwargs cleanly)::

    @register_op("rms_norm", backend="triton", capability=Capability(modules=["triton"]))
    def rms_norm_triton(x, w, eps, *, mode=None, offset=0.0):
        ...  # may delegate to a torch.autograd.Function internally

Function class registration (positional-only — see limitation below)::

    @register_op(
        "rms_norm",
        backend="triton",
        capability=Capability(modules=["triton"]),
        preference_rank=50,
        tolerances={torch.bfloat16: dict(atol_fwd=2e-2, atol_bwd=1e-1)},
    )
    class LigerRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(...): ...
        @staticmethod
        def backward(...): ...

**Limitation of class registration:** ``torch.autograd.Function.apply()`` does
not accept keyword arguments. :meth:`OpImpl.call` strips the dispatcher's
``mode`` meta-kwarg (variant selection happens via the impl's registered
``modes`` declaration instead) and raises ``TypeError`` on any other kwargs.
If callers need to pass kwargs (``offset=``, ``casting_mode=``, …), use the
plain-forward pattern and let it delegate to an autograd Function internally
with positional arguments only.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type

from liger_kernel.backends.capability import Capability

logger = logging.getLogger(__name__)


def _invalidate_available_cache() -> None:
    """Clear the dispatcher's memoized auto-select lists.

    Resolved via ``sys.modules`` (NOT ``from liger_kernel.backends import
    dispatch`` — the package re-exports the ``dispatch`` *function*, which
    shadows the submodule on attribute lookup). If the dispatcher module was
    never imported there is no memo to clear yet.
    """
    _dispatch_mod = sys.modules.get("liger_kernel.backends.dispatch")
    if _dispatch_mod is None:
        try:
            _dispatch_mod = importlib.import_module("liger_kernel.backends.dispatch")
        except ImportError:  # pragma: no cover - import-order edge
            # Dispatcher not importable yet (registration during its own import);
            # there is no memo to clear. Genuine init errors are not swallowed.
            logger.debug("dispatch module not importable during cache invalidation; skipping.")
            return
    _clear = getattr(_dispatch_mod, "clear_available_cache", None)
    if _clear is not None:
        _clear()


@dataclass(frozen=True)
class OpImpl:
    """A single implementation of a single op.

    An *implementation* is a ``(backend, DSL)`` pair such as ``nvidia-triton``,
    ``nvidia-cutile``, ``nvidia-cutedsl``, or ``ascend-triton`` — stored as the
    hyphenated string in :attr:`impl_name`.

    The legacy attribute name :attr:`backend` is preserved as a property alias
    pointing at :attr:`impl_name`, so legacy callers that read ``op.backend`` do
    not break.
    """

    op_name: str
    impl_name: str
    capability: Capability = field(default_factory=Capability)
    forward: Optional[Callable[..., Any]] = None
    autograd_function: Optional[Type] = None
    modes: Tuple[str, ...] = ()
    default_mode: Optional[str] = None
    preference_rank: int = 100
    tolerances: Mapping[Any, Mapping[str, float]] = field(default_factory=dict)
    notes: str = ""

    @property
    def backend(self) -> str:  # back-compat alias for legacy callers
        return self.impl_name

    def call(self, *args, **kwargs):
        """Invoke this impl, preferring the autograd_function if present.

        ``torch.autograd.Function.apply()`` does not accept keyword arguments
        (modern torch raises ``TypeError``). The dispatcher uses ``mode`` as
        a meta-kwarg to select kernel variants; we strip it here and let the
        registered ``modes`` declaration handle variant selection at registration
        time. Other kwargs cannot be forwarded through ``.apply()`` and are
        rejected with a clear error so callers don't hit a confusing torch
        internal error.
        """
        if self.autograd_function is not None:
            # ``mode`` is dispatcher meta — the autograd_function path picks
            # variants by the impl's registered ``modes``, not by kwarg.
            kwargs.pop("mode", None)
            if kwargs:
                raise TypeError(
                    f"OpImpl({self.op_name!r}, {self.impl_name!r}) uses an "
                    f"autograd_function registration, which cannot accept "
                    f"keyword arguments via ``.apply()``. Got kwargs: "
                    f"{sorted(kwargs)}. Either pass these positionally, or "
                    f"register a plain forward callable instead of a "
                    f"``torch.autograd.Function`` subclass."
                )
            return self.autograd_function.apply(*args)
        if self.forward is not None:
            return self.forward(*args, **kwargs)
        raise RuntimeError(f"OpImpl({self.op_name}, {self.impl_name}) has no callable")


# Global, process-wide registry. Keyed by (op_name, backend).
_REGISTRY: Dict[Tuple[str, str], OpImpl] = {}
_REGISTRY_LOCK = threading.Lock()

# Map of op_name -> set of impl-module paths that should be import-probed when
# the dispatcher first sees that op. Populated by ``declare_op_locations``.
_DISCOVERY_MAP: Dict[str, Tuple[str, ...]] = {}
_DISCOVERED: Dict[str, bool] = {}


def register_impl(
    op_name: str,
    *,
    impl_name: Optional[str] = None,
    capability: Optional[Capability] = None,
    modes: Tuple[str, ...] = (),
    default_mode: Optional[str] = None,
    preference_rank: int = 100,
    tolerances: Optional[Mapping[Any, Mapping[str, float]]] = None,
    notes: str = "",
    backend: Optional[str] = None,  # back-compat alias for impl_name
):
    """Decorator that registers a kernel implementation in the global registry.

    The implementation is identified by ``(op_name, impl_name)`` where
    ``impl_name`` is the hyphenated ``<backend>-<dsl>`` form:
    ``"nvidia-triton"``, ``"nvidia-cutile"``, ``"nvidia-cutedsl"``,
    ``"ascend-triton"``, …

    For back-compat, the legacy keyword ``backend=`` is accepted as
    an alias for ``impl_name=`` and forwards to the same field.

    Works on either a ``torch.autograd.Function`` subclass or a plain function.
    The decorator returns the decorated object unchanged.
    """
    if impl_name is None and backend is None:
        raise TypeError("register_impl requires impl_name= (or back-compat backend=)")
    if impl_name is None:
        impl_name = backend  # legacy back-compat

    def _decorator(obj):
        if isinstance(obj, type):
            impl = OpImpl(
                op_name=op_name,
                impl_name=impl_name,
                capability=capability or Capability(),
                autograd_function=obj,
                modes=tuple(modes),
                default_mode=default_mode,
                preference_rank=preference_rank,
                tolerances=tolerances or {},
                notes=notes,
            )
        elif callable(obj):
            impl = OpImpl(
                op_name=op_name,
                impl_name=impl_name,
                capability=capability or Capability(),
                forward=obj,
                modes=tuple(modes),
                default_mode=default_mode,
                preference_rank=preference_rank,
                tolerances=tolerances or {},
                notes=notes,
            )
        else:
            raise TypeError(f"@register_impl expects a class or callable, got {type(obj).__name__}")

        key = (op_name, impl_name)
        with _REGISTRY_LOCK:
            existing = _REGISTRY.get(key)
            if existing is not None and existing is not impl:
                logger.debug("register_impl: overwriting impl for %s", key)
            _REGISTRY[key] = impl

        # Any (re)registration can change auto-select outcomes.
        _invalidate_available_cache()

        return obj

    return _decorator


# Legacy back-compat alias. ``register_op`` was the original name; new code
# should use ``register_impl``.
register_op = register_impl


def declare_op_locations(op_name: str, module_paths: Tuple[str, ...]) -> None:
    """Tell the dispatcher where to find impls for ``op_name``.

    Used so that ``available_impls("rms_norm")`` (a.k.a. legacy
    ``available_backends``) can lazily import only the modules that might
    register an impl for that op, rather than eagerly importing every
    backend at package load.
    """
    _DISCOVERY_MAP[op_name] = tuple(module_paths)


_DISCOVERY_FAILURES: Dict[str, Tuple[type, str]] = {}
"""Records (module_path -> (exc_type, message)) for failed discovery imports.
Surfaced by :func:`get_discovery_failures` so users can see why a
backend isn't appearing in :func:`available_backends`."""


# Per-op completion events. A thread that wins the discovery race creates the
# event when it claims discovery and ``.set()``s it after the import pass
# completes (success or failure). Concurrent callers ``.wait()`` on the event
# so they only see the registry once registration has actually happened — the
# original implementation only had a "claim" flag, which let racing threads
# return with an incomplete registry while the import was mid-flight.
_DISCOVERY_EVENTS: Dict[str, threading.Event] = {}


def _discover(op_name: str) -> None:
    """Import the impl modules declared for ``op_name``, once.

    Catches a broad ``Exception`` because impl modules may fail to import for
    reasons other than ``ImportError`` (e.g., a transitive module uses
    Python-3.10-only union syntax on an older interpreter, or a kernel JIT
    fails to compile at import time). Either way: the impl can't run, so the
    dispatcher should skip it rather than crash the whole process. We log
    everything at WARNING level so users can investigate via
    :func:`get_discovery_failures`.

    Additionally, we snapshot ``_REGISTRY`` before and after the import pass
    and warn if a module imported cleanly but did not register anything for
    ``op_name`` — that's almost always a conditional ``@register_op`` guard
    (capability check, feature flag) silently disabling the impl, which would
    otherwise surface as a cryptic "No backend available" downstream.

    Concurrency: the first caller for each op creates a per-op
    ``threading.Event`` under ``_REGISTRY_LOCK`` and signals it after the
    import pass completes. Subsequent callers wait on the event so they
    cannot observe an in-flight, partially-populated registry. Discovery
    runs OUTSIDE ``_REGISTRY_LOCK`` to avoid deadlocking against
    ``register_impl`` calls executed by the import side-effect.
    """
    with _REGISTRY_LOCK:
        if _DISCOVERED.get(op_name):
            event = _DISCOVERY_EVENTS.get(op_name)
            if event is None:
                # Already-completed discovery from before the event scheme
                # existed (or test-fixture mutation). Treat as done.
                return
        else:
            event = _DISCOVERY_EVENTS.get(op_name)
            if event is None:
                event = threading.Event()
                _DISCOVERY_EVENTS[op_name] = event
                _DISCOVERED[op_name] = True  # claim discovery
                we_own_discovery = True
            else:
                we_own_discovery = False
            if not we_own_discovery:
                # Another thread is mid-import; fall through to wait below.
                pass
            else:
                before = {k for k in _REGISTRY.keys() if k[0] == op_name}

    # Hot path for late callers: imports already done — quick wait, return.
    if not _DISCOVERY_EVENTS.get(op_name) or _DISCOVERY_EVENTS[op_name].is_set():
        return
    if not locals().get("we_own_discovery", False):
        # Block until the owner finishes the import pass + registration.
        _DISCOVERY_EVENTS[op_name].wait()
        return

    try:
        for path in _DISCOVERY_MAP.get(op_name, ()):
            try:
                importlib.import_module(path)
            except Exception as e:  # noqa: BLE001 — intentionally broad
                _DISCOVERY_FAILURES[path] = (type(e), str(e))
                logger.warning(
                    "Backend impl '%s' for op '%s' could not be imported: %s: %s",
                    path,
                    op_name,
                    type(e).__name__,
                    e,
                )

        with _REGISTRY_LOCK:
            after = {k for k in _REGISTRY.keys() if k[0] == op_name}
        new_pairs = after - before
        op_paths = set(_DISCOVERY_MAP.get(op_name, ()))
        # Only suppress the warning if a failure is for *this op's* declared
        # paths — failures for unrelated ops shouldn't mask a silent-
        # registration bug for the current op.
        this_op_failures = op_paths & set(_DISCOVERY_FAILURES.keys())
        if not new_pairs and op_paths and not this_op_failures and not before:
            # All declared paths imported successfully but nothing registered
            # for this op. Almost always a conditional decorator guard. Skip
            # when ``before`` is non-empty (sibling ops in the same module
            # already triggered the @register_op decorators).
            logger.warning(
                "_discover(%r): all declared paths imported but no @register_op ran. "
                "Check for conditional guards (capability check, feature flag, "
                "torch.cuda.is_available()) in: %s",
                op_name,
                list(_DISCOVERY_MAP[op_name]),
            )
    finally:
        # Signal completion regardless of import outcome so waiting threads
        # don't hang on a partial failure.
        _DISCOVERY_EVENTS[op_name].set()
        # Discovery may have registered new impls (or silently failed to);
        # drop the dispatcher's memoized auto-select lists so the next
        # ``available_impls`` re-evaluates against the final registry state.
        _invalidate_available_cache()


def get_discovery_failures() -> Dict[str, Tuple[type, str]]:
    """Return a snapshot of impl-module imports that failed during discovery.

    Keys are dotted module paths; values are ``(exception_type, message)``
    tuples. Used by tests and diagnostics.
    """
    return dict(_DISCOVERY_FAILURES)


def get_registered(op_name: str, backend: str) -> Optional[OpImpl]:
    _discover(op_name)
    with _REGISTRY_LOCK:
        return _REGISTRY.get((op_name, backend))


def list_registered(op_name: str) -> Tuple[OpImpl, ...]:
    """Return all registered impls for ``op_name`` (any availability).

    Holds ``_REGISTRY_LOCK`` while snapshotting so concurrent ``register_op``
    calls don't trip a ``dictionary changed size during iteration`` error in
    multi-threaded servers (vLLM, TGI, etc.).
    """
    _discover(op_name)
    with _REGISTRY_LOCK:
        return tuple(impl for (n, _b), impl in _REGISTRY.items() if n == op_name)


def all_registered_ops() -> Tuple[str, ...]:
    with _REGISTRY_LOCK:
        return tuple(sorted({op for (op, _b) in _REGISTRY.keys()}))
