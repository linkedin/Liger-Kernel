"""Multi-DSL implementation registry for Liger-Kernel.

This package lets one op (e.g., ``rms_norm``) be implemented in multiple
``(backend, DSL)`` pairs — *implementations* such as ``nvidia-triton``,
``nvidia-cutile``, ``nvidia-cutedsl``, ``ascend-triton`` — and selected at
call time based on hardware capability, user override, or environment.

Vocabulary
----------
- **Backend** = hardware vendor (``nvidia``, ``ascend``, ``rocm``, ``intel``).
- **DSL** = kernel language (``triton``, ``cutile``, ``cutedsl``, ``tilelang``).
- **Implementation** = a ``(backend, DSL)`` pair, named ``<backend>-<dsl>``.

Public surface (new canonical names)
------------------------------------
- :func:`register_impl` — decorator used by impl files to register themselves.
- :func:`dispatch` — resolver that picks an impl and calls it.
- :func:`set_impl` / :func:`get_impl` — process-wide override.
- :func:`available_impls` — introspection: which impls are usable here.
- :class:`Capability` — declarative gate (min compute capability, required
  modules, etc.).
- :class:`OpImpl` — what a registered impl looks like internally.

Legacy back-compat
------------------
The old names ``register_op`` / ``set_backend`` / ``get_backend`` /
``available_backends`` / ``resolve_backend`` / ``BackendNotAvailableError`` /
``NoBackendAvailableError`` / ``LigerBackendFallbackWarning`` are kept as
aliases to the new names so existing callers do not break.
"""

from liger_kernel.backends.capability import Capability
from liger_kernel.backends.dispatch import BackendNotAvailableError
from liger_kernel.backends.dispatch import ImplNotAvailableError
from liger_kernel.backends.dispatch import LigerBackendFallbackWarning
from liger_kernel.backends.dispatch import LigerImplFallbackWarning
from liger_kernel.backends.dispatch import NoBackendAvailableError
from liger_kernel.backends.dispatch import NoImplAvailableError

# Legacy back-compat — same objects under the old names.
from liger_kernel.backends.dispatch import all_backends

# New canonical API (impl-named).
from liger_kernel.backends.dispatch import all_impls
from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import available_impls
from liger_kernel.backends.dispatch import clear_available_cache
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.dispatch import get_backend
from liger_kernel.backends.dispatch import get_impl
from liger_kernel.backends.dispatch import resolve_backend
from liger_kernel.backends.dispatch import resolve_impl
from liger_kernel.backends.dispatch import set_backend
from liger_kernel.backends.dispatch import set_impl
from liger_kernel.backends.registry import OpImpl
from liger_kernel.backends.registry import register_impl
from liger_kernel.backends.registry import register_op

__all__ = [
    "Capability",
    "OpImpl",
    # New canonical names
    "all_impls",
    "available_impls",
    "clear_available_cache",
    "dispatch",
    "get_impl",
    "ImplNotAvailableError",
    "LigerImplFallbackWarning",
    "NoImplAvailableError",
    "register_impl",
    "resolve_impl",
    "set_impl",
    # Legacy back-compat aliases
    "all_backends",
    "available_backends",
    "BackendNotAvailableError",
    "get_backend",
    "LigerBackendFallbackWarning",
    "NoBackendAvailableError",
    "register_op",
    "resolve_backend",
    "set_backend",
]
