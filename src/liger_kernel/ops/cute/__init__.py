"""LigerCute — fused MoE + NVSHMEM kernels (ported from LigerCommKernels).

The native kernels live in a SEPARATE top-level package ``liger_cute_kernels``,
shipped by its own CUDA/torch-version prefixed **lck wheel** — not by the
top-level ``liger_kernel`` wheel, which stays pure Python/Triton. This module is
just the entry point that loads ``liger_cute_kernels.tvm_ffi`` when it is installed::

    liger_cute_kernels/           # the lck wheel (optional, separate package)
      __init__.py
      tvm_ffi.py                  # TVM FFI facade over the native core
      libliger_cute_kernels.so    # torch-free CUTLASS + NVSHMEM core
      libnvshmem_host.so          # bundled nvshmem

The fused MoE op (``moe_fused``) lives in ``cute/ops/`` and is wired up as the
opt-in ``cute`` implementation (``LIGER_KERNEL_IMPL=cute``). The lck wheel is
optional, so the rest of ``liger_kernel`` keeps working without it — the ops
module is imported only when the ``cute`` implementation is actively selected.
"""

from __future__ import annotations

import importlib

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

# Cached handle to the TVM FFI facade (from the separate liger_cute_kernels
# package). None until first loaded.
_tvm_ffi = None


def _load_tvm_ffi():
    """Import the ``liger_cute_kernels.tvm_ffi`` facade, or raise a helpful error."""
    global _tvm_ffi
    if _tvm_ffi is not None:
        return _tvm_ffi
    try:
        _tvm_ffi = importlib.import_module("liger_cute_kernels.tvm_ffi")
        if not _tvm_ffi.is_available():
            raise ImportError("liger_cute_kernels.tvm_ffi could not load the native core")
    except ImportError as exc:  # pragma: no cover - depends on a CUDA build
        raise ImportError(
            "liger_cute_kernels is not installed. Install the matching lck wheel "
            "for your CUDA/torch environment, or build it locally (see the "
            "liger_cute_kernels/ module at the repo root)."
        ) from exc
    return _tvm_ffi


def is_available() -> bool:
    """True if the TVM FFI facade can load the native core."""
    try:
        return _load_tvm_ffi() is not None
    except ImportError:
        return False


__all__ = ["is_available"]


# Self-register as the opt-in "cute" implementation. Like cutile, ``cute`` has no
# default_devices, so it is never auto-applied — users select it explicitly via
# ``LIGER_KERNEL_IMPL=cute``. Registration is pure metadata and must not import
# the native extension: this __init__ is imported during impl discovery even when
# the lck wheel is absent. The ops module (``cute.ops``) — which does load the
# extension — is imported only when this implementation is actively selected.
register_impl(
    ImplInfo(
        name="cute",
        devices=("cuda",),
        module_path=f"{__name__}.ops",  # liger_kernel.ops.cute.ops
    )
)
