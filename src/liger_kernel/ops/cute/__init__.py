"""LigerCute — fused MoE + NVSHMEM kernels (ported from LigerCommKernels).

The native kernels live in a SEPARATE top-level package ``liger_cute_kernel``,
shipped by its own CUDA/torch-version prefixed **lck wheel** — not by the
top-level ``liger_kernel`` wheel, which stays pure Python/Triton. This module is
just the entry point that loads ``liger_cute_kernel._C`` when it is installed::

    liger_cute_kernel/            # the lck wheel (optional, separate package)
      __init__.py
      _C.*.so                     # pybind shim
      libliger_cute_kernels.so    # torch-free CUTLASS + NVSHMEM core
      libnvshmem_host.so          # bundled nvshmem

Harness stage: no functional kernels are wired up yet. The lck wheel is
optional, so the rest of ``liger_kernel`` keeps working without it.
"""

from __future__ import annotations

import importlib

# Cached handle to the compiled extension (from the separate liger_cute_kernel
# package). None until first loaded.
_ext = None


def _load_extension():
    """Import the ``liger_cute_kernel._C`` extension, or raise a helpful error."""
    global _ext
    if _ext is not None:
        return _ext
    # _C links libtorch, so torch must be imported first to load libtorch.so
    # into the process before the extension's NEEDED entry is resolved.
    import torch  # noqa: F401

    try:
        _ext = importlib.import_module("liger_cute_kernel._C")
    except ImportError as exc:  # pragma: no cover - depends on a CUDA build
        raise ImportError(
            "liger_cute_kernel is not installed. Install the matching lck wheel "
            "for your CUDA/torch environment, or build it locally (see "
            "liger_kernel/ops/cute/backend/README.md)."
        ) from exc
    return _ext


def is_available() -> bool:
    """True if the compiled ``_C`` extension can be imported."""
    try:
        return _load_extension() is not None
    except ImportError:
        return False


__all__ = ["is_available"]
