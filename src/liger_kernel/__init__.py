"""Liger-Kernel: high-performance Triton kernels for LLM training.

Top-level helpers exposed here let users select a kernel-DSL backend
(Triton, cuTile, ...) without reaching into submodules. The backend
abstraction is implemented in ``liger_kernel.backends``.
"""

# Importing ``functional`` registers op-location declarations with the
# dispatcher (``declare_op_locations`` calls). Without this side effect,
# ``available_impls("rms_norm")`` would return ``[]`` because the dispatcher
# would not know where to find the impl modules to discover them.
from liger_kernel import functional  # noqa: F401

# Legacy back-compat aliases (same objects).
from liger_kernel.backends import available_backends
from liger_kernel.backends import available_impls
from liger_kernel.backends import get_backend
from liger_kernel.backends import get_impl
from liger_kernel.backends import set_backend
from liger_kernel.backends import set_impl

__all__ = [
    # New canonical names
    "available_impls",
    "get_impl",
    "set_impl",
    # Legacy back-compat aliases
    "available_backends",
    "get_backend",
    "set_backend",
]
