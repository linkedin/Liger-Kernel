"""cuTile (NVIDIA cuda.tile) backend implementations for Liger-Kernel ops.

This package holds per-op cuTile kernel implementations registered with the
multi-DSL dispatcher via :func:`liger_kernel.backends.register_op`. Importing
modules in this package triggers their registration as a side effect; the
modules themselves import ``cuda.tile``, so callers without that package will
see :class:`ImportError`, which the dispatcher's discovery layer catches.

This file is intentionally empty (no ``register_vendor`` call) — vendor-level
op swapping (NPU/XPU) happens in ``liger_kernel.ops.backends`` instead.
"""
