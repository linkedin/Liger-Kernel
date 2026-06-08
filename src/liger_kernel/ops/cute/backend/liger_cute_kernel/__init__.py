"""liger_cute_kernel — native CUTLASS + NVSHMEM MoE kernels (the "lck" wheel).

Standalone top-level package, kept separate from ``liger_kernel`` so the native
libraries don't mix into the pure-Python package. It ships the compiled
extension and its support libraries side by side::

    liger_cute_kernel/
      __init__.py
      _C.*.so                    # pybind shim (links libtorch)
      libliger_cute_kernels.so   # torch-free CUTLASS + NVSHMEM core
      libnvshmem_host.so         # bundled nvshmem

``_C`` finds ``libliger_cute_kernels.so`` (and nvshmem) via an ``$ORIGIN`` RPATH
since they sit in this directory. ``_C`` links libtorch, so torch is imported
here first to load ``libtorch.so`` before ``_C``'s NEEDED entry is resolved.

Consumers should go through ``liger_kernel.ops.cute`` rather than importing this
package directly.
"""

from __future__ import annotations

# Ensure libtorch is loaded before any `import liger_cute_kernel._C`.
import torch  # noqa: F401
