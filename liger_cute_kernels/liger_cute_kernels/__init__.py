"""liger_cute_kernels — native CUTLASS + NVSHMEM MoE kernels (the "lck" wheel).

Standalone top-level package, kept separate from ``liger_kernel`` so the native
libraries don't mix into the pure-Python package. It ships the compiled
extension and its support libraries side by side::

    liger_cute_kernels/
      __init__.py
      libliger_cute_kernels.so   # torch-free CUTLASS + NVSHMEM core
      libnvshmem_host.so         # bundled nvshmem

The Python API loads the core through TVM FFI, so the runtime boundary is the
torch-free core ABI rather than a Torch extension.

Consumers should go through ``liger_kernel.ops.cute`` rather than importing this
package directly.
"""

from __future__ import annotations
