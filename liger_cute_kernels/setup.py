"""Build script for the **lck wheel** — the native CUTLASS + NVSHMEM MoE
kernels, shipped as the standalone top-level package ``liger_cute_kernels``.

This is a SEPARATE distribution from the top-level ``liger_kernel`` wheel (which
is pure Python/Triton) and is intentionally its OWN package so it does not mix
into ``liger_kernel``. The lck wheel:

  * builds the torch-free core ``libliger_cute_kernels.so`` with TVM FFI exports
    compiled into that same shared library,
  * ships it as ``liger_cute_kernels/{libliger_cute_kernels.so,
    libnvshmem_host.so}`` (plus its own ``__init__.py``),
  * is tagged with the CUDA + torch version as a PEP 440 local version, e.g.
    ``liger_cute_kernels-0.1.0+cu130.torch2.9.1-cp312-cp312-linux_x86_64.whl``.

``liger_kernel.ops.cute`` (from the liger wheel) imports ``liger_cute_kernels.tvm_ffi``
at runtime, so the two packages stay cleanly separated.

Build against the LOCAL torch/CUDA (no build isolation), from this directory:

    pip wheel . --no-deps --no-build-isolation -w dist

Reuse a core built once across the torch matrix by pointing at its dir:

    LIGER_CUTE_CORE_DIR=/abs/dir-with-core \
        pip wheel . --no-deps --no-build-isolation -w dist
"""

from cute_build import CMakeExtension
from cute_build import LckBuildExt
from cute_build import lck_local_version
from setuptools import setup

BASE_VERSION = "0.1.0"

setup(
    name="liger_cute_kernels",
    version=f"{BASE_VERSION}+{lck_local_version()}",
    description="Native CUTLASS + NVSHMEM MoE kernels (lck) for liger_kernel.ops.cute",
    python_requires=">=3.9",
    install_requires=["torch", "apache-tvm-ffi"],
    extras_require={"nvshmem-pypi": ["nvidia-nvshmem-cu13"]},
    # Self-contained package: its __init__.py is packaged by build_py and the
    # .so are placed beside it by LckBuildExt. The ext module also makes this a
    # platform wheel (cpXY/abi/platform tags).
    packages=["liger_cute_kernels"],
    package_dir={"liger_cute_kernels": "liger_cute_kernels"},
    package_data={"liger_cute_kernels": ["tvm_ffi_bindings.cpp"]},
    ext_modules=[CMakeExtension("liger_cute_kernels.libliger_cute_kernels")],
    cmdclass={"build_ext": LckBuildExt},
    zip_safe=False,
)
