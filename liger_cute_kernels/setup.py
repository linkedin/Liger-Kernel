"""Build script for the native CUTLASS + NVSHMEM kernel wheel.

The wheel installs the standalone top-level package ``liger_cute_kernels``.

This is a SEPARATE distribution from the top-level ``liger_kernel`` wheel (which
is pure Python/Triton) and is intentionally its OWN package so it does not mix
into ``liger_kernel``. The native wheel:

  * builds torch-free SM90a and SM100f cores with TVM FFI exports,
  * selects the matching core at runtime from the active GPU capability,
  * offers CUDA 12/13 NVSHMEM dependencies through optional extras instead of
    copying their shared libraries into this wheel, and
  * uses the same public version as the top-level ``liger-kernel`` package.

``liger_kernel.ops.cute`` (from the liger wheel) imports ``liger_cute_kernels.tvm_ffi``
at runtime, so the two packages stay cleanly separated.

Build against the LOCAL torch/CUDA (no build isolation), from this directory:

    pip wheel . --no-deps --no-build-isolation -w dist

Reuse a core built once across the torch matrix by pointing at its dir:

    LIGER_CUTE_CORE_DIR=/abs/dir-with-core \
        pip wheel . --no-deps --no-build-isolation -w dist
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cute_build import CMakeExtension
from cute_build import LckBdistWheel
from cute_build import LckBuildExt
from cute_build import lck_version
from setuptools import setup

HERE = Path(__file__).resolve().parent

setup(
    name="liger-cute-kernels",
    version=lck_version(),
    description="Native CUTLASS + NVSHMEM kernels for liger_kernel.ops.cute",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="BSD-2-Clause",
    license_files=["LICENSE"],
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "apache-tvm-ffi",
    ],
    extras_require={
        "cu12": ["nvidia-nvshmem-cu12==3.6.5"],
        "cu13": ["nvidia-nvshmem-cu13==3.6.5"],
    },
    # Self-contained package: its __init__.py is packaged by build_py and the
    # .so are placed beside it by LckBuildExt. The marker extension makes this a
    # platform wheel, while LckBdistWheel records that the TVM FFI core is not
    # tied to the CPython ABI used to package it.
    packages=["liger_cute_kernels"],
    package_dir={"liger_cute_kernels": "liger_cute_kernels"},
    package_data={"liger_cute_kernels": ["tvm_ffi_bindings.cpp", "*.so", "*.cubin"]},
    ext_modules=[CMakeExtension("liger_cute_kernels.libliger_cute_kernels")],
    cmdclass={"bdist_wheel": LckBdistWheel, "build_ext": LckBuildExt},
    zip_safe=False,
)
