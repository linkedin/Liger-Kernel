"""Build helpers for the LigerCute native core (the "lck" library).

This module lives in the standalone ``liger_cute_kernels`` module at the repo
root and is self-contained: it knows how to compile the torch-free core
``libliger_cute_kernels.so`` from the CMake project sitting next to it.

It is deliberately decoupled from the top-level ``liger_kernel`` wheel — that
wheel is pure Python/Triton and never builds native code. The separate,
CUDA/torch-version-prefixed **lck wheel** (its own setup.py + optional install)
is handled separately; see this module's README.md.

Phase 3.1 — build the torch-free core (no torch required)::

    from cute_build import build_core
    build_core(out_dir)   # -> out_dir/libliger_cute_kernels.so (+ libnvshmem_host.so)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
CMAKE_DIR = HERE  # CMakeLists.txt sits beside this module

CORE_SO = "libliger_cute_kernels.so"
NVSHMEM_SO = "libnvshmem_host.so"

# In-wheel location of the native libraries: the lck wheel's own top-level
# package, kept separate from liger_kernel so it doesn't mix with it.
PKG_REL = Path("liger_cute_kernels")


def _cmake_base_args() -> list[str]:
    args = [f"-DPython_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"]
    if shutil.which("ninja") is not None:
        args.insert(0, "-GNinja")
    return args


def _find_nvshmem_so() -> Path | None:
    home = Path(os.environ.get("NVSHMEM_HOME", "/usr/local/nvshmem"))
    for libdir in ("lib", "lib64"):
        cand = home / libdir / NVSHMEM_SO
        if cand.exists():
            return cand
    return None


def _stage_nvshmem(out_dir: Path) -> None:
    """Copy nvshmem's host .so into out_dir (next to the core)."""
    nvshmem = _find_nvshmem_so()
    if nvshmem is not None:
        shutil.copy2(nvshmem, out_dir / NVSHMEM_SO)
    else:
        print(
            f"WARNING: {NVSHMEM_SO} not found under NVSHMEM_HOME; the lck wheel will not bundle nvshmem",
            file=sys.stderr,
        )


def build_core(out_dir: Path | str, build_temp: Path | str | None = None) -> Path:
    """Build the torch-free core. Returns the path to the core .so.

    Stages the core + nvshmem host .so into ``out_dir``. Requires
    NVSHMEM_HOME / CUTLASS_HOME to be discoverable but does NOT require torch
    (configured with -DLIGER_CUTE_BUILD_BINDINGS=OFF).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_temp = Path(build_temp or (out_dir / "_cmake_core"))
    build_temp.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        ["cmake", "-S", str(CMAKE_DIR), "-B", str(build_temp), *_cmake_base_args(), "-DLIGER_CUTE_BUILD_BINDINGS=OFF"]
    )
    subprocess.check_call(
        ["cmake", "--build", str(build_temp), "--config", "Release", "-j", "--target", "liger_cute_kernels"]
    )

    shutil.copy2(next(build_temp.rglob(CORE_SO)), out_dir / CORE_SO)
    _stage_nvshmem(out_dir)
    return out_dir / CORE_SO


# ── lck wheel build (used by backend/setup.py) ───────────────────────────────


class CMakeExtension(Extension):
    """Marker extension; the real work happens in LckBuildExt."""

    def __init__(self, name: str):
        super().__init__(name, sources=[])


class LckBuildExt(build_ext):
    """Build the lck wheel: compile _C (+ core) and place the .so into the
    wheel's own ``liger_cute_kernels`` package dir.

    If ``LIGER_CUTE_CORE_DIR`` points at a prebuilt core, it is linked as an
    imported lib (no core recompile); otherwise the core is built from source.
    """

    def build_extension(self, ext: Extension) -> None:  # noqa: ARG002
        build_temp = Path(self.build_temp) / "cute"
        build_temp.mkdir(parents=True, exist_ok=True)

        core_dir = os.environ.get("LIGER_CUTE_CORE_DIR")
        use_prebuilt = bool(core_dir) and (Path(core_dir) / CORE_SO).exists()

        cmake_args = [*_cmake_base_args(), "-DLIGER_CUTE_BUILD_BINDINGS=ON"]
        if use_prebuilt:
            cmake_args.append(f"-DLIGER_CUTE_CORE_IMPORTED_DIR={core_dir}")
        subprocess.check_call(["cmake", "-S", str(CMAKE_DIR), "-B", str(build_temp), *cmake_args])
        targets = ["_C"] if use_prebuilt else ["liger_cute_kernels", "_C"]
        subprocess.check_call(["cmake", "--build", str(build_temp), "--config", "Release", "-j", "--target", *targets])

        # Place the three .so into the lck wheel's own liger_cute_kernels package
        # (its __init__.py comes from build_py). build_ext expects _C at
        # get_ext_fullpath(); the cmake output name matches, so copying it there
        # satisfies setuptools.
        dest = Path(self.build_lib) / PKG_REL
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(next(build_temp.rglob("_C*.so")), dest)
        if use_prebuilt:
            shutil.copy2(Path(core_dir) / CORE_SO, dest / CORE_SO)
        else:
            shutil.copy2(next(build_temp.rglob(CORE_SO)), dest / CORE_SO)
        _stage_nvshmem(dest)
        print(f"staged lck artifacts -> {dest}")


def lck_local_version() -> str:
    """PEP 440 local-version segment encoding the CUDA + torch build variant,
    e.g. ``cu130.torch2.9.1``. Override with ``LIGER_CUTE_LOCAL_VERSION``.
    """
    override = os.environ.get("LIGER_CUTE_LOCAL_VERSION")
    if override:
        return override
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "building the lck wheel needs torch importable to tag the wheel "
            "with the CUDA/torch version; install torch or set "
            "LIGER_CUTE_LOCAL_VERSION"
        ) from exc
    cuda = (torch.version.cuda or "nocuda").replace(".", "")
    torch_ver = torch.__version__.split("+")[0]
    return f"cu{cuda}.torch{torch_ver}"
