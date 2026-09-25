"""Build helpers for the LigerCute native CUTLASS + NVSHMEM core.

This module lives in the standalone ``liger_cute_kernels`` module at the repo
root and is self-contained: it knows how to compile the torch-free core
``libliger_cute_kernels.so`` from the CMake project sitting next to it.

It is deliberately decoupled from the top-level ``liger_kernel`` wheel — that
wheel is pure Python/Triton and never builds native code. The separate native
wheel uses the same public version and is handled by its own setup.py; see this
module's README.md.

Phase 3.1 — build the torch-free core (no torch required)::

    from cute_build import build_core
    build_core(out_dir)   # -> core .so and optional cubin
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path

from setuptools import Extension
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
CMAKE_DIR = HERE  # CMakeLists.txt sits beside this module

CORE_SO = "libliger_cute_kernels.so"
SM90_NONRDC_MOE_CUBIN = "liger_moe_sm90_nonrdc.cubin"
SM90_NONRDC_MOE_BUILD_ENV = "LIGER_CUTE_ENABLE_SM90_NONRDC_MOE"
SM90_NONRDC_MOE_ALL_CONFIGS_ENV = "LIGER_CUTE_SM90_NONRDC_ALL_CONFIGS"
CUDA_ARCH_ENV = "LIGER_CUTE_CUDA_ARCH"
CUDA_ARCHES_ENV = "LIGER_CUTE_CUDA_ARCHS"
VERSION_ENV = "LIGER_CUTE_VERSION"
STRIP_NATIVE_ENV = "LIGER_CUTE_STRIP_NATIVE"
CLEAN_BUILD_ENV = "LIGER_CUTE_CLEAN_BUILD"
BUILD_JOBS_ENV = "LIGER_CUTE_BUILD_JOBS"
WHEEL_PLATFORM_ENV = "LIGER_CUTE_WHEEL_PLATFORM_TAG"
ROOT_PYPROJECT = HERE.parent / "pyproject.toml"

# In-wheel location of the native libraries: the native wheel's own top-level
# package, kept separate from liger_kernel so it doesn't mix with it.
PKG_REL = Path("liger_cute_kernels")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None or value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def lck_version() -> str:
    """Return the same public version used by the root ``liger-kernel`` package."""
    override = os.environ.get(VERSION_ENV)
    if override:
        return override
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        ROOT_PYPROJECT.read_text(),
    )
    if match is None:
        raise RuntimeError(f"could not read the project version from {ROOT_PYPROJECT}")
    return match.group(1)


def lck_cuda_arches() -> tuple[str, ...]:
    """Return the architecture cores to package, preserving legacy single-arch builds."""
    plural = os.environ.get(CUDA_ARCHES_ENV)
    singular = os.environ.get(CUDA_ARCH_ENV)
    if plural and singular:
        raise ValueError(f"set only one of {CUDA_ARCHES_ENV} or {CUDA_ARCH_ENV}")
    raw = plural or singular or "90a"
    arches = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not arches:
        raise ValueError("at least one CUDA architecture is required")
    if len(set(arches)) != len(arches):
        raise ValueError(f"duplicate CUDA architectures in {raw!r}")
    for arch in arches:
        if re.fullmatch(r"\d+[a-z]?", arch) is None:
            raise ValueError(f"invalid CUDA architecture {arch!r}")
    return arches


def packaged_core_name(arch: str, *, multi_arch: bool) -> str:
    if not multi_arch:
        return CORE_SO
    return f"libliger_cute_kernels_sm{arch}.so"


def lck_wheel_tag(default_platform: str) -> tuple[str, str, str]:
    """Return a Python-ABI-independent platform tag for the TVM FFI wheel."""
    platform_tag = os.environ.get(WHEEL_PLATFORM_ENV, default_platform)
    if re.fullmatch(r"[a-z0-9_.]+", platform_tag) is None:
        raise ValueError(f"{WHEEL_PLATFORM_ENV} contains an invalid platform tag: {platform_tag!r}")
    return "py3", "none", platform_tag


def _cmake_base_args(cuda_arch: str | None = None) -> list[str]:
    cuda_arch = cuda_arch or os.environ.get(CUDA_ARCH_ENV)
    nonrdc_enabled = _env_flag(SM90_NONRDC_MOE_BUILD_ENV) and cuda_arch in (None, "90a")
    enabled = "ON" if nonrdc_enabled else "OFF"
    all_configs = "ON" if nonrdc_enabled and _env_flag(SM90_NONRDC_MOE_ALL_CONFIGS_ENV) else "OFF"
    args = [
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DLIGER_CUTE_ENABLE_SM90_NONRDC_MOE={enabled}",
        f"-DLIGER_CUTE_SM90_NONRDC_ALL_CONFIGS={all_configs}",
    ]
    if shutil.which("ninja") is not None:
        args.insert(0, "-GNinja")
    if cuda_arch:
        args.append(f"-DLIGER_CUTE_CUDA_ARCH={cuda_arch}")
    return args


def _nvshmem_home_candidates() -> list[Path]:
    candidates = [
        Path(os.environ["NVSHMEM_HOME"]) if os.environ.get("NVSHMEM_HOME") else None,
        Path("/usr/local/nvshmem"),
    ]
    try:
        spec = importlib.util.find_spec("nvidia.nvshmem")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        if spec.origin:
            candidates.append(Path(spec.origin).resolve().parent)
        for location in spec.submodule_search_locations or []:
            candidates.append(Path(location).resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _prepare_nvshmem_home(build_temp: Path) -> Path | None:
    for home in _nvshmem_home_candidates():
        if (home / "include" / "nvshmem.h").exists() and (
            (home / "lib" / "libnvshmem_host.so").exists() or (home / "lib" / "libnvshmem_host.so.3").exists()
        ):
            if (home / "lib" / "libnvshmem_host.so").exists():
                return home
            compat = build_temp / "nvshmem_home"
            compat_lib = compat / "lib"
            compat.mkdir(parents=True, exist_ok=True)
            compat_lib.mkdir(parents=True, exist_ok=True)
            include = compat / "include"
            if not include.exists():
                include.symlink_to(home / "include")
            for so in (home / "lib").glob("*.so.3"):
                link = compat_lib / so.name.removesuffix(".3")
                if not link.exists():
                    link.symlink_to(so)
            device = home / "lib" / "libnvshmem_device.a"
            if device.exists():
                link = compat_lib / "libnvshmem_device.a"
                if not link.exists():
                    link.symlink_to(device)
            return compat
    return None


def _stage_optional_core_artifacts(source_root: Path, out_dir: Path, *, required: bool) -> None:
    direct = source_root / SM90_NONRDC_MOE_CUBIN
    matches = [direct] if direct.is_file() else list(source_root.rglob(SM90_NONRDC_MOE_CUBIN))
    if len(matches) > 1:
        raise RuntimeError(f"multiple {SM90_NONRDC_MOE_CUBIN} artifacts found under {source_root}")
    if not matches:
        if required:
            raise RuntimeError(f"{SM90_NONRDC_MOE_BUILD_ENV}=1 but {SM90_NONRDC_MOE_CUBIN} was not built")
        return
    shutil.copy2(matches[0], out_dir / SM90_NONRDC_MOE_CUBIN)


def _strip_native_binary(path: Path) -> None:
    if not _env_flag(STRIP_NATIVE_ENV):
        return
    strip = shutil.which("strip")
    if strip is None:
        raise RuntimeError(f"{STRIP_NATIVE_ENV}=1 requires the strip executable")
    subprocess.check_call([strip, "--strip-unneeded", str(path)])


def _build_core_for_arch(build_temp: Path, cuda_arch: str) -> Path:
    build_temp.mkdir(parents=True, exist_ok=True)
    cmake_args = [*_cmake_base_args(cuda_arch), "-DLIGER_CUTE_BUILD_BINDINGS=OFF"]
    nvshmem_home = _prepare_nvshmem_home(build_temp)
    if nvshmem_home is not None:
        cmake_args.append(f"-DNVSHMEM_HOME={nvshmem_home}")
    subprocess.check_call(["cmake", "-S", str(CMAKE_DIR), "-B", str(build_temp), *cmake_args])
    build_command = [
        "cmake",
        "--build",
        str(build_temp),
        "--config",
        "Release",
        "--target",
        "liger_cute_kernels",
    ]
    jobs = os.environ.get(BUILD_JOBS_ENV)
    if jobs:
        if not jobs.isdigit() or int(jobs) < 1:
            raise ValueError(f"{BUILD_JOBS_ENV} must be a positive integer, got {jobs!r}")
        build_command.extend(["-j", jobs])
    else:
        build_command.append("-j")
    subprocess.check_call(build_command)
    return next(build_temp.rglob(CORE_SO))


def build_core(out_dir: Path | str, build_temp: Path | str | None = None) -> Path:
    """Build the torch-free core. Returns the path to the core .so.

    Stages the core and optional SM90 non-RDC MoE cubin into ``out_dir``. Requires
    NVSHMEM_HOME / CUTLASS_HOME to be discoverable but does NOT require torch
    (configured with -DLIGER_CUTE_BUILD_BINDINGS=OFF).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_temp = Path(build_temp or (out_dir / "_cmake_core"))
    build_temp.mkdir(parents=True, exist_ok=True)

    arches = lck_cuda_arches()
    if len(arches) != 1:
        raise ValueError(f"build_core requires one architecture, got {arches}")
    shutil.copy2(_build_core_for_arch(build_temp, arches[0]), out_dir / CORE_SO)
    _strip_native_binary(out_dir / CORE_SO)
    _stage_optional_core_artifacts(
        build_temp,
        out_dir,
        required=_env_flag(SM90_NONRDC_MOE_BUILD_ENV) and arches[0] == "90a",
    )
    return out_dir / CORE_SO


# ── native wheel build (used by setup.py) ────────────────────────────────────


class CMakeExtension(Extension):
    """Marker extension; the real work happens in LckBuildExt."""

    def __init__(self, name: str):
        super().__init__(name, sources=[])


class LckBdistWheel(bdist_wheel):
    """Tag the native TVM FFI libraries independently of the build Python ABI."""

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return lck_wheel_tag(platform_tag)


class LckBuildExt(build_ext):
    """Build the native wheel: compile the core plus TVM FFI shim.

    If ``LIGER_CUTE_CORE_DIR`` points at a prebuilt core, it is linked as an
    imported lib (no core recompile); otherwise the core is built from source.
    """

    def build_extension(self, ext: Extension) -> None:  # noqa: ARG002
        build_root = Path(self.build_temp) / "cute"
        core_dir = os.environ.get("LIGER_CUTE_CORE_DIR")
        arches = lck_cuda_arches()
        multi_arch = len(arches) > 1
        dest = Path(self.build_lib) / PKG_REL
        dest.mkdir(parents=True, exist_ok=True)
        for arch in arches:
            arch_build = build_root / arch
            packaged_name = packaged_core_name(arch, multi_arch=multi_arch)
            prebuilt = None
            if core_dir:
                candidates = [
                    Path(core_dir) / packaged_name,
                    Path(core_dir) / arch / CORE_SO,
                ]
                if not multi_arch:
                    candidates.insert(0, Path(core_dir) / CORE_SO)
                prebuilt = next((candidate for candidate in candidates if candidate.is_file()), None)
                if prebuilt is None:
                    raise RuntimeError(f"no prebuilt {arch} core found under {core_dir}")
            core = prebuilt or _build_core_for_arch(arch_build, arch)
            shutil.copy2(core, dest / packaged_name)
            _strip_native_binary(dest / packaged_name)
            if arch == "90a":
                _stage_optional_core_artifacts(
                    prebuilt.parent if prebuilt is not None else arch_build,
                    dest,
                    required=_env_flag(SM90_NONRDC_MOE_BUILD_ENV),
                )
            if prebuilt is None and _env_flag(CLEAN_BUILD_ENV):
                shutil.rmtree(arch_build)
        print(f"staged native artifacts -> {dest}")
