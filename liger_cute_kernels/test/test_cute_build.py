from __future__ import annotations

import importlib.util

from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "cute_build.py"
_SPEC = importlib.util.spec_from_file_location("lck_cute_build", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cute_build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cute_build)


def test_prepare_nvshmem_home_adapts_versioned_pypi_layout(tmp_path, monkeypatch):
    home = tmp_path / "site-packages" / "nvidia" / "nvshmem"
    (home / "include").mkdir(parents=True)
    (home / "lib").mkdir()
    (home / "include" / "nvshmem.h").touch()
    (home / "lib" / "libnvshmem_host.so.3").touch()
    (home / "lib" / "libnvshmem_device.a").touch()
    monkeypatch.setattr(cute_build, "_nvshmem_home_candidates", lambda: [home])

    compat = cute_build._prepare_nvshmem_home(tmp_path / "build")

    assert compat is not None
    assert (compat / "include").resolve() == (home / "include").resolve()
    assert (compat / "lib" / "libnvshmem_host.so").resolve() == (home / "lib" / "libnvshmem_host.so.3").resolve()
    assert (compat / "lib" / "libnvshmem_device.a").resolve() == (home / "lib" / "libnvshmem_device.a").resolve()


def test_build_core_passes_prepared_nvshmem_home(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    build_temp = tmp_path / "cmake"
    compat_home = tmp_path / "nvshmem-compat"
    calls = []

    def fake_check_call(command):
        calls.append(command)
        if "--build" in command:
            build_temp.mkdir(parents=True, exist_ok=True)
            (build_temp / cute_build.CORE_SO).touch()

    monkeypatch.setattr(cute_build, "_prepare_nvshmem_home", lambda _: compat_home)
    monkeypatch.setattr(cute_build.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(cute_build, "_stage_nvshmem", lambda _: None)

    result = cute_build.build_core(out_dir, build_temp)

    assert result == out_dir / cute_build.CORE_SO
    assert result.exists()
    configure = calls[0]
    assert f"-DNVSHMEM_HOME={compat_home}" in configure
    assert "-DLIGER_CUTE_BUILD_BINDINGS=OFF" in configure
