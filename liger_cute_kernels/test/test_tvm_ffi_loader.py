from __future__ import annotations

import importlib.util

from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "liger_cute_kernels" / "tvm_ffi.py"
_SPEC = importlib.util.spec_from_file_location("lck_tvm_ffi", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
tvm_ffi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tvm_ffi)


def test_selects_sm90_core(tmp_path, monkeypatch):
    module = tmp_path / "libliger_cute_kernels_sm90a.so"
    module.touch()
    (tmp_path / "libliger_cute_kernels_sm100f.so").touch()
    monkeypatch.setattr(tvm_ffi.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tvm_ffi.torch.cuda, "get_device_capability", lambda: (9, 0))

    assert tvm_ffi._select_core_module(tmp_path) == module


def test_selects_sm100f_core_for_blackwell_family(tmp_path, monkeypatch):
    (tmp_path / "libliger_cute_kernels_sm90a.so").touch()
    module = tmp_path / "libliger_cute_kernels_sm100f.so"
    module.touch()
    monkeypatch.setattr(tvm_ffi.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tvm_ffi.torch.cuda, "get_device_capability", lambda: (10, 3))

    assert tvm_ffi._select_core_module(tmp_path) == module


def test_select_core_rejects_unsupported_gpu(tmp_path, monkeypatch):
    (tmp_path / "libliger_cute_kernels_sm90a.so").touch()
    (tmp_path / "libliger_cute_kernels_sm100f.so").touch()
    monkeypatch.setattr(tvm_ffi.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tvm_ffi.torch.cuda, "get_device_capability", lambda: (12, 0))

    with pytest.raises(RuntimeError, match="no Liger core"):
        tvm_ffi._select_core_module(tmp_path)


def test_nvshmem_loader_uses_dependency_package(tmp_path, monkeypatch):
    package_root = tmp_path / "site-packages" / "nvidia" / "nvshmem"
    library_dir = package_root / "lib"
    library_dir.mkdir(parents=True)
    host = library_dir / "libnvshmem_host.so.3"
    uid = library_dir / "nvshmem_bootstrap_uid.so.3"
    host.touch()
    uid.touch()
    spec = SimpleNamespace(
        origin=None,
        submodule_search_locations=[str(package_root)],
    )
    loaded = []
    monkeypatch.setattr(tvm_ffi.importlib.util, "find_spec", lambda _: spec)
    monkeypatch.setattr(tvm_ffi.ctypes, "CDLL", lambda path, mode: loaded.append((path, mode)))
    monkeypatch.setattr(tvm_ffi, "_NVSHMEM_LIBS_LOADED", False)

    tvm_ffi._load_nvshmem_libraries(tmp_path / "wheel-package")

    assert [path for path, _ in loaded] == [str(host), str(uid)]
