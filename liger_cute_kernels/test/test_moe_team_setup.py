from __future__ import annotations

import importlib.util
import sys
import types

from pathlib import Path

from liger_cute_kernels import nvshmem


def _load_moe_module(monkeypatch):
    fake_cute = types.ModuleType("liger_kernel.ops.cute")
    fake_cute._load_tvm_ffi = lambda: object()
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.cute", fake_cute)

    module_path = Path(__file__).resolve().parents[2] / "src" / "liger_kernel" / "ops" / "cute" / "ops" / "moe.py"
    spec = importlib.util.spec_from_file_location("lck_test_moe", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_moe_runtime_only_reads_prepared_team(monkeypatch):
    moe = _load_moe_module(monkeypatch)
    pg = object()
    calls = []

    def fake_resolve_team(process_group, *, create=True):
        calls.append((process_group, create))
        return 23

    monkeypatch.setattr(nvshmem, "resolve_team", fake_resolve_team)

    assert moe._resolve_team(pg) == 23
    assert calls == [(pg, False)]
