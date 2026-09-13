from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "cmake" / "export_nonrdc_nvshmem_ptx.py"
_SPEC = importlib.util.spec_from_file_location("export_nonrdc_nvshmem_ptx", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
export_ptx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_ptx)


def _sample_ptx() -> str:
    declarations = "\n".join(f".const .align 8 .b8 {symbol}[8];" for symbol in export_ptx.REQUIRED_SYMBOLS)
    return (
        ".version 8.8\n"
        ".target sm_90a\n"
        f"{declarations}\n"
        "setmaxnreg.dec.sync.aligned.u32 24;\n"
        "setmaxnreg.inc.sync.aligned.u32 240;\n"
    )


def test_export_symbols_marks_only_required_constants_visible():
    transformed = export_ptx.export_symbols(_sample_ptx())

    for symbol in export_ptx.REQUIRED_SYMBOLS:
        assert f".visible .const .align 8 .b8 {symbol}[8];" in transformed
    assert transformed.count(".visible .const") == len(export_ptx.REQUIRED_SYMBOLS)


def test_export_symbols_rejects_missing_symbol():
    source = _sample_ptx().replace(f".const .align 8 .b8 {export_ptx.REQUIRED_SYMBOLS[0]}[8];\n", "")

    with pytest.raises(ValueError, match="expected one PTX declaration"):
        export_ptx.export_symbols(source)


def test_export_symbols_rejects_duplicate_symbol():
    declaration = f".const .align 8 .b8 {export_ptx.REQUIRED_SYMBOLS[0]}[8];\n"
    source = _sample_ptx().replace(declaration, declaration * 2)

    with pytest.raises(ValueError, match="expected one PTX declaration"):
        export_ptx.export_symbols(source)


@pytest.mark.parametrize(
    "probe",
    (
        "setmaxnreg.dec.sync.aligned.u32 24;\n",
        "setmaxnreg.inc.sync.aligned.u32 240;\n",
    ),
)
def test_export_symbols_rejects_missing_setmaxnreg_probe(probe):
    with pytest.raises(ValueError, match="missing the SETMAXNREG build probe"):
        export_ptx.export_symbols(_sample_ptx().replace(probe, ""))


def test_export_symbols_rejects_non_accelerated_target():
    with pytest.raises(ValueError, match="must target sm_90a"):
        export_ptx.export_symbols(_sample_ptx().replace("sm_90a", "sm_90"))


def test_main_rejects_identical_resolved_paths(tmp_path, monkeypatch):
    path = tmp_path / "module.ptx"
    path.write_text(_sample_ptx())
    monkeypatch.setattr(sys, "argv", [str(_MODULE_PATH), str(path), str(path)])

    with pytest.raises(ValueError, match="input and output PTX paths must differ"):
        export_ptx.main()
