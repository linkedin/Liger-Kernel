"""Experimental source-packaged C++ kernel built through TVM FFI."""

from pathlib import Path

import torch

_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        import tvm_ffi.cpp

        source = Path(__file__).resolve().parent / "csrc" / "tvm_ffi_double.cpp"
        if not source.exists():
            raise FileNotFoundError(f"Missing packaged TVM FFI source: {source}")
        _MODULE = tvm_ffi.cpp.load(
            name="liger_tvm_ffi_double",
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
        )
    return _MODULE


def tvm_ffi_double(x: torch.Tensor) -> torch.Tensor:
    """Return ``x * 2`` through a CPU TVM FFI C++ implementation."""
    if x.device.type != "cpu":
        raise RuntimeError("tvm_ffi_double currently supports CPU tensors only")
    if x.dtype != torch.float32:
        raise RuntimeError("tvm_ffi_double currently supports torch.float32 tensors only")

    x = x.contiguous()
    output = torch.empty_like(x)
    _load_module().tvm_ffi_double(x, output)
    return output
