"""
Shared helpers for the FlyDSL backend ops.
"""

from __future__ import annotations

import torch


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def warp_size(device: torch.device | None = None) -> int:
    """AMD wavefront size: 32 on RDNA, 64 on CDNA.

    Uses FlyDSL's ``is_rdna_arch`` as the source of truth rather than an inline
    ``gfx*`` prefix test. Note that ``gfx1250`` is *not* RDNA here (it runs
    wave64), so a naive ``gfx12`` prefix check would wrongly return 32.
    """
    from flydsl.runtime.device import is_rdna_arch

    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    arch = (getattr(props, "gcnArchName", None) or "").split(":")[0]
    return 32 if is_rdna_arch(arch) else 64


_TORCH_TO_FLYDSL_STR = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def dtype_to_flydsl_str(dtype: torch.dtype) -> str:
    """FlyDSL type-name string ('f32' / 'f16' / 'bf16') for a torch dtype."""
    try:
        return _TORCH_TO_FLYDSL_STR[dtype]
    except KeyError:
        supported = ", ".join(_TORCH_TO_FLYDSL_STR.values())
        raise TypeError(
            f"FlyDSL backend supports {{{supported}}} logits, got {dtype}. "
            f"Select a different backend via LIGER_KERNEL_IMPL, or cast the input."
        ) from None


def flydsl_elem_type(dtype_str: str):
    """FlyDSL numeric type (e.g. ``fx.Float32``) for a type-name string."""
    import flydsl.expr as fx

    return {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]


def elem_bits(dtype_str: str) -> int:
    """Element bit width for a type-name string, read from the fx numeric type."""
    return flydsl_elem_type(dtype_str).width
