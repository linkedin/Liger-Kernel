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
    """AMD wavefront size: 32 on RDNA, 64 on CDNA."""
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    arch = (getattr(props, "gcnArchName", None) or "").split(":")[0]
    if arch.startswith(("gfx10", "gfx11", "gfx12")):
        return 32
    return 64


def dtype_to_flydsl_str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise TypeError(f"FlyDSL backend does not support dtype {dtype}")
