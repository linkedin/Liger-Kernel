"""Shared registration metadata for thin cuTile dispatcher adapters."""

from typing import Optional

import torch

from liger_kernel.backends import Capability
from liger_kernel.ops._nvidia_shared import cutile_compiler_available

CUTILE_CAPABILITY = Capability(
    min_cc=(10, 0),
    modules=["cuda.tile", "torch"],
    predicate=cutile_compiler_available,
)

CUTILE_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}


def validate_default_mode(op_name: str, mode: Optional[str]) -> None:
    if mode not in (None, "default"):
        raise ValueError(f"cuTile {op_name} has only mode='default'; got mode={mode!r}.")
