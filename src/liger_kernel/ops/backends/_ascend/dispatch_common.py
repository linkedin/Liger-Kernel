"""Shared capability / rank for Ascend Triton dispatcher adapters."""

from __future__ import annotations

import torch

from liger_kernel.backends import Capability
from liger_kernel.utils import is_npu_available

ASCEND_TRITON_CAPABILITY = Capability(modules=["triton"], predicate=is_npu_available)
ASCEND_TRITON_RANK = 10

DEFAULT_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}
