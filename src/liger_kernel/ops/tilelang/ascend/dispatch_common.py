"""Shared capability, rank and tolerances for Ascend TileLang implementations."""

import torch

from liger_kernel.backends import Capability
from liger_kernel.utils import is_npu_available

ASCEND_TILELANG_CAPABILITY = Capability(
    modules=["tilelang.ascend.language"],
    predicate=lambda: is_npu_available() and "950" in torch.npu.get_device_name(),
)
ASCEND_TILELANG_RANK = 20

DEFAULT_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}
