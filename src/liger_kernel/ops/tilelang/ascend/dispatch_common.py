"""Shared capability, rank and tolerances for Ascend TileLang implementations."""

import importlib.util

import torch

from liger_kernel.backends import Capability
from liger_kernel.utils import is_npu_available


def _is_available():
    # Discovery must not import TileLang's TVM before a Triton/TBE call.
    return (
        is_npu_available()
        and torch.npu.get_device_name().startswith("Ascend950")
        and importlib.util.find_spec("tilelang") is not None
    )


ASCEND_TILELANG_CAPABILITY = Capability(predicate=_is_available)
ASCEND_TILELANG_RANK = 20

DEFAULT_TOLERANCES = {
    torch.float16: {"atol_fwd": 5e-3, "atol_bwd": 5e-2, "rtol_fwd": 1e-3, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_fwd": 1e-2, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_fwd": 1e-5, "rtol_bwd": 1e-4},
}
