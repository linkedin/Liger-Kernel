"""Shared Ascend constants, dtype mappings and SIMD helpers for TileLang kernels."""

import tilelang.ascend.language as T
import torch

# Ascend 950 UB capacity, matching TileLang's Ascend scheduler.
ASCEND_950_UB_SIZE_BYTES = 248 * 1024
DTYPE_MAP = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


@T.macro
def load_float(buf, offset, dtype):
    if dtype == "float32":
        value = T.simd.vld(buf[offset])
    else:
        packed = T.simd.vld(buf[offset], dist="UNPK_B16")
        value = T.simd.vcvt(packed, "float32")
    return value


@T.macro
def store_float(buf, offset, value, dtype):
    if dtype == "float32":
        T.simd.vsts(buf[offset], value)
    else:
        narrow = T.simd.vcvt(value, dtype, sat=False)
        T.simd.vsts(buf[offset], narrow, dist="PK_B32")
