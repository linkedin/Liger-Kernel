"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/utils.py#L23
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

Modifications made by Yanning Chen, 2024.
"""

import functools
import importlib
import os
from typing import Callable

import torch
import triton
from packaging.version import Version


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def calculate_gemm_settings(m, n, k):
    total_memory = torch.cuda.get_device_properties(0).total_memory

    if total_memory >= 48 * 1000 * 1000 * 1000:  # >=48 GB VRAM
        if m * n * k > 1e9:  # large matmul
            return (
                128,  # block_m
                128,  # block_n
                512,  # block_k
                4,  # num_stages
                8,  # num_warps
                4,  # split_k
                16,  # group_m
            )
        elif m * n * k > 1e6:  # mid matmul
            return (
                128,  # block_m
                128,  # block_n
                256,  # block_k
                3,  # num_stages
                8,  # num_warps
                2,  # split_k
                8,  # group_m
            )
        else:  # small matmul
            return (
                64,  # block_m
                64,  # block_n
                128,  # block_k
                3,  # num_stages
                4,  # num_warps
                2,  # split_k
                4,  # group_m
            )
    else:  # <48 GB VRAM
        if m * n * k > 1e9:  # large matmul
            return (
                64,  # block_m
                64,  # block_n
                128,  # block_k
                3,  # num_stages
                4,  # num_warps
                2,  # split_k
                8,  # group_m
            )
        elif m * n * k > 1e4:  # mid matmul
            return (
                64,  # block_m
                64,  # block_n
                64,  # block_k
                3,  # num_stages
                4,  # num_warps
                1,  # split_k
                4,  # group_m
            )
        else:  # small matmul
            return (
                64,  # block_m
                64,  # block_n
                64,  # block_k
                3,  # num_stages
                4,  # num_warps
                1,  # split_k
                2,  # group_m
            )


def check_compute_capability_for_fp8(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            if compute_capability[0] >= 9:  # SM_90+
                os.environ["ENABLE_TMA"] = "1"
                os.environ["ENABLE_MMA_V3"] = "1"
            elif compute_capability[0] == 8 and compute_capability[1] == 9:  # SM_89
                os.environ["ENABLE_TMA"] = "0"
                os.environ["ENABLE_MMA_V3"] = "0"
            else:
                raise SystemExit(
                    "This kernel requires SM_89 or higher for native FP8 support."
                )
        else:
            raise SystemExit(
                "CUDA is not available. This kernel requires CUDA with SM_89 or higher for native FP8 support."
            )
        return fn(ctx, *args, **kwargs)

    return wrapper


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))
