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
    device_properties = torch.cuda.get_device_properties(0)
    total_memory = device_properties.total_memory
    shared_memory_per_block = 100 * 1024  # 100KB max smem per block
    compute_capability = torch.cuda.get_device_capability(0)

    def get_settings(
        block_m, block_n, block_k, num_stages, num_warps, split_k, group_m
    ):
        required_shared_memory = block_m * block_n * block_k * num_stages
        while required_shared_memory > shared_memory_per_block:
            if block_m > 32 and block_n > 32 and block_k > 32:
                block_m //= 2
                block_n //= 2
                block_k //= 2
            elif num_stages > 1:
                num_stages -= 1
            else:
                raise RuntimeError(
                    f"Out of resource: shared memory. Required: {required_shared_memory}, "
                    f"Hardware limit: {shared_memory_per_block}. Reducing block sizes or `num_stages` may help."
                )
            required_shared_memory = block_m * block_n * block_k * num_stages
        return block_m, block_n, block_k, num_stages, num_warps, split_k, group_m

    def determine_initial_settings(m, n, k, total_memory, compute_capability):
        if compute_capability[0] >= 9:  # SM_90+
            if m * n * k > 1e9:  # large matmul
                return 256, 256, 512, 4, 16, 4, 32
            elif m * n * k > 1e6:  # mid matmul
                return 128, 128, 256, 3, 8, 2, 16
            else:  # small matmul
                return 64, 64, 128, 3, 4, 2, 8
        elif total_memory >= 48 * 1000 * 1000 * 1000:  # >=48 GB VRAM
            if m * n * k > 1e9:  # large matmul
                return 128, 128, 512, 4, 8, 4, 16
            elif m * n * k > 1e6:  # mid matmul
                return 128, 128, 256, 3, 8, 2, 8
            else:  # small matmul
                return 64, 64, 128, 3, 4, 2, 4
        else:  # <48 GB VRAM
            if m * n * k > 1e9:  # large matmul
                return 64, 64, 128, 3, 4, 2, 8
            elif m * n * k > 1e4:  # mid matmul
                return 64, 64, 64, 3, 4, 1, 4
            else:  # small matmul
                return 64, 64, 64, 3, 4, 1, 2

    block_m, block_n, block_k, num_stages, num_warps, split_k, group_m = (
        determine_initial_settings(m, n, k, total_memory, compute_capability)
    )
    return get_settings(
        block_m, block_n, block_k, num_stages, num_warps, split_k, group_m
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
