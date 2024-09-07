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


def calculate_settings_mnk(M, N, K, R=1, S=1, group_size=True):
    # default profile
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    num_warps = 4

    if M > 1024:
        BLOCK_SIZE_M = 256
    elif M > 512:
        BLOCK_SIZE_M = 128
    elif M > 256:
        BLOCK_SIZE_M = 64
    else:
        BLOCK_SIZE_M = 32

    if N > 512:
        BLOCK_SIZE_N = 256
    elif N > 256:
        BLOCK_SIZE_N = 128
    elif N > 128:
        BLOCK_SIZE_N = 64
    else:
        BLOCK_SIZE_N = 32

    if K * R * S > 1024:
        BLOCK_SIZE_K = 128
    elif K * R * S > 512:
        BLOCK_SIZE_K = 64
    elif K * R * S > 256:
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_K = 16

    if group_size:
        if N > 512:
            GROUP_SIZE_M = 16
        elif N > 256:
            GROUP_SIZE_M = 8
        else:
            GROUP_SIZE_M = 4

    total_threads = BLOCK_SIZE_M * BLOCK_SIZE_N // 32
    if total_threads > 128:
        num_warps = 8
    elif total_threads > 64:
        num_warps = 4
    elif total_threads > 32:
        num_warps = 2
    else:
        num_warps = 1

    if group_size:
        return BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps
    else:
        return BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))
