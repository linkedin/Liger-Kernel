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
from typing import Callable, Tuple

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


@functools.lru_cache(maxsize=128)
def calculate_settings_mnk(
    M: int, N: int, K: int, R: int = 1, S: int = 1, group_size: bool = True
) -> Tuple[int, ...]:
    block_sizes_m = [32, 64, 128, 256]
    block_sizes_n = [32, 64, 128, 256]
    block_sizes_k = [16, 32, 64, 128]
    group_sizes_m = [4, 8, 16]
    warp_sizes = [1, 2, 4, 8]

    def choose_optimal(sizes, threshold):
        return next((size for size in reversed(sizes) if threshold >= size), sizes[0])

    # compute optimal block sizes
    BLOCK_SIZE_M = choose_optimal(block_sizes_m, M)
    BLOCK_SIZE_N = choose_optimal(block_sizes_n, N)
    BLOCK_SIZE_K = choose_optimal(block_sizes_k, K * R * S)
    GROUP_SIZE_M = choose_optimal(group_sizes_m, N) if group_size else None

    total_threads = BLOCK_SIZE_M * BLOCK_SIZE_N // 32
    num_warps = choose_optimal(warp_sizes, total_threads)

    # compute grid
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    if group_size:
        return BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, grid
    else:
        return BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, grid


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))
