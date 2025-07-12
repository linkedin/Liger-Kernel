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
import operator

from typing import Callable

import torch
import triton
import triton.language as tl

from packaging.version import Version

from liger_kernel.utils import infer_device


def is_hip() -> bool:
    return torch.version.hip is not None


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def _calculate_max_fused_size():
    torch_device_props = _get_device_properties()

    MIN_SIZE = 4096
    MAX_SIZE = 131072
    base_size = 65536

    if torch_device_props["max_regs"] > 65536:
        reg_limit = 131072
    elif torch_device_props["max_regs"] > 49152:
        reg_limit = 65536
    elif torch_device_props["max_regs"] > 32768:
        reg_limit = 32768
    else:
        reg_limit = 16384

    if torch_device_props["num_sm"] >= 80:
        sm_factor = 2.0
    elif torch_device_props["num_sm"] >= 68:
        sm_factor = 1.5
    elif torch_device_props["num_sm"] >= 46:
        sm_factor = 1.0
    else:
        sm_factor = 0.5

    if torch_device_props["max_shared_mem"] >= 98304:
        smem_factor = 1.5
    elif torch_device_props["max_shared_mem"] >= 49152:
        smem_factor = 1.0
    else:
        smem_factor = 0.5

    if torch_device_props["is_hip"]:
        base_size = min(base_size, 32768)
        sm_factor *= 0.75
        smem_factor *= 0.75

    calculated_size = int(base_size * min(sm_factor, smem_factor))
    final_size = min(reg_limit, calculated_size)

    final_size = triton.next_power_of_2(final_size)
    return max(MIN_SIZE, min(MAX_SIZE, final_size))


def _prev_power_of_2(n):
    # Triton requires powers of 2 for num_warps
    if n <= 0:
        return 1
    return 1 << (n.bit_length() - 1)


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = _calculate_max_fused_size()
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 65536:
        num_warps = 64 if not is_hip() else 16  # H100 has 64 schedulable warps maximum per SM
    elif BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    torch_device_props = _get_device_properties()
    max_threads_per_sm = torch_device_props["max_threads_per_sm"]
    warp_size = torch_device_props["warp_size"]
    max_warps_per_sm = max_threads_per_sm // warp_size

    num_warps = min(num_warps, max_warps_per_sm)

    # Triton requires num_warps to be a power of 2.
    num_warps = _prev_power_of_2(num_warps)

    return BLOCK_SIZE, num_warps


def _get_device_properties():
    device = torch.cuda.current_device()
    torch_device_props = torch.cuda.get_device_properties(device)

    max_shared_mem = 49152  # value for RTX 3090.
    if hasattr(torch_device_props, "shared_memory_per_block"):
        max_shared_mem = torch_device_props.shared_memory_per_block
    elif hasattr(torch_device_props, "sharedMemPerBlock"):
        max_shared_mem = torch_device_props.sharedMemPerBlock
    elif hasattr(torch_device_props, "max_shared_memory_per_block"):
        max_shared_mem = torch_device_props.max_shared_memory_per_block

    is_hip_device: bool = is_hip()

    return {
        "num_sm": torch_device_props.multi_processor_count,
        "max_shared_mem": max_shared_mem,
        "max_threads_per_sm": torch_device_props.max_threads_per_multi_processor,
        "warp_size": 64 if is_hip_device else 32,
        "max_regs": torch_device_props.regs_per_multiprocessor,
        "is_hip": is_hip_device,
    }


def calculate_num_stages():
    torch_device_props = _get_device_properties()

    reg_stages = 2
    if torch_device_props["max_regs"] > 65536:
        reg_stages = 8
    elif torch_device_props["max_regs"] >= 65536:
        reg_stages = 6
    elif torch_device_props["max_regs"] > 49152:
        reg_stages = 4
    elif torch_device_props["max_regs"] > 32768:
        reg_stages = 4
    elif torch_device_props["max_regs"] > 16384:
        reg_stages = 3

    smem_stages = 2
    if torch_device_props["max_shared_mem"] > 163840:
        smem_stages = 8
    elif torch_device_props["max_shared_mem"] > 131072:
        smem_stages = 6
    elif torch_device_props["max_shared_mem"] > 98304:
        smem_stages = 4
    elif torch_device_props["max_shared_mem"] >= 49152:
        smem_stages = 4
    elif torch_device_props["max_shared_mem"] > 32768:
        smem_stages = 3
    elif torch_device_props["max_shared_mem"] > 16384:
        smem_stages = 2
    else:
        smem_stages = 1

    num_sm = torch_device_props["num_sm"]
    sm_stages = 2
    if num_sm >= 108:
        sm_stages = 8
    elif num_sm >= 84:
        sm_stages = 6
    elif num_sm >= 68:
        sm_stages = 4
    elif num_sm >= 46:
        sm_stages = 3
    elif num_sm >= 28:
        sm_stages = 2
    else:
        sm_stages = 1

    max_threads_per_sm = torch_device_props["max_threads_per_sm"]
    thread_stages = 2
    if max_threads_per_sm >= 2048:
        thread_stages = 8
    elif max_threads_per_sm >= 1792:
        thread_stages = 6
    elif max_threads_per_sm >= 1536:
        thread_stages = 4
    elif max_threads_per_sm >= 1024:
        thread_stages = 3
    else:
        thread_stages = 2

    if torch_device_props["is_hip"]:
        reg_stages = max(1, reg_stages // 2)
        smem_stages = max(1, smem_stages // 2)
        sm_stages = max(1, sm_stages // 2)
        thread_stages = max(1, thread_stages // 2)

    final_stages = min(reg_stages, smem_stages, sm_stages, thread_stages)

    return max(1, min(8, final_stages))


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)
