# adapted from: https://github.com/pytorch-labs/applied-ai/blob/main/kernels/triton/inference/fp8/splitk_gemm_fp8.py
# Valid Architectures: [SM_89, SM_90(a)]
import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_gemm_settings,
    check_compute_capability_for_fp8,
    ensure_contiguous,
)

dtypes = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2, torch.float8_e4m3fn: 3}


"""
Split-K GEMM is preferred in scenarios where:
- The matrix dimensions (m, n, k) are large, leading to the need for splitting the computation across multiple blocks.
- The available shared memory per block is insufficient to handle the required shared memory for a single block.
    - in our case this is 100KB SMEM per block.
"""


@triton.jit
def grouped_launch(
    pid,
    m,  # rows
    n,  # cols
    block_m: tl.constexpr,  # rows in a block
    block_n: tl.constexpr,  # cols in a block
    group_m: tl.constexpr,  # blocks in group along row dimension
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    pid_in_group = pid % width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel_forward(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_k: tl.constexpr,
    group_m: tl.constexpr,
    compute_type_int: tl.constexpr,
):
    """
    FP8 GEMM with FP8 E4M3 FP8 representation.

    This kernel performs matmul of two input matrices A and B, and stores the result in matrix C.
    The computation is split across multiple blocks to handle large matrix dimensions and efficiently utilize GPU resources.

    Design:
    - The kernel uses a split-K strategy to divide the computation of the K dimension across multiple blocks.
    - Each block computes a partial result for a submatrix of C, and the results are accumulated using atomic operations.
    - The kernel supports different compute types (FP16, BF16, FP32, FP8) based on the input argument `compute_type_int`.

    Args:
        a_ptr: Pointer to the first input matrix A.
        b_ptr: Pointer to the second input matrix B.
        c_ptr: Pointer to the output matrix C.
        stride_am: Stride of matrix A along the m dimension.
        stride_ak: Stride of matrix A along the k dimension.
        stride_bk: Stride of matrix B along the k dimension.
        stride_bn: Stride of matrix B along the n dimension.
        stride_cm: Stride of matrix C along the m dimension.
        stride_cn: Stride of matrix C along the n dimension.
        m: Number of rows in matrix A and matrix C.
        n: Number of columns in matrix B and matrix C.
        k: Number of columns in matrix A and rows in matrix B.
        block_m: Number of rows in a block.
        block_n: Number of columns in a block.
        block_k: Number of columns in a block for the k dimension.
        split_k: Factor to split the k dimension.
        group_m: Number of blocks in a group along the row dimension.
        compute_type_int: Integer representing the compute type (0: float16, 1: bfloat16, 2: float32, 3: float8_e4m3fn).

    Returns:
        None

    Structural Representation:
    - Two matrices A (m x k) and B (k x n) | C = A @ B.
    - Computation is divided into blocks of size (block_m x block_n) and splits the K dimension into chunks
      of size (block_k).

    ```
    Matrix A (m x k):
    +-------------------+-------------------+-------------------+
    |       Block 0     |       Block 1     |       Block 2     |
    |                   |                   |                   |
    |       (m x k0)    |       (m x k1)    |       (m x k2)    |
    +-------------------+-------------------+-------------------+

    Matrix B (k x n):
    +-------------------+-------------------+-------------------+
    |       Block 0     |       Block 1     |       Block 2     |
    |                   |                   |                   |
    |       (k0 x n)    |       (k1 x n)    |       (k2 x n)    |
    +-------------------+-------------------+-------------------+

    Matrix C (m x n):
    +-------------------+-------------------+-------------------+
    |       Block 0     |       Block 1     |       Block 2     |
    |                   |                   |                   |
    |       (m x n)     |       (m x n)     |       (m x n)     |
    +-------------------+-------------------+-------------------+
    ```

    - Each block computes a partial result for a submatrix of C and accumulates the results using `triton.language.atomic_add`.
    """
    if compute_type_int == 0 or compute_type_int == 3:
        compute_type = tl.float16
    elif compute_type_int == 1:
        compute_type = tl.bfloat16
    elif compute_type_int == 2:
        compute_type = tl.float32

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = tl.multiple_of(pid_m * block_m + tl.arange(0, block_m), block_m)
    offs_n = tl.multiple_of(pid_n * block_n + tl.arange(0, block_n), block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_ in range(0, grid_k, step=2):
        k_remaining = k - k_ * (block_k * split_k)

        mask_a = offs_k[None, :] < k_remaining
        mask_b = offs_k[:, None] < k_remaining

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # fp8 input dot product (supported types: [fp8e4nv, fp8e5, fp8e4b15])
        acc = tl.dot(a, b, acc)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(c_ptrs, acc.to(compute_type), mask=mask)


def forward_kernel(ctx, a, b):
    ctx.save_for_backward(a, b)
    input_dtype = a.dtype

    # cast to FP8
    # structure:
    #   | 1 bit sign | 4 bit exponent | 3 bit mantissa |
    a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)

    m, k = a.shape
    _, n = b.shape

    block_m, block_n, block_k, num_stages, num_warps, split_k, group_m = (
        calculate_gemm_settings(m, n, k)
    )

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    gemm_split_k_kernel_forward[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        group_m,
        num_stages=num_stages,
        num_warps=num_warps,
        compute_type_int=dtypes[input_dtype],
    )

    return c


@triton.jit
def gemm_split_k_kernel_backward(
    grad_output_ptr,
    a_ptr,
    b_ptr,
    grad_a_ptr,
    grad_b_ptr,
    stride_gom,
    stride_gon,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_gam,
    stride_gak,
    stride_gbm,
    stride_gbn,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)

    offs_m = tl.multiple_of(pid_m * block_m + tl.arange(0, block_m), block_m)
    offs_n = tl.multiple_of(pid_n * block_n + tl.arange(0, block_n), block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_gom = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_gon = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    grad_output_ptrs = grad_output_ptr + (
        offs_gom[:, None] * stride_gom + offs_gon[None, :] * stride_gon
    )
    a_ptrs = a_ptr + (offs_gom[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_gon[None, :] * stride_bn)

    grad_a_acc = tl.zeros((block_m, block_k), dtype=tl.float32)
    grad_b_acc = tl.zeros((block_k, block_n), dtype=tl.float32)

    for k_ in range(0, grid_k, step=2):
        k_remaining = k - k_ * (block_k * split_k)

        mask_a = offs_k[None, :] < k_remaining
        mask_b = offs_k[:, None] < k_remaining

        grad_output = tl.load(grad_output_ptrs, mask=mask_a, other=0.0)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        grad_a_acc += tl.dot(grad_output, b.T)
        grad_b_acc += tl.dot(a.T, grad_output)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk

    grad_a_acc = grad_a_acc.to(tl.float16)
    grad_b_acc = grad_b_acc.to(tl.float16)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    grad_a_ptrs = grad_a_ptr + (
        offs_m[:, None] * stride_gam + offs_k[None, :] * stride_gak
    )
    grad_b_ptrs = grad_b_ptr + (
        offs_k[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
    )

    mask_a = (offs_m < m)[:, None] & (offs_k < k)[None, :]
    mask_b = (offs_k < k)[:, None] & (offs_n < n)[None, :]

    tl.atomic_add(grad_a_ptrs, grad_a_acc, mask=mask_a)
    tl.atomic_add(grad_b_ptrs, grad_b_acc, mask=mask_b)


def backward_kernel(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = torch.zeros_like(a, dtype=torch.float16)
    grad_b = torch.zeros_like(b, dtype=torch.float16)

    grad_output_fp16 = grad_output.to(torch.float16)
    a_fp16 = a.to(torch.float16)
    b_fp16 = b.to(torch.float16)

    m, k = a.shape
    _, n = b.shape

    block_m, block_n, block_k, num_stages, num_warps, split_k, group_m = (
        calculate_gemm_settings(m, n, k)
    )

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    gemm_split_k_kernel_backward[grid](
        grad_output_fp16,
        a_fp16,
        b_fp16,
        grad_a,
        grad_b,
        grad_output.stride(0),
        grad_output.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        grad_a.stride(0),
        grad_a.stride(1),
        grad_b.stride(0),
        grad_b.stride(1),
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        group_m,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    if ctx.needs_input_grad[0]:
        grad_a = grad_a.to(a.dtype)
    if ctx.needs_input_grad[1]:
        grad_b = grad_b.to(b.dtype)

    return grad_a, grad_b


class LigerFP8GemmSplitKFunction(torch.autograd.Function):
    @staticmethod
    @check_compute_capability_for_fp8
    @ensure_contiguous
    def forward(ctx, a, b):
        return forward_kernel(ctx, a, b)

    @staticmethod
    @check_compute_capability_for_fp8
    @ensure_contiguous
    def backward(ctx, grad_output):
        return backward_kernel(ctx, grad_output)
