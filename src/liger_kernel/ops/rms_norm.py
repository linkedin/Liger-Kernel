"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/rms_norm.py#L30
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22

Modifications made by Yanning Chen, 2024.
"""

import math
import operator

import torch
import triton
import triton.language as tl

# torch.distributed.tensor is a lazy submodule on torch 2.12+; bind it once at
# import so downstream ``torch.distributed.tensor.DTensor`` attribute access
# never raises AttributeError on a missing-attribute path. Only ImportError is
# expected (torch built without the distributed package); let other failures
# surface so they can be debugged rather than silently masked.
try:
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_device_multiprocessor_count
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


# Some torch builds don't eagerly import the ``torch.distributed.tensor`` submodule,
# so ``torch.distributed.tensor.DTensor`` can raise
# ``AttributeError: module 'torch.distributed' has no attribute 'tensor'``. Import
# DTensor defensively: the import triggers the submodule load when available and gives
# a class to isinstance against; when unavailable, ``()`` makes the isinstance checks
# a safe no-op (a plain, non-distributed tensor is never a DTensor).
try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:
    _DTensor = ()


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    else:
        # Scalar kernel params are specialized to fp32 by eager Triton but to
        # fp64 by Inductor when this kernel is launched from inside
        # torch.compile. An fp64 scalar silently promotes mean_square, rsqrt
        # and the weight multiply to float64, roughly halving throughput.
        # Pinning to fp32 is a no-op in eager and keeps both paths identical.
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(rstd_base, rstd)

    X_row = X_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        # Pin the fp64 scalar Inductor passes to fp32 (see _rms_norm_forward_kernel).
        W_row = W_row + offset.to(tl.float32)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride

        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        # Different backward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        dX_row = rstd_row * m

        dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        if elementwise_affine:
            # calculate the gradient of W
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if elementwise_affine:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


@triton.jit
def _block_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    else:
        # See _rms_norm_forward_kernel: pin fp64 scalars from Inductor to fp32.
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    X_row = X_row * rstd[:, None]

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)[None, :]
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        # Pin the fp64 scalar Inductor passes to fp32 (see _rms_norm_forward_kernel).
        W_row = W_row + offset.to(tl.float32)

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row[None, :]).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols) * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None] * X_row
        )

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                # TODO(tcc): use tl.sum(..., dtype=tl.float32) once we upgrade to triton>=3.3.0
                dW_row += tl.sum((dY_row * (X_row * rstd_row[:, None]).to(X_dtype)).to(tl.float32), 0)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    if elementwise_affine:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


@triton.jit
def _rms_group_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_groups,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grouped RMSNorm forward: X of shape (..., H) is viewed as (..., G, group_size) and each group is
    normalized independently: y = (x / rms(x_group)) * (offset + w_group).

    The launch grid covers every (token, group) pair, i.e. rows of the (n_rows * n_groups, n_cols) view.
    Row `r` corresponds to token `r // n_groups` and group `r % n_groups`; the weight slice for that
    group starts at `(r % n_groups) * n_cols` in W.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    group_id = row_idx % n_groups

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride
    # W is a flat (n_groups * n_cols,) tensor; group k owns the slice [k * n_cols, (k + 1) * n_cols)
    w_base = W_ptr + group_id * n_cols

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(w_base + col_offsets, mask=mask, other=0)

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)
    elif casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)
    else:
        # _CASTING_MODE_NONE: keep everything in the input dtype
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    if casting_mode != _CASTING_MODE_NONE:
        # Match the scalar discipline in the generic RMSNorm forward kernel.
        # Inductor otherwise specializes Python scalar parameters as fp64 and
        # silently promotes the grouped normalization math under torch.compile.
        eps = tl.cast(eps, tl.float32)
        offset = tl.cast(offset, tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    tl.store(rstd_base, rstd)

    X_row = X_row * rstd

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    Y_row = X_row * (offset + W_row)

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_group_norm_backward_row(dY_row, X_row, W_row, rstd_row, n_cols, casting_mode: tl.constexpr, X_dtype):
    """Compute one generic grouped-RMSNorm backward row."""
    X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_LLAMA:
        m = (dY_row * W_row).to(tl.float32)
    elif casting_mode == _CASTING_MODE_GEMMA:
        dY_row = dY_row.to(tl.float32)
        m = dY_row * W_row
    else:
        m = dY_row * W_row

    dX_row = rstd_row * m
    dX_row += rstd_row * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

    if casting_mode == _CASTING_MODE_LLAMA:
        dW_update = dY_row * (X_row * rstd_row).to(X_dtype)
    else:
        dW_update = dY_row * (X_row * rstd_row)
    return dX_row.to(X_dtype), dW_update


@triton.jit
def _rms_group_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_groups,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grouped RMSNorm backward. The grid is (num_row_blocks * n_groups,); program `pid` handles the
    column slice of group `k = pid % n_groups` for tokens assigned to row block `pid // n_groups`.
    Keeping a single group per program lets each program write one complete compact dW row.
    """
    pid = tl.program_id(0).to(tl.int64)
    row_block_id = pid // n_groups
    group_id = pid % n_groups

    row_start = row_block_id * rows_per_program
    row_end = min(row_start + rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # W is a flat (n_groups * n_cols,) tensor; group k owns the slice [k * n_cols, (k + 1) * n_cols)
    W_row = tl.load(W_ptr + group_id * n_cols + col_offsets, mask=mask, other=0.0)
    # ``offset`` is a Triton scalar in eager but a Python scalar while Inductor
    # analyzes kernel mutations. ``tl.cast`` handles both representations.
    W_row = W_row + tl.cast(offset, tl.float32)

    for row_idx in range(row_start, row_end):
        # Flatten (token, group) to the row index of the (n_rows * n_groups, n_cols) views
        flat_row = row_idx * n_groups + group_id

        dy_base = dY_ptr + flat_row * dY_row_stride
        dx_base = dX_ptr + flat_row * dX_row_stride
        x_base = X_ptr + flat_row * X_row_stride
        rstd_base = RSTD_ptr + flat_row * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        rstd_row = tl.load(rstd_base)
        dX_row, dW_update = _rms_group_norm_backward_row(dY_row, X_row, W_row, rstd_row, n_cols, casting_mode, X_dtype)
        dW_row += dW_update
        tl.store(dx_base + col_offsets, dX_row, mask=mask)

    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=mask)


@triton.jit
def _rms_group_norm_backward_add_kernel(
    dY0_ptr,
    dY1_ptr,
    dY2_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_groups: tl.constexpr,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    N_GRADIENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Grouped RMSNorm backward with an in-register sum of two or three upstream gradients."""
    pid = tl.program_id(0).to(tl.int64)
    row_block_id = pid // n_groups
    group_id = pid % n_groups

    row_start = row_block_id * rows_per_program
    row_end = min(row_start + rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    W_row = tl.load(W_ptr + group_id * n_cols + col_offsets, mask=mask, other=0.0)
    W_row = W_row + tl.cast(offset, tl.float32)

    for row_idx in range(row_start, row_end):
        flat_row = row_idx * n_groups + group_id
        dy_offsets = flat_row * dY_row_stride + col_offsets
        dx_base = dX_ptr + flat_row * dX_row_stride
        x_base = X_ptr + flat_row * X_row_stride
        rstd_base = RSTD_ptr + flat_row * RSTD_row_stride

        dY0_row = tl.load(dY0_ptr + dy_offsets, mask=mask, other=0.0)
        dY_dtype = dY0_row.dtype
        dY1_row = tl.load(dY1_ptr + dy_offsets, mask=mask, other=0.0)
        # Sum in gradient dtype before Gemma-mode fp32 math. The two-gradient specialization avoids
        # loading an autograd-materialized zero while preserving the order of the real additions.
        dY_row = (dY0_row + dY1_row).to(dY_dtype)
        if N_GRADIENTS == 3:
            dY2_row = tl.load(dY2_ptr + dy_offsets, mask=mask, other=0.0)
            dY_row = (dY_row + dY2_row).to(dY_dtype)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)
        dX_row, dW_update = _rms_group_norm_backward_row(dY_row, X_row, W_row, rstd_row, n_cols, casting_mode, X_dtype)
        dW_row += dW_update
        tl.store(dx_base + col_offsets, dX_row, mask=mask)

    # Every program owns one complete group slice, so a compact [program, group_size] buffer is
    # sufficient for both this multi-gradient path and the generic grouped backward.
    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode, n_groups=None):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]

    if n_groups is not None:
        # Grouped RMSNorm: normalize each of the `n_groups` slices of the last dim independently.
        # The (..., dim) tensor is viewed as (n_rows * n_groups, group_size) so every program in the
        # grouped kernels sees a plain row of length `group_size`.
        if not isinstance(n_groups, int) or n_groups <= 0:
            raise ValueError(f"n_groups must be a positive integer, got {n_groups}.")
        if W is None:
            raise ValueError("Grouped RMSNorm requires an elementwise-affine weight.")
        if W.dim() != 1 or W.numel() != dim or dim % n_groups != 0:
            raise ValueError(f"Weight shape {tuple(W.shape)} incompatible with n_groups={n_groups} and last dim {dim}.")
        group_size = dim // n_groups
        X = X.view(-1, group_size)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        # RSTD is to cache rstd for each (row, group) pair
        rstd_dtype = (
            torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
        )
        RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

        with device_context(X.device):
            _rms_group_norm_forward_kernel[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                RSTD,
                RSTD.stride(0),
                n_groups,
                n_cols,
                eps,
                offset,
                casting_mode,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode

    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    # RSTD is always computed/stored in fp32 if we are using Llama or Gemma casting mode
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        # Check constraints.
        assert X.shape[1] == W.shape[0], (
            "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"
        )
        elementwise_affine = True
    else:
        elementwise_affine = False

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)
    with device_context(X.device):
        if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
            _rms_norm_forward_kernel[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
        else:
            BLOCK_ROW = 16
            kernel_args["BLOCK_ROW"] = BLOCK_ROW
            _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def rms_group_norm_backward_add3(
    grad0,
    grad1,
    grad2,
    X,
    W,
    RSTD,
    offset,
    casting_mode,
    BLOCK_SIZE,
    num_warps,
    n_groups,
    n_gradients=3,
):
    """Run grouped RMSNorm backward while summing BF16/FP16 gradients in registers."""
    if n_gradients not in (2, 3):
        raise ValueError(f"n_gradients must be 2 or 3, got {n_gradients}.")
    if n_gradients == 3 and grad2 is None:
        raise ValueError("grad2 must be provided when n_gradients=3.")
    shape = grad0.shape
    dim = shape[-1]
    group_size = dim // n_groups
    grad0 = grad0.view(-1, group_size)
    grad1 = grad1.view(-1, group_size)
    # Triton still requires a valid pointer for its compile-time-elided third input.
    grad2 = grad1 if n_gradients == 2 else grad2
    grad2 = grad2.view(-1, group_size)
    n_rows, n_cols = grad0.shape
    n_token_rows = n_rows // n_groups

    sm_count = get_device_multiprocessor_count(X.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # The kernel writes every dX element and every compact partial-dW element, so neither output
    # needs a zero-fill. Program order is [row_block, group], matching this reduction view.
    dX = torch.empty_like(grad0)
    partial_dW = torch.empty((sm_count * n_groups, group_size), dtype=torch.float32, device=W.device)
    rows_per_program = math.ceil(n_token_rows / sm_count)

    with device_context(X.device):
        _rms_group_norm_backward_add_kernel[(sm_count * n_groups,)](
            grad0,
            grad1,
            grad2,
            grad0.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            RSTD,
            RSTD.stride(0),
            partial_dW,
            partial_dW.stride(0),
            n_token_rows,
            n_groups,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            N_GRADIENTS=n_gradients,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    dW = partial_dW.view(sm_count, n_groups, group_size).sum(dim=0).reshape(dim).to(W.dtype)
    return dX.view(*shape), dW


def rms_group_norm_backward_add2(
    grad0,
    grad1,
    X,
    W,
    RSTD,
    offset,
    casting_mode,
    BLOCK_SIZE,
    num_warps,
    n_groups,
):
    """Run grouped RMSNorm backward for exactly two real upstream gradients."""
    return rms_group_norm_backward_add3(
        grad0,
        grad1,
        None,
        X,
        W,
        RSTD,
        offset,
        casting_mode,
        BLOCK_SIZE,
        num_warps,
        n_groups,
        n_gradients=2,
    )


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode, n_groups=None):
    shape = dY.shape
    dim = shape[-1]

    if n_groups is not None:
        group_size = dim // n_groups
        dY = dY.view(-1, group_size)
        n_rows, n_cols = dY.shape  # (n_tokens * n_groups, group_size)
        n_token_rows = n_rows // n_groups

        sm_count = get_device_multiprocessor_count(X.device)

        # Programs are (row_block, group) pairs and each writes one complete compact group slice.
        # No columns are left unwritten, so this workspace does not need a zero-fill.
        _dW = torch.empty((sm_count * n_groups, group_size), dtype=torch.float32, device=W.device)

        if n_cols > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        rows_per_program = math.ceil(n_token_rows / sm_count)

        if in_place is True:
            dX = dY
        else:
            dX = torch.empty_like(dY)

        with device_context(X.device):
            _rms_group_norm_backward_kernel[(sm_count * n_groups,)](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                torch_to_triton_dtype[X.dtype],
                W,
                RSTD,
                RSTD.stride(0),
                _dW,
                _dW.stride(0),
                n_token_rows,
                n_groups,
                n_cols,
                offset,
                rows_per_program,
                casting_mode,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        dW = _dW.view(sm_count, n_groups, group_size).sum(dim=0).reshape(dim).to(W.dtype)
        return dX.view(*shape), dW

    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    if W is not None:
        # fp32 for numerical stability especially.
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
        elementwise_affine = True
    else:
        _dW = None
        elementwise_affine = False

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    with device_context(X.device):
        if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
            _rms_norm_backward_kernel[grid](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                torch_to_triton_dtype[X.dtype],
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                _dW,
                _dW.stride(0) if elementwise_affine else 0,
                n_rows,
                n_cols,
                offset,
                rows_per_program,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
        else:
            BLOCK_ROW = 16
            kernel_args["BLOCK_ROW"] = BLOCK_ROW
            _block_rms_norm_backward_kernel[grid](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                torch_to_triton_dtype[X.dtype],
                W,
                W.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                _dW,
                _dW.stride(0) if elementwise_affine else 0,
                n_rows,
                n_cols,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                **kernel_args,  # XPU-specific optimization
            )
    dX = dX.view(*shape)

    if elementwise_affine:
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = None

    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None, n_groups=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        n_groups: optional number of groups the last dim of X is split into for grouped RMSNorm.
        Each of the `n_groups` slices of size H // n_groups is normalized independently against its
        own weight slice `W[g * (H // n_groups) : (g + 1) * (H // n_groups)]`.
        """
        if isinstance(X, _DTensor):
            # Input tensor is output of a tensor parallel module and
            # needs to be gathered to a local tensor to compute
            # RMSE layer norm on each TP worker.
            # TODO: support CP.
            X = X.full_tensor()

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
            X, W, eps, offset, casting_mode, row_mode, n_groups=n_groups
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.n_groups = n_groups
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if isinstance(dY, _DTensor):
            # Gradients are output of a tensor parallel module and
            # needs to be gathered to a local tensor for computing RMSE layer.
            # TODO: support CP.
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
            ctx.row_mode,
            n_groups=ctx.n_groups,
        )
        return dX, dW, None, None, None, None, None, None
