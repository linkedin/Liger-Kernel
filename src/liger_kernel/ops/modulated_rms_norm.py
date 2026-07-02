"""
This file extends Liger RMSNorm, which incorporates code from Unsloth licensed
under the Apache License, Version 2.0. See src/liger_kernel/ops/rms_norm.py for
the original attribution.
"""

import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import is_dtensor
from liger_kernel.ops.utils import set_large_grf_mode
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _modulated_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    Scale_ptr,
    Scale_row_stride,
    Shift_ptr,
    Shift_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    mod_row_idx = row_idx // rows_per_modulation
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride
    scale_base = Scale_ptr + mod_row_idx * Scale_row_stride
    rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    Scale_row = tl.load(scale_base + col_offsets, mask=mask, other=0.0)
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    if has_shift:
        shift_base = Shift_ptr + mod_row_idx * Shift_row_stride
        Shift_row = tl.load(shift_base + col_offsets, mask=mask, other=0.0)

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(rstd_base, rstd)

    X_row = X_row * rstd

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    Y_row = Y_row * (1.0 + Scale_row)
    if has_shift:
        Y_row = Y_row + Shift_row

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _modulated_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    Scale_ptr,
    Scale_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    dScale_ptr,
    dScale_row_stride,
    dShift_ptr,
    dShift_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset

    for row_idx in range(row_start, row_end):
        mod_row_idx = row_idx // rows_per_modulation
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride
        x_base = X_ptr + row_idx * X_row_stride
        scale_base = Scale_ptr + mod_row_idx * Scale_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride
        dscale_base = dScale_ptr + mod_row_idx * dScale_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        Scale_row = tl.load(scale_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)
        X_norm = X_row * rstd_row
        Mod_row = 1.0 + Scale_row
        dRms_row = dY_row * Mod_row

        if casting_mode == _CASTING_MODE_LLAMA:
            X_norm_for_output = X_norm.to(X_dtype)
            if elementwise_affine:
                rms_output = X_norm_for_output * W_row
                m = (dRms_row * W_row).to(tl.float32)
            else:
                rms_output = X_norm_for_output
                m = dRms_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dRms_row = dRms_row.to(tl.float32)
            if elementwise_affine:
                rms_output = (X_norm * W_row).to(X_dtype)
                m = dRms_row * W_row
            else:
                rms_output = X_norm.to(X_dtype)
                m = dRms_row

        else:
            if elementwise_affine:
                rms_output = X_norm * W_row
                m = dRms_row * W_row
            else:
                rms_output = X_norm
                m = dRms_row

        dX_row = rstd_row * m
        dX_row += rstd_row * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dRms_row * X_norm.to(X_dtype)
            else:
                dW_row += dRms_row * X_norm

        dScale_row = dY_row * rms_output
        if rows_per_modulation == 1:
            tl.store(dscale_base + col_offsets, dScale_row, mask=mask)
        else:
            tl.atomic_add(dscale_base + col_offsets, dScale_row, sem="relaxed", mask=mask)
        if has_shift:
            dshift_base = dShift_ptr + mod_row_idx * dShift_row_stride
            if rows_per_modulation == 1:
                tl.store(dshift_base + col_offsets, dY_row, mask=mask)
            else:
                tl.atomic_add(dshift_base + col_offsets, dY_row, sem="relaxed", mask=mask)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if elementwise_affine:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def _check_modulation_shape(X, scale, shift):
    dim = X.shape[-1]
    assert scale.numel() % dim == 0, "Scale element count must be a multiple of the hidden size."
    n_rows = X.numel() // dim
    scale_rows = scale.numel() // dim
    assert scale_rows > 0, "Scale must have at least one row."
    assert n_rows % scale_rows == 0, "Scale rows must divide hidden state rows for broadcasting."

    if shift is not None:
        assert shift.numel() == scale_rows * dim, "Shift must use the same broadcast rows as scale."

    return scale_rows, n_rows // scale_rows


def modulated_rms_norm_forward(X, W, scale, shift, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    scale_rows, rows_per_modulation = _check_modulation_shape(X, scale, shift)

    X = X.view(-1, dim)
    scale = scale.view(scale_rows, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension between X and W."
        elementwise_affine = True
    else:
        elementwise_affine = False

    has_shift = shift is not None
    if has_shift:
        shift = shift.view(scale_rows, dim)
    else:
        shift = scale

    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    _modulated_rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0) if elementwise_affine else 0,
        scale,
        scale.stride(0),
        shift,
        shift.stride(0) if has_shift else 0,
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        casting_mode,
        elementwise_affine=elementwise_affine,
        has_shift=has_shift,
        rows_per_modulation=rows_per_modulation,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )

    return Y.view(*shape), RSTD, BLOCK_SIZE, num_warps, casting_mode, rows_per_modulation


def modulated_rms_norm_backward(
    dY,
    X,
    W,
    scale,
    shift,
    RSTD,
    offset,
    casting_mode,
    BLOCK_SIZE,
    num_warps,
    rows_per_modulation,
    in_place,
):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    X = X.view(-1, dim)
    scale_shape = scale.shape
    scale = scale.view(-1, dim)
    n_rows, n_cols = dY.shape
    scale_rows = scale.shape[0]

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    elif X.device.type == "npu":
        sm_count = get_npu_core_count()

    elementwise_affine = W is not None
    if elementwise_affine:
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
    else:
        _dW = None

    modulation_grad_dtype = scale.dtype if rows_per_modulation == 1 else torch.float32
    dScale = torch.empty((scale_rows, n_cols), dtype=modulation_grad_dtype, device=scale.device)
    if rows_per_modulation != 1:
        dScale.zero_()

    has_shift = shift is not None
    if has_shift:
        shift_shape = shift.shape
        shift = shift.view(scale_rows, dim)
        modulation_grad_dtype = shift.dtype if rows_per_modulation == 1 else torch.float32
        dShift = torch.empty((scale_rows, n_cols), dtype=modulation_grad_dtype, device=shift.device)
        if rows_per_modulation != 1:
            dShift.zero_()
    else:
        shift_shape = None
        dShift = dScale

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    kernel_args = {}
    if X.device.type == "xpu":
        set_large_grf_mode(kernel_args)

    _modulated_rms_norm_backward_kernel[grid](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        W,
        W.stride(0) if elementwise_affine else 0,
        scale,
        scale.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0) if elementwise_affine else 0,
        dScale,
        dScale.stride(0),
        dShift,
        dShift.stride(0) if has_shift else 0,
        n_rows,
        n_cols,
        offset,
        rows_per_program,
        casting_mode,
        elementwise_affine=elementwise_affine,
        has_shift=has_shift,
        rows_per_modulation=rows_per_modulation,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype) if elementwise_affine else None
    dScale = dScale.to(scale.dtype).view(*scale_shape) if dScale.dtype != scale.dtype else dScale.view(*scale_shape)
    if has_shift:
        dShift = dShift.to(shift.dtype).view(*shift_shape) if dShift.dtype != shift.dtype else dShift.view(*shift_shape)
    else:
        dShift = None

    return dX, dW, dScale, dShift


class LigerModulatedRMSNormFunction(torch.autograd.Function):
    """
    Performs modulated RMSNorm in a single fused kernel: ``y = (1 + scale) * RMSNorm(x) + shift``.

    The base RMSNorm matches ``LigerRMSNormFunction`` semantics (``offset``, ``casting_mode``,
    ``in_place`` behave identically). On top of that, ``scale`` and the optional ``shift`` apply
    an affine modulation commonly used in Diffusion Transformer style architectures (AdaLN).

    Broadcasting of ``scale`` / ``shift`` is determined by the leading shape, given hidden size H
    and a flattened token count N = X.numel() / H:
    - shape ``(H,)`` or ``(1, H)``: shared across every token (N rows reuse one modulation row).
    - shape ``(B, H)`` with 3D X of shape ``(B, T, H)``: shared along the sequence axis.
    - shape ``(N, H)`` (or matching X): per-token modulation, no atomic accumulation needed.
    In general, ``scale.numel() // H`` must divide N. ``shift`` (if provided) must broadcast
    identically to ``scale``.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, scale, shift, eps, offset=0.0, casting_mode="llama", in_place=True):
        if is_dtensor(X):
            # Match LigerRMSNormFunction: gather TP-sharded input to a local tensor.
            X = X.full_tensor()
        Y, RSTD, BLOCK_SIZE, num_warps, casting_mode, rows_per_modulation = modulated_rms_norm_forward(
            X,
            W,
            scale,
            shift,
            eps,
            offset,
            casting_mode,
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.rows_per_modulation = rows_per_modulation
        ctx.has_weight = W is not None
        ctx.has_shift = shift is not None
        if W is not None and shift is not None:
            ctx.save_for_backward(X, W, scale, shift, RSTD)
        elif W is not None:
            ctx.save_for_backward(X, W, scale, RSTD)
        elif shift is not None:
            ctx.save_for_backward(X, scale, shift, RSTD)
        else:
            ctx.save_for_backward(X, scale, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        if is_dtensor(dY):
            dY = dY.full_tensor()
        if ctx.has_weight and ctx.has_shift:
            X, W, scale, shift, RSTD = ctx.saved_tensors
        elif ctx.has_weight:
            X, W, scale, RSTD = ctx.saved_tensors
            shift = None
        elif ctx.has_shift:
            X, scale, shift, RSTD = ctx.saved_tensors
            W = None
        else:
            X, scale, RSTD = ctx.saved_tensors
            W = None
            shift = None

        dX, dW, dScale, dShift = modulated_rms_norm_backward(
            dY,
            X,
            W,
            scale,
            shift,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.rows_per_modulation,
            ctx.in_place,
        )

        return dX, dW, dScale, dShift, None, None, None, None
