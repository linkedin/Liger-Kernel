import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import torch_to_triton_dtype
from liger_kernel.utils import get_npu_multi_processor_count
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


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes the following:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    This is a commonly used pattern in the decoder layers of LLMs.
    Some examples:
    1. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/qwen3/modeling_qwen3.py#L271
    2. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/llama4/modeling_llama4.py#L393

    This kernel is inspired by the rms_norm forward kernel, and is adapted to support the residual addition in the forward pass.
    The backward pass is also adapted to support the residual addition in the backward pass.
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    S_ptr += row_idx * S_row_stride
    X_ptr += row_idx * X_row_stride
    R_ptr += row_idx * R_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)
    S_row = X_row + R_row
    tl.store(S_ptr + col_offsets, S_row, mask=mask)
    S_row_dtype = S_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        S_row = S_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(S_row_dtype)
        offset = offset.to(S_row_dtype)

    mean_square = tl.sum(S_row * S_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    S_row = S_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(S_row_dtype)

    Y_row = S_row * (offset + W_row)

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _fused_add_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
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
    rows_per_program: tl.constexpr,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_dS_out: tl.constexpr,
):
    """
    This kernel is adapted from the rms_norm backward kernel, and is adapted to support the residual
    addition in the backward pass. For the following code pattern:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    The gradient of hidden_states and residual comes out be exactly same. The value of this gradient is
    the sum of the gradient of the hidden_states in step 3 and the gradient of the residual in step 2.

    The backward pass computation logic is same as the rms_norm backward kernel, except that the gradient
    of the hidden_states in step 3 and the gradient of the residual in step 2 are summed up.
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride
    if has_dS_out:
        dS_out_ptr += row_start * dS_out_row_stride

    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset

    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row).to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row
        else:
            m = dY_row * W_row

        dX_row = rstd_row * m

        if has_dS_out:
            dS_out_row = tl.load(dS_out_ptr + col_offsets, mask=mask, other=0.0)
            dX_row += (rstd_row) * (
                -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
            ) + dS_out_row
            dS_out_ptr += dS_out_row_stride
        else:
            dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        # calculate the gradient of W
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
        else:
            # here X_row is already in fp32 (see previous if block)
            dW_row += dY_row * (X_row * rstd_row)

        tl.store(dX_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    R = R.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    # RSTD is always computed/stored in fp32 if we are using Llama or Gemma casting mode
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # Check constraints.
    assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    # TODO: add _block_fused_add_rms_norm_forward_kernel
    _fused_add_rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        X,
        X.stride(0),
        R,
        R.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,  # XPU-specific optimization
    )

    return Y.view(*shape), S.view(*shape), RSTD, BLOCK_SIZE, num_warps, casting_mode


def fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    dS_out = dS_out.view(-1, dim)
    S = S.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if S.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(S.device).multi_processor_count
    elif S.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(S.device).gpu_eu_count
    elif S.device.type == "npu":
        sm_count = get_npu_multi_processor_count()

    # fp32 for numerical stability especially.
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    # XPU-specific optimization
    kernel_args = {}
    if S.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    # TODO: add _block_fused_add_rms_norm_backward_kernel
    _fused_add_rms_norm_backward_kernel[grid](
        dY,
        dY.stride(0),
        dS_out,
        dS_out.stride(0),
        dX,
        dX.stride(0),
        S,
        S.stride(0),
        torch_to_triton_dtype[S.dtype],
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        offset,
        rows_per_program,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        has_dS_out=dS_out is not None,
        **kernel_args,  # XPU-specific optimization
    )

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)

    return dX, dX, dW  # dR is equal to dX


class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """
    Performs a fused operation that first adds a residual tensor to the hidden_states tensor (`X`), then applies RMSNorm (Root Mean Square Normalization) to the result using the weight tensor `W`, with optional offset and casting mode.

    This class implements the following sequence, commonly used in transformer decoder layers:
        1. hidden_states = residual + hidden_states
        2. residual = hidden_states (after addition)
        3. hidden_states = rmsnorm(hidden_states)

    Both the normalized hidden_states and the updated residual are returned as outputs.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 before starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    The `in_place` option determines whether to modify dY in-place to store dX. This defaults to `True` to save memory.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, R, W, eps, offset=0.0, casting_mode="llama", in_place=False):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        # TODO: add row_mode
        Y, S, RSTD, BLOCK_SIZE, num_warps, casting_mode = fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(S, W, RSTD)
        return Y, S

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        """
        Y: (B, T, H) or (BxT, H)
        """
        S, W, RSTD = ctx.saved_tensors
        dX, dR, dW = fused_add_rms_norm_backward(
            dY,
            dS_out,
            S,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
        )

        return dX, dR, dW, None, None, None, None, None
