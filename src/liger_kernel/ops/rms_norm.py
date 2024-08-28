import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


_CASTING_MODE_NONE = tl.constexpr(-1)
_CASTING_MODE_LLAMA = tl.constexpr(0)
_CASTING_MODE_GEMMA = tl.constexpr(1)


@triton.jit
def _rms_norm_forward(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    r_ptr,
    r_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    r_ptr += row_idx * r_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only inv_rms is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_rms = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(r_ptr, inv_rms)

    X_row = X_row * inv_rms

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    Y_row = X_row * (offset + W_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    r_ptr,
    r_row_stride,
    dW_ptr,
    dW_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    r_ptr += row_idx * r_row_stride
    dW_ptr += row_idx * dW_row_stride

    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    original_x_dtype = X_row.dtype

    # Get cached rms
    inv_rms_row = tl.load(r_ptr)

    W_row = W_row + offset

    # Different bacward graphs for different casting modes
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)
        m = (dY_row * W_row).to(tl.float32)
        dX_row = inv_rms_row * m

        dX_row += (inv_rms_row) * (
            -(1 / n_cols)
            * inv_rms_row
            * inv_rms_row
            * tl.sum(m * X_row, axis=0)
            * X_row
        )

    if casting_mode == _CASTING_MODE_GEMMA:
        dY_row, W_row, X_row = (
            dY_row.to(tl.float32),
            W_row.to(tl.float32),
            X_row.to(tl.float32),
        )
        dX_row = inv_rms_row * dY_row * W_row

        dX_row += (inv_rms_row) * (
            -(1 / n_cols)
            * inv_rms_row
            * inv_rms_row
            * tl.sum(dY_row * W_row * X_row, axis=0)
            * X_row
        )

    # calculate the gradient of W
    if casting_mode == _CASTING_MODE_LLAMA:
        dW_row = dY_row * (X_row * inv_rms_row).to(original_x_dtype)
    else:
        # here X_row is already in fp32 (see previous if block)
        dW_row = dY_row * (X_row * inv_rms_row)

    tl.store(dY_ptr + col_offsets, dX_row, mask=mask)
    tl.store(dW_ptr + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


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
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama"):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if not isinstance(casting_mode, int):
            assert (
                casting_mode in _str_to_casting_mode
            ), f"Invalid casting mode: {casting_mode}"
            casting_mode = _str_to_casting_mode[casting_mode]
        else:
            assert (
                casting_mode in _str_to_casting_mode.values()
            ), f"Invalid casting mode: {casting_mode}"

        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
        # r is to cache (1/rms) for each row
        # r is always computed/stored in fp32 if we are using Llama or Gemma casting mode
        r_dtype = (
            torch.float32
            if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value)
            else X.dtype
        )
        r = torch.empty(n_rows, dtype=r_dtype, device=X.device)

        # Check constraints.
        assert (
            X.shape[1] == W.shape[0]
        ), "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

        _rms_norm_forward[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            r,
            r.stride(0),
            n_cols,
            eps,
            offset,
            casting_mode,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """

        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = torch.empty_like(
            X,
            dtype=(
                torch.float32
                if ctx.casting_mode == _CASTING_MODE_GEMMA.value
                else W.dtype
            ),
        )

        # Here we use dY to store the value of dX to save memory
        _rms_norm_backward[(n_rows,)](
            dY,
            dY.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            r,
            r.stride(0),
            dW,
            dW.stride(0),
            n_cols,
            ctx.eps,
            ctx.offset,
            ctx.casting_mode,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        dW = torch.sum(dW, dim=0).to(W.dtype)
        return dX, dW, None, None, None
