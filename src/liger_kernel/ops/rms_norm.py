import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    r_ptr += row_idx * r_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)

    # trick: row_var is tiny compared to X_row because it just has one per row we can save 4 ops (*, sum, /, rqrt) if we cache it
    tl.store(r_ptr, inv_var)

    normed = X_row * inv_var

    output = normed * W_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)


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
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / var(x)) * (dy * w - (1/N) * (dy * w) dot x) * x
    dw = sum(dy * (x / var(x)))
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

    # Get saved row variance
    inv_var = tl.load(r_ptr)

    normed = X_row * inv_var

    dY_W = dY_row * W_row
    dY_normed = dY_row * normed

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dY_ptr + col_offsets, output, mask=mask)

    # calculate the gradient of W
    tl.store(dW_ptr + col_offsets, dY_normed, mask=mask)


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda")
        r = torch.empty(n_rows, dtype=X.dtype, device="cuda")

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
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = torch.zeros_like(X)

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
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        dW = torch.sum(dW, dim=0)
        return dX, dW, None
