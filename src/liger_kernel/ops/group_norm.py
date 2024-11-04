import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
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

MAX_FUSED_SIZE = 65536

@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride, # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride
    
    # Compute mean
    s = 0.0
    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        s += tl.sum(X)
    
    m = s/hidden_size
    
    # Compute variance
    variance = 0.0
    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        # We need to mask out of index with mean to ensure that the variance remains unaffected
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
        diff = X - m
        variance += tl.sum(diff * diff)
    
    variance = variance / hidden_size
    # 1/std
    rstd = rsqrt(variance + eps)
    

    # Normalize
    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
        Y = (X - m) * rstd
        tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

    
    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)

        
@triton.jit
def _group_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_channels, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    W_ptr,  # pointer to weights, shape (n_channels)
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_ptr_row_stride,  # stride of each column in mean
    Mean_ptr_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_groups, hidden_size)
    DW_ptr,  # pointer to weights grad, shape (n_channels)
    DB_ptr,  # pointer to bias grad, shape (n_channels)
    UPSTREAM_ptr,  # pointer to output grad, shape (n_rows, n_channels, hidden_size)
    hidden_size: tl.constexpr, # hidden size
    channels_per_group: tl.constexpr, # number of groups in group norm
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    group_idx = channel_idx // channels_per_group

    # X_col_stide will correspond to the number of groups
    X_ptr += batch_idx * X_row_stride + channel_idx * X_col_stride
    DX_ptr += batch_idx * X_row_stride + channel_idx * X_col_stride
    UPSTREAM_ptr += batch_idx * X_row_stride + channel_idx * X_col_stride

    # Mean and rstd are the same shape so have the same strides
    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    W = tl.load(W_ptr + channel_idx)
    
    dW = 0.0
    dB = 0.0

    for i in range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=mean)
        UPSTREAM_grad = tl.load(UPSTREAM_ptr + hidden_size_offsets, mask=mask, other=0.0)
        """
        Y = (X - mean) * rstd

        h(x) = rstd = 1/(sqrt(var + eps))
        f(x) = x * h(x) = X * rstd
        g(x) = - mean * h(x) = - mean * rstd

        Y = f(x) + g(x)
        dy_dx = df_dx + dg_dx
        """

        # dh_dx = -0.5 * (rstd**3) * dvar_dx
        c1 = 1 / hidden_size
        c2 = X - mean
        # c4 = tl.sum(c2)
        # dmean_dx = c1
        # dvar_dx = c4 * c1
        # drstd_dx = - (rstd*rstd*rstd) * dvar_dx

        # df_dx = rstd + X * drstd_dx
        # dg_dx = - (dmean_dx * rstd + mean * drstd_dx)
        # dY_dx = df_dx + dg_dx

        # dX = W * UPSTREAM_grad * dY_dx
        
        c3 = c2 * rstd
        dW += tl.sum(UPSTREAM_grad * c3)
        dB += tl.sum(UPSTREAM_grad)

        x_hat = (X - mean) * rstd
        wdy = W * UPSTREAM_grad
        c1 = tl.sum(x_hat * wdy) / (hidden_size * num_groups)
        c2 = tl.sum(wdy, axis=0) / (hidden_size * num_groups)
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + hidden_size_offsets, dx.to(dtype), mask=mask)

        # tl.store(DX_ptr + hidden_size_offsets, dX, mask=mask)
    
    # Need to ensure additions to the same channel are atomic
    tl.atomic_add(DW_ptr + channel_idx, dW.to(dtype))
    tl.atomic_add(DB_ptr + channel_idx, dB.to(dtype))

def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    # Reshape X so that the mean and std are computed across the groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    # print(f"Mean is {X.view(-1).mean()}")
    # print(f"RSTD is {1/torch.sqrt(torch.var(X.view(-1), unbiased=False) + 1e-6)}")
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    # print(f"Init mean is {Mean}")
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    # print(f"Init RSTD is {RSTD}")
    
    _group_norm_forward_kernel[(batch_size, num_groups)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # print(f"After Init mean {Mean}")
    # print(f"After Init rstd {RSTD}")
    Y = Y.view(*shape)
    affine_shape = [1] * len(shape)
    affine_shape[1] = num_channels
    Y = Y * W.view(affine_shape) + B.view(affine_shape)
    return Y, X.view(*shape), Mean, RSTD, BLOCK_SIZE


def group_norm_backward(dY, X, W, B, Mean, RSTD, num_channels, num_groups):
    # print(f"Sum of upstream is : {dY.sum()}")
    shape = dY.shape
    batch_size = shape[0]
    hidden_size = dY.shape[-1]
    channels_per_group = num_channels // num_groups
    DX = torch.empty((batch_size, num_channels, hidden_size), dtype=X.dtype, device=X.device)
    DW = torch.zeros((num_channels), dtype=W.dtype, device=W.device)
    DB = torch.zeros((num_channels), dtype=B.dtype, device=B.device)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16
    # print(Mean)
    # print(RSTD)
    # ans = (X - Mean[0]) * RSTD[0]
    # ans = ans * dY
    # print(f"Torch ans is {ans.sum(dim=-1)}")
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    _group_norm_backward_kernel[(batch_size, num_channels)](
        X,
        X.stride(0),
        X.stride(1),
        W,
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        DX,
        DW,
        DB,
        dY,
        hidden_size,
        channels_per_group,
        num_groups,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype
    )
    print(Mean)
    a = (X.view(batch_size, num_groups, -1) - Mean.view(batch_size, num_groups, -1)) * RSTD.view(batch_size, num_groups, -1)
    b = a * dY.view(batch_size, num_groups, -1)
    print(f"Pure torch output: {b.view(batch_size, num_channels, -1).sum(-1)}")
    return DX, DW, DB


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, affine_scaling_weight, affine_shifting_bias, num_channels, num_groups, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE = group_norm_forward(X, num_channels, num_groups, affine_scaling_weight, affine_shifting_bias, eps)
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.save_for_backward(X, affine_scaling_weight, affine_shifting_bias, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(dY, X, W, B, Mean, RSTD, ctx.num_channels, ctx.num_groups)
        return DX, DW, DB, None, None, None
