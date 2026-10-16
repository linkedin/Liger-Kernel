import operator
import os

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import infer_device
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

if infer_device() == "npu":
    # Ascend UB (192KB/core, inflated by auto multi-buffering) cannot hold
    # 16K fp32 tiles; cap the block size, overridable for tuning.
    MAX_FUSED_SIZE = int(os.environ.get("LIGER_GN_MAX_FUSED_SIZE", 4096))
else:
    MAX_FUSED_SIZE = 65536

# The fused 2D paths are validated on Ascend NPU; other backends keep using
# the original kernels until validated there.
_FUSED_2D_ENABLED = infer_device() == "npu"


def _h_log2(n: int) -> int:
    """log2(n) when n is a power of two, else -1 (selects the shift path)."""
    return n.bit_length() - 1 if n > 0 and (n & (n - 1)) == 0 else -1


@triton.jit
def _group_norm_fwd_2d(
    Y_ptr,
    X_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    X_row_stride,
    X_col_stride,
    B,  # batch size (loop bound)
    eps: tl.constexpr,
    NG: tl.constexpr,  # group-flat size, must be a power of two
    G: tl.constexpr,  # num_groups
    CPG: tl.constexpr,  # channels per group
    HP: tl.constexpr,  # hidden size per channel (NG == CPG * HP)
    GPB: tl.constexpr,  # groups per program
):
    """Fused forward for the pow2 single-tile case.

    1D grid over group chunks; each program loops over the batch. One
    iteration handles GPB whole groups as a [GPB, NG] tile: row-wise
    reductions (axis=1) for stats, affine applied via a [GPB, CPG] load
    broadcast along H (no gather, no vector integer division).
    """
    g0 = tl.program_id(0) * GPB
    rg = tl.arange(0, GPB)
    rr = tl.arange(0, NG)
    cc = tl.arange(0, CPG)
    W4 = tl.load(W_ptr + (g0 + rg)[:, None] * CPG + cc[None, :]).to(tl.float32)
    B4 = tl.load(B_ptr + (g0 + rg)[:, None] * CPG + cc[None, :]).to(tl.float32)
    for b in tl.range(0, B):
        ptrs = b * X_row_stride + (g0 + rg)[:, None] * X_col_stride + rr[None, :]
        Xv = tl.load(X_ptr + ptrs).to(tl.float32)
        s = tl.sum(Xv, axis=1)
        sq = tl.sum(Xv * Xv, axis=1)
        m = s / NG
        var = tl.maximum(sq / NG - m * m, 0.0)
        rstd = rsqrt(var + eps)
        X3 = tl.reshape(Xv, (GPB, CPG, HP))
        Y3 = (X3 - m[:, None, None]) * rstd[:, None, None] * W4[:, :, None] + B4[:, :, None]
        Yv = tl.reshape(Y3, (GPB, NG))
        tl.store(Y_ptr + ptrs, Yv.to(Y_ptr.dtype.element_ty))
        tl.store(Mean_ptr + b * G + g0 + rg, m)
        tl.store(RSTD_ptr + b * G + g0 + rg, rstd)


@triton.jit
def _group_norm_bwd_2d(
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    DY_ptr,
    X_row_stride,
    X_col_stride,
    B,  # batch size (loop bound)
    NG: tl.constexpr,  # group-flat size, must be a power of two
    G: tl.constexpr,
    CPG: tl.constexpr,
    HP: tl.constexpr,
    GPB: tl.constexpr,
):
    """Fused backward for the pow2 single-tile case.

    1D grid over group chunks; each program loops over the batch so the
    dW/dB channel reductions accumulate in registers and are stored once
    (no atomics, no zero-init, no dtype cast kernels afterwards).
    """
    g0 = tl.program_id(0) * GPB
    rg = tl.arange(0, GPB)
    rr = tl.arange(0, NG)
    cc = tl.arange(0, CPG)
    W4 = tl.load(W_ptr + (g0 + rg)[:, None] * CPG + cc[None, :]).to(tl.float32)
    W3 = tl.reshape(tl.broadcast_to(W4[:, :, None], (GPB, CPG, HP)), (GPB, NG))
    dW_acc = tl.zeros([GPB, CPG], tl.float32)
    dB_acc = tl.zeros([GPB, CPG], tl.float32)
    for b in tl.range(0, B):
        rows = b * G + g0 + rg
        ptrs = b * X_row_stride + (g0 + rg)[:, None] * (CPG * X_col_stride) + rr[None, :]
        mean = tl.load(Mean_ptr + rows)
        rstd = tl.load(RSTD_ptr + rows)
        Xv = tl.load(X_ptr + ptrs).to(tl.float32)
        dYv = tl.load(DY_ptr + ptrs).to(tl.float32)
        x_hat = (Xv - mean[:, None]) * rstd[:, None]
        # segmented per-channel reductions via reshape, no where-select chains
        dW_acc += tl.sum(tl.reshape(dYv * x_hat, (GPB, CPG, HP)), axis=2)
        dB_acc += tl.sum(tl.reshape(dYv, (GPB, CPG, HP)), axis=2)
        wdy = W3 * dYv
        c1 = tl.sum(x_hat * wdy, axis=1) / NG
        c2 = tl.sum(wdy, axis=1) / NG
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(DX_ptr + ptrs, dx.to(DX_ptr.dtype.element_ty))
    outs = (g0 + rg)[:, None] * CPG + cc[None, :]
    tl.store(DW_ptr + outs, dW_acc.to(DW_ptr.dtype.element_ty))
    tl.store(DB_ptr + outs, dB_acc.to(DB_ptr.dtype.element_ty))


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride,  # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    W_ptr,  # pointer to W
    B_ptr,  # pointer to B
    hidden_size,  # hidden size of X (group-flat: channels_per_group * H)
    channels_per_group,  # the number of channels per group
    eps,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_TILE: tl.constexpr,  # hidden_size <= BLOCK_SIZE: fuse stats+normalize
    H_LOG2: tl.constexpr,  # log2(hidden_size_per_channel) if pow2 else -1
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

    block_range = tl.arange(0, BLOCK_SIZE)
    hidden_size_per_channel = hidden_size // channels_per_group

    if SINGLE_TILE:
        # Whole group fits in one tile: load X once, keep it in registers for
        # both the statistics and the normalize+affine pass.
        mask = block_range < hidden_size
        X = tl.load(X_ptr + block_range, mask=mask, other=0.0).to(tl.float32)

        s = tl.sum(X)
        squared_sum = tl.sum(X * X)
        m = s / hidden_size
        variance = (squared_sum / hidden_size) - (m * m)
        # guard against catastrophic cancellation producing tiny negatives
        variance = tl.maximum(variance, 0.0)
        rstd = rsqrt(variance + eps.to(tl.float32))

        # Determine which channel each element belongs to, then load W/B.
        # Shift instead of vector integer division when H is a power of two.
        if H_LOG2 >= 0:
            local_channel = block_range >> H_LOG2
        else:
            local_channel = block_range // hidden_size_per_channel
        global_channel = group_idx * channels_per_group + local_channel
        W = tl.load(W_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
        B = tl.load(B_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
        Y = (X - m) * rstd * W + B
        tl.store(Y_ptr + block_range, Y.to(Y_ptr.dtype.element_ty), mask=mask)
    else:
        # Compute mean and variance using the online algorithm (fp32 accumulation)
        s = 0.0
        squared_sum = 0.0
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(X)
            # X**2
            squared_sum += tl.sum(X * X)

        m = s / hidden_size

        # variance = E[X**2] - E[X]**2
        variance = (squared_sum / hidden_size) - (m * m)
        variance = tl.maximum(variance, 0.0)

        # 1/std
        rstd = rsqrt(variance + eps.to(tl.float32))

        # Normalize — flat loop over full hidden_size (not per-channel)
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m).to(tl.float32)
            if H_LOG2 >= 0:
                local_channel = hidden_size_offsets >> H_LOG2
            else:
                local_channel = hidden_size_offsets // hidden_size_per_channel
            global_channel = group_idx * channels_per_group + local_channel
            W = tl.load(W_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
            B = tl.load(B_ptr + global_channel, mask=mask, other=0.0).to(tl.float32)
            Y = (X - m) * rstd * W + B
            tl.store(Y_ptr + hidden_size_offsets, Y.to(Y_ptr.dtype.element_ty), mask=mask)

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
    DW_ptr,  # pointer to weights grad, shape (n_channels), fp32 scratch
    DB_ptr,  # pointer to bias grad, shape (n_channels), fp32 scratch
    UPSTREAM_ptr,  # pointer to output grad, shape (n_rows, n_channels, hidden_size)
    hidden_size: tl.constexpr,  # hidden size per channel (H)
    channels_per_group: tl.constexpr,  # number of channels per group
    BLOCK_SIZE: tl.constexpr,
    SINGLE_TILE: tl.constexpr,  # hidden_size * channels_per_group <= BLOCK_SIZE
    H_LOG2: tl.constexpr,  # log2(hidden_size) if pow2 else -1
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md

    The backprop equations are the same for group_norm and layer_norm
    the only difference here is that we load the Mean, Rstd corresponding to the
    group we're computing gradients for and the mean and rstd are computed over n-channels
    so the total number of elements we compute the mean over is num_channels_per_group * hidden_size

    We also need to load the Weights corresponding to the current channel to compute the gradients.
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    # Move the pointers to the correct batch
    X_ptr += batch_idx * X_row_stride
    DX_ptr += batch_idx * X_row_stride
    UPSTREAM_ptr += batch_idx * X_row_stride

    # Mean and rstd are the same shape so have the same strides
    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)

    block_range = tl.arange(0, BLOCK_SIZE)

    if SINGLE_TILE:
        # Whole group in one tile: load X/dY once, compute dW/dB with
        # register-resident segmented reductions, dx in the same pass.
        group_numel = hidden_size * channels_per_group
        group_offset = group_idx * channels_per_group * X_col_stride
        mask = block_range < group_numel
        X = tl.load(X_ptr + group_offset + block_range, mask=mask, other=0.0).to(tl.float32)
        UPSTREAM_grad = tl.load(UPSTREAM_ptr + group_offset + block_range, mask=mask, other=0.0).to(tl.float32)

        x_hat = (X - mean) * rstd
        if H_LOG2 >= 0:
            local_channel = block_range >> H_LOG2
        else:
            local_channel = block_range // hidden_size
        W = tl.load(W_ptr + group_idx * channels_per_group + local_channel, mask=mask, other=0.0).to(tl.float32)

        dy_xhat = UPSTREAM_grad * x_hat
        # Segmented reduction per channel (unrolled, register-resident)
        for c in tl.static_range(channels_per_group):
            sel = local_channel == c
            dW = tl.sum(tl.where(sel, dy_xhat, 0.0))
            dB = tl.sum(tl.where(sel, UPSTREAM_grad, 0.0))
            # Need to ensure additions to the same channel are atomic
            tl.atomic_add(DW_ptr + group_idx * channels_per_group + c, dW)
            tl.atomic_add(DB_ptr + group_idx * channels_per_group + c, dB)

        N = group_numel
        wdy = W * UPSTREAM_grad
        c1 = tl.sum(x_hat * wdy) / N
        c2 = tl.sum(wdy) / N
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + group_offset + block_range, dx.to(DX_ptr.dtype.element_ty), mask=mask)
    else:
        c1 = 0.0
        c2 = 0.0

        # We need to compute the sum terms of the backprop equations across all channels in the group
        for channel_idx in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
            dW = 0.0
            dB = 0.0
            # Move the pointers to the correct channel
            W = tl.load(W_ptr + channel_idx).to(tl.float32)
            for i in tl.range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                X = tl.load(
                    X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                UPSTREAM_grad = tl.load(
                    UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

                x_hat = (X - mean) * rstd
                dW += tl.sum(UPSTREAM_grad * x_hat)
                dB += tl.sum(UPSTREAM_grad)

                wdy = W * UPSTREAM_grad
                c1 += tl.sum(x_hat * wdy)
                c2 += tl.sum(wdy)

            # Need to ensure additions to the same channel are atomic
            tl.atomic_add(DW_ptr + channel_idx, dW)
            tl.atomic_add(DB_ptr + channel_idx, dB)

        N = hidden_size * channels_per_group
        c1 = c1 / N
        c2 = c2 / N

        for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
            # Move the pointers to the correct channel
            W = tl.load(W_ptr + channel_idx).to(tl.float32)
            for i in range(0, hidden_size, BLOCK_SIZE):
                hidden_size_offsets = i + block_range
                mask = hidden_size_offsets < hidden_size
                X = tl.load(
                    X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                UPSTREAM_grad = tl.load(
                    UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)

                x_hat = (X - mean) * rstd
                wdy = W * UPSTREAM_grad
                dx = (wdy - (x_hat * c1 + c2)) * rstd
                tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx.to(DX_ptr.dtype.element_ty), mask=mask)


def _pick_gpb(num_groups: int) -> int:
    """Groups per program for the 2D path; must divide num_groups."""
    if num_groups % 2 == 0:
        return 2
    return 1


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# Cache of CompiledKernel handles for the hot fused paths. Triton's python
# dispatch costs ~18us per launch; a cached CompiledKernel launch is ~9us,
# which matters at small batch sizes where the kernel itself is ~15us.
_KERNEL_CACHE = {}


def _launch_cached(jit_fn, grid, cache_key, *args, **constexprs):
    entry = _KERNEL_CACHE.get(cache_key)
    if entry is None:
        ck = jit_fn[grid](*args, **constexprs)
        _KERNEL_CACHE[cache_key] = ck
        return
    g = tuple(grid) + (1,) * (3 - len(grid))
    entry[g](*args)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    # Reshape X so that the mean and std are computed across the groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    # Mean/RSTD are internal buffers; keep them fp32 regardless of input dtype
    Mean = torch.empty((batch_size, num_groups), dtype=torch.float32, device=X.device)
    RSTD = torch.empty((batch_size, num_groups), dtype=torch.float32, device=X.device)

    if _FUSED_2D_ENABLED and _is_pow2(hidden_size) and hidden_size <= MAX_FUSED_SIZE:
        # fused 2D path: whole group(s) per tile, no masks, no gather
        GPB = _pick_gpb(num_groups)
        _launch_cached(
            _group_norm_fwd_2d,
            (num_groups // GPB,),
            ("fwd2d", X.dtype, batch_size, hidden_size, num_groups, GPB, X.stride(0), X.stride(1), eps),
            Y,
            X,
            Mean,
            RSTD,
            W,
            B,
            X.stride(0),
            X.stride(1),
            batch_size,
            eps=eps,
            NG=hidden_size,
            G=num_groups,
            CPG=channels_per_group,
            HP=hidden_size // channels_per_group,
            GPB=GPB,
        )
    else:
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
            W,
            B,
            hidden_size,
            channels_per_group,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            SINGLE_TILE=hidden_size <= BLOCK_SIZE,
            H_LOG2=_h_log2(hidden_size // channels_per_group),
        )
    # Return tensors in the original shape
    return Y.view(*shape), X.view(*shape), Mean, RSTD, BLOCK_SIZE


def group_norm_backward(dY, X, W, B, Mean, RSTD, num_channels, num_groups):
    shape = dY.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    dY = dY.view(batch_size, num_groups, -1)
    # Number of elements per channel, so it has to be measured after the flatten:
    # the last dimension of an unflattened (N, C, H, W) input is only W
    hidden_size = dY.shape[-1] // channels_per_group
    DX = torch.empty(
        (batch_size, num_groups, hidden_size * channels_per_group),
        dtype=X.dtype,
        device=X.device,
    )

    group_numel = hidden_size * channels_per_group
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_numel))
    if _FUSED_2D_ENABLED and _is_pow2(group_numel) and group_numel <= MAX_FUSED_SIZE:
        # fused 2D path: batch-loop kernel, DW/DB written directly (1 launch)
        GPB = _pick_gpb(num_groups)
        DW = torch.empty((num_channels), dtype=W.dtype, device=W.device)
        DB = torch.empty((num_channels), dtype=B.dtype, device=B.device)
        _launch_cached(
            _group_norm_bwd_2d,
            (num_groups // GPB,),
            ("bwd2d", X.dtype, W.dtype, batch_size, group_numel, num_groups, GPB, X.stride(0), X.stride(1)),
            X,
            W,
            Mean,
            RSTD,
            DX,
            DW,
            DB,
            dY,
            X.stride(0),
            X.stride(1),
            batch_size,
            NG=group_numel,
            G=num_groups,
            CPG=channels_per_group,
            HP=hidden_size,
            GPB=GPB,
        )
        return DX.view(*shape), DW, DB

    # fallback path: fp32 accumulation buffers + atomics
    DW = torch.zeros((num_channels), dtype=torch.float32, device=W.device)
    DB = torch.zeros((num_channels), dtype=torch.float32, device=B.device)
    _group_norm_backward_kernel[(batch_size, num_groups)](
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
        BLOCK_SIZE=BLOCK_SIZE,
        SINGLE_TILE=group_numel <= BLOCK_SIZE,
        H_LOG2=_h_log2(hidden_size),
    )

    # Return tensors in the original shape
    return DX.view(*shape), DW.to(W.dtype), DB.to(B.dtype)


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    ):
        Y, X, Mean, RSTD, BLOCK_SIZE = group_norm_forward(
            X,
            num_channels,
            num_groups,
            affine_scaling_weight,
            affine_shifting_bias,
            eps,
        )
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
