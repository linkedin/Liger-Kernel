import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings_mnk


@triton.jit
def conv2d_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    K,
    P,
    Q,
    R,
    S,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dila_h,
    dila_w,
    GEMM_M,
    GEMM_N,
    GEMM_K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # constant offset pre-computation for speedup
    HWC = H * W * C
    SC = S * C
    RSC = R * S * C

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        r = gemm_k // SC
        rsc_residual = gemm_k % SC
        s = rsc_residual // C
        c = rsc_residual % C

        h = p[:, None] * stride_h + r[None, :] * dila_h - pad_h
        w = q[:, None] * stride_w + s[None, :] * dila_w - pad_w

        mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_w = (r < R) & (s < S) & (c < C)

        offs_x = n[:, None] * HWC + h * W * C + w * C + c
        offs_w = k[None, :] * RSC + r[:, None] * SC + s[:, None] * C + c[:, None]

        x_ptrs = x_ptr + offs_x
        w_ptrs = w_ptr + offs_w

        x_data = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_data = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)
        accumulator += tl.dot(x_data, w_data)

    c_data = accumulator.to(tl.float16)

    offs_y = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_y = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    y_ptrs = y_ptr + offs_y
    tl.store(y_ptrs, c_data, mask=mask_y)


def conv2d_forward(
    x: torch.Tensor, w: torch.Tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
):
    N, C, H, W = x.shape
    K, C, R, S = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dila_h, dila_w = dilation
    P = (H + 2 * pad_h - dila_h * (R - 1) - 1) // stride_h + 1
    Q = (W + 2 * pad_w - dila_w * (S - 1) - 1) // stride_w + 1
    y = torch.empty((N, K, P, Q), device=x.device, dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, grid = (
        calculate_settings_mnk(GEMM_M, GEMM_N, GEMM_K, R, S)
    )

    conv2d_forward_kernel[grid](
        x,
        w,
        y,
        N,
        C,
        H,
        W,
        K,
        P,
        Q,
        R,
        S,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dila_h,
        dila_w,
        GEMM_M,
        GEMM_N,
        GEMM_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    return y.requires_grad_()


class LigerConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, stride, padding, dilation):
        ctx.save_for_backward(x, w)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        output = conv2d_forward(x, w, stride, padding, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        grad_x = torch.nn.grad.conv2d_input(
            x.shape, w, grad_output, stride, padding, dilation
        )
        grad_w = torch.nn.grad.conv2d_weight(
            x, w.shape, grad_output, stride, padding, dilation
        )

        return grad_x, grad_w, None, None, None
