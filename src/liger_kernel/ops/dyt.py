import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import infer_device

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _dyt_fwd_kernel(
    x_ptr,
    x_row_stride,
    alpha_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reference:
    https://arxiv.org/abs/2503.10622

    Shapes:
        - x: (BT, C)
        - alpha: (1)
        - gamma: (C)
        - beta: (C)
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask)
    beta = tl.load(beta_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = gamma * tanh((alpha * x).cast(tl.float32)) + beta
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _dyt_bwd_kernel(
    x_ptr,
    x_row_stride,
    dy_ptr,
    dy_row_stride,
    dx_ptr,
    dx_row_stride,
    alpha_ptr,
    dalpha_ptr,
    gamma_ptr,
    dgamma_ptr,
    dgamma_row_stride,
    n_cols,
    n_rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reference:
    https://arxiv.org/abs/2503.10622

    Shapes:
        - x: (BT, C)
        - alpha: (1)
        - gamma: (C)
        - dx: (BT, C)
        - dy: (BT, C)
        - dgamma: (sm_count, C)
        - dalpha: (sm_count,)
    """
    # d(gamma * tanh(alpha * x) + beta) / dx
    # = gamma * (1 - tanh^2(alpha * x)) * alpha
    # d(gamma * tanh(alpha * x) + beta) / dalpha
    # = gamma * (1 - tanh^2(alpha * x)) * x
    # d(gamma * tanh(alpha * x) + beta) / dgamma
    # = tanh(alpha * x)
    # d(gamma * tanh(alpha * x)) / dbeta = 1
    pid = tl.program_id(0)

    row_start = pid * ROWS_PER_PROGRAM
    row_end = min((pid + 1) * ROWS_PER_PROGRAM, n_rows)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    dalpha = 0.0
    dgamma = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    x_ptr += row_start * x_row_stride
    dx_ptr += row_start * dx_row_stride
    dy_ptr += row_start * dy_row_stride
    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)

    for _ in tl.range(row_start, row_end):
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tanh_ax = tanh((alpha * x).cast(tl.float32))
        sech2_ax = 1 - tanh_ax * tanh_ax

        dx = dy * gamma * sech2_ax * alpha
        dalpha += tl.sum(dy * gamma * sech2_ax * x)
        dgamma += dy * tanh_ax
        tl.store(dx_ptr + offsets, dx, mask=mask)

        dy_ptr += dy_row_stride
        x_ptr += x_row_stride
        dx_ptr += dx_row_stride

    tl.store(dgamma_ptr + pid * dgamma_row_stride + offsets, dgamma, mask=mask)
    tl.store(dalpha_ptr + pid, dalpha)

    pass


def liger_dyt_fwd(x, alpha, gamma, beta):
    shape = x.shape
    dim = shape[-1]
    x = x.view(-1, dim)
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _dyt_fwd_kernel[(n_rows,)](
        x_ptr=x,
        alpha_ptr=alpha,
        gamma_ptr=gamma,
        beta_ptr=beta,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.view(*shape)


def liger_dyt_bwd(dy, x, alpha, gamma):
    shape = dy.shape
    dtype = x.dtype
    dim = shape[-1]
    dy = dy.view(-1, dim)
    x = x.view(-1, dim)
    n_rows, n_cols = dy.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    sm_count = 1
    device = infer_device()
    if device == "cuda":
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    elif device == "xpu":
        sm_count = torch.xpu.get_device_properties(x.device).gpu_subslice_count
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            f"Feature dimension {dim} exceeds maximum supported size of {BLOCK_SIZE}. Consider using a smaller feature dimension."
        )

    dx = torch.empty_like(x, dtype=torch.float32)
    _dalpha = torch.empty((sm_count,), dtype=torch.float32, device=x.device)
    _dgamma = torch.empty((sm_count, n_cols), dtype=torch.float32, device=x.device)

    grid = (sm_count,)
    rows_per_program = triton.cdiv(n_rows, sm_count)
    _dyt_bwd_kernel[grid](
        x_ptr=x,
        x_row_stride=x.stride(0),
        dy_ptr=dy,
        dy_row_stride=dy.stride(0),
        dx_ptr=dx,
        dx_row_stride=dx.stride(0),
        alpha_ptr=alpha,
        dalpha_ptr=_dalpha,
        gamma_ptr=gamma,
        dgamma_ptr=_dgamma,
        dgamma_row_stride=_dgamma.stride(0),
        n_cols=n_cols,
        n_rows=n_rows,
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dalpha = _dalpha.sum(dim=0, keepdim=True).to(dtype)
    dgamma = _dgamma.sum(dim=0).to(dtype)
    dbeta = dy.sum(dim=0).to(dtype)
    return dx.view(*shape), dalpha, dgamma, dbeta


class LigerDyTFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, alpha, gamma, beta):
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        x, alpha, gamma = ctx.saved_tensors
        dx, dalpha, dgamma, dbeta = liger_dyt_bwd(
            grad_output,
            x,
            alpha,
            gamma,
        )

        return (dx, dalpha, dgamma, dbeta)
