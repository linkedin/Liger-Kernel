import torch
import triton
import triton.language as tl

from triton.language.math import tanh

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# -----------------------------------------------------------------------------
# Forward Kernel
# -----------------------------------------------------------------------------


@triton.jit
def _dyt_fwd_kernel(
    X,
    Y,
    Alpha,
    Gamma,
    Beta,
    HAVE_BETA: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Forward kernel for DYT: y = tanh(α·x) · γ + β

    Grid: (num_col_blocks, num_row_programs)
    Each program processes multiple rows using grid-stride loop
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    num_row_programs = tl.num_programs(1)

    col_start = pid_n * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    alpha = tl.load(Alpha).to(tl.float32)
    gamma = tl.load(Gamma + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    if HAVE_BETA:
        beta = tl.load(Beta + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Grid-stride loop over rows
    for row_idx in range(pid_m, M, num_row_programs):
        row_offset = row_idx * N

        x = tl.load(X + row_offset + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

        # Compute: y = tanh(α·x) · γ + β
        tanh_x = tanh(alpha * x)
        y = tanh_x * gamma

        if HAVE_BETA:
            y += beta

        tl.store(Y + row_offset + col_offsets, y, mask=col_mask)


# -----------------------------------------------------------------------------
# Backward Kernel
# -----------------------------------------------------------------------------


@triton.jit
def _dyt_bwd_kernel(
    DY,
    DX,
    DA,
    DG,
    DB,
    X,
    Alpha,
    Gamma,
    HAVE_BETA: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward kernel for DYT

    Grid: (num_col_blocks, num_row_programs)
    Each program processes multiple rows using grid-stride loop
    Accumulates gradients in local buffers, then stores to global memory
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    num_row_programs = tl.num_programs(1)

    col_start = pid_n * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    alpha = tl.load(Alpha).to(tl.float32)
    gamma = tl.load(Gamma + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    da_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dg_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAVE_BETA:
        db_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Grid-stride loop over rows
    for row_idx in range(pid_m, M, num_row_programs):
        row_offset = row_idx * N

        x = tl.load(X + row_offset + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row_offset + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

        tanh_x = tanh(alpha * x)

        if HAVE_BETA:
            db_acc += dy

        dg_acc += dy * tanh_x

        # Compute intermediate: tmp = (1 - tanh²) · dy · γ
        tmp = (1.0 - tanh_x * tanh_x) * dy * gamma

        # Accumulate dα = Σ(x · tmp)
        da_vec += x * tmp

        # Compute dx = α · tmp
        dx = alpha * tmp
        tl.store(DX + row_offset + col_offsets, dx, mask=col_mask)

    da_acc = tl.sum(da_vec, 0)
    da_offset = pid_m * triton.cdiv(N, BLOCK_N) + pid_n
    tl.store(DA + da_offset, da_acc)

    dg_offset = pid_m * N + col_offsets
    tl.store(DG + dg_offset, dg_acc, mask=col_mask)

    if HAVE_BETA:
        db_offset = pid_m * N + col_offsets
        tl.store(DB + db_offset, db_acc, mask=col_mask)


def get_optimal_block_size(total_elements, is_backward=False):
    """
    Calculate optimal Block Size using compute_default_tiling_strategy
    """
    multiplier = 8.0 if is_backward else 4.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((total_elements,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return 2048


def _compute_grid_size(n_cols, n_rows, block_n):
    """
    Compute grid size to avoid launching idle programs

    Args:
        n_cols: Number of columns
        n_rows: Number of rows
        block_n: Block size for column dimension

    Returns:
        (num_col_blocks, num_row_programs)
    """
    num_cores = get_npu_core_count()
    num_col_blocks = triton.cdiv(n_cols, block_n)
    num_row_blocks = n_rows

    num_row_programs = min(max(1, (num_cores // num_col_blocks)), num_row_blocks)

    return num_col_blocks, num_row_programs


# -----------------------------------------------------------------------------
# Python Wrapper Functions
# -----------------------------------------------------------------------------


def liger_dyt_fwd(x, alpha, gamma, beta):
    """
    Forward pass of DYT: y = tanh(α·x) · γ + β

    Args:
        x: Input tensor of shape [..., N]
        alpha: Scalar parameter
        gamma: Vector parameter of shape [N]
        beta: Vector parameter of shape [N] (optional)

    Returns:
        y: Output tensor of same shape as x
    """
    assert x.is_contiguous()
    HAVE_BETA = beta is not None

    # Flatten to 2D
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape

    # Allocate output
    y = torch.empty_like(x)

    block_n = get_optimal_block_size(N, is_backward=False)

    # Compute grid size
    num_col_blocks, num_row_programs = _compute_grid_size(N, M, block_n)
    grid = (num_col_blocks, num_row_programs)

    # Launch kernel
    _dyt_fwd_kernel[grid](x, y, alpha, gamma, beta, HAVE_BETA, M, N, BLOCK_N=block_n)

    return y.view(input_shape)


def liger_dyt_bwd(dy, x, alpha, gamma, beta):
    """
    Backward pass of DYT

    Args:
        dy: Upstream gradient of shape [..., N]
        x: Input tensor of shape [..., N]
        alpha: Scalar parameter
        gamma: Vector parameter of shape [N]
        beta: Vector parameter of shape [N] (optional)

    Returns:
        dx: Gradient w.r.t. x
        dalpha: Gradient w.r.t. alpha
        dgamma: Gradient w.r.t. gamma
        dbeta: Gradient w.r.t. beta (or None)
    """
    assert dy.is_contiguous()
    HAVE_BETA = beta is not None

    # Flatten to 2D
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    dy = dy.view(-1, input_shape[-1])
    M, N = x.shape

    block_n = get_optimal_block_size(N, is_backward=True)

    # Compute grid size
    num_col_blocks, num_row_programs = _compute_grid_size(N, M, block_n)
    grid = (num_col_blocks, num_row_programs)

    da = torch.zeros(num_row_programs, triton.cdiv(N, block_n), dtype=torch.float32, device=x.device)
    dg = torch.empty(num_row_programs, N, dtype=torch.float32, device=x.device)
    db = torch.empty(num_row_programs, N, dtype=torch.float32, device=x.device) if HAVE_BETA else None
    dx = torch.empty_like(dy)

    _dyt_bwd_kernel[grid](dy, dx, da, dg, db, x, alpha, gamma, HAVE_BETA, M, N, BLOCK_N=block_n)

    da = da.sum().to(x.dtype).unsqueeze(0)
    dg = dg.sum(0).to(gamma.dtype)
    db = db.sum(0).to(x.dtype) if HAVE_BETA else None

    return dx.view(input_shape), da, dg, db


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------


class LigerDyTFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, alpha, gamma, beta):
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy):
        x, alpha, gamma, beta = ctx.saved_tensors
        dx, dalpha, dgamma, dbeta = liger_dyt_bwd(dy, x, alpha, gamma, beta)
        return dx, dalpha, dgamma, dbeta
