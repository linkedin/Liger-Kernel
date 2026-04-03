import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# -----------------------------------------------------------------------------
# Optimized Forward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _layer_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    n_cols_inv: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    OPTIMIZED NPU layer_norm forward kernel for small n_cols (<= 2048).

    Key optimizations:
    1. Pre-compute n_cols_inv to avoid repeated scalar division
    2. Hoist W and B loads outside the loop (already done)
    3. Minimize per-iteration scalar operations
    4. Use vectorized operations for mask handling
    5. Optimize cache hints for memory access patterns
    6. Reduce type conversions by keeping intermediate results in float32
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Pre-compute grid stride constants (done once, not per iteration)
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    # Load W and B once (already optimized - kept outside loop)
    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    B_row = tl.load(B_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    base_row_idx = pid * BLOCK_SIZE_M

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + base_row_idx + row_offsets
        row_mask = row_idx < n_rows

        block_mask = row_mask[:, None] & col_mask[None, :]

        X_block_ptr = X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :]

        X_rows = tl.load(
            X_block_ptr,
            mask=block_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        # Compute mean with vectorized operations
        row_sum = tl.sum(X_rows, axis=1)
        mean_rows = row_sum * n_cols_inv  # Multiplication is faster than division

        # Center the data (vectorized operation)
        X_centered = X_rows - mean_rows[:, None]

        X_centered_masked = tl.where(block_mask, X_centered, 0.0)
        var_rows = tl.sum(X_centered_masked * X_centered_masked, axis=1) * n_cols_inv

        rstd_rows = rsqrt(var_rows + eps)

        Mean_ptr_offset = Mean_ptr + row_idx * Mean_row_stride
        RSTD_ptr_offset = RSTD_ptr + row_idx * RSTD_row_stride

        tl.store(Mean_ptr_offset, mean_rows, mask=row_mask)
        tl.store(RSTD_ptr_offset, rstd_rows, mask=row_mask)

        Y_f32 = X_centered * rstd_rows[:, None] * W_row[None, :] + B_row[None, :]

        # Store output with coalesced memory access
        Y_block_ptr = Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :]
        tl.store(Y_block_ptr, Y_f32, mask=block_mask)


# -----------------------------------------------------------------------------
# Forward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _layer_norm_forward_kernel_npu(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """NPU-optimized layer_norm forward kernel with column blocking."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols

    for row_idx in range(pid, n_rows, num_progs):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        Mean_row_ptr = Mean_ptr + row_idx * Mean_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        row_sum = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

            row_sum += tl.sum(X_block)

        mean = row_sum * n_cols_inv

        var_sum = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

            X_centered = X_block - mean
            var_sum += tl.sum(tl.where(mask, X_centered * X_centered, 0.0))

        var = var_sum * n_cols_inv
        rstd = rsqrt(var + eps)

        tl.store(Mean_row_ptr, mean)
        tl.store(RSTD_row_ptr, rstd)

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            B_block = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            X_centered = X_block - mean
            Y_f32 = X_centered * rstd * W_block + B_block

            tl.store(Y_row_ptr + col_offsets, Y_f32.to(X_block.dtype), mask=mask)


# -----------------------------------------------------------------------------
# Optimized Backward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _layer_norm_backward_kernel_no_tiling(
    X_ptr,
    X_row_stride,
    W_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    DX_ptr,
    DX_row_stride,
    DW_scratch_ptr,
    DW_scratch_stride,
    DB_scratch_ptr,
    DB_scratch_stride,
    DY_ptr,
    DY_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_cols_inv: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    OPTIMIZED NPU layer_norm backward kernel for small n_cols (<= 2048).

    Key optimizations:
    1. Pre-compute n_cols_inv to avoid repeated division
    2. Minimize scalar operations in the hot path
    3. Reduce redundant mask computations
    4. Optimize memory access patterns with better cache hints
    5. Keep intermediate results in float32 to reduce conversions
    6. Use vectorized operations throughout
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Per-program accumulators for dW/dB
    dW_acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    dB_acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    base_row_idx = pid * BLOCK_SIZE_M

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + base_row_idx + row_offsets
        row_mask = row_idx < n_rows

        # Pre-compute block mask once
        block_mask = row_mask[:, None] & col_mask[None, :]

        X_block_ptr = X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :]
        DY_block_ptr = DY_ptr + row_idx[:, None] * DY_row_stride + col_offsets[None, :]
        Mean_row_ptr = Mean_ptr + row_idx * Mean_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load all required data with appropriate cache hints
        # .cg = cache global (read once, don't pollute cache)
        X_rows = tl.load(X_block_ptr, mask=block_mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        DY_rows = tl.load(DY_block_ptr, mask=block_mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        mean_rows = tl.load(Mean_row_ptr, mask=row_mask, other=0.0).to(tl.float32)
        rstd_rows = tl.load(RSTD_row_ptr, mask=row_mask, other=0.0).to(tl.float32)

        x_hat = (X_rows - mean_rows[:, None]) * rstd_rows[:, None]
        wdy = W_row[None, :] * DY_rows

        x_hat_wdy_masked = tl.where(block_mask, x_hat * wdy, 0.0)
        wdy_masked = tl.where(block_mask, wdy, 0.0)

        c1 = tl.sum(x_hat_wdy_masked, axis=1) * n_cols_inv
        c2 = tl.sum(wdy_masked, axis=1) * n_cols_inv

        DX_f32 = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_rows[:, None]

        # Store dX with coalesced memory access
        DX_block_ptr = DX_ptr + row_idx[:, None] * DX_row_stride + col_offsets[None, :]
        tl.store(DX_block_ptr, DX_f32.to(X_ptr.dtype.element_ty), mask=block_mask)

        dW_acc += tl.sum(tl.where(block_mask, DY_rows * x_hat, 0.0), axis=0)
        dB_acc += tl.sum(tl.where(block_mask, DY_rows, 0.0), axis=0)

    # Write accumulated gradients to scratch buffers
    DW_scratch_offset = DW_scratch_ptr + pid * DW_scratch_stride + col_offsets
    DB_scratch_offset = DB_scratch_ptr + pid * DB_scratch_stride + col_offsets

    tl.store(DW_scratch_offset, dW_acc, mask=col_mask)
    tl.store(DB_scratch_offset, dB_acc, mask=col_mask)


# -----------------------------------------------------------------------------
# Backward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _layer_norm_backward_kernel_npu(
    X_ptr,
    X_row_stride,
    W_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    DX_ptr,
    DX_row_stride,
    DW_ptr,
    DB_ptr,
    DY_ptr,
    DY_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """NPU-optimized layer_norm backward kernel with column blocking."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols_inv = 1.0 / n_cols

    for row_idx in range(pid, n_rows, num_progs):
        X_row_ptr = X_ptr + row_idx * X_row_stride
        DY_row_ptr = DY_ptr + row_idx * DY_row_stride
        DX_row_ptr = DX_ptr + row_idx * DX_row_stride
        Mean_row_ptr = Mean_ptr + row_idx * Mean_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        mean = tl.load(Mean_row_ptr).to(tl.float32)
        rstd = tl.load(RSTD_row_ptr).to(tl.float32)

        sum_x_hat_wdy = 0.0
        sum_wdy = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            DY_block = tl.load(DY_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            x_hat = (X_block - mean) * rstd
            wdy = W_block * DY_block

            sum_x_hat_wdy += tl.sum(tl.where(mask, x_hat * wdy, 0.0))
            sum_wdy += tl.sum(tl.where(mask, wdy, 0.0))

        c1 = sum_x_hat_wdy * n_cols_inv
        c2 = sum_wdy * n_cols_inv

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            DY_block = tl.load(DY_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            x_hat = (X_block - mean) * rstd
            wdy = W_block * DY_block

            DX_block = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_row_ptr + col_offsets, DX_block.to(X_ptr.dtype.element_ty), mask=mask)

            dW_block = DY_block * x_hat
            dB_block = DY_block

            tl.atomic_add(DW_ptr + col_offsets, dW_block, mask=mask)
            tl.atomic_add(DB_ptr + col_offsets, dB_block, mask=mask)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_optimal_block_size(n_cols, is_forward: bool):
    """
    Calculate optimal block size using compute_default_tiling_strategy.

    Memory analysis for forward pass (per row):
    - Load: X_block, W_block, B_block (3 blocks)
    - Store: Y_block, Mean, RSTD (3 blocks)
    - Compute: X_centered, Y intermediate (2 blocks)
    - Total: conservative estimate 10 blocks of memory

    Memory analysis for backward pass (per row):
    - Load: X_block, DY_block, W_block, Mean, RSTD, existing_DW, existing_DB (7 blocks)
    - Store: DX_block, new_DW, new_DB (3 blocks)
    - Compute: x_hat, wdy, DX intermediate, dW_block, dB_block (5 blocks)
    - Total: conservative estimate 15 blocks of memory

    Args:
        n_cols: Number of columns in the tensor
        is_forward: Whether this is for forward pass (True) or backward pass (False)

    Returns:
        Optimal block size
    """
    if n_cols <= 2048:
        return triton.next_power_of_2(n_cols)

    memory_multiplier = 10.0 if is_forward else 15.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((n_cols,),),
        tiling_dims=(0,),
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return max(2048, block_size)
    else:
        return 2048


def _compute_grid_size(n_rows: int, block_size_m: int, num_cores: int) -> int:
    """
    Compute the effective grid size for no-tiling kernels.

    OPTIMIZATION: Balances parallelism with overhead
    - Ensures enough work per program to amortize launch costs
    - Avoids launching idle programs
    - Caps at 2x core count for hardware concurrency
    """
    num_row_blocks = triton.cdiv(n_rows, block_size_m)

    return min(num_cores * 2, num_row_blocks)


# -----------------------------------------------------------------------------
# Forward and Backward Functions
# -----------------------------------------------------------------------------


def layer_norm_forward(X, W, B, eps):
    """
    NPU-optimized forward pass for LayerNorm.

    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (output, input, mean, rstd)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # Get optimal block sizes
    BLOCK_SIZE = get_optimal_block_size(n_cols, True)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    # Allocate output tensors
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)

    num_cores = get_npu_core_count()

    # Choose kernel
    if n_cols <= 2048:
        grid_size = _compute_grid_size(n_rows, BLOCK_SIZE_M, num_cores)
        n_cols_inv = 1.0 / float(n_cols)

        _layer_norm_forward_kernel_no_tiling[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            B,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            n_cols_inv,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        grid_size = min(num_cores, n_rows)
        _layer_norm_forward_kernel_npu[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            B,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return Y.view(*shape), X, Mean, RSTD


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    """
    NPU-optimized backward pass for LayerNorm.

    Args:
        dY: Gradient of output
        X: Input tensor
        W: Weight tensor
        B: Bias tensor
        Mean: Pre-computed mean
        RSTD: Pre-computed reciprocal standard deviation

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Get optimal block sizes
    BLOCK_SIZE = get_optimal_block_size(n_cols, False)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    num_cores = get_npu_core_count()

    # Allocate gradient tensors
    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    # Choose kernel
    if n_cols <= 2048:
        grid_size = _compute_grid_size(n_rows, BLOCK_SIZE_M, num_cores)
        DW_scratch = torch.empty((grid_size, n_cols), dtype=torch.float32, device=W.device)
        DB_scratch = torch.empty((grid_size, n_cols), dtype=torch.float32, device=W.device)

        n_cols_inv = 1.0 / float(n_cols)

        _layer_norm_backward_kernel_no_tiling[(grid_size,)](
            X,
            X.stride(0),
            W,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            DX,
            DX.stride(0),
            DW_scratch,
            DW_scratch.stride(0),
            DB_scratch,
            DB_scratch.stride(0),
            dY,
            dY.stride(0),
            n_rows,
            n_cols,
            n_cols_inv,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )

        DW = DW_scratch.sum(dim=0)
        DB = DB_scratch.sum(dim=0)
    else:
        grid_size = min(num_cores, n_rows)

        DW = torch.zeros(n_cols, dtype=torch.float32, device=W.device)
        DB = torch.zeros(n_cols, dtype=torch.float32, device=W.device)

        _layer_norm_backward_kernel_npu[(grid_size,)](
            X,
            X.stride(0),
            W,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            DX,
            DX.stride(0),
            DW,
            DB,
            dY,
            dY.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return DX.view(*shape), DW.to(W.dtype), DB.to(B.dtype)


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------


class LigerLayerNormFunction(torch.autograd.Function):
    """
    OPTIMIZED NPU LayerNorm operation.

    Key optimizations for no-tiling kernels:
    1. Pre-compute 1/n_cols to avoid scalar division (40.6% → <30% target)
    2. Minimize per-iteration scalar operations in grid-stride loops
    3. Hoist constant computations outside loops
    4. Use vectorized operations throughout
    5. Optimize memory access patterns with better cache hints
    6. Reduce type conversions by keeping intermediates in float32
    7. Improve grid sizing for better work distribution
    """

    @staticmethod
    @ensure_contiguous
    def forward(X, W, B, eps):
        Y, X, Mean, RSTD = layer_norm_forward(X, W, B, eps)
        return Y, X, Mean, RSTD

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, W, B, _ = inputs
        Y, X, Mean, RSTD = output
        ctx.save_for_backward(X, W, B, Mean, RSTD)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, _grad_X, _grad_Mean, _grad_RSTD):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
