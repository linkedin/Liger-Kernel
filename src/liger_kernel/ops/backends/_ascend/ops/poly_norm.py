import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# -----------------------------------------------------------------------------
# Forward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _poly_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,  # weight: [3] for [w0, w1, w2]
    B_ptr,  # bias: scalar
    RSTD_ptr,  # cache rstd for backward: shape (n_rows, 3)
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    NPU-optimized PolyNorm forward kernel for small n_cols (<= 2048).

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-stride loop setup
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    # Load weights and bias
    w0 = tl.load(W_ptr + 0)
    w1 = tl.load(W_ptr + 1)
    w2 = tl.load(W_ptr + 2)
    b = tl.load(B_ptr)

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        # Load input rows
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            cache_modifier=".cg",
        )

        X_f32 = X_rows.to(tl.float32)

        # Compute x³, x², x
        X_pow3 = X_f32 * X_f32 * X_f32
        X_pow2 = X_f32 * X_f32
        X_pow1 = X_f32

        # Compute norm(x³): norm(u) = u * rsqrt(mean(u²) + eps)
        # Mask out out-of-bounds positions to prevent contaminating the sum
        mean_square_3 = tl.sum(X_pow3 * X_pow3, axis=1) / n_cols
        rstd_3 = rsqrt(mean_square_3 + eps)
        norm_x3 = X_pow3 * rstd_3[:, None]

        # Compute norm(x²)
        mean_square_2 = tl.sum(X_pow2 * X_pow2, axis=1) / n_cols
        rstd_2 = rsqrt(mean_square_2 + eps)
        norm_x2 = X_pow2 * rstd_2[:, None]

        # Compute norm(x)
        mean_square_1 = tl.sum(X_pow1 * X_pow1, axis=1) / n_cols
        rstd_1 = rsqrt(mean_square_1 + eps)
        norm_x1 = X_pow1 * rstd_1[:, None]

        # Cache rstd values for backward (store 3 values per row)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride + 0, rstd_3.to(X_rows.dtype), mask=row_mask)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride + 1, rstd_2.to(X_rows.dtype), mask=row_mask)
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride + 2, rstd_1.to(X_rows.dtype), mask=row_mask)

        # Compute output: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        Y_f32 = w0 * norm_x3 + w1 * norm_x2 + w2 * norm_x1 + b

        # Store output
        tl.store(
            Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
            Y_f32.to(X_rows.dtype),
            mask=block_mask,
        )


# -----------------------------------------------------------------------------
# Forward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _poly_norm_forward_kernel_npu(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,  # weight: [3] for [w0, w1, w2]
    B_ptr,  # bias: scalar
    RSTD_ptr,  # cache rstd for backward: shape (n_rows, 3)
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NPU-optimized PolyNorm forward kernel with column blocking.

    This kernel processes rows using a grid-stride loop pattern:
    1. Each program handles multiple rows
    2. For each row, we process it in column chunks of BLOCK_SIZE
    3. Grid size is limited to NPU core count to avoid resource overflow

    Three-pass algorithm per row:
    - First pass: compute mean_square and rstd for x³, x², x across all column blocks
    - Second pass: apply normalization and affine transformation

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    offsets = tl.arange(0, BLOCK_SIZE)

    # Load weights and bias
    w0 = tl.load(W_ptr + 0)
    w1 = tl.load(W_ptr + 1)
    w2 = tl.load(W_ptr + 2)
    b = tl.load(B_ptr)

    # Grid-stride loop over rows
    for row_idx in range(pid, n_rows, num_progs):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # First pass: compute mean_square for x³, x², x
        sum_square_3 = 0.0
        sum_square_2 = 0.0
        sum_square_1 = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

            # Compute powers
            X_pow3 = X_block * X_block * X_block
            X_pow2 = X_block * X_block
            X_pow1 = X_block

            sum_square_3 += tl.sum(X_pow3 * X_pow3)
            sum_square_2 += tl.sum(X_pow2 * X_pow2)
            sum_square_1 += tl.sum(X_pow1 * X_pow1)

        # Compute rstd values
        mean_square_3 = sum_square_3 / n_cols
        mean_square_2 = sum_square_2 / n_cols
        mean_square_1 = sum_square_1 / n_cols

        rstd_3 = rsqrt(mean_square_3 + eps)
        rstd_2 = rsqrt(mean_square_2 + eps)
        rstd_1 = rsqrt(mean_square_1 + eps)

        # Store rstd values
        tl.store(RSTD_row_ptr + 0, rstd_3)
        tl.store(RSTD_row_ptr + 1, rstd_2)
        tl.store(RSTD_row_ptr + 2, rstd_1)

        # Second pass: normalize and apply affine transformation
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            # Load input
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

            # Compute powers
            X_pow3 = X_block * X_block * X_block
            X_pow2 = X_block * X_block
            X_pow1 = X_block

            # Apply normalization
            norm_x3 = X_pow3 * rstd_3
            norm_x2 = X_pow2 * rstd_2
            norm_x1 = X_pow1 * rstd_1

            # Compute output: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
            Y_f32 = w0 * norm_x3 + w1 * norm_x2 + w2 * norm_x1 + b

            # Store result
            tl.store(Y_row_ptr + col_offsets, Y_f32.to(X_block.dtype), mask=mask)


# -----------------------------------------------------------------------------
# Backward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _poly_norm_backward_kernel_no_tiling(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_scratch_ptr,  # shape: (n_programs, 3)
    dW_scratch_stride,
    dB_scratch_ptr,  # shape: (n_programs,)
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    NPU-optimized PolyNorm backward kernel for small n_cols (<= 2048).

    Backward pass equations:
        ∂L/∂x_i = Σ_p w_p * [p*x_i^(p-1) * grad_i/D_p - (p/d)*x_i^(2p-1) * S_p/(D_p³)]

    where:
        - D_p = RMS(x^p) = 1/rstd_p
        - S_p = sum(grad * x^p) over the row
        - d = n_cols
        - p ∈ {3, 2, 1}
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-stride loop setup
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    # Load weights
    w0 = tl.load(W_ptr + 0).to(tl.float32)
    w1 = tl.load(W_ptr + 1).to(tl.float32)
    w2 = tl.load(W_ptr + 2).to(tl.float32)

    # Each program accumulates its own dW/dB contribution to avoid atomic contention
    dW0_acc = 0.0
    dW1_acc = 0.0
    dW2_acc = 0.0
    dB_acc = 0.0

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        # Load input and gradient data
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            cache_modifier=".cg",
        )
        dY_rows = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            cache_modifier=".cg",
        )

        # Load cached rstd values (3 values per row)
        rstd_3 = tl.load(RSTD_ptr + row_idx * RSTD_row_stride + 0, mask=row_mask, other=0.0).to(tl.float32)
        rstd_2 = tl.load(RSTD_ptr + row_idx * RSTD_row_stride + 1, mask=row_mask, other=0.0).to(tl.float32)
        rstd_1 = tl.load(RSTD_ptr + row_idx * RSTD_row_stride + 2, mask=row_mask, other=0.0).to(tl.float32)

        X_f32 = X_rows.to(tl.float32)
        dY_f32 = dY_rows.to(tl.float32)

        # Compute powers
        X_pow3 = X_f32 * X_f32 * X_f32
        X_pow2 = X_f32 * X_f32
        X_pow1 = X_f32

        # Accumulate bias gradient: dB = sum(dY)
        dB_acc += tl.sum(dY_f32)

        # Compute gradient w.r.t. input using closed-form formula
        # For p=3: ∂L/∂x from w0 * norm(x³)
        S_3 = tl.sum(dY_f32 * X_pow3, axis=1)  # sum over columns for each row
        grad_x_3 = w0 * (
            3.0 * X_pow2 * rstd_3[:, None] * dY_f32
            - (3.0 / n_cols) * X_pow2 * X_pow3 * (rstd_3[:, None] * rstd_3[:, None] * rstd_3[:, None]) * S_3[:, None]
        )

        # For p=2: ∂L/∂x from w1 * norm(x²)
        S_2 = tl.sum(dY_f32 * X_pow2, axis=1)
        grad_x_2 = w1 * (
            2.0 * X_pow1 * rstd_2[:, None] * dY_f32
            - (2.0 / n_cols) * X_pow1 * X_pow2 * (rstd_2[:, None] * rstd_2[:, None] * rstd_2[:, None]) * S_2[:, None]
        )

        # For p=1: ∂L/∂x from w2 * norm(x)
        S_1 = tl.sum(dY_f32 * X_pow1, axis=1)
        grad_x_1 = w2 * (
            1.0 * rstd_1[:, None] * dY_f32
            - (1.0 / n_cols) * X_pow1 * (rstd_1[:, None] * rstd_1[:, None] * rstd_1[:, None]) * S_1[:, None]
        )

        # Total gradient
        dX_f32 = grad_x_3 + grad_x_2 + grad_x_1

        # Store dX
        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_f32.to(X_ptr.dtype.element_ty),
            mask=block_mask,
        )

        # Accumulate weight gradients using closed-form: dW_p = rstd_p * S_p
        dW0_acc += tl.sum(rstd_3 * S_3)
        dW1_acc += tl.sum(rstd_2 * S_2)
        dW2_acc += tl.sum(rstd_1 * S_1)

    # Write this program's accumulated dW/dB to its dedicated scratch row
    tl.store(dW_scratch_ptr + pid * dW_scratch_stride + 0, dW0_acc)
    tl.store(dW_scratch_ptr + pid * dW_scratch_stride + 1, dW1_acc)
    tl.store(dW_scratch_ptr + pid * dW_scratch_stride + 2, dW2_acc)
    tl.store(dB_scratch_ptr + pid, dB_acc)


# -----------------------------------------------------------------------------
# Backward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _poly_norm_backward_kernel_npu(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dB_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NPU-optimized PolyNorm backward kernel with column blocking.

    Each program processes multiple rows using grid-stride loop.
    For each row, we process columns in blocks to avoid UB overflow.

    Two-pass algorithm:
    - First pass: compute S_p = sum(grad * x^p) for p ∈ {3, 2, 1}
    - Second pass: compute gradients dX, dW, dB
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    offsets = tl.arange(0, BLOCK_SIZE)

    # Load weights
    w0 = tl.load(W_ptr + 0).to(tl.float32)
    w1 = tl.load(W_ptr + 1).to(tl.float32)
    w2 = tl.load(W_ptr + 2).to(tl.float32)

    dw0_acc = 0.0
    dw1_acc = 0.0
    dw2_acc = 0.0
    db_acc = 0.0

    # Grid-stride loop over rows
    for row_idx in range(pid, n_rows, num_progs):
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load cached rstd values
        rstd_3 = tl.load(RSTD_row_ptr + 0).to(tl.float32)
        rstd_2 = tl.load(RSTD_row_ptr + 1).to(tl.float32)
        rstd_1 = tl.load(RSTD_row_ptr + 2).to(tl.float32)

        # First pass: compute S_p = sum(grad * x^p)
        S_3 = 0.0
        S_2 = 0.0
        S_1 = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            # Compute powers
            X_pow3 = X_block * X_block * X_block
            X_pow2 = X_block * X_block
            X_pow1 = X_block

            S_3 += tl.sum(dY_block * X_pow3)
            S_2 += tl.sum(dY_block * X_pow2)
            S_1 += tl.sum(dY_block * X_pow1)

        # Second pass: compute gradients
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            # Compute powers
            X_pow3 = X_block * X_block * X_block
            X_pow2 = X_block * X_block
            X_pow1 = X_block

            # Compute gradient w.r.t. input using closed-form formula
            # For p=3: ∂L/∂x from w0 * norm(x³)
            grad_x_3 = w0 * (
                3.0 * X_pow2 * rstd_3 * dY_block - (3.0 / n_cols) * X_pow2 * X_pow3 * (rstd_3 * rstd_3 * rstd_3) * S_3
            )

            # For p=2: ∂L/∂x from w1 * norm(x²)
            grad_x_2 = w1 * (
                2.0 * X_pow1 * rstd_2 * dY_block - (2.0 / n_cols) * X_pow1 * X_pow2 * (rstd_2 * rstd_2 * rstd_2) * S_2
            )

            # For p=1: ∂L/∂x from w2 * norm(x)
            grad_x_1 = w2 * (1.0 * rstd_1 * dY_block - (1.0 / n_cols) * X_pow1 * (rstd_1 * rstd_1 * rstd_1) * S_1)

            # Total gradient
            dX_block = grad_x_3 + grad_x_2 + grad_x_1

            # Store dX
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_ptr.dtype.element_ty), mask=mask)

            dw0_acc += tl.sum(rstd_3 * dY_block * X_pow3)
            dw1_acc += tl.sum(rstd_2 * dY_block * X_pow2)
            dw2_acc += tl.sum(rstd_1 * dY_block * X_pow1)
            db_acc += tl.sum(dY_block)

    tl.store(dW_ptr + 0, dw0_acc)
    tl.store(dW_ptr + 1, dw1_acc)
    tl.store(dW_ptr + 2, dw2_acc)
    tl.store(dB_ptr, db_acc)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_optimal_block_size(n_cols, is_forward: bool):
    """
    Calculate optimal block size using compute_default_tiling_strategy.

    Memory analysis for forward pass (per row):
    - Load: X_block (1 block)
    - Compute: X_pow3, X_pow2, X_pow1, norm_x3, norm_x2, norm_x1 (6 blocks)
    - Total: conservative estimate 8 blocks of memory

    Memory analysis for backward pass (per row):
    - Load: X_block, dY_block, RSTD (3 blocks)
    - Compute: X_pow3, X_pow2, X_pow1, grad_x_3, grad_x_2, grad_x_1 (6 blocks)
    - Total: conservative estimate 10 blocks of memory

    Args:
        n_cols: Number of columns in the tensor
        is_forward: Whether this is for forward pass (True) or backward pass (False)

    Returns:
        Optimal block size
    """
    if n_cols <= 2048:
        return triton.next_power_of_2(n_cols)

    memory_multiplier = 8.0 if is_forward else 10.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.8,
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

    Limits the grid to the minimum of:
    - The number of row blocks actually needed (ceil(n_rows / BLOCK_SIZE_M)), which
      prevents launching idle programs that would waste core cycles
    - NPU core count, which is the hardware concurrency upper bound

    Args:
        n_rows: Total number of rows to process
        block_size_m: Number of rows each program handles per iteration
        num_cores: Number of available NPU cores

    Returns:
        Effective grid size
    """
    num_row_blocks = triton.cdiv(n_rows, block_size_m)
    return min(num_cores, num_row_blocks)


# -----------------------------------------------------------------------------
# Forward and Backward Functions
# -----------------------------------------------------------------------------


def poly_norm_forward(X, W, B, eps=1e-6):
    """
    PolyNorm Forward Pass

    Args:
        X: input tensor of shape (*, H) where H is hidden dimension
        W: weight tensor of shape (3,) for [w0, w1, w2]
        B: bias scalar tensor
        eps: epsilon for numerical stability

    Returns:
        Y: output tensor of same shape as X
        X: reshaped input (for backward)
        RSTD: cached rstd values (for backward)
        BLOCK_SIZE: block size used
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape

    # Check constraints
    assert W.shape[0] == 3, "Weight tensor must have shape (3,)"
    assert B.numel() == 1, "Bias must be a scalar"

    # Get optimal block sizes
    BLOCK_SIZE = get_optimal_block_size(n_cols, True)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    # RSTD is to cache rstd for each row (3 values per row)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    RSTD = torch.empty((n_rows, 3), dtype=torch.float32, device=X.device)

    # Grid size
    num_cores = get_npu_core_count()

    # Choose kernel based on n_cols
    if n_cols <= 2048:
        # Small kernel: use 2D tensor loading
        grid_size = _compute_grid_size(n_rows, BLOCK_SIZE_M, num_cores)

        _poly_norm_forward_kernel_no_tiling[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            B,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        # Large kernel: use column blocking
        grid_size = min(num_cores, n_rows)

        _poly_norm_forward_kernel_npu[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            B,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return Y.view(*shape), X, RSTD


def poly_norm_backward(dY, X, W, RSTD, in_place):
    """
    PolyNorm Backward Pass

    Args:
        dY: gradient of output
        X: input tensor (already reshaped to 2D)
        W: weight tensor
        RSTD: cached rstd values from forward
        BLOCK_SIZE: block size from forward
        in_place: whether to in-place modify dY to store dX (saves memory)

    Returns:
        dX: gradient w.r.t. input
        dW: gradient w.r.t. weight
        dB: gradient w.r.t. bias
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Get optimal block sizes
    BLOCK_SIZE_BACKWARD = get_optimal_block_size(n_cols, False)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE_BACKWARD

    # Grid size
    num_cores = get_npu_core_count()

    # Allocate or reuse gradients
    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    # Choose kernel based on n_cols
    if n_cols <= 2048:
        # Small kernel: use 2D tensor loading with scratch buffers
        grid_size = _compute_grid_size(n_rows, BLOCK_SIZE_M, num_cores)

        # Allocate per-program scratch buffers for dW and dB
        dW_scratch = torch.empty((grid_size, 3), dtype=torch.float32, device=W.device)
        dB_scratch = torch.empty((grid_size,), dtype=torch.float32, device=W.device)

        _poly_norm_backward_kernel_no_tiling[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            dW_scratch,
            dW_scratch.stride(0),
            dB_scratch,
            n_rows,
            n_cols,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_BACKWARD,
        )

        dW = dW_scratch.sum(dim=0).to(W.dtype)
        dB = dB_scratch.sum().to(W.dtype)
    else:
        # Large kernel: use column blocking with atomic operations
        grid_size = min(num_cores, n_rows)

        dW = torch.zeros(3, dtype=torch.float32, device=W.device)
        dB = torch.zeros(1, dtype=torch.float32, device=W.device)

        _poly_norm_backward_kernel_npu[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            dW,
            dB,
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE_BACKWARD,
        )

        dW = dW.to(W.dtype)
        dB = dB.squeeze().to(W.dtype)

    # Reshape dX back to original shape
    dX = dX.view(*shape)

    return dX, dW, dB


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------


class LigerPolyNormFunction(torch.autograd.Function):
    """
    PolyNorm Function with forward and backward pass

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    Backward uses closed-form gradient:
        ∂L/∂x_i = Σ_p w_p * [p*x_i^(p-1) * grad_i/D_p - (p/d)*x_i^(2p-1) * S_p/(D_p³)]
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps=1e-6, in_place=True):
        """
        Args:
            X: input tensor of shape (B, T, H) or (BxT, H)
            W: weight tensor of shape (3,) for [w0, w1, w2]
            B: bias scalar
            eps: epsilon for numerical stability
            in_place: whether to in-place modify grad_output in backward (saves memory)

        Returns:
            Y: output tensor of same shape as X
        """
        Y, X, RSTD = poly_norm_forward(X, W, B, eps)
        ctx.in_place = in_place
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: gradient of output

        Returns:
            dX, dW, dB: gradients w.r.t. X, W, B
        """
        X, W, RSTD = ctx.saved_tensors
        dX, dW, dB = poly_norm_backward(grad_output, X, W, RSTD, ctx.in_place)
        return dX, dW, dB, None, None
