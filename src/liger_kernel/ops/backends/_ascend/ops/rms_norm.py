import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import torch_to_triton_dtype

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


def torch_dtype_to_triton(dtype):
    mapping = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }
    return mapping.get(dtype, tl.float32)


# -----------------------------------------------------------------------------
# Forward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    NPU-optimized rms_norm forward kernel for small n_cols (< 2048).

    Performance optimizations:
    1. Use 2D vector loading to maximize UB utilization (e.g., (1,2048), (2,1024), (4,512))
    2. Process multiple rows at once using 2D indexing
    3. Keep data in registers, minimize conversions
    4. Use optimal cache policies

    Used when n_cols < 2048 to avoid the overhead of column blocking.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    # Grid-stride loop setup for 2D blocks
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        # Load multiple rows at once using 2D indexing
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        # Compute sum_square for all rows
        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            X_rows = X_rows.to(tl.float32)

        sum_squares = tl.sum(tl.where(block_mask, X_rows * X_rows, 0.0), axis=1)

        # Compute rstd for all rows
        mean_squares = sum_squares / n_cols
        rstd_rows = rsqrt(mean_squares + eps)

        # Store rstd_rows
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd_rows, mask=row_mask)

        # Apply casting based on mode
        if casting_mode == _CASTING_MODE_GEMMA:
            X_rows = X_rows.to(tl.float32)
            if elementwise_affine:
                W_row_fp32 = W_row.to(tl.float32)
        elif casting_mode == _CASTING_MODE_LLAMA:
            X_rows = X_rows.to(tl.float32)

        # Normalize
        X_rows = X_rows * rstd_rows[:, None]

        # Cast back for Llama mode before weight multiplication
        if casting_mode == _CASTING_MODE_LLAMA:
            X_rows = X_rows.to(X_DTYPE)

        # Apply weight
        if elementwise_affine:
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_rows = X_rows * (offset + W_row_fp32[None, :])
            else:
                Y_rows = X_rows * (offset + W_row[None, :])
        else:
            Y_rows = X_rows

        # Cast back for Gemma mode
        if casting_mode == _CASTING_MODE_GEMMA:
            Y_rows = Y_rows.to(X_DTYPE)

        # Store results
        tl.store(Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :], Y_rows, mask=block_mask)


# -----------------------------------------------------------------------------
# Forward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_forward_kernel_tiled(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NPU-optimized rms_norm forward kernel for large n_cols (>= 2048).

    This kernel processes rows using a grid-stride loop pattern:
    1. Each program handles multiple rows
    2. For each row, we process it in column chunks of BLOCK_SIZE
    3. Grid size is limited to NPU core count to avoid resource overflow

    This solves two problems:
    1. UB overflow when n_cols is too large (original kernel used n_cols as BLOCK_SIZE)
    2. Efficient multi-row processing within a single kernel launch
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    offsets = tl.arange(0, BLOCK_SIZE)
    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Accumulator for mean_square computation across all column blocks
        sum_square = 0.0

        # First pass: accumulate sum of squares
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")

            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)

            # Accumulate sum of squares (only for valid elements)
            sum_square += tl.sum(X_block * X_block)

        # Compute rstd for this row
        mean_square = sum_square / n_cols

        rstd = rsqrt(mean_square + eps)

        # Store rstd
        tl.store(RSTD_row_ptr, rstd)

        # Second pass: normalize and multiply by weight
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            # Load X_block
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca")

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            # Apply casting based on mode
            if casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)
                if elementwise_affine:
                    W_block = W_block.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                X_block = X_block.to(tl.float32)

            # Normalize
            X_block = X_block * rstd

            # Cast back for Llama mode before weight multiplication
            if casting_mode == _CASTING_MODE_LLAMA:
                X_block = X_block.to(X_DTYPE)

            # Apply weight
            if elementwise_affine:
                Y_block = X_block * (offset + W_block)
            else:
                Y_block = X_block

            # Cast back for Gemma mode
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_block = Y_block.to(X_DTYPE)

            # Store result
            tl.store(Y_row_ptr + col_offsets, Y_block, mask=mask)


# -----------------------------------------------------------------------------
# Backward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_backward_kernel_no_tiling(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    NPU-optimized rms_norm backward kernel for small n_cols (< 2048).

    Performance optimizations:
    1. Keep all data in registers, minimize conversions
    2. Reuse X_normalized (X * rstd) for both dX and dW
    3. Optimize computation order to reduce register pressure
    4. Combine operations where possible
    5. Use 2D vector loading to maximize UB utilization (e.g., (1,2048), (2,1024), (4,512))
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-stride loop setup for 2D blocks
    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    # Load W once for all iterations
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_offset = W_row + offset

    # Grid-stride loop over row blocks
    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]

        dY_rows = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        # Load rstd for all rows in the block
        rstd_rows = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, mask=row_mask, other=0.0)

        # Convert X to fp32 once
        X_rows = X_rows.to(tl.float32)

        # Compute X_normalized (reused in dX and dW)
        X_normalized = X_rows * rstd_rows[:, None]

        # Compute m based on casting mode and elementwise_affine
        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                m_rows = (dY_rows * W_offset[None, :]).to(tl.float32)
                # For dW in Llama mode, we need X_normalized in original dtype
                X_normalized = X_normalized.to(X_dtype)
            elif casting_mode == _CASTING_MODE_GEMMA:
                m_rows = dY_rows.to(tl.float32) * W_offset[None, :]
            else:
                m_rows = dY_rows * W_offset[None, :]
        else:
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                m_rows = dY_rows.to(tl.float32)
            else:
                m_rows = dY_rows

        # Compute sum(m * X) for correction factor
        sum_m_X = tl.sum(tl.where(block_mask, m_rows * X_rows, 0.0), axis=1)

        # Compute correction factor
        correction_factors = -(1.0 / n_cols) * rstd_rows * rstd_rows * sum_m_X

        # Compute dX = rstd * m + rstd * correction_factor * X
        dX_rows = rstd_rows[:, None] * m_rows + rstd_rows[:, None] * correction_factors[:, None] * X_rows

        # Store dX
        tl.store(dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :], dX_rows.to(X_dtype), mask=block_mask)

        if elementwise_affine:
            # Compute dW contribution: dY * X_normalized
            dW_rows = (dY_rows * X_normalized).to(tl.float32)

            # Accumulate to per-program dW buffer
            dW_row_ptr = dW_ptr + pid * dW_row_stride
            existing_dW = tl.load(dW_row_ptr + col_offsets, mask=col_mask, other=0.0)
            new_dW = existing_dW + tl.sum(tl.where(block_mask, dW_rows, 0.0), axis=0)
            tl.store(dW_row_ptr + col_offsets, new_dW, mask=col_mask)


# -----------------------------------------------------------------------------
# Backward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _rms_norm_backward_kernel_tiled(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NPU-optimized rms_norm backward kernel for large n_cols (>= 2048).

    Each program processes multiple rows using grid-stride loop.
    For each row, we process columns in blocks to avoid UB overflow.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Initialize dW accumulator (per-program, will be reduced later)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs):
        # Base pointers for this row
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Load rstd for this row
        rstd = tl.load(RSTD_row_ptr)

        # First pass: compute sum(m * X) for the correction term
        sum_m_X = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")

            # Convert to fp32 for computation
            X_block = X_block.to(tl.float32)

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
                W_offset = W_block + offset

                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    dY_block = dY_block.to(tl.float32)
                    m = dY_block * W_offset
                else:
                    m = dY_block * W_offset
            else:
                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = dY_block.to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block

            # Accumulate sum(m * X)
            sum_m_X += tl.sum(m * X_block)

        # Compute the correction factor
        correction_factor = -(1.0 / n_cols) * rstd * rstd * sum_m_X

        # Second pass: compute gradients
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)

            X_block = X_block.to(tl.float32)

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset

                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dY_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    dY_block = dY_block.to(tl.float32)
                    m = dY_block * W_offset
                else:
                    m = dY_block * W_offset
            else:
                # Compute m based on casting mode
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = dY_block.to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dY_block.to(tl.float32)
                else:
                    m = dY_block

            # Compute dX
            dX_block = rstd * m + rstd * correction_factor * X_block

            # Store dX
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_dtype), mask=mask)

            if elementwise_affine:
                # Compute dW contribution (accumulate per program)
                if casting_mode == _CASTING_MODE_LLAMA:
                    dW_block = dY_block * (X_block * rstd).to(X_dtype)
                else:
                    dW_block = dY_block * (X_block * rstd)

                # Atomic add to dW_ptr (each program writes to its own row)
                dW_row_ptr = dW_ptr + pid * dW_row_stride

                # Load existing dW, add contribution, store back
                existing_dW = tl.load(dW_row_ptr + col_offsets, mask=mask, other=0.0)
                new_dW = existing_dW + dW_block.to(tl.float32)
                tl.store(dW_row_ptr + col_offsets, new_dW, mask=mask)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_optimal_block_size(n_cols, is_forward: bool):
    """
    Calculate optimal block size for forward pass using compute_default_tiling_strategy.

    Memory analysis for forward pass (per row):
    - Load: X_block, W_block (2 blocks)
    - Compute: X_block (fp32), Y_block (1-2 blocks)
    - Total: conservative estimate 6 blocks of memory

    Memory analysis for backward pass (per row):
    - Load: dY_block, X_block, W_block (3 blocks)
    - Compute: m, dX_block, dW_block (3 blocks)
    - Store: dX_block, accumulated dW (2 blocks)
    - Total: conservative estimate 8 blocks of memory

    Args:
        n_cols: Number of columns in the tensor
        is_forward: Whether this is for forward pass

    Returns:
        Optimal block size
    """
    if n_cols <= 2048:
        return triton.next_power_of_2(n_cols)

    memory_multiplier = 6.0 if is_forward else 8.0

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


# -----------------------------------------------------------------------------
# Forward and Backward Functions
# -----------------------------------------------------------------------------


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def rms_norm_forward(X, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    X_DTYPE = torch_dtype_to_triton(X.dtype)

    # Get optimal block size for column processing
    BLOCK_SIZE = get_optimal_block_size(n_cols, True)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    # RSTD is always fp32 for Llama/Gemma modes
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        # Check constraints
        assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension"
        elementwise_affine = True
    else:
        elementwise_affine = False

    # Grid size limited to NPU core count
    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    # Choose kernel based on n_cols
    if n_cols <= 2048:
        # Use no-tiling kernel for small n_cols
        _rms_norm_forward_kernel_no_tiling[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine,
            X_DTYPE,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        # Use tiled kernel for large n_cols
        _rms_norm_forward_kernel_tiled[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine,
            X_DTYPE,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return Y.view(*shape), X, RSTD, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Get NPU core count for grid size
    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    # Get optimal block size for backward pass
    BLOCK_SIZE = get_optimal_block_size(n_cols, False)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    if W is not None:
        # fp32 for numerical stability
        _dW = torch.zeros((grid_size, n_cols), dtype=torch.float32, device=W.device)
        elementwise_affine = True
    else:
        _dW = None
        elementwise_affine = False

    if in_place:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    # Choose kernel based on n_cols
    if n_cols <= 2048:
        # Use no-tiling kernel for small n_cols
        _rms_norm_backward_kernel_no_tiling[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0) if elementwise_affine else 0,
            n_rows,
            n_cols,
            offset,
            casting_mode,
            elementwise_affine,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        # Use tiled kernel for large n_cols
        _rms_norm_backward_kernel_tiled[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0) if elementwise_affine else 0,
            n_rows,
            n_cols,
            offset,
            casting_mode,
            elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    dX = dX.view(*shape)

    if elementwise_affine:
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = None

    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if isinstance(X, torch.distributed.tensor.DTensor):
            # Input tensor is output of a tensor parallel module and
            # needs to be gathered to a local tensor to compute
            # RMSE layer norm on each TP worker.
            # TODO: support CP.
            X = X.full_tensor()

        Y, X, RSTD, casting_mode = rms_norm_forward(X, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None
        if isinstance(dY, torch.distributed.tensor.DTensor):
            # Gradients are output of a tensor parallel module and
            # needs to be gathered to a local tensor for computing RMSE layer.
            # TODO: support CP.
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.in_place)
        return dX, dW, None, None, None, None, None
