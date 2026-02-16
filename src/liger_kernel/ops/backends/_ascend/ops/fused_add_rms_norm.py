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
# Forward Kernel - Grid-Stride Loop with Row-wise Processing
# -----------------------------------------------------------------------------


@triton.jit
def _fused_add_rms_norm_forward_kernel_npu(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    NPU-optimized fused_add_rms_norm forward kernel.

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
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        S_row_ptr = S_ptr + row_idx * S_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        R_row_ptr = R_ptr + row_idx * R_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride

        # Accumulator for mean_square computation across all column blocks
        sum_square = 0.0

        # First pass: compute S_row = X_row + R_row and accumulate sum of squares
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            R_block = tl.load(R_row_ptr + col_offsets, mask=mask, other=0.0, eviction_policy="evict_first")
            S_block = X_block + R_block

            # Store S_row
            tl.store(S_row_ptr + col_offsets, S_block, mask=mask, cache_modifier=".cg")

            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                S_block = S_block.to(tl.float32)

            # Accumulate sum of squares (only for valid elements)
            sum_square += tl.sum(tl.where(mask, S_block * S_block, 0.0))

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

            # Load S_block (already computed in first pass)
            S_block = tl.load(S_row_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".ca")
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            # Apply casting based on mode
            if casting_mode == _CASTING_MODE_GEMMA:
                S_block = S_block.to(tl.float32)
                W_block = W_block.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_block = S_block.to(tl.float32)

            # Normalize
            S_block = S_block * rstd

            # Cast back for Llama mode before weight multiplication
            if casting_mode == _CASTING_MODE_LLAMA:
                S_block = S_block.to(X_DTYPE)
            # Apply weight
            Y_block = S_block * (offset + W_block)

            # Cast back for Gemma mode
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_block = Y_block.to(X_DTYPE)

            # Store result
            tl.store(Y_row_ptr + col_offsets, Y_block, mask=mask)


# -----------------------------------------------------------------------------
# Backward Kernel - Grid-Stride Loop with Row-wise Processing
# -----------------------------------------------------------------------------


@triton.jit
def _fused_add_rms_norm_backward_kernel_npu(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
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
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    has_dS_out: tl.constexpr,
):
    """
    NPU-optimized fused_add_rms_norm backward kernel.

    Each program processes multiple rows using grid-stride loop.
    For each row, we process columns in blocks to avoid UB overflow.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Initialize dW accumulator (per-program, will be reduced later)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Grid-stride loop over rows
    for row_idx in tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES):
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

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            # Convert to fp32 for computation
            X_block = X_block.to(tl.float32)

            # Compute m based on casting mode
            W_offset = W_block + offset

            if casting_mode == _CASTING_MODE_LLAMA:
                m = (dY_block * W_offset).to(tl.float32)
            elif casting_mode == _CASTING_MODE_GEMMA:
                dY_block = dY_block.to(tl.float32)
                m = dY_block * W_offset
            else:
                m = dY_block * W_offset

            # Accumulate sum(m * X)
            sum_m_X += tl.sum(tl.where(mask, m * X_block, 0.0))

        # Compute the correction factor
        correction_factor = -(1.0 / n_cols) * rstd * rstd * sum_m_X

        # Second pass: compute gradients
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            X_block = X_block.to(tl.float32)

            # Compute m based on casting mode
            W_offset = W_block + offset

            if casting_mode == _CASTING_MODE_LLAMA:
                m = (dY_block * W_offset).to(tl.float32)
            elif casting_mode == _CASTING_MODE_GEMMA:
                dY_block_fp32 = dY_block.to(tl.float32)
                m = dY_block_fp32 * W_offset
            else:
                m = dY_block * W_offset

            # Compute dX
            dX_block = rstd * m + rstd * correction_factor * X_block

            # Add dS_out gradient if present
            if has_dS_out:
                dS_out_row_ptr = dS_out_ptr + row_idx * dS_out_row_stride
                dS_out_block = tl.load(dS_out_row_ptr + col_offsets, mask=mask, other=0.0)
                dX_block += dS_out_block

            # Store dX
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_dtype), mask=mask)

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
    - Load: X_block, R_block, W_block (3 blocks)
    - Store: S_block, Y_block (2 blocks)
    - Compute: S_block, Y_block intermediate (2 blocks)
    - Total: conservative estimate 8 blocks of memory

    Memory analysis for backward pass (per row):
    - Load: dY_block, X_block, W_block, existing_dW (4 blocks)
    - Store: dX_block, new_dW (2 blocks)
    - Compute: m, dX_block intermediate, dW_block intermediate (3 blocks)
    - Additional: dS_out_block if present (1 block)
    - Total: conservative estimate 12 blocks of memory

    Args:
        n_cols: Number of columns in the tensor

    Returns:
        Optimal block size
    """
    if n_cols <= 2048:
        return triton.next_power_of_2(n_cols)

    memory_multiplier = 8.0 if is_forward else 12.0

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


def fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    R = R.view(-1, dim)
    n_rows, n_cols = X.shape
    X_DTYPE = torch_dtype_to_triton(X.dtype)

    # Get optimal block size for column processing
    BLOCK_SIZE = get_optimal_block_size(n_cols, True)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    # RSTD is always fp32 for Llama/Gemma modes
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # Check constraints
    assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension"

    # Grid size limited to NPU core count
    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    _fused_add_rms_norm_forward_kernel_npu[(grid_size,)](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        X,
        X.stride(0),
        R,
        R.stride(0),
        W,
        RSTD,
        RSTD.stride(0),
        n_rows,
        n_cols,
        eps,
        offset,
        casting_mode,
        X_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=4,
    )

    return Y.view(*shape), S.view(*shape), RSTD, casting_mode


def fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD, offset, casting_mode, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    if dS_out is not None:
        dS_out = dS_out.view(-1, dim)
    S = S.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Get NPU core count for grid size
    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    # Get optimal block size for backward pass
    BLOCK_SIZE = get_optimal_block_size(n_cols, False)

    # fp32 for numerical stability
    _dW = torch.zeros((grid_size, n_cols), dtype=torch.float32, device=W.device)

    if in_place:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    _fused_add_rms_norm_backward_kernel_npu[(grid_size,)](
        dY,
        dY.stride(0),
        dS_out if dS_out is not None else dY,  # Dummy pointer if dS_out is None
        dS_out.stride(0) if dS_out is not None else 0,
        dX,
        dX.stride(0),
        S,
        S.stride(0),
        torch_to_triton_dtype[S.dtype],
        W,
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        offset,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=4,
        has_dS_out=dS_out is not None,
    )

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)

    return dX, dX, dW  # dR is equal to dX


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------


class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """
    NPU-optimized fused operation for residual addition and RMSNorm.

    This implementation solves two key issues:
    1. UB overflow when n_cols is too large (by using column-wise blocking)
    2. Efficient multi-row processing (by using grid-stride loop with core count limit)
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, R, W, eps, offset=0.0, casting_mode="llama", in_place=False):
        """
        X: (B, T, H) or (BxT, H)
        R: (B, T, H) or (BxT, H)
        W: (H,)
        """
        Y, S, RSTD, casting_mode = fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.save_for_backward(S, W, RSTD)
        return Y, S

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        """
        dY: (B, T, H) or (BxT, H)
        dS_out: (B, T, H) or (BxT, H)
        """
        S, W, RSTD = ctx.saved_tensors
        dX, dR, dW = fused_add_rms_norm_backward(
            dY,
            dS_out,
            S,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.in_place,
        )

        return dX, dR, dW, None, None, None, None
