import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch.nn.modules.utils import _pair

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.backends._ascend.ub_manager import get_ub_manager
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def _fused_mask_softmax_fwd_kernel(
    scores_ptr,
    out_ptr,
    stride_b,
    stride_row,
    N,
    L,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused forward kernel: causal masking + softmax with grid-stride loop.

    Args:
        scores_ptr: Input scores tensor pointer
        out_ptr: Output softmax probabilities pointer
        stride_b: Batch stride
        stride_row: Row stride
        N: Number of batches
        L: Sequence length
        BLOCK_SIZE: Block size for processing columns
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_rows = N * L

    # Grid-stride loop over all rows
    for linear_idx in tl.range(pid, total_rows, num_progs):
        batch_id = linear_idx // L
        row_idx = linear_idx % L

        row_ptr = scores_ptr + batch_id * stride_b + row_idx * stride_row
        out_row_ptr = out_ptr + batch_id * stride_b + row_idx * stride_row

        valid_len = row_idx + 1

        # First pass: online softmax
        max_val = float("-inf")
        d_sum = 0.0
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            vals = tl.load(row_ptr + col_idx, mask=col_mask, other=float("-inf"))
            m_block = tl.max(vals)
            m_new = tl.maximum(max_val, m_block)
            d_sum = d_sum * tl.exp(max_val - m_new) + tl.sum(tl.exp(vals - m_new))
            max_val = m_new

        # Second pass: normalize and store
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            vals = tl.load(row_ptr + col_idx, mask=col_mask, other=float("-inf"))
            exp_vals = tl.exp(vals - max_val)
            probs = exp_vals / d_sum
            tl.store(out_row_ptr + col_idx, probs, mask=col_mask)

        # Store zeros for masked positions
        for block_start in range(valid_len, L, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < L
            tl.store(out_row_ptr + col_idx, 0.0, mask=col_mask)


@triton.jit
def _fused_mask_softmax_bwd_kernel(
    grad_out_ptr,
    probs_ptr,
    grad_scores_ptr,
    stride_b,
    stride_row,
    N,
    L,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused backward kernel: softmax gradient + causal masking with grid-stride loop.

    Computes: grad_scores = probs * (grad_out - dot(grad_out, probs))
    where dot is computed only over valid (non-masked) positions.

    Args:
        grad_out_ptr: Gradient w.r.t. softmax output
        probs_ptr: Softmax probabilities from forward pass
        grad_scores_ptr: Output gradient w.r.t. input scores
        stride_b: Batch stride
        stride_row: Row stride
        N: Number of batches
        L: Sequence length
        BLOCK_SIZE: Block size for processing columns
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_rows = N * L

    # Grid-stride loop over all rows
    for linear_idx in tl.range(pid, total_rows, num_progs):
        batch_id = linear_idx // L
        row_idx = linear_idx % L

        grad_row_ptr = grad_out_ptr + batch_id * stride_b + row_idx * stride_row
        probs_row_ptr = probs_ptr + batch_id * stride_b + row_idx * stride_row
        out_row_ptr = grad_scores_ptr + batch_id * stride_b + row_idx * stride_row

        valid_len = row_idx + 1

        # First pass: compute dot product (grad_out * probs)
        dot = 0.0
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            grad_vals = tl.load(grad_row_ptr + col_idx, mask=col_mask, other=0.0)
            prob_vals = tl.load(probs_row_ptr + col_idx, mask=col_mask, other=0.0)
            dot += tl.sum(tl.where(col_mask, grad_vals * prob_vals, 0.0))

        # Second pass: compute gradient
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            grad_vals = tl.load(grad_row_ptr + col_idx, mask=col_mask, other=0.0)
            prob_vals = tl.load(probs_row_ptr + col_idx, mask=col_mask, other=0.0)
            grad_scores = prob_vals * (grad_vals - dot)
            tl.store(out_row_ptr + col_idx, grad_scores, mask=col_mask)

        # Zero out masked positions
        for block_start in range(valid_len, L, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < L
            tl.store(out_row_ptr + col_idx, 0.0, mask=col_mask)


@triton.jit
def _fused_mask_sparsemax_bwd_kernel(
    grad_out_ptr,
    probs_ptr,
    grad_scores_ptr,
    stride_b,
    stride_row,
    N,
    L,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused backward kernel: sparsemax gradient + causal masking.

    Sparsemax backward:
    grad_input[i] = grad_out[i] - (sum_support(grad_out) / |support|) if output[i] > 0
                  = 0 otherwise

    Args:
        grad_out_ptr: Gradient w.r.t. sparsemax output
        probs_ptr: Sparsemax probabilities from forward pass
        grad_scores_ptr: Output gradient w.r.t. input scores
        stride_b: Batch stride
        stride_row: Row stride
        L: Sequence length
        BLOCK_SIZE: Block size for processing columns
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_rows = N * L

    # Grid-stride loop over all rows
    for linear_idx in tl.range(pid, total_rows, num_progs):
        batch_id = linear_idx // L
        row_idx = linear_idx % L

        grad_row_ptr = grad_out_ptr + batch_id * stride_b + row_idx * stride_row
        probs_row_ptr = probs_ptr + batch_id * stride_b + row_idx * stride_row
        out_row_ptr = grad_scores_ptr + batch_id * stride_b + row_idx * stride_row

        valid_len = row_idx + 1

        # First pass: compute support count and gradient sum
        supp_cnt = 0.0
        go_sum = 0.0
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            prob_vals = tl.load(probs_row_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
            grad_vals = tl.load(grad_row_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
            supp = prob_vals > 0.0
            go_sum += tl.sum(tl.where(supp & col_mask, grad_vals, 0.0))
            supp_cnt += tl.sum(tl.where(supp & col_mask, 1.0, 0.0))

        # Second pass: compute gradient
        avg_grad = go_sum / tl.maximum(supp_cnt, 1e-6)
        for block_start in range(0, valid_len, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < valid_len
            prob_vals = tl.load(probs_row_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
            grad_vals = tl.load(grad_row_ptr + col_idx, mask=col_mask, other=0.0).to(tl.float32)
            supp = prob_vals > 0.0
            grad_scores = tl.where(supp, grad_vals - avg_grad, 0.0)
            tl.store(out_row_ptr + col_idx, grad_scores, mask=col_mask)

        # Zero out masked positions
        for block_start in range(valid_len, L, BLOCK_SIZE):
            col_idx = block_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_idx < L
            tl.store(out_row_ptr + col_idx, 0.0, mask=col_mask)


@triton.jit
def _mask_row_kernel(
    scores_ptr,
    out_ptr,
    stride_b,
    stride_row,
    L,
    mask_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D-tiled causal masking kernel.

    Grid:
        axis 0 (pid_m): row-block index
        axis 1 (pid_n): col-block index
        axis 2 (pid_b): batch index

    Each program handles a BLOCK_SIZE x BLOCK_SIZE tile.
    """

    pid_m = tl.program_id(0)  # row block
    pid_n = tl.program_id(1)  # col block
    pid_b = tl.program_id(2)  # batch

    # Compute row/col indices
    row_start = pid_m * BLOCK_SIZE
    col_start = pid_n * BLOCK_SIZE

    row_idx = row_start + tl.arange(0, BLOCK_SIZE)
    col_idx = col_start + tl.arange(0, BLOCK_SIZE)

    row_mask = row_idx < L
    col_mask = col_idx < L

    ptrs = scores_ptr + pid_b * stride_b + row_idx[:, None] * stride_row + col_idx[None, :]
    out_ptrs = out_ptr + pid_b * stride_b + row_idx[:, None] * stride_row + col_idx[None, :]

    vals = tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    causal = col_idx[None, :] <= row_idx[:, None]
    masked_vals = tl.where(causal, vals, mask_val)

    tl.store(out_ptrs, masked_vals, mask=row_mask[:, None] & col_mask[None, :])


def get_2d_optimal_block_size(
    n: int,
    dtype_size: int,
    memory_multiplier: float,
    safety_margin: float = 0.9,
    min_block: int = 16,
    max_block: int = 512,
):
    """
    Compute optimal 2D block sizes (BLOCK_SIZE, BLOCK_SIZE) for a square matrix.

    Args:
        n: matrix size (N x N)
        ub_capacity_bits: UB capacity in bits
        dtype_size: bytes per element
        memory_multiplier: estimated live buffer multiplier
        safety_margin: UB safety factor
        min_block: minimum block size
        max_block: maximum block size

    Returns:
        BLOCK_SIZE
    """
    ub_manager = get_ub_manager()

    dtype_size = max(dtype_size, 4)

    # Safe UB budget
    safe_ub = int(ub_manager.ub_capacity_bits * safety_margin)

    # Max tile area
    max_area = safe_ub // (memory_multiplier * dtype_size * 8)
    max_area = max(1, max_area)

    # Ideal square tile size
    block = int(math.sqrt(max_area))

    # Clamp to problem size
    block = min(block, n, max_block)
    block = max(block, min_block)

    block = triton.next_power_of_2(block)
    if block > n:
        block = n

    return block


def get_optimal_size_fused_mask_softmax(L: int, is_forward: bool = True, dtype_size: int = 2):
    """
    Compute optimal block size for fused mask kernel.
    """

    if L <= 2048:
        return triton.next_power_of_2(L)

    memory_multiplier = 6.0 if is_forward else 8.0  # 3 loads + 1 store + reduction overhead

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=dtype_size,
        memory_multiplier=memory_multiplier,
        shapes=((L,),),
        tiling_dims=(0,),
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return max(2048, block_size)

    return 2048


def mask_zero_rowwise(scores: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for causal masking with zero values.
    Uses 1D row-wise processing.

    Args:
        scores: Input scores tensor of shape (*batch, L, L)

    Returns:
        Masked scores tensor with future positions set to 0.0
    """
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.view(N, L, L)
    out = torch.empty_like(scores_f)

    BLOCK_SIZE = get_2d_optimal_block_size(L, dtype_size=scores_f.element_size(), memory_multiplier=12.0)

    grid = (triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N)
    _mask_row_kernel[grid](
        scores_f,
        out,
        scores_f.stride(0),
        scores_f.stride(1),
        L,
        mask_val=0.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(*batch, L, L)


def fused_mask_softmax_forward(scores: torch.Tensor) -> torch.Tensor:
    """
    Fused forward pass: causal masking + softmax.

    Args:
        scores: Input scores tensor of shape (*batch, L, L)

    Returns:
        Softmax probabilities with causal masking applied
    """
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.view(N, L, L)
    out = torch.empty_like(scores_f)

    stride_b = scores_f.stride(0)
    stride_row = scores_f.stride(1)

    BLOCK_SIZE = get_optimal_size_fused_mask_softmax(L, is_forward=True)

    # Grid size limited to NPU core count for better resource utilization
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, N * L)

    _fused_mask_softmax_fwd_kernel[(grid_size,)](
        scores_f,
        out,
        stride_b,
        stride_row,
        N,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(*batch, L, L)


def fused_mask_softmax_backward(grad_out: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    Fused backward pass: softmax gradient + causal masking.

    Args:
        grad_out: Gradient w.r.t. softmax output of shape (*batch, L, L)
        probs: Softmax probabilities from forward pass of shape (*batch, L, L)

    Returns:
        Gradient w.r.t. input scores with causal masking applied
    """
    *batch, L, _ = grad_out.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_out_f = grad_out.view(N, L, L)
    probs_f = probs.view(N, L, L)
    grad_scores = torch.empty_like(grad_out_f)

    BLOCK_SIZE = get_optimal_size_fused_mask_softmax(L, is_forward=False)

    # Grid size limited to NPU core count for better resource utilization
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, N * L)

    _fused_mask_softmax_bwd_kernel[(grid_size,)](
        grad_out_f,
        probs_f,
        grad_scores,
        grad_out_f.stride(0),
        grad_out_f.stride(1),
        N,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_scores.view(*batch, L, L)


def fused_mask_sparsemax_forward(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass: causal masking + sparsemax using reference implementation.
    Uses one-axis grid (one program per row).

    Args:
        scores: Input scores tensor of shape (*batch, L, L)

    Returns:
        Tuple of (sparsemax probabilities, flattened output for backward)
    """
    from liger_kernel.ops.sparsemax import _sparsemax_forward

    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.view(N, L, L)
    scores_masked = torch.empty_like(scores_f)

    # BLOCK_SIZE = get_optimal_size_fused_mask_softmax(L, is_forward=True)
    BLOCK_SIZE = get_2d_optimal_block_size(L, dtype_size=scores_f.element_size(), memory_multiplier=12.0)

    grid = (triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N)
    _mask_row_kernel[grid](
        scores_f,
        scores_masked,
        scores_f.stride(0),
        scores_f.stride(1),
        L,
        mask_val=-1e9,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Apply sparsemax row by row
    probs, probs_flat = _sparsemax_forward(scores_masked, dim=-1)

    return probs.view(*batch, L, L), probs_flat


def fused_mask_sparsemax_backward(grad_out: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    Fused backward pass: sparsemax gradient + causal masking.

    Args:
        grad_out: Gradient w.r.t. sparsemax output of shape (*batch, L, L)
        probs: Sparsemax probabilities from forward pass of shape (*batch, L, L)

    Returns:
        Gradient w.r.t. input scores with causal masking applied
    """
    *batch, L, _ = grad_out.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_out_f = grad_out.view(N, L, L)
    probs_f = probs.view(N, L, L)
    grad_scores = torch.empty_like(grad_out_f)

    BLOCK_SIZE = get_optimal_size_fused_mask_softmax(L, is_forward=False)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, N * L)

    _fused_mask_sparsemax_bwd_kernel[(grid_size,)](
        grad_out_f,
        probs_f,
        grad_scores,
        grad_out_f.stride(0),
        grad_out_f.stride(1),
        N,
        L,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_scores.view(*batch, L, L)


class LigerMultiTokenAttentionFunction(torch.autograd.Function):
    """
    NPU-optimized Multi-Token Attention using 1D row-wise processing.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, scores, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, sparse=False):
        # Use fused mask+softmax or mask+sparsemax kernel
        if sparse:
            probs, probs_flat = fused_mask_sparsemax_forward(scores)
            ctx.save_for_backward(probs, probs_flat, weight, bias)
            ctx.sparse = True
        else:
            probs = fused_mask_softmax_forward(scores)
            ctx.save_for_backward(probs, weight, bias)
            ctx.sparse = False

        # Apply convolution with the attention weights
        out_conv = F.conv2d(
            probs,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        out_masked = mask_zero_rowwise(out_conv)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups

        return out_masked

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out):
        if ctx.sparse:
            probs, probs_flat, weight, bias = ctx.saved_tensors
        else:
            probs, weight, bias = ctx.saved_tensors

        stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        grad_out_masked = mask_zero_rowwise(grad_out)

        # Backward through convolution
        grad_probs = F.conv_transpose2d(
            grad_out_masked, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

        grad_weight = torch.nn.grad.conv2d_weight(
            input=probs,
            weight_size=weight.shape,
            grad_output=grad_out_masked,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        grad_bias = None
        if bias is not None:
            grad_bias = grad_out_masked.sum(dim=(0, 2, 3))

        grad_scores = grad_probs

        # Use fused softmax or sparsemax backward kernel
        if ctx.sparse:
            grad_scores = fused_mask_sparsemax_backward(grad_probs, probs)
        else:
            grad_scores = fused_mask_softmax_backward(grad_probs, probs)

        return (grad_scores, grad_weight, grad_bias, None, None, None, None, None)
