from typing import Optional

import torch
import triton
import triton.language as tl

from triton.language.math import tanh

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def liger_cross_entropy_forward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    lse_ptr,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_rows,
    ce_stats_ptr,
    ignore_index,
    ls_eps,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    Triton kernel: cross-entropy forward (per-row loss, optional z-loss, LSE, accuracy, argmax).

    Parameters:
        X_ptr: Logits pointer; logical shape ``(n_rows, n_cols)``, row-major with stride ``X_stride``.
        X_stride (int64): Stride between consecutive rows of logits (columns contiguous).
        Y_ptr: Target class indices per row; ignored rows compare equal to ``ignore_index``.
        weight_ptr: Per-class weights ``[n_cols]`` when ``HAS_WEIGHT``; unused otherwise.
        loss_ptr: Output per-row (unreduced) loss including optional z-loss term.
        z_loss_ptr: Per-row z-loss ``lse_square_scale * lse**2`` when ``RETURN_Z_LOSS``; unused otherwise.
        lse_ptr: Per-row log-sum-exp when ``RETURN_LSE``; caller may pass an unused tensor binding when false.
        token_accuracy_ptr: Per-row float ``1.0`` if argmax matches target else ``0.0`` when ``RETURN_TOKEN_ACCURACY``.
        token_accuracy_stride (int64): Row stride of ``token_accuracy_ptr`` (or ``0`` when disabled).
        predicted_tokens_ptr: Per-row argmax index (``int64``) when ``RETURN_PREDICTED_TOKENS``; ``-1`` for ignored rows.
        predicted_tokens_stride (int64): Row stride of ``predicted_tokens_ptr`` (or ``0`` when disabled).
        n_cols (int): Vocabulary size (number of classes).
        n_rows (int): Number of rows (batch * sequence length).
        ce_stats_ptr: Float32 vector ``[inv_n_scale, inv_sum_weight_scale, weight_sum]`` for mean reduction and smoothing.
        ignore_index (int): Label value to skip (no loss contribution; optional outputs zeroed or ``-1``).
        ls_eps (float): ``label_smoothing / n_cols`` used in the weighted smoothing path when ``label_smoothing > 0``.
        lse_square_scale (constexpr float): Coefficient on ``lse**2`` added into per-row loss (z-loss stabilizer).
        label_smoothing (constexpr float): Label smoothing amount in ``[0, 1)``.
        reduction (constexpr str): ``"mean"`` scales by stats from ``ce_stats_ptr``; ``"sum"`` / ``"none"`` use scale ``1``.
        softcap (float): If ``HAS_SOFTCAPPING``, logits are ``softcap * tanh(x / softcap)``.
        RETURN_Z_LOSS (constexpr bool): Write z-loss rows to ``z_loss_ptr``.
        RETURN_LSE (constexpr bool): Write per-row LSE to ``lse_ptr``.
        RETURN_TOKEN_ACCURACY (constexpr bool): Write per-row correctness to ``token_accuracy_ptr``.
        RETURN_PREDICTED_TOKENS (constexpr bool): Write per-row argmax to ``predicted_tokens_ptr``.
        BLOCK_SIZE (constexpr int): Block size along the vocabulary dimension.
        HAS_WEIGHT (constexpr bool): Load ``weight_ptr[target]`` and per-class weights for smoothing when true.
        HAS_SOFTCAPPING (constexpr bool): Apply tanh soft-capping with ``softcap`` when true.

    Returns:
        None. Effects are writes to ``loss_ptr`` and optional output pointers listed above.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Contiguous row ranges per program improve MTE3 store locality and reduce
    # scalar address arithmetic versus interleaving (pid, pid+num_progs, ...).
    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)
    n_local = row_end - row_start

    inv_n_scale = tl.load(ce_stats_ptr + 0)
    inv_sum_weight_scale = tl.load(ce_stats_ptr + 1)
    weight_sum = tl.load(ce_stats_ptr + 2)

    # One int64 row base per program; advance row pointers with small integer adds
    # instead of row_idx.to(int64) * stride each iteration.
    row_start_int64 = row_start.to(tl.int64)
    target_row_ptr = Y_ptr + row_start_int64
    logits_row_ptr = X_ptr + row_start_int64 * X_stride
    loss_row_ptr = loss_ptr + row_start_int64
    lse_row_ptr = lse_ptr + row_start_int64
    if RETURN_Z_LOSS:
        z_loss_row_ptr = z_loss_ptr + row_start_int64
    if RETURN_TOKEN_ACCURACY:
        token_accuracy_row_ptr = token_accuracy_ptr + row_start_int64 * token_accuracy_stride
    if RETURN_PREDICTED_TOKENS:
        predicted_tokens_row_ptr = predicted_tokens_ptr + row_start_int64 * predicted_tokens_stride

    for _ in range(n_local):
        y = tl.load(target_row_ptr)

        if y == ignore_index:
            if RETURN_LSE:
                tl.store(lse_row_ptr, 0.0)
            if RETURN_TOKEN_ACCURACY:
                tl.store(token_accuracy_row_ptr, 0.0)
            if RETURN_PREDICTED_TOKENS:
                tl.store(predicted_tokens_row_ptr, -1)
        else:
            if HAS_WEIGHT:
                weight_y = tl.load(weight_ptr + y).cast(tl.float32)

            m = float("-inf")
            d = 0.0
            argmax_idx = tl.full((), 0, dtype=tl.int64)
            ori_X_y = tl.load(logits_row_ptr + y).cast(tl.float32)
            if HAS_SOFTCAPPING:
                ori_X_y = softcap * tanh(ori_X_y / softcap)

            scaled_x_sum = 0.0

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    logits_row_ptr + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                    eviction_policy="evict_first",
                ).cast(tl.float32)
                if HAS_SOFTCAPPING:
                    X_block = softcap * tanh(X_block / softcap)
                block_max = tl.max(X_block)

                if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
                    is_max_mask = X_block == block_max
                    masked_offsets = X_offsets + (n_cols - X_offsets) * (1 - is_max_mask.to(tl.int64))
                    current_block_argmax_idx = tl.min(masked_offsets)
                    is_new_max = block_max > m
                    argmax_idx = argmax_idx + is_new_max.to(tl.int64) * (current_block_argmax_idx - argmax_idx)

                if label_smoothing > 0:
                    # Mask logits before multiplying by weight so padding slots (-inf/logits mask)
                    # do not produce NaNs (0 * (-inf)); out-of-range weights are already 0 from load.
                    if HAS_WEIGHT:
                        weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols, other=0.0)
                        scaled_x_sum += tl.sum(-ls_eps * X_block * weight_block)
                    else:
                        scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -ls_eps * X_block, 0.0))

                m_new = tl.maximum(m, block_max)
                d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
                m = m_new

            lse = m + tl.log(d)
            if RETURN_LSE:
                tl.store(lse_row_ptr, lse)
            loss = lse - ori_X_y
            if HAS_WEIGHT:
                loss = weight_y * loss

            if label_smoothing > 0:
                if HAS_WEIGHT:
                    smooth_loss = scaled_x_sum + ls_eps * lse * weight_sum
                else:
                    smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss

            z_loss = lse_square_scale * lse * lse
            if reduction == "mean":
                if HAS_WEIGHT:
                    loss = loss * inv_sum_weight_scale
                else:
                    loss = loss * inv_n_scale
                z_loss = z_loss * inv_n_scale
            loss += z_loss

            tl.store(loss_row_ptr, loss)
            if RETURN_Z_LOSS:
                tl.store(z_loss_row_ptr, z_loss)
            if RETURN_TOKEN_ACCURACY:
                tl.store(token_accuracy_row_ptr, (argmax_idx == y).to(tl.float32))
            if RETURN_PREDICTED_TOKENS:
                tl.store(predicted_tokens_row_ptr, argmax_idx)

        target_row_ptr = target_row_ptr + 1
        logits_row_ptr = logits_row_ptr + X_stride
        loss_row_ptr = loss_row_ptr + 1
        if RETURN_LSE:
            lse_row_ptr = lse_row_ptr + 1
        if RETURN_Z_LOSS:
            z_loss_row_ptr = z_loss_row_ptr + 1
        if RETURN_TOKEN_ACCURACY:
            token_accuracy_row_ptr = token_accuracy_row_ptr + token_accuracy_stride
        if RETURN_PREDICTED_TOKENS:
            predicted_tokens_row_ptr = predicted_tokens_row_ptr + predicted_tokens_stride


@triton.jit
def liger_cross_entropy_backward_kernel_no_weight(
    X_ptr,
    X_stride,
    Y_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_output_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_rows,
    ce_stats_ptr,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD_OUTPUT_VECTOR: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    """
    Specialized backward kernel for the common path without class weights, softcap, z-loss, or label smoothing.
    Optimized for Ascend NPU memory bandwidth utilization.

    Parameters:
        X_ptr: Logits pointer; shape ``(n_rows, n_cols)``, row-major with stride ``X_stride``.
        X_stride (int64): Stride between consecutive logits rows.
        Y_ptr: Target class index per row; ignored rows receive zero ``dX``.
        lse_ptr: Per-row LSE (fp32) when ``HAS_LSE``; otherwise aliases per-row loss buffer for LSE reconstruction.
        grad_output_ptr: Scalar loss gradient or per-row vector (see ``HAS_GRAD_OUTPUT_VECTOR``).
        grad_output_stride (int64): Stride for vector ``grad_output`` (``0`` when scalar).
        dX_ptr: Output logits gradient; same logical layout as ``X_ptr`` with stride ``dX_stride``.
        dX_stride (int64): Stride between ``dX`` rows.
        n_cols (int): Vocabulary size.
        n_rows (int): Number of rows (flattened batch * time).
        ce_stats_ptr: Stats vector (at least ``inv_n_scale`` at index 0 for mean scaling).
        ignore_index (int): Label value skipped for gradient (zero row).
        reduction (constexpr str): ``"mean"`` vs ``sum``/``none`` affects LSE reconstruction when ``not HAS_LSE``.
        BLOCK_SIZE (constexpr int): Tile size along vocabulary.
        HAS_GRAD_OUTPUT_VECTOR (constexpr bool): ``True`` if ``grad_output`` is per-row (``reduction="none"`` style).
        HAS_LSE (constexpr bool): ``True`` if ``lse_ptr`` holds forward LSE; ``False`` to reconstruct from loss + ``x[y]``.

    When ``HAS_LSE`` is True, ``lse_ptr`` holds per-row log-sum-exp from forward.
    When ``HAS_LSE`` is False, ``lse_ptr`` aliases the per-row loss buffer; LSE is reconstructed as
    ``loss_row + x_y`` (``reduction`` not ``mean``/mean-over-batch semantics matching stored rows) or
    ``loss_row / inv_n + x_y`` (``reduction`` ``mean``), avoiding a dedicated fp32 LSE tensor.

    Returns:
        None. Writes softmax-derived gradients into ``dX_ptr``.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    scale_factor = tl.load(ce_stats_ptr + 0)

    for row_idx in range(row_start, row_end):
        program_id = row_idx.to(tl.int64)
        y = tl.load(Y_ptr + program_id)
        X_ptr_offset = program_id * X_stride
        dX_ptr_offset = program_id * dX_stride

        grad_scale = (
            tl.load(grad_output_ptr + program_id * grad_output_stride)
            if HAS_GRAD_OUTPUT_VECTOR
            else tl.load(grad_output_ptr)
        ).cast(tl.float32)

        final_scale = grad_scale * scale_factor

        if y == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, 0.0, mask=X_offsets < n_cols)
        else:
            if HAS_LSE:
                lse = tl.load(lse_ptr + program_id).cast(tl.float32)
            else:
                loss_row = tl.load(lse_ptr + program_id).cast(tl.float32)
                x_y = tl.load(X_ptr + X_ptr_offset + y).cast(tl.float32)
                if reduction == "mean":
                    inv_n = tl.load(ce_stats_ptr + 0).cast(tl.float32)
                    lse = loss_row / inv_n + x_y
                else:
                    # Per-row loss was stored without mean scaling (``reduction`` ``sum`` or ``none``).
                    lse = loss_row + x_y

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                    eviction_policy="evict_first",
                ).cast(tl.float32)
                grad = tl.exp(X_block - lse) * final_scale
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, grad, mask=X_offsets < n_cols)

            target_ptr = dX_ptr + dX_ptr_offset + y
            target_grad = tl.load(target_ptr).cast(tl.float32)
            tl.store(target_ptr, target_grad - final_scale)


@triton.jit
def liger_cross_entropy_backward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    weight_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_output_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_rows,
    ce_stats_ptr,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRAD_OUTPUT_VECTOR: tl.constexpr,
):
    """
    General backward kernel: gradients of cross-entropy w.r.t. logits (optional weight, softcap, smoothing, z-loss).

    Parameters:
        X_ptr: Logits pointer; shape ``(n_rows, n_cols)``, row-major with stride ``X_stride``.
        X_stride (int64): Stride between logits rows.
        Y_ptr: Target class index per row.
        weight_ptr: Per-class weights ``[n_cols]`` when ``HAS_WEIGHT``; unused otherwise.
        lse_ptr: Per-row log-sum-exp from forward (fp32).
        grad_output_ptr: Upstream gradient scalar or per-row vector.
        grad_output_stride (int64): Stride for vector ``grad_output`` (``0`` when scalar).
        dX_ptr: Output gradient buffer with stride ``dX_stride``.
        dX_stride (int64): Stride between ``dX`` rows.
        n_cols (int): Vocabulary size.
        n_rows (int): Number of rows.
        ce_stats_ptr: ``[inv_n_non_ignore, inv_sum_non_ignore_weight, weight_sum]`` for reduction and smoothing.
        ignore_index (int): Label value for skipped rows (zero gradient).
        lse_square_scale (constexpr float): Forward z-loss coefficient (affects ``d`` loss / ``d`` LSE chain).
        label_smoothing (constexpr float): Label smoothing amount.
        reduction (constexpr str): ``"mean"`` or ``"sum"`` / ``"none"`` scaling for gradient contribution.
        softcap (float): Tanh soft-cap scale when ``HAS_SOFTCAPPING``.
        BLOCK_SIZE (constexpr int): Tile size along vocabulary.
        HAS_WEIGHT (constexpr bool): Use ``weight_ptr`` in gradient expression.
        HAS_SOFTCAPPING (constexpr bool): Apply softcap derivative chain when true.
        HAS_GRAD_OUTPUT_VECTOR (constexpr bool): Broadcast vs per-row ``grad_output``.

    Returns:
        None. Writes into ``dX_ptr``.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    row_chunk = (n_rows + num_progs - 1) // num_progs
    row_start = pid * row_chunk
    row_end = tl.minimum(row_start + row_chunk, n_rows)

    inv_n_non_ignore = tl.load(ce_stats_ptr + 0)
    inv_sum_non_ignore_weight = tl.load(ce_stats_ptr + 1)
    weight_sum = tl.load(ce_stats_ptr + 2)

    for row_idx in range(row_start, row_end):
        program_id = row_idx.to(tl.int64)
        y = tl.load(Y_ptr + program_id)
        X_ptr_offset = program_id * X_stride
        dX_ptr_offset = program_id * dX_stride
        grad_scale = (
            tl.load(grad_output_ptr + program_id * grad_output_stride)
            if HAS_GRAD_OUTPUT_VECTOR
            else tl.load(grad_output_ptr)
        ).cast(tl.float32)

        if y == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, 0.0, mask=X_offsets < n_cols)
        else:
            if HAS_WEIGHT:
                weight_y = tl.load(weight_ptr + y).cast(tl.float32)
            lse = tl.load(lse_ptr + program_id).cast(tl.float32)
            eps = label_smoothing / n_cols
            eps_weight_sum = eps * weight_sum
            z_scale = 1.0 + 2.0 * lse_square_scale * lse
            one_minus_ls = 1.0 - label_smoothing
            z_deriv = 2.0 * lse_square_scale * lse

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                    eviction_policy="evict_first",
                ).cast(tl.float32)
                if HAS_SOFTCAPPING:
                    intermediate = tanh(X_block / softcap)
                    X_block = softcap * intermediate

                softmax_X = tl.exp(X_block - lse)
                if not HAS_WEIGHT:
                    X_block = softmax_X * z_scale - eps
                    if y >= i and y < i + BLOCK_SIZE:
                        y_mask = (X_offsets == y).to(tl.float32)
                        X_block = X_block - y_mask * one_minus_ls
                    if reduction == "mean":
                        X_block = X_block * inv_n_non_ignore
                else:
                    weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                    dloss_ori = one_minus_ls * softmax_X
                    if y >= i and y < i + BLOCK_SIZE:
                        y_mask = (X_offsets == y).to(tl.float32)
                        dloss_ori = dloss_ori - y_mask * one_minus_ls
                    dloss_ori = dloss_ori * weight_y
                    dloss_smooth = -eps * weight_block + softmax_X * eps_weight_sum
                    dz_loss = z_deriv * softmax_X
                    if reduction == "mean":
                        dloss_ori = dloss_ori * inv_sum_non_ignore_weight
                        dloss_smooth = dloss_smooth * inv_sum_non_ignore_weight
                        dz_loss = dz_loss * inv_n_non_ignore
                    X_block = dloss_ori + dloss_smooth + dz_loss

                if HAS_SOFTCAPPING:
                    X_block = X_block * (1 - intermediate * intermediate)

                X_block = X_block * grad_scale
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, X_block, mask=X_offsets < n_cols)


def get_optimal_block_size(n_cols, has_gradients=True):
    """
    Pick Triton block size along the vocabulary dimension for Ascend.

    Uses fixed thresholds when ``has_gradients`` is True; otherwise falls back to
    ``compute_default_tiling_strategy`` with an NPU-oriented memory multiplier.

    Args:
        n_cols (int): Vocabulary size (number of columns).
        has_gradients (bool): If True, use the fast heuristic table for backward-style kernels;
            if False, query tiling strategy for forward-only sizing.

    Returns:
        int: Block size (positive). Defaults to 4096 when tiling yields nothing.
    """
    if has_gradients:
        if n_cols <= 32768:
            return 1024
        if n_cols <= 131072:
            return 2048
        return 4096

    multiplier = 12.0 if has_gradients else 8.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((n_cols,),), tiling_dims=(0,)
    )
    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return 4096


def get_no_weight_fast_path_block_size(n_cols):
    """
    Block size for the no-weight backward fast path kernel.

    Args:
        n_cols (int): Vocabulary size.

    Returns:
        int: ``2048`` when ``n_cols <= 4096``, otherwise ``get_optimal_block_size(n_cols, True)``.
    """
    if n_cols <= 4096:
        return 2048
    return get_optimal_block_size(n_cols, has_gradients=True)


def _make_ce_stats_buffer(
    target: torch.Tensor,
    ignore_index: int,
    weight: Optional[torch.Tensor],
    reduction: str,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build reduction/smoothing statistics on device for the Triton kernels.

    Args:
        target (torch.Tensor): Class indices of shape ``(n_rows,)`` (flattened batch * time).
        ignore_index (int): Label value treated as padding / ignored.
        weight (torch.Tensor, optional): Per-class weights of shape ``(V,)``; ``None`` for uniform.
        reduction (str): ``"mean"``, ``"sum"``, or ``"none"`` (stats still encode scaling hooks used in-kernel).
        target_mask (torch.Tensor, optional): Boolean mask of valid targets; default ``target != ignore_index``.

    Returns:
        torch.Tensor: Float32 vector of shape ``[3]`` on ``target.device``:
        ``[inv_n_scale, inv_sum_weight_scale, weight_sum]`` (last two entries meaningful when weighting is used).
        Built without ``.item()`` so launch avoids host sync on NPU.
    """
    device = target.device
    dtype = torch.float32
    if target_mask is None:
        target_mask = target != ignore_index
    sum_n = target_mask.sum(dtype=dtype)

    if reduction == "mean":
        inv_n = sum_n.clamp(min=1.0).reciprocal_()
    else:
        inv_n = torch.ones((), dtype=dtype, device=device)

    if weight is not None:
        non_ignore_targets = target.masked_select(target_mask)
        sum_w = torch.gather(weight, dim=0, index=non_ignore_targets).sum(dtype=dtype)
        w_sum = weight.sum(dtype=dtype)
        if reduction == "mean":
            inv_sum_w = sum_w.clamp(min=1e-12).reciprocal_()
        else:
            inv_sum_w = torch.ones((), dtype=dtype, device=device)
    else:
        inv_sum_w = torch.ones((), dtype=dtype, device=device)
        w_sum = torch.zeros((), dtype=dtype, device=device)

    return torch.stack((inv_n, inv_sum_w, w_sum))


def _forward_returns_fp32_lse_rows(
    input_requires_grad: bool,
    weight: Optional[torch.Tensor],
    label_smoothing: float,
    softcap: Optional[float],
    lse_square_scale: float,
    input_dtype: torch.dtype,
) -> bool:
    """
    Whether forward should allocate fp32 per-row LSE and have the kernel write them for backward.

    Args:
        input_requires_grad: If False, backward is not needed; LSE buffer for backward is unnecessary.
        weight: Optional per-class weights.
        label_smoothing: Label smoothing amount.
        softcap: Optional tanh soft-cap value.
        lse_square_scale: Z-loss coefficient on ``lse**2``.
        input_dtype: Logits element dtype.

    Returns:
        bool: ``True`` if ``input_requires_grad`` and forward must allocate and fill fp32 per-row LSE.
        Returns ``False`` on the no-weight fast path where backward can recover LSE from per-row loss
        and ``X[target]`` (requires ``weight is None``, ``label_smoothing == 0.0``, ``softcap is None``,
        ``lse_square_scale == 0.0``, and ``input_dtype == torch.float32``).
    """
    return input_requires_grad and not (
        weight is None
        and label_smoothing == 0.0
        and softcap is None
        and lse_square_scale == 0.0
        and input_dtype == torch.float32
    )


def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    return_lse=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = get_optimal_block_size(V, has_gradients=False)

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    # Triton requires a tensor pointer; when ``return_lse`` is False the kernel never reads/writes LSE
    # (``RETURN_LSE`` false), so we pass ``loss_1d`` as the unused binding (``None`` is not supported).
    lse_buffer_fp32 = torch.empty(n_rows, dtype=torch.float32, device=_input.device) if return_lse else None
    lse_ptr_for_kernel = lse_buffer_fp32 if return_lse else loss_1d
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    )
    predicted_tokens_1d = (
        torch.full((n_rows,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None
    )

    target_mask = target != ignore_index
    invalid_target_mask = target_mask & ((target < 0) | (target >= V))
    assert not torch.any(invalid_target_mask), (
        f"Target tensor contains out of bounds values. Expected targets in [0, {V}) or ignore_index={ignore_index}"
    )

    ce_stats = _make_ce_stats_buffer(target, ignore_index, weight, reduction, target_mask=target_mask)
    if weight is not None:
        # ensure weight is contiguous
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # NPU-optimized grid configuration
    # grid_size = min(get_npu_core_count(), n_rows)

    cores = min(get_npu_core_count(), n_rows)
    plain_lm = (
        weight is None
        and label_smoothing == 0.0
        and softcap is None
        and float(lse_square_scale) == 0.0
        and not return_z_loss
        and not return_lse
        and not return_token_accuracy
        and not return_predicted_tokens
    )
    if plain_lm:
        ts = compute_default_tiling_strategy(
            safety_margin=0.9,
            dtype_size=4,
            memory_multiplier=2.5,
            shapes=((V,),),
            tiling_dims=(0,),
        )
        BLOCK_SIZE = max(256, ts[0][0]) if ts else 8192
        grid_size = n_rows if n_rows <= 1024 else cores
    else:
        BLOCK_SIZE = get_optimal_block_size(V, has_gradients=False)
        grid_size = cores

    ls_eps = float(label_smoothing) / float(V) if label_smoothing else 0.0
    liger_cross_entropy_forward_kernel[(grid_size,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        weight_ptr=weight,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        lse_ptr=lse_ptr_for_kernel,
        token_accuracy_ptr=token_accuracy_1d,
        token_accuracy_stride=token_accuracy_1d.stride(-1)
        if return_token_accuracy
        else 0,  # always 1 if accuracy is enabled
        predicted_tokens_ptr=predicted_tokens_1d,
        predicted_tokens_stride=predicted_tokens_1d.stride(-1)
        if return_predicted_tokens
        else 0,  # always 1 if predicted tokens is enabled
        n_cols=V,
        n_rows=n_rows,
        ce_stats_ptr=ce_stats,
        ignore_index=ignore_index,
        ls_eps=ls_eps,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        RETURN_LSE=return_lse,
        RETURN_TOKEN_ACCURACY=return_token_accuracy,
        RETURN_PREDICTED_TOKENS=return_predicted_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # For accuracy, we compute the mean across all non-ignored tokens
        token_accuracy = (
            torch.sum(token_accuracy_1d) / target_mask.sum(dtype=torch.float32).clamp(min=1.0)
            if return_token_accuracy
            else None
        )

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return (
        loss,
        z_loss,
        token_accuracy,
        predicted_tokens,
        _input,
        lse_buffer_fp32 if return_lse else None,
        ce_stats,
        loss_1d,
    )


def cross_entropy_backward(
    _input,
    target,
    weight,
    lse,
    grad_output,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    ce_stats=None,
    derive_lse_from_loss: bool = False,
):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = get_optimal_block_size(V, has_gradients=True)

    if ce_stats is None:
        ce_stats = _make_ce_stats_buffer(target, ignore_index, weight, reduction)
    if weight is not None and weight.stride(-1) != 1:
        weight = weight.contiguous()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()
    if grad_output.ndim > 0 and grad_output.stride(-1) != 1:
        grad_output = grad_output.contiguous()

    grad_input = torch.empty_like(_input)
    grid_size = min(get_npu_core_count(), n_rows)
    grad_output_stride = grad_output.stride(-1) if grad_output.ndim > 0 else 0

    use_no_weight_fast_path = weight is None and softcap is None and label_smoothing == 0.0 and lse_square_scale == 0.0

    if use_no_weight_fast_path:
        fast_path_block_size = get_no_weight_fast_path_block_size(V)
        liger_cross_entropy_backward_kernel_no_weight[(grid_size,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            lse_ptr=lse,
            grad_output_ptr=grad_output,
            grad_output_stride=grad_output_stride,
            dX_ptr=grad_input,
            dX_stride=grad_input.stride(-2),
            n_cols=V,
            n_rows=n_rows,
            ce_stats_ptr=ce_stats,
            ignore_index=ignore_index,
            reduction=reduction,
            BLOCK_SIZE=fast_path_block_size,
            # Gradients w.r.t. per-row loss when ``grad_output`` is 1-D (typically ``reduction="none"``).
            HAS_GRAD_OUTPUT_VECTOR=grad_output.ndim > 0,
            HAS_LSE=not derive_lse_from_loss,
        )
    else:
        liger_cross_entropy_backward_kernel[(grid_size,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            weight_ptr=weight,
            lse_ptr=lse,
            grad_output_ptr=grad_output,
            grad_output_stride=grad_output_stride,
            dX_ptr=grad_input,
            dX_stride=grad_input.stride(-2),
            n_cols=V,
            n_rows=n_rows,
            ce_stats_ptr=ce_stats,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=True if weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            # Same semantics as the no-weight backward kernel above.
            HAS_GRAD_OUTPUT_VECTOR=grad_output.ndim > 0,
        )

    return grad_input


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger Cross Entropy loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        weight(Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index (int): The index to ignore in the target.
        lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str): The reduction to apply to the output: "none" | "mean | "sum".
        softcap (Optional[float]): The upper threshold for scaling logits to the range (-softcap, +softcap).
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss, token_accuracy, predicted_tokens) instead of (loss, None, None, None). Default: `False`
        return_token_accuracy (bool): When `return_token_accuracy` is `True`, computes and returns per-token accuracy without materializing logits. Default: `False`
        return_predicted_tokens (bool): When `return_predicted_tokens` is `True`, returns per-token predicted class indices (argmax) without materializing logits. Default: `False`

        Returns:
        tuple: A tuple with the computed losses, accuracy, and predicted tokens: (loss, z_loss, token_accuracy, predicted_tokens). z_loss, token_accuracy, and predicted_tokens are None if not requested.
        """
        input_requires_grad = _input.requires_grad
        return_lse = _forward_returns_fp32_lse_rows(
            input_requires_grad,
            weight,
            label_smoothing,
            softcap,
            lse_square_scale,
            _input.dtype,
        )

        (
            loss,
            z_loss,
            token_accuracy,
            predicted_tokens,
            _input,
            lse,
            ce_stats,
            loss_1d,
        ) = cross_entropy_forward(
            _input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            return_lse=return_lse,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )
        if input_requires_grad:
            if not return_lse:
                saved_tensors = [_input.detach(), target.detach(), loss_1d]
            else:
                saved_tensors = [_input.detach(), target.detach(), lse.detach()]
            if weight is not None:
                saved_tensors.append(weight.detach())
            ctx.save_for_backward(*saved_tensors)
        ctx.derive_lse_from_loss = bool(input_requires_grad and not return_lse)
        ctx.ce_stats = ce_stats
        ctx.has_weight = weight is not None
        ctx.ignore_index = ignore_index
        ctx.lse_square_scale = lse_square_scale
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.softcap = softcap
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tensor): No use. Gradient for z_loss (not used as z_loss is only for logging).
        grad_output3 (tensor): No use. Gradient for token_accuracy (not used as token_accuracy is only for metrics).
        grad_output4 (tensor): No use. Gradient for predicted_tokens (not used as predicted_tokens is only for metrics).
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics

        if ctx.has_weight:
            _input, target, lse_or_loss, weight = ctx.saved_tensors
        else:
            _input, target, lse_or_loss = ctx.saved_tensors
            weight = None
        _input = cross_entropy_backward(
            _input,
            target,
            weight,
            lse_or_loss,
            grad_output,
            ctx.ignore_index,
            ctx.lse_square_scale,
            ctx.label_smoothing,
            ctx.reduction,
            ctx.softcap,
            ctx.ce_stats,
            derive_lse_from_loss=ctx.derive_lse_from_loss,
        )
        return (
            _input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
