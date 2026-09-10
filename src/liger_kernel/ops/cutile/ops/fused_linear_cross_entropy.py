# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile fused linear cross entropy for Hopper (SM90) and Blackwell (SM100).

The projection and input-gradient GEMMs use PyTorch's tuned BLAS path. cuTile
computes partitioned CE statistics, overwrites a reusable ``[min(N, C), V]``
logits buffer with dZ, and accumulates dW with an FP32-accumulating MMA schedule.
Tokens are processed in chunks so the transient logits workspace is
``O(min(N, C) * V)`` instead of the full ``O(N * V)`` materialization; the
retained gradients (raw BF16 dX and the FP32 dW accumulator) add ``O(N*H + V*H)``.

The upstream/mean scale is applied out-of-place in backward: the FP32 dW
accumulator is retained until backward and cast to the weight dtype exactly once,
after multiplying by the combined ``upstream * mean_normalizer`` factor. Casting
after (not before) that multiply preserves the numeric range of small gradients
that would otherwise flush to zero in BF16.

Two low-level interfaces are exported. ``fused_linear_cross_entropy_forward`` /
``fused_linear_cross_entropy_backward`` retain the original unchunked contract
(the forward returns the full ``[N, V]`` dZ and the 7-argument backward runs the
dW matmul). The token-chunked path used by the autograd Function lives in
``chunked_fused_linear_cross_entropy_forward`` /
``chunked_fused_linear_cross_entropy_backward``.
"""

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import validate_flce_chunk_size

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]

LOG2E = 1.4426950408889634
DW_TILE_M = 128
DW_TILE_N = 128
DW_TILE_K = 64
SCALE_CAST_TILE_M = 128
SCALE_CAST_TILE_N = 128
CE_BLOCK_SIZE = 2048
LOGITS_STATS_BLOCK_SIZE = 4096
MAX_STATS_BLOCK_SIZE = 1024


@ct.kernel(num_worker_warps=ct.ByTarget(sm_90=8, default=8), opt_level=3)
def _matmul_tn_kernel(
    a,
    b,
    output,
    scale,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    ACCUMULATE: ConstBool,
):
    """Compute ``output (+)= (a.T @ b) * scale`` into an FP32 accumulator tile.

    When ``ACCUMULATE`` is ``False`` the tile overwrites ``output`` (first token
    chunk); otherwise the freshly computed contribution is added to the value
    already present in the FP32 ``output`` buffer (subsequent chunks).
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    a_view = a.tiled_view((TILE_K, TILE_M), padding_mode=ct.PaddingMode.ZERO)
    b_view = b.tiled_view((TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    output_view = output.tiled_view((TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    for pid_k in range(a_view.num_tiles(0)):
        a_tile = a_view.load((pid_k, pid_m), latency=10)
        b_tile = b_view.load((pid_k, pid_n), latency=10)
        acc = ct.mma(ct.transpose(a_tile), b_tile, acc)

    tile_scale = ct.astype(ct.gather(scale, (), check_bounds=False), ct.float32)
    result = acc * tile_scale
    if ACCUMULATE:
        result = result + ct.astype(output_view.load((pid_m, pid_n), latency=2), ct.float32)
    output_view.store((pid_m, pid_n), ct.astype(result, output.dtype), latency=1)


@ct.kernel(num_worker_warps=ct.ByTarget(sm_90=8, default=8), opt_level=3)
def _matmul_tn_n4_kernel(
    a,
    b,
    output,
    scale,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    LOAD_LATENCY: ConstInt,
    ACCUMULATE: ConstBool,
):
    """Compute four adjacent dW column tiles while sharing each dZ load.

    Accumulates into the FP32 ``output`` buffer: overwrite on the first token
    chunk (``ACCUMULATE`` is ``False``), add on subsequent chunks.
    """
    pid_n = ct.bid(0) * 4
    pid_m = ct.bid(1)
    a_view = a.tiled_view((TILE_K, TILE_M), padding_mode=ct.PaddingMode.ZERO)
    b_view = b.tiled_view((TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    output_view = output.tiled_view((TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    acc0 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc1 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc2 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    acc3 = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    for pid_k in range(a_view.num_tiles(0)):
        a_tile = ct.transpose(a_view.load((pid_k, pid_m), latency=LOAD_LATENCY))
        acc0 = ct.mma(a_tile, b_view.load((pid_k, pid_n), latency=LOAD_LATENCY), acc0)
        acc1 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 1), latency=LOAD_LATENCY), acc1)
        acc2 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 2), latency=LOAD_LATENCY), acc2)
        acc3 = ct.mma(a_tile, b_view.load((pid_k, pid_n + 3), latency=LOAD_LATENCY), acc3)

    tile_scale = ct.astype(ct.gather(scale, (), check_bounds=False), ct.float32)
    result0 = acc0 * tile_scale
    result1 = acc1 * tile_scale
    result2 = acc2 * tile_scale
    result3 = acc3 * tile_scale
    if ACCUMULATE:
        result0 = result0 + ct.astype(output_view.load((pid_m, pid_n), latency=2), ct.float32)
        result1 = result1 + ct.astype(output_view.load((pid_m, pid_n + 1), latency=2), ct.float32)
        result2 = result2 + ct.astype(output_view.load((pid_m, pid_n + 2), latency=2), ct.float32)
        result3 = result3 + ct.astype(output_view.load((pid_m, pid_n + 3), latency=2), ct.float32)
    output_view.store((pid_m, pid_n), ct.astype(result0, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 1), ct.astype(result1, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 2), ct.astype(result2, output.dtype), latency=1)
    output_view.store((pid_m, pid_n + 3), ct.astype(result3, output.dtype), latency=1)


@ct.kernel(num_worker_warps=ct.ByTarget(sm_90=8, default=8), opt_level=3)
def _scale_cast_kernel(
    src,
    dst,
    scale,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
):
    """Scale an FP32 ``src`` tile by a device scalar and cast into ``dst.dtype``.

    Used to turn the retained FP32 dW accumulator into the final weight-dtype
    gradient without allocating an O(V*H) FP32 temporary: the multiply by the
    combined ``upstream * mean_normalizer`` scalar happens in FP32 and the BF16
    cast is applied exactly once, per element, on store.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    src_view = src.tiled_view((TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    dst_view = dst.tiled_view((TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    tile = ct.astype(src_view.load((pid_m, pid_n), latency=2), ct.float32)
    tile_scale = ct.astype(ct.gather(scale, (), check_bounds=False), ct.float32)
    dst_view.store((pid_m, pid_n), ct.astype(tile * tile_scale, dst.dtype), latency=1)


@ct.kernel(occupancy=8)
def _fused_cross_entropy_dz_kernel(
    logits,
    target,
    loss,
    loss_scale,
    partial_max,
    partial_sum,
    completion_count,
    vocab_size,
    num_partitions,
    ignore_index,
    STATS_BLOCK_SIZE: ConstInt,
    CE_BLOCK_SIZE: ConstInt,
    PARTIALS_BLOCK_SIZE: ConstInt,
    HAS_GRADIENTS: ConstBool,
    REDUCTION_MEAN: ConstBool,
):
    program = ct.bid(0)
    row = program // num_partitions
    partition = program % num_partitions
    label = ct.load(target, row, shape=())
    stat_cols = ct.arange(STATS_BLOCK_SIZE, dtype=ct.int32) + partition * STATS_BLOCK_SIZE
    values = ct.astype(
        ct.gather(logits, (row, stat_cols), check_bounds=True, padding_value=-float("inf"), latency=4),
        ct.float32,
    )
    block_max = ct.max(values, 0, keepdims=False)
    block_sum = ct.sum(ct.exp2((values - block_max) * LOG2E, flush_to_zero=True), 0, keepdims=False)
    ct.scatter(partial_max, (row, partition), block_max, check_bounds=False)
    ct.scatter(partial_sum, (row, partition), block_sum, check_bounds=False)

    completed = ct.atomic_add(
        completion_count,
        row,
        1,
        check_bounds=False,
        memory_order=ct.MemoryOrder.ACQ_REL,
        memory_scope=ct.MemoryScope.DEVICE,
    )
    if completed != num_partitions - 1:
        return

    if label == ignore_index:
        if HAS_GRADIENTS:
            for chunk in range((vocab_size + CE_BLOCK_SIZE - 1) // CE_BLOCK_SIZE):
                cols = ct.arange(CE_BLOCK_SIZE, dtype=ct.int32) + chunk * CE_BLOCK_SIZE
                ct.scatter(logits, (row, cols), ct.zeros((CE_BLOCK_SIZE,), dtype=logits.dtype), check_bounds=True)
        ct.scatter(loss, row, ct.astype(0.0, loss.dtype))
        return

    partial_cols = ct.arange(PARTIALS_BLOCK_SIZE, dtype=ct.int32)
    tile_max = ct.astype(
        ct.gather(partial_max, (row, partial_cols), check_bounds=True, padding_value=-float("inf"), latency=2),
        ct.float32,
    )
    tile_sum = ct.astype(
        ct.gather(partial_sum, (row, partial_cols), check_bounds=True, padding_value=0.0, latency=2),
        ct.float32,
    )
    valid_stats = ct.less(partial_cols, num_partitions)
    tile_max = ct.where(valid_stats, tile_max, -float("inf"))
    tile_sum = ct.where(valid_stats, tile_sum, 0.0)

    max_value = ct.max(tile_max, 0, keepdims=False)
    exp_sum = ct.sum(tile_sum * ct.exp2((tile_max - max_value) * LOG2E, flush_to_zero=True), 0, keepdims=False)
    target_logit = ct.astype(ct.gather(logits, (row, label), check_bounds=False), ct.float32)
    row_scale = 1.0
    if REDUCTION_MEAN:
        row_scale = ct.astype(ct.gather(loss_scale, (), check_bounds=False), ct.float32)
    ct.scatter(loss, row, ct.astype((max_value + ct.log(exp_sum) - target_logit) * row_scale, loss.dtype))

    if HAS_GRADIENTS:
        inv_sum = 1.0 / exp_sum
        for chunk in range((vocab_size + CE_BLOCK_SIZE - 1) // CE_BLOCK_SIZE):
            cols = ct.arange(CE_BLOCK_SIZE, dtype=ct.int32) + chunk * CE_BLOCK_SIZE
            logits_tile = ct.astype(
                ct.gather(logits, (row, cols), check_bounds=True, padding_value=-float("inf"), latency=4),
                ct.float32,
            )
            gradient = ct.exp2((logits_tile - max_value) * LOG2E, flush_to_zero=True) * inv_sum
            gradient = ct.where(ct.equal(cols, label), gradient - 1.0, gradient)
            ct.scatter(logits, (row, cols), ct.astype(gradient, logits.dtype), check_bounds=True)


def _launch_matmul_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
    accumulate: bool,
) -> None:
    if output.shape[1] % 256 == 0:
        _launch_cutile(
            output.device,
            (
                output.shape[1] // 256,
                (output.shape[0] + DW_TILE_M - 1) // DW_TILE_M,
                1,
            ),
            _matmul_tn_n4_kernel,
            (a, b, output, scale, DW_TILE_M, 64, DW_TILE_K, 5, accumulate),
        )
        return

    tile_n = min(256, _next_power_of_2(output.shape[1]))
    _launch_cutile(
        output.device,
        (
            (output.shape[0] + DW_TILE_M - 1) // DW_TILE_M,
            (output.shape[1] + tile_n - 1) // tile_n,
            1,
        ),
        _matmul_tn_kernel,
        (a, b, output, scale, DW_TILE_M, tile_n, DW_TILE_K, accumulate),
    )


def _launch_cutile(device: torch.device, grid, kernel, args) -> None:
    with torch.cuda.device(device):
        ct.launch(torch.cuda.current_stream(device), grid, kernel, args)


def _launch_scale_cast(src: torch.Tensor, dst: torch.Tensor, scale: torch.Tensor) -> None:
    """Compute ``dst = (src * scale).astype(dst.dtype)`` without an FP32 temporary."""
    _launch_cutile(
        dst.device,
        (
            (src.shape[0] + SCALE_CAST_TILE_M - 1) // SCALE_CAST_TILE_M,
            (src.shape[1] + SCALE_CAST_TILE_N - 1) // SCALE_CAST_TILE_N,
            1,
        ),
        _scale_cast_kernel,
        (src, dst, scale, SCALE_CAST_TILE_M, SCALE_CAST_TILE_N),
    )


def _reject_unsupported(
    bias,
    ce_weight,
    lse_square_scale,
    label_smoothing,
    softcap,
    return_z_loss,
    accum_dtype,
    use_token_scaling,
    return_token_accuracy,
    return_predicted_tokens,
) -> None:
    unsupported = {
        "bias": bias is not None,
        "class weights": ce_weight is not None,
        "z-loss": bool(lse_square_scale),
        "label smoothing": bool(label_smoothing),
        "softcap": softcap is not None,
        "return_z_loss": return_z_loss,
        "token scaling": use_token_scaling,
        "return_token_accuracy": return_token_accuracy,
        "return_predicted_tokens": return_predicted_tokens,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise NotImplementedError(f"cuTile FLCE does not support: {', '.join(enabled)}")
    # FP32 accumulation is the default and only supported accumulation dtype.
    if accum_dtype is not None and accum_dtype != torch.float32:
        raise NotImplementedError(
            f"cuTile FLCE only supports FP32 accumulation (accum_dtype=torch.float32 or None), got {accum_dtype!r}"
        )


# Impl names that identify this cuTile kernel itself; accepted so callers that
# route by name can target it explicitly. These are self-identity only -- this
# Function does not dispatch to alternative CE implementations.
_SELF_CE_IMPLS = (None, "cutile", "nvidia-cutile")


def _validate_ce_dispatch(ce_impl, ce_mode) -> None:
    if ce_impl not in _SELF_CE_IMPLS:
        raise NotImplementedError(
            f"cuTile FLCE does not dispatch to ce_impl={ce_impl!r}; "
            f"expected one of {tuple(name for name in _SELF_CE_IMPLS)}"
        )
    if ce_mode is not None:
        raise NotImplementedError(f"cuTile FLCE only supports the default CE mode (ce_mode=None), got {ce_mode!r}")


def _largest_power_of_2_le(n: int) -> int:
    """Return the largest power of two less than or equal to ``n`` (>= 1)."""
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _resolve_chunk_size(chunk_size, tokens: int) -> int:
    """Resolve the token-chunk size, clamped to ``tokens``.

    An explicit ``chunk_size`` is validated and clamped by the shared
    ``validate_flce_chunk_size`` helper: it must be a positive Python ``int``
    (not ``bool``), is clamped to ``tokens`` but never rounded, and any invalid
    value raises ``ValueError``. When ``None`` the CuTe-native heuristic
    ``min(N, 1024, largest_pow2 <= max(512, N // 4))`` is used.
    """
    if chunk_size is None:
        return min(tokens, 1024, _largest_power_of_2_le(max(512, tokens // 4)))
    return validate_flce_chunk_size(chunk_size, tokens)


def _validate_inputs(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
) -> None:
    if _input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("cuTile FLCE supports BF16 input and weight only")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M, H], weight[V, H], and target[M]")
    if 0 in _input.shape or 0 in weight.shape or target.shape[0] == 0:
        raise ValueError(
            f"cuTile FLCE does not support empty tensors: input {tuple(_input.shape)}, "
            f"weight {tuple(weight.shape)}, target {tuple(target.shape)}"
        )
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"incompatible input {tuple(_input.shape)}, weight {tuple(weight.shape)}, and target {tuple(target.shape)}"
        )
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    # Reject CPU tensors before touching CUDA device-capability queries.
    if _input.device.type != "cuda" or weight.device.type != "cuda" or target.device.type != "cuda":
        raise RuntimeError("cuTile FLCE requires CUDA (Hopper 9.0 or Blackwell SM100 10.0) tensors")
    if not torch.cuda.is_available():
        raise RuntimeError("cuTile FLCE requires an available CUDA device")
    if not _input.is_contiguous() or not weight.is_contiguous() or not target.is_contiguous():
        raise ValueError("cuTile FLCE requires contiguous input, weight, and target tensors")
    capability = torch.cuda.get_device_capability(_input.device)
    if capability not in ((9, 0), (10, 0)):
        raise RuntimeError(
            f"cuTile FLCE requires a Hopper (compute capability 9.0) or Blackwell SM100 (10.0) GPU, got {capability}"
        )
    num_stats_partitions = (weight.shape[0] + LOGITS_STATS_BLOCK_SIZE - 1) // LOGITS_STATS_BLOCK_SIZE
    if num_stats_partitions > MAX_STATS_BLOCK_SIZE:
        raise NotImplementedError(
            f"cuTile FLCE supports at most {MAX_STATS_BLOCK_SIZE * LOGITS_STATS_BLOCK_SIZE} vocabulary entries"
        )


def _ce_setup(_input, weight, target, reduction, ignore_index):
    """Shared prologue: derive shapes, validate targets, and build the scales.

    Returns the geometry the CE kernel needs plus ``loss_scale`` (passed to the
    kernel for the mean-reduced per-row loss) and ``gradient_scale`` (the global
    mean normalizer, ``1`` for ``sum``) applied to the gradients in backward.
    """
    tokens, hidden_size = _input.shape
    vocab_size = weight.shape[0]
    num_vocab_tiles = (vocab_size + LOGITS_STATS_BLOCK_SIZE - 1) // LOGITS_STATS_BLOCK_SIZE
    stats_block_size = _next_power_of_2(num_vocab_tiles)
    needs_dz = _input.requires_grad or weight.requires_grad
    reduction_mean = reduction == "mean"

    target_mask = target != ignore_index
    valid_target = ~target_mask | ((target >= 0) & (target < vocab_size))
    torch._assert_async(valid_target.all(), f"target values must be in [0, {vocab_size}) or equal ignore_index")
    # CE normalization uses the GLOBAL count of non-ignored tokens across all chunks.
    loss_scale = (
        target_mask.sum().clamp_min(1).to(torch.float32).reciprocal()
        if reduction_mean
        else torch.empty((), dtype=torch.float32, device=_input.device)
    )
    gradient_scale = loss_scale if reduction_mean else torch.ones((), dtype=torch.float32, device=_input.device)
    return (
        tokens,
        hidden_size,
        vocab_size,
        num_vocab_tiles,
        stats_block_size,
        needs_dz,
        reduction_mean,
        loss_scale,
        gradient_scale,
    )


def _project_and_dz(
    input_chunk,
    weight,
    logits_chunk,
    target_chunk,
    loss_chunk,
    loss_scale,
    partial_max,
    partial_sum,
    completion_count,
    vocab_size,
    num_vocab_tiles,
    stats_block_size,
    ignore_index,
    needs_dz,
    reduction_mean,
):
    """Project a token chunk to logits and overwrite it in place with dZ.

    ``torch.mm`` fills ``logits_chunk`` with the BF16 projection; the fused CE
    kernel then writes the per-token loss and (when ``needs_dz``) replaces the
    logits with the softmax-minus-one-hot gradient dZ. Shared by the legacy
    unchunked path (one full-batch call) and the chunked path (per chunk).
    """
    rows = logits_chunk.shape[0]
    torch.mm(input_chunk, weight.t(), out=logits_chunk)
    # Counters must be reset before every partition reduction.
    completion_count.zero_()
    _launch_cutile(
        logits_chunk.device,
        (num_vocab_tiles * rows, 1, 1),
        _fused_cross_entropy_dz_kernel,
        (
            logits_chunk,
            target_chunk,
            loss_chunk,
            loss_scale,
            partial_max,
            partial_sum,
            completion_count,
            vocab_size,
            num_vocab_tiles,
            ignore_index,
            LOGITS_STATS_BLOCK_SIZE,
            CE_BLOCK_SIZE,
            stats_block_size,
            needs_dz,
            reduction_mean,
        ),
    )


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """Legacy unchunked forward.

    Materializes the full ``[N, V]`` logits buffer, overwrites it with dZ, and
    returns that dZ as the third element (``grad_logits``). The paired
    :func:`fused_linear_cross_entropy_backward` consumes ``grad_logits`` to run
    the dW matmul. This is the original low-level contract, retained for
    compatibility; the autograd Function uses the bounded
    :func:`chunked_fused_linear_cross_entropy_forward` instead.
    """
    _reject_unsupported(
        bias,
        ce_weight,
        lse_square_scale,
        label_smoothing,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    _validate_inputs(_input, weight, target, reduction)

    (
        tokens,
        hidden_size,
        vocab_size,
        num_vocab_tiles,
        stats_block_size,
        needs_dz,
        reduction_mean,
        loss_scale,
        gradient_scale,
    ) = _ce_setup(_input, weight, target, reduction, ignore_index)

    logits = torch.empty((tokens, vocab_size), dtype=torch.bfloat16, device=_input.device)
    loss_1d = torch.empty(tokens, dtype=torch.float32, device=_input.device)
    partial_max = torch.empty((tokens, num_vocab_tiles), dtype=torch.float32, device=_input.device)
    partial_sum = torch.empty_like(partial_max)
    completion_count = torch.zeros(tokens, dtype=torch.int32, device=_input.device)

    _project_and_dz(
        _input,
        weight,
        logits,
        target,
        loss_1d,
        loss_scale,
        partial_max,
        partial_sum,
        completion_count,
        vocab_size,
        num_vocab_tiles,
        stats_block_size,
        ignore_index,
        needs_dz,
        reduction_mean,
    )

    grad_input = None
    if _input.requires_grad:
        grad_input = torch.empty((tokens, hidden_size), dtype=_input.dtype, device=_input.device)
        torch.mm(logits, weight, out=grad_input)

    loss = loss_1d.sum()
    # Third element is the full [N, V] dZ (legacy contract): the paired backward
    # runs the dW matmul from it. Scaling is deferred to backward.
    return loss, grad_input, logits if weight.requires_grad else None, gradient_scale


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_logits: Optional[torch.Tensor],
    _input: Optional[torch.Tensor],
    gradient_scale: torch.Tensor,
    weight_shape: torch.Size,
    weight_dtype: torch.dtype,
):
    """Legacy 7-argument backward paired with the unchunked forward.

    Applies ``upstream * gradient_scale`` to dX and runs the dW matmul
    ``dZ.T @ X`` (with the same scale folded in) directly into a ``weight_dtype``
    tensor. All operations are out-of-place, so the saved tensors are not
    mutated across repeated backward calls.
    """
    total_scale = grad_output * gradient_scale
    grad_input_out = None
    if grad_input is not None:
        grad_input_out = grad_input * total_scale

    grad_weight = None
    if grad_logits is not None:
        grad_weight = torch.empty(weight_shape, dtype=weight_dtype, device=grad_logits.device)
        _launch_matmul_tn(grad_logits, _input, grad_weight, total_scale, False)

    return grad_input_out, grad_weight


def chunked_fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
    chunk_size=None,
):
    """Token-chunked forward used by the autograd Function.

    Processes tokens in row chunks so the transient logits/dZ workspace is
    ``O(min(N, chunk_size) * V)``. dX is written as the raw BF16 projection and
    dW is accumulated in an FP32 buffer; NEITHER is scaled here. The returned
    third element is the raw FP32 dW accumulator (``O(V*H)``), which
    :func:`chunked_fused_linear_cross_entropy_backward` scales and casts to the
    weight dtype exactly once -- deferring the BF16 cast preserves small
    gradients that would flush to zero if cast per-chunk in the forward.
    """
    _reject_unsupported(
        bias,
        ce_weight,
        lse_square_scale,
        label_smoothing,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    _validate_inputs(_input, weight, target, reduction)

    (
        tokens,
        hidden_size,
        vocab_size,
        num_vocab_tiles,
        stats_block_size,
        needs_dz,
        reduction_mean,
        loss_scale,
        gradient_scale,
    ) = _ce_setup(_input, weight, target, reduction, ignore_index)

    chunk = _resolve_chunk_size(chunk_size, tokens)
    buffer_rows = min(tokens, chunk)

    # Reusable per-chunk logits/dZ buffer of shape [min(N, C), V] and the
    # partitioned CE statistics/counter scratch of shape [min(N, C), num_partitions].
    logits = torch.empty((buffer_rows, vocab_size), dtype=torch.bfloat16, device=_input.device)
    loss_1d = torch.empty(tokens, dtype=torch.float32, device=_input.device)
    partial_max = torch.empty((buffer_rows, num_vocab_tiles), dtype=torch.float32, device=_input.device)
    partial_sum = torch.empty_like(partial_max)
    completion_count = torch.zeros(buffer_rows, dtype=torch.int32, device=_input.device)

    grad_input = None
    if _input.requires_grad:
        grad_input = torch.empty((tokens, hidden_size), dtype=_input.dtype, device=_input.device)

    # FP32 dW accumulator reused across chunks. It is retained UNSCALED for
    # backward; the mean normalizer and upstream scalar are combined and the
    # single BF16 cast is applied there, never per chunk.
    dweight_accum = None
    unit_scale = None
    if weight.requires_grad:
        dweight_accum = torch.empty((vocab_size, hidden_size), dtype=torch.float32, device=_input.device)
        unit_scale = torch.ones((), dtype=torch.float32, device=_input.device)

    for chunk_index, start in enumerate(range(0, tokens, chunk)):
        stop = min(start + chunk, tokens)
        rows = stop - start
        logits_chunk = logits[:rows]
        input_chunk = _input[start:stop]

        _project_and_dz(
            input_chunk,
            weight,
            logits_chunk,
            target[start:stop],
            loss_1d[start:stop],
            loss_scale,
            partial_max[:rows],
            partial_sum[:rows],
            completion_count[:rows],
            vocab_size,
            num_vocab_tiles,
            stats_block_size,
            ignore_index,
            needs_dz,
            reduction_mean,
        )

        # logits_chunk now holds dZ. dX is scaled AFTER the low-precision GEMM
        # (in backward), matching the baseline; store the raw projection here.
        if grad_input is not None:
            torch.mm(logits_chunk, weight, out=grad_input[start:stop])

        # Accumulate raw dZ.T @ X into the FP32 dW buffer: overwrite on the first
        # chunk, add on subsequent chunks.
        if dweight_accum is not None:
            _launch_matmul_tn(logits_chunk, input_chunk, dweight_accum, unit_scale, chunk_index > 0)

    loss = loss_1d.sum()
    # Retain the RAW FP32 dW accumulator (scaled + cast in backward) so small
    # gradients are not lost to an early BF16 cast.
    return loss, grad_input, dweight_accum, gradient_scale


def chunked_fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    dweight_accum: Optional[torch.Tensor],
    gradient_scale: torch.Tensor,
    weight_dtype: torch.dtype,
):
    """Backward for the chunked forward.

    dX is the raw BF16 projection, scaled out-of-place by
    ``upstream * gradient_scale``. dW is the retained FP32 accumulator: the
    combined ``upstream * gradient_scale`` factor is applied in FP32 and the
    result cast to ``weight_dtype`` exactly once by a fused scale-and-cast
    kernel (no O(V*H) FP32 temporary, and small gradients survive the cast).
    Every output is a fresh tensor, so the saved gradients are never mutated --
    repeated backward with a different upstream (including ``1``) stays correct.
    """
    grad_input_out = None
    if grad_input is not None:
        grad_input_out = grad_input * (grad_output * gradient_scale)

    grad_weight_out = None
    if dweight_accum is not None:
        combined_scale = (grad_output * gradient_scale).to(torch.float32)
        grad_weight_out = torch.empty(dweight_accum.shape, dtype=weight_dtype, device=dweight_accum.device)
        _launch_scale_cast(dweight_accum, grad_weight_out, combined_scale)

    return grad_input_out, grad_weight_out


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    # This Function computes dX/dW during forward with token chunking. The
    # transient logits/dZ workspace is O(min(N, chunk_size) * V) instead of the
    # full O(N * V); the retained gradients add O(N*H + V*H).
    supports_chunk_size = True
    # ce_impl / ce_mode are accepted only as self-identity placeholders (see
    # _validate_ce_dispatch); this Function does NOT dispatch to alternative CE
    # implementations, so it does not advertise inner-impl dispatch.
    supports_inner_impl_dispatch = False

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
        ce_impl=None,
        ce_mode=None,
        chunk_size=None,
    ):
        _validate_ce_dispatch(ce_impl, ce_mode)
        loss, grad_input, dweight_accum, gradient_scale = chunked_fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            ce_weight=ce_weight,
            bias=bias,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
            chunk_size=chunk_size,
        )

        # Retain the raw BF16 dX, the raw FP32 dW accumulator, and the scale.
        # Neither the full [N, V] dZ nor the input activations are saved. The
        # target weight dtype is tracked so backward can cast dW once.
        ctx.save_for_backward(grad_input, dweight_accum, gradient_scale)
        ctx.weight_dtype = weight.dtype
        return loss, None, None, None

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        del grad_output2, grad_output3, grad_output4
        grad_input, dweight_accum, gradient_scale = ctx.saved_tensors
        grad_input, grad_weight = chunked_fused_linear_cross_entropy_backward(
            grad_output,
            grad_input,
            dweight_accum,
            gradient_scale,
            ctx.weight_dtype,
        )
        return (
            grad_input,
            grad_weight,
            None,  # target
            None,  # bias
            None,  # ce_weight
            None,  # ignore_index
            None,  # lse_square_scale
            None,  # label_smoothing
            None,  # reduction
            None,  # softcap
            None,  # return_z_loss
            None,  # accum_dtype
            None,  # use_token_scaling
            None,  # return_token_accuracy
            None,  # return_predicted_tokens
            None,  # ce_impl
            None,  # ce_mode
            None,  # chunk_size
        )


__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "chunked_fused_linear_cross_entropy_backward",
    "chunked_fused_linear_cross_entropy_forward",
    "fused_linear_cross_entropy_backward",
    "fused_linear_cross_entropy_forward",
]
