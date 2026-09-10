# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile fused linear cross entropy for Hopper (SM90) and Blackwell (SM100).

All three GEMMs -- the projection, the input-gradient (dX), and the
weight-gradient (dW) -- use PyTorch's tuned cuBLAS path. cuTile computes
partitioned CE statistics, overwrites a reusable ``[min(N, C), V]`` logits
buffer with dZ, and (for the chunked path) normalizes dZ in the CE kernel. dW is
accumulated across token chunks by cuBLAS ``mm`` (first chunk) then ``addmm``
(subsequent chunks), so the transient logits workspace is ``O(min(N, C) * V)``
instead of the full ``O(N * V)`` materialization; the retained gradients (BF16
dX and BF16 dW) add ``O(N*H + V*H)``.

Precision/normalization/cast order (chunked/autograd path) matches the Triton
FLCE backend's mean/sum policy: the CE kernel folds the global non-ignored mean
normalizer (``1/N``; ``1`` for ``sum``) into dZ in FP32 -- after the target
subtract-1 (so ``softmax(x_y) - 1`` cancellation keeps full FP32 precision) and
before the BF16 cast -- so dX and the dW accumulation run on the already
normalized low-precision dZ. The dW accumulator dtype follows ``accum_dtype``
(``None`` -> weight dtype/BF16, ``torch.float32`` -> FP32). A BF16 accumulator
rounds to BF16 after EACH chunk, so the end-of-forward ``.to`` weight-dtype cast
is a no-op; an explicit FP32 accumulator is converted to the weight dtype exactly
once at the end of the forward. Backward then only applies
the upstream gradient (never re-normalizes); the unchunked legacy backward is the
FP32-only accumulation path. This is a precision-STORAGE policy
aligned with Triton; it does not claim bitwise-identical CE reductions.

Two low-level interfaces are exported. ``fused_linear_cross_entropy_forward`` /
``fused_linear_cross_entropy_backward`` retain the original unchunked contract
(the forward returns the full ``[N, V]`` RAW dZ and the 7-argument backward runs
the dW matmul, applying the combined ``upstream * mean_normalizer`` scale in FP32
before a single weight-dtype cast so small gradients survive -- legacy FP32-only
accumulation). The token-chunked path used by the autograd Function lives in
``chunked_fused_linear_cross_entropy_forward`` /
``chunked_fused_linear_cross_entropy_backward`` and returns/consumes the final
weight-dtype gradients directly (``loss, grad_input, grad_weight``).
"""

import operator

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.fused_linear_cross_entropy import _get_chunk_size
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import compare_version

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]

LOG2E = 1.4426950408889634
SCALE_CAST_TILE_M = 128
SCALE_CAST_TILE_N = 128
CE_BLOCK_SIZE = 2048
LOGITS_STATS_BLOCK_SIZE = 4096
MAX_STATS_BLOCK_SIZE = 1024

# The FP32 dW accumulation uses the ``torch.mm`` / ``torch.addmm``
# ``out_dtype=torch.float32`` overload (BF16 operands -> FP32 accumulator) added
# in torch 2.8. On older torch the operands are upcast to FP32 first; see
# ``_accumulate_dw``. This is a runtime capability check only -- it does not
# raise the package floor.
_ADDMM_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")


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
    NORMALIZE_GRADIENTS: ConstBool,
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
            # Fold the global mean normalizer (1/N) into the gradient in FP32 --
            # AFTER the target subtract-1 (so the (softmax(x_y) - 1) cancellation
            # keeps full FP32 precision) and BEFORE the cast to the logits dtype.
            # This matches the Triton kernel, which normalizes dZ in-kernel and
            # then runs the dX/dW GEMMs on the already-normalized low-precision dZ.
            # ``sum`` reduction has a unit normalizer, so it is skipped. Legacy
            # callers pass NORMALIZE_GRADIENTS=False to keep the raw dZ contract.
            if NORMALIZE_GRADIENTS and REDUCTION_MEAN:
                gradient = gradient * row_scale
            ct.scatter(logits, (row, cols), ct.astype(gradient, logits.dtype), check_bounds=True)


def _accumulate_dw(
    dweight_accum: torch.Tensor,
    dz: torch.Tensor,
    x: torch.Tensor,
    accumulate: bool,
) -> None:
    """Accumulate ``dZ.T @ X`` into the ``dweight_accum`` buffer via cuBLAS.

    ``dz`` is ``[rows, V]`` and ``x`` is ``[rows, H]``; the contribution
    ``dZ.T @ X`` is ``[V, H]``. The first token chunk (``accumulate`` is
    ``False``) overwrites the buffer with ``torch.mm``; subsequent chunks add
    with ``torch.addmm``. The accumulation dtype follows ``dweight_accum.dtype``:

    * **Matching dtype** (e.g. an ``accum_dtype=None``/BF16 buffer whose dtype
      already equals ``dz``): accumulate straight into the buffer with a plain
      ``mm``/``addmm`` (``out=dweight_accum``), mirroring the Triton backend's
      direct low-precision ``addmm`` -- no FP32 upcast, no parameter-sized FP32
      temporary. Doing the running sum with ``addmm`` (not a Python ``mm`` +
      ``+=``) avoids the extra rounding of a separate BF16 partial product.
    * **FP32 buffer with half operands** (``accum_dtype=torch.float32`` or the
      legacy path): on torch>=2.8 the BF16 operands drive the
      ``out_dtype=torch.float32`` overload directly; on older torch (no overload)
      only the bounded BF16 operands are upcast to FP32 before the same mm/addmm
      into the FP32 buffer.

    This single helper is shared by the chunked and legacy paths.
    """
    dz_t = dz.t()  # [V, rows] view
    buf_dtype = dweight_accum.dtype
    if buf_dtype == dz.dtype:
        # Matching-dtype accumulator: direct low-precision mm/addmm, no upcast.
        if accumulate:
            torch.addmm(dweight_accum, dz_t, x, out=dweight_accum)
        else:
            torch.mm(dz_t, x, out=dweight_accum)
    elif buf_dtype == torch.float32 and dz.dtype in (torch.float16, torch.bfloat16):
        if _ADDMM_SUPPORTS_OUT_DTYPE:
            if accumulate:
                torch.addmm(dweight_accum, dz_t, x, out_dtype=torch.float32, out=dweight_accum)
            else:
                torch.mm(dz_t, x, out_dtype=torch.float32, out=dweight_accum)
        else:
            dz_t_fp32 = dz_t.to(torch.float32)
            x_fp32 = x.to(torch.float32)
            if accumulate:
                torch.addmm(dweight_accum, dz_t_fp32, x_fp32, out=dweight_accum)
            else:
                torch.mm(dz_t_fp32, x_fp32, out=dweight_accum)
    else:
        raise NotImplementedError(
            f"cuTile FLCE cannot accumulate dW with buffer dtype {buf_dtype!r} and dZ dtype {dz.dtype!r}"
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


def _resolve_accum_dtype(accum_dtype, weight_dtype: torch.dtype, *, allow_low_precision: bool) -> torch.dtype:
    """Resolve the dW-accumulation buffer dtype from ``accum_dtype``.

    The chunked (default) path mirrors Triton's ``accum_dtype`` policy:
    ``None`` inherits the weight dtype (BF16 here) so dW accumulates across
    chunks in the parameter dtype, ``torch.float32`` forces an FP32 accumulator,
    and an explicit BF16 that matches the weight dtype is accepted (same buffer
    dtype). Anything else is rejected clearly rather than silently downgraded.

    The legacy unchunked path (``allow_low_precision=False``) has always
    accumulated dW in FP32; it accepts only ``None``/``torch.float32`` and
    rejects an explicit low-precision accum it cannot honor instead of silently
    ignoring it.
    """
    if accum_dtype is None:
        return weight_dtype if allow_low_precision else torch.float32
    if accum_dtype == torch.float32:
        return torch.float32
    if allow_low_precision and accum_dtype == weight_dtype:
        return accum_dtype
    if not allow_low_precision:
        raise NotImplementedError(
            f"legacy cuTile FLCE accumulates dW in FP32 only (accum_dtype=torch.float32 or None); got {accum_dtype!r}"
        )
    raise NotImplementedError(
        f"cuTile FLCE supports accum_dtype in (None, torch.float32, {weight_dtype!r}); got {accum_dtype!r}"
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
    normalize_gradients,
):
    """Project a token chunk to logits and overwrite it in place with dZ.

    ``torch.mm`` fills ``logits_chunk`` with the BF16 projection; the fused CE
    kernel then writes the per-token loss and (when ``needs_dz``) replaces the
    logits with the softmax-minus-one-hot gradient dZ. When
    ``normalize_gradients`` is set the kernel folds the global mean normalizer
    into dZ in FP32 before the BF16 cast (the chunked path), so the dX/dW GEMMs
    run on already-normalized dZ; the legacy path passes ``False`` to preserve
    the raw dZ contract. Shared by the legacy unchunked path (one full-batch
    call) and the chunked path (per chunk).
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
            normalize_gradients,
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

    Precision policy (legacy, unchanged): dZ is stored RAW (unnormalized) and dW
    is accumulated in FP32; the combined ``upstream * mean_normalizer`` scale is
    applied in FP32 and cast to the weight dtype exactly once in backward, so
    small gradients survive. Because dW is FP32-only here, an explicit
    low-precision ``accum_dtype`` is rejected rather than silently ignored.
    """
    _reject_unsupported(
        bias,
        ce_weight,
        lse_square_scale,
        label_smoothing,
        softcap,
        return_z_loss,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    _validate_inputs(_input, weight, target, reduction)
    _resolve_accum_dtype(accum_dtype, weight.dtype, allow_low_precision=False)

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
        False,  # normalize_gradients: legacy keeps RAW dZ; scale is applied in backward
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

    Applies ``upstream * gradient_scale`` to dX. dW is computed in FP32 by cuBLAS
    (``dZ.T @ X`` accumulated into an FP32 buffer), then the combined
    ``upstream * gradient_scale`` factor is applied in FP32 and the result cast to
    ``weight_dtype`` exactly once by the fused scale-and-cast kernel. Casting
    after (not before) the upstream multiply preserves small gradients that would
    flush to zero if rounded to BF16 first. All operations are out-of-place, so
    the saved tensors are not mutated across repeated backward calls.
    """
    total_scale = grad_output * gradient_scale
    grad_input_out = None
    if grad_input is not None:
        grad_input_out = grad_input * total_scale

    grad_weight = None
    if grad_logits is not None:
        dweight_accum = torch.empty(weight_shape, dtype=torch.float32, device=grad_logits.device)
        _accumulate_dw(dweight_accum, grad_logits, _input, accumulate=False)
        combined_scale = total_scale.to(torch.float32)
        grad_weight = torch.empty(weight_shape, dtype=weight_dtype, device=grad_logits.device)
        _launch_scale_cast(dweight_accum, grad_weight, combined_scale)

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
):
    """Token-chunked forward used by the autograd Function.

    Processes tokens in row chunks so the transient logits/dZ workspace is
    ``O(min(N, C) * V)``, where the token-chunk size ``C`` comes from the shared
    :func:`liger_kernel.ops.fused_linear_cross_entropy._get_chunk_size` helper --
    the EXACT same default geometry as the Triton FLCE backend, with no public
    ``chunk_size`` knob.

    Precision/normalization/cast order matches the Triton backend's mean/sum
    path: the CE kernel normalizes dZ by the global non-ignored mean (``1/N``,
    ``1`` for ``sum``) in FP32 BEFORE storing it to BF16 and BEFORE any gradient
    GEMM, so dX (``normalized_dZ @ weight``) and the dW accumulation both run on
    the already-normalized low-precision dZ -- there is no post-GEMM
    normalization. The dW accumulator dtype follows ``accum_dtype``
    (see :func:`_resolve_accum_dtype`): ``None`` inherits the weight dtype (BF16)
    and accumulates across chunks in that dtype via ``addmm``, rounding to BF16
    after EACH chunk so the end-of-forward weight-dtype ``.to`` is a no-op;
    ``torch.float32`` accumulates in FP32 and is converted to the weight dtype
    exactly ONCE, here at the end of the forward, so the
    returned ``grad_weight`` is already the final weight-dtype gradient.

    Returns ``(loss, grad_input, grad_weight)`` -- both gradients are in the
    parameter dtype (BF16). :func:`chunked_fused_linear_cross_entropy_backward`
    only applies the upstream gradient (no re-normalization).
    """
    _reject_unsupported(
        bias,
        ce_weight,
        lse_square_scale,
        label_smoothing,
        softcap,
        return_z_loss,
        use_token_scaling,
        return_token_accuracy,
        return_predicted_tokens,
    )
    _validate_inputs(_input, weight, target, reduction)
    accum_buffer_dtype = _resolve_accum_dtype(accum_dtype, weight.dtype, allow_low_precision=True)

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

    chunk = _get_chunk_size(tokens, hidden_size, vocab_size)
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

    # dW accumulator reused across chunks. Its dtype follows accum_dtype: BF16
    # (accum_dtype=None/BF16, matching the weight) or FP32 (accum_dtype=fp32).
    # It is cast to the weight dtype exactly once at the end of the forward.
    dweight_accum = None
    if weight.requires_grad:
        dweight_accum = torch.empty((vocab_size, hidden_size), dtype=accum_buffer_dtype, device=_input.device)

    for chunk_index, start in enumerate(range(0, tokens, chunk)):
        stop = min(start + chunk, tokens)
        rows = stop - start
        logits_chunk = logits[:rows]
        input_chunk = _input[start:stop]

        # normalize_gradients=True: the CE kernel folds the global mean normalizer
        # into dZ (FP32) before the BF16 cast, so the GEMMs below see normalized dZ.
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
            True,
        )

        # logits_chunk now holds the NORMALIZED dZ. dX is the direct projection of
        # that normalized dZ (no post-GEMM scaling).
        if grad_input is not None:
            torch.mm(logits_chunk, weight, out=grad_input[start:stop])

        # Accumulate normalized dZ.T @ X into the dW buffer via cuBLAS: overwrite
        # (mm) on the first chunk, add (addmm) on subsequent chunks.
        if dweight_accum is not None:
            _accumulate_dw(dweight_accum, logits_chunk, input_chunk, chunk_index > 0)

    loss = loss_1d.sum()
    # Cast the completed dW accumulator to the weight dtype exactly once (matches
    # the Triton mean/sum path, which casts grad_weight to weight.dtype at the end
    # of forward). dX is already the parameter dtype.
    grad_weight = dweight_accum.to(weight.dtype) if dweight_accum is not None else None
    return loss, grad_input, grad_weight


def chunked_fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
):
    """Backward for the chunked forward.

    Both gradients are already normalized (by the CE kernel) and cast to the
    weight dtype (at the end of the forward), so backward only applies the
    upstream gradient -- it NEVER re-applies the mean normalizer. The multiply is
    out-of-place and the result is kept in the gradient's own dtype (mirroring
    the Triton ``element_mul`` in-place BF16 multiply without mutating the saved
    tensor), so repeated backward with a different upstream (including ``1``) and
    retained-graph alias safety both hold.
    """
    grad_input_out = None
    if grad_input is not None:
        grad_input_out = (grad_input * grad_output).to(grad_input.dtype)

    grad_weight_out = None
    if grad_weight is not None:
        grad_weight_out = (grad_weight * grad_output).to(grad_weight.dtype)

    return grad_input_out, grad_weight_out


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    # This Function computes dX/dW during forward with token chunking. The
    # transient logits/dZ workspace is O(min(N, C) * V) instead of the full
    # O(N * V); the retained gradients add O(N*H + V*H). The token-chunk size C
    # is the shared default Triton geometry (_get_chunk_size); there is no public
    # chunk_size knob.
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
    ):
        _validate_ce_dispatch(ce_impl, ce_mode)
        loss, grad_input, grad_weight = chunked_fused_linear_cross_entropy_forward(
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
        )

        # Retain only the COMPLETED weight-dtype gradients: BF16 dX and BF16 dW
        # (already normalized in the CE kernel and cast once at the end of the
        # forward). Neither the full [N, V] dZ nor the input activations nor an
        # FP32 accumulator are saved. Backward only applies the upstream gradient.
        ctx.save_for_backward(grad_input, grad_weight)
        return loss, None, None, None

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        del grad_output2, grad_output3, grad_output4
        grad_input, grad_weight = ctx.saved_tensors
        grad_input, grad_weight = chunked_fused_linear_cross_entropy_backward(
            grad_output,
            grad_input,
            grad_weight,
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
        )


__all__ = [
    "LigerFusedLinearCrossEntropyFunction",
    "chunked_fused_linear_cross_entropy_backward",
    "chunked_fused_linear_cross_entropy_forward",
    "fused_linear_cross_entropy_backward",
    "fused_linear_cross_entropy_forward",
]
