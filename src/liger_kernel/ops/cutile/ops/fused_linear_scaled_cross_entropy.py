"""cuTile interface for fused linear scaled cross entropy."""

import math

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import LOG2E
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.cutile.ops.utils import _select_cross_entropy_block_size

ConstInt = ct.Constant[int]


def _validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


def _validate_inputs(_input, weight, target, ignore_index):
    if _input.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[M,H], weight[V,H], and target[M]")
    if _input.shape[0] != target.shape[0] or _input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input {tuple(_input.shape)}, weight {tuple(weight.shape)} and target "
            f"{tuple(target.shape)} shapes are incompatible"
        )
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("input, hidden, and vocabulary dimensions must be non-empty")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError("input, weight, and target must be on the same device")
    if _input.dtype != weight.dtype or not torch.is_floating_point(_input):
        raise TypeError("input and weight must have the same floating-point dtype")
    if target.dtype != torch.int64:
        raise TypeError("target must be an int64 tensor")

    out_of_range = ((target < 0) | (target >= weight.shape[0])) & (target != ignore_index)
    if bool(out_of_range.any()):
        raise ValueError(
            f"target contains values outside [0, {weight.shape[0]}) that are not ignore_index={ignore_index}"
        )


@ct.kernel(occupancy=4)
def _fused_scaled_cross_entropy_forward_kernel_ct(
    logits,
    target,
    nll,
    entropy,
    lse,
    vocab_size,
    inv_temperature,
    ignore_index,
    BLOCK_SIZE: ConstInt,
    RETURN_ENTROPY: ConstInt,
):
    row_idx = ct.bid(0)
    target_idx = ct.load(target, row_idx, shape=())

    if target_idx == ignore_index:
        ct.scatter(nll, row_idx, ct.astype(0.0, nll.dtype))
        ct.scatter(lse, row_idx, ct.astype(0.0, lse.dtype))
        if RETURN_ENTROPY:
            ct.scatter(entropy, row_idx, ct.astype(0.0, entropy.dtype))
        return

    target_col = ct.astype(target_idx, ct.int32)
    target_logit = ct.astype(
        ct.gather(logits, (row_idx, target_col), check_bounds=False),
        ct.float32,
    )
    target_logit *= inv_temperature

    running_max = ct.float32(-math.inf)
    running_sum = ct.float32(0.0)
    running_weighted_sum = ct.float32(0.0)
    num_chunks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    for chunk_idx in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), chunk_idx * BLOCK_SIZE)
        in_bounds = ct.less(col_idx, vocab_size)
        logits_chunk = ct.astype(
            ct.gather(logits, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf),
            ct.float32,
        )
        logits_chunk *= inv_temperature

        chunk_max = ct.max(logits_chunk, 0, keepdims=False)
        exp_chunk = ct.exp2((logits_chunk - chunk_max) * LOG2E, flush_to_zero=True)
        chunk_sum = ct.sum(exp_chunk, 0, keepdims=False)

        new_max = ct.maximum(running_max, chunk_max)
        previous_scale = ct.exp2((running_max - new_max) * LOG2E, flush_to_zero=True)
        chunk_scale = ct.exp2((chunk_max - new_max) * LOG2E, flush_to_zero=True)

        running_sum = running_sum * previous_scale + chunk_sum * chunk_scale
        if RETURN_ENTROPY:
            safe_logits_chunk = ct.where(in_bounds, logits_chunk, 0.0)
            chunk_weighted_sum = ct.sum(exp_chunk * safe_logits_chunk, 0, keepdims=False)
            running_weighted_sum = running_weighted_sum * previous_scale + chunk_weighted_sum * chunk_scale
        running_max = new_max

    logsumexp = running_max + ct.log(running_sum)

    ct.scatter(nll, row_idx, ct.astype(logsumexp - target_logit, nll.dtype))
    ct.scatter(lse, row_idx, ct.astype(logsumexp, lse.dtype))
    if RETURN_ENTROPY:
        entropy_value = logsumexp - running_weighted_sum / running_sum
        ct.scatter(entropy, row_idx, ct.astype(entropy_value, entropy.dtype))


def fused_scaled_cross_entropy_forward(
    _input,
    weight,
    target,
    temperature=1.0,
    ignore_index=-100,
    m_tiles_per_cluster=1,
    return_entropy=False,
):
    """Compute per-token NLL and optional entropy in token chunks.

    Returns ``(nll, entropy, lse, input, weight, hidden_size)``. The final
    three values are retained by the autograd wrapper for backward.
    """
    _validate_inputs(_input, weight, target, ignore_index)
    _validate_temperature(temperature)
    if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
        raise TypeError("m_tiles_per_cluster must be an int")
    if m_tiles_per_cluster < 1:
        raise ValueError("m_tiles_per_cluster must be >= 1")
    if not isinstance(return_entropy, bool):
        raise TypeError("return_entropy must be a bool")

    BT, H = _input.shape
    V = weight.shape[0]
    inc_factor = (V + H - 1) // H
    chunk_size = _next_power_of_2((BT + inc_factor - 1) // inc_factor)
    block_size = _select_cross_entropy_block_size(V)

    nll = torch.empty(BT, dtype=torch.float32, device=_input.device)
    entropy = torch.empty(BT, dtype=_input.dtype, device=_input.device) if return_entropy else None
    lse = torch.empty(BT, dtype=torch.float32, device=_input.device)
    dummy_entropy = torch.empty(1, dtype=_input.dtype, device=_input.device)
    target = target.contiguous()

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        logits_chunk = (_input[start:end] @ weight.t()).contiguous()
        entropy_arg = entropy[start:end] if return_entropy else dummy_entropy

        ct.launch(
            torch.cuda.current_stream(),
            (end - start, 1, 1),
            _fused_scaled_cross_entropy_forward_kernel_ct,
            (
                logits_chunk,
                target[start:end],
                nll[start:end],
                entropy_arg,
                lse[start:end],
                int(V),
                float(1.0 / temperature),
                int(ignore_index),
                int(block_size),
                int(return_entropy),
            ),
        )

    return nll, entropy, lse, _input, weight, H


class _LigerFusedLinearScaledCrossEntropyCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, temperature, ignore_index, return_entropy):
        ctx.set_materialize_grads(False)
        nll, entropy, lse, saved_input, saved_weight, _ = fused_scaled_cross_entropy_forward(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            return_entropy=return_entropy,
        )

        saved_entropy = entropy if entropy is not None else _input.new_empty(0)
        ctx.save_for_backward(saved_input, saved_weight, target, lse, saved_entropy)
        ctx.temperature = temperature
        ctx.ignore_index = ignore_index
        ctx.return_entropy = return_entropy
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        raise NotImplementedError("cuTile fused linear scaled cross entropy backward is not implemented")


class LigerFusedLinearScaledCrossEntropyFunction:
    """Compute per-token scaled NLL and optional vocabulary entropy.

    ``m_tiles_per_cluster`` is accepted for API compatibility but does not
    change the cuTile schedule.
    """

    @staticmethod
    def apply(
        _input,
        weight,
        target,
        temperature=1.0,
        ignore_index=-100,
        m_tiles_per_cluster=1,
        return_entropy=False,
    ):
        if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
            raise TypeError("m_tiles_per_cluster must be an int")
        if m_tiles_per_cluster < 1:
            raise ValueError("m_tiles_per_cluster must be >= 1")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")

        return _LigerFusedLinearScaledCrossEntropyCuTileFunction.apply(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            return_entropy,
        )
