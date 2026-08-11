"""cuTile interface for fused linear scaled cross entropy."""

import math
import os

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import LOG2E
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.cutile.ops.utils import _select_cross_entropy_block_size
from liger_kernel.utils import infer_device_arch

ConstInt = ct.Constant[int]

_MIB = 1024 * 1024
_PORTABLE_LOGITS_WORKSPACE_BYTES = 256 * _MIB
_BLACKWELL_LOGITS_WORKSPACE_BYTES = 512 * _MIB
_BLACKWELL_MIN_TOKENS = 4096
_BLACKWELL_MIN_VOCAB_SIZE = 131072
_WORKSPACE_MB_ENV = "LIGER_CUTILE_SCALED_CE_WORKSPACE_MB"


def _validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")


def _validate_input_metadata(_input, weight, target):
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


def _validate_inputs(_input, weight, target, ignore_index):
    _validate_input_metadata(_input, weight, target)
    out_of_range = ((target < 0) | (target >= weight.shape[0])) & (target != ignore_index)
    if bool(out_of_range.any()):
        raise ValueError(
            f"target contains values outside [0, {weight.shape[0]}) that are not ignore_index={ignore_index}"
        )


def _select_logits_workspace_bytes(device_id, token_count, vocab_size, element_size):
    workspace_override = os.environ.get(_WORKSPACE_MB_ENV)
    if workspace_override is not None:
        try:
            workspace_mb = int(workspace_override)
        except ValueError as exc:
            raise ValueError(f"{_WORKSPACE_MB_ENV} must be a positive integer, got {workspace_override!r}") from exc
        if workspace_mb <= 0:
            raise ValueError(f"{_WORKSPACE_MB_ENV} must be a positive integer, got {workspace_override!r}")
        return workspace_mb * _MIB

    if (
        element_size == 2
        and token_count >= _BLACKWELL_MIN_TOKENS
        and vocab_size >= _BLACKWELL_MIN_VOCAB_SIZE
        and infer_device_arch(device_id).startswith("blackwell")
    ):
        return _BLACKWELL_LOGITS_WORKSPACE_BYTES
    return _PORTABLE_LOGITS_WORKSPACE_BYTES


def _calculate_token_chunk_size(
    token_count,
    vocab_size,
    element_size,
    workspace_bytes=_PORTABLE_LOGITS_WORKSPACE_BYTES,
):
    bytes_per_token = vocab_size * element_size
    max_tokens_per_chunk = max(1, workspace_bytes // bytes_per_token)
    power_of_two_chunk = _next_power_of_2(max_tokens_per_chunk + 1) // 2
    if token_count <= power_of_two_chunk:
        return token_count

    minimum_chunks = (token_count + max_tokens_per_chunk - 1) // max_tokens_per_chunk
    power_of_two_chunks = (token_count + power_of_two_chunk - 1) // power_of_two_chunk
    power_of_two_tail = token_count - (power_of_two_chunks - 1) * power_of_two_chunk

    if power_of_two_chunks > minimum_chunks or power_of_two_tail * 2 < power_of_two_chunk:
        return (token_count + minimum_chunks - 1) // minimum_chunks
    return power_of_two_chunk


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

    Returns per-token NLL, optional entropy, and log-sum-exp.
    """
    _validate_inputs(_input, weight, target, ignore_index)
    _validate_temperature(temperature)
    if not isinstance(m_tiles_per_cluster, int) or isinstance(m_tiles_per_cluster, bool):
        raise TypeError("m_tiles_per_cluster must be an int")
    if m_tiles_per_cluster < 1:
        raise ValueError("m_tiles_per_cluster must be >= 1")
    if not isinstance(return_entropy, bool):
        raise TypeError("return_entropy must be a bool")

    BT = _input.shape[0]
    V = weight.shape[0]
    device_id = _input.device.index if _input.device.index is not None else torch.cuda.current_device()
    workspace_bytes = _select_logits_workspace_bytes(device_id, BT, V, _input.element_size())
    chunk_size = _calculate_token_chunk_size(BT, V, _input.element_size(), workspace_bytes)
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

    return nll, entropy, lse


@ct.kernel(occupancy=4)
def _fused_scaled_cross_entropy_backward_kernel_ct(
    logits,
    target,
    lse,
    entropy,
    grad_nll,
    grad_entropy,
    vocab_size,
    inv_temperature,
    ignore_index,
    BLOCK_SIZE: ConstInt,
    HAS_NLL_GRAD: ConstInt,
    HAS_ENTROPY_GRAD: ConstInt,
):
    row_idx = ct.bid(0)
    target_idx = ct.load(target, row_idx, shape=())
    num_chunks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    if target_idx == ignore_index:
        for chunk_idx in range(num_chunks):
            col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), chunk_idx * BLOCK_SIZE)
            zero_chunk = ct.full((BLOCK_SIZE,), 0.0, dtype=logits.dtype)
            ct.scatter(logits, (row_idx, col_idx), zero_chunk, check_bounds=True)
        return

    target_col = ct.astype(target_idx, ct.int32)
    row_lse = ct.astype(ct.load(lse, row_idx, shape=()), ct.float32)
    row_grad_nll = ct.float32(0.0)
    if HAS_NLL_GRAD:
        row_grad_nll = ct.astype(ct.load(grad_nll, row_idx, shape=()), ct.float32)
    row_entropy = ct.float32(0.0)
    row_grad_entropy = ct.float32(0.0)
    if HAS_ENTROPY_GRAD:
        row_entropy = ct.astype(ct.load(entropy, row_idx, shape=()), ct.float32)
        row_grad_entropy = ct.astype(ct.load(grad_entropy, row_idx, shape=()), ct.float32)

    for chunk_idx in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), chunk_idx * BLOCK_SIZE)
        raw_logits = ct.astype(
            ct.gather(logits, (row_idx, col_idx), check_bounds=True, padding_value=-math.inf),
            ct.float32,
        )
        scaled_logits = raw_logits * inv_temperature
        probabilities = ct.exp(scaled_logits - row_lse)

        grad_scaled_logits = probabilities * row_grad_nll
        is_target = ct.equal(col_idx, target_col)
        grad_scaled_logits = ct.where(is_target, grad_scaled_logits - row_grad_nll, grad_scaled_logits)
        if HAS_ENTROPY_GRAD:
            log_probabilities = scaled_logits - row_lse
            grad_scaled_logits -= row_grad_entropy * probabilities * (log_probabilities + row_entropy)
        grad_raw_logits = grad_scaled_logits * inv_temperature

        ct.scatter(
            logits,
            (row_idx, col_idx),
            ct.astype(grad_raw_logits, logits.dtype),
            check_bounds=True,
        )


def fused_scaled_cross_entropy_backward(
    grad_nll,
    _input,
    weight,
    target,
    lse,
    temperature,
    ignore_index,
    *,
    entropy=None,
    grad_entropy=None,
):
    """Compute gradients for per-token NLL and optional entropy outputs.

    Returns gradients for ``_input`` and ``weight``. ``grad_nll`` may be
    ``None`` for entropy-only backward; ``entropy`` and ``grad_entropy`` are
    needed only when entropy contributes to the gradient.
    """
    _validate_input_metadata(_input, weight, target)
    _validate_temperature(temperature)
    if grad_nll is None and grad_entropy is None:
        raise ValueError("at least one of grad_nll or grad_entropy must be provided")
    if grad_nll is not None and grad_nll.shape != target.shape:
        raise ValueError(f"grad_nll must have shape {tuple(target.shape)}, got {tuple(grad_nll.shape)}")
    if lse.shape != target.shape:
        raise ValueError(f"lse must have shape {tuple(target.shape)}, got {tuple(lse.shape)}")
    if grad_entropy is not None:
        if entropy is None:
            raise ValueError("entropy is required when grad_entropy is provided")
        if entropy.shape != target.shape or grad_entropy.shape != target.shape:
            raise ValueError(
                f"entropy and grad_entropy must have shape {tuple(target.shape)}, "
                f"got {tuple(entropy.shape)} and {tuple(grad_entropy.shape)}"
            )

    BT = _input.shape[0]
    V = weight.shape[0]
    device_id = _input.device.index if _input.device.index is not None else torch.cuda.current_device()
    workspace_bytes = _select_logits_workspace_bytes(device_id, BT, V, _input.element_size())
    chunk_size = _calculate_token_chunk_size(BT, V, _input.element_size(), workspace_bytes)
    block_size = _select_cross_entropy_block_size(V)

    grad_input = torch.empty_like(_input) if _input.requires_grad else None
    grad_weight = torch.empty_like(weight) if weight.requires_grad else None
    grad_logits_workspace = torch.empty(
        min(BT, chunk_size),
        V,
        dtype=_input.dtype,
        device=_input.device,
    )
    dummy_f32 = torch.empty(1, dtype=torch.float32, device=_input.device)
    dummy_input_dtype = torch.empty(1, dtype=_input.dtype, device=_input.device)

    target = target.contiguous()
    lse = lse.contiguous()
    grad_nll = grad_nll.contiguous() if grad_nll is not None else None
    entropy = entropy.contiguous() if entropy is not None else None
    grad_entropy = grad_entropy.contiguous() if grad_entropy is not None else None

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        input_chunk = _input[start:end]
        grad_logits_chunk = grad_logits_workspace[: end - start]
        torch.mm(input_chunk, weight.t(), out=grad_logits_chunk)

        ct.launch(
            torch.cuda.current_stream(),
            (end - start, 1, 1),
            _fused_scaled_cross_entropy_backward_kernel_ct,
            (
                grad_logits_chunk,
                target[start:end],
                lse[start:end],
                entropy[start:end] if entropy is not None else dummy_input_dtype,
                grad_nll[start:end] if grad_nll is not None else dummy_f32,
                grad_entropy[start:end] if grad_entropy is not None else dummy_input_dtype,
                int(V),
                float(1.0 / temperature),
                int(ignore_index),
                int(block_size),
                int(grad_nll is not None),
                int(grad_entropy is not None),
            ),
        )

        if grad_input is not None:
            torch.mm(grad_logits_chunk, weight, out=grad_input[start:end])
        if grad_weight is not None:
            if start == 0:
                torch.mm(grad_logits_chunk.t(), input_chunk, out=grad_weight)
            else:
                # Accumulate the GEMM directly into grad_weight instead of
                # materializing a full V x H temporary for every token chunk.
                torch.addmm(grad_weight, grad_logits_chunk.t(), input_chunk, out=grad_weight)

    return grad_input, grad_weight


class _LigerFusedLinearScaledCrossEntropyCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, temperature, ignore_index, return_entropy):
        ctx.set_materialize_grads(False)
        nll, entropy, lse = fused_scaled_cross_entropy_forward(
            _input,
            weight,
            target,
            temperature,
            ignore_index,
            return_entropy=return_entropy,
        )

        saved_entropy = entropy if entropy is not None else _input.new_empty(0)
        ctx.save_for_backward(_input, weight, target, lse, saved_entropy)
        ctx.temperature = temperature
        ctx.ignore_index = ignore_index
        ctx.return_entropy = return_entropy
        return (nll, entropy) if return_entropy else nll

    @staticmethod
    def backward(ctx, grad_nll, grad_entropy=None):
        _input, weight, target, lse, entropy = ctx.saved_tensors
        grad_input, grad_weight = fused_scaled_cross_entropy_backward(
            grad_nll,
            _input,
            weight,
            target,
            lse,
            ctx.temperature,
            ctx.ignore_index,
            entropy=entropy if ctx.return_entropy else None,
            grad_entropy=grad_entropy,
        )
        return grad_input, grad_weight, None, None, None, None


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
