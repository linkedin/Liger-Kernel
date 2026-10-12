# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cut Cross Entropy (CCE) without materializing the full logits matrix.

Implements *Cut Your Losses in Large-Vocabulary Language Models* (Wijmans et
al., ICLR 2025): https://openreview.net/forum?id=E4Fk3YuG56.

This implementation is based on the clean-room kernel introduced in
huggingface/trl#6859 and includes portability and correctness fixes maintained
by Liger Kernel.
"""

import operator

from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh

NEG = tl.constexpr(-1.0e30)
_BWD_BUFFER_BYTES = 256 * 1024 * 1024


@triton.jit
def _softcap(x, softcap):
    return softcap * tanh(x / softcap)


@triton.jit
def _cce_forward_kernel(
    E,
    C,
    Bias,
    M_p,
    S_p,
    T_p,
    A_p,
    N,
    H,
    V,
    stride_en,
    stride_eh,
    stride_cv,
    stride_ch,
    logit_scale,
    softcap,
    V_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    NEED_ENTROPY: tl.constexpr,
    NEED_ARGMAX: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_h = tl.arange(0, BLOCK_H)
    offs_v = tl.arange(0, BLOCK_V)

    v_begin = pid_s * V_SPLIT
    v_end = tl.minimum(v_begin + V_SPLIT, V)

    m = tl.full([BLOCK_N], NEG, tl.float32)
    s = tl.zeros([BLOCK_N], tl.float32)
    t = tl.zeros([BLOCK_N], tl.float32)
    argmax = tl.zeros([BLOCK_N], tl.int32)

    e_ptrs = E + offs_n[:, None] * stride_en + offs_h[None, :] * stride_eh
    for v0 in range(v_begin, v_end, BLOCK_V):
        vv = v0 + offs_v
        v_mask = vv < v_end
        c_ptrs = C + vv[:, None] * stride_cv + offs_h[None, :] * stride_ch

        acc = tl.zeros([BLOCK_N, BLOCK_V], tl.float32)
        for h0 in range(0, H, BLOCK_H):
            if EVEN_H:
                a = tl.load(e_ptrs + h0 * stride_eh, mask=n_mask[:, None], other=0.0)
                b = tl.load(c_ptrs + h0 * stride_ch, mask=v_mask[:, None], other=0.0)
            else:
                h_mask = h0 + offs_h < H
                a = tl.load(e_ptrs + h0 * stride_eh, mask=n_mask[:, None] & h_mask[None, :], other=0.0)
                b = tl.load(c_ptrs + h0 * stride_ch, mask=v_mask[:, None] & h_mask[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(b), acc)

        if HAS_BIAS:
            acc += tl.load(Bias + vv, mask=v_mask, other=0.0).to(tl.float32)[None, :]
        acc *= logit_scale
        if HAS_SOFTCAP:
            acc = _softcap(acc, softcap)
        acc = tl.where(v_mask[None, :], acc, NEG)

        tile_m = tl.max(acc, 1)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.exp(m - new_m)
        p = tl.exp(acc - new_m[:, None])
        p = tl.where(v_mask[None, :], p, 0.0)
        s = s * alpha + tl.sum(p, 1)
        if NEED_ENTROPY:
            t = t * alpha + tl.sum(p * acc, 1)
        if NEED_ARGMAX:
            argmax = tl.where(tile_m > m, v0 + tl.argmax(acc, 1), argmax)
        m = new_m

    out_offsets = pid_s * N + offs_n
    tl.store(M_p + out_offsets, m, mask=n_mask)
    tl.store(S_p + out_offsets, s, mask=n_mask)
    if NEED_ENTROPY:
        tl.store(T_p + out_offsets, t, mask=n_mask)
    if NEED_ARGMAX:
        tl.store(A_p + out_offsets, argmax, mask=n_mask)


@triton.jit
def _cce_target_logit_kernel(
    E,
    C,
    Bias,
    Targets,
    Out,
    N,
    H,
    V,
    ignore_index,
    stride_en,
    stride_eh,
    stride_cv,
    stride_ch,
    logit_scale,
    softcap,
    BLOCK_H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    pid = tl.program_id(0)
    target = tl.load(Targets + pid)
    # Keep ignored and invalid targets in bounds. Invalid non-ignored targets are
    # reported by the asynchronous validation performed by the Python wrapper.
    safe_target = tl.where((target == ignore_index) | (target < 0) | (target >= V), 0, target)

    acc = tl.zeros([BLOCK_H], tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offsets = h0 + tl.arange(0, BLOCK_H)
        h_mask = offsets < H
        a = tl.load(E + pid * stride_en + offsets * stride_eh, mask=h_mask, other=0.0)
        b = tl.load(C + safe_target * stride_cv + offsets * stride_ch, mask=h_mask, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)
    out = tl.sum(acc)

    if HAS_BIAS:
        out += tl.load(Bias + safe_target).to(tl.float32)
    out *= logit_scale
    if HAS_SOFTCAP:
        out = _softcap(out, softcap)
    tl.store(Out + pid, out)


@triton.jit
def _cce_backward_d_kernel(
    E,
    C,
    Bias,
    Targets,
    LSE,
    W,
    D,
    N,
    H,
    v_begin,
    v_end,
    stride_en,
    stride_eh,
    stride_cv,
    stride_ch,
    stride_dn,
    logit_scale,
    softcap,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_h = tl.arange(0, BLOCK_H)
    vv = v_begin + pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = vv < v_end

    e_ptrs = E + offs_n[:, None] * stride_en + offs_h[None, :] * stride_eh
    c_ptrs = C + vv[:, None] * stride_cv + offs_h[None, :] * stride_ch

    acc = tl.zeros([BLOCK_N, BLOCK_V], tl.float32)
    for h0 in range(0, H, BLOCK_H):
        if EVEN_H:
            a = tl.load(e_ptrs + h0 * stride_eh, mask=n_mask[:, None], other=0.0)
            b = tl.load(c_ptrs + h0 * stride_ch, mask=v_mask[:, None], other=0.0)
        else:
            h_mask = h0 + offs_h < H
            a = tl.load(e_ptrs + h0 * stride_eh, mask=n_mask[:, None] & h_mask[None, :], other=0.0)
            b = tl.load(c_ptrs + h0 * stride_ch, mask=v_mask[:, None] & h_mask[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)

    if HAS_BIAS:
        acc += tl.load(Bias + vv, mask=v_mask, other=0.0).to(tl.float32)[None, :]
    acc *= logit_scale
    if HAS_SOFTCAP:
        acc = _softcap(acc, softcap)

    lse = tl.load(LSE + offs_n, mask=n_mask, other=0.0)
    d = tl.exp(acc - lse[:, None])
    targets = tl.load(Targets + offs_n, mask=n_mask, other=-1)
    d -= (vv[None, :] == targets[:, None]).to(tl.float32)

    if HAS_SOFTCAP:
        d *= 1.0 - (acc / softcap) * (acc / softcap)
    d *= logit_scale

    w = tl.load(W + offs_n, mask=n_mask, other=0.0)
    d *= w[:, None]
    d_ptrs = D + offs_n[:, None] * stride_dn + (vv - v_begin)[None, :]
    tl.store(d_ptrs, d.to(D.dtype.element_ty), mask=n_mask[:, None] & v_mask[None, :])


def _is_nvidia_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.version.cuda is not None and not is_hip()


def _cce_config(n_tokens: int, hidden_size: int, vocab_size: int, dtype: torch.dtype, device: torch.device) -> dict:
    if not _is_nvidia_device(device):
        # Portable schedule for AMD LDS limits and non-CUDA Triton backends.
        config = {"BLOCK_N": 64, "BLOCK_V": 64, "BLOCK_H": 32, "num_warps": 4, "num_stages": 1}
    elif dtype == torch.float32:
        config = {"BLOCK_N": 128, "BLOCK_V": 128, "BLOCK_H": 32, "num_warps": 8, "num_stages": 3}
    else:
        config = {"BLOCK_N": 128, "BLOCK_V": 256, "BLOCK_H": 64, "num_warps": 8, "num_stages": 4}
    config["BLOCK_N"] = max(16, min(config["BLOCK_N"], triton.next_power_of_2(n_tokens)))
    config["BLOCK_V"] = max(16, min(config["BLOCK_V"], triton.next_power_of_2(vocab_size)))
    config["BLOCK_H"] = max(16, min(config["BLOCK_H"], triton.next_power_of_2(hidden_size)))
    return config


def _cce_num_splits(n_tiles: int, vocab_size: int, block_v: int, device: torch.device) -> tuple[int, int]:
    backend = getattr(torch, device.type, None)
    n_sms = None
    try:
        properties = backend.get_device_properties(device)
        n_sms = next(
            int(getattr(properties, name))
            for name in ("multi_processor_count", "gpu_subslice_count", "gpu_eu_count", "cube_core_num")
            if getattr(properties, name, None) is not None
        )
    except (AssertionError, AttributeError, RuntimeError, StopIteration, TypeError):
        pass
    v_tiles = triton.cdiv(vocab_size, block_v)
    if n_sms is None:
        return 1, v_tiles * block_v
    target = max(1, (2 * n_sms) // max(n_tiles, 1))
    splits = max(1, min(target, v_tiles // 8))
    v_split = triton.cdiv(triton.cdiv(vocab_size, splits), block_v) * block_v
    return triton.cdiv(vocab_size, v_split), v_split


class LigerCCEFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    @amp_custom_fwd
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: Optional[torch.Tensor],
        ignore_index: int,
        logit_scale: float,
        softcap: Optional[float],
        reduction: str,
        return_metrics: bool,
    ):
        n_tokens, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        device, dtype = hidden.device, hidden.dtype
        config = _cce_config(n_tokens, hidden_size, vocab_size, dtype, device)
        block_n, block_v, block_h = config["BLOCK_N"], config["BLOCK_V"], config["BLOCK_H"]
        n_tiles = triton.cdiv(n_tokens, block_n)
        splits, v_split = _cce_num_splits(n_tiles, vocab_size, block_v, device)

        m_partial = torch.empty((splits, n_tokens), dtype=torch.float32, device=device)
        s_partial = torch.empty_like(m_partial)
        t_partial = torch.empty_like(m_partial) if return_metrics else m_partial
        a_partial = torch.empty((splits, n_tokens), dtype=torch.int32, device=device) if return_metrics else m_partial

        with device_context(device):
            _cce_forward_kernel[(n_tiles, splits)](
                hidden,
                weight,
                bias if bias is not None else hidden,
                m_partial,
                s_partial,
                t_partial,
                a_partial,
                n_tokens,
                hidden_size,
                vocab_size,
                hidden.stride(0),
                hidden.stride(1),
                weight.stride(0),
                weight.stride(1),
                logit_scale,
                softcap if softcap is not None else 1.0,
                V_SPLIT=v_split,
                HAS_BIAS=bias is not None,
                HAS_SOFTCAP=softcap is not None,
                NEED_ENTROPY=return_metrics,
                NEED_ARGMAX=return_metrics,
                EVEN_H=hidden_size % block_h == 0,
                BLOCK_N=block_n,
                BLOCK_V=block_v,
                BLOCK_H=block_h,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )

        global_max, global_source = m_partial.max(0)
        alpha = (m_partial - global_max).exp()
        exp_sum = (s_partial * alpha).sum(0)
        lse = global_max + exp_sum.log()

        target_logit = torch.empty(n_tokens, dtype=torch.float32, device=device)
        with device_context(device):
            _cce_target_logit_kernel[(n_tokens,)](
                hidden,
                weight,
                bias if bias is not None else hidden,
                targets,
                target_logit,
                n_tokens,
                hidden_size,
                vocab_size,
                ignore_index,
                hidden.stride(0),
                hidden.stride(1),
                weight.stride(0),
                weight.stride(1),
                logit_scale,
                softcap if softcap is not None else 1.0,
                BLOCK_H=min(1024, triton.next_power_of_2(hidden_size)),
                HAS_BIAS=bias is not None,
                HAS_SOFTCAP=softcap is not None,
                num_warps=4,
            )

        valid = targets != ignore_index
        per_token = torch.where(valid, lse - target_logit, torch.zeros((), dtype=torch.float32, device=device))
        n_valid = valid.sum().clamp(min=1)
        if reduction == "mean":
            loss = per_token.sum() / n_valid
        elif reduction == "sum":
            loss = per_token.sum()
        else:
            loss = per_token

        if return_metrics:
            weighted_logits = (t_partial * alpha).sum(0)
            per_token_entropy = torch.where(
                valid, lse - weighted_logits / exp_sum, torch.zeros((), dtype=torch.float32, device=device)
            )
            entropy_sum = per_token_entropy.sum()
            argmax = a_partial.gather(0, global_source.unsqueeze(0)).squeeze(0)
            num_correct_tokens = ((argmax == targets) & valid).sum()
            ctx.mark_non_differentiable(entropy_sum, num_correct_tokens)
        else:
            entropy_sum = None
            num_correct_tokens = None

        ctx.save_for_backward(hidden, weight, bias, targets, lse, valid, n_valid)
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.reduction = reduction
        return loss, entropy_sum, num_correct_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_loss, grad_entropy_sum, grad_num_correct_tokens):
        hidden, weight, bias, targets, lse, valid, n_valid = ctx.saved_tensors
        n_tokens, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        device, dtype = hidden.device, hidden.dtype

        if ctx.reduction == "mean":
            row_weights = (grad_loss / n_valid).expand(n_tokens)
        elif ctx.reduction == "sum":
            row_weights = grad_loss.expand(n_tokens)
        else:
            row_weights = grad_loss
        row_weights = torch.where(
            valid, row_weights.float(), torch.zeros((), dtype=torch.float32, device=device)
        ).contiguous()

        need_hidden = ctx.needs_input_grad[0]
        need_weight = ctx.needs_input_grad[1]
        need_bias = bias is not None and ctx.needs_input_grad[3]
        grad_hidden = torch.zeros((n_tokens, hidden_size), dtype=torch.float32, device=device) if need_hidden else None
        grad_weight = torch.empty_like(weight) if need_weight else None
        grad_bias = torch.empty_like(bias) if need_bias else None

        config = _cce_config(n_tokens, hidden_size, vocab_size, dtype, device)
        block_n, block_v, block_h = config["BLOCK_N"], config["BLOCK_V"], config["BLOCK_H"]
        chunk = max(
            block_v,
            (_BWD_BUFFER_BYTES // (n_tokens * hidden.element_size())) // block_v * block_v,
        )
        chunk = min(chunk, triton.cdiv(vocab_size, block_v) * block_v)
        d_buffer = torch.empty((n_tokens, chunk), dtype=dtype, device=device)

        for v_begin in range(0, vocab_size, chunk):
            v_end = min(v_begin + chunk, vocab_size)
            chunk_size = v_end - v_begin
            with device_context(device):
                _cce_backward_d_kernel[(triton.cdiv(n_tokens, block_n), triton.cdiv(chunk_size, block_v))](
                    hidden,
                    weight,
                    bias if bias is not None else hidden,
                    targets,
                    lse,
                    row_weights,
                    d_buffer,
                    n_tokens,
                    hidden_size,
                    v_begin,
                    v_end,
                    hidden.stride(0),
                    hidden.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    d_buffer.stride(0),
                    ctx.logit_scale,
                    ctx.softcap if ctx.softcap is not None else 1.0,
                    HAS_BIAS=bias is not None,
                    HAS_SOFTCAP=ctx.softcap is not None,
                    EVEN_H=hidden_size % block_h == 0,
                    BLOCK_N=block_n,
                    BLOCK_V=block_v,
                    BLOCK_H=block_h,
                    num_warps=config["num_warps"],
                    num_stages=config["num_stages"],
                )
            d_logits = d_buffer[:, :chunk_size]
            if need_hidden:
                grad_hidden += torch.mm(d_logits, weight[v_begin:v_end])
            if need_weight:
                torch.mm(d_logits.t(), hidden, out=grad_weight[v_begin:v_end])
            if need_bias:
                grad_bias[v_begin:v_end] = d_logits.sum(0, dtype=torch.float32).to(bias.dtype)

        return (
            grad_hidden.to(dtype) if need_hidden else None,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_cce_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor],
    logit_scale: float,
    softcap: Optional[float],
    reduction: str,
) -> None:
    if hidden.ndim != 2 or hidden.numel() == 0:
        raise ValueError(f"hidden must be a non-empty 2D tensor, got shape {tuple(hidden.shape)}")
    if hidden.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"hidden must have dtype float16, bfloat16, or float32, got {hidden.dtype}")
    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] != hidden.shape[1]:
        raise ValueError(f"weight must have shape (V, {hidden.shape[1]}) with V > 0, got {tuple(weight.shape)}")
    if targets.shape != (hidden.shape[0],):
        raise ValueError(f"targets must have shape ({hidden.shape[0]},), got {tuple(targets.shape)}")
    if targets.dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"targets must have dtype int32 or int64, got {targets.dtype}")
    if weight.dtype != hidden.dtype:
        raise TypeError(f"weight dtype {weight.dtype} does not match hidden dtype {hidden.dtype}")
    if weight.device != hidden.device or targets.device != hidden.device:
        raise ValueError("hidden, weight, and targets must be on the same device")
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"bias must have shape ({weight.shape[0]},), got {tuple(bias.shape)}")
        if bias.dtype != hidden.dtype:
            raise TypeError(f"bias dtype {bias.dtype} does not match hidden dtype {hidden.dtype}")
        if bias.device != hidden.device:
            raise ValueError("bias must be on the same device as hidden")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
    if softcap is not None and softcap <= 0:
        raise ValueError(f"softcap must be greater than zero or None, got {softcap}")
    if not isinstance(logit_scale, (int, float)):
        raise TypeError(f"logit_scale must be a real number, got {type(logit_scale).__name__}")


def liger_cce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    logit_scale: float = 1.0,
    softcap: Optional[float] = None,
    reduction: str = "mean",
    return_metrics: bool = False,
):
    """Compute CCE for ``hidden @ weight.T + bias`` without storing logits.

    When ``return_metrics`` is true, returns ``(loss, metrics)`` where metrics
    contains the raw ``num_correct_tokens`` and ``entropy_sum`` over non-ignored
    tokens. Raw sums are suitable for aggregation across distributed ranks.
    """
    _validate_cce_inputs(hidden, weight, targets, bias, logit_scale, softcap, reduction)
    valid_targets = (targets == ignore_index) | ((targets >= 0) & (targets < weight.shape[0]))
    torch._assert_async(valid_targets.all(), "targets contain a class index outside the vocabulary")

    loss, entropy_sum, num_correct_tokens = LigerCCEFunction.apply(
        hidden,
        weight,
        targets,
        bias,
        ignore_index,
        float(logit_scale),
        softcap,
        reduction,
        return_metrics,
    )
    if return_metrics:
        return loss, {"num_correct_tokens": num_correct_tokens, "entropy_sum": entropy_sum}
    return loss
