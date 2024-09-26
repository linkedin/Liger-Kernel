import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
from torch import Tensor

from src.liger_kernel.ops.flash_attention.backward.compute_delta import _compute_delta
from src.liger_kernel.ops.flash_attention.backward.kernel import _bwd_kernel
from src.liger_kernel.ops.flash_attention.utils import attention_pack, attention_unpack, torch_ignore_deterministic, infer_bias_strides, handle_dropout, encode_dtype


def _flash_attn_backward(
    dO: Tensor,  # [batch_size, seqlen_q, nheads_q, head_dim]
    q: Tensor,  # [batch_size, seqlen_q, nheads_q, head_dim]
    k: Tensor,  # [batch_size, seqlen_k, nheads_kv, head_dim]
    v: Tensor,  # [batch_size, seqlen_k, nheads_kv, head_dim]
    bias: Optional[Tensor],  # [1 | batch_size, 1 | nheads_q, seqlen_q, seqlen_k]
    attention_mask: Optional[Tensor],  # [batch_size, seqlen_qk]
    o: Tensor,  # [batch_size, seqlen_q, nheads_q, head_dim]
    lse: Tensor,  # [batch_size, nheads_q, max_seqlen_q_rounded]
    dropout_p: float,
    causal: bool,
    softmax_scale: Optional[float],
    dropout_seed: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:

    if attention_mask is not None:
        assert bias is None, "Attention mask is not supported along with attention bias. Just use bias instead."
        assert q.size(1) == k.size(1), "Attention mask is not supported with seqlen_q != seqlen_k"
        varlen_mode = (attention_mask.size(0) > 1)
        useless_padding = attention_mask.size(1) - attention_mask.sum(-1).max().item()
        if useless_padding > 0:
            dO = dO[:, :-useless_padding]
            q = q[:, :-useless_padding]
            k = k[:, :-useless_padding]
            v = v[:, :-useless_padding]
            attention_mask = attention_mask[:, :-useless_padding]
            o = o[:, :-useless_padding]
    else:
        varlen_mode = False
        useless_padding = 0

    # Retrieve and check shapes
    dO = dO.contiguous() if dO.stride(-1) != 1 else dO
    batch_size, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    max_seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    assert nheads_q % nheads_kv == 0, f"{nheads_q = } is not divisible by {nheads_kv =}"
    assert lse.shape == (batch_size, nheads_q, max_seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1

    # Depending on attention_mask, switch to varlen
    if varlen_mode:
        # Compute padding-related statistics
        cum_seqlens_q = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        cum_seqlens_k = torch.zeros(size=(attention_mask.size(0)+1,), device=attention_mask.device, dtype=torch.int32)
        with torch_ignore_deterministic():
            cum_seqlens_q[1:] = attention_mask.sum(dim=1).cumsum(0)
            cum_seqlens_k[1:] = attention_mask.sum(dim=1).cumsum(0)
        # cum_seqlens_q = [0, seqlen_q1, seqlen_q1+seqlen_q2, ..., seqlen_q1+...+seqlen_qB] of shape [B+1]
        max_seqlen_q: int = attention_mask.size(1)
        max_seqlen_k: int = attention_mask.size(1)
        # Collate all matrices
        q = attention_pack(q, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        k = attention_pack(k, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        v = attention_pack(v, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        o = attention_pack(o, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        dO = attention_pack(dO, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        # Update seqlens
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
    else:
        cum_seqlens_q = None
        cum_seqlens_k = None
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

    # Handle bias and dropout
    stride_bb, stride_bh, stride_bm = infer_bias_strides(bias, batch_size, nheads_q, seqlen_q, seqlen_k)
    dropout_seed = handle_dropout(dropout_p, dropout_seed, is_forward=False)

    # Prepare gradient accumulators # TODO: maybe we can initialize this as empty -- check pre hook
    dq = torch.zeros_like(q, dtype=torch.float32)  # [batch_size|1, seqlen_q|sum_seqlens_qk, nheads_q, head_dim]
    dk = torch.zeros(size=(k.size(0), k.size(1), q.size(2), k.size(3)), device=k.device, dtype=k.dtype)
    dv = torch.zeros(size=(v.size(0), v.size(1), q.size(2), v.size(3)), device=v.device, dtype=v.dtype)
    delta = torch.zeros_like(lse)  # [batch_size, nheads_q, max_seqlen_q_rounded]

    # Infer problem size
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    # Launch the delta computation kernel
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch_size * nheads_q)  # noqa: E731
    _compute_delta[grid](
        o,
        dO,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        dO.stride(0),
        dO.stride(2),
        dO.stride(1),
        nheads_q,
        seqlen_q,
        max_seqlen_q_rounded,
        cum_seqlens_q,
        head_dim,
        max_seqlen_q // 32,
        encode_dtype(o),
        VARLEN=varlen_mode,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # Launch backward kernel
    head_ratio = nheads_q // nheads_kv
    grid = lambda META: (  # noqa: E731
        triton.cdiv(seqlen_k, META["BLOCK_N1"]) + triton.cdiv(seqlen_q, META["BLOCK_M2"]),
        batch_size * nheads_q,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        dO,
        dq,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        dropout_p,
        dropout_seed,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        stride_bb, stride_bh, stride_bm,
        dO.stride(0), dO.stride(2), dO.stride(1),
        dq.stride(0), dq.stride(2), dq.stride(1),
        dk.stride(0), dk.stride(2), dk.stride(1),
        dv.stride(0), dv.stride(2), dv.stride(1),
        nheads_q,
        head_ratio,
        seqlen_q,
        cum_seqlens_q,
        seqlen_k,
        cum_seqlens_k,
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 32,
        max_seqlen_k // 32,  # key for triton cache (limit number of compilations)
        encode_dtype(q),
        VARLEN=varlen_mode,
        IS_CAUSAL=causal,
        BIAS_ON=(bias is not None),
        USE_DROPOUT=(dropout_p > 0),
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # GQA reduction
    if head_ratio > 1:
        dk = dk.unflatten(dim=2, sizes=(nheads_kv, head_ratio)).sum(-2)
        dv = dv.unflatten(dim=2, sizes=(nheads_kv, head_ratio)).sum(-2)

    # In case of variable length mode, we need to unpack the gradients
    if varlen_mode:
        dq = attention_unpack(dq, cum_seqlens_q, batch_size, max_seqlen_q)
        dk = attention_unpack(dk, cum_seqlens_k, batch_size, max_seqlen_k)
        dv = attention_unpack(dv, cum_seqlens_k, batch_size, max_seqlen_k)
    # And add back the useless padding if there was any
    if useless_padding > 0:
        dq = F.pad(dq, (0, 0, 0, 0, 0, useless_padding))
        dk = F.pad(dk, (0, 0, 0, 0, 0, useless_padding))
        dv = F.pad(dv, (0, 0, 0, 0, 0, useless_padding))

    return dq, dk, dv
