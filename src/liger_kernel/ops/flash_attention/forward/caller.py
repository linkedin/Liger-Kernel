import math
from typing import Optional, Tuple

import torch
import triton
from torch import Tensor

from src.liger_kernel.ops.flash_attention.forward.kernel import _fwd_kernel
from src.liger_kernel.ops.flash_attention.utils import attention_pack, attention_unpack, torch_ignore_deterministic, infer_bias_strides, handle_dropout, encode_dtype


def _flash_attn_forward(
    q: Tensor,  # [batch_size, seqlen_q, num_heads_q, head_dim]
    k: Tensor,  # [batch_size, seqlen_k, num_heads_kv, head_dim]
    v: Tensor,  # [batch_size, seqlen_k, num_heads_kv, head_dim]
    attention_mask: Optional[Tensor],  # [batch_size, seqlen_qk]
    bias: Optional[Tensor],  # [1 | batch_size, 1 | num_heads_q, seqlen_q, seqlen_k]
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dropout_seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, float, int]:

    # Currently, variable length (varlen) mode is mutually exclusive with attention masking (TODO)
    if attention_mask is not None:
        varlen_mode = True
        assert bias is None, "Attention mask is not supported along with attention bias. Just use bias instead."
        assert q.size(1) == k.size(1), "Attention mask is not supported with seqlen_q != seqlen_k"
    else:
        varlen_mode = False

    # Retrieve and check shapes (TODO: remove as much as possible of those)
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    expected_kv_shape = (batch, seqlen_k, nheads_kv, head_dim)
    assert nheads_q % nheads_kv == 0, f"{nheads_q = } is not divisible by {nheads_kv =}"
    assert k.shape == expected_kv_shape, f"{k.shape = } <> {expected_kv_shape = }"
    assert v.shape == expected_kv_shape, f"{v.shape = } <> {expected_kv_shape = }"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

    # Depending on attention_mask, switch to varlen
    varlen_mode = varlen_mode and (batch > 1)
    if varlen_mode:
        # Compute padding-related statistics
        cum_seqlens_q = torch.zeros(size=(attention_mask.size(0) + 1,), device=attention_mask.device, dtype=torch.int32)
        with torch_ignore_deterministic():
            cum_seqlens_q[1:] = attention_mask.sum(dim=1).cumsum(0)
        # cum_seqlens_q = [0, seqlen_q1, seqlen_q1+seqlen_q2, ..., seqlen_q1+...+seqlen_qB] of shape [B+1]
        max_seqlen_q: int = attention_mask.size(1)
        max_seqlen_k: int = attention_mask.size(1)
        # Collate all matrices
        q = attention_pack(q, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        k = attention_pack(k, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        v = attention_pack(v, attention_mask)  # [1, sum_seqlens_qk, num_head, head_dim]
        # Update seqlens
        seqlen_q = q.size(1)
    else:
        cum_seqlens_q = None
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k

    # Account for bias and dropout
    stride_bb, stride_bh, stride_bm = infer_bias_strides(bias, batch, nheads_q, seqlen_q, seqlen_k)
    dropout_seed = handle_dropout(dropout_p, dropout_seed, is_forward=True)

    # Setup output accumulator
    o = torch.zeros_like(q)

    # Setup LSE accumulators: in varlen mode, batch is still equal to the nb of queries
    max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128  # wastefull in varlen and not (just use mask)
    lse = torch.zeros((batch, nheads_q, max_seqlen_q_rounded), device=q.device, dtype=torch.float32)

    # Infer problem size and launch kernel
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    PADDED_HEADS = BLOCK_HEADDIM > head_dim
    # BLOCK = 128
    # num_warps = 4 if head_dim <= 64 else 8
    head_ratio = nheads_q // nheads_kv
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads_q)  # noqa: E731
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        bias,
        softmax_scale,
        dropout_p,
        dropout_seed,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        stride_bb, stride_bh, stride_bm,
        nheads_q,
        head_ratio,
        seqlen_q,
        cum_seqlens_q,  # array containing [seqlen_q_1, ..., seqlen_q_B] , if VARLEN, else None
        seqlen_k,
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 128,
        max_seqlen_k // 128,  # key for triton cache (limit number of compilations)
        encode_dtype(q),
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # VARLEN=varlen_mode, IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        VARLEN=varlen_mode,
        USE_DROPOUT=(dropout_p > 0),
        IS_CAUSAL=causal,
        BIAS_ON=(bias is not None),
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        PADDED_HEADS=PADDED_HEADS,
    )

    # When in variable length mode, we need to unpack the packed tensors
    if varlen_mode:
        o = attention_unpack(o, cum_seqlens_q, *attention_mask.shape)

    return o, lse, softmax_scale, dropout_seed  # softmax_scale could have been updated
