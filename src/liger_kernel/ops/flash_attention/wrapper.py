from typing import Optional

import torch
from torch import Tensor

from src.liger_kernel.ops.flash_attention.backward.caller import _flash_attn_backward
from src.liger_kernel.ops.flash_attention.forward.caller import _flash_attn_forward


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        softmax_scale: Optional[Tensor] = None,
        dropout_seed: Optional[int] = None,
    ):
        """
        Compute the forward pass of the FlashAttention function.
        Args:
            - ctx (): the autograd.Function context
            - q (Tensor): the query projection tensor, of shape [batch_size, seqlen_q, num_heads, head_dim]
            - k (Tensor): the key projection tensor, of shape [batch_size, seqlen_k, num_heads, head_dim]
            - v (Tensor): the values projection tensor, of shape [batch_size, seqlen_k, num_heads, head_dim]
            - attention_mask (Optional[Tensor]): an optional attention mask of shape [batch_size, seqlen_q].
                Forces seqlen_q == seqlen_k.
            - causal (bool): a boolean to indicate whether or not to use causal attention
            - softmax_scale (Optional[float]): an optional float to scale the pre-softmax attention scores. Defaults
                to 1 / sqrt(head_dim)
        Return:
            the attention output tensor
        """
        # Make sure that the last dimension is contiguous
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        v = v if v.stride(-1) == 1 else v.contiguous()
        attention_bias = None if (attention_bias is None) else attention_bias.contiguous()
        o, lse, ctx.softmax_scale, ctx.dropout_seed = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            bias=attention_bias,
            dropout_p=dropout_p,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_seed=dropout_seed,
        )
        ctx.save_for_backward(q, k, v, attention_bias, attention_mask, o, lse)
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Compute the backward pass of the FlashAttention function.
        Args:
            - ctx (): the autograd.Function context
            - do (Tensor): the gradient of the output tensor, of shape [batch_size, seqlen_q, num_heads, head_dim]
        Return:
            three tensors, the gradients of q, k and v respectively (check forward for shape info)
        """
        q, k, v, bias, attention_mask, o, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(
            dO=do,
            q=q,
            k=k,
            v=v,
            bias=bias,
            attention_mask=attention_mask,
            o=o,
            lse=lse,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            dropout_seed=ctx.dropout_seed,
        )
        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[Tensor] = None,
    dropout_seed: Optional[int] = None,
) -> Tensor:
    return FlashAttnFunc.apply(q, k, v, attention_mask, attention_bias, dropout_p, causal, softmax_scale, dropout_seed)
