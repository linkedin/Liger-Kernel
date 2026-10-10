import torch
import torch.nn as nn

from liger_kernel.ops import LigerAttnResFunction


class LigerAttnRes(nn.Module):
    """Attention Residuals (AttnRes) from Kimi/Moonshot AI (arXiv:2603.15031).

    Replaces the standard residual connection ``h = h_prev + f(RMSNorm(h_prev))``
    with a softmax attention over the depth (block) dimension: the stacked outputs
    of ``N`` blocks are each RMSNorm'd, scored against a learned pseudo-query, and
    the resulting per-block softmax weights produce a weighted sum. This is the
    module wrapper around :func:`~liger_kernel.transformers.functional.liger_attn_res`.

    Args:
        hidden_size: hidden dimension ``D`` of each block output.
        eps: epsilon for the per-block RMSNorm (default: 1e-6).
        query_init_std: standard deviation of the normal initialization for the
            learned pseudo-query ``w_query`` (default: 0.02). ``w_norm`` (the
            per-block RMSNorm weight) is initialized to ones.

    Shape:
        - Input: ``[N, B, T, D]`` stacked block outputs, or a list of ``N``
          tensors each of shape ``[B, T, D]``.
        - Output: ``[B, T, D]``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, query_init_std: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.query_init_std = query_init_std
        self.w_query = nn.Parameter(torch.randn(hidden_size) * query_init_std)
        self.w_norm = nn.Parameter(torch.ones(hidden_size))

    def forward(self, V):
        if isinstance(V, (list, tuple)):
            V = torch.stack(V)
        return LigerAttnResFunction.apply(V, self.w_query, self.w_norm, self.eps)

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
