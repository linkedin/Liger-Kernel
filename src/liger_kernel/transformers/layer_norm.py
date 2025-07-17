import torch
import torch.nn as nn

from liger_kernel.ops.layer_norm import LigerLayerNormFunction


class LigerLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, bias=False, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size) if bias else torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return LigerLayerNormFunction.apply(hidden_states, self.weight, self.bias, self.variance_epsilon)

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"
