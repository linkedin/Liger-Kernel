import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=0.0, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon = eps
        self.offset = offset

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon, self.offset
        )
