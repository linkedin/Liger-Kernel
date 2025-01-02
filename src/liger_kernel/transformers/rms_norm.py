import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place = (
            eps,
            offset,
            casting_mode,
            in_place,
        )

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}"
        )
