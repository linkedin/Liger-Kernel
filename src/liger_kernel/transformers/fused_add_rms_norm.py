import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedAddRMSNormFunction
from liger_kernel.ops.fused_add_rms_norm import fused_add_rms_norm_output_only


class LigerFusedAddRMSNorm(nn.Module):
    """Add a residual and apply RMSNorm.

    Set return_residual=False to return only the normalized output. This path
    uses Triton and needs torch.no_grad() when any input or weight requires gradients.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=False,
        return_residual=True,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place = (eps, offset, casting_mode, in_place)
        self.return_residual = return_residual

    def forward(self, hidden_states, residual):
        if not self.return_residual:
            return fused_add_rms_norm_output_only(
                hidden_states, residual, self.weight, self.variance_epsilon, self.offset, self.casting_mode
            )
        return LigerFusedAddRMSNormFunction.apply(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, "
            f"in_place={self.in_place}, return_residual={self.return_residual}"
        )
