from typing import Optional

import torch

from liger_kernel.ops import LigerFusedLinearCrossEntropyFunction
from liger_kernel.transformers.functional import CrossEntropyOutput
from liger_kernel.transformers.functional import build_flce_apply_args


class LigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be 'mean' or 'sum' or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling
        self.return_token_accuracy = return_token_accuracy
        self.return_predicted_tokens = return_predicted_tokens
        self.chunk_size = chunk_size

    def forward(self, lin_weight, _input, target, bias=None):
        base_args = (
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.accum_dtype,
            self.use_token_scaling,
            self.return_token_accuracy,
            self.return_predicted_tokens,
        )
        # Only append the trailing ce_impl/ce_mode/chunk_size args the active backend
        # Function advertises support for. The prior unconditional ``(None, None)`` broke
        # backends (native CuTe DSL, cuTile, Ascend) whose Function takes only 15 args.
        apply_args = build_flce_apply_args(LigerFusedLinearCrossEntropyFunction, base_args, chunk_size=self.chunk_size)
        loss, z_loss, token_accuracy, predicted_tokens = LigerFusedLinearCrossEntropyFunction.apply(*apply_args)
        if not self.return_z_loss and not self.return_token_accuracy and not self.return_predicted_tokens:
            return loss

        return CrossEntropyOutput(
            loss=loss, z_loss=z_loss, token_accuracy=token_accuracy, predicted_tokens=predicted_tokens
        )
