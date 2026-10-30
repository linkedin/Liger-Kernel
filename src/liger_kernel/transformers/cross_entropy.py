from typing import Optional

import torch

from liger_kernel.backends import dispatch
from liger_kernel.backends import resolve_impl
from liger_kernel.ops import LigerCrossEntropyFunction  # noqa: F401 — re-exported for backend-routing checks
from liger_kernel.ops.cross_entropy import check_loss_dtype
from liger_kernel.transformers.functional import CrossEntropyOutput


class LigerCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
        loss_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        check_loss_dtype(loss_dtype)
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.weight = weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.return_token_accuracy = return_token_accuracy
        self.return_predicted_tokens = return_predicted_tokens
        self.loss_dtype = loss_dtype

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        # Only the Triton backend implements loss_dtype, so other backends never see the kwarg by default.
        extra = {}
        if self.loss_dtype is not None:
            device = _input.device if _input.is_cuda else None
            impl_name = resolve_impl("cross_entropy", device=device).impl_name
            if impl_name != "nvidia-triton":
                raise ValueError(
                    f"loss_dtype is only supported by the Triton cross_entropy backend, but '{impl_name}' was selected."
                )
            extra["loss_dtype"] = self.loss_dtype
        loss, z_loss, token_accuracy, predicted_tokens = dispatch(
            "cross_entropy",
            _input,
            target,
            self.weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.return_token_accuracy,
            self.return_predicted_tokens,
            **extra,
        )
        if not self.return_z_loss and not self.return_token_accuracy and not self.return_predicted_tokens:
            return loss

        return CrossEntropyOutput(
            loss=loss, z_loss=z_loss, token_accuracy=token_accuracy, predicted_tokens=predicted_tokens
        )
