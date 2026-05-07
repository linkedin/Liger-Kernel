from typing import Optional

import torch

try:
    from tilegym.suites.liger.cutile.fused_linear_jsd import FusedLinearJSDFunction as LigerFusedLinearJSDFunction
    from tilegym.suites.liger.cutile.fused_linear_jsd import _fused_linear_jsd_forward as _cutile_fused_linear_jsd_forward
    from tilegym.suites.liger.cutile.fused_linear_jsd import fused_linear_jsd as _cutile_fused_linear_jsd

    _TILEGYM_IMPORT_ERROR = None
    _TILEGYM_AVAILABLE = True
except ImportError as exc:
    LigerFusedLinearJSDFunction = None
    _cutile_fused_linear_jsd = None
    _cutile_fused_linear_jsd_forward = None
    _TILEGYM_IMPORT_ERROR = exc
    _TILEGYM_AVAILABLE = False


def _require_tilegym() -> None:
    if not _TILEGYM_AVAILABLE:
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from _TILEGYM_IMPORT_ERROR


def fused_linear_jsd_forward(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    jsd_beta: float,
    ignore_index: int,
    has_label: bool,
    temperature: float,
):
    _require_tilegym()
    return _cutile_fused_linear_jsd_forward(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta,
        ignore_index,
        has_label,
        temperature,
        compute_grad_input=student_input.requires_grad,
        compute_grad_weight=student_weight.requires_grad,
    )


def fused_linear_jsd_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
):
    _require_tilegym()
    scale_is_one = torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device))
    if scale_is_one:
        return grad_input, grad_weight
    return (
        (grad_input * grad_output) if grad_input is not None else None,
        (grad_weight * grad_output) if grad_weight is not None else None,
    )


class TileGymLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(H, V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(H, V, bias=False, dtype=dtype, device=device)
        self.beta = beta
        self.ignore_index = ignore_index
        self.temperature = temperature

    def forward(
        self,
        student_input: torch.Tensor,
        teacher_input: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ):
        _require_tilegym()
        return _cutile_fused_linear_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            shift_labels=label,
            beta=self.beta,
            ignore_index=self.ignore_index,
            temperature=self.temperature,
        )
