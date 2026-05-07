from typing import Optional

import torch

try:
    from tilegym.suites.liger.cutile.jsd import JSDFunction as LigerJSDFunction
    from tilegym.suites.liger.cutile.jsd import _jsd_backward as _cutile_jsd_backward
    from tilegym.suites.liger.cutile.jsd import _jsd_forward as _cutile_jsd_forward
    from tilegym.suites.liger.cutile.jsd import jsd as _cutile_jsd

    _TILEGYM_IMPORT_ERROR = None
    _TILEGYM_AVAILABLE = True
except ImportError as exc:
    LigerJSDFunction = None
    _cutile_jsd = None
    _cutile_jsd_backward = None
    _cutile_jsd_forward = None
    _TILEGYM_IMPORT_ERROR = exc
    _TILEGYM_AVAILABLE = False


def _require_tilegym() -> None:
    if not _TILEGYM_AVAILABLE:
        raise ImportError(
            "tilegym cutile backend is not available. Install it from the ocean repo."
        ) from _TILEGYM_IMPORT_ERROR


def jsd_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor],
    beta: float,
    ignore_index: int,
    has_label: bool,
):
    _require_tilegym()
    return _cutile_jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)


def jsd_backward(dX: torch.Tensor, grad_output: torch.Tensor):
    _require_tilegym()
    return _cutile_jsd_backward(dX, grad_output)


class TileGymJSD(torch.nn.Module):
    def __init__(self, beta: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
    ):
        _require_tilegym()
        return _cutile_jsd(
            input,
            target,
            shift_labels=shift_labels,
            beta=self.beta,
            ignore_index=self.ignore_index,
        )
