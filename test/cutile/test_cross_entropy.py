import importlib.util

import pytest
import torch
import torch.nn.functional as F

from test.utils import COLLAPSE_DTYPES
from test.utils import FP64_SAFE_COLLAPSE_MARGINS
from test.utils import assert_cross_entropy_does_not_collapse
from test.utils import assert_grad_close_per_row
from test.utils import dominant_target_logits
from test.utils import set_seed


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.cross_entropy import LigerCrossEntropyFunction
else:
    LigerCrossEntropyFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile cross entropy requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]


def _cutile_ce(x, target, weight=None, label_smoothing=0.0, reduction="mean"):
    return LigerCrossEntropyFunction.apply(x, target, weight, -100, 0.0, label_smoothing, reduction)[0]


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("dtype, rtol", COLLAPSE_DTYPES)
def test_dominant_target_does_not_collapse(reduction, dtype, rtol):
    assert_cross_entropy_does_not_collapse(_cutile_ce, dtype, rtol, "cuda", reduction)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("use_weight", [False, True])
def test_dominant_target_does_not_collapse_smoothed_weighted(label_smoothing, use_weight):
    set_seed()
    V = 4096
    logits, target = dominant_target_logits(FP64_SAFE_COLLAPSE_MARGINS, V=V, device="cuda")
    weight = torch.rand(V, device="cuda") + 0.5 if use_weight else None
    _input = logits.clone().requires_grad_(True)
    _ref = logits.double().requires_grad_(True)

    loss = _cutile_ce(_input, target, weight=weight, label_smoothing=label_smoothing, reduction="none")
    ref = F.cross_entropy(
        _ref,
        target,
        weight=None if weight is None else weight.double(),
        label_smoothing=label_smoothing,
        reduction="none",
    )
    loss.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(loss.double(), ref.detach(), rtol=1e-4, atol=0)
    # With smoothing, softmax - eps cancels where softmax ~ eps, so compare at the row's scale
    assert_grad_close_per_row(_input.grad, _ref.grad, rtol=1e-4)
