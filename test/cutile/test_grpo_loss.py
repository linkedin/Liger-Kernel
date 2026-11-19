import importlib.util

import pytest
import torch

from test.utils import COLLAPSE_DTYPES
from test.utils import assert_grpo_dominant_target
from test.utils import assert_grpo_k3_near_reference


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.grpo_loss import GrpoLossFunction
else:
    GrpoLossFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile GRPO loss requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]


def _cutile_grpo_loss(
    logits,
    old_logp,
    ref_logp,
    completion_ids,
    advantages,
    temperature,
    beta,
    eps_low,
    eps_high,
    inplace,
    loss_type,
    importance_sampling_level,
    reduce,
):
    return GrpoLossFunction.apply(
        logits,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        None,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace,
        loss_type,
        None,
        reduce,
        importance_sampling_level,
    )


@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
@pytest.mark.parametrize("dtype, rtol", COLLAPSE_DTYPES)
def test_dominant_target_does_not_collapse(importance_sampling_level, dtype, rtol):
    assert_grpo_dominant_target(_cutile_grpo_loss, importance_sampling_level, dtype, rtol, "cuda")


@pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
def test_k3_kl_near_reference(importance_sampling_level):
    assert_grpo_k3_near_reference(_cutile_grpo_loss, importance_sampling_level, "cuda")
