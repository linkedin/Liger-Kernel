import os
import random
import warnings

import numpy as np
import pytest
import torch

from liger_kernel.ops.helion.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyHelion
from liger_kernel.utils import infer_device

device = infer_device()


def supports_bfloat16():
    if device == "cuda":
        return torch.cuda.get_device_capability() >= (8, 0)  # Ampere and newer
    elif device == "xpu":
        return True
    else:
        return False


def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if device == "cuda":
        # If you are using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == "xpu":
        # If you are using XPU
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


class TorchLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, target):
        logits = self.lm_head(x).to(torch.float32)
        return self.ce_loss(logits, target)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.flce = LigerFusedLinearCrossEntropyHelion(
            ignore_index=ignore_index, reduction=reduction, grad_in_forward=True
        )

    def forward(self, x, target):
        return self.flce(x, self.lm_head.weight, target)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 1024, 4096, 32000),  # llama
        (3, 423, 1000, 10000),  # weird shapes
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float32, 1e-1, 1e-2),
    ],
)
def test_fused_linear_cross_entropy_correctness(B, T, H, V, reduction, dtype, atol, rtol):
    input = torch.randn(B * T, H, device=device, requires_grad=True)
    weight = torch.randn(V, H, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), device=device)

    ref_lm_head_ce = TorchLMHeadCE(H, V, dtype=dtype, reduction=reduction).to(device=device)
    liger_lm_head_ce = LigerLMHeadCE(H, V, dtype=dtype, reduction=reduction).to(device=device)

    ref_lm_head_ce.lm_head.weight.data = weight.data
    liger_lm_head_ce.lm_head.weight.data = weight.data

    ref_input = input.detach().clone().requires_grad_(True)
    liger_input = input.detach().clone().requires_grad_(True)

    # Forward pass
    ref_loss: torch.Tensor = ref_lm_head_ce(ref_input, target)
    liger_loss: torch.Tensor = liger_lm_head_ce(liger_input, target)

    torch.testing.assert_close(liger_loss, ref_loss, rtol=rtol, atol=atol)

    # Backward pass (backward() with reduction=="none" is not supported yet)
    if reduction == "none":
        warnings.warn("backward() with reduction='none' is not supported yet", UserWarning)

    else:
        liger_loss.backward()
        ref_loss.backward()

        assert liger_lm_head_ce.lm_head.weight.grad.isnan().sum() == 0, "lm_head.weight of liger contains nan"
        assert ref_lm_head_ce.lm_head.weight.grad.isnan().sum() == 0, "lm_head.weight of ref contains nan"
        assert liger_input.grad.isnan().sum() == 0
        assert liger_input.grad.isinf().sum() == 0
        torch.testing.assert_close(liger_input.grad, ref_input.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            liger_lm_head_ce.lm_head.weight.grad,
            ref_lm_head_ce.lm_head.weight.grad,
            rtol=rtol,
            atol=atol,
        )
