from test.utils import assert_verbose_allclose, set_seed
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)
from liger_kernel.utils import infer_device

device = infer_device()

# set random seed globally
set_seed()


class TorchLMHeadPreference(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based preference loss.

    :param H: hidden size
    :param V: vocab size
    :param bias: whether to use bias
    :param beta: weight for the odds ratio loss
    :param softcap: scaler for softcapping logits
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        softcap: Optional[float] = None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=bias, dtype=dtype
        )
        self.ignore_index = ignore_index
        self.beta = beta
        self.softcap = softcap

    def forward(self, x, target):
        logits = self.lin(x).to(torch.float32)
        if self.softcap is not None and self.softcap != 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)

        log_probs = F.log_softmax(logits, dim=-1)
        
        len_chosen = target.shape[0] // 2
        loss_mask = target != self.ignore_index
        label = torch.where(loss_mask, target, 0)
        
        per_token_logps = log_probs.gather(-1, label.unsqueeze(-1)).squeeze(-1)
        average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        
        chosen_logps = average_log_prob[:len_chosen]
        rejected_logps = average_log_prob[len_chosen:]
        
        # Simple preference loss
        preference_loss = -self.beta * (chosen_logps - rejected_logps).mean()
        
        return preference_loss


class LigerLMHeadPreference(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        softcap: Optional[float] = None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=bias, dtype=dtype
        )
        self.ignore_index = ignore_index
        self.beta = beta
        self.softcap = softcap

    def forward(self, x, target):
        def simple_preference_loss(chosen_logps, rejected_logps, target, beta=0.1):
            return -beta * (chosen_logps - rejected_logps).mean()

        loss, *_ = LigerFusedLinearPreferenceBase.apply(
            x,
            self.lin.weight,
            target,
            self.lin.bias,
            simple_preference_loss,
            chunk_size=1,
            ignore_index=self.ignore_index,
            beta=self.beta,
            compute_nll_loss=False,
            compiled=True,
            softcap=self.softcap,
        )
        return loss


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "ignore_index, beta, softcap",
    [
        (-100, 0.1, None),
        (42, 0.2, 30.0),  # Pass non-default values to ensure all params work
    ],
)
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    bias,
    ignore_index,
    beta,
    softcap,
    atol,
    rtol,
):
    torch_lm_head = TorchLMHeadPreference(
        H=H,
        V=V,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        softcap=softcap,
        dtype=dtype,
    ).to(device)
    
    liger_lm_head = LigerLMHeadPreference(
        H=H,
        V=V,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        softcap=softcap,
        dtype=dtype,
    ).to(device)

    # init the linear layers with the same weights
    torch_lm_head.lin.weight.data = liger_lm_head.lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head.lin.bias.data = liger_lm_head.lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    # Create input tensors
    _tensor = torch.randn(B * T * 2, H, device=device, dtype=dtype) * scalar  # *2 for chosen/rejected pairs
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    # Create target tensor
    target = torch.randint(0, V, (B * T * 2,), device=device, dtype=torch.long)
    
    # Assign some random elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T, (1,)).item()
    indices_to_assign = torch.randperm(B * T * 2)[:num_elements_to_assign]
    target[indices_to_assign] = ignore_index

    # Forward pass
    output1 = torch_lm_head(_input1, target)
    output2 = liger_lm_head(_input2, target)

    # Check outputs match
    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    # Backward pass
    output1.backward()
    output2.backward()

    # Check gradients match
    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head.lin.weight.grad,
        liger_lm_head.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_lm_head.lin.bias.grad,
            liger_lm_head.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )
