from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_orpo
from liger_kernel.chunked_loss.orpo_loss import LigerFusedLinearORPOFunction
from liger_kernel.utils import infer_device
from test.utils import HFAlignmentLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class HFORPOLoss(HFAlignmentLoss):
    """
    Implementation of the Odds Ratio Preference Optimization (ORPO) loss,
    adapted from Hugging Face's implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/orpo_trainer.py
    """

    def __init__(self, ignore_index: int = -100, beta: float = 0.1):
        super().__init__(beta=beta, ignore_index=ignore_index)

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://huggingface.co/papers/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        losses = -self.beta * ratio

        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps

        return (
            losses,
            chosen_rewards,
            rejected_rewards,
            torch.mean(ratio),
            torch.mean(log_odds),
        )


class TorchLMHeadORPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.orpo_loss = HFORPOLoss(ignore_index=ignore_index, beta=beta).get_batch_loss_metrics

    def forward(self, x, y, nll_target=None):
        return self.orpo_loss(self.lin.weight, x, y, self.lin.bias, nll_target=nll_target)


class LigerLMHeadORPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.orpo_loss = LigerFusedLinearORPOLoss(ignore_index=ignore_index, beta=beta)

    def forward(self, x, y, nll_target=None):
        return self.orpo_loss(self.lin.weight, x, y, self.lin.bias, nll_target=nll_target)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
def test_correctness(B, T, H, V, scalar, dtype, atol, rtol, bias, ignore_index, beta):
    B = 2 * B  # orpo loss requires B to be even
    torch_lm_head_orpo = TorchLMHeadORPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_orpo = LigerLMHeadORPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_orpo.lin.weight.data = liger_lm_head_orpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_orpo.lin.bias.data = liger_lm_head_orpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device=device,
        dtype=torch.long,
    )
    nll_target = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    loss1, aggregated_aux_outputs1 = torch_lm_head_orpo(input1, target, nll_target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_orpo(input2, target, nll_target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        assert_verbose_allclose(
            aggregated_aux_outputs1[i],
            aggregated_aux_outputs2[i],
            atol=atol,
            rtol=rtol,
        )

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_orpo.lin.weight.grad,
        liger_lm_head_orpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_orpo.lin.bias.grad,
            liger_lm_head_orpo.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias):
    B = 2 * B

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device=device,
        dtype=torch.long,
    )

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    loss1, _ = LigerFusedLinearORPOFunction.apply(input1, weight1, target, bias1)
    loss2, _ = liger_fused_linear_orpo(input2, weight2, target, bias2)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
