from test.utils import HFAlignmentLoss, assert_verbose_allclose, set_seed
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.cpo_loss import LigerFusedLinearCPOFunction

# set random seed globally
set_seed()


class HFCPOLoss(HFAlignmentLoss):
    """
    HF's implementation of CPO loss in TRL. https://github.com/huggingface/trl/blob/main/trl/trainer/cpo_trainer.py
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__(alpha=alpha, beta=beta, ignore_index=ignore_index)
        # Sigmoid defaults to the CPO loss defined in the paper listed above.
        self.loss_type = "sigmoid"
        self.label_smoothing = label_smoothing

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = policy_chosen_logps - policy_rejected_logps

        # The beta is a temperature parameter for the CPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative CPO loss.
        if self.loss_type == "sigmoid":
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = (
                F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid']"
            )

        return losses


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        # (1, 2, 12, 128),
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-3),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "ignore_index, beta, alpha", [(-100, 0.1, 1.0), (42, 0.2, 0.85)]
)
def test_correctness(
    B, T, H, V, scalar, dtype, atol, rtol, bias, ignore_index, beta, alpha
):
    B = 2 * B  # cpo loss requires B to be even

    _input = torch.randn(B, T, H, device="cuda", dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device="cuda",
        dtype=torch.long,
    )
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    _weight = torch.randn(V, H, device="cuda", dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device="cuda", dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    loss1 = HFCPOLoss(ignore_index=ignore_index, beta=beta).get_batch_loss_metrics(
        input1, weight1, target, bias1, alpha=alpha
    )
    loss2 = LigerFusedLinearCPOFunction.apply(
        input2, weight2, target, bias2, ignore_index, beta, alpha, True
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
