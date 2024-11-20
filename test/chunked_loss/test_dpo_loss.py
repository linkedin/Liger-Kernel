from test.utils import HFAlignmentLoss, assert_verbose_allclose, set_seed

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction

# set random seed globally
set_seed()


class HF_DPO_Loss(HFAlignmentLoss):
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
    ):
        """Compute DPO loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the DPO loss for each example in the batch.
        """
        # Derived from https://huggingface.co/papers/2305.18290
        logits_diff = self.beta * (policy_chosen_logps - policy_rejected_logps)
        losses = -F.logsigmoid(logits_diff)
        return losses


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
        (1.0, torch.float32, 2e-2, 5e-1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
def test_correctness(B, T, H, V, scalar, dtype, atol, rtol, bias, ignore_index, beta):
    B = 2 * B  # dpo loss requires B to be even

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

    loss1 = HF_DPO_Loss(ignore_index=ignore_index, beta=beta).get_batch_loss_metrics(
        input1, weight1, target, bias1
    )
    loss2 = LigerFusedLinearDPOFunction.apply(
        input2, weight2, target, bias2, ignore_index, beta, True
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
