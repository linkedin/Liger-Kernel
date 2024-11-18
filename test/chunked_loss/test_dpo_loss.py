from test.utils import assert_verbose_allclose, set_seed
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction

# set random seed globally
set_seed()


class HF_DPO_Loss:
    """
    Implementation of the Direct Preference Optimization (DPO) loss,
    adapted from Hugging Face's implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
    """

    def __init__(self, ignore_index: int = -100, beta: float = 0.1):
        self.ignore_index = ignore_index
        self.beta = beta

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of ignore_index are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        loss_mask = labels != self.ignore_index

        # dummy token; we'll ignore the losses on these tokens later
        labels = torch.where(labels == self.ignore_index, 0, labels)

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
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

    def concatenated_forward(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        len_chosen = _input.shape[0] // 2

        outputs = _input @ weight.t()
        if bias is not None:
            outputs = outputs + bias
        all_logits = outputs.float()

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = target
        chosen_nll_loss = cross_entropy_loss(
            all_logits[:len_chosen], labels[:len_chosen]
        )

        all_logps = self.get_batch_logps(
            all_logits,
            target,
            average_log_prob=True,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    def get_batch_loss_metrics(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""

        forward_output = self.concatenated_forward(_input, weight, target, bias)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]

        losses = self.dpo_loss(policy_chosen_logps, policy_rejected_logps)
        # full DPO loss
        loss = policy_nll_loss - losses.mean()
        return loss


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
