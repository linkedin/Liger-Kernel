from test.utils import HFAlignmentLossKTO, assert_verbose_allclose, set_seed
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.kto_loss import LigerFusedLinearKTOFunction

# set random seed globally
set_seed()


class HFKTOLoss(HFAlignmentLossKTO):
    """
    HF's implementation of KTO loss in TRL. https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        simpo_gamma: float = 0.5,
        loss_type: str = "kto",
    ):
        super().__init__(alpha=alpha, beta=beta, ignore_index=ignore_index)
        # Sigmoid defaults to the CPO loss defined in the paper listed above.
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.simpo_gamma = simpo_gamma

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The KL tensor contains the detached KL divergence estimate between the policy and reference models.
        """

        kl = torch.zeros(1).to(policy_chosen_logps.device)

        # Chosen losses
        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1)

            if self.loss_type == "kto":
                # Eqn (7) of the KTO paper (https://huggingface.co/papers/2402.01306)
                chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            elif self.loss_type == "apo_zero_unpaired":
                # Unpaired variant of Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                chosen_losses = 1 - F.sigmoid(self.beta * chosen_logratios)

            chosen_rewards = self.beta * chosen_logratios.detach()

        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([])#.to(self.accelerator.device)
            chosen_rewards = torch.Tensor([])#.to(self.accelerator.device)

        # Rejected losses
        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1)

            if self.loss_type == "kto":
                rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            elif self.loss_type == "apo_zero_unpaired":
                rejected_losses = F.sigmoid(self.beta * rejected_logratios)

            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([])#.to(self.accelerator.device)
            rejected_rewards = torch.Tensor([])#.to(self.accelerator.device)
        desirable_weight = 1.0
        undesirable_weight = 1.0
        losses = torch.cat(
            (desirable_weight * chosen_losses, undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, kl


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
    reference_logps = _input.detach().clone().requires_grad_(True)

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
    labels = torch.randint(
        0,
        2,
         (
             B,
             1,
         ),
        device="cuda",
        dtype=torch.float,
    )
    reference_logps = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device="cuda",
        dtype=torch.float,
    )

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index
    #labels.view(-1)[indices_to_assign] = ignore_index

    _weight = torch.randn(V, H, device="cuda", dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device="cuda", dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None
    # _input: torch.FloatTensor,
    # weight: torch.FloatTensor,
    # target: torch.LongTensor,
    # labels: List,
    # reference_logps: np.array,
    # bias: torch.FloatTensor = None,
    # alpha: float = 1.0,
    # average_log_prob: bool = True,
    loss1 = HFKTOLoss(ignore_index=ignore_index, beta=beta).get_batch_loss_metrics(
        input1, weight1, target, labels, reference_logps,bias1,alpha=alpha
    )
    loss2 = LigerFusedLinearKTOFunction.apply(
        input2, weight2, target,labels, reference_logps, bias2, ignore_index, beta, alpha, True
    )
    print("loss1",loss1)
    print("loss2", loss2)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
