from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearCPOLoss
from liger_kernel.chunked_loss.cpo_loss import LigerFusedLinearCPOFunction
from liger_kernel.chunked_loss.functional import liger_fused_linear_cpo
from liger_kernel.utils import infer_device
from test.utils import HFAlignmentLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

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
        simpo_gamma: float = 0.5,
        loss_type: str = "sigmoid",
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
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "simpo":
            logits = logits - (self.simpo_gamma / self.beta)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid']")

        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps

        return losses, chosen_rewards, rejected_rewards


class TorchLMHeadCPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        simpo_gamma: float = 0.5,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.cpo_loss = HFCPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            simpo_gamma=simpo_gamma,
        ).get_batch_loss_metrics
        self.average_log_prob = loss_type == "simpo"

    def forward(self, x, y):
        return self.cpo_loss(self.lin.weight, x, y, self.lin.bias, average_log_prob=self.average_log_prob)


class LigerLMHeadCPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.cpo_loss = LigerFusedLinearCPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )

    def forward(self, x, y):
        return self.cpo_loss(self.lin.weight, x, y, self.lin.bias)


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
        (1.0, torch.bfloat16, 5e-2, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ignore_index, beta, alpha", [(-100, 0.1, 1.0), (42, 0.2, 0.85)])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    ignore_index,
    beta,
    alpha,
    label_smoothing,
):
    B = 2 * B  # cpo loss requires B to be even

    torch_lm_head_cpo = TorchLMHeadCPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        label_smoothing=label_smoothing,
    )
    liger_lm_head_cpo = LigerLMHeadCPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    torch_lm_head_cpo.lin.weight.data = liger_lm_head_cpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_cpo.lin.bias.data = liger_lm_head_cpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)

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
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    loss1, aggregated_aux_outputs1 = torch_lm_head_cpo(input1, target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_cpo(input2, target)

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
        torch_lm_head_cpo.lin.weight.grad,
        liger_lm_head_cpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_cpo.lin.bias.grad,
            liger_lm_head_cpo.lin.bias.grad,
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

    loss1, aggregated_aux_outputs1 = LigerFusedLinearCPOFunction.apply(input1, weight1, target, bias1)
    loss2, aggregated_aux_outputs2 = liger_fused_linear_cpo(input2, weight2, target, bias2)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
