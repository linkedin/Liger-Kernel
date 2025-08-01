import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
from liger_kernel.chunked_loss.functional import liger_fused_linear_dpo
from liger_kernel.utils import infer_device
from test.utils import HFAlignmentLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class HFDPOLoss(HFAlignmentLoss):
    """
    Implementation of the Direct Preference Optimization (DPO) loss,
    adapted from Hugging Face's implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute DPO loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the DPO loss for each example in the batch.
        """
        # Derived from https://huggingface.co/papers/2305.18290
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits_diff)
        return losses, chosen_rewards, rejected_rewards


class HFAPOZeroLoss(HFAlignmentLoss):
    """
    Implementation of the APO-zero loss.
    Reference: https://huggingface.co/papers/2408.06266
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute APO-zero loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the APO-zero loss for each example in the batch.
        """
        # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # Use this loss when you believe the chosen outputs are better than your model's default output
        losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
        losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
        losses = losses_chosen + losses_rejected

        return losses, chosen_rewards, rejected_rewards


class HFAPODownLoss(HFAlignmentLoss):
    """
    Implementation of the APO-down loss.
    Reference: https://huggingface.co/papers/2408.06266
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute APO-down loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the APO-down loss for each example in the batch.
        """
        # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # Use this loss when you believe the chosen outputs are worse than your model's default output.
        # Decrease chosen likelihood and decrease rejected likelihood more
        losses_chosen = F.sigmoid(self.beta * chosen_logratios)
        losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
        losses = losses_chosen + losses_rejected

        return losses, chosen_rewards, rejected_rewards


class HFSPPPOHARDLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        a = policy_chosen_logps - ref_chosen_logps
        b = policy_rejected_logps - ref_rejected_logps
        losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        return losses, chosen_rewards, rejected_rewards


class HFNCAPAIRLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        losses = (
            -F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) - 0.5 * F.logsigmoid(-rejected_rewards)
        )

        return losses, chosen_rewards, rejected_rewards


class TorchLMHeadDPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.dpo_loss = HFDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.dpo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadAPOZero(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.apo_loss = HFAPOZeroLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.apo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadAPODown(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.apo_loss = HFAPODownLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.apo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadSPPOHARD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.sppo_hard = HFSPPPOHARDLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.sppo_hard(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadNCAPAIR(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.nca_pair = HFNCAPAIRLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.nca_pair(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class LigerLMHeadDPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.dpo_loss = LigerFusedLinearDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
            average_log_prob=True,
            loss_type=loss_type,
        )

    def forward(self, x, ref_x, y):
        return self.dpo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
        )


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
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
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
    ref_bias,
    compute_nll_loss,
    ignore_index,
    beta,
):
    B = 2 * B  # dpo loss requires B to be even

    torch_lm_head_dpo = TorchLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_dpo = LigerLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_dpo.lin.weight.data = liger_lm_head_dpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_dpo.ref_lin.weight.data = liger_lm_head_dpo.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_dpo.lin.bias.data = liger_lm_head_dpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_dpo.ref_lin.bias.data = liger_lm_head_dpo.ref_lin.bias.data = torch.randn(
            V, device=device, dtype=dtype
        )

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

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

    loss1, aggregated_aux_outputs1 = torch_lm_head_dpo(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_dpo(input2, ref_input, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        if i > 4 and dtype == torch.bfloat16:
            # numerical instability in bf16 for chosen_rewards and rejected_rewards
            # temporary fix. TODO: investigate how to reduce numercial instabiltiy issue
            assert_verbose_allclose(
                aggregated_aux_outputs1[i],
                aggregated_aux_outputs2[i],
                atol=5e-1,
                rtol=rtol,
            )
            continue
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
        torch_lm_head_dpo.lin.weight.grad,
        liger_lm_head_dpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_dpo.lin.bias.grad,
            liger_lm_head_dpo.lin.bias.grad,
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
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("compute_nll_loss", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, compute_nll_loss):
    B = 2 * B

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

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

    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_weight1 = _ref_weight.detach().clone().requires_grad_(True)
    ref_weight2 = _ref_weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    _ref_bias = torch.randn(V, device=device, dtype=dtype) if ref_bias else None
    ref_bias1 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None
    ref_bias2 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None

    loss1, aggregated_aux_outputs1 = LigerFusedLinearDPOFunction.apply(
        input1,
        weight1,
        target,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        -100,
        0.1,
        compute_nll_loss,
    )
    loss2, aggregated_aux_outputs2 = liger_fused_linear_dpo(
        input2,
        weight2,
        target,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
        -100,
        0.1,
        compute_nll_loss,
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


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
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
@pytest.mark.parametrize("loss_type", ["apo_zero", "apo_down", "sppo_hard", "nca_pair"])
def test_correctness_apo_loss_types(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    ref_bias,
    compute_nll_loss,
    ignore_index,
    beta,
    loss_type,
):
    B = 2 * B  # dpo loss requires B to be even

    # Select the appropriate HF reference implementation
    if loss_type == "apo_zero":
        torch_lm_head = TorchLMHeadAPOZero
    elif loss_type == "apo_down":
        torch_lm_head = TorchLMHeadAPODown
    elif loss_type == "sppo_hard":
        torch_lm_head = TorchLMHeadSPPOHARD
    elif loss_type == "nca_pair":
        torch_lm_head = TorchLMHeadNCAPAIR
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    torch_lm_head_apo = torch_lm_head(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_apo = LigerLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
        loss_type=loss_type,
    )

    torch_lm_head_apo.lin.weight.data = liger_lm_head_apo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_apo.ref_lin.weight.data = liger_lm_head_apo.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_apo.lin.bias.data = liger_lm_head_apo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_apo.ref_lin.bias.data = liger_lm_head_apo.ref_lin.bias.data = torch.randn(
            V, device=device, dtype=dtype
        )

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

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

    loss1, aggregated_aux_outputs1 = torch_lm_head_apo(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_apo(input2, ref_input, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        if i > 4 and dtype == torch.bfloat16:
            # numerical instability in bf16 for chosen_rewards and rejected_rewards
            # temporary fix. TODO: investigate how to reduce numerical instability issue
            assert_verbose_allclose(
                aggregated_aux_outputs1[i],
                aggregated_aux_outputs2[i],
                atol=5e-1,
                rtol=rtol,
            )
            continue
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
        torch_lm_head_apo.lin.weight.grad,
        liger_lm_head_apo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_apo.lin.bias.grad,
            liger_lm_head_apo.lin.bias.grad,
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
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("loss_type", ["apo_zero", "apo_down", "sppo_hard", "nca_pair"])
def test_correctness_functional_apo_loss_types(
    B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, compute_nll_loss, loss_type
):
    B = 2 * B

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

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

    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_weight1 = _ref_weight.detach().clone().requires_grad_(True)
    ref_weight2 = _ref_weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    _ref_bias = torch.randn(V, device=device, dtype=dtype) if ref_bias else None
    ref_bias1 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None
    ref_bias2 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None

    # Call with loss_type parameter for LigerFusedLinearDPOFunction
    loss1, aggregated_aux_outputs1 = LigerFusedLinearDPOFunction.apply(
        input1,
        weight1,
        target,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        -100,
        0.1,
        compute_nll_loss,
        True,  # compiled
        True,  # use_ref_model
        False,  # average_log_prob
        1,  # chunk_size
        loss_type,  # loss_type
    )

    # For comparison, create a LigerFusedLinearDPOLoss with the loss_type
    dpo_loss_fn = LigerFusedLinearDPOLoss(
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=compute_nll_loss,
        loss_type=loss_type,
    )

    loss2, aggregated_aux_outputs2 = dpo_loss_fn(
        weight2,
        input2,
        target,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


def test_invalid_loss_type():
    """Test that invalid loss types raise ValueError"""
    with pytest.raises(ValueError, match="Unsupported loss_type"):
        LigerFusedLinearDPOLoss(loss_type="invalid_loss_type")

    # Test that valid loss types don't raise errors
    valid_loss_types = ["sigmoid", "apo_zero", "apo_down", "sppo_hard", "nca_pair"]
    for loss_type in valid_loss_types:
        # Should not raise an exception
        loss_fn = LigerFusedLinearDPOLoss(loss_type=loss_type)
        assert loss_fn.loss_type == loss_type
