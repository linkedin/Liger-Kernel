import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearIRPOLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_irpo
from liger_kernel.chunked_loss.irpo_loss import LigerFusedLinearIRPOFunction
from liger_kernel.utils import infer_device
from test.utils import HFAlignmentLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class HFIRPOLoss(HFAlignmentLoss):
    """
    Implementation of the Iterative Reasoning Preference Optimization (IRPO) loss,
    adapted from Hugging Face's implementation.
    Reference: https://arxiv.org/abs/2404.19733 (Equation 1)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        use_ref_model: bool = True,
    ):
        super().__init__(
            beta=beta,
            alpha=alpha,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=True,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute the IRPO preference (DPO) term for a batch of policy log probabilities.
        The NLL term on the chosen response is composed by HFAlignmentLoss itself.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            ref_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            The losses tensor contains the IRPO preference term for each pair in the batch.
        """
        # DPO term of Equation (1) of https://arxiv.org/abs/2404.19733
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits_diff)
        return losses, chosen_rewards, rejected_rewards


class TorchLMHeadIRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.irpo_loss = HFIRPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            alpha=alpha,
            use_ref_model=True,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.irpo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=False,
        )


class LigerLMHeadIRPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.irpo_loss = LigerFusedLinearIRPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            alpha=alpha,
            use_ref_model=True,
            average_log_prob=False,
        )

    def forward(self, x, ref_x, y):
        return self.irpo_loss(
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
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
def test_correctness(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, ignore_index, beta):
    B = 2 * B  # irpo loss requires B to be even

    torch_lm_head_irpo = TorchLMHeadIRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        ignore_index=ignore_index,
        beta=beta,
    )
    liger_lm_head_irpo = LigerLMHeadIRPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_irpo.lin.weight.data = liger_lm_head_irpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_irpo.ref_lin.weight.data = liger_lm_head_irpo.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_irpo.lin.bias.data = liger_lm_head_irpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_irpo.ref_lin.bias.data = liger_lm_head_irpo.ref_lin.bias.data = torch.randn(
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

    loss1, aggregated_aux_outputs1 = torch_lm_head_irpo(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_irpo(input2, ref_input, target)

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
        torch_lm_head_irpo.lin.weight.grad,
        liger_lm_head_irpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_irpo.lin.bias.grad,
            liger_lm_head_irpo.lin.bias.grad,
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
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias):
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

    loss1, aggregated_aux_outputs1 = LigerFusedLinearIRPOFunction.apply(
        input1,
        weight1,
        target,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        -100,  # ignore_index
        0.1,  # beta
        1.0,  # alpha
    )
    loss2, aggregated_aux_outputs2 = liger_fused_linear_irpo(
        input2,
        weight2,
        target,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
        -100,  # ignore_index
        0.1,  # beta
        1.0,  # alpha
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_irpo_frozen_weight(compiled, has_bias):
    B = 4
    T = 16
    H = 32
    V = 64
    chunk_size = 2

    torch.manual_seed(42)
    _input = torch.randn(2 * B, T, H, device=device, dtype=torch.float32)
    _target = torch.randint(0, V, (2 * B, T), device=device, dtype=torch.long)
    _weight = torch.randn(V, H, device=device, dtype=torch.float32)
    _bias = torch.randn(V, device=device, dtype=torch.float32) if has_bias else None

    # Full fine-tuning (weight requires grad)
    input_full = _input.clone().detach().requires_grad_(True)
    weight_full = _weight.clone().detach().requires_grad_(True)
    bias_full = _bias.clone().detach().requires_grad_(True) if has_bias else None

    loss_fn_full = LigerFusedLinearIRPOLoss(beta=0.1, use_ref_model=False, chunk_size=chunk_size, compiled=compiled)
    loss_full, *aux_full = loss_fn_full(weight_full, input_full, _target, bias=bias_full)
    loss_full.backward()
    assert weight_full.grad is not None
    if has_bias:
        assert bias_full.grad is not None

    # LoRA / frozen weight (weight does not require grad)
    input_lora = _input.clone().detach().requires_grad_(True)
    weight_lora = _weight.clone().detach().requires_grad_(False)
    bias_lora = _bias.clone().detach().requires_grad_(False) if has_bias else None

    loss_fn_lora = LigerFusedLinearIRPOLoss(beta=0.1, use_ref_model=False, chunk_size=chunk_size, compiled=compiled)
    loss_lora, *aux_lora = loss_fn_lora(weight_lora, input_lora, _target, bias=bias_lora)
    loss_lora.backward()

    assert weight_lora.grad is None
    if has_bias:
        assert bias_lora.grad is None

    assert_verbose_allclose(loss_full, loss_lora, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(input_full.grad, input_lora.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_irpo_alpha_scales_nll_loss(dtype):
    """
    Verify that alpha is actually forwarded and scales the NLL component.
    The NLL term is always on for IRPO, so loss(alpha=2) must differ from loss(alpha=1).
    """
    B, T, H, V = 4, 16, 32, 64
    atol = 1e-4 if dtype == torch.float32 else 5e-2

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    def run(alpha):
        inp = _input.detach().clone().requires_grad_(True)
        w = _weight.detach().clone().requires_grad_(True)
        rw = _ref_weight.detach().clone().requires_grad_(True)
        loss_fn = LigerFusedLinearIRPOLoss(
            beta=0.1,
            alpha=alpha,
            use_ref_model=True,
            average_log_prob=False,
        )
        loss, _ = loss_fn(w, inp, target, None, _input.detach(), rw, None)
        return loss

    loss_alpha1 = run(alpha=1.0)
    loss_alpha2 = run(alpha=2.0)

    assert not torch.allclose(loss_alpha1, loss_alpha2, atol=atol), (
        f"Expected losses to differ when alpha changes, but got {loss_alpha1} vs {loss_alpha2}"
    )
