import math

import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_jsd
from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction
from liger_kernel.utils import infer_device
from test.utils import HFDistillationLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class HFJSDLoss(HFDistillationLoss):
    """
    Naive implementation of a distillation loss using Jensen-Shannon Divergence (JSD).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        ignore_index: int = -100,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
    ):
        super().__init__(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        )

    def distillation_loss(self, student_logits, teacher_logits, target=None, ignore_index=-100, beta=0.5):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len, vocab_size).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len, vocab_size).
            target (torch.Tensor): Target labels for masking. Shape: (batch_size * seq_len,).
            ignore_index (int): Index to ignore in loss computation.
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            log_mean_probs = torch.logsumexp(
                torch.stack([student_log_probs + math.log(1 - beta), teacher_log_probs + math.log(beta)], dim=0), dim=0
            )
            student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="none", log_target=True)
            teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="none", log_target=True)
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl

        # Sum over vocab dimension
        jsd_loss = jsd_loss.sum(dim=-1)

        # Apply ignore_index mask
        if target is not None:
            mask = target != ignore_index
            jsd_loss = jsd_loss * mask.float()
            num_valid_tokens = mask.sum().clamp_min(1)
            return jsd_loss.sum() / num_valid_tokens

        return jsd_loss.sum()


class TorchLMHeadJSD(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.
    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param weight_hard_loss: weight_hard_loss
    :param weight_soft_loss: weight_soft_loss
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool,
        device: torch.device,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        # smaller student model weights
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)
        self.beta = beta
        self.jsd = HFJSDLoss(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        ).get_batch_loss_metrics

    def forward(self, student_input, teacher_input, target):
        jsd_loss = self.jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
            beta=self.beta,
        )
        return jsd_loss

    def backward_with_grad_and_value(self, student_input, teacher_input, target):
        """
        Compute gradients using grad_and_value on NPU to match Liger implementation.
        This method is used in tests on NPU devices to ensure consistency.
        """
        # Use grad_and_value to compute gradients and loss
        if self.student_lin.bias is not None:

            def loss_fn(student_input, student_weight, student_bias):
                return self.jsd(
                    student_input,
                    student_weight,
                    teacher_input,
                    self.teacher_lin.weight,
                    target,
                    student_bias,
                    self.teacher_lin.bias,
                    beta=self.beta,
                )

            (grad_input, grad_weight, grad_bias), loss = torch.func.grad_and_value(loss_fn, argnums=(0, 1, 2))(
                student_input, self.student_lin.weight, self.student_lin.bias
            )

            # Set gradients
            student_input.grad = grad_input
            self.student_lin.weight.grad = grad_weight
            self.student_lin.bias.grad = grad_bias
        else:

            def loss_fn(student_input, student_weight):
                return self.jsd(
                    student_input,
                    student_weight,
                    teacher_input,
                    self.teacher_lin.weight,
                    target,
                    None,  # student_bias is None when bias=False
                    self.teacher_lin.bias,
                    beta=self.beta,
                )

            (grad_input, grad_weight), loss = torch.func.grad_and_value(loss_fn, argnums=(0, 1))(
                student_input, self.student_lin.weight
            )

            # Set gradients
            student_input.grad = grad_input
            self.student_lin.weight.grad = grad_weight

        return loss


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool,
        device: torch.device,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        # smaller student model weights
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)
        self.chunked_jsd = LigerFusedLinearJSDLoss(
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            ignore_index=ignore_index,
            temperature=temperature,
            beta=beta,
        )

    def forward(self, student_input, teacher_input, target):
        return self.chunked_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
        )


#############################################################################
# Test the correctness of the fused linear JSD
#############################################################################


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
@pytest.mark.parametrize(
    "temperature, weight_hard_loss, weight_soft_loss, beta",
    [
        (1.0, 0.5, 0.5, 0.5),
        (2.0, 0.0, 1.0, 0.8),
        (0.5, 1.0, 0.0, 0.2),
    ],
)
@pytest.mark.parametrize("ignore_index", [-100, 42])
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
    temperature,
    weight_hard_loss,
    weight_soft_loss,
    beta,
    ignore_index,
):
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        temperature=temperature,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        beta=beta,
        ignore_index=ignore_index,
    )
    liger_lm_head_jsd = LigerLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        temperature=temperature,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        beta=beta,
        ignore_index=ignore_index,
    )

    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_jsd.student_lin.bias.data = liger_lm_head_jsd.student_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )
        torch_lm_head_jsd.teacher_lin.bias.data = liger_lm_head_jsd.teacher_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target[indices_to_assign] = ignore_index

    # Assign some random number of elements as ignore_index
    # On NPU, use grad_and_value for reference implementation to match Liger implementation
    if device == "npu":
        loss1 = torch_lm_head_jsd.backward_with_grad_and_value(student_input1, teacher_input, target)
        loss2 = liger_lm_head_jsd(student_input2, teacher_input, target)
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
        loss2.backward()
    else:
        loss1 = torch_lm_head_jsd(student_input1, teacher_input, target)
        loss2 = liger_lm_head_jsd(student_input2, teacher_input, target)
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
        loss1.backward()
        loss2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        liger_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_lm_head_jsd.student_lin.bias.grad,
            liger_lm_head_jsd.student_lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (9, 7, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-2),
        (1.0, torch.float32, 1e-4, 5e-3),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "temperature, weight_hard_loss, weight_soft_loss, beta, ignore_index",
    [(1.0, 0.5, 0.5, 0.5, -100), (2.0, 0.1, 0.9, 0.5, 42)],
)
def test_correctness_functional(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    bias,
    weight_hard_loss,
    weight_soft_loss,
    beta,
    ignore_index,
    temperature,
    atol,
    rtol,
):
    _weight = torch.rand(V, H // 2, device=device, dtype=dtype)
    student_weight1 = _weight.detach().clone().requires_grad_(True)
    student_weight2 = _weight.detach().clone().requires_grad_(True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    if bias:
        _bias = torch.rand(V, device=device, dtype=dtype)
        student_bias1 = _bias.detach().clone().requires_grad_(True)
        student_bias2 = _bias.detach().clone().requires_grad_(True)
        teacher_bias = torch.rand(V, device=device, dtype=dtype)
    else:
        student_bias1 = student_bias2 = teacher_bias = None

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = liger_fused_linear_jsd(
        student_input1,
        student_weight1,
        teacher_input,
        teacher_weight,
        label,
        student_bias1,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
    )
    output2 = LigerFusedLinearJSDFunction.apply(
        student_input2,
        student_weight2,
        teacher_input,
        teacher_weight,
        label,
        student_bias2,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(student_weight1.grad, student_weight2.grad, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(student_bias1.grad, student_bias2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_jsd_frozen_student_weight(compiled, bias):
    B, T, H, V = 2, 8, 32, 64
    torch.manual_seed(42)
    student_input = torch.randn(B * T, H, device=device, dtype=torch.float32)
    teacher_input = torch.randn(B * T, H, device=device, dtype=torch.float32)
    student_weight = torch.randn(V, H, device=device, dtype=torch.float32)
    teacher_weight = torch.randn(V, H, device=device, dtype=torch.float32)
    student_bias = torch.randn(V, device=device, dtype=torch.float32) if bias else None
    teacher_bias = torch.randn(V, device=device, dtype=torch.float32) if bias else None
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    def run(weight_requires_grad, bias_requires_grad):
        x = student_input.clone().requires_grad_(True)
        w = student_weight.clone().requires_grad_(weight_requires_grad)
        b = student_bias.clone().requires_grad_(bias_requires_grad) if bias else None
        loss_fn = LigerFusedLinearJSDLoss(
            weight_hard_loss=0.5, weight_soft_loss=0.5, beta=0.5, chunk_size=4, compiled=compiled
        )
        loss = loss_fn(x, w, teacher_input, teacher_weight, target, b, teacher_bias)
        loss.backward()
        return loss, x, w, b

    loss_full, x_full, w_full, b_full = run(True, True)
    loss_lora, x_lora, w_lora, b_lora = run(False, False)

    assert w_full.grad is not None
    assert w_lora.grad is None
    if bias:
        assert b_full.grad is not None
        assert b_lora.grad is None
    assert_verbose_allclose(loss_full, loss_lora, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(x_full.grad, x_lora.grad, atol=1e-5, rtol=1e-5)

    if bias:
        # trainable weight, frozen bias: the weight gradient is unchanged and the bias gets none
        _, x_mixed, w_mixed, b_mixed = run(True, False)
        assert b_mixed.grad is None
        assert_verbose_allclose(w_full.grad, w_mixed.grad, atol=1e-5, rtol=1e-5)
        assert_verbose_allclose(x_full.grad, x_mixed.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("compiled", [False, True])
def test_jsd_no_grad_required(compiled):
    B, T, H, V = 2, 8, 32, 64
    torch.manual_seed(42)
    student_input = torch.randn(B * T, H, device=device, dtype=torch.float32)
    teacher_input = torch.randn(B * T, H, device=device, dtype=torch.float32)
    student_weight = torch.randn(V, H, device=device, dtype=torch.float32)
    teacher_weight = torch.randn(V, H, device=device, dtype=torch.float32)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    loss_fn = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.5, weight_soft_loss=0.5, beta=0.5, chunk_size=4, compiled=compiled
    )

    loss_frozen = loss_fn(student_input, student_weight, teacher_input, teacher_weight, target)
    assert not loss_frozen.requires_grad
    loss_trainable = loss_fn(
        student_input.clone().requires_grad_(True),
        student_weight.clone().requires_grad_(True),
        teacher_input,
        teacher_weight,
        target,
    )
    assert_verbose_allclose(loss_frozen, loss_trainable, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("weight_requires_grad", [False, True])
def test_jsd_differentiates_only_params_that_require_grad(monkeypatch, weight_requires_grad):
    """A frozen student weight is left out of the differentiated arguments, so its gradient GEMM and buffer
    are skipped rather than computed and discarded."""
    argnums_seen = []
    real = torch.func.grad_and_value

    def spy(func, argnums=0, has_aux=False):
        argnums_seen.append(argnums)
        return real(func, argnums=argnums, has_aux=has_aux)

    monkeypatch.setattr(torch.func, "grad_and_value", spy)
    B, T, H, V = 2, 8, 32, 64
    torch.manual_seed(42)
    student_input = torch.randn(B * T, H, device=device, requires_grad=True)
    student_weight = torch.randn(V, H, device=device, requires_grad=weight_requires_grad)
    loss_fn = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.5, weight_soft_loss=0.5, beta=0.5, chunk_size=4, compiled=False
    )
    loss = loss_fn(
        student_input,
        student_weight,
        torch.randn(B * T, H, device=device),
        torch.randn(V, H, device=device),
        torch.randint(0, V, (B * T,), device=device),
    )
    loss.backward()
    assert argnums_seen and all(a == ((0, 1) if weight_requires_grad else (0,)) for a in argnums_seen)
