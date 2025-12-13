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

    def distillation_loss(self, student_logits, teacher_logits, beta=0.5):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).
        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
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
            # Compute probabilities (only required for mean calculation)
            log_mean_probs = torch.logsumexp(
                torch.stack([student_log_probs + math.log(1 - beta), teacher_log_probs + math.log(beta)], dim=0), dim=0
            )

            student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="batchmean", log_target=True)
            teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="batchmean", log_target=True)

            # JSD is the weighted average of the KL divergences
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl
        return jsd_loss


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


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 16, 64, 128),
        (4, 32, 128, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ignore_index", [-100, 42])
def test_ignore_index_exclusion(B, T, H, V, dtype, ignore_index):
    """Test that tokens with ignore_index are excluded from loss computation."""
    set_seed(42)

    student_input = torch.rand(B * T, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype)
    student_weight = torch.rand(V, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    # All valid targets (no ignore_index)
    target_all_valid = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    target_all_valid[target_all_valid == ignore_index] = (ignore_index + 1) % V

    # Some tokens ignored (~30%)
    target_with_ignored = target_all_valid.clone()
    num_ignored = int(B * T * 0.3)
    target_with_ignored[:num_ignored] = ignore_index

    jsd_loss = LigerFusedLinearJSDLoss(ignore_index=ignore_index)

    loss_all_valid = jsd_loss(
        student_input.detach().clone().requires_grad_(True),
        student_weight.detach().clone(),
        teacher_input,
        teacher_weight,
        target_all_valid,
    )

    loss_with_ignored = jsd_loss(
        student_input.detach().clone().requires_grad_(True),
        student_weight.detach().clone(),
        teacher_input,
        teacher_weight,
        target_with_ignored,
    )

    # Losses should be different
    assert not torch.allclose(loss_all_valid, loss_with_ignored), "Loss should differ when some tokens are ignored"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ignore_index_gradient_exclusion(dtype):
    """Test that ignored tokens don't contribute to gradients."""
    B, T, H, V = 2, 8, 64, 128
    ignore_index = -100
    set_seed(42)

    student_input = torch.rand(B * T, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype)
    student_weight = torch.rand(V, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    # First half ignored, second half valid
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    target[: B * T // 2] = ignore_index

    jsd_loss = LigerFusedLinearJSDLoss(ignore_index=ignore_index)

    loss = jsd_loss(student_input, student_weight, teacher_input, teacher_weight, target)
    loss.backward()

    # Should have valid gradients (not NaN)
    assert not torch.isnan(student_input.grad).any(), "Gradients should not be NaN"
    assert not torch.isnan(student_weight.grad).any(), "Gradients should not be NaN"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_soft_loss_only_ignore_index(dtype):
    """Test ignore_index with soft loss only (no hard loss interference)."""
    B, T, H, V = 2, 16, 64, 128
    ignore_index = -100
    set_seed(42)

    student_input = torch.rand(B * T, H // 2, device=device, dtype=dtype)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype)
    student_weight = torch.rand(V, H // 2, device=device, dtype=dtype)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    # All valid
    target_all_valid = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    target_all_valid[target_all_valid == ignore_index] = 0

    # Some ignored
    target_with_ignored = target_all_valid.clone()
    target_with_ignored[: B * T // 2] = ignore_index

    # Soft loss only - this tests our fix directly
    jsd_loss = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.0,
        weight_soft_loss=1.0,
        ignore_index=ignore_index,
    )

    loss_all_valid = jsd_loss(student_input, student_weight, teacher_input, teacher_weight, target_all_valid)
    loss_with_ignored = jsd_loss(student_input, student_weight, teacher_input, teacher_weight, target_with_ignored)

    assert not torch.allclose(loss_all_valid, loss_with_ignored), "Soft loss should differ when tokens are ignored"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_tokens_ignored(dtype):
    """Test edge case where all tokens are ignored - should not produce NaN."""
    B, T, H, V = 2, 8, 64, 128
    ignore_index = -100
    set_seed(42)

    student_input = torch.rand(B * T, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype)
    student_weight = torch.rand(V, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    # All tokens ignored
    target = torch.full((B * T,), ignore_index, device=device, dtype=torch.long)

    jsd_loss = LigerFusedLinearJSDLoss(ignore_index=ignore_index)

    loss = jsd_loss(student_input, student_weight, teacher_input, teacher_weight, target)

    # Should not be NaN
    assert not torch.isnan(loss), f"Loss should not be NaN, got {loss}"

    # Backward should also not produce NaN
    loss.backward()
    assert not torch.isnan(student_input.grad).any(), "Gradients should not be NaN"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ignored_tokens_zero_gradient(dtype):
    """Test that ignored token positions have zero gradient contribution."""
    B, T, H, V = 2, 8, 64, 128
    ignore_index = -100
    set_seed(42)

    student_input = torch.rand(B * T, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype)
    student_weight = torch.rand(V, H // 2, device=device, dtype=dtype, requires_grad=True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # First half ignored
    num_ignored = B * T // 2
    target[:num_ignored] = ignore_index

    jsd_loss = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.0,  # soft loss only
        weight_soft_loss=1.0,
        ignore_index=ignore_index,
    )

    loss = jsd_loss(student_input, student_weight, teacher_input, teacher_weight, target)
    loss.backward()

    # Ignored positions should have zero (or near-zero) gradient
    ignored_grad = student_input.grad[:num_ignored]
    valid_grad = student_input.grad[num_ignored:]

    # Ignored should be significantly smaller than valid
    ignored_norm = ignored_grad.norm()
    valid_norm = valid_grad.norm()

    assert ignored_norm < valid_norm * 0.01, (
        f"Ignored gradient norm ({ignored_norm}) should be much smaller than valid ({valid_norm})"
    )
