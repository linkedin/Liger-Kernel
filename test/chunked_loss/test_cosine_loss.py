import pytest
import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityFunction
from liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_cosine
from liger_kernel.utils import infer_device
from test.utils import HFDistillationLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()
set_seed()


class HFCosineLoss(HFDistillationLoss):
    """
    implementation of a distilltion loss using cosine similarity
    """

    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__(
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            ignore_index=ignore_index,
            temperature=temperature,
        )

    def distillation_loss(self, student_logits, teacher_logits, beta=1.0):
        # Compute normalized logits
        print(f"student_logits.shape: {student_logits.shape}")
        student_norm = F.normalize(student_logits, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=-1)
        # cosine_sim = (student_norm * teacher_norm).sum(dim=1).mean()
        # loss =  beta * (1 - cosine_sim)
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)

        loss = beta * (1 - cosine_sim)
        return loss.mean()


class TorchCosineLoss(torch.nn.Module):
    """
    Reference implementation for Cosine Similarity Loss using standard torch operations.
    Computes the loss as 1 - cosine_similarity averaged over all tokens.
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
        beta: float = 1.0,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        # Note: student inputs are expected to have hidden size H//2 while teacher inputs have H.
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)
        self.beta = beta
        self.cosine = HFCosineLoss(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        ).get_batch_loss_metrics

    def forward(self, student_input, teacher_input, target):
        loss = self.cosine(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
            beta=self.beta,
        )
        return loss


class LigerCosineLoss(torch.nn.Module):
    """
    Liger implementation that uses fused cosine similarity loss.
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
        compiled: bool = True,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.chunked_cosine = LigerFusedLinearCosineSimilarityLoss(
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            beta=beta,
            ignore_index=ignore_index,
            temperature=temperature,
            compiled=compiled,
            chunk_size=chunk_size,
        )
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)

    def forward(self, student_input, teacher_input, target):
        return self.chunked_cosine(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
        )


###############################################################################
# Test correctness of the module implementations
###############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 32, 128),  # H must be even
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
    B, T, H, V, scalar, dtype, atol, rtol, bias, temperature, weight_hard_loss, weight_soft_loss, beta
):
    torch_cosine = TorchCosineLoss(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        temperature=temperature,
        beta=beta,
    )
    liger_cosine = LigerCosineLoss(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        temperature=temperature,
        beta=beta,
    )
    # Ensure both implementations start with the same weights and biases.
    torch_cosine.student_lin.weight.data = liger_cosine.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_cosine.teacher_lin.weight.data = liger_cosine.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )
    if bias:
        torch_cosine.student_lin.bias.data = liger_cosine.student_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )
        torch_cosine.teacher_lin.bias.data = liger_cosine.teacher_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.clone().detach().requires_grad_(True)
    student_input2 = _tensor.clone().detach().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    # Dummy target (not used in cosine computation)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    loss1 = torch_cosine(student_input1, teacher_input, target)
    loss2 = liger_cosine(student_input2, teacher_input, target)
    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    print("loss1 shape : {loss1.shape}")
    loss2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_cosine.student_lin.weight.grad, liger_cosine.student_lin.weight.grad, atol=atol, rtol=rtol
    )
    if bias:
        assert_verbose_allclose(
            torch_cosine.student_lin.bias.grad, liger_cosine.student_lin.bias.grad, atol=atol, rtol=rtol
        )


###############################################################################
# Test correctness of the functional interface
###############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (9, 7, 40, 40),  # H must be even
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
    [
        (1.0, 0.5, 0.5, 0.5, -100),
        (2.0, 0.1, 0.9, 0.5, 42),
    ],
)
def test_correctness_functional(
    B, T, H, V, scalar, dtype, bias, weight_hard_loss, weight_soft_loss, beta, ignore_index, temperature, atol, rtol
):
    # Prepare weights and biases for functional testing.
    student_weight1 = torch.rand(V, H // 2, device=device, dtype=dtype).detach().clone().requires_grad_(True)
    student_weight2 = student_weight1.clone().detach().requires_grad_(True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    if bias:
        student_bias1 = torch.rand(V, device=device, dtype=dtype).detach().clone().requires_grad_(True)
        student_bias2 = student_bias1.clone().detach().requires_grad_(True)
        teacher_bias = torch.rand(V, device=device, dtype=dtype)
    else:
        student_bias1 = student_bias2 = teacher_bias = None

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.clone().detach().requires_grad_(True)
    student_input2 = _tensor.clone().detach().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Functional call using the fused cosine similarity function
    output1 = liger_fused_linear_cosine(
        student_input1,
        student_weight1,
        teacher_input,
        teacher_weight,
        target,
        student_bias1,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
        True,
        1024,
    )
    output2 = LigerFusedLinearCosineSimilarityFunction.apply(
        student_input2,
        student_weight2,
        teacher_input,
        teacher_weight,
        target,
        student_bias2,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
        True,
        1024,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(student_weight1.grad, student_weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(student_bias1.grad, student_bias2.grad, atol=atol, rtol=rtol)
