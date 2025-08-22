from typing import Optional

import pytest
import torch

from test.transformers.test_cross_entropy import CrossEntropyWithZLoss
from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()

# set random seed globally
set_seed()


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    :param label_smoothing: label_smoothing to apply on target
    :param lse_square_scale: scaler of lse ^ 2 to compute z loss

    # TODO: if we bump CI env's `transformers` version to >= 4.46, we should just directly
    # call https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L32
    # to be consistent with Hugging Face model implementation.
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = CrossEntropyWithZLoss(
            weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            return_z_loss=return_z_loss,
        )
        self.softcap = softcap

    def forward(self, x, y):
        logits = self.lin(x).to(torch.float32)
        if self.softcap is not None and self.softcap != 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ce_weight: Optional[torch.FloatTensor] = None,
        bias: bool = False,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype=None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y, self.lin.bias)


#############################################################################
# Test the correctness of the fused linear cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "reduction, scalar, dtype, atol, rtol",
    [
        ("mean", 1.0, torch.bfloat16, 5e-3, 5e-2),
        ("mean", 1.0, torch.float32, 1e-5, 5e-4),
        ("sum", 1.0, torch.bfloat16, 5e-0, 5e1),
        ("sum", 1.0, torch.float32, 1e-3, 5e-2),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "has_ce_weight, label_smoothing, ignore_index, lse_square_scale, softcap, return_z_loss, accum_dtype",
    [
        (False, 0, -100, 0, None, False, None),
        # Pass non-default values once to ensure all params work along
        (True, 0.1, 42, 1e-4, 30.0, True, torch.float32),
    ],
)
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    bias,
    has_ce_weight,
    lse_square_scale,
    label_smoothing,
    ignore_index,
    reduction,
    softcap,
    return_z_loss,
    accum_dtype,
    atol,
    rtol,
):
    if has_ce_weight:
        ce_weight = torch.rand(V, device=device, dtype=torch.float32)
    else:
        ce_weight = None
    torch_lm_head_ce = TorchLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        ce_weight=ce_weight,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
        softcap=softcap,
        return_z_loss=return_z_loss,
        dtype=dtype,
    ).to(device)
    liger_lm_head_ce = LigerLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        ce_weight=ce_weight,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
        softcap=softcap,
        return_z_loss=return_z_loss,
        dtype=dtype,
        accum_dtype=accum_dtype,
    ).to(device)

    # init the linear in all CEs with the same weights
    torch_lm_head_ce.lin.weight.data = liger_lm_head_ce.lin.weight.data = torch.rand(V, H, device=device, dtype=dtype)

    if bias:
        torch_lm_head_ce.lin.bias.data = liger_lm_head_ce.lin.bias.data = torch.rand(V, device=device, dtype=dtype)

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    if return_z_loss:
        output1, z_output1 = torch_lm_head_ce(_input1, target)
        output2, z_output2 = liger_lm_head_ce(_input2, target)
    else:
        output1 = torch_lm_head_ce(_input1, target)
        output2 = liger_lm_head_ce(_input2, target)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    if return_z_loss:
        assert_verbose_allclose(z_output1, z_output2, atol=atol, rtol=rtol)

    grad_output = torch.ones_like(output1)
    output1.backward(gradient=grad_output)
    output2.backward(gradient=grad_output)

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_ce.lin.weight.grad,
        liger_lm_head_ce.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_lm_head_ce.lin.bias.grad,
            liger_lm_head_ce.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        # weird shapes
        (9, 7, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("ce_weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, bias, ce_weight, atol, rtol):
    _input = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype) if bias else None

    ce_weight = torch.randn(V, device=device) if ce_weight else None
    y1, z1 = liger_fused_linear_cross_entropy(
        input=x1,
        weight=weight,
        target=target,
        bias=bias,
        ce_weight=ce_weight,
        ignore_index=-100,
        lse_square_scale=1e-4,
        label_smoothing=0.1,
        reduction="mean",
        softcap=30.0,
        return_z_loss=True,
        accum_dtype=torch.float32,
    )
    y2, z2 = LigerFusedLinearCrossEntropyFunction.apply(
        x2, weight, target, bias, ce_weight, -100, 1e-4, 0.1, "mean", 30.0, True, torch.float32
    )

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)
    assert torch.allclose(z1, z2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (4, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "bias, cast_dtype, atol, rtol",
    [
        (True, torch.bfloat16, 5e-3, 5e-2),
        (True, torch.float16, 5e-3, 5e-2),
        (False, torch.bfloat16, 5e-3, 5e-2),
        (False, torch.float16, 5e-3, 5e-2),
    ],
)
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_amp(B, T, H, V, bias, cast_dtype, accum_dtype, atol, rtol):
    dtype = torch.float32
    torch_lm_head_ce = TorchLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        label_smoothing=0.0,
        reduction="mean",
        dtype=dtype,
    ).to(device)
    liger_lm_head_ce = LigerLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        label_smoothing=0.0,
        reduction="mean",
        dtype=dtype,
        accum_dtype=accum_dtype,
    ).to(device)

    # init the linear in all CEs with the same weights
    torch_lm_head_ce.lin.weight.data = liger_lm_head_ce.lin.weight.data = torch.rand(V, H, device=device, dtype=dtype)

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype)
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    with torch.autocast(device_type=device, dtype=cast_dtype):
        output1 = torch_lm_head_ce(_input1, target)
        output2 = liger_lm_head_ce(_input2, target)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    with torch.autocast(device_type=device, dtype=cast_dtype):
        output1.backward()
        output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_ce.lin.weight.grad,
        liger_lm_head_ce.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )


def test_correctness_token_scaling():
    """Test that token scaling produces the correct loss values and gradients."""
    B, T, H, V = 2, 4, 8, 16
    dtype = torch.float32

    # Create inputs
    _input = torch.randn(B * T, H, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Create weights
    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype)

    # Test using functional API with token scaling
    loss_scaled = liger_fused_linear_cross_entropy(
        input=_input,
        weight=weight,
        target=target,
        bias=bias,
        ignore_index=-100,
        reduction="none",  # Use "none" to get per-token losses
        use_token_scaling=True,
    )

    # Compare with manual implementation
    # Compute logits
    logits = _input @ weight.t()
    if bias is not None:
        logits = logits + bias

    # Compute standard cross entropy loss per token
    ce_loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100, reduction="none")

    # Compute predicted probabilities for target tokens
    pred_probs = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(-1)).squeeze(-1).detach()

    # Scale by predicted probabilities
    expected_loss = ce_loss * pred_probs

    # Check that losses are close
    assert torch.allclose(loss_scaled, expected_loss, atol=1e-4, rtol=1e-4)

    # Test gradients
    loss_scaled.sum().backward(retain_graph=True)
    grad_scaled = _input.grad.clone()
    _input.grad.zero_()

    expected_loss.sum().backward(retain_graph=True)
    grad_expected = _input.grad.clone()
    _input.grad.zero_()

    # Check that gradients are close
    assert torch.allclose(grad_scaled, grad_expected, atol=1e-4, rtol=1e-4)


def test_correctness_token_scaling_consistency():
    """Test that token scaling is consistent between functional and module APIs."""
    B, T, H, V = 2, 4, 8, 16
    dtype = torch.float32

    # Create inputs
    _input = torch.randn(B * T, H, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Create weights
    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype)

    # Test functional API
    loss_functional = liger_fused_linear_cross_entropy(
        input=_input,
        weight=weight,
        target=target,
        bias=bias,
        ignore_index=-100,
        reduction="sum",
        use_token_scaling=True,
    )

    # Test module API
    ce_loss_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
        use_token_scaling=True,
    )

    loss_module = ce_loss_module(weight, _input, target, bias)

    # Check that losses are identical
    assert torch.allclose(loss_functional, loss_module, atol=1e-6, rtol=1e-6)

    # Test gradients
    loss_functional.backward(retain_graph=True)
    grad_functional = _input.grad.clone()
    _input.grad.zero_()

    loss_module.backward(retain_graph=True)
    grad_module = _input.grad.clone()
    _input.grad.zero_()

    # Check that gradients are identical
    assert torch.allclose(grad_functional, grad_module, atol=1e-6, rtol=1e-6)


def test_correctness_token_scaling_functional():
    """Test token scaling using the functional API."""
    B, T, H, V = 2, 4, 8, 16
    dtype = torch.float32

    # Create inputs
    _input = torch.randn(B * T, H, device=device, dtype=dtype)
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Create weights
    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype)

    # Test using functional API with token scaling
    y1 = liger_fused_linear_cross_entropy(
        input=x1,
        weight=weight,
        target=target,
        bias=bias,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="sum",  # Use sum for easier verification
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=True,
    )

    # Compare with manual implementation
    # Compute logits
    logits = x2 @ weight.t()
    if bias is not None:
        logits = logits + bias

    # Compute softmax probabilities
    probs = torch.softmax(logits.detach(), dim=-1)  # Detach to avoid gradient flow

    # Get predicted probabilities for target tokens
    pred_probs = torch.gather(probs, -1, target.unsqueeze(-1)).squeeze(-1)

    # Compute standard cross entropy loss
    ce_loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100, reduction="none")

    # Scale by predicted probabilities
    scaled_loss = ce_loss * pred_probs

    # Sum over all tokens
    y2 = scaled_loss.sum()

    # Check that losses are close
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)

    # Test gradients
    y1.backward()
    y2.backward()

    # Check that gradients are close
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


def test_correctness_token_scaling_module():
    """Test token scaling using the module API."""
    B, T, H, V = 2, 4, 8, 16
    dtype = torch.float32

    # Create inputs
    _input = torch.randn(B * T, H, device=device, dtype=dtype)
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Create module with token scaling
    ce_loss = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
        use_token_scaling=True,
    )

    # Create weights
    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype)

    # Test using module API with token scaling
    y1 = ce_loss(weight, x1, target, bias)

    # Compare with manual implementation
    # Compute logits
    logits = x2 @ weight.t()
    if bias is not None:
        logits = logits + bias

    # Compute softmax probabilities
    probs = torch.softmax(logits.detach(), dim=-1)  # Detach to avoid gradient flow

    # Get predicted probabilities for target tokens
    pred_probs = torch.gather(probs, -1, target.unsqueeze(-1)).squeeze(-1)

    # Compute standard cross entropy loss
    ce_loss_manual = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100, reduction="none")

    # Scale by predicted probabilities
    scaled_loss = ce_loss_manual * pred_probs

    # Sum over all tokens
    y2 = scaled_loss.sum()

    # Check that losses are close
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)

    # Test gradients
    y1.backward()
    y2.backward()

    # Check that gradients are close
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
