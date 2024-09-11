from test.transformers.test_cross_entropy import CrossEntropyWithZLoss
from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch

from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

# set random seed globally
set_seed()


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=bias, dtype=dtype
        )
        self.ce_loss = CrossEntropyWithZLoss(
            ignore_index=ignore_index,
            reduction="mean",
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
        )

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=bias, dtype=dtype
        )
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y, self.lin.bias)


#############################################################################
# Test the correctness of the fused linear cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        # (2, 4, 512, 512),  # The test does not work on some CI GPUs. Issue #160
        (8, 2048, 4096, 32000),  # llama2, mistral
        # Comment out to speed up testing
        # (4, 2048, 4096, 128256),  # llama3 8B
        # (4, 1024, 8192, 128256),  # llama3 70B
        (4, 423, 8192, 32000),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("lse_square_scale", [0, 1e-4])
@pytest.mark.parametrize("label_smoothing", [0, 0.1])
def test_correctness(
    B, T, H, V, scalar, dtype, bias, lse_square_scale, label_smoothing, atol, rtol
):
    device = "cuda"
    torch_lm_head_ce = TorchLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        dtype=dtype,
    ).to(device)
    liger_lm_head_ce = LigerLMHeadCE(
        H=H,
        V=V,
        bias=bias,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        dtype=dtype,
    ).to(device)

    # init the linear in all CEs with the same weights
    torch_lm_head_ce.lin.weight.data = liger_lm_head_ce.lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_ce.lin.bias.data = liger_lm_head_ce.lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = torch_lm_head_ce(_input1, target)
    output2 = liger_lm_head_ce(_input2, target)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

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
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, bias, atol, rtol):
    device = "cuda"

    _input = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    x1 = _input.detach().clone().requires_grad_(True)
    x2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype) if bias else None

    y1 = liger_fused_linear_cross_entropy(x1, weight, target, bias)
    y2 = LigerFusedLinearCrossEntropyFunction.apply(x2, weight, target, bias)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
