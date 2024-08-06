from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.linear_fused_cross_entropy import (
    LigerLinearFusedCrossEntropyLoss,
    LigerStatelessLCE,
)

# set random seed globally
set_seed()


class TorchLinearFusedCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LinearLigerStatelessLCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = LigerStatelessLCE(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


class LinearFusedLigerCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with liger cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.liger_ce_loss = LigerCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        logits = self.lin(x)
        return self.liger_ce_loss(logits, y)


#############################################################################
# Test the correctness of the linear fused cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 4, 512, 512),
        (8, 2048, 4096, 32000),  # llama2, mistral
        (4, 2048, 4096, 128256),  # llama3
        (4, 1024, 8192, 128256),  # llama3
        (4, 423, 8192, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
def test_correctness(B, T, H, V, scalar, dtype, atol, rtol):
    device = "cuda"
    torch_lf_ce = TorchLinearFusedCE(H=H, V=V, dtype=dtype).to(device)
    lf_liger_ce = LinearFusedLigerCE(H=H, V=V, dtype=dtype).to(device)
    liger_lf_ce = (
        LigerLinearFusedCrossEntropyLoss(in_features=H, num_classes=V)
        .type(dtype)
        .to(device)
    )

    # init the linear in all CEs with the same weights
    torch_lf_ce.lin.weight.data = lf_liger_ce.lin.weight.data = (
        liger_lf_ce.linear.data
    ) = torch.randn(V, H, device=device, dtype=dtype)

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    _input3 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = torch_lf_ce(_input1, target)
    output2 = lf_liger_ce(_input2, target)
    output3 = liger_lf_ce(_input3, target)
    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)
    assert_verbose_allclose(output1, output3, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()
    output3.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(_input1.grad, _input3.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lf_ce.lin.weight.grad, lf_liger_ce.lin.weight.grad, atol=atol, rtol=rtol
    )
    assert_verbose_allclose(
        torch_lf_ce.lin.weight.grad, liger_lf_ce.linear.grad, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 4, 512, 512),
        (8, 2048, 4096, 32000),  # llama2, mistral
        (4, 2048, 4096, 128256),  # llama3
        (4, 1024, 8192, 128256),  # llama3
        (4, 423, 8192, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        # (1.0, torch.bfloat16, 5e-3, 5e-2),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
def test_correctness_liger_stateless(B, T, H, V, scalar, dtype, atol, rtol):
    device = "cuda"
    torch_lf_ce = TorchLinearFusedCE(H=H, V=V, dtype=dtype).to(device)
    liger_stateless = LinearLigerStatelessLCE(H=H, V=V, dtype=dtype).to(device)

    # init the linear in all CEs with the same weights
    torch_lf_ce.lin.weight.data = liger_stateless.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    _tensor = torch.randn(B * T, H, device=device, dtype=dtype) * scalar
    _input1 = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = torch_lf_ce(_input1, target)
    output2 = liger_stateless(_input2, target)

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(_input1.grad, _input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lf_ce.lin.weight.grad,
        liger_stateless.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
