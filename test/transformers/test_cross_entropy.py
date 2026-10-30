import importlib

import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16
from torch.nn import CrossEntropyLoss

from liger_kernel.ops import LigerCrossEntropyFunction
from liger_kernel.ops.cross_entropy import cross_entropy_forward
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import CrossEntropyOutput
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.utils import infer_device

device = infer_device()
liger_ce_module = importlib.import_module("liger_kernel.transformers.cross_entropy")
set_seed(42)


class CrossEntropyWithZLoss(torch.nn.Module):
    def __init__(
        self,
        weight=None,
        lse_square_scale=0.0,
        reduction="mean",
        ignore_index=-100,
        label_smoothing=0.0,
        return_z_loss=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.weight = weight
        self.lse_square_scale = lse_square_scale
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.return_z_loss = return_z_loss
        self.label_smoothing = label_smoothing
        self.dtype = dtype

    def forward(self, logits, targets):
        # Loss calculations are all in float32
        logits = logits.to(torch.float32)

        target_mask = targets != self.ignore_index

        # Standard cross entropy loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        # Compute log-sum-exp term
        lse = torch.logsumexp(logits, dim=-1)

        # Z-loss term
        z_loss = torch.where(targets != self.ignore_index, self.lse_square_scale * (lse**2), 0.0)

        if self.reduction == "mean":
            z_loss = z_loss.sum() / target_mask.sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss
        ce_loss = ce_loss.to(self.dtype)
        z_loss = z_loss.to(self.dtype)

        # Final loss: cross-entropy loss + Z-loss
        total_loss = ce_loss + z_loss
        if self.return_z_loss:
            return total_loss, z_loss
        else:
            return total_loss


def _test_correctness_once(target_ce, B, T, V, reduction, scalar, dtype, atol, rtol):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_ignore_index_once(target_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol):
    torch_ce = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_label_smoothing_once(target_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol):
    torch_ce = CrossEntropyLoss(label_smoothing=label_smoothing)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_label_smoothing_with_ignore_index_once(
    target_ce, B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
):
    torch_ce = CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_softcap_once(target_ce, B, T, V, softcap, reduction, scalar, dtype, atol, rtol):
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # upcasting to match liger's casting strategy
    # and downcasting to original dtype
    output = torch_ce(softcap * torch.tanh(_input.to(torch.float32) / softcap), target).to(dtype)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))

    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_z_loss_once(
    target_ce,
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
):
    torch.manual_seed(0)
    torch_ce = CrossEntropyWithZLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        dtype=dtype,
    )

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    if return_z_loss:
        output, z_output = torch_ce(_input, target)
        result2 = target_ce(_input2, target)
        if isinstance(result2, CrossEntropyOutput):
            output2 = result2.loss
            z_output2 = result2.z_loss
        else:
            output2, z_output2 = result2
    else:
        output = torch_ce(_input, target)
        output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    if return_z_loss:
        assert torch.allclose(z_output, z_output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()

    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_z_loss_with_other_params_once(
    target_ce,
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
    label_smoothing,
    ignore_index,
    reduction,
):
    torch.manual_seed(0)
    torch_ce = CrossEntropyWithZLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
        dtype=dtype,
    )

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    if return_z_loss:
        output, z_output = torch_ce(_input, target)
        result2 = target_ce(_input2, target)
        output2 = result2.loss
        z_output2 = result2.z_loss
    else:
        output = torch_ce(_input, target)
        output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    if return_z_loss:
        assert torch.allclose(z_output, z_output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert_verbose_allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_out_of_bounds_target_once(target_ce, B, T, V, ignore_index):
    torch.manual_seed(0)

    _tensor = torch.randn(B * T, V, device=device, dtype=torch.bfloat16)
    _input = _tensor.detach().clone().requires_grad_(True)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    # Assign out of bounds target
    num_out_of_bounds = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_out_of_bounds]  # Randomly select indices
    target[indices_to_assign] = torch.randint(V, 2 * V, (num_out_of_bounds,)).to(device)

    try:
        _ = target_ce(_input, target)
        assert False, "Should have thrown an error"
    except AssertionError as e:
        assert "out of bounds" in str(e)


def _test_correctness_with_weight_once(target_ce, B, T, V, reduction, weight, scalar, dtype, atol, rtol):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss(weight=weight, reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_weight_with_other_params_once(
    target_ce,
    B,
    T,
    V,
    reduction,
    weight,
    lse_square_scale,
    ignore_index,
    label_smoothing,
    softcap,
    scalar,
    dtype,
    atol,
    rtol,
):
    torch.manual_seed(0)
    torch_ce = CrossEntropyWithZLoss(
        weight=weight,
        lse_square_scale=lse_square_scale,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        dtype=dtype,
    )

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    # upcasting to match liger's casting strategy
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    output = torch_ce(softcap * torch.tanh(_input.to(torch.float32) / softcap), target).to(dtype)
    output2 = target_ce(_input2, target)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    output.backward(gradient=torch.ones_like(output))
    output2.backward(gradient=torch.ones_like(output))
    assert_verbose_allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_not_last_layer_once(target_ce, B, T, V, reduction, scalar, dtype, atol, rtol):
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    loss1 = output * 3
    loss2 = output2 * 3

    grad_output = torch.rand_like(output)
    loss1.backward(gradient=grad_output)
    loss2.backward(gradient=grad_output)
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_not_last_layer_with_other_params_once(
    target_ce,
    B,
    T,
    V,
    reduction,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    softcap,
    scalar,
    dtype,
    atol,
    rtol,
):
    torch_ce = CrossEntropyWithZLoss(
        reduction=reduction,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
    )

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    # upcasting to match liger's casting strategy
    # and downcasting to original dtype
    output = torch_ce(softcap * torch.tanh(_input.to(torch.float32) / softcap), target).to(dtype)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    loss1 = output * 3
    loss2 = output2 * 3

    grad_output = torch.rand_like(output)
    loss1.backward(gradient=grad_output)
    loss2.backward(gradient=grad_output)
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_forward_only(target_ce, B, T, V, reduction, dtype, scalar, atol, rtol):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype) * scalar
    _input = _tensor.detach().clone()
    _input2 = _tensor.detach().clone()

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    with torch.no_grad():
        output = torch_ce(_input, target)
        output2 = target_ce(_input2, target)
        assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    try:
        # Try running backward on liger output
        output2.backward(gradient=torch.ones_like(output))
    except RuntimeError as e:
        assert "does not require grad" in str(e)


def _test_correctness_functional(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
):
    _input = torch.randn(B * T, V, device=device, dtype=dtype) * scalar

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    result = liger_cross_entropy(
        x1,
        target,
        None,
        ignore_index=0,
        lse_square_scale=1e-4,
        label_smoothing=0.1,
        reduction="mean",
        softcap=30.0,
        return_z_loss=True,
    )
    y1 = result.loss
    y1_z = result.z_loss
    y2, y2_z, _, _ = LigerCrossEntropyFunction.apply(x2, target, None, 0, 1e-4, 0.1, "mean", 30.0, True, False, False)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)
    assert torch.allclose(y1_z, y2_z, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)

    y1.backward(grad)
    y2.backward(grad)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


#############################################################################
# Test the correctness of the liger cross entropy loss
#############################################################################


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama
        (3, 423, 32000),  # weird shapes
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness(B, T, V, scalar, dtype, reduction, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(reduction=reduction)
    _test_correctness_once(liger_ce, B, T, V, reduction, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_functional(B, T, V, scalar, dtype, atol, rtol):
    _test_correctness_functional(B, T, V, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V, ignore_index",
    [
        (2, 4096, 32000, 2),
        # weird shapes
        (3, 423, 32000, -123),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_ignore_index(B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    _test_correctness_with_ignore_index_once(liger_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V, label_smoothing",
    [
        (2, 4096, 32000, 0.1),
        # weird shapes
        (3, 423, 32000, 0.1),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_label_smoothing_once(B, T, V, label_smoothing, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(label_smoothing=label_smoothing)
    _test_correctness_with_label_smoothing_once(liger_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V, ignore_index, label_smoothing",
    [
        (2, 4096, 32000, 1, 0.1),
        # weird shapes
        (3, 423, 32000, -300, 0.2),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_label_smoothing_with_ignore_index_once(
    B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    _test_correctness_with_label_smoothing_with_ignore_index_once(
        liger_ce, B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V, softcap",
    [
        (2, 4096, 32000, 30.0),  # llama2, mistral
        # weird shapes
        (3, 423, 32000, 30.0),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_softcap_once(B, T, V, softcap, reduction, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(softcap=softcap, reduction=reduction)
    _test_correctness_with_softcap_once(liger_ce, B, T, V, softcap, reduction, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2
        # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
@pytest.mark.parametrize("return_z_loss", [True, False])
@pytest.mark.parametrize(
    "lse_square_scale",
    [
        1e-4,  # PaLM
        1e-5,  # Chameleon
    ],
)
def test_correctness_with_z_loss_once(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
):
    test_ce = LigerCrossEntropyLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
    )
    _test_correctness_with_z_loss_once(
        test_ce,
        B,
        T,
        V,
        scalar,
        dtype,
        atol,
        rtol,
        lse_square_scale,
        return_z_loss,
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
@pytest.mark.parametrize(
    "return_z_loss, lse_square_scale",
    [
        (True, 1e-4),
        (False, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "label_smoothing, ignore_index, reduction",
    [
        (0.1, 42, "mean"),
        (0.2, -42, "sum"),
    ],
)
def test_correctness_with_z_loss_with_other_params_once(
    B,
    T,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    lse_square_scale,
    return_z_loss,
    label_smoothing,
    ignore_index,
    reduction,
):
    test_ce = LigerCrossEntropyLoss(
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    _test_correctness_with_z_loss_with_other_params_once(
        test_ce,
        B,
        T,
        V,
        scalar,
        dtype,
        atol,
        rtol,
        lse_square_scale,
        return_z_loss,
        label_smoothing,
        ignore_index,
        reduction,
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_weight_once(B, T, V, reduction, scalar, dtype, atol, rtol):
    weight = torch.rand(V, device=device, dtype=dtype)
    test_ce = LigerCrossEntropyLoss(weight=weight, reduction=reduction)
    _test_correctness_with_weight_once(test_ce, B, T, V, reduction, weight, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize(
    "ignore_index, lse_square_scale, label_smoothing, softcap",
    [
        (-100, 1e-4, 0.1, 30.0),
        (42, 1e-5, 0.2, 40.0),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_weight_with_other_params_once(
    B,
    T,
    V,
    reduction,
    lse_square_scale,
    ignore_index,
    label_smoothing,
    softcap,
    scalar,
    dtype,
    atol,
    rtol,
):
    weight = torch.rand(V, device=device, dtype=torch.float32)  # match softcap casting
    test_ce = LigerCrossEntropyLoss(
        weight=weight,
        lse_square_scale=lse_square_scale,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        softcap=softcap,
    )
    _test_correctness_with_weight_with_other_params_once(
        test_ce,
        B,
        T,
        V,
        reduction,
        weight,
        lse_square_scale,
        ignore_index,
        label_smoothing,
        softcap,
        scalar,
        dtype,
        atol,
        rtol,
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_not_last_layer(B, T, V, reduction, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(reduction=reduction)
    _test_correctness_not_last_layer_once(liger_ce, B, T, V, reduction, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 1024, 32000),  # llama2, mistral
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "ignore_index, lse_square_scale, label_smoothing, softcap",
    [
        (-100, 1e-4, 0.1, 30.0),
        (42, 1e-5, 0.2, 40.0),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (1.0, torch.float32, 1e-8, 1e-5),
    ],
)
def test_correctness_not_last_layer_with_other_params(
    B, T, V, reduction, ignore_index, lse_square_scale, label_smoothing, softcap, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(
        reduction=reduction,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        softcap=softcap,
    )
    _test_correctness_not_last_layer_with_other_params_once(
        liger_ce,
        B,
        T,
        V,
        reduction,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        softcap,
        scalar,
        dtype,
        atol,
        rtol,
    )


def test_float32_internal():
    """
    This test validates that the internal softmax calculations occur in float32,
    even if the input dtype is bfloat16.
    """
    # Set up test parameters
    batch_size = 4
    n_cols = 128256
    n_non_ignore = batch_size
    ignore_index = -100
    label_smoothing = 0.0
    lse_square_scale = 0.0
    softcap = 0.0
    BLOCK_SIZE = 4096 if device == "npu" else 32768
    reduction = "mean"

    # Initialize input tensors
    X_init = torch.randn(batch_size, n_cols, dtype=torch.bfloat16, device=device)
    Y = torch.randint(0, n_cols, (batch_size,), device=device)

    # Run kernel for bfloat16
    X_bf16 = X_init.clone()
    loss_bf16 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    token_accuracy_bf16 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    predicted_tokens_bf16 = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_bf16,
        X_stride=X_bf16.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        weight_ptr=X_bf16,  # dummy ptr, not used
        z_loss_ptr=loss_bf16,  # dummy ptr, not used
        loss_ptr=loss_bf16,
        loss_stride=loss_bf16.stride(-1),
        token_accuracy_ptr=token_accuracy_bf16,
        token_accuracy_stride=token_accuracy_bf16.stride(-1),
        predicted_tokens_ptr=predicted_tokens_bf16,
        predicted_tokens_stride=predicted_tokens_bf16.stride(-1),
        n_cols=n_cols,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=n_non_ignore,  # not used
        weight_sum=0.0,  # not used
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=0,  # False
        RETURN_TOKEN_ACCURACY=0,
        RETURN_PREDICTED_TOKENS=0,
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        HAS_GRADIENTS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    # Run kernel for float32
    X_fp32 = X_init.float()
    loss_fp32 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    token_accuracy_fp32 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    predicted_tokens_fp32 = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_fp32,
        X_stride=X_fp32.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        weight_ptr=X_fp32,  # dummy ptr, not used
        loss_ptr=loss_fp32,
        z_loss_ptr=loss_fp32,  # dummy ptr, not used
        loss_stride=loss_fp32.stride(-1),
        token_accuracy_ptr=token_accuracy_fp32,
        token_accuracy_stride=token_accuracy_fp32.stride(-1),
        predicted_tokens_ptr=predicted_tokens_fp32,
        predicted_tokens_stride=predicted_tokens_fp32.stride(-1),
        n_cols=n_cols,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=n_non_ignore,  # not used
        weight_sum=n_non_ignore,  # not used
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=0,  # False
        RETURN_TOKEN_ACCURACY=0,
        RETURN_PREDICTED_TOKENS=0,
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        HAS_GRADIENTS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    torch.allclose(X_bf16, X_fp32.bfloat16())
    torch.allclose(loss_bf16, loss_fp32)


@pytest.mark.parametrize(
    "B, T, V, ignore_index",
    [
        (2, 4096, 32000, 2),
        # weird shapes
        (3, 423, 32000, -123),
    ],
)
def test_correctness_with_out_of_bounds_target_once(B, T, V, ignore_index):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index)
    _test_correctness_with_out_of_bounds_target_once(liger_ce, B, T, V, ignore_index)


@pytest.mark.parametrize(
    "B, T, V, ignore_index",
    [
        (2, 4096, 32000, -100),
        (3, 423, 32000, 2),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "dtype, scalar, atol, rtol",
    [
        (torch.float32, 1.0, 1e-4, 1e-4),
        (torch.float16, 1.0, 1e-2, 1e-2),
        (torch.bfloat16, 1.0, 1e-2, 1e-2),
    ],
)
def test_correctness_with_forward_only(B, T, V, ignore_index, reduction, dtype, scalar, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    _test_correctness_with_forward_only(liger_ce, B, T, V, reduction, dtype, scalar, atol, rtol)


@pytest.mark.parametrize(
    "return_z_loss, return_token_accuracy, return_predicted_tokens",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_liger_cross_entropy_structured_output(return_z_loss, return_token_accuracy, return_predicted_tokens):
    logits = torch.tensor(
        [[2.0, 0.5, -1.0], [0.1, 1.5, 0.3], [0.7, -0.2, 0.9]],
        device=device,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2], device=device)

    original_logits = logits.detach().clone()

    result = liger_cross_entropy(
        logits,
        targets,
        reduction="mean",
        return_z_loss=return_z_loss,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
    )

    if not return_z_loss and not return_token_accuracy and not return_predicted_tokens:
        assert isinstance(result, torch.Tensor)
        assert result.shape == ()
        result.backward()
        assert logits.grad is not None
        logits.grad.zero_()
        return

    assert isinstance(result, CrossEntropyOutput)
    assert result.loss.shape == ()

    if return_z_loss:
        assert result.z_loss is not None
        assert isinstance(result.z_loss, torch.Tensor)
    else:
        assert result.z_loss is None

    if return_token_accuracy:
        assert result.token_accuracy is not None
        with torch.no_grad():
            predictions = original_logits.argmax(dim=-1)
            correct = (predictions == targets).float()
            expected_accuracy = correct.mean()
        assert torch.allclose(result.token_accuracy, expected_accuracy, atol=1e-6)
    else:
        assert result.token_accuracy is None

    if return_predicted_tokens:
        assert result.predicted_tokens is not None
        assert result.predicted_tokens.dtype == torch.int64
        assert result.predicted_tokens.shape == (3,)
        with torch.no_grad():
            expected_predictions = original_logits.argmax(dim=-1)
        assert torch.equal(result.predicted_tokens, expected_predictions)
        # When both are enabled, predicted_tokens and token_accuracy should be consistent
        if return_token_accuracy:
            correct_from_predictions = (result.predicted_tokens == targets).float().mean()
            assert torch.allclose(result.token_accuracy, correct_from_predictions, atol=1e-6)
    else:
        assert result.predicted_tokens is None

    result.loss.backward()
    assert logits.grad is not None
    logits.grad.zero_()


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 128, 512),
        (3, 47, 31),  # weird shapes
    ],
)
@pytest.mark.parametrize("ignore_index", [-100, 2])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_with_predicted_tokens(B, T, V, ignore_index, dtype):
    torch.manual_seed(42)

    _tensor = torch.randn(B * T, V, device=device, dtype=dtype)
    _input = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Assign some elements as ignore_index
    num_ignore = B * T // 4
    indices_to_ignore = torch.randperm(B * T)[:num_ignore]
    target[indices_to_ignore] = ignore_index

    # Compute expected argmax BEFORE the kernel modifies _input in-place
    with torch.no_grad():
        expected_predictions = _tensor.float().argmax(dim=-1)

    liger_ce = LigerCrossEntropyLoss(
        ignore_index=ignore_index,
        return_predicted_tokens=True,
    )
    result = liger_ce(_input, target)

    assert isinstance(result, CrossEntropyOutput)
    assert result.predicted_tokens is not None
    assert result.predicted_tokens.shape == (B * T,)
    assert result.predicted_tokens.dtype == torch.int64

    # For non-ignored tokens, predicted_tokens should match argmax
    non_ignore_mask = target != ignore_index
    assert torch.equal(result.predicted_tokens[non_ignore_mask], expected_predictions[non_ignore_mask])

    # For ignored tokens, predicted_tokens should be -1
    assert torch.all(result.predicted_tokens[~non_ignore_mask] == -1)

    # Verify backward still works
    result.loss.backward()
    assert _input.grad is not None


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        pytest.param(torch.float16),
    ],
)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_correctness_true_class_grad_confident_predictions(dtype, label_smoothing, reduction):
    """
    dx_y = (softmax(x_y) - (1 - label_smoothing)) / N cancels catastrophically once the model is
    confident, so the -(1 - label_smoothing) term must be folded in before the result is rounded
    to the low-precision dtype of the in-place gradient buffer. Random logits leave softmax(x_y)
    near zero and never expose a lossy round-trip through that buffer; confident ones do.

    The true-class gradient is checked against an fp32 reference computed from the same
    low-precision logits, and is required to be no less accurate than torch's own low-precision
    backward on those logits.
    """
    torch.manual_seed(0)
    B, T, V = 2, 64, 4096

    logits = torch.randn(B * T, V, device=device, dtype=torch.float32)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    # Drive softmax(x_y) close to 1 so the true-class gradient is a small difference of
    # comparatively large terms.
    logits[torch.arange(B * T, device=device), target] = logits.max(dim=-1).values + 10.0
    _tensor = logits.to(dtype)

    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)
    _input_ref = _tensor.detach().clone().float().requires_grad_(True)

    torch_ce = CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    liger_ce = LigerCrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    torch_ce(_input, target).backward()
    liger_ce(_input2, target).backward()
    torch_ce(_input_ref, target).backward()

    rows = torch.arange(B * T, device=device)
    ref = _input_ref.grad[rows, target].double()
    torch_err = (_input.grad[rows, target].double() - ref).abs().max().item()
    liger_err = (_input2.grad[rows, target].double() - ref).abs().max().item()

    # Rounding the softmax term before the subtraction inflates this error by ~1/(1 - softmax(x_y)),
    # i.e. orders of magnitude, so a small constant factor over torch is a wide margin.
    assert liger_err <= max(4 * torch_err, torch.finfo(dtype).tiny), (
        f"true-class grad error {liger_err:.3e} exceeds torch's {torch_err:.3e} "
        f"(reduction={reduction}, label_smoothing={label_smoothing}, dtype={dtype})"
    )


_LOSS_DTYPE_SHAPES = [(512, 4096), (256, 32001)]


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("BT, V", _LOSS_DTYPE_SHAPES)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("use_weight", [False, True])
def test_bf16_loss_dtype_float32_matches_fp32_reference(reduction, BT, V, label_smoothing, use_weight):
    # The tolerances are estimates that have not been confirmed on a GPU; the kernel evaluates exp with an
    # approximate hardware instruction, so they leave room for ~1e-5 relative error.
    torch.manual_seed(0)
    logits = (torch.randn(BT, V, device=device) * 4).to(torch.bfloat16)
    target = torch.randint(0, V, (BT,), device=device)
    target[::11] = -100
    weight = torch.rand(V, device=device) + 0.5 if use_weight else None
    ref = F.cross_entropy(logits.float(), target, weight=weight, label_smoothing=label_smoothing, reduction=reduction)

    out_fp32 = liger_cross_entropy(
        logits.clone(),
        target,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction=reduction,
        loss_dtype=torch.float32,
    )
    assert out_fp32.dtype == torch.float32
    torch.testing.assert_close(out_fp32, ref, rtol=1e-5, atol=1e-5)

    # Default keeps the input dtype and equals the fp32 result rounded once.
    out = LigerCrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction=reduction)(
        logits.clone(), target
    )
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_fp32.to(torch.bfloat16))

    out_module = LigerCrossEntropyLoss(
        weight=weight, label_smoothing=label_smoothing, reduction=reduction, loss_dtype=torch.float32
    )(logits.clone(), target)
    assert torch.equal(out_module, out_fp32)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("BT, V", _LOSS_DTYPE_SHAPES)
def test_bf16_default_reduction_is_at_least_as_close_as_per_token_rounding(reduction, BT, V):
    torch.manual_seed(0)
    logits = (torch.randn(BT, V, device=device) * 4).to(torch.bfloat16)
    target = torch.randint(0, V, (BT,), device=device)
    target[::11] = -100
    ref = F.cross_entropy(logits.float(), target, reduction=reduction).double()

    # Emulates the previous algorithm: the per-token losses (already divided by n_non_ignore for "mean")
    # were stored in bf16 before being summed.
    per_token = F.cross_entropy(logits.float(), target, reduction="none")
    if reduction == "mean":
        per_token = per_token / (target != -100).sum()
    previous = per_token.to(torch.bfloat16).sum().double()

    out = liger_cross_entropy(logits.clone(), target, reduction=reduction).double()
    # Rounding the exact value once gives the nearest bf16 number, so it cannot lose to rounding every token first;
    # the slack only covers the kernel's approximate exp.
    assert (out - ref).abs() <= (previous - ref).abs() + 1e-5 * ref.abs()


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_bf16_z_loss_dtype_float32_matches_fp32_reference():
    torch.manual_seed(0)
    BT, V = 2048, 8192
    logits = (torch.randn(BT, V, device=device) * 4).to(torch.bfloat16)
    target = torch.randint(0, V, (BT,), device=device)
    lse_square_scale = 1e-2
    ref_z = lse_square_scale * torch.logsumexp(logits.float(), dim=-1) ** 2

    out = liger_cross_entropy(
        logits.clone(),
        target,
        reduction="none",
        lse_square_scale=lse_square_scale,
        return_z_loss=True,
        loss_dtype=torch.float32,
    )
    assert out.loss.dtype == out.z_loss.dtype == torch.float32
    torch.testing.assert_close(out.z_loss, ref_z, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("loss_dtype", [torch.float16, torch.bfloat16, torch.float64, "float32"])
def test_loss_dtype_rejects_unsupported_values(loss_dtype):
    logits = torch.randn(4, 8, device=device)
    target = torch.randint(0, 8, (4,), device=device)
    with pytest.raises(ValueError, match="loss_dtype"):
        liger_cross_entropy(logits, target, loss_dtype=loss_dtype)
    with pytest.raises(ValueError, match="loss_dtype"):
        LigerCrossEntropyLoss(loss_dtype=loss_dtype)


def test_loss_dtype_requires_triton_backend(monkeypatch):
    class _FakeImpl:
        impl_name = "nvidia-cutedsl"

    monkeypatch.setattr(liger_ce_module, "resolve_impl", lambda *args, **kwargs: _FakeImpl())
    logits = torch.randn(4, 8, device=device)
    target = torch.randint(0, 8, (4,), device=device)
    with pytest.raises(ValueError, match="only supported by the Triton"):
        LigerCrossEntropyLoss(loss_dtype=torch.float32)(logits, target)


def _run_backward(logits, target, backward_fn, kernel_kwargs=None):
    """Returns (grad, grad_output seen by the op)."""
    _input = logits.detach().clone().requires_grad_(True)
    loss = liger_cross_entropy(_input, target, reduction="none", **(kernel_kwargs or {}))
    seen = []
    loss.register_hook(lambda g: seen.append(g))
    backward_fn(loss)
    return _input.grad.clone(), seen[0]


def _none_backward_fns(BT, per_token_weight):
    strided = torch.rand(2 * BT, device=device)[::2]
    return {
        "mean": lambda loss: loss.mean().backward(),
        "sum": lambda loss: loss.sum().backward(),
        "weighted_sum": lambda loss: (loss * per_token_weight).sum().backward(),
        "strided": lambda loss: loss.backward(strided),
    }


_NONE_REDUCERS = ["mean", "sum", "weighted_sum", "strided"]
_NONE_SHAPES = [(512, 4096), (256, 32001), (1024, 8192)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reducer", _NONE_REDUCERS)
@pytest.mark.parametrize("BT, V", _NONE_SHAPES)
def test_none_reduction_grad_matches_per_row_multiply(dtype, reducer, BT, V):
    if dtype == torch.bfloat16 and not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    torch.manual_seed(0)
    logits = torch.randn(BT, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device)
    target[::7] = -100
    per_token_weight = torch.rand(BT, device=device) + 0.5

    grad, g_seen = _run_backward(logits, target, _none_backward_fns(BT, per_token_weight)[reducer])
    assert g_seen.ndim == 1

    # The per-row multiply that the op used to do out of place: fp32 math, a single rounding to the input dtype.
    # cross_entropy_forward leaves the unscaled gradient in its returned input.
    unscaled = cross_entropy_forward(
        logits.clone().requires_grad_(True), target, None, -100, 0.0, 0.0, "none", None, False
    )[4]
    expected = (unscaled.float() * g_seen.float().unsqueeze(1)).to(dtype)
    assert torch.equal(grad, expected)


@pytest.mark.parametrize("reducer", _NONE_REDUCERS)
@pytest.mark.parametrize("BT, V", _NONE_SHAPES)
def test_none_reduction_grad_matches_torch(reducer, BT, V):
    torch.manual_seed(0)
    logits = torch.randn(BT, V, device=device)
    target = torch.randint(0, V, (BT,), device=device)
    target[::7] = -100
    per_token_weight = torch.rand(BT, device=device) + 0.5
    backward_fn = _none_backward_fns(BT, per_token_weight)[reducer]

    ref_input = logits.clone().requires_grad_(True)
    backward_fn(F.cross_entropy(ref_input, target, reduction="none"))
    grad, _ = _run_backward(logits, target, backward_fn)
    torch.testing.assert_close(grad, ref_input.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(device != "cuda", reason="requires CUDA memory statistics")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reducer", ["mean", "sum", "weighted_sum"])
@pytest.mark.parametrize("BT, V", [(2048, 32000), (1024, 32001)])
def test_none_reduction_backward_allocates_no_bt_x_v_temporary(dtype, reducer, BT, V):
    if dtype == torch.bfloat16 and not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    torch.manual_seed(0)
    logits = torch.randn(BT, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device)
    per_token_weight = torch.rand(BT, device=device) + 0.5
    backward_fn = _none_backward_fns(BT, per_token_weight)[reducer]
    logits_bytes = logits.numel() * logits.element_size()

    _input = logits.detach().clone().requires_grad_(True)
    loss = liger_cross_entropy(_input, target, reduction="none")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    backward_fn(loss)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - base

    # Before this change the backward allocated a BT x V temporary here.
    assert peak_extra < 0.5 * logits_bytes


def test_none_reduction_second_backward_raises():
    torch.manual_seed(0)
    logits = torch.randn(64, 512, device=device, requires_grad=True)
    target = torch.randint(0, 512, (64,), device=device)
    loss = liger_cross_entropy(logits, target, reduction="none")
    loss.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match=r"second backward \(retain_graph=True\)"):
        loss.sum().backward()


def test_scalar_backward_with_non_unit_grad_twice_raises():
    torch.manual_seed(0)
    logits = torch.randn(64, 512, device=device, requires_grad=True)
    target = torch.randint(0, 512, (64,), device=device)
    loss = liger_cross_entropy(logits, target, reduction="sum")
    (loss * 2.0).backward(retain_graph=True)
    with pytest.raises(RuntimeError, match=r"second backward \(retain_graph=True\)"):
        (loss * 2.0).backward()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_scalar_backward_with_unit_grad_supports_retain_graph(reduction):
    torch.manual_seed(0)
    logits = torch.randn(64, 512, device=device)
    target = torch.randint(0, 512, (64,), device=device)

    ref_input = logits.clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_input, target, reduction=reduction)
    ref_loss.backward(retain_graph=True)
    ref_loss.backward()

    _input = logits.clone().requires_grad_(True)
    loss = liger_cross_entropy(_input, target, reduction=reduction)
    loss.backward(retain_graph=True)
    loss.backward()
    torch.testing.assert_close(_input.grad, ref_input.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_none_reduction_loss_dtype_float32_backward_keeps_input_dtype():
    torch.manual_seed(0)
    logits = torch.randn(256, 1024, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, 1024, (256,), device=device)
    weight = torch.rand(256, device=device) + 0.5

    ref_input = logits.float().requires_grad_(True)
    (F.cross_entropy(ref_input, target, reduction="none") * weight).sum().backward()
    grad, _ = _run_backward(
        logits,
        target,
        lambda loss: (loss * weight).sum().backward(),
        kernel_kwargs={"loss_dtype": torch.float32},
    )
    assert grad.dtype == torch.bfloat16
    # bf16 gradient against an fp32 reference: about 2**-8 relative rounding error.
    torch.testing.assert_close(grad.float(), ref_input.grad, rtol=2e-2, atol=2e-2)
