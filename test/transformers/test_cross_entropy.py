import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16
from torch.nn import CrossEntropyLoss

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.utils import infer_device

device = infer_device()
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
        output2, z_output2 = target_ce(_input2, target)

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
        output2, z_output2 = target_ce(_input2, target)

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

    loss1.backward(gradient=torch.ones_like(output))
    loss2.backward(gradient=torch.ones_like(output))
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


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

    y1, y1_z = liger_cross_entropy(
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
    y2, y2_z = LigerCrossEntropyFunction.apply(x2, target, None, 0, 1e-4, 0.1, "mean", 30.0, True)

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
    BLOCK_SIZE = 32768
    reduction = "mean"

    # Initialize input tensors
    X_init = torch.randn(batch_size, n_cols, dtype=torch.bfloat16, device=device)
    Y = torch.randint(0, n_cols, (batch_size,), device=device)

    # Run kernel for bfloat16
    X_bf16 = X_init.clone()
    loss_bf16 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_bf16,
        X_stride=X_bf16.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        weight_ptr=X_bf16,  # dummy ptr, not used
        z_loss_ptr=loss_bf16,  # dummy ptr, not used
        loss_ptr=loss_bf16,
        loss_stride=loss_bf16.stride(-1),
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
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    # Run kernel for float32
    X_fp32 = X_init.float()
    loss_fp32 = torch.zeros(batch_size, dtype=torch.float32, device=device)
    liger_cross_entropy_kernel[(batch_size,)](
        X_ptr=X_fp32,
        X_stride=X_fp32.stride(-2),
        Y_ptr=Y,
        Y_stride=Y.stride(-1),
        weight_ptr=X_fp32,  # dummy ptr, not used
        loss_ptr=loss_fp32,
        z_loss_ptr=loss_fp32,  # dummy ptr, not used
        loss_stride=loss_fp32.stride(-1),
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
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
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
