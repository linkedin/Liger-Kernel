from test.utils import set_seed, supports_bfloat16

import pytest
import torch
from torch.nn import CrossEntropyLoss

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy

set_seed(42)


def _test_correctness_once(target_ce, B, T, V, reduction, scalar, dtype, atol, rtol):
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_ignore_index_once(
    target_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
):
    torch_ce = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    _tensor = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_label_smoothing_once(
    target_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol
):
    torch_ce = CrossEntropyLoss(label_smoothing=label_smoothing)

    _tensor = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_with_label_smoothing_with_ignore_index_once(
    target_ce, B, T, V, ignore_index, label_smoothing, scalar, dtype, atol, rtol
):
    torch_ce = CrossEntropyLoss(
        ignore_index=ignore_index, label_smoothing=label_smoothing
    )

    _tensor = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(
        1, B * T // 2, (1,)
    ).item()  # Random number of elements to set to ignore_index
    indices_to_assign = torch.randperm(B * T)[
        :num_elements_to_assign
    ]  # Randomly select indices
    target[indices_to_assign] = ignore_index

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)

    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    output.backward()
    output2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_not_last_layer_once(
    target_ce, B, T, V, reduction, scalar, dtype, atol, rtol
):
    torch_ce = CrossEntropyLoss(reduction=reduction)

    _tensor = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar
    _input = _tensor.detach().clone().requires_grad_(True)
    _input2 = _tensor.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    output = torch_ce(_input, target)
    output2 = target_ce(_input2, target)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)

    loss1 = output * 3
    loss2 = output2 * 3

    loss1.backward()
    loss2.backward()
    assert torch.allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)


def _test_correctness_functional(B, T, V, scalar, dtype, atol, rtol):
    _input = torch.randn(B * T, V, device="cuda", dtype=dtype) * scalar

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.long)

    y1 = liger_cross_entropy(x1, target, 0)
    y2 = LigerCrossEntropyFunction.apply(x2, target, 0)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

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
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
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
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        pytest.param(
            1.0,
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_ignore_index(
    B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    _test_correctness_with_ignore_index_once(
        liger_ce, B, T, V, ignore_index, reduction, scalar, dtype, atol, rtol
    )


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
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_label_smoothing_once(
    B, T, V, label_smoothing, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(label_smoothing=label_smoothing)
    _test_correctness_with_label_smoothing_once(
        liger_ce, B, T, V, label_smoothing, scalar, dtype, atol, rtol
    )


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
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
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
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_not_last_layer(B, T, V, reduction, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss(reduction=reduction)
    _test_correctness_not_last_layer_once(
        liger_ce, B, T, V, reduction, scalar, dtype, atol, rtol
    )
