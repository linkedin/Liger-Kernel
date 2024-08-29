import pytest
import torch
from torch.nn import CrossEntropyLoss

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy

SLEEP_SECONDS = 0.1


def _test_correctness_once(target_ce, B, T, V, scalar, dtype, atol, rtol):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss()

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
    target_ce, B, T, V, ignore_index, scalar, dtype, atol, rtol
):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss(ignore_index=ignore_index)

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
    target_ce, B, T, V, scalar, dtype, atol, rtol
):
    torch.manual_seed(0)
    torch_ce = CrossEntropyLoss()

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
    torch.manual_seed(0)

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
        (2, 4096, 32000),  # llama2, mistral
        (2, 4096, 32000),  # llama2, mistral
        (1, 4096, 128256),  # llama3
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.1, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (10.0, torch.bfloat16, 1e-7, 5e-2),
        (0.1, torch.float32, 1e-8, 1e-6),
        (1.0, torch.float32, 1e-8, 1e-6),
        (10.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness(B, T, V, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss()
    _test_correctness_once(liger_ce, B, T, V, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        (2, 4096, 32000),  # llama2, mistral
        (1, 4096, 128256),  # llama3
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.1, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (10.0, torch.bfloat16, 1e-7, 5e-2),
        (0.1, torch.float32, 1e-8, 1e-6),
        (1.0, torch.float32, 1e-8, 1e-6),
        (10.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_functional(B, T, V, scalar, dtype, atol, rtol):
    _test_correctness_functional(B, T, V, scalar, dtype, atol, rtol)


@pytest.mark.parametrize(
    "B, T, V, ignore_index",
    [
        (2, 4096, 32000, -100),  # llama2, mistral
        (2, 4096, 32000, 2),  # llama2, mistral
        (1, 4096, 128256, -300),  # llama3
        # weird shapes
        (3, 423, 32000, -123),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (0.1, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (10.0, torch.bfloat16, 1e-8, 5e-2),
        (0.1, torch.float32, 1e-8, 1e-6),
        (1.0, torch.float32, 1e-8, 1e-6),
        (10.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_with_ignore_index(
    B, T, V, ignore_index, scalar, dtype, atol, rtol
):
    liger_ce = LigerCrossEntropyLoss(ignore_index=ignore_index)
    _test_correctness_with_ignore_index_once(
        liger_ce, B, T, V, ignore_index, scalar, dtype, atol, rtol
    )


@pytest.mark.parametrize(
    "B, T, V",
    [
        (2, 4096, 32000),  # llama2, mistral
        (2, 4096, 32000),  # llama2, mistral
        (1, 4096, 128256),  # llama3
        # # weird shapes
        (3, 423, 32000),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 1e-8, 5e-2),
        (1.0, torch.float32, 1e-8, 1e-6),
    ],
)
def test_correctness_not_last_layer(B, T, V, scalar, dtype, atol, rtol):
    liger_ce = LigerCrossEntropyLoss()
    _test_correctness_not_last_layer_once(liger_ce, B, T, V, scalar, dtype, atol, rtol)


#############################################################################
# Test full pass of the liger cross entropy loss to ensure it doesn't crash
#############################################################################


def _full_pass_once(B, T, V):
    torch.manual_seed(0)
    liger_ce = LigerCrossEntropyLoss()

    _input = torch.randn(
        B * T, V, requires_grad=True, device="cuda", dtype=torch.bfloat16
    )
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    output = liger_ce(_input, target)
    output.backward()


@pytest.mark.parametrize(
    "B, T, V",
    [
        (
            8,
            8192,
            128256,
        ),  # _input = 16GB, total = ~32GB, 8405385216 > 2,147,483,647, so we need int64
        (8, 16384, 128256),  # _input = 32GB, total = ~64GB
    ],
)
@pytest.mark.skipif(
    torch.cuda.get_device_properties(0).total_memory < 64 * 1000 * 1000 * 1000,
    reason="Needs 64GB+ GPU memory.",
)
def test_large_no_exception(B, T, V):
    # The large inputs were hitting cuda illegal memory access because of
    # https://github.com/triton-lang/triton/issues/1058
    _full_pass_once(B, T, V)
