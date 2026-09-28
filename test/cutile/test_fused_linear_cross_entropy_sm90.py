import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed

pytest.importorskip("cuda.tile")

from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile FLCE requires CUDA"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="cuTile FLCE requires Hopper",
    ),
]

set_seed()


def _reference(x, weight, target, ignore_index, reduction):
    return F.cross_entropy(
        F.linear(x, weight).float(),
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )


@pytest.mark.parametrize("shape", [(128, 128, 256), (129, 96, 384), (32, 64, 8192)])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_correctness(shape, reduction):
    tokens, hidden_size, vocab_size = shape
    ignore_index = -100
    upstream = torch.tensor(0.7, device="cuda")
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[: max(1, tokens // 8)] = ignore_index

    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, reduction)
    loss_ref.backward(upstream)

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss, _, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        x,
        weight,
        target,
        None,
        None,
        ignore_index,
        0.0,
        0.0,
        reduction,
    )
    loss.backward(upstream)

    atol = 5e-3 if reduction == "mean" else 5e-2
    rtol = 5e-2
    assert_verbose_allclose(loss_ref, loss, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("requires_input_grad, requires_weight_grad", [(True, False), (False, True)])
def test_independent_gradient_requirements(requires_input_grad, requires_weight_grad):
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=requires_input_grad)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=requires_weight_grad)
    target = torch.randint(256, (128,), device="cuda")

    loss = LigerFusedLinearCrossEntropyFunction.apply(x, weight, target)[0]
    loss.backward()

    assert (x.grad is not None) == requires_input_grad
    assert (weight.grad is not None) == requires_weight_grad


def test_rejects_unsupported_features():
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device="cuda")
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="bias"):
        LigerFusedLinearCrossEntropyFunction.apply(x, weight, target, bias)


def test_retained_backward_preserves_saved_gradient():
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device="cuda")
    upstream = torch.tensor(0.7, device="cuda")
    loss = LigerFusedLinearCrossEntropyFunction.apply(x, weight, target)[0]

    loss.backward(upstream, retain_graph=True)
    first_x_grad = x.grad.clone()
    first_weight_grad = weight.grad.clone()
    x.grad = None
    weight.grad = None
    loss.backward(upstream, retain_graph=True)

    assert_verbose_allclose(first_x_grad, x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(first_weight_grad, weight.grad, atol=0.0, rtol=0.0)


def test_large_token_grid():
    tokens = 65536
    x = torch.randn(tokens, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(128, (tokens,), device="cuda")

    loss = LigerFusedLinearCrossEntropyFunction.apply(x, weight, target)[0]

    assert torch.isfinite(loss)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two Hopper GPUs")
def test_noncurrent_cuda_device():
    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    x = torch.randn(128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device=device)

    loss = LigerFusedLinearCrossEntropyFunction.apply(x, weight, target)[0]
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None
    assert weight.grad is not None
