import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="CuTe DSL fused linear cross entropy requires Hopper (SM90)",
)


def _operator():
    pytest.importorskip("cutlass")
    from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy_sm90 import LigerFusedLinearCrossEntropySM90Function

    return LigerFusedLinearCrossEntropySM90Function


def _reference(x, weight, target, ignore_index, reduction, grad_output):
    x_ref = x.float().detach().requires_grad_()
    weight_ref = weight.float().detach().requires_grad_()
    loss = torch.nn.functional.cross_entropy(
        x_ref @ weight_ref.T,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    loss.backward(grad_output)
    return loss.detach(), x_ref.grad.detach(), weight_ref.grad.detach()


def _relative_max_error(actual, expected):
    return (actual.float() - expected.float()).abs().max() / expected.float().abs().max().clamp_min(1e-9)


@pytest.mark.parametrize("shape", [(128, 256, 512), (256, 512, 768)])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("with_ignored_tokens", [False, True])
def test_sm90_fused_linear_cross_entropy(shape, reduction, with_ignored_tokens):
    torch.manual_seed(0)
    tokens, hidden_size, vocab_size = shape
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    if with_ignored_tokens:
        target[::17] = -100
    grad_output = torch.tensor(0.75, device="cuda")

    x_actual = x.detach().clone().requires_grad_()
    weight_actual = weight.detach().clone().requires_grad_()
    loss_actual = _operator().apply(
        x_actual,
        weight_actual,
        target,
        -100,
        reduction,
    )
    loss_actual.backward(grad_output)

    loss_expected, dx_expected, dw_expected = _reference(
        x,
        weight,
        target,
        -100,
        reduction,
        grad_output,
    )

    assert torch.allclose(loss_actual.float(), loss_expected, atol=5e-3, rtol=5e-3)
    assert _relative_max_error(x_actual.grad, dx_expected) < 3e-2
    assert _relative_max_error(weight_actual.grad, dw_expected) < 3e-2


def test_sm90_fused_linear_cross_entropy_rejects_unsupported_shapes_and_dtypes():
    function = _operator()
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(512, (128,), device="cuda")

    with pytest.raises(TypeError, match="BF16"):
        function.apply(x.float(), weight.float(), target)
    with pytest.raises(ValueError, match="reduction"):
        function.apply(x, weight, target, -100, "none")
    with pytest.raises(NotImplementedError, match="multiple of 128"):
        function.apply(x[:127], weight, target[:127])
    with pytest.raises(NotImplementedError, match="bias"):
        function.apply(x, weight, target, -100, "mean", torch.zeros(512, device="cuda"))


def test_sm90_fused_linear_cross_entropy_repeated_backward():
    torch.manual_seed(1)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(512, (128,), device="cuda")
    loss = _operator().apply(x, weight, target)

    first = torch.autograd.grad(loss, (x, weight), retain_graph=True)
    second = torch.autograd.grad(loss, (x, weight), retain_graph=True)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_sm90_fused_linear_cross_entropy_repeated_training_steps():
    torch.manual_seed(2)
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    for step in range(20):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(512, (128,), device="cuda")
        if step % 3 == 0:
            target[::17] = -100
        _operator().apply(x, weight, target).backward()
        torch.cuda.synchronize()
        weight.grad = None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two Hopper GPUs")
def test_sm90_fused_linear_cross_entropy_noncurrent_cuda_device():
    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    x = torch.randn(128, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(512, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(512, (128,), device=device)

    loss = _operator().apply(x, weight, target)
    loss.backward()

    assert loss.device == device
    assert x.grad is not None
    assert weight.grad is not None
