import pytest
import torch

from test.transformers.test_fused_linear_scaled_cross_entropy import _FRONTEND


@pytest.fixture(params=["fallback", "cutile", "sm90"])
def backend(request):
    name = request.param
    if name == "fallback":
        return name, _FRONTEND._FusedLinearPPOFallbackFunction, "cpu", torch.float32
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if name == "sm90":
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("requires Hopper")
        pytest.importorskip("cutlass")
        from liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90 import (
            LigerFusedScaledCrossEntropySM90Function,
        )

        return name, LigerFusedScaledCrossEntropySM90Function, "cuda", torch.bfloat16
    pytest.importorskip("cuda.tile")
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    return name, LigerFusedLinearScaledCrossEntropyFunction, "cuda", torch.bfloat16


def apply(backend, x, weight, target, ids, temperature=1.0, entropy=True):
    name, function, _, _ = backend
    args = (x, weight, target, temperature, -100)
    if name != "fallback":
        args += (1,)
    return function.apply(*args, entropy, ids)


def reference(name, x, weight, target, ids, temperature, entropy):
    if name == "sm90":
        logits = (x.float() @ weight.float().t()).half().float() / temperature
    elif name == "cutile":
        logits = (x @ weight.t()).float() / temperature
    else:
        logits = ((x @ weight.t()) / temperature).float()
    log_probs = logits.log_softmax(-1)
    valid = target != -100
    nll = -log_probs.gather(1, target.masked_fill(~valid, 0)[:, None]).squeeze(1)
    selected = log_probs[:, ids].masked_fill(~valid[:, None], 0)
    outputs = (nll.masked_fill(~valid, 0),)
    if entropy:
        ent = -(log_probs.exp() * log_probs).sum(-1).to(x.dtype).masked_fill(~valid, 0)
        outputs += (ent,)
    return outputs + (selected,)


def inputs(backend, shape):
    _, _, device, dtype = backend
    tokens, hidden, vocab = shape
    x = (torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.25).requires_grad_()
    weight = (torch.randn(vocab, hidden, device=device, dtype=dtype) / hidden**0.5).requires_grad_()
    target = torch.randint(vocab, (tokens,), device=device)
    target[::7] = -100
    return x, weight, target


@torch.no_grad()
def native_gradients(name, x, weight, target, ids, temperature, entropy, gn, ge, gs):
    if name == "sm90":
        raw_logits = (x.float() @ weight.float().t()).half().float()
    else:
        raw_logits = (x @ weight.t()).float()
    q = raw_logits / temperature
    log_probs = q.log_softmax(-1)
    probabilities = log_probs.exp()
    dz = probabilities * (gn - gs.sum(-1))[:, None]
    dz.scatter_add_(1, ids.expand(x.shape[0], -1), gs)
    valid = target != -100
    dz.scatter_add_(1, target.masked_fill(~valid, 0)[:, None], -gn[:, None])
    if ge is not None:
        dz -= ge.float()[:, None] * probabilities * (log_probs + entropy.float()[:, None])
    dz.masked_fill_(~valid[:, None], 0)
    dz = (dz / temperature).to(x.dtype)
    expected_dx = dz @ weight
    wave = 1024 if name == "sm90" else x.shape[0]
    expected_dw = torch.zeros_like(weight)
    for start in range(0, x.shape[0], wave):
        expected_dw.add_(dz[start : start + wave].t() @ x[start : start + wave])
    return expected_dx, expected_dw


@pytest.mark.parametrize("shape", [(7, 8, 13), (521, 70, 517), (1025, 64, 67)])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
@pytest.mark.parametrize("entropy", [False, True])
def test_selected_outputs_and_combined_gradients(backend, shape, temperature, entropy):
    name, _, device, _ = backend
    x, weight, target = inputs(backend, shape)
    # A strided selection with duplicate and boundary IDs.
    ids = torch.tensor([shape[2] - 1, 2, 0, 2, 3, 2, 0, 2], device=device)[::2]
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()

    actual = apply(backend, x, weight, target, ids, temperature, entropy)
    expected = reference(name, expected_x, expected_weight, target, ids, temperature, entropy)
    grads = tuple(torch.randn_like(output) for output in expected)
    torch.autograd.backward(actual, grads)
    torch.autograd.backward(expected, grads)

    for value, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, ref, atol=2e-4, rtol=2e-4)
        assert torch.count_nonzero(value[target == -100]) == 0
    assert actual[-1].dtype == torch.float32
    assert actual[-1].shape == (shape[0], ids.numel())
    if name == "fallback":
        torch.testing.assert_close(x.grad, expected_x.grad, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(weight.grad, expected_weight.grad, atol=3e-5, rtol=3e-5)
    else:
        # Native kernels round dZ to BF16 and accumulate dW per wave.
        dx, dw = native_gradients(
            name,
            x,
            weight,
            target,
            ids,
            temperature,
            expected[1] if entropy else None,
            grads[0],
            grads[1] if entropy else None,
            grads[-1],
        )
        torch.testing.assert_close(x.grad, dx, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(weight.grad, dw, atol=3e-3, rtol=2e-2)
        for value, ref in ((x.grad, expected_x.grad), (weight.grad, expected_weight.grad)):
            relative_error = (value.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
            assert relative_error < 0.02
    assert torch.count_nonzero(x.grad[target == -100]) == 0


@pytest.mark.parametrize("output", ["nll", "entropy", "selected"])
@pytest.mark.parametrize("trainable", ["both", "input", "weight"])
def test_independent_output_backward(backend, output, trainable):
    name, _, device, _ = backend
    x, weight, target = inputs(backend, (19, 64, 131))
    x.requires_grad_(trainable != "weight")
    weight.requires_grad_(trainable != "input")
    ids = torch.tensor([130, 0, 1], device=device)
    target[1] = 50
    expected_x = x.detach().clone().requires_grad_(x.requires_grad)
    expected_weight = weight.detach().clone().requires_grad_(weight.requires_grad)
    index = {"nll": 0, "entropy": 1, "selected": 2}[output]

    actual = apply(backend, x, weight, target, ids)[index]
    expected = reference(name, expected_x, expected_weight, target, ids, 1.0, True)[index]
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    for value, ref in ((x, expected_x), (weight, expected_weight)):
        if value.requires_grad:
            torch.testing.assert_close(value.grad, ref.grad, atol=4e-3, rtol=4e-2)
        else:
            assert value.grad is None


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("ignored", [False, True])
def test_empty_selection_and_ignored_rows(backend, empty, ignored):
    _, _, device, _ = backend
    x, weight, target = inputs(backend, (9, 64, 131))
    ids = torch.tensor([] if empty else [0, 130], dtype=torch.int64, device=device)
    if ignored:
        target.fill_(-100)

    nll, selected = apply(backend, x, weight, target, ids, entropy=False)
    selected.sum().backward()

    assert selected.shape == (9, ids.numel())
    if empty or ignored:
        assert torch.count_nonzero(selected) == 0
        assert torch.count_nonzero(x.grad) == 0
        assert torch.count_nonzero(weight.grad) == 0
    if ignored:
        assert torch.count_nonzero(nll) == 0


@pytest.mark.parametrize("count", [1, 16])
def test_noncontiguous_inputs_and_larger_selection(backend, count):
    name, _, device, dtype = backend
    x = (torch.randn(70, 193, device=device, dtype=dtype) * 0.25).t().detach().requires_grad_()
    weight = (torch.randn(70, 1537, device=device, dtype=dtype) / 70**0.5).t().detach().requires_grad_()
    target = torch.randint(1537, (193,), device=device)
    ids = torch.linspace(0, 1536, count, device=device).long()
    actual = apply(backend, x, weight, target, ids, entropy=False)
    expected = reference(name, x, weight, target, ids, 1.0, False)
    for value, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, ref, atol=2e-4, rtol=2e-4)
    gs = torch.randn_like(actual[-1])
    actual[-1].backward(gs)
    if name == "fallback":
        dx, dw = torch.autograd.grad(expected[-1], (x, weight), gs)
    else:
        dx, dw = native_gradients(name, x, weight, target, ids, 1.0, None, torch.zeros_like(actual[0]), None, gs)
    torch.testing.assert_close(x.grad, dx, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(weight.grad, dw, atol=3e-3, rtol=2e-2)


def test_selection_device_mismatch(backend):
    _, _, device, _ = backend
    x, weight, target = inputs(backend, (7, 64, 13))
    ids = torch.empty(1, dtype=torch.int64, device="meta" if device == "cpu" else "cpu")
    with pytest.raises(ValueError, match="same device"):
        apply(backend, x, weight, target, ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cutile_fp32_selected_gradients_with_multiple_chunks(monkeypatch):
    pytest.importorskip("cuda.tile")
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    monkeypatch.setenv("LIGER_CUTILE_SCALED_CE_WORKSPACE_MB", "1")
    backend = ("cutile", LigerFusedLinearScaledCrossEntropyFunction, "cuda", torch.float32)
    x, weight, target = inputs(backend, (521, 31, 1025))
    ids = torch.tensor([1024, 0, 1, 0], device="cuda")
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    actual = apply(backend, x, weight, target, ids, 0.7)
    expected = reference("cutile", expected_x, expected_weight, target, ids, 0.7, True)
    grads = tuple(torch.randn_like(value) for value in expected)
    torch.autograd.backward(actual, grads)
    torch.autograd.backward(expected, grads)
    for value, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=6e-5, rtol=6e-5)
    torch.testing.assert_close(weight.grad, expected_weight.grad, atol=6e-5, rtol=6e-5)


@pytest.mark.parametrize("objective", ["token_set", "vocab"])
def test_genrank_objectives(backend, objective):
    name, _, device, _ = backend
    x, weight, target = inputs(backend, (13, 64, 131))
    ids = torch.tensor([0, 130], device=device)
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    labels = torch.arange(13, device=device) % 2
    valid = target != -100
    actual = apply(backend, x, weight, target, ids, entropy=False)[-1][valid]
    expected = reference(name, expected_x, expected_weight, target, ids, 1.0, False)[-1][valid]

    def loss(log_probs):
        if objective == "token_set":
            return torch.nn.functional.cross_entropy(log_probs, labels[valid])
        return torch.nn.functional.binary_cross_entropy(log_probs[:, 1].exp(), labels[valid].float())

    actual_loss, expected_loss = loss(actual), loss(expected)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=4e-4, rtol=4e-2)
    torch.testing.assert_close(weight.grad, expected_weight.grad, atol=4e-4, rtol=4e-2)


def test_quantized_combined_backward_matches_full_matrix(backend):
    name, _, device, _ = backend
    if name == "fallback":
        pytest.skip("native BF16 staging contract")
    x, weight, target = inputs(backend, (1025, 70, 517))
    ids = torch.tensor([516, 0, 3, 0], device=device)
    temperature = 0.7
    nll, entropy, selected = apply(backend, x, weight, target, ids, temperature)
    gn, ge, gs = (torch.randn_like(value) for value in (nll, entropy, selected))
    torch.autograd.backward((nll, entropy, selected), (gn, ge, gs))

    expected_dx, expected_dw = native_gradients(name, x, weight, target, ids, temperature, entropy, gn, ge, gs)

    torch.testing.assert_close(x.grad, expected_dx, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(weight.grad, expected_dw, atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ([0], TypeError, "int64 tensor"),
        (torch.tensor([0.0]), TypeError, "int64 tensor"),
        (torch.tensor([[0]]), ValueError, r"shape \[K\]"),
        (torch.tensor([-1]), ValueError, r"in \[0, 13\)"),
        (torch.tensor([13]), ValueError, r"in \[0, 13\)"),
    ],
)
def test_invalid_selection(backend, selection, error, message):
    _, _, device, _ = backend
    x, weight, target = inputs(backend, (7, 64, 13))
    if isinstance(selection, torch.Tensor):
        selection = selection.to(device)
    with pytest.raises(error, match=message):
        apply(backend, x, weight, target, selection)


def test_frontend_forwards_selection_and_preserves_default_contract(monkeypatch):
    ids = torch.tensor([1, 0])
    calls = []

    class Stub:
        @staticmethod
        def apply(*args):
            calls.append(args)

    monkeypatch.setattr(_FRONTEND, "_resolve_implementation", lambda _: Stub)
    x, weight, target = torch.randn(2, 3), torch.randn(4, 3), torch.tensor([0, 1])
    _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(x, weight, target)
    _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(x, weight, target, selected_token_ids=ids)
    assert len(calls[0]) == 7
    assert len(calls[1]) == 8
    assert calls[1][-1] is ids


def test_frontend_cpu_selected_output():
    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    target = torch.tensor([1, 0, -100])
    ids = torch.tensor([0, 3, 3])
    nll, selected = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        x, weight, target, selected_token_ids=ids
    )
    expected = reference("fallback", x, weight, target, ids, 1.0, False)
    torch.testing.assert_close(nll, expected[0])
    torch.testing.assert_close(selected, expected[1])
    selected.sum().backward()
    assert x.grad is not None and weight.grad is not None
    assert torch.count_nonzero(x.grad[-1]) == 0


def test_float64_recovery_reference_values_and_gradients():
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1, 0, 6])
    ids = torch.tensor([0, 3, 3])

    def recover(x, weight):
        logits = x @ weight.t() / 0.7
        anchor = logits.log_softmax(-1).gather(1, target[:, None])
        return anchor + logits[:, ids] - logits.gather(1, target[:, None])

    expected = (x @ weight.t() / 0.7).log_softmax(-1)[:, ids]
    torch.testing.assert_close(recover(x, weight), expected)
    assert torch.autograd.gradcheck(recover, (x, weight))
