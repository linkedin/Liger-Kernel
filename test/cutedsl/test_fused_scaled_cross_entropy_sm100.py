"""Numerical coverage for native non-TP Blackwell scaled cross entropy."""

import pytest
import torch

from test.utils import set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Native Blackwell scaled cross entropy requires SM100 or SM103",
)


def _module():
    import liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm100 as module

    return module


def _reference(x, weight, target, temperature, return_entropy, selected_ids, projection_gradients=None):
    projection = x.float() @ weight.float().t()
    if projection_gradients is not None:
        projection.register_hook(projection_gradients.append)
    log_probs = (projection / temperature).log_softmax(-1)
    ignored = target == -100
    nll = -log_probs.gather(1, target.masked_fill(ignored, 0)[:, None]).squeeze(1)
    outputs = (nll.masked_fill(ignored, 0),)
    if return_entropy:
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        outputs += (entropy.masked_fill(ignored, 0).to(x.dtype),)
    if selected_ids is not None:
        outputs += (log_probs[:, selected_ids].masked_fill(ignored[:, None], 0),)
    return outputs


def _assert_gradients(x, weight, xr, wr, projection_gradient, chunk_size=None):
    # Model BF16 dZ, partial-gradient, and cross-chunk rounding independently.
    # Keep the separate error bound against unrounded autograd.
    staged = projection_gradient.to(torch.bfloat16).float()
    chunk_size = x.shape[0] if chunk_size is None else chunk_size
    staged_weight_gradient = torch.zeros_like(weight)
    for start in range(0, x.shape[0], chunk_size):
        partial = staged[start : start + chunk_size].t() @ x.detach()[start : start + chunk_size].float()
        staged_weight_gradient.add_(partial.to(weight.dtype))
    staged_gradients = (
        (staged @ weight.detach().float()).to(x.dtype),
        staged_weight_gradient,
    )
    for actual, reference, expected in zip((x, weight), (xr, wr), staged_gradients):
        if not actual.requires_grad:
            assert actual.grad is None
            continue
        torch.testing.assert_close(actual.grad.float(), expected.float(), atol=2e-3, rtol=2e-2)
        error = (actual.grad.float() - reference.grad.float()).norm()
        scale = reference.grad.float().norm().clamp_min(1e-12)
        assert (error / scale).item() < 0.01


@pytest.mark.parametrize("shape", [(17, 64, 97), (193, 70, 517), (513, 128, 1024), (513, 512, 4096)])
@pytest.mark.parametrize("mode", ["nll", "entropy", "selected", "combined"])
def test_sm100_outputs_and_signed_gradients(shape, mode, monkeypatch):
    module = _module()
    set_seed(42)
    tokens, hidden, vocab = shape
    x = (torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    weight = (torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5).requires_grad_()
    target = torch.randint(vocab, (tokens,), device="cuda")
    target[::7] = -100
    selected_ids = torch.tensor([0, vocab - 1, 0], device="cuda") if mode in ("selected", "combined") else None
    return_entropy = mode in ("entropy", "combined")
    monkeypatch.setattr(module, "_BACKWARD_WORKSPACE_BYTES", 256 * module._align(vocab) * 2)
    temperature = 0.7

    actual = module.LigerFusedScaledCrossEntropySM100Function.apply(
        x, weight, target, temperature, -100, 1, return_entropy, selected_ids
    )
    actual = actual if isinstance(actual, tuple) else (actual,)
    xr, wr = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    projection_gradients = []
    expected = _reference(xr, wr, target, temperature, return_entropy, selected_ids, projection_gradients)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    if mode in ("entropy", "selected"):
        used_actual, used_expected = actual[-1:], expected[-1:]
    else:
        used_actual, used_expected = actual, expected
    upstream = tuple(torch.randn_like(output) for output in used_actual)
    torch.autograd.backward(used_actual, upstream)
    torch.autograd.backward(used_expected, upstream)

    _assert_gradients(x, weight, xr, wr, projection_gradients[0], chunk_size=256)


@pytest.mark.parametrize("backend", ["sm100", "cutile"])
@pytest.mark.parametrize("mode", ["selected", "combined"])
@pytest.mark.parametrize("tokens,chunk_size,seed", [(513, 256, 42), (4097, 256, 43), (4097, 2048, 43)])
def test_matched_bf16_selected_gradients_against_fp32(backend, mode, tokens, chunk_size, seed, monkeypatch):
    if backend == "sm100":
        module = _module()
        monkeypatch.setattr(module, "_BACKWARD_WORKSPACE_BYTES", chunk_size * 1024 * 2)
        function = module.LigerFusedScaledCrossEntropySM100Function
    else:
        pytest.importorskip("cuda.tile")
        import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as module

        monkeypatch.setattr(module, "_calculate_token_chunk_size", lambda *args, **kwargs: chunk_size)
        function = module.LigerFusedLinearScaledCrossEntropyFunction
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    set_seed(seed)
    x = (torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
    weight = (torch.randn(1024, 128, device="cuda", dtype=torch.bfloat16) / 128**0.5).requires_grad_()
    target = torch.randint(1024, (tokens,), device="cuda")
    target[::7] = -100
    selected_ids = torch.tensor([0, 1023, 0], device="cuda")
    return_entropy = mode == "combined"
    xr = x.detach().float().requires_grad_()
    wr = weight.detach().float().requires_grad_()

    actual = function.apply(x, weight, target, 0.7, -100, 1, return_entropy, selected_ids)
    expected = _reference(xr, wr, target, 0.7, return_entropy, selected_ids)
    used_actual = actual if return_entropy else actual[-1:]
    used_expected = expected if return_entropy else expected[-1:]
    upstream = tuple(torch.randn_like(value) for value in used_actual)
    torch.autograd.backward(used_actual, upstream)
    torch.autograd.backward(used_expected, upstream)

    for value, reference in ((x.grad, xr.grad), (weight.grad, wr.grad)):
        relative_error = (value.float() - reference).norm() / reference.norm().clamp_min(1e-12)
        assert relative_error.item() < 0.01
    assert torch.count_nonzero(x.grad[target == -100]) == 0


@pytest.mark.parametrize("needs_grad", [(True, False), (False, True), (True, True)])
def test_sm100_noncontiguous_inputs_and_empty_selection(needs_grad):
    module = _module()
    set_seed(43)
    x = torch.randn(33, 140, device="cuda", dtype=torch.bfloat16)[:, ::2].requires_grad_(needs_grad[0])
    weight = (torch.randn(194, 70, device="cuda", dtype=torch.bfloat16)[::2] / 70**0.5).requires_grad_(needs_grad[1])
    target = torch.randint(97, (66,), device="cuda")[::2]
    ids = torch.empty(0, device="cuda", dtype=torch.int64)

    nll, selected = module.LigerFusedScaledCrossEntropySM100Function.apply(x, weight, target, 1.5, -100, 1, False, ids)
    xr = x.detach().clone().requires_grad_(needs_grad[0])
    wr = weight.detach().clone().requires_grad_(needs_grad[1])
    projection_gradients = []
    expected_nll, expected_selected = _reference(xr, wr, target, 1.5, False, ids, projection_gradients)
    upstream = torch.randn_like(nll)
    nll.backward(upstream)
    expected_nll.backward(upstream)

    torch.testing.assert_close(nll, expected_nll, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(selected, expected_selected)
    _assert_gradients(x, weight, xr, wr, projection_gradients[0])


@pytest.mark.parametrize("tokens", [17, 513])
def test_sm100_all_ignored_outputs_and_gradients_are_zero(tokens, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_BACKWARD_WORKSPACE_BYTES", 256 * module._align(97) * 2)
    x = torch.randn(tokens, 70, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(97, 70, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.full((tokens,), -100, device="cuda")
    ids = torch.tensor([0, 96], device="cuda")

    outputs = module.LigerFusedScaledCrossEntropySM100Function.apply(x, weight, target, 0.7, -100, 1, True, ids)
    torch.autograd.backward(outputs, tuple(torch.randn_like(output) for output in outputs))

    for value in (*outputs, x.grad, weight.grad):
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)


@pytest.mark.parametrize("vocab,hidden", [(64, 64), (320, 192), (4096, 576)])
def test_sm100_weight_gradient_tma_preserves_fp32_accumulation(vocab, hidden):
    module = _module()
    dz = torch.ones(256, vocab, device="cuda", dtype=torch.bfloat16)
    first_dz = dz.clone()
    first_dz[0] += 2**-7
    x = torch.ones(256, hidden, device="cuda", dtype=torch.bfloat16)
    storage = torch.full((vocab + 64, hidden), float("nan"), device="cuda", dtype=torch.float32)
    actual = storage[:vocab]
    module.ensure_cuda_context()

    # Preserve low bits in both the GEMM partial and the accumulated destination.
    for index, partial in enumerate((first_dz, dz * 2**-10, -dz)):
        module._gemm(
            partial.t(),
            x.t(),
            module._weight_gradient_epilogue,
            (),
            ("dw", index != 0),
            tma_output=actual,
            reduce_add_output=index != 0,
        )

    torch.testing.assert_close(actual, torch.full_like(actual, 0.25 + 2**-7), atol=0, rtol=0)
    assert torch.isnan(storage[vocab:]).all()


def test_sm100_public_frontend_uses_native_without_override(monkeypatch):
    import liger_kernel.ops.fused_linear_scaled_cross_entropy as frontend

    from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction

    module = _module()
    monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
    assert frontend._resolve_implementation(torch.device("cuda:0"), torch.bfloat16) is (
        module.LigerFusedScaledCrossEntropySM100Function
    )
    x = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(97, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(97, (17,), device="cuda")

    nll = LigerFusedLinearScaledCrossEntropyFunction.apply(x, weight, target)
    expected = _reference(x, weight, target, 1.0, False, None)[0]

    torch.testing.assert_close(nll, expected, atol=2e-5, rtol=2e-5)
    assert "SM100" in type(nll.grad_fn).__name__
