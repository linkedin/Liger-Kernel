import importlib.util
import os

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_FRONTEND_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "liger_kernel" / "ops" / "fused_linear_scaled_cross_entropy.py"
)
_SPEC = importlib.util.spec_from_file_location("_fused_linear_scaled_cross_entropy_frontend", _FRONTEND_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FRONTEND = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FRONTEND)

_CUTILE_ENABLED = os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() == "cutile"


@pytest.mark.parametrize(
    ("device", "cuda_available", "hip_version"),
    [
        ("cpu", True, None),
        ("cuda:0", False, None),
        ("cuda:0", True, "6.3"),
    ],
    ids=["cpu-input", "cuda-unavailable", "rocm"],
)
def test_frontend_uses_fallback_without_sm90_nvidia(monkeypatch, device, cuda_available, hip_version):
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", hip_version)
    monkeypatch.setattr(
        _FRONTEND.torch.cuda,
        "get_device_capability",
        lambda _: pytest.fail("compute capability must not be queried without an NVIDIA GPU"),
    )

    assert _FRONTEND._resolve_implementation(torch.device(device)) is _FRONTEND._FusedLinearPPOFallbackFunction


def test_frontend_uses_fallback_for_unsupported_nvidia_compute_capability(monkeypatch):
    device = torch.device("cuda:2")
    seen = []

    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(
        _FRONTEND.torch.cuda,
        "get_device_capability",
        lambda actual: seen.append(actual) or (8, 9),
    )
    monkeypatch.setattr(
        _FRONTEND,
        "_load_sm90_function",
        lambda: pytest.fail("SM90 implementation must not load for an unsupported capability"),
    )

    assert _FRONTEND._resolve_implementation(device) is _FRONTEND._FusedLinearPPOFallbackFunction
    assert seen == [device]


def test_frontend_dispatches_sm90_and_forwards_arguments(monkeypatch):
    device = torch.device("cuda:3")
    _input = SimpleNamespace(device=device)
    weight = object()
    target = object()
    calls = []

    class StubSM90Function:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return "sm90-result"

    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda actual: (9, 0))
    monkeypatch.setattr(_FRONTEND, "_load_sm90_function", lambda: StubSM90Function)

    result = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        _input,
        weight,
        target,
        0.7,
        -1,
        3,
        True,
    )

    assert result == "sm90-result"
    assert calls == [(_input, weight, target, 0.7, -1, 3, True)]


def test_frontend_fallback_matches_torch_with_entropy_and_ignored_rows():
    torch.manual_seed(42)
    token_count, hidden_size, vocab_size = 521, 7, 13
    target = torch.randint(vocab_size, (token_count,), dtype=torch.long)
    target[::11] = -100
    grad_nll = torch.rand(token_count)
    grad_entropy = torch.rand(token_count) - 0.5
    temperature = 0.7

    actual_input = torch.randn(token_count, hidden_size, requires_grad=True)
    actual_weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
    actual_nll, actual_entropy = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_input,
        actual_weight,
        target,
        temperature,
        -100,
        2,
        True,
    )
    torch.autograd.backward((actual_nll, actual_entropy), (grad_nll, grad_entropy))

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    logits = (expected_input @ expected_weight.t()).float() / temperature
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    valid = target != -100
    safe_target = target.masked_fill(~valid, 0)
    expected_nll = torch.where(
        valid,
        -log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1),
        torch.zeros(token_count),
    )
    expected_entropy = torch.where(
        valid,
        -(probs * log_probs).sum(dim=-1),
        torch.zeros(token_count),
    )
    torch.autograd.backward((expected_nll, expected_entropy), (grad_nll, grad_entropy))

    torch.testing.assert_close(actual_nll, expected_nll)
    torch.testing.assert_close(actual_entropy, expected_entropy)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)


def test_frontend_fallback_returns_only_nll_when_entropy_is_disabled():
    _input = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    target = torch.arange(4)

    output = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(_input, weight, target)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (4,)
    output.sum().backward()
    assert _input.grad is not None
    assert weight.grad is not None


def test_frontend_fallback_supports_entropy_only_backward():
    _input = torch.randn(6, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    target = torch.arange(6)

    _, entropy = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        _input,
        weight,
        target,
        1.0,
        -100,
        1,
        True,
    )
    entropy.sum().backward()

    assert _input.grad is not None
    assert weight.grad is not None


def test_frontend_is_exported_from_ops_root():
    try:
        import liger_kernel.ops as ops
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        pytest.skip("root ops package requires Triton")

    expected_module = (
        "liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy"
        if _CUTILE_ENABLED
        else "liger_kernel.ops.fused_linear_scaled_cross_entropy"
    )
    assert ops.LigerFusedLinearScaledCrossEntropyFunction.__module__ == expected_module
