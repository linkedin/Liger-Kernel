import importlib.util
import os

from contextlib import contextmanager
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
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
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


def test_frontend_dispatches_sm103_from_nondefault_input_device(monkeypatch):
    device = torch.device("cuda:7")
    seen_capability = []
    seen_arch = []

    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(
        _FRONTEND.torch.cuda,
        "get_device_capability",
        lambda actual: seen_capability.append(actual) or (10, 3),
    )
    monkeypatch.setattr(
        _FRONTEND,
        "infer_device_arch",
        lambda device_id: seen_arch.append(device_id) or "blackwell_ultra",
    )

    assert (
        _FRONTEND._resolve_implementation(device, torch.bfloat16)
        is _FRONTEND._FusedLinearScaledCrossEntropySM103Function
    )
    assert seen_capability == [device]
    assert seen_arch == [7]


@pytest.mark.parametrize(
    ("capability", "architecture"),
    [
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell_consumer"),
        ((10, 3), "cuda"),
    ],
)
def test_frontend_keeps_non_sm103_devices_on_existing_paths(monkeypatch, capability, architecture):
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(_FRONTEND, "infer_device_arch", lambda _: architecture)

    assert _FRONTEND._resolve_implementation(torch.device("cuda:2")) is _FRONTEND._FusedLinearPPOFallbackFunction


def test_frontend_keeps_unsupported_dtype_on_fallback(monkeypatch):
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: (10, 3))
    monkeypatch.setattr(
        _FRONTEND,
        "infer_device_arch",
        lambda _: pytest.fail("architecture lookup is unnecessary for an unsupported dtype"),
    )

    assert (
        _FRONTEND._resolve_implementation(torch.device("cuda:2"), torch.float64)
        is _FRONTEND._FusedLinearPPOFallbackFunction
    )


@pytest.mark.parametrize(
    ("token_count", "vocab_size", "element_size", "expected"),
    [
        (1, 131072, 2, 1),
        (2048, 131072, 2, 2048),
        (4096, 131072, 2, 2048),
        (4097, 131072, 2, 1366),
        (4096, 65536, 4, 2048),
    ],
)
def test_sm103_token_chunk_size_is_bounded_and_balanced(token_count, vocab_size, element_size, expected):
    assert _FRONTEND._calculate_sm103_token_chunk_size(token_count, vocab_size, element_size) == expected


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


def _require_sm103():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device_id = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_id) != (10, 3):
        pytest.skip("requires SM103")


def _scaled_ce_reference(hidden, weight, target, temperature, return_entropy):
    logits = hidden.float() @ weight.float().t()
    logits /= temperature
    valid = target != -100
    safe_target = target.masked_fill(~valid, 0)
    log_probs = logits.log_softmax(dim=-1)
    nll = torch.where(
        valid,
        -log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(log_probs[:, 0]),
    )
    if not return_entropy:
        return nll
    entropy = torch.where(
        valid,
        -(log_probs.exp() * log_probs).sum(dim=-1),
        torch.zeros_like(log_probs[:, 0]),
    ).to(hidden.dtype)
    return nll, entropy


@pytest.mark.parametrize(
    ("dtype", "shape", "temperature"),
    [
        (torch.float32, (17, 31, 257), 0.5),
        (torch.float16, (7, 33, 32769), 1.0),
        (torch.bfloat16, (16, 64, 4096), 0.8),
        (torch.bfloat16, (2, 16, 131072), 1.7),
    ],
    ids=["fp32-ragged", "fp16-wide-ragged", "bf16-power-of-two", "bf16-large-vocab"],
)
@pytest.mark.parametrize("gradient_mode", ["nll", "entropy", "mixed"])
def test_sm103_matches_fp32_reference(dtype, shape, temperature, gradient_mode):
    _require_sm103()
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")

    token_count, hidden_size, vocab_size = shape
    generator = torch.Generator(device="cuda").manual_seed(token_count + vocab_size)
    hidden_data = torch.randn(token_count, hidden_size, device="cuda", dtype=dtype, generator=generator) * 0.25
    weight_data = (
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, generator=generator) / hidden_size**0.5
    )
    target = torch.randint(vocab_size, (token_count,), device="cuda", generator=generator)
    target[::5] = -100
    grad_nll = torch.randn(token_count, device="cuda", dtype=torch.float32, generator=generator)
    grad_entropy = torch.randn(token_count, device="cuda", dtype=dtype, generator=generator)
    return_entropy = gradient_mode != "nll"

    actual_hidden = hidden_data.clone().requires_grad_(True)
    actual_weight = weight_data.clone().requires_grad_(True)
    actual = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_hidden,
        actual_weight,
        target,
        temperature,
        -100,
        1,
        return_entropy,
    )
    if gradient_mode == "nll":
        actual_outputs = (actual,)
        actual_grad_outputs = (grad_nll,)
    elif gradient_mode == "entropy":
        actual_outputs = (actual[1],)
        actual_grad_outputs = (grad_entropy,)
    else:
        actual_outputs = actual
        actual_grad_outputs = (grad_nll, grad_entropy)
    actual_grads = torch.autograd.grad(actual_outputs, (actual_hidden, actual_weight), actual_grad_outputs)

    expected_hidden = hidden_data.float().requires_grad_(True)
    expected_weight = weight_data.float().requires_grad_(True)
    expected = _scaled_ce_reference(expected_hidden, expected_weight, target, temperature, return_entropy)
    if gradient_mode == "nll":
        expected_outputs = (expected,)
        expected_grad_outputs = (grad_nll,)
    elif gradient_mode == "entropy":
        expected_outputs = (expected[1],)
        expected_grad_outputs = (grad_entropy.float(),)
    else:
        expected_outputs = expected
        expected_grad_outputs = (grad_nll, grad_entropy.float())
    expected_grads = torch.autograd.grad(expected_outputs, (expected_hidden, expected_weight), expected_grad_outputs)

    atol, rtol = {
        torch.float32: (6e-5, 6e-5),
        torch.float16: (3e-2, 3e-2),
        torch.bfloat16: (8e-2, 5e-2),
    }[dtype]
    actual_all_outputs = actual if return_entropy else (actual,)
    expected_all_outputs = expected if return_entropy else (expected,)
    for actual_output, expected_output in zip(actual_all_outputs, expected_all_outputs):
        torch.testing.assert_close(actual_output.float(), expected_output.float(), atol=atol, rtol=rtol)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), atol=atol, rtol=rtol)


def test_sm103_noncontiguous_inputs_partial_grads_repeated_backward_and_context(monkeypatch):
    _require_sm103()
    generator = torch.Generator(device="cuda").manual_seed(47)
    hidden_base = torch.randn(11, 66, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight_base = torch.randn(259, 66, device="cuda", dtype=torch.bfloat16, generator=generator)
    hidden = hidden_base[:, ::2].requires_grad_(True)
    weight = weight_base[:, ::2]
    target = torch.randint(259, (22,), device="cuda", generator=generator)[::2]
    grad = torch.randn(22, device="cuda", dtype=torch.float32, generator=generator)[::2]
    hidden_before = hidden.detach().clone()
    weight_before = weight.detach().clone()
    seen_devices = []
    real_device_context = _FRONTEND.device_context

    @contextmanager
    def recording_device_context(device):
        seen_devices.append(torch.device(device))
        with real_device_context(device):
            yield

    monkeypatch.setattr(_FRONTEND, "device_context", recording_device_context)
    output = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(hidden, weight, target, 1.0, -100, 1, False)
    first_grad = torch.autograd.grad(output, hidden, grad, retain_graph=True)[0]
    second_grad = torch.autograd.grad(output, hidden, grad)[0]

    assert not hidden.is_contiguous()
    assert not weight.is_contiguous()
    assert not target.is_contiguous()
    assert not grad.is_contiguous()
    assert output.shape == target.shape
    assert output.dtype == torch.float32
    torch.testing.assert_close(first_grad, second_grad, atol=0, rtol=0)
    torch.testing.assert_close(hidden, hidden_before, atol=0, rtol=0)
    torch.testing.assert_close(weight, weight_before, atol=0, rtol=0)
    assert seen_devices == [hidden.device, hidden.device, hidden.device]


def test_sm103_all_ignored_rows_return_zero_outputs_and_gradients():
    _require_sm103()
    hidden = torch.randn(5, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(513, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.full((5,), -100, device="cuda", dtype=torch.int64)

    nll, entropy = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        hidden,
        weight,
        target,
        1.0,
        -100,
        1,
        True,
    )
    torch.autograd.backward((nll, entropy), (torch.randn_like(nll), torch.randn_like(entropy)))

    assert torch.count_nonzero(nll) == 0
    assert torch.count_nonzero(entropy) == 0
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


def test_sm103_weight_only_gradient_and_compiled_forward_match_eager():
    _require_sm103()
    from liger_kernel.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    generator = torch.Generator(device="cuda").manual_seed(48)
    hidden = torch.randn(13, 35, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(1023, 35, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_(True)
    target = torch.randint(1023, (13,), device="cuda", generator=generator)
    grad = torch.randn(13, device="cuda", dtype=torch.float32, generator=generator)

    def forward(input_, weight_, labels):
        return LigerFusedLinearScaledCrossEntropyFunction.apply(
            input_,
            weight_,
            labels,
            0.8,
            -100,
            1,
            False,
        )

    eager = forward(hidden, weight, target)
    eager_grad = torch.autograd.grad(eager, weight, grad, retain_graph=True)[0]
    compiled = torch.compile(forward)(hidden, weight, target)
    compiled_grad = torch.autograd.grad(compiled, weight, grad)[0]

    torch.testing.assert_close(compiled, eager, atol=0, rtol=0)
    torch.testing.assert_close(compiled_grad, eager_grad, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("shape", "dtype", "atol", "rtol"),
    [
        ((17, 31, 257), torch.float32, 6e-5, 6e-5),
        ((1025, 64, 131072), torch.bfloat16, 8e-2, 5e-2),
    ],
    ids=["fp32-ragged", "bf16-multichunk"],
)
def test_cutile_matches_verl_fallback_forward_and_combined_backward(shape, dtype, atol, rtol):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")

    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    torch.manual_seed(42)
    token_count, hidden_size, vocab_size = shape
    temperature = 0.8
    target = torch.randint(vocab_size, (token_count,), device="cuda", dtype=torch.int64)
    target[::11] = -100
    grad_nll = torch.randn(token_count, device="cuda", dtype=torch.float32)
    grad_entropy = torch.randn(token_count, device="cuda", dtype=dtype)

    actual_input = (torch.randn(token_count, hidden_size, device="cuda", dtype=dtype) * 0.25).requires_grad_(True)
    actual_weight = (
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype) / hidden_size**0.5
    ).requires_grad_(True)
    actual_nll, actual_entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_input,
        actual_weight,
        target,
        temperature,
        -100,
        3,
        True,
    )
    torch.autograd.backward((actual_nll, actual_entropy), (grad_nll, grad_entropy))

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    expected_nll, expected_entropy = _FRONTEND._FusedLinearPPOFallbackFunction.apply(
        expected_input,
        expected_weight,
        target,
        temperature,
        -100,
        True,
    )
    torch.autograd.backward((expected_nll, expected_entropy), (grad_nll, grad_entropy))

    torch.testing.assert_close(actual_nll, expected_nll, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_entropy, expected_entropy, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize("output", ["nll", "entropy"])
def test_cutile_supports_single_output_backward(output):
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    torch.manual_seed(43)
    token_count, hidden_size, vocab_size = 19, 33, 259
    target = torch.randint(vocab_size, (token_count,), device="cuda", dtype=torch.int64)
    target[::5] = -100
    grad_output = torch.randn(token_count, device="cuda")

    actual_input = (torch.randn(token_count, hidden_size, device="cuda") * 0.25).requires_grad_(True)
    actual_weight = (torch.randn(vocab_size, hidden_size, device="cuda") / hidden_size**0.5).requires_grad_(True)
    actual_nll, actual_entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
        actual_input,
        actual_weight,
        target,
        1.2,
        -100,
        2,
        True,
    )
    (actual_nll if output == "nll" else actual_entropy).backward(grad_output)

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    expected_nll, expected_entropy = _FRONTEND._FusedLinearPPOFallbackFunction.apply(
        expected_input,
        expected_weight,
        target,
        1.2,
        -100,
        True,
    )
    (expected_nll if output == "nll" else expected_entropy).backward(grad_output)

    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=6e-5, rtol=6e-5)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=6e-5, rtol=6e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("token_count", "vocab_size", "element_size", "expected"),
    [
        (4096, 131072, 2, 1024),
        (1025, 131072, 2, 513),
        (2500, 100000, 2, 1250),
        (5000, 131072, 2, 1024),
    ],
)
def test_cutile_token_chunk_size_balances_small_tails(token_count, vocab_size, element_size, expected):
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import _calculate_token_chunk_size

    assert _calculate_token_chunk_size(token_count, vocab_size, element_size) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize(
    ("architecture", "token_count", "vocab_size", "element_size", "expected_mb"),
    [
        ("hopper", 4096, 131072, 2, 256),
        ("blackwell", 4096, 131072, 2, 512),
        ("blackwell_ultra", 4096, 131072, 2, 512),
        ("blackwell", 2048, 131072, 2, 256),
        ("blackwell", 4096, 65536, 2, 256),
        ("blackwell", 4096, 131072, 4, 256),
    ],
)
def test_cutile_selects_workspace_by_architecture(
    monkeypatch,
    architecture,
    token_count,
    vocab_size,
    element_size,
    expected_mb,
):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.delenv(scaled_ce._WORKSPACE_MB_ENV, raising=False)
    monkeypatch.setattr(scaled_ce, "infer_device_arch", lambda _: architecture)

    actual = scaled_ce._select_logits_workspace_bytes(0, token_count, vocab_size, element_size)
    assert actual == expected_mb * scaled_ce._MIB


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
def test_cutile_workspace_override(monkeypatch):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.setenv(scaled_ce._WORKSPACE_MB_ENV, "1024")
    workspace_bytes = scaled_ce._select_logits_workspace_bytes(0, 4096, 131072, 2)

    assert workspace_bytes == 1024 * scaled_ce._MIB
    assert scaled_ce._calculate_token_chunk_size(4096, 131072, 2, workspace_bytes) == 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_cutile_rejects_invalid_workspace_override(monkeypatch, value):
    import liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy as scaled_ce

    monkeypatch.setenv(scaled_ce._WORKSPACE_MB_ENV, value)
    with pytest.raises(ValueError, match=scaled_ce._WORKSPACE_MB_ENV):
        scaled_ce._select_logits_workspace_bytes(0, 4096, 131072, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile requires CUDA")
@pytest.mark.skipif(not _CUTILE_ENABLED, reason="requires LIGER_KERNEL_IMPL=cutile")
def test_cutile_rejects_out_of_range_target():
    from liger_kernel.ops.cutile.ops.fused_linear_scaled_cross_entropy import LigerFusedLinearScaledCrossEntropyFunction

    _input = torch.randn(2, 8, device="cuda")
    weight = torch.randn(16, 8, device="cuda")
    target = torch.tensor([0, 16], device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match=r"outside \[0, 16\)"):
        LigerFusedLinearScaledCrossEntropyFunction.apply(_input, weight, target)


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
