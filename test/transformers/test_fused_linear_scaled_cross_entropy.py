import builtins
import importlib.util
import inspect
import logging
import os

from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError
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

_NATIVE_FRONTEND_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "liger_kernel"
    / "ops"
    / "cute"
    / "fused_linear_scaled_cross_entropy_tp.py"
)
_NATIVE_SPEC = importlib.util.spec_from_file_location(
    "_fused_linear_scaled_cross_entropy_native_frontend", _NATIVE_FRONTEND_PATH
)
assert _NATIVE_SPEC is not None and _NATIVE_SPEC.loader is not None
_NATIVE_FRONTEND = importlib.util.module_from_spec(_NATIVE_SPEC)
_NATIVE_SPEC.loader.exec_module(_NATIVE_FRONTEND)

_CUTILE_ENABLED = os.environ.get("LIGER_KERNEL_IMPL", "").strip().lower() == "cutile"
_SM100_MODULE = "liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm100"


@pytest.fixture
def sm100_loader():
    loader = _FRONTEND._load_sm100_function
    loader.cache_clear()
    yield loader
    loader.cache_clear()


@pytest.fixture
def sm100_imports(monkeypatch):
    native = object()
    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "cutlass.cute":
            return SimpleNamespace(cute=object())
        if name == _SM100_MODULE:
            return SimpleNamespace(LigerFusedScaledCrossEntropySM100Function=native)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    return native


@pytest.mark.parametrize("installed_version", [None, "4.5.0", "4.6.0rc1", "invalid"])
def test_sm100_loader_rejects_missing_or_unsupported_version_once(monkeypatch, caplog, sm100_loader, installed_version):
    def installed(name):
        assert name == "nvidia-cutlass-dsl"
        if installed_version is None:
            raise PackageNotFoundError(name)
        return installed_version

    original_import = builtins.__import__

    def reject_cute_import(name, *args, **kwargs):
        if name.startswith("cutlass") or name == _SM100_MODULE:
            pytest.fail("An unsupported CuTe DSL version must not import native kernels")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(_FRONTEND, "version", installed)
    monkeypatch.setattr(builtins, "__import__", reject_cute_import)
    with caplog.at_level(logging.WARNING, logger=_FRONTEND.__name__):
        assert sm100_loader() is None
        assert sm100_loader() is None
    messages = [record.getMessage() for record in caplog.records if record.name == _FRONTEND.__name__]
    assert len(messages) == 1
    assert "using the PyTorch fallback" in messages[0]
    assert "nvidia-cutlass-dsl>=4.6.0" in messages[0]


@pytest.mark.parametrize("failed_module", ["cutlass.cute", _SM100_MODULE])
def test_sm100_loader_falls_back_on_import_error_once(monkeypatch, caplog, sm100_loader, sm100_imports, failed_module):
    monkeypatch.setattr(_FRONTEND, "version", lambda _: "4.6.0")
    original_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == failed_module:
            if name == "cutlass.cute":
                raise ModuleNotFoundError("No module named 'cutlass.cute'", name=name)
            raise ImportError("missing optional CuTe binding")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with caplog.at_level(logging.WARNING, logger=_FRONTEND.__name__):
        assert sm100_loader() is None
        assert sm100_loader() is None
    messages = [record.getMessage() for record in caplog.records if record.name == _FRONTEND.__name__]
    assert len(messages) == 1
    assert "using the PyTorch fallback" in messages[0]


@pytest.mark.parametrize("installed_version", ["4.6.0", "4.10.0"])
def test_sm100_loader_accepts_supported_versions(monkeypatch, sm100_loader, sm100_imports, installed_version):
    monkeypatch.setattr(_FRONTEND, "version", lambda _: installed_version)

    assert sm100_loader() is sm100_imports


def test_sm100_loader_does_not_hide_unexpected_errors(monkeypatch, sm100_loader, sm100_imports):
    monkeypatch.setattr(_FRONTEND, "version", lambda _: "4.6.0")
    original_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == _SM100_MODULE:
            raise RuntimeError("unexpected native initialization error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(RuntimeError, match="unexpected native initialization error"):
        sm100_loader()


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_frontend_blackwell_opt_out_before_and_after_loading(monkeypatch, sm100_loader, sm100_imports, capability):
    monkeypatch.setattr(_FRONTEND, "version", lambda _: "4.6.0")
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: capability)
    device = torch.device("cuda:0")
    monkeypatch.setenv("LIGER_DISABLE_SM100_SCALED_CE", "1")
    assert _FRONTEND._resolve_implementation(device, torch.bfloat16) is _FRONTEND._FusedLinearPPOFallbackFunction
    monkeypatch.setenv("LIGER_DISABLE_SM100_SCALED_CE", "0")
    assert _FRONTEND._resolve_implementation(device, torch.bfloat16) is sm100_imports
    monkeypatch.setenv("LIGER_DISABLE_SM100_SCALED_CE", "1")
    assert _FRONTEND._resolve_implementation(device, torch.bfloat16) is _FRONTEND._FusedLinearPPOFallbackFunction


def test_frontend_blackwell_opt_out_does_not_disable_hopper(monkeypatch):
    native = object()
    monkeypatch.setenv("LIGER_DISABLE_SM100_SCALED_CE", "1")
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(_FRONTEND, "_load_sm90_function", lambda: native)

    assert _FRONTEND._resolve_implementation(torch.device("cuda:0"), torch.bfloat16) is native


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


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_frontend_dispatches_blackwell_without_global_override(monkeypatch, capability, accum_dtype):
    device = torch.device("cuda:2")
    x = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(dtype=torch.bfloat16)
    target, selected_ids = object(), object()
    calls = []
    seen_devices = []

    class NativeFunction:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return "blackwell"

    monkeypatch.delenv("LIGER_KERNEL_IMPL", raising=False)
    monkeypatch.delenv("LIGER_DISABLE_SM100_SCALED_CE", raising=False)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(
        _FRONTEND.torch.cuda, "get_device_capability", lambda actual: seen_devices.append(actual) or capability
    )
    monkeypatch.setattr(_FRONTEND, "_load_sm100_function", lambda: NativeFunction)

    result = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(
        x, weight, target, 0.7, -1, 2, True, selected_ids, accum_dtype
    )

    assert result == "blackwell"
    accum_args = (accum_dtype,) if accum_dtype is not None else ()
    assert calls == [(x, weight, target, 0.7, -1, 2, True, selected_ids, *accum_args)]
    assert seen_devices == [device]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_frontend_blackwell_retains_fallback_for_unsupported_dtype(monkeypatch, dtype):
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(_FRONTEND, "_load_sm100_function", lambda: pytest.fail("unsupported dtype must not load CuTe"))

    assert _FRONTEND._resolve_implementation(torch.device("cuda:0"), dtype) is _FRONTEND._FusedLinearPPOFallbackFunction


def test_frontend_blackwell_retains_fallback_without_cute(monkeypatch):
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(_FRONTEND, "_load_sm100_function", lambda: None)

    assert (
        _FRONTEND._resolve_implementation(torch.device("cuda:0"), torch.bfloat16)
        is _FRONTEND._FusedLinearPPOFallbackFunction
    )


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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backward_inside_autocast", [False, True])
def test_fallback_fp32_parameters_backward_under_cuda_autocast(autocast_dtype, backward_inside_autocast):
    if not _FRONTEND._SUPPORTS_OUT_DTYPE or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires the CUDA FP32-output addmm path")
    tokens, hidden, vocab = 521, 32, 16
    x = (
        torch.linspace(0.125, 1.125, tokens * hidden, device="cuda", dtype=torch.float32)
        .reshape(tokens, hidden)
        .requires_grad_()
    )
    weight = torch.zeros(vocab, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.zeros(tokens, device="cuda", dtype=torch.int64)
    target[::11] = -100
    upstream = torch.where(torch.arange(tokens, device="cuda") % 2 == 0, 0.5, -0.25)

    with torch.autocast("cuda", dtype=autocast_dtype):
        nll = _FRONTEND._FusedLinearPPOFallbackFunction.apply(x, weight, target, 1.0, -100, False)
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=backward_inside_autocast):
        nll.backward(upstream)

    # Zero weights give uniform probabilities, so dW is an analytic weighted sum of X.
    projected_x = x.detach().to(autocast_dtype).float() if backward_inside_autocast else x.detach()
    weighted_x = (projected_x * upstream.masked_fill(target == -100, 0)[:, None]).sum(dim=0)
    class_gradient = torch.full((vocab, 1), 1 / vocab, device="cuda")
    class_gradient[0] -= 1
    expected_dw = class_gradient * weighted_x
    assert x.grad.dtype == torch.float32 and weight.grad.dtype == torch.float32
    torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, expected_dw, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize(
    "capability",
    [(9, 0), (10, 0), (10, 3)],
    ids=["hopper", "blackwell-b200", "blackwell-b300"],
)
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_tp_frontend_dispatches_supported_bf16_cuda_to_native(monkeypatch, capability, accum_dtype):
    process_group = object()
    device = torch.device("cuda:3")
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(shape=(16, 32), dtype=torch.bfloat16)
    target = object()
    calls = []
    seen = []

    class StubNativeFunction:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return "native-result"

    monkeypatch.setattr(_FRONTEND, "_tp_group_info", lambda actual: (actual, 2))
    monkeypatch.setattr(_FRONTEND, "_validate_tp_inputs", lambda *args: None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(
        _FRONTEND.torch.cuda,
        "get_device_capability",
        lambda actual: seen.append(actual) or capability,
    )
    monkeypatch.setattr(_FRONTEND, "_load_native_tp_function", lambda: StubNativeFunction)
    monkeypatch.setattr(
        _FRONTEND,
        "_apply_tp_fallback",
        lambda *args: pytest.fail("the Liger fallback must not run when the native package is available"),
    )

    result = _FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction.apply(
        _input,
        weight,
        target,
        process_group,
        0.5,
        -1,
        2,
        True,
        accum_dtype,
    )

    assert result == "native-result"
    accum_args = (accum_dtype,) if accum_dtype is not None else ()
    assert calls == [(_input, weight, target, 32, 0.5, -1, 2, True, process_group, *accum_args)]
    assert seen == [device]


def test_tp_frontend_uses_fallback_for_unsupported_nvidia_capability(monkeypatch):
    process_group = object()
    device = torch.device("cuda:0")
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(shape=(8, 16))
    target = object()
    calls = []

    monkeypatch.setattr(_FRONTEND, "_tp_group_info", lambda actual: (actual, 1))
    monkeypatch.setattr(_FRONTEND, "_validate_tp_inputs", lambda *args: None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda actual: (8, 9))
    monkeypatch.setattr(
        _FRONTEND,
        "_load_native_tp_function",
        lambda: pytest.fail("native TP must not load for an unsupported capability"),
    )
    monkeypatch.setattr(
        _FRONTEND,
        "_apply_tp_fallback",
        lambda *args: calls.append(args) or "fallback-result",
    )

    result = _FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction.apply(
        _input,
        weight,
        target,
        process_group,
    )

    assert result == "fallback-result"
    assert calls == [(_input, weight, target, 1.0, -100, False, process_group)]


def test_tp_group_is_a_call_argument():
    parameters = inspect.signature(_FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction.apply).parameters

    assert list(parameters)[3] == "tp_group"


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize(
    "accum_dtype,error", [("float32", TypeError), (torch.int32, ValueError), (torch.float64, ValueError)]
)
def test_accum_dtype_rejected_before_dispatch(tp, accum_dtype, error):
    x = torch.zeros(3, 4, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    target = torch.zeros(3, dtype=torch.int64)
    function = (
        _FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction
        if tp
        else _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction
    )
    args = (x, weight, target, object()) if tp else (x, weight, target)
    with pytest.raises(error, match="accum_dtype"):
        function.apply(*args, accum_dtype=accum_dtype)


@pytest.mark.parametrize("tp", [False, True])
def test_fallback_fp32_accumulation_rounds_only_final_weight_gradient(monkeypatch, tp):
    monkeypatch.setattr(_FRONTEND, "_FALLBACK_CHUNK_SIZE", 16)
    monkeypatch.setattr(_FRONTEND.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(_FRONTEND.dist, "get_world_size", lambda group: 1)
    tokens, hidden, vocab = 4097, 7, 4
    target = torch.zeros(tokens, dtype=torch.int64)
    target[::7] = -100
    gradients = []
    for accum_dtype in (None, torch.float32, torch.bfloat16):
        x = torch.full((tokens, hidden), 0.125, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.zeros(vocab, hidden, dtype=torch.bfloat16, requires_grad=True)
        if tp:
            nll = _FRONTEND._apply_tp_fallback(x, weight, target, 1.0, -100, False, object(), accum_dtype)
        else:
            nll = _FRONTEND.LigerFusedLinearScaledCrossEntropyFunction.apply(x, weight, target, accum_dtype=accum_dtype)
        nll.sum().backward()
        assert x.grad.dtype == x.dtype and weight.grad.dtype == weight.dtype
        assert torch.count_nonzero(x.grad) == 0
        gradients.append(weight.grad)

    expected = torch.full((vocab, hidden), 1 / vocab)
    expected[0] -= 1
    expected *= (target != -100).sum() * 0.125
    torch.testing.assert_close(gradients[1], expected.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[2], atol=0, rtol=0)
    assert not torch.equal(gradients[0], gradients[1])


@pytest.mark.parametrize("native_result", [None, ImportError("native package unavailable")])
def test_tp_frontend_uses_liger_fallback_when_native_is_unavailable(monkeypatch, native_result):
    process_group = object()
    device = torch.device("cuda:0")
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(shape=(8, 16))
    target = object()
    calls = []

    monkeypatch.setattr(_FRONTEND, "_tp_group_info", lambda actual: (actual, 1))
    monkeypatch.setattr(_FRONTEND, "_validate_tp_inputs", lambda *args: None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_FRONTEND.torch.version, "hip", None)
    monkeypatch.setattr(_FRONTEND.torch.cuda, "get_device_capability", lambda actual: (10, 3))

    def load_native():
        if isinstance(native_result, BaseException):
            raise native_result
        return native_result

    monkeypatch.setattr(_FRONTEND, "_load_native_tp_function", load_native)
    monkeypatch.setattr(
        _FRONTEND,
        "_apply_tp_fallback",
        lambda *args: calls.append(args) or "fallback-result",
    )

    result = _FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction.apply(
        _input,
        weight,
        target,
        process_group,
    )

    assert result == "fallback-result"
    assert calls == [(_input, weight, target, 1.0, -100, False, process_group)]


def test_tp_fallback_matches_torch_with_entropy_and_ignored_rows(monkeypatch):
    process_group = object()
    token_count, hidden_size, vocab_size = 521, 7, 13
    temperature = 0.7
    target = torch.randint(vocab_size, (token_count,), dtype=torch.int64)
    target[::5] = -100
    grad_nll = torch.rand(token_count)
    grad_entropy = torch.rand(token_count) - 0.5

    monkeypatch.setattr(_FRONTEND.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(_FRONTEND.dist, "get_world_size", lambda group: 1)

    actual_input = torch.randn(token_count, hidden_size, requires_grad=True)
    actual_weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
    actual_nll, actual_entropy = _FRONTEND._apply_tp_fallback(
        actual_input,
        actual_weight,
        target,
        temperature,
        -100,
        True,
        process_group,
    )
    torch.autograd.backward((actual_nll, actual_entropy), (grad_nll, grad_entropy))

    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_weight = actual_weight.detach().clone().requires_grad_(True)
    logits = (expected_input @ expected_weight.t()).float() / temperature
    log_probs = logits.log_softmax(dim=-1)
    probabilities = log_probs.exp()
    valid = target != -100
    safe_target = target.masked_fill(~valid, 0)
    expected_nll = torch.where(
        valid,
        -log_probs.gather(-1, safe_target[:, None]).squeeze(-1),
        torch.zeros(token_count),
    )
    expected_entropy = torch.where(
        valid,
        -(probabilities * log_probs).sum(dim=-1),
        torch.zeros(token_count),
    )
    torch.autograd.backward((expected_nll, expected_entropy), (grad_nll, grad_entropy))

    torch.testing.assert_close(actual_nll, expected_nll)
    torch.testing.assert_close(actual_entropy, expected_entropy)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_native_tp_autograd_forwards_runtime_and_inverse_temperature(monkeypatch, accum_dtype):
    calls = {}

    class FakeTvmFfi:
        @staticmethod
        def fused_linear_scaled_cross_entropy_forward(*args):
            calls["forward"] = args
            tokens = args[0].shape[0]
            return torch.arange(tokens, dtype=torch.float32), torch.ones(tokens), torch.full((tokens,), 2.0)

        @staticmethod
        def fused_linear_scaled_cross_entropy_backward(*args):
            calls["backward"] = args
            return torch.full_like(args[2], 3.0), torch.full_like(args[3], 4.0, dtype=accum_dtype)

    monkeypatch.setattr(_NATIVE_FRONTEND, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(_NATIVE_FRONTEND, "_pad_hidden", lambda x, weight: (x, weight, x.shape[1]))
    monkeypatch.setattr(_NATIVE_FRONTEND, "_prepare_native_call", lambda *args: 17)
    monkeypatch.setattr(_NATIVE_FRONTEND, "_get_tvm_ffi", lambda: FakeTvmFfi)

    _input = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([0, 1, 2], dtype=torch.int64)
    nll, entropy = _NATIVE_FRONTEND.LigerFusedLinearScaledCrossEntropyNativeTPFunction.apply(
        _input,
        weight,
        target,
        10,
        0.5,
        -100,
        2,
        True,
        object(),
        accum_dtype,
    )
    torch.autograd.backward((nll, entropy), (torch.ones_like(nll), torch.full_like(entropy, 0.25)))

    assert calls["forward"][3:] == (10, -100, 2.0, 17, True)
    torch.testing.assert_close(calls["backward"][0], torch.ones(3))
    torch.testing.assert_close(calls["backward"][1], torch.full((3,), 0.25))
    accum_args = (accum_dtype,) if accum_dtype is not None else ()
    assert calls["backward"][7:] == (10, -100, 2.0, 17, 2, True, *accum_args)
    torch.testing.assert_close(_input.grad, torch.full_like(_input, 3.0))
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 4.0))


def test_native_call_uses_prepared_group_without_global_state(monkeypatch):
    process_group = object()
    calls = []

    class FakeNvshmem:
        @staticmethod
        def resolve_team(group, *, create=True):
            calls.append(("team", group, create))
            return 17

    class FakeTvmFfi:
        @staticmethod
        def fused_linear_scaled_cross_entropy_configure_backward(*args):
            calls.append(("backward_config", args))

        @staticmethod
        def fused_linear_scaled_cross_entropy_configure_forward(*args):
            calls.append(("forward_config", args))

    monkeypatch.setattr(_NATIVE_FRONTEND, "_get_nvshmem", lambda: FakeNvshmem)
    monkeypatch.setattr(_NATIVE_FRONTEND, "_get_tvm_ffi", lambda: FakeTvmFfi)
    monkeypatch.setattr(_NATIVE_FRONTEND.torch.cuda, "device", lambda device: nullcontext())

    team = _NATIVE_FRONTEND._prepare_native_call(
        process_group,
        torch.device("cuda:1"),
        128,
        2048,
        320,
        2,
    )
    repeated = _NATIVE_FRONTEND._prepare_native_call(
        process_group,
        torch.device("cuda:1"),
        64,
        1024,
        160,
        1,
    )
    assert team == 17
    assert repeated == 17
    assert calls == [
        ("team", process_group, False),
        ("backward_config", (128, 2048, 320, 2, 17)),
        ("forward_config", (128, 320, 17)),
        ("team", process_group, False),
        ("backward_config", (64, 1024, 160, 1, 17)),
        ("forward_config", (64, 160, 17)),
    ]


@pytest.mark.parametrize("supports_contexts", [False, True])
def test_native_availability_requires_context_aware_bindings(monkeypatch, supports_contexts):
    class NativeModule:
        fused_linear_scaled_cross_entropy_configure_backward = object()
        fused_linear_scaled_cross_entropy_configure_forward = object()
        fused_linear_scaled_cross_entropy_backward = object()
        fused_linear_scaled_cross_entropy_forward = object()

    class Bindings:
        @staticmethod
        def _load_module():
            return NativeModule()

    class Teams:
        @staticmethod
        def resolve_team(*args, **kwargs):
            raise AssertionError("availability must not resolve teams")

    if supports_contexts:
        NativeModule.fused_linear_scaled_cross_entropy_configure_context = object()
    monkeypatch.setattr(_NATIVE_FRONTEND, "_get_tvm_ffi", lambda: Bindings)
    monkeypatch.setattr(_NATIVE_FRONTEND, "_get_nvshmem", lambda: Teams)
    assert _NATIVE_FRONTEND.is_available() is supports_contexts


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
    assert ops.LigerFusedLinearScaledCrossEntropyTPFunction.__module__ == (
        "liger_kernel.ops.fused_linear_scaled_cross_entropy"
    )
