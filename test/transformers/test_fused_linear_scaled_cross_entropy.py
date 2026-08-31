import importlib.util
import inspect
import os

from contextlib import nullcontext
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

_LCK_FRONTEND_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "liger_kernel"
    / "ops"
    / "cute"
    / "fused_linear_scaled_cross_entropy_tp.py"
)
_LCK_SPEC = importlib.util.spec_from_file_location(
    "_fused_linear_scaled_cross_entropy_lck_frontend", _LCK_FRONTEND_PATH
)
assert _LCK_SPEC is not None and _LCK_SPEC.loader is not None
_LCK_FRONTEND = importlib.util.module_from_spec(_LCK_SPEC)
_LCK_SPEC.loader.exec_module(_LCK_FRONTEND)

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


def test_tp_frontend_dispatches_hopper_bf16_to_lck(monkeypatch):
    process_group = object()
    device = torch.device("cuda:3")
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(shape=(16, 32))
    target = object()
    calls = []

    class StubLckFunction:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return "lck-result"

    monkeypatch.setattr(_FRONTEND, "_tp_group_info", lambda actual: (actual, 2))
    monkeypatch.setattr(_FRONTEND, "_validate_tp_inputs", lambda *args: None)
    monkeypatch.setattr(_FRONTEND, "_is_hopper", lambda actual: actual == device)
    monkeypatch.setattr(_FRONTEND, "_load_lck_tp_function", lambda group, actual_device: StubLckFunction)
    monkeypatch.setattr(
        _FRONTEND,
        "_apply_tp_fallback",
        lambda *args: pytest.fail("the Liger fallback must not run when LCK is available"),
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
    )

    assert result == "lck-result"
    assert calls == [(_input, weight, target, 32, 0.5, -1, 2, True, process_group)]


def test_tp_group_is_a_call_argument():
    parameters = inspect.signature(_FRONTEND.LigerFusedLinearScaledCrossEntropyTPFunction.apply).parameters

    assert list(parameters)[3] == "tp_group"


@pytest.mark.parametrize("lck_result", [None, ImportError("lck unavailable")])
def test_tp_frontend_uses_liger_fallback_when_lck_is_unavailable(monkeypatch, lck_result):
    process_group = object()
    device = torch.device("cuda:0")
    _input = SimpleNamespace(device=device, dtype=torch.bfloat16)
    weight = SimpleNamespace(shape=(8, 16))
    target = object()
    calls = []

    monkeypatch.setattr(_FRONTEND, "_tp_group_info", lambda actual: (actual, 1))
    monkeypatch.setattr(_FRONTEND, "_validate_tp_inputs", lambda *args: None)
    monkeypatch.setattr(_FRONTEND, "_is_hopper", lambda actual: True)

    def load_lck(group, actual_device):
        if isinstance(lck_result, BaseException):
            raise lck_result
        return lck_result

    monkeypatch.setattr(_FRONTEND, "_load_lck_tp_function", load_lck)
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


def test_lck_tp_autograd_forwards_runtime_and_inverse_temperature(monkeypatch):
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
            return torch.full_like(args[2], 3.0), torch.full_like(args[3], 4.0)

    monkeypatch.setattr(_LCK_FRONTEND, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(_LCK_FRONTEND, "_pad_hidden", lambda x, weight: (x, weight, x.shape[1]))
    monkeypatch.setattr(_LCK_FRONTEND, "_prepare_lck_call", lambda *args: 17)
    monkeypatch.setattr(_LCK_FRONTEND, "_borrow_nccl_communicator", lambda group, device: 23)
    monkeypatch.setattr(_LCK_FRONTEND, "_get_tvm_ffi", lambda: FakeTvmFfi)

    _input = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2], dtype=torch.int64)
    nll, entropy = _LCK_FRONTEND.LigerFusedLinearScaledCrossEntropyLckTPFunction.apply(
        _input,
        weight,
        target,
        10,
        0.5,
        -100,
        2,
        True,
        object(),
    )
    torch.autograd.backward((nll, entropy), (torch.ones_like(nll), torch.full_like(entropy, 0.25)))

    assert calls["forward"][3:] == (10, -100, 2.0, 23, True)
    torch.testing.assert_close(calls["backward"][0], torch.ones(3))
    torch.testing.assert_close(calls["backward"][1], torch.full((3,), 0.25))
    assert calls["backward"][7:] == (10, -100, 2.0, 17, 2, True)
    torch.testing.assert_close(_input.grad, torch.full_like(_input, 3.0))
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 4.0))


def test_lck_call_uses_prepared_group_without_global_state(monkeypatch):
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

    monkeypatch.setattr(_LCK_FRONTEND, "_get_nvshmem", lambda: FakeNvshmem)
    monkeypatch.setattr(_LCK_FRONTEND, "_get_tvm_ffi", lambda: FakeTvmFfi)
    monkeypatch.setattr(_LCK_FRONTEND.torch.cuda, "device", lambda device: nullcontext())

    team = _LCK_FRONTEND._prepare_lck_call(
        process_group,
        torch.device("cuda:1"),
        128,
        2048,
        320,
        2,
    )
    repeated = _LCK_FRONTEND._prepare_lck_call(
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
        ("forward_config", (128, 320)),
        ("team", process_group, False),
        ("backward_config", (64, 1024, 160, 1, 17)),
        ("forward_config", (64, 160)),
    ]


def test_lck_borrows_matching_process_group_nccl_communicator(monkeypatch):
    calls = []
    comm_handles = iter((0, 23))
    device = torch.device("cuda:2")

    class FakeBackend:
        @staticmethod
        def get_runtime_nccl_version():
            return (2, 27, 6)

        @staticmethod
        def eager_connect_single_device(actual_device):
            calls.append(("eager", actual_device))

        @staticmethod
        def _comm_ptr():
            calls.append(("comm_ptr",))
            return next(comm_handles)

    class FakeProcessGroup:
        @staticmethod
        def _get_backend(actual_device):
            calls.append(("backend", actual_device))
            return FakeBackend()

    fake_tvm_ffi = SimpleNamespace(nccl_runtime_version=lambda: (2, 27, 6))
    monkeypatch.setattr(_LCK_FRONTEND, "_get_tvm_ffi", lambda: fake_tvm_ffi)

    assert _LCK_FRONTEND._borrow_nccl_communicator(FakeProcessGroup(), device) == 23
    assert calls == [("backend", device), ("comm_ptr",), ("eager", device), ("comm_ptr",)]


def test_lck_rejects_mismatched_nccl_runtime(monkeypatch):
    class FakeBackend:
        @staticmethod
        def get_runtime_nccl_version():
            return (2, 27, 6)

        @staticmethod
        def eager_connect_single_device(device):
            pytest.fail(f"must not initialize a mismatched communicator on {device}")

        @staticmethod
        def _comm_ptr():
            return 23

    class FakeProcessGroup:
        @staticmethod
        def _get_backend(device):
            return FakeBackend()

    monkeypatch.setattr(
        _LCK_FRONTEND,
        "_get_tvm_ffi",
        lambda: SimpleNamespace(nccl_runtime_version=lambda: (2, 26, 5)),
    )

    with pytest.raises(RuntimeError, match="NCCL runtime versions must match"):
        _LCK_FRONTEND._borrow_nccl_communicator(FakeProcessGroup(), torch.device("cuda:0"))


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
