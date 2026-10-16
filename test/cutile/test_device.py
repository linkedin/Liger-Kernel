"""Device/stream contracts for the six thin cuTile adapters."""

import importlib

from contextlib import contextmanager
from contextlib import nullcontext

import pytest
import torch

ct = pytest.importorskip("cuda.tile")

from liger_kernel.backends import dispatch  # noqa: E402
from test.cutile.test_cutile_backends_parity import _cutile_impl  # noqa: E402
from test.cutile.test_rope import _rotate_reference  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_FUNCTIONS = {
    "cross_entropy": "LigerCrossEntropyFunction",
    "fused_add_rms_norm": "LigerFusedAddRMSNormFunction",
    "geglu": "LigerGELUMulFunction",
    "kl_div": "LigerKLDivLossFunction",
    "rope": "LigerRopeFunction",
    "swiglu": "LigerSiLUMulFunction",
}
_CASES = [
    ("cross_entropy", 64),
    ("fused_add_rms_norm", 512),
    ("fused_add_rms_norm", 8192),
    ("geglu", 512),
    ("kl_div", 64),
    ("kl_div", 31),
    ("rope", 8),
    ("rope", 12),
    ("swiglu", 512),
    ("swiglu", 7),
]


def _inputs(op, width, device):
    def randn(*shape):
        return torch.randn(*shape, device=device, requires_grad=True)

    if op == "rope":
        angles = torch.randn(1, 4, width // 2, device=device)
        return randn(1, 2, 4, width), randn(1, 1, 4, width), angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)
    x = randn(4, width)
    if op == "cross_entropy":
        return x, torch.arange(4, device=device), None, -100, 0.0, 0.0, "mean", None, False
    if op == "fused_add_rms_norm":
        return x, randn(4, width), randn(width), 1e-6
    if op == "kl_div":
        prediction = x.detach().log_softmax(-1).requires_grad_(True)
        target = torch.randn_like(x).softmax(-1)
        return prediction, target, "none" if width == 31 else "batchmean", False, 1e-10
    return x, randn(4, width)


def _reference(op, args):
    if op == "cross_entropy":
        return (torch.nn.functional.cross_entropy(args[0], args[1]),)
    if op == "fused_add_rms_norm":
        x, residual, weight, eps = args
        summed = x + residual
        return summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * weight, summed
    if op == "geglu":
        return (torch.nn.functional.gelu(args[0], approximate="tanh") * args[1],)
    if op == "kl_div":
        return (torch.nn.functional.kl_div(args[0], args[1], reduction=args[2]),)
    if op == "rope":
        return _rotate_reference(args[0], args[2], args[3]), _rotate_reference(args[1], args[2], args[3])
    return (torch.nn.functional.silu(args[0]) * args[1],)


@contextmanager
def _require_guarded_launches(monkeypatch, device):
    real_device = torch.cuda.device
    real_launch = ct.launch
    expected_stream = torch.cuda.current_stream(device)
    active = []
    launches = []

    class track_device(real_device):
        def __enter__(self):
            result = super().__enter__()
            active.append(torch.cuda.current_device())
            return result

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                active.pop()

    def check_launch(stream, *args, **kwargs):
        assert active and active[-1] == device.index, "cuTile launch missing input-device guard"
        assert torch.cuda.current_device() == device.index
        assert stream.device == device and stream.cuda_stream == expected_stream.cuda_stream
        launches.append(stream)
        return real_launch(stream, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "device", track_device)
        patch.setattr(ct, "launch", check_launch)
        yield
    assert launches, "no native cuTile launch was observed"
    assert not active


def _run_case(op, width, entrypoint, device, monkeypatch=None, phase=None):
    args = _inputs(op, width, device)
    reference_args = [
        value.detach().clone().requires_grad_(value.requires_grad) if isinstance(value, torch.Tensor) else value
        for value in args
    ]
    expected = _reference(op, reference_args)

    with _require_guarded_launches(monkeypatch, device) if phase == "forward" else nullcontext():
        if entrypoint == "native":
            module = importlib.import_module(f"liger_kernel.ops.cutile.ops.{op}")
            result = getattr(module, _FUNCTIONS[op]).apply(*args)
        else:
            result = dispatch(op, *args, impl="nvidia-cutile", mode="default")
    actual = result if isinstance(result, tuple) else (result,)
    if op == "cross_entropy":
        actual = actual[:1]
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected):
        assert output.device == device
        torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-4)

    upstream = [torch.randn_like(output) for output in actual]
    with _require_guarded_launches(monkeypatch, device) if phase == "backward" else nullcontext():
        torch.autograd.backward(actual, tuple(grad.clone() for grad in upstream))
    torch.autograd.backward(expected, tuple(grad.clone() for grad in upstream))
    for value, reference in zip(args, reference_args):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            assert value.grad is not None and reference.grad is not None
            assert value.grad.device == device
            torch.testing.assert_close(value.grad, reference.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("entrypoint", ["native", "dispatcher"])
@pytest.mark.parametrize(
    "op,width,phase",
    [
        (op, width, phase)
        for op, width in _CASES
        for phase in ("forward", "backward")
        if op != "cross_entropy" or phase == "forward"
    ],
)
def test_cutile_launches_guard_input_device_on_nondefault_stream(op, width, phase, entrypoint, monkeypatch):
    _cutile_impl(op)
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _run_case(op, width, entrypoint, device, monkeypatch, phase)
    stream.synchronize()


@pytest.mark.parametrize("entrypoint", ["native", "dispatcher"])
@pytest.mark.parametrize("op,width", _CASES)
def test_cutile_noncurrent_device_forward_backward(op, width, entrypoint):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for non-current-device execution")
    supported = [
        index for index in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(index) >= (10, 0)
    ]
    if not supported:
        pytest.skip("requires a Blackwell input device for cuTile")
    device = torch.device("cuda", supported[0])
    other = next(index for index in range(torch.cuda.device_count()) if index != device.index)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _cutile_impl(op)
        with torch.cuda.device(other):
            _run_case(op, width, entrypoint, device)
            assert torch.cuda.current_device() == other
    stream.synchronize()
