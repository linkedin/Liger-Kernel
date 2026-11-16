"""Ascend ReLU² dtype, layout, boundary and autograd regression tests."""

import gc

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops.relu_squared import LigerReLUSquaredFunction
from liger_kernel.ops.relu_squared import _ascend_relu_squared_cached_op
from liger_kernel.ops.relu_squared import _ascend_relu_squared_native_supported

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available() or not _ascend_relu_squared_native_supported(),
    reason="Ascend NPU with the validated Triton-Ascend 3.2.2 / torch-npu 2.9.0 stack is required",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 7), (1,), (31,), (1023,), (4097,), (2, 3, 5, 7), (3, 257)])
def test_shapes_and_dtypes(dtype, shape):
    x = torch.randn(shape, device="npu", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    output = LigerReLUSquaredFunction.apply(x)
    expected = torch.square(torch.relu(reference))
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    tol = {torch.float16: (1e-2, 1e-2), torch.bfloat16: (1e-1, 5e-2), torch.float32: (1e-5, 1e-5)}[dtype]
    torch.testing.assert_close(output, expected, atol=tol[0], rtol=tol[1])
    torch.testing.assert_close(x.grad, reference.grad, atol=tol[0], rtol=tol[1])
    assert output.shape == x.shape and output.dtype == x.dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["transpose", "stride"])
def test_noncontiguous_inputs_and_gradients(dtype, layout):
    original = torch.randn(33, 65, device="npu", dtype=dtype)
    x = (original.T if layout == "transpose" else original[:, ::2]).detach().requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    output = LigerReLUSquaredFunction.apply(x)
    expected = torch.square(torch.relu(reference))
    grad = torch.randn(output.shape[::-1], device="npu", dtype=dtype).T
    output.backward(grad)
    expected.backward(grad)
    atol, rtol = (1e-1, 5e-2) if dtype == torch.bfloat16 else (1e-3, 1e-2)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, reference.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_special_values(dtype):
    x = torch.tensor(
        [-float("inf"), -1.0, -0.0, 0.0, 1.0, 40000.0, float("inf"), float("nan")],
        device="npu",
        dtype=dtype,
        requires_grad=True,
    )
    reference = x.detach().clone().requires_grad_(True)
    output = LigerReLUSquaredFunction.apply(x)
    expected = torch.square(torch.relu(reference))
    grad = torch.tensor(
        [float("inf"), float("inf"), float("inf"), float("inf"), 1.0, 0.001, 1.0, 1.0], device="npu", dtype=dtype
    )
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(output, expected, equal_nan=True)
    torch.testing.assert_close(x.grad, reference.grad, equal_nan=True)


def test_numerical_gradient():
    x = torch.tensor([-0.8, -0.3, 0.2, 0.7, 1.1], device="npu", dtype=torch.float32, requires_grad=True)
    assert torch.autograd.gradcheck(LigerReLUSquaredFunction.apply, (x,), eps=1e-3, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contiguous_storage_offset(dtype):
    # A contiguous view need not have a 16-byte-aligned data pointer.
    x = torch.linspace(-2, 2, 34, device="npu", dtype=dtype)[1:].requires_grad_(True)
    expected = torch.square(torch.relu(x))
    torch.testing.assert_close(LigerReLUSquaredFunction.apply(x), expected)


def test_cached_launcher_uses_new_tensors_and_current_stream():
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    results = []
    for value, stream in zip([2.0, 3.0], streams):
        with torch.npu.stream(stream):
            x = torch.full((128,), value, device="npu", requires_grad=True)
            y = LigerReLUSquaredFunction.apply(x)
            y.sum().backward()
            results.append((value, x, y))
    for stream in streams:
        stream.synchronize()
    for value, x, y in results:
        torch.testing.assert_close(y, torch.full_like(y, value**2))
        torch.testing.assert_close(x.grad, torch.full_like(x, 2 * value))


def test_graph_replay_reads_updated_input():
    x = torch.full((128,), 2.0, device="npu")
    # Compile outside capture, then verify replay uses the updated values.
    LigerReLUSquaredFunction.apply(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y = LigerReLUSquaredFunction.apply(x)
    x.fill_(3.0)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y, torch.full_like(y, 9.0))


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_disabled_autograd(context):
    x = torch.randn(31, device="npu", requires_grad=True)
    with context():
        y = LigerReLUSquaredFunction.apply(x)
        torch.testing.assert_close(y, torch.square(torch.relu(x)))
    assert not y.requires_grad and y.grad_fn is None


def test_retain_graph_uses_each_gradient():
    x = torch.randn(257, device="npu", requires_grad=True)
    y = LigerReLUSquaredFunction.apply(x)
    for scale in [1.0, 3.0]:
        x.grad = None
        y.backward(torch.full_like(y, scale), retain_graph=True)
        torch.testing.assert_close(x.grad, 2 * torch.relu(x) * scale)


def test_saved_input_version_check():
    x = torch.randn(31, device="npu", requires_grad=True)
    y = LigerReLUSquaredFunction.apply(x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


def test_saved_tensor_hooks():
    calls = []

    def pack(x):
        calls.append("pack")
        return x

    def unpack(x):
        calls.append("unpack")
        return x

    x = torch.randn(31, device="npu", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = LigerReLUSquaredFunction.apply(x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, 2 * torch.relu(x))
    assert calls == ["pack", "unpack"]


def test_parallel_threads_have_independent_gradients():
    LigerReLUSquaredFunction.apply(torch.ones(257, device="npu"))

    def run(value):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(4):
                x = torch.full((257,), value, device="npu", requires_grad=True)
                y = LigerReLUSquaredFunction.apply(x)
                y.sum().backward()
        stream.synchronize()
        torch.testing.assert_close(y, torch.full_like(y, value * value))
        torch.testing.assert_close(x.grad, torch.full_like(x, 2 * value))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, value) for value in [2.0, 3.0]]
        for future in futures:
            future.result(timeout=30)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_native_context_checks_input_metadata(mismatch):
    apply = _ascend_relu_squared_cached_op().context(torch.ones(31, device="npu")).apply
    x = torch.ones(
        32 if mismatch == "shape" else 31,
        dtype=torch.float16 if mismatch == "dtype" else torch.float32,
        device="cpu" if mismatch == "device" else "npu",
    )
    with pytest.raises(RuntimeError, match="input metadata mismatch"):
        apply(x)


def test_cached_launch_observes_dynamic_hooks():
    x = torch.ones(31, device="npu")
    LigerReLUSquaredFunction.apply(x)
    saved = (CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook)
    calls = []

    def enter(metadata):
        assert metadata is not None
        calls.append("enter")

    def exit_hook(metadata):
        assert metadata is not None
        calls.append("exit")

    try:
        CompiledKernel.launch_enter_hook = enter
        CompiledKernel.launch_exit_hook = exit_hook
        y = LigerReLUSquaredFunction.apply(x)
    finally:
        CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook = saved
    torch.testing.assert_close(y, torch.ones_like(y))
    assert calls == ["enter", "exit"]


def test_finished_graphs_release_input_storage():
    def iteration():
        x = torch.ones(257, device="npu", requires_grad=True)
        y = LigerReLUSquaredFunction.apply(x)
        y.sum().backward()

    for _ in range(3):
        iteration()
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(100):
        iteration()
    gc.collect()
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() == before
