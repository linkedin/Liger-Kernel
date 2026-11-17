from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops import LigerSoftmaxFunction
from liger_kernel.transformers.functional import liger_softmax
from liger_kernel.transformers.softmax import LigerSoftmax

NPU_FORMAT_ND = 2


def _validated_stack_available():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    from liger_kernel.ops.backends._ascend.ops.softmax import _native_softmax_supported

    return _native_softmax_supported()


pytestmark = pytest.mark.skipif(
    not _validated_stack_available(), reason="Validated Ascend 3.2.2 / torch-npu 2.9.0 stack is required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((), 0),
        ((0, 3), -1),
        ((3, 0), 0),
        ((2, 1), -1),
        ((97, 2048), -1),
        ((49, 4096), -1),
        ((2048, 4096), -1),
        ((5, 8192), -1),
        ((31,), -1),
        ((3, 5, 7), 0),
        ((3, 5, 7), 1),
        ((3, 5, 7), -1),
        ((3, 3584), -1),
        ((31, 3584), -1),
        ((3, 5120), -1),
        ((3, 7168), -1),
        ((31, 7168), -1),
        ((2, 4097), -1),
        ((1, 65537), -1),
    ],
)
def test_shapes_dims_gradients(dtype, shape, dim):
    x = torch.randn(shape, dtype=dtype, device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    actual = LigerSoftmax(dim)(x)
    expected = torch.softmax(ref, dim)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
def test_noncontiguous(dim):
    base = torch.randn(4, 5, 7, device="npu", requires_grad=True)
    x = base.transpose(0, 1)
    ref = x.detach().clone().requires_grad_(True)
    y = liger_softmax(x, dim)
    expected = torch.softmax(ref, dim)
    assert y.stride() == expected.stride()
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn(4, 5, 7, device="npu").transpose(0, 1)
    y.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(base.grad.transpose(0, 1), ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dim", [-4, 3])
def test_invalid_dim(dim):
    with pytest.raises(IndexError, match="Dimension out of range"):
        liger_softmax(torch.randn(2, 3, 4, device="npu"), dim)


def test_original_function_signature():
    x = torch.randn(3, 7, device="npu", requires_grad=True)
    y = LigerSoftmaxFunction.apply(x)
    y.backward(torch.randn_like(y))
    assert x.grad is not None


def test_numerical_gradient():
    x = torch.randn(2, 3, device="npu", requires_grad=True)
    assert torch.autograd.gradcheck(liger_softmax, (x,), eps=1e-3, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize(
    "values",
    [
        [1e4, 1e4 - 1, -1e4],
        [float("nan"), 0, 1],
        [float("inf"), 0, 1],
        [-float("inf"), -float("inf"), -float("inf")],
        [-float("inf"), 1, 2],
    ],
)
def test_special_values(values):
    x = torch.tensor([values], device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    y, expected = liger_softmax(x), torch.softmax(ref, -1)
    torch.testing.assert_close(y, expected, equal_nan=True)
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, ref.grad, equal_nan=True)


def test_same_elements_different_columns():
    for shape in [(8, 16), (4, 32), (2, 64), (8, 16)]:
        x = torch.randn(shape, device="npu", requires_grad=True)
        ref = x.detach().clone().requires_grad_(True)
        y, expected = liger_softmax(x), torch.softmax(ref, -1)
        torch.testing.assert_close(y, expected)
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(x.grad, ref.grad)


def test_saved_output_version_check():
    x = torch.randn(2, 7, device="npu", requires_grad=True)
    y = liger_softmax(x)
    with torch.no_grad():
        y.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward(torch.ones_like(y))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nd_output_composes_with_autograd(dtype):
    from torch_npu import get_npu_format

    x = torch.randn(3, 2, 17, device="npu", dtype=dtype).transpose(0, 1).detach().requires_grad_(True)
    ref = x.detach().clone().requires_grad_(True)
    weights = torch.randn(x.shape, device="npu", dtype=torch.float32)
    y = liger_softmax(x)
    expected = torch.softmax(ref, -1)
    assert y.is_contiguous()
    assert get_npu_format(y) == NPU_FORMAT_ND
    (y.float() * weights).square().sum().backward()
    (expected.float() * weights).square().sum().backward()
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)


def test_concurrent_backward_streams():
    warmup = torch.randn(16, 64, device="npu", requires_grad=True)
    liger_softmax(warmup).sum().backward()
    torch.npu.synchronize()

    def run():
        torch.npu.set_device(0)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            x = torch.randn(16, 64, device="npu", requires_grad=True)
            ref = x.detach().clone().requires_grad_(True)
            grad = torch.randn_like(x)
            liger_softmax(x).backward(grad)
            torch.softmax(ref, -1).backward(grad)
        stream.synchronize()
        return x.grad, ref.grad

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        for future in futures:
            actual, expected = future.result()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_saved_tensor_hooks_and_retained_graph():
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor.detach().clone()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        x = torch.randn(2, 7, device="npu", requires_grad=True)
        y = liger_softmax(x)
    grad = torch.randn_like(y)
    first = torch.autograd.grad(y, x, grad, retain_graph=True)[0]
    second = torch.autograd.grad(y, x, grad)[0]
    torch.testing.assert_close(first, second)
    assert saved


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_no_grad(mode):
    with mode():
        x = torch.randn(2, 7, device="npu")
        y = liger_softmax(x)
        torch.testing.assert_close(y, torch.softmax(x, -1))
        assert y.grad_fn is None


def test_graph_replay():
    x = torch.randn(4, 32, device="npu")
    for _ in range(3):
        liger_softmax(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y = liger_softmax(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y, torch.softmax(x, -1))


def test_unaligned_view_and_current_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.randn(130, device="npu")[1:129].reshape(4, 32).requires_grad_(True)
        ref = x.detach().clone().requires_grad_(True)
        y, expected = liger_softmax(x), torch.softmax(ref, -1)
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
    stream.synchronize()
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(x.grad, ref.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("columns", [33, 2049, 3585, 5121, 7169, 8193, 16385])
def test_irregular_widths(dtype, columns):
    # Cover masked tails in every fused/split/long dispatch, including odd rows.
    x = torch.randn(5, columns, dtype=dtype, device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    y = liger_softmax(x)
    expected = torch.softmax(ref, -1)
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(y, expected, atol=tolerance, rtol=tolerance)
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("value", [-float("inf"), float("inf"), float("nan")])
def test_long_row_special_values(value):
    x = torch.zeros(2, 16385, device="npu")
    x[:, :8192] = -float("inf")
    x[1, -1] = value
    x.requires_grad_(True)
    ref = x.detach().clone().requires_grad_(True)
    y, expected = liger_softmax(x), torch.softmax(ref, -1)
    torch.testing.assert_close(y, expected, equal_nan=True)
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, ref.grad, equal_nan=True)


def test_backward_releases_saved_output():
    x = torch.randn(2, 7, device="npu", requires_grad=True)
    y = liger_softmax(x)
    y.sum().backward()
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        y.sum().backward()


def test_launch_hooks_enabled_after_forward():
    x = torch.randn(2, 7, device="npu", requires_grad=True)
    y = liger_softmax(x)
    events = []
    previous_enter = CompiledKernel.launch_enter_hook
    previous_exit = CompiledKernel.launch_exit_hook
    try:
        CompiledKernel.launch_enter_hook = lambda metadata: events.append("enter")
        CompiledKernel.launch_exit_hook = lambda metadata: events.append("exit")
        y.backward(torch.ones_like(y))
    finally:
        CompiledKernel.launch_enter_hook = previous_enter
        CompiledKernel.launch_exit_hook = previous_exit
    assert events == ["enter", "exit"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_initialization_preserves_input_rng_and_gradients(dtype):
    x = torch.randn(257, 4096, dtype=dtype, device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    rng = torch.npu.get_rng_state().clone()
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}[dtype]
    for _ in range(2):
        y = liger_softmax(x)
        expected = torch.softmax(ref, -1)
        torch.testing.assert_close(x, ref, atol=0, rtol=0)
        torch.testing.assert_close(torch.npu.get_rng_state(), rng, atol=0, rtol=0)
        torch.testing.assert_close(y, expected, atol=tolerance, rtol=tolerance)
        dy = torch.linspace(-1, 1, x.numel(), device="npu", dtype=dtype).reshape_as(x)
        y.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)
        x.grad = None
        ref.grad = None


@pytest.mark.parametrize(
    "triton_version,torch_npu_version,expected",
    [
        ("3.2.2", "2.9.0", True),
        ("3.2.2", "2.9.0+local", True),
        ("3.2.2", "2.9.0.post1", False),
        ("3.6.0", "2.10.0.post7", False),
        ("3.2.1", "2.9.0", False),
        (None, "2.9.0", False),
    ],
)
def test_native_stack_dispatch(monkeypatch, triton_version, torch_npu_version, expected):
    from importlib.metadata import PackageNotFoundError

    from liger_kernel.ops.backends._ascend.ops import softmax

    def package_version(name):
        value = {"triton-ascend": triton_version, "torch-npu": torch_npu_version}[name]
        if value is None:
            raise PackageNotFoundError(name)
        return value

    supported = softmax._native_softmax_supported
    supported.cache_clear()
    try:
        monkeypatch.setattr(softmax, "version", package_version)
        assert supported() is expected
    finally:
        supported.cache_clear()


def test_existing_backend_fallback():
    from liger_kernel.ops.backends._ascend.ops import softmax

    x = torch.randn(2, 3, 7, device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    y = softmax._LegacySoftmaxFunction.apply(x, 1)
    expected = torch.softmax(ref, dim=1)
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(x.grad, ref.grad)
