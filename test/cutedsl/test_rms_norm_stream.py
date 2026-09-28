"""CUDA Graph backward regressions for the shared CuTe RMSNorm implementation.

Keep these in the CuTe suite so the H100/B200 GPU jobs install the required DSL
and execute the regressions. Triton provides a control for the capture harness.
"""

import pytest
import torch

import liger_kernel.functional

from liger_kernel.backends.dispatch import available_backends

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph tests require CUDA")


@pytest.mark.parametrize("backend", ["nvidia-triton", "nvidia-cutedsl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("in_place", [False, True])
def test_rms_norm_backward_cuda_graph(backend, dtype, in_place):
    """Backward replay must use the new upstream gradient on the capture stream."""
    if backend not in available_backends("rms_norm"):
        pytest.skip(f"{backend} is unavailable")

    torch.manual_seed(42)
    # H=4096 is eligible for the CuTe implementation, without a Triton fallback.
    x = torch.randn(128, 4096, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    y = liger_kernel.functional.rms_norm(x, w, impl=backend, in_place=in_place)
    upstream = [torch.randn_like(y) for _ in range(2)]
    # The same backend's default-stream eager result isolates stream correctness
    # from differences in casting and reduction order between implementations.
    expected = [torch.autograd.grad(y, (x, w), dy.clone(), retain_graph=True) for dy in upstream]
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        static_x = x.detach().clone().requires_grad_(True)
        static_w = w.detach().clone().requires_grad_(True)
        static_dy = upstream[0].clone()
        static_y = liger_kernel.functional.rms_norm(static_x, static_w, impl=backend, in_place=in_place)

        def backward():
            # Clone inside the graph so in-place backward cannot overwrite the
            # upstream buffer used by later replays.
            return torch.autograd.grad(static_y, (static_x, static_w), static_dy.clone(), retain_graph=True)

        for _ in range(3):
            backward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = backward()

        for dy, reference in zip(upstream, expected):
            static_dy.copy_(dy)
            graph.replay()
            stream.synchronize()
            for actual, ref in zip(captured, reference):
                torch.testing.assert_close(actual, ref, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["nvidia-triton", "nvidia-cutedsl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual_grad", [False, True])
def test_fused_add_rms_norm_backward_cuda_graph(backend, dtype, with_residual_grad):
    """The shared RMSNorm backward must be captured, including the dS epilogue."""
    if backend not in available_backends("fused_add_rms_norm"):
        pytest.skip(f"{backend} is unavailable")

    torch.manual_seed(42)
    # This shape is supported by CuTe without falling back to Triton.
    x = torch.randn(128, 4096, device="cuda", dtype=dtype, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True)
    w = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    y, s = liger_kernel.functional.fused_add_rms_norm(x, r, w, impl=backend, in_place=False)
    outputs = (y, s) if with_residual_grad else (y,)
    upstream = [tuple(torch.randn_like(out) for out in outputs) for _ in range(2)]
    # Use default-stream eager execution of the same backend as the reference.
    expected = [torch.autograd.grad(outputs, (x, r, w), grads, retain_graph=True) for grads in upstream]
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        static_x = x.detach().clone().requires_grad_(True)
        static_r = r.detach().clone().requires_grad_(True)
        static_w = w.detach().clone().requires_grad_(True)
        static_grads = tuple(grad.clone() for grad in upstream[0])
        static_y, static_s = liger_kernel.functional.fused_add_rms_norm(
            static_x, static_r, static_w, impl=backend, in_place=False
        )
        static_outputs = (static_y, static_s) if with_residual_grad else (static_y,)

        def backward():
            return torch.autograd.grad(static_outputs, (static_x, static_r, static_w), static_grads, retain_graph=True)

        for _ in range(3):
            backward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = backward()

        for grads, reference in zip(upstream, expected):
            for static_grad, grad in zip(static_grads, grads):
                static_grad.copy_(grad)
            graph.replay()
            stream.synchronize()
            for actual, ref in zip(captured, reference):
                torch.testing.assert_close(actual, ref, atol=0, rtol=0)
