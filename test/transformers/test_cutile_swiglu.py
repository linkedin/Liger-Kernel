import importlib.util

import pytest
import torch

from test.utils import supports_bfloat16


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.swiglu import LigerSiLUMulFunction as CuTileSiLUMulFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction as TritonSiLUMulFunction
else:
    CuTileSiLUMulFunction = None
    TritonSiLUMulFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile SwiGLU requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]


# ---------------------------------------------------------------------------
# Shared plain SiLU-Mul coverage (kept identical across triton/cutedsl/cutile test files)
# ---------------------------------------------------------------------------
_SILU_SHAPES = [
    (4, 2048),  # 2D, power-of-2 aligned
    (2, 256, 512),  # 3D, small
    (6, 42, 431),  # non-aligned / predicated path, odd width
    (1, 1, 7),  # tiny
    (3, 1023),  # odd width
    (4, 11008),  # non-power-of-2 "cliff" (Qwen2.5-7B)
    (2, 3, 13824),  # non-power-of-2 "cliff" (Qwen2.5-3B), 3D
    (2, 18944),  # non-power-of-2 "cliff" (Qwen2.5-14B)
]
_SILU_MULTIPLIERS = [(1.0, 1.0), (1.5, 0.75), (0.7, 1.3)]

# fp32, fp16, bf16 (bf16 guarded) — used by the harmonized SiLU-Mul tests below.
_SILU_DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        id="bf16",
    ),
]
# Multiplier sweep runs on the fp32 + bf16 subset (fp16 adds no path coverage here).
_SILU_MULT_DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        id="bf16",
    ),
]


def _ref_silu_mul(a, b, gate=1.0, down=1.0):
    """Ground-truth SiLU-Mul reference (silu(a * gate) * b * down) computed in fp32."""
    return torch.nn.functional.silu(a.float() * gate) * b.float() * down


def _silu_tol(dtype):
    """Harmonized (atol, rtol) for the shared plain SiLU-Mul tests."""
    if dtype == torch.float32:
        return 2e-4, 1e-4
    return 1e-2, 1e-2  # fp16 / bf16


# ---------------------------------------------------------------------------
# Plain SiLU-Mul tests (parallel across triton / cutedsl / cutile).
# ---------------------------------------------------------------------------
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("shape", _SILU_SHAPES)
@pytest.mark.parametrize("dtype", _SILU_DTYPES)
def test_cutile_correctness(shape, dtype):
    """Forward + backward vs the fp32 SiLU-Mul reference across shapes/dtypes."""
    torch.manual_seed(0)
    atol, rtol = _silu_tol(dtype)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape, device="cuda", dtype=dtype)

    ref_a = a.float().detach().requires_grad_(True)
    ref_b = b.float().detach().requires_grad_(True)
    ref_out = _ref_silu_mul(ref_a, ref_b)
    ref_out.backward(grad.float())

    test_a = a.clone().detach().requires_grad_(True)
    test_b = b.clone().detach().requires_grad_(True)
    test_out = CuTileSiLUMulFunction.apply(test_a, test_b)
    test_out.backward(grad.clone())

    torch.testing.assert_close(test_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_a.grad.float(), ref_a.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_b.grad.float(), ref_b.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("shape", [(2, 3, 13824), (2, 256, 512)])
@pytest.mark.parametrize("gate_multiplier, down_multiplier", _SILU_MULTIPLIERS)
@pytest.mark.parametrize("dtype", _SILU_MULT_DTYPES)
def test_cutile_multipliers(shape, gate_multiplier, down_multiplier, dtype):
    """Forward + backward with gate/down multipliers vs the fp32 reference."""
    torch.manual_seed(0)
    atol, rtol = _silu_tol(dtype)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape, device="cuda", dtype=dtype)

    ref_a = a.float().detach().requires_grad_(True)
    ref_b = b.float().detach().requires_grad_(True)
    ref_out = _ref_silu_mul(ref_a, ref_b, gate_multiplier, down_multiplier)
    ref_out.backward(grad.float())

    test_a = a.clone().detach().requires_grad_(True)
    test_b = b.clone().detach().requires_grad_(True)
    test_out = CuTileSiLUMulFunction.apply(test_a, test_b, gate_multiplier, down_multiplier)
    test_out.backward(grad.clone())

    torch.testing.assert_close(test_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_a.grad.float(), ref_a.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_b.grad.float(), ref_b.grad.float(), atol=atol, rtol=rtol)


def test_cutile_default_multipliers_backward_compat():
    """Calling without multipliers must equal calling with (1.0, 1.0)."""
    torch.manual_seed(0)
    a = torch.randn(4, 16, 32, device="cuda", dtype=torch.float32)
    b = torch.randn(4, 16, 32, device="cuda", dtype=torch.float32)
    grad = torch.randn(4, 16, 32, device="cuda", dtype=torch.float32)

    a1 = a.clone().detach().requires_grad_(True)
    b1 = b.clone().detach().requires_grad_(True)
    a2 = a.clone().detach().requires_grad_(True)
    b2 = b.clone().detach().requires_grad_(True)

    y_default = CuTileSiLUMulFunction.apply(a1, b1)
    y_explicit = CuTileSiLUMulFunction.apply(a2, b2, 1.0, 1.0)

    torch.testing.assert_close(y_default, y_explicit)

    y_default.backward(grad.clone())
    y_explicit.backward(grad.clone())

    torch.testing.assert_close(a1.grad, a2.grad)
    torch.testing.assert_close(b1.grad, b2.grad)


@pytest.mark.parametrize("shape", [(512,), (123,), (2, 256, 512), (3, 5, 7), (2, 4, 8, 16)])
def test_cutile_shape_flexibility(shape):
    """Arbitrary >=1-D shapes: forward + backward vs the fp32 reference; shape preserved."""
    torch.manual_seed(0)
    dtype = torch.float32
    atol, rtol = _silu_tol(dtype)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape, device="cuda", dtype=dtype)

    ref_a = a.clone().detach().requires_grad_(True)
    ref_b = b.clone().detach().requires_grad_(True)
    ref_out = _ref_silu_mul(ref_a, ref_b)
    ref_out.backward(grad)

    test_a = a.clone().detach().requires_grad_(True)
    test_b = b.clone().detach().requires_grad_(True)
    test_out = CuTileSiLUMulFunction.apply(test_a, test_b)
    assert test_out.shape == a.shape
    test_out.backward(grad.clone())

    torch.testing.assert_close(test_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_a.grad.float(), ref_a.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_b.grad.float(), ref_b.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape, dtype", [((73, 13824), torch.bfloat16), ((4, 2048), torch.float32)])
def test_cutile_cuda_graph(shape, dtype):
    """CUDA-graph capture + replay of the forward must be bit-identical to eager."""
    if dtype == torch.bfloat16 and not supports_bfloat16():
        pytest.skip("bfloat16 not supported on this GPU")
    torch.manual_seed(0)
    static_a = torch.randn(*shape, device="cuda", dtype=dtype)
    static_b = torch.randn(*shape, device="cuda", dtype=dtype)

    def run():
        with torch.no_grad():
            return CuTileSiLUMulFunction.apply(static_a, static_b)

    # Warm up (build / cache the kernel) on a side stream before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = run()

    # Replay on fresh data copied into the static input buffers.
    fresh_a = torch.randn(*shape, device="cuda", dtype=dtype)
    fresh_b = torch.randn(*shape, device="cuda", dtype=dtype)
    static_a.copy_(fresh_a)
    static_b.copy_(fresh_b)
    g.replay()
    torch.cuda.synchronize()
    graph_out = static_out.clone()

    with torch.no_grad():
        eager_out = CuTileSiLUMulFunction.apply(fresh_a, fresh_b)

    max_abs_diff = (graph_out.float() - eager_out.float()).abs().max().item()
    assert max_abs_diff == 0.0, f"graph vs eager mismatch: max_abs_diff={max_abs_diff}"


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("shape", _SILU_SHAPES)
@pytest.mark.parametrize("dtype", _SILU_DTYPES)
def test_cutile_matches_triton(shape, dtype):
    """cuTile Function must match the Triton Function (output + grads)."""
    torch.manual_seed(0)
    atol, rtol = _silu_tol(dtype)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape, device="cuda", dtype=dtype)

    tri_a = a.clone().detach().requires_grad_(True)
    tri_b = b.clone().detach().requires_grad_(True)
    tri_out = TritonSiLUMulFunction.apply(tri_a, tri_b)
    tri_out.backward(grad.clone())

    test_a = a.clone().detach().requires_grad_(True)
    test_b = b.clone().detach().requires_grad_(True)
    test_out = CuTileSiLUMulFunction.apply(test_a, test_b)
    test_out.backward(grad.clone())

    torch.testing.assert_close(test_out.float(), tri_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_a.grad.float(), tri_a.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(test_b.grad.float(), tri_b.grad.float(), atol=atol, rtol=rtol)
