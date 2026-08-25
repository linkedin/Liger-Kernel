import importlib.util

import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import supports_bfloat16


def _has_cuda_tile():
    try:
        return importlib.util.find_spec("cuda.tile") is not None
    except ModuleNotFoundError:
        return False


_CUDA_TILE_AVAILABLE = _has_cuda_tile()
if _CUDA_TILE_AVAILABLE:
    from liger_kernel.ops.cutile.ops.swiglu import LigerSiLUMulFunction as CuTileSiLUMulFunction
else:
    CuTileSiLUMulFunction = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile SwiGLU requires CUDA"),
    pytest.mark.skipif(not _CUDA_TILE_AVAILABLE, reason="cuda-tile is not installed"),
]


def _reference_swiglu(a, b, gate_multiplier, down_multiplier):
    # Compute in fp32 then cast back, matching the kernel's internal fp32 math.
    return (F.silu(a.float() * gate_multiplier) * b.float() * down_multiplier).to(a.dtype)


# n_cols chosen to cover: power-of-2 (aligned) sizes, the non-power-of-2
# intermediate sizes that hit the old bounds-checking cliff (Qwen2.5 11008 /
# 13824 / 18944, Gemma-ish 4608), and small / odd widths that exercise the full
# power-of-2 remainder ladder (down to width 1).
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "n_cols",
    [
        512,  # < base: pure remainder ladder
        2048,  # exactly one base tile
        4096,  # aligned, multiple base tiles
        8192,  # aligned
        4608,  # Gemma-ish (non-power-of-2)
        11008,  # Qwen2.5-7B
        13824,  # Qwen2.5-3B
        18944,  # Qwen2.5-14B
        1023,  # odd: 512+256+...+1 ladder
        431,  # odd remainder
        7,  # tiny (widths 4+2+1)
        1,  # single column
    ],
)
@pytest.mark.parametrize("n_rows", [4, 73])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 2e-4, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_cutile_swiglu_correctness(n_cols, n_rows, dtype, atol, rtol):
    torch.manual_seed(0)
    a = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    b = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    grad_output = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)

    ref_a = a.clone().requires_grad_(True)
    ref_b = b.clone().requires_grad_(True)
    ref_out = _reference_swiglu(ref_a, ref_b, 1.0, 1.0)
    ref_out.backward(grad_output)

    cutile_a = a.clone().requires_grad_(True)
    cutile_b = b.clone().requires_grad_(True)
    cutile_out = CuTileSiLUMulFunction.apply(cutile_a, cutile_b)
    cutile_out.backward(grad_output.clone())

    assert_verbose_allclose(cutile_out, ref_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(cutile_a.grad, ref_a.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(cutile_b.grad, ref_b.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("n_cols", [4096, 13824, 18944])
@pytest.mark.parametrize(
    "gate_multiplier, down_multiplier",
    [
        (1.0, 1.0),
        (0.5, 2.0),
        (2.0, 0.5),
    ],
)
def test_cutile_swiglu_multipliers(n_cols, gate_multiplier, down_multiplier):
    dtype, atol, rtol = torch.float32, 2e-4, 1e-5
    torch.manual_seed(0)
    a = torch.randn(16, n_cols, device="cuda", dtype=dtype)
    b = torch.randn(16, n_cols, device="cuda", dtype=dtype)
    grad_output = torch.randn(16, n_cols, device="cuda", dtype=dtype)

    ref_a = a.clone().requires_grad_(True)
    ref_b = b.clone().requires_grad_(True)
    ref_out = _reference_swiglu(ref_a, ref_b, gate_multiplier, down_multiplier)
    ref_out.backward(grad_output)

    cutile_a = a.clone().requires_grad_(True)
    cutile_b = b.clone().requires_grad_(True)
    cutile_out = CuTileSiLUMulFunction.apply(cutile_a, cutile_b, gate_multiplier, down_multiplier)
    cutile_out.backward(grad_output.clone())

    assert_verbose_allclose(cutile_out, ref_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(cutile_a.grad, ref_a.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(cutile_b.grad, ref_b.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("n_cols", [13824, 18944])
def test_cutile_swiglu_3d_input(n_cols):
    dtype, atol, rtol = torch.float32, 2e-4, 1e-5
    torch.manual_seed(0)
    shape = (2, 3, n_cols)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    grad_output = torch.randn(*shape, device="cuda", dtype=dtype)

    ref_a = a.clone().requires_grad_(True)
    ref_b = b.clone().requires_grad_(True)
    ref_out = _reference_swiglu(ref_a, ref_b, 1.0, 1.0)
    ref_out.backward(grad_output)

    cutile_a = a.clone().requires_grad_(True)
    cutile_b = b.clone().requires_grad_(True)
    cutile_out = CuTileSiLUMulFunction.apply(cutile_a, cutile_b)
    cutile_out.backward(grad_output.clone())

    assert cutile_out.shape == shape
    assert_verbose_allclose(cutile_out, ref_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(cutile_a.grad, ref_a.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(cutile_b.grad, ref_b.grad, atol=atol, rtol=rtol, max_print=20)
