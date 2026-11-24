import os

import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerAttnResFunction
from liger_kernel.transformers.functional import liger_attn_res
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def pytorch_attn_res(V, w_query, w_norm, eps=1e-6):
    """
    Reference PyTorch implementation.
    V: [N, B, T, D], w_query: [D], w_norm: [D]
    """
    V_f32 = V.float()
    rms = torch.sqrt(V_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    K = (V_f32 / rms).to(V.dtype) * w_norm

    scores = torch.einsum("d, n b t d -> n b t", w_query.float(), K.float())
    alpha = scores.softmax(dim=0)

    h = torch.einsum("n b t, n b t d -> b t d", alpha, V.float()).to(V.dtype)
    return h


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "N, B, T, D",
    [
        (4, 2, 64, 4096),
        (8, 2, 64, 4096),
        (4, 2, 64, 8192),
        # more blocks than the old MAX_BLOCKS ceiling of 32
        (40, 1, 32, 128),
        # weird shapes
        (3, 5, 37, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-5),
        (torch.float16, 1e-2, 1e-3),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(N, B, T, D, dtype, atol, rtol):
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)

    # Reference
    ref = pytorch_attn_res(V, w_query, w_norm)
    # Triton
    out = LigerAttnResFunction.apply(V, w_query, w_norm, 1e-6)

    assert_verbose_allclose(ref, out, atol=atol, rtol=rtol)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "N, B, T, D",
    [
        (4, 2, 64, 4096),
        (8, 2, 64, 4096),
        # more blocks than the old MAX_BLOCKS ceiling of 32
        (40, 1, 32, 128),
        # weird shapes
        (3, 5, 37, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-4),
        (torch.float16, 5e-2, 5e-3),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_backward(N, B, T, D, dtype, atol, rtol):
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)
    do = torch.randn(B, T, D, device=device, dtype=dtype)

    # Reference
    V_ref = V.clone().requires_grad_(True)
    wq_ref = w_query.clone().requires_grad_(True)
    wn_ref = w_norm.clone().requires_grad_(True)
    h_ref = pytorch_attn_res(V_ref, wq_ref, wn_ref)
    h_ref.backward(do, retain_graph=True)

    # Triton
    V_tri = V.clone().requires_grad_(True)
    wq_tri = w_query.clone().requires_grad_(True)
    wn_tri = w_norm.clone().requires_grad_(True)
    h_tri = LigerAttnResFunction.apply(V_tri, wq_tri, wn_tri, 1e-6)
    h_tri.backward(do, retain_graph=True)

    assert_verbose_allclose(V_ref.grad, V_tri.grad, atol=atol, rtol=rtol)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "N, B, T, D",
    [
        (4, 2, 64, 512),
        (8, 2, 32, 256),
    ],
)
def test_correctness_list_input(N, B, T, D):
    """Test that passing a list of tensors works the same as stacked tensor."""
    dtype = torch.float32
    blocks = [torch.randn(B, T, D, device=device, dtype=dtype) for _ in range(N)]
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)

    V_stacked = torch.stack(blocks)

    out_stacked = LigerAttnResFunction.apply(V_stacked, w_query, w_norm, 1e-6)
    out_list = LigerAttnResFunction.apply(torch.stack(blocks), w_query, w_norm, 1e-6)

    assert_verbose_allclose(out_stacked, out_list, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "N, B, T, D",
    [
        (4, 2, 64, 512),
        (8, 2, 32, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-5),
        (torch.float16, 1e-2, 1e-3),
    ],
)
def test_correctness_functional(N, B, T, D, dtype, atol, rtol):
    """Test that functional API matches direct function call."""
    V = torch.randn(N, B, T, D, device=device, dtype=dtype)
    w_query = torch.randn(D, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(D, device=device, dtype=dtype)

    # Direct function call
    y1 = LigerAttnResFunction.apply(V, w_query, w_norm, 1e-6)
    # Functional API
    y2 = liger_attn_res(V, w_query, w_norm, eps=1e-6)

    assert_verbose_allclose(y1, y2, atol=atol, rtol=rtol)

    # Test backward
    V1 = V.clone().requires_grad_(True)
    V2 = V.clone().requires_grad_(True)
    wq1 = w_query.clone().requires_grad_(True)
    wq2 = w_query.clone().requires_grad_(True)
    wn1 = w_norm.clone().requires_grad_(True)
    wn2 = w_norm.clone().requires_grad_(True)

    y1 = LigerAttnResFunction.apply(V1, wq1, wn1, 1e-6)
    y2 = liger_attn_res(V2, wq2, wn2, eps=1e-6)

    grad = torch.randn_like(y1)
    y1.backward(grad)
    y2.backward(grad)

    assert_verbose_allclose(V1.grad, V2.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_attn_res_int32_offset_overflow():
    # Regression: V/dV global-memory offsets must be int64. For
    # N * (B*T) * D > 2**31 the previous int32 offset arithmetic overflowed,
    # causing an illegal memory access / silently wrong output on large shapes.
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gib < 20:
        pytest.skip("needs >=20 GiB GPU to cross the 2**31 offset threshold")
    dev, dt = "cuda", torch.bfloat16
    N, B, T, D = 4, 1, 140_000, 4096
    assert N * (B * T) * D > 2**31
    set_seed(42)
    V = torch.randn(N, B, T, D, device=dev, dtype=dt, requires_grad=True)
    wq = torch.randn(D, device=dev, dtype=dt)
    wn = torch.randn(D, device=dev, dtype=dt)

    def ref_chunked(V, wq, wn, eps=1e-6, chunk=2048):
        # memory-lean, token-chunked equivalent of pytorch_attn_res
        # (each token position is independent across the T dim)
        _, _, Tt, Dd = V.shape
        wqf = wq.float()
        out = torch.empty(V.shape[1], Tt, Dd, device=V.device, dtype=V.dtype)
        with torch.no_grad():
            for s in range(0, Tt, chunk):
                e = min(s + chunk, Tt)
                Vcf = V[:, :, s:e, :].float()
                rms = torch.sqrt(Vcf.pow(2).mean(-1, keepdim=True) + eps)
                K = (Vcf / rms).to(V.dtype) * wn
                a = torch.einsum("d,nbcd->nbc", wqf, K.float()).softmax(0)
                out[:, s:e, :] = torch.einsum("nbc,nbcd->bcd", a, Vcf).to(V.dtype)
        return out

    out = LigerAttnResFunction.apply(V, wq, wn, 1e-6)
    exp = ref_chunked(V, wq, wn)
    assert_verbose_allclose(out, exp, atol=1e-1, rtol=1e-2)
    out.sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(V.grad).all()
