"""Correctness tests for the CuteDSL RoPE kernel.

These tests require the optional ``nvidia-cutlass-dsl`` package and an NVIDIA
GPU; they are skipped otherwise. Numerics are checked against the existing
Triton kernel (``LigerRopeFunction``) and HuggingFace's ``apply_rotary_pos_emb``
using *real* rotary embeddings (whose two head-dim halves are duplicated).
"""

import importlib.util

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest
import torch

from torch.utils.checkpoint import checkpoint
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch
from test.utils import supports_bfloat16

device = infer_device()

cutedsl_available = importlib.util.find_spec("cutlass") is not None and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not cutedsl_available,
    reason="nvidia-cutlass-dsl + CUDA GPU required for CuteDSL RoPE",
)

if cutedsl_available:
    from liger_kernel.ops.cutedsl.ops.rope import LigerRopeCuteDSLFunction


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q / k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird, non-tile-aligned shapes (exercise predicated kernel)
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
        # long sequence (the perf-relevant shape)
        (1, 8192, 32, 8, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 1e-5),
    ],
)
@pytest.mark.parametrize("expand_position_ids", [True, False])
def test_cutedsl_matches_triton_and_hf(
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol, expand_position_ids
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim)},
    )

    _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)
    _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q_hf = _q.clone().requires_grad_(True)
    k_hf = _k.clone().requires_grad_(True)
    q_tt = _q.clone().requires_grad_(True)
    k_tt = _k.clone().requires_grad_(True)
    q_cu = _q.clone().requires_grad_(True)
    k_cu = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k_hf, pos_ids)

    # forward
    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos, sin)
    cu_q, cu_k = LigerRopeCuteDSLFunction.apply(q_cu, k_cu, cos, sin)

    assert torch.allclose(cu_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(cu_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = torch.autograd.grad((cu_q, cu_k), (q_cu, k_cu), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 8, 64),
        (1, 2048, 32, 8, 128),
        # token-fallback head_dim (no TMA): fast path must still be correct
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 1e-5),
    ],
)
def test_cutedsl_backward_contiguous_grad(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    """Exercise the no-copy backward fast path.

    When the incoming gradient is already ``(bsz, n_head, seq, hd)``-contiguous
    (e.g. eager attention, or any model that calls ``.contiguous()`` after RoPE),
    ``rope_backward`` rotates it in place with the TMA kernel in ``SEQ_INNER`` mode
    instead of materialising a token-major copy. The autograd-driven test above
    only feeds transpose-view grads (the fallback path), so this covers the fast
    path directly and checks numerics against HuggingFace.
    """
    from liger_kernel.ops.cutedsl.ops.rope import rope_backward

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim)},
    )

    _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)
    _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)
    q_hf = _q.clone().requires_grad_(True)
    k_hf = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k_hf, pos_ids)

    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)

    # A genuinely (bsz, n_head, seq, hd)-CONTIGUOUS gradient triggers the fast path.
    dq = torch.randn(bsz, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    dk = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    assert dq.is_contiguous() and dk.is_contiguous()

    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = rope_backward(dq.clone(), dk.clone(), cos, sin)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim",
    [
        # TMA fast path (even head_dim)
        (128, 32, 64),
        (256, 16, 128),
        # token fallback (head_dim=92 is not TMA-vectorizable)
        (423, 73, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
        (torch.float16, 1e-2, 1e-5),
    ],
)
def test_cutedsl_vision_2d_cos_sin(seq_len, num_heads, head_dim, dtype, atol, rtol):
    """Vision RoPE feeds a 2-D ``(seq_len, head_dim)`` cos/sin table (no batch dim).

    ``liger_rotary_pos_emb_vision`` transposes q/k to 4-D but forwards cos/sin
    *unchanged* as rank-2 tensors, so the kernel must accept a rank-2 table (the
    CuteDSL ``rope_forward``/``rope_backward`` handle this via a ``cos.dim()==2``
    unsqueeze). Numerics are checked against HuggingFace (fed the equivalent 3-D
    table) and the Triton kernel (fed the same 2-D table), for both the TMA and
    the token-fallback code paths.
    """
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_heads, head_dim=head_dim)},
    )

    _q = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)
    _k = torch.randn((seq_len, num_heads, head_dim), device=device).to(dtype)

    def to_4d(x):
        return x.unsqueeze(0).transpose(1, 2).contiguous()

    q_hf = to_4d(_q).clone().requires_grad_(True)
    k_hf = to_4d(_k).clone().requires_grad_(True)
    q_tt = to_4d(_q).clone().requires_grad_(True)
    k_tt = to_4d(_k).clone().requires_grad_(True)
    q_cu = to_4d(_q).clone().requires_grad_(True)
    k_cu = to_4d(_k).clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos_3d, sin_3d = rotary_emb(k_hf, pos_ids)  # (1, seq_len, head_dim)
    cos_2d, sin_2d = cos_3d[0], sin_3d[0]  # (seq_len, head_dim) -- the vision table

    hf_q, hf_k = apply_rotary_pos_emb(q_hf, k_hf, cos_3d, sin_3d)
    tt_q, tt_k = LigerRopeFunction.apply(q_tt, k_tt, cos_2d, sin_2d)
    cu_q, cu_k = LigerRopeCuteDSLFunction.apply(q_cu, k_cu, cos_2d, sin_2d)

    assert torch.allclose(cu_q, hf_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, hf_k, atol=atol, rtol=rtol)
    assert torch.allclose(cu_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(cu_k, tt_k, atol=atol, rtol=rtol)

    dq = torch.randn_like(hf_q)
    dk = torch.randn_like(hf_k)
    q_hf_grad, k_hf_grad = torch.autograd.grad((hf_q, hf_k), (q_hf, k_hf), (dq, dk), allow_unused=True)
    q_cu_grad, k_cu_grad = torch.autograd.grad((cu_q, cu_k), (q_cu, k_cu), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q_cu_grad, q_hf_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k_cu_grad, k_hf_grad, atol=atol, rtol=rtol)


@pytest.fixture
def cached_rope():
    import liger_kernel.ops.cutedsl.ops.rope as module

    module._tma_supported_cached.cache_clear()
    yield module
    module._tma_supported_cached.cache_clear()


def test_tma_support_cache_resolves_current_device_and_preserves_shape_gates(cached_rope, monkeypatch):
    current_device = [0]
    capability = Mock(side_effect=lambda dev: (9, 0) if dev.index == 0 else (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device[0])

    for _ in range(2):
        assert cached_rope._tma_supported(torch.bfloat16, 128)
        assert cached_rope._tma_supported(torch.bfloat16, 128, torch.device("cuda"))
        assert cached_rope._tma_supported(torch.bfloat16, 128, torch.device("cuda:0"))
        assert cached_rope._tma_supported(torch.bfloat16, 128, 0)
        assert cached_rope._tma_supported(torch.bfloat16, 128, "cuda:0")
    assert capability.call_count == 1

    current_device[0] = 1
    assert not cached_rope._tma_supported(torch.bfloat16, 128)
    assert not cached_rope._tma_supported(torch.bfloat16, 128, torch.device("cuda"))
    assert cached_rope._tma_supported(torch.bfloat16, 128, torch.device("cuda:0"))
    assert capability.call_count == 2
    assert cached_rope._tma_supported(torch.float32, 8, torch.device("cuda:0"))
    assert not cached_rope._tma_supported(torch.bfloat16, 8, torch.device("cuda:0"))
    assert not cached_rope._tma_supported(torch.bfloat16, 92, torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not cached_rope._tma_supported(torch.bfloat16, 128, torch.device("cuda:0"))


@pytest.fixture(params=[False, True], ids=["pointer", "tvm-ffi"])
def tma_rope(request, monkeypatch):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires Hopper or newer")
    import liger_kernel.ops.cutedsl.ops.rope as module

    if request.param and not module._TVM_FFI_PRESENT:
        pytest.skip("apache-tvm-ffi is not installed")
    monkeypatch.setattr(module, "_TVM_FFI_PRESENT", request.param)
    return module


def _tma_tables(seq, head_dim, dtype):
    angles = torch.randn(1, seq, head_dim // 2, device=device)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _tma_reference(x, cos, sin, backward=False):
    x32 = x.float()
    first, second = x32.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    sign = -1 if backward else 1
    return x32 * cos.float()[:, None] + sign * rotated * sin.float()[:, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_tma_repeated_calls_bind_fresh_tensors_and_gradients(tma_rope, dtype, checkpointed):
    seq, head_dim = 129, 128
    cos, sin = _tma_tables(seq, head_dim, dtype)
    retained = []
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    for iteration, batch in enumerate((1, 2, 1)):
        if iteration == 2:
            cos, sin = _tma_tables(seq, head_dim, dtype)
        qb = torch.randn(batch, seq, 16, head_dim, device=device, dtype=dtype, requires_grad=True)
        kb = torch.randn(batch, seq, 8, head_dim, device=device, dtype=dtype, requires_grad=True)
        qr = qb.detach().float().requires_grad_()
        kr = kb.detach().float().requires_grad_()

        def rotate(qb, kb):
            return tma_rope.LigerRopeCuteDSLFunction.apply((qb * 1).transpose(1, 2), (kb * 1).transpose(1, 2), cos, sin)

        actual = checkpoint(rotate, qb, kb, use_reentrant=False) if checkpointed else rotate(qb, kb)
        expected = (_tma_reference(qr.transpose(1, 2), cos, sin), _tma_reference(kr.transpose(1, 2), cos, sin))
        gq, gk = torch.randn_like(qb).transpose(1, 2), torch.randn_like(kb).transpose(1, 2)
        actual_grads = torch.autograd.grad(actual, (qb, kb), (gq.clone(), gk.clone()))
        expected_grads = torch.autograd.grad(expected, (qr, kr), (gq.float(), gk.float()))
        for value, ref in zip((*actual, *actual_grads), (*expected, *expected_grads)):
            torch.testing.assert_close(value.float(), ref, atol=tolerance, rtol=tolerance)
        # Keep old allocations alive so address recycling cannot hide stale bindings.
        retained.append((qb, kb, *actual))


@pytest.mark.parametrize("backward", [False, True])
def test_tma_bindings_handle_offset_views_and_current_stream(tma_rope, backward):
    seq, heads, dim, offset = 33, 8, 128, 16
    count = seq * heads * dim
    q_storage = torch.zeros(count + 2 * offset, device=device)
    k_storage = torch.zeros_like(q_storage)
    q = q_storage[offset:-offset].view(1, seq, heads, dim).transpose(1, 2)
    k = k_storage[offset:-offset].view(1, seq, heads, dim).transpose(1, 2)
    cos, sin = torch.zeros(1, seq, dim, device=device), torch.ones(1, seq, dim, device=device)
    run = tma_rope.rope_backward if backward else tma_rope.rope_forward
    run(q, k, cos, sin)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        q.fill_(1)
        k.fill_(2)
        actual = run(q, k, cos, sin)[:2]
    stream.synchronize()

    expected = (
        _tma_reference(torch.ones_like(q), cos, sin, backward),
        _tma_reference(torch.full_like(k, 2), cos, sin, backward),
    )
    for value, ref in zip(actual, expected):
        torch.testing.assert_close(value, ref, atol=0, rtol=0)
    for storage in (q_storage, k_storage):
        assert torch.count_nonzero(storage[:offset]) == 0
        assert torch.count_nonzero(storage[-offset:]) == 0


def test_tma_bindings_are_independent_across_host_threads(tma_rope, monkeypatch):
    monkeypatch.setattr(tma_rope, "_COMPILE_CACHE", {})
    cos, sin = _tma_tables(33, 128, torch.float32)
    batches = [
        (
            torch.randn(1, 33, 8, 128, device=device).transpose(1, 2),
            torch.randn(1, 33, 8, 128, device=device).transpose(1, 2),
        )
        for _ in range(2)
    ]
    tma_rope.rope_forward(batches[0][0].clone(), batches[0][1].clone(), cos, sin)
    expected = [(_tma_reference(q, cos, sin), _tma_reference(k, cos, sin)) for q, k in batches]
    torch.cuda.synchronize()
    barrier = Barrier(2)

    def wrap(compiled):
        def launch(*args):
            barrier.wait(timeout=30)
            return compiled(*args)

        return launch

    for key, compiled in list(tma_rope._COMPILE_CACHE.items()):
        monkeypatch.setitem(tma_rope._COMPILE_CACHE, key, wrap(compiled))
    cuda_device = batches[0][0].device

    def worker(batch):
        with torch.cuda.device(cuda_device):
            stream = torch.cuda.Stream(device=cuda_device)
            with torch.cuda.stream(stream):
                result = tma_rope.rope_forward(*batch, cos, sin)[:2]
            stream.synchronize()
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(worker, batches))
    for result, refs in zip(actual, expected):
        for value, ref in zip(result, refs):
            torch.testing.assert_close(value, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backward", [False, True])
def test_tma_cuda_graph_replay_uses_updated_contents(tma_rope, backward):
    q = torch.zeros(1, 33, 8, 128, device=device).transpose(1, 2)
    k = torch.zeros_like(q)
    cos, sin = _tma_tables(33, 128, torch.float32)
    run = tma_rope.rope_backward if backward else tma_rope.rope_forward
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(q, k, cos, sin)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(q, k, cos, sin)[:2]
    for _ in range(3):
        q.normal_()
        k.normal_()
        new_cos, new_sin = _tma_tables(33, 128, torch.float32)
        cos.copy_(new_cos)
        sin.copy_(new_sin)
        expected = (_tma_reference(q, cos, sin, backward), _tma_reference(k, cos, sin, backward))
        graph.replay()
        torch.cuda.synchronize()
        for value, ref in zip(actual, expected):
            torch.testing.assert_close(value, ref, atol=1e-5, rtol=1e-5)
