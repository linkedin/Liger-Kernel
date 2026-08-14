"""Tests for the frontend ``LigerExpertParallelFusedMoe`` ``nn.Module``.

Two layers of coverage, mirroring ``test_cute_moe_autograd.py``:

  * ``test_frontend_import_guard`` — pure-Python, always runs. The frontend
    imports the fused-MoE backend op eagerly, so importing
    ``liger_kernel.transformers.moe`` must succeed iff the lck wheel is present
    and raise a clear ``ImportError`` otherwise. When it does import, the module
    is a weightless ``nn.Module`` op-wrapper. Needs neither a GPU nor the wheel.

  * ``test_frontend_module_forward`` — functional, **skipped unless the lck wheel
    is installed and >= 2 CUDA devices are present**. Drives the module end to
    end across real PEs and checks the behaviour the wrapper adds on top of the
    raw ``moe_fused`` op: the ``(B, S, D)`` reshape round-trip, the int32 cast of
    the routing indices, and the expert-parallel ``process_group`` threading —
    against a plain-torch reference, including a backward pass.

Self-contained on purpose: like the sibling autograd test it spawns one real
process per GPU (NCCL + NVSHMEM via ``nvshmem.init_from_pg``) and gathers the
per-rank experts for the reference. Shapes (T=512, D=4096, I=2048, E=16, K=2)
match a row the tuned dispatch table covers so the auto path resolves a config.
"""

from __future__ import annotations

import importlib
import os
import tempfile

from datetime import timedelta

import pytest

# ── shared setup / availability probing ──────────────────────────────────────

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.functional as Fnn

    from liger_kernel.ops import cute as _cute

    _LCK_AVAILABLE = _cute.is_available()
except ImportError:  # torch missing, or liger_kernel import failure
    torch = None
    dist = None
    mp = None
    nn = None
    Fnn = None
    _cute = None
    _LCK_AVAILABLE = False

_NDEV = torch.cuda.device_count() if (torch is not None and torch.cuda.is_available()) else 0

# MoE shape — matches a row the tuned table covers, E divisible by every world
# size we run. T is chosen so it factors cleanly into a (B, S) grid below.
_T, _D, _I, _E, _K = 512, 4096, 2048, 16, 2
_B, _S = 4, _T // 4  # (batch, seq) view that flattens back to T tokens


# ── always-on import-guard check ─────────────────────────────────────────────


def test_frontend_import_guard():
    """Importing the frontend tracks lck availability; the module is weightless.

    The op is imported eagerly at module import, so the import itself is the
    backend-availability gate: it succeeds with the wheel and raises a clear
    ``ImportError`` without it. When present, ``LigerExpertParallelFusedMoe`` is
    an ``nn.Module`` op-wrapper that owns no parameters of its own.
    """
    if _LCK_AVAILABLE:
        mod = importlib.import_module("liger_kernel.transformers.moe")
        layer_cls = mod.LigerExpertParallelFusedMoe
        assert issubclass(layer_cls, nn.Module)

        layer = layer_cls()
        assert layer.process_group is None
        # Pure op-wrapper: holds no weights / buffers of its own.
        assert list(layer.parameters()) == []
        assert list(layer.buffers()) == []

        sentinel = object()
        assert layer_cls(process_group=sentinel).process_group is sentinel
    else:
        with pytest.raises(ImportError):
            importlib.import_module("liger_kernel.transformers.moe")


# ── functional test (needs the lck wheel + GPUs) ──────────────────────────────


def _world_size() -> int:
    """Largest power-of-two divisor of E that fits the available GPUs (>= 2)."""
    want = int(os.environ.get("LIGER_TEST_NPROC", "8"))
    for w in (8, 4, 2):
        if w <= min(want, _NDEV) and _E % w == 0:
            return w
    return 2


def _route(X, gate_W, top_k):
    """Standard top-K gate in plain torch (stands in for the native router)."""
    logits = X.float() @ gate_W.float().t()  # [T, E]
    weights, indices = torch.topk(logits, top_k, dim=1)
    weights = torch.softmax(weights, dim=1)
    return indices.to(torch.int32).contiguous(), weights.to(torch.bfloat16).contiguous()


def _torch_reference_moe(X, expert_indices, expert_weights, all_B_global, all_C_global, all_A_global, top_k):
    T, D = X.shape
    E = all_B_global.size(0)
    device = X.device

    X_f32 = X.to(torch.float32)
    B_f32 = all_B_global.to(torch.float32)
    C_f32 = all_C_global.to(torch.float32)
    A_f32 = all_A_global.to(torch.float32)
    w_f32 = expert_weights.to(torch.float32)

    Y_slots = torch.zeros(T * top_k, D, dtype=torch.float32, device=device)
    idx_cpu = expert_indices.cpu()

    for e in range(E):
        mask = idx_cpu == e
        if not mask.any():
            continue
        rows, slots_k = mask.nonzero(as_tuple=True)
        rows = rows.to(device)
        slots = rows * top_k + slots_k.to(device)
        X_e = X_f32.index_select(0, rows)
        BX = X_e @ B_f32[e].t()
        CX = X_e @ C_f32[e].t()
        Z = Fnn.silu(BX) * CX
        Y_e = Z @ A_f32[e].t()
        Y_slots.index_copy_(0, slots, Y_e)

    weighted = Y_slots * w_f32.reshape(-1, 1)
    return weighted.reshape(T, top_k, D).sum(1).to(X.dtype)


def _gather_experts(local, group):
    world = dist.get_world_size(group)
    chunks = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(chunks, local.contiguous(), group=group)
    return torch.cat(chunks, dim=0).contiguous()


def _check_close(ours, ref, *, mean_rel_tol=0.15, atol_for_rel=1e-3):
    ours_f32 = ours.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    assert ours_f32.shape == ref_f32.shape, (ours_f32.shape, ref_f32.shape)
    assert torch.isfinite(ours_f32).all(), "ours has non-finite values"
    assert torch.isfinite(ref_f32).all(), "ref has non-finite values"
    diff = (ours_f32 - ref_f32).abs()
    rel = diff / ref_f32.abs().clamp_min(atol_for_rel)
    mean_rel = rel.mean().item()
    assert mean_rel < mean_rel_tol, (
        f"reference mismatch: mean_rel={mean_rel:.4f} (tol {mean_rel_tol}), max_abs={diff.max().item():.4f}"
    )


def _make_inputs(rank, world_size):
    experts_per_pe = _E // world_size
    torch.manual_seed(0)
    X = torch.randn(_T, _D, dtype=torch.bfloat16, device="cuda")
    gate_W = torch.randn(_E, _D, dtype=torch.bfloat16, device="cuda")
    torch.manual_seed(100 + rank)
    all_B = torch.randn(experts_per_pe, _I, _D, dtype=torch.bfloat16, device="cuda")
    all_C = torch.randn(experts_per_pe, _I, _D, dtype=torch.bfloat16, device="cuda")
    all_A = torch.randn(experts_per_pe, _D, _I, dtype=torch.bfloat16, device="cuda")
    expert_indices, expert_weights = _route(X, gate_W, _K)
    return X, all_B, all_C, all_A, expert_indices, expert_weights


def _module_worker(rank, world_size, init_file):
    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi

    from liger_kernel.transformers.moe import LigerExpertParallelFusedMoe

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    nvshmem.init_from_pg()
    try:
        tvm_ffi.moe_configure_symmetric(
            max_tokens=_T,
            hidden_dim=_D,
            max_num_experts=_E,
            max_top_k=_K,
            num_pes=nvshmem.n_pes(),
            num_hosts=1,
            gpus_per_host=world_size,
        )
        X_data, all_B_data, all_C_data, all_A_data, ei, ew = _make_inputs(rank, world_size)

        # Reference uses the globally-gathered experts and flat [T, D] tensors.
        all_B_g = _gather_experts(all_B_data, dist.group.WORLD)
        all_C_g = _gather_experts(all_C_data, dist.group.WORLD)
        all_A_g = _gather_experts(all_A_data, dist.group.WORLD)
        Y_ref = _torch_reference_moe(X_data, ei, ew, all_B_g, all_C_g, all_A_g, _K)

        layer = LigerExpertParallelFusedMoe(process_group=dist.group.WORLD)

        # Feed (B, S, D) / (B, S, K) so the module's flatten→op→restore round-trip
        # is exercised — the output must come back with the same leading dims.
        hidden = X_data.view(_B, _S, _D)
        ei_3d = ei.view(_B, _S, _K)
        ew_3d = ew.view(_B, _S, _K)

        # ── No-grad path: shape round-trip + numerical match. ──
        with torch.no_grad():
            Y_ng = layer(hidden, ei_3d, ew_3d, all_B_data, all_C_data, all_A_data, _E)
        assert Y_ng.shape == (_B, _S, _D), Y_ng.shape
        assert not Y_ng.requires_grad
        _check_close(Y_ng.reshape(_T, _D), Y_ref)

        # The module casts non-int32 routing indices itself (int64 is the common
        # output of torch.topk) — same result.
        with torch.no_grad():
            Y_i64 = layer(hidden, ei_3d.to(torch.int64), ew_3d, all_B_data, all_C_data, all_A_data, _E)
        _check_close(Y_i64.reshape(_T, _D), Y_ref)

        # ── Grad path: weights supplied at call time receive finite grads. ──
        X = X_data.clone().detach().view(_B, _S, _D).requires_grad_(True)
        all_B = all_B_data.clone().detach().requires_grad_(True)
        all_C = all_C_data.clone().detach().requires_grad_(True)
        all_A = all_A_data.clone().detach().requires_grad_(True)

        Y = layer(X, ei_3d, ew_3d, all_B, all_C, all_A, _E)
        assert Y.shape == (_B, _S, _D), Y.shape
        assert Y.requires_grad, "grad path must yield a tensor that requires grad"
        _check_close(Y.reshape(_T, _D), Y_ref)

        Y.float().sum().backward()
        for name, t in [("X", X), ("all_B", all_B), ("all_C", all_C), ("all_A", all_A)]:
            assert t.grad is not None, f"{name}.grad is None after backward"
            assert t.grad.shape == t.shape, f"{name}.grad shape {tuple(t.grad.shape)} != {tuple(t.shape)}"
            assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise


@pytest.mark.skipif(not _LCK_AVAILABLE, reason="liger_cute_kernels (lck wheel) not installed")
@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_frontend_module_forward():
    """``LigerExpertParallelFusedMoe`` runs end to end and matches the reference.

    Exercises the value the frontend adds over the raw op — the (B, S, D)
    reshape round-trip, the int32 cast of routing indices, and process-group
    threading — across real PEs, in both the no-grad and grad paths.
    """
    import shutil

    world_size = _world_size()
    rdzv = tempfile.mkdtemp(prefix="lck_frontend_")
    init_file = os.path.join(rdzv, "store")
    try:
        mp.spawn(_module_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        shutil.rmtree(rdzv, ignore_errors=True)
