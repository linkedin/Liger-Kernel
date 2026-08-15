"""CUDA-graph capture/replay tests for the fused MoE fwd/bwd kernels.

Ported from LigerCommKernels' ``tests/python/test_autograd.py`` (the
``test_moe_fused_cuda_graph`` / fwd+bwd graph tests). Those upstream tests drove
the kernels through the ``lck.moe_fused`` autograd wrapper and ``moe_router_fwd``;
neither is in scope for this package yet, so here we drive the **raw TVM FFI
bindings** directly — which is exactly what the graph tests are about: proving the
*kernel itself* is CUDA-graph-safe (no host syncs, no non-capturable NVSHMEM calls,
stable symmetric-stack offsets across replays). Routing is computed in plain torch
(a standard top-K gate), and the same indices/weights feed both the kernel and the
torch reference, so the comparison is consistent.

The whole kernel design (device-resident iter counter, ``nvshmemx_barrier_on_stream``
instead of host-side ``nvshmem_team_sync``) exists to be graph-capturable, so this
is the highest-signal regression test for the port.

No mocks. Each test spawns one real process per GPU with ``torch.multiprocessing``,
initialises a real NCCL group + NVSHMEM runtime via ``nvshmem.init_from_pg``, and
captures the kernel under ``torch.cuda.graph``. Requires >= 2 CUDA devices; skipped
otherwise. Shapes (T=512, D=4096, I=2048, E=16, K=2) match an entry the tuned
dispatch table covers so the auto path resolves a config.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from datetime import timedelta

import pytest

try:
    import liger_cute_kernels.tvm_ffi as tvm_ffi
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as Fnn

    from liger_cute_kernels import nvshmem
except ImportError:
    torch = None
    dist = None
    mp = None
    tvm_ffi = None
    nvshmem = None

pytestmark = pytest.mark.skipif(
    tvm_ffi is None or not tvm_ffi.is_available(),
    reason="liger_cute_kernels not built/installed; build it to run these.",
)

_NDEV = (
    torch.cuda.device_count()
    if (tvm_ffi is not None and tvm_ffi.is_available() and torch is not None and torch.cuda.is_available())
    else 0
)

# MoE shape — matches a row the tuned table covers (so the auto path resolves a
# config) and keeps E divisible by every world size we run.
_T, _D, _I, _E, _K = 512, 4096, 2048, 16, 2


def _world_size() -> int:
    """Largest power-of-two divisor of E that fits the available GPUs (>= 2)."""
    want = int(os.environ.get("LIGER_TEST_NPROC", "8"))
    for w in (8, 4, 2):
        if w <= min(want, _NDEV) and _E % w == 0:
            return w
    return 2


# ── torch reference + comparison (ported verbatim from test_kernels.py) ───────


def _route(X: "torch.Tensor", gate_W: "torch.Tensor", top_k: int):
    """Standard top-K gate, in plain torch (stands in for moe_router_fwd_bf16).

    Returns (expert_indices [T,K] int32, expert_weights [T,K] bf16). The same
    tensors feed the kernel and the reference, so absolute routing values are
    irrelevant — only consistency matters.
    """
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


# ── distributed harness (file-based rendezvous, like test_nvshmem.py) ─────────


def _init(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    nvshmem.init_from_pg()


def _make_inputs(rank: int, world_size: int, tokens: int = _T):
    experts_per_pe = _E // world_size
    torch.manual_seed(0)
    X = torch.randn(tokens, _D, dtype=torch.bfloat16, device="cuda")
    gate_W = torch.randn(_E, _D, dtype=torch.bfloat16, device="cuda")
    torch.manual_seed(100 + rank)
    all_B = torch.randn(experts_per_pe, _I, _D, dtype=torch.bfloat16, device="cuda")
    all_C = torch.randn(experts_per_pe, _I, _D, dtype=torch.bfloat16, device="cuda")
    all_A = torch.randn(experts_per_pe, _D, _I, dtype=torch.bfloat16, device="cuda")
    expert_indices, expert_weights = _route(X, gate_W, _K)
    return X, gate_W, all_B, all_C, all_A, expert_indices, expert_weights


def _configure(world_size: int, max_tokens: int = _T):
    tvm_ffi.moe_configure_symmetric(
        max_tokens=max_tokens,
        hidden_dim=_D,
        max_num_experts=_E,
        max_top_k=_K,
        num_pes=nvshmem.n_pes(),
        num_hosts=1,
        gpus_per_host=world_size,
    )


def _run(world_size: int, worker, *worker_args):
    rdzv = tempfile.mkdtemp(prefix="lck_graph_")
    init_file = os.path.join(rdzv, "store")
    try:
        mp.spawn(
            worker,
            args=(world_size, init_file, *worker_args),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(rdzv, ignore_errors=True)


# ── forward: capture + replay ─────────────────────────────────────────────────


def _fwd_graph_worker(rank: int, world_size: int, init_file: str):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    try:
        _configure(world_size)
        X, gate_W, all_B, all_C, all_A, ei, ew = _make_inputs(rank, world_size)

        def fwd():
            return tvm_ffi.moe_fused_fwd_bf16(
                X, ei, ew, all_B, all_C, all_A, num_experts=_E, top_k=_K, team_handle=team
            )

        # Reference (global experts gathered across PEs).
        all_B_g = _gather_experts(all_B, dist.group.WORLD)
        all_C_g = _gather_experts(all_C, dist.group.WORLD)
        all_A_g = _gather_experts(all_A, dist.group.WORLD)
        Y_ref = _torch_reference_moe(X, ei, ew, all_B_g, all_C_g, all_A_g, _K).cpu()

        # Warmup: triggers all lazy/non-capturable work (team cache, symm sizing,
        # dispatch-table walk). Release the symm buffers immediately (no-grad path).
        Y_w = fwd()[0]
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()

        # Eager output, snapshotted to CPU before the capture mempool is opened.
        Y_eager = fwd()[0]
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()
        Y_eager_cpu = Y_eager.detach().cpu().clone()
        del Y_eager, Y_w

        # Capture. Explicit pool + thread_local error mode mirror Megatron's
        # CudaGraphManager so input/symm addresses stay stable across replays.
        mempool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=mempool, capture_error_mode="thread_local"):
            Y_captured = fwd()[0]
            tvm_ffi.moe_pop_fwd()  # host-side stack bookkeeping; nothing recorded

        results = []
        for _ in range(2):  # second replay must hit the same symm offsets
            g.replay()
            torch.cuda.synchronize()
            results.append(Y_captured.detach().cpu().clone())

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for i, Y in enumerate(results):
        assert torch.isfinite(Y).all(), f"replay {i} produced non-finite output"
        _check_close(Y, Y_eager_cpu)
        _check_close(Y, Y_ref)


# ── forward: packed w13 strided views ─────────────────────────────────────────


def _strided_fwd_worker(rank: int, world_size: int, init_file: str):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    try:
        _configure(world_size)
        X, gate_W, all_B, all_C, all_A, ei, ew = _make_inputs(rank, world_size)
        experts_per_pe = all_B.size(0)

        packed_w13 = torch.stack((all_B, all_C), dim=1).reshape(experts_per_pe, 2 * _I, _D)
        strided_B = packed_w13[:, :_I, :]
        strided_C = packed_w13[:, _I:, :]
        assert not strided_B.is_contiguous()
        assert not strided_C.is_contiguous()
        assert strided_B.stride() == (2 * _I * _D, _D, 1)
        assert strided_C.stride() == strided_B.stride()

        all_B_g = _gather_experts(all_B, dist.group.WORLD)
        all_C_g = _gather_experts(all_C, dist.group.WORLD)
        all_A_g = _gather_experts(all_A, dist.group.WORLD)
        Y_ref = _torch_reference_moe(X, ei, ew, all_B_g, all_C_g, all_A_g, _K).cpu()

        Y_contiguous = tvm_ffi.moe_fused_fwd_bf16(
            X, ei, ew, all_B, all_C, all_A, num_experts=_E, top_k=_K, team_handle=team
        )[0]
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()
        Y_contiguous = Y_contiguous.cpu()

        Y_strided = tvm_ffi.moe_fused_fwd_bf16(
            X,
            ei,
            ew,
            strided_B,
            strided_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
        )[0]
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()
        Y_strided = Y_strided.cpu()

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    _check_close(Y_strided, Y_contiguous, mean_rel_tol=0.01)
    _check_close(Y_strided, Y_ref)


# ── forward + backward: two graphs ────────────────────────────────────────────


def _fwd_bwd_graph_worker(rank: int, world_size: int, init_file: str):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    try:
        _configure(world_size)
        X, gate_W, all_B, all_C, all_A, ei, ew = _make_inputs(rank, world_size)

        def fwd():
            return tvm_ffi.moe_fused_fwd_bf16(
                X, ei, ew, all_B, all_C, all_A, num_experts=_E, top_k=_K, team_handle=team
            )

        def bwd(dY, fwd_out):
            (Y, x_sorted, y_buf, all_off, tok_slots, tile_ids, tile_m) = fwd_out
            return tvm_ffi.moe_fused_bwd_bf16(
                dY,
                y_buf,
                x_sorted,
                tok_slots,
                tile_ids,
                all_off,
                ei,
                ew,
                all_B,
                all_C,
                all_A,
                num_experts=_E,
                top_k=_K,
                team_handle=team,
                fwd_tile_m=tile_m,
            )

        # Warmup fwd+bwd (eager), then release symm buffers.
        out = fwd()
        dY = torch.randn_like(out[0])
        bwd(dY, out)
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()

        # Eager reference grads, snapshotted to CPU.
        out = fwd()
        dY = torch.randn_like(out[0])
        dY_cpu = dY.detach().cpu().clone()
        dX_e, dB_e, dC_e, dA_e, dW_e = bwd(dY, out)
        torch.cuda.synchronize()
        dX_ref = dX_e.detach().cpu().clone()
        dB_ref = dB_e.detach().cpu().clone()
        dC_ref = dC_e.detach().cpu().clone()
        dA_ref = dA_e.detach().cpu().clone()
        tvm_ffi.moe_pop_fwd()
        del out, dY, dX_e, dB_e, dC_e, dA_e, dW_e
        torch.cuda.synchronize()

        # Capture fwd into one graph (keep the symm buffers — bwd consumes them).
        mempool = torch.cuda.graph_pool_handle()
        fwd_g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd_g, pool=mempool, capture_error_mode="thread_local"):
            fwd_out = fwd()
        torch.cuda.synchronize()

        # Capture bwd into a second graph; release symm after (host-side pop).
        dY_static = torch.zeros_like(fwd_out[0])
        bwd_g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bwd_g, pool=mempool, capture_error_mode="thread_local"):
            grads = bwd(dY_static, fwd_out)
            tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()

        # Replay fwd -> bwd with the real upstream grad.
        dY_static.copy_(dY_cpu.to(dY_static.device))
        fwd_g.replay()
        bwd_g.replay()
        torch.cuda.synchronize()
        dX_c, dB_c, dC_c, dA_c, _dW_c = [t.detach().cpu().clone() for t in grads]

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for name, got in [("dX", dX_c), ("dB", dB_c), ("dC", dC_c), ("dA", dA_c)]:
        assert torch.isfinite(got).all(), f"{name} replay produced non-finite values"
    _check_close(dX_c, dX_ref)
    _check_close(dB_c, dB_ref)
    _check_close(dC_c, dC_ref)
    _check_close(dA_c, dA_ref)


# ── forward: partial combine tile ─────────────────────────────────────────────


def _unaligned_fwd_worker(
    rank: int,
    world_size: int,
    init_file: str,
    unaligned_tokens: int = 45,
):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    try:
        _configure(world_size, max_tokens=unaligned_tokens)
        X, _gate_W, all_B, all_C, all_A, ei, ew = _make_inputs(
            rank,
            world_size,
            tokens=unaligned_tokens,
        )

        all_B_g = _gather_experts(all_B, dist.group.WORLD)
        all_C_g = _gather_experts(all_C, dist.group.WORLD)
        all_A_g = _gather_experts(all_A, dist.group.WORLD)
        Y_ref = _torch_reference_moe(X, ei, ew, all_B_g, all_C_g, all_A_g, _K).cpu()

        Y = tvm_ffi.moe_fused_fwd_bf16(
            X,
            ei,
            ew,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
        )[0].clone()
        tvm_ffi.moe_pop_fwd()
        torch.cuda.synchronize()
        Y_cpu = Y.detach().cpu()

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    assert torch.count_nonzero(Y_ref) > 0
    _check_close(Y_cpu, Y_ref)
    if unaligned_tokens > 32:
        _check_close(Y_cpu[32:], Y_ref[32:])


# ── forward + backward: partial combine tile ─────────────────────────────────


def _unaligned_fwd_bwd_worker(rank: int, world_size: int, init_file: str):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    unaligned_tokens = 45
    aligned_tokens = 64
    try:
        _configure(world_size, max_tokens=aligned_tokens)
        X, _gate_W, all_B, all_C, all_A, ei, ew = _make_inputs(
            rank,
            world_size,
            tokens=unaligned_tokens,
        )
        torch.manual_seed(1000 + rank)
        dY = torch.randn_like(X)

        unaligned_out = tvm_ffi.moe_fused_fwd_bf16(
            X,
            ei,
            ew,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
        )
        unaligned_grads = tvm_ffi.moe_fused_bwd_bf16(
            dY,
            unaligned_out[2],
            unaligned_out[1],
            unaligned_out[4],
            unaligned_out[5],
            unaligned_out[3],
            ei,
            ew,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
            fwd_tile_m=unaligned_out[6],
        )
        torch.cuda.synchronize()
        unaligned_cpu = [tensor.detach().cpu().clone() for tensor in unaligned_grads]
        tvm_ffi.moe_pop_fwd()

        X_padded = torch.zeros(
            aligned_tokens,
            _D,
            dtype=X.dtype,
            device=X.device,
        )
        X_padded[:unaligned_tokens].copy_(X)
        dY_padded = torch.zeros_like(X_padded)
        dY_padded[:unaligned_tokens].copy_(dY)
        ei_padded = torch.empty(
            aligned_tokens,
            _K,
            dtype=ei.dtype,
            device=ei.device,
        )
        ew_padded = torch.empty(
            aligned_tokens,
            _K,
            dtype=ew.dtype,
            device=ew.device,
        )
        ei_padded[:unaligned_tokens].copy_(ei)
        ew_padded[:unaligned_tokens].copy_(ew)
        dummy_experts = torch.arange(
            0,
            _E,
            _E // _K,
            dtype=ei.dtype,
            device=ei.device,
        )
        ei_padded[unaligned_tokens:].copy_(dummy_experts.expand(aligned_tokens - unaligned_tokens, -1))
        ew_padded[unaligned_tokens:].fill_(1.0 / _K)

        padded_out = tvm_ffi.moe_fused_fwd_bf16(
            X_padded,
            ei_padded,
            ew_padded,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
        )
        padded_grads = tvm_ffi.moe_fused_bwd_bf16(
            dY_padded,
            padded_out[2],
            padded_out[1],
            padded_out[4],
            padded_out[5],
            padded_out[3],
            ei_padded,
            ew_padded,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            team_handle=team,
            fwd_tile_m=padded_out[6],
        )
        torch.cuda.synchronize()
        padded_cpu = [tensor.detach().cpu().clone() for tensor in padded_grads]
        tvm_ffi.moe_pop_fwd()

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for name, unaligned, padded in zip(
        ("dX", "dB", "dC", "dA", "dW"),
        unaligned_cpu,
        padded_cpu,
    ):
        if name in ("dX", "dW"):
            padded = padded[:unaligned_tokens]
        assert torch.isfinite(unaligned).all(), f"{name} has non-finite values"
        _check_close(unaligned, padded)


# ── forward + backward: single-token automatic dispatch ──────────────────────


def _single_token_auto_dispatch_worker(
    rank: int,
    world_size: int,
    init_file: str,
):
    _init(rank, world_size, init_file)
    team = nvshmem.team_world()
    tokens = 1 + rank * 7
    hidden_dim = 512
    intermediate_dim = 256
    num_experts = 128
    top_k = 8
    experts_per_pe = num_experts // world_size
    fwd_force = "128,64,4,64,128,64,4,64,4,3,128"
    bwd_force = "2,128,64,4,128,256,64,2,32,64,64,2,128"

    try:
        tvm_ffi.moe_configure_symmetric(
            max_tokens=64,
            hidden_dim=hidden_dim,
            max_num_experts=num_experts,
            max_top_k=top_k,
            num_pes=nvshmem.n_pes(),
            num_hosts=1,
            gpus_per_host=world_size,
        )
        torch.manual_seed(10_000 + rank)
        X = torch.randn(tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
        gate_W = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device="cuda")
        all_B = torch.randn(
            experts_per_pe,
            intermediate_dim,
            hidden_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        all_C = torch.randn_like(all_B)
        all_A = torch.randn(
            experts_per_pe,
            hidden_dim,
            intermediate_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        ei, ew = _route(X, gate_W, top_k)
        dY = torch.randn_like(X)

        def fwd():
            return tvm_ffi.moe_fused_fwd_bf16(
                X,
                ei,
                ew,
                all_B,
                all_C,
                all_A,
                num_experts=num_experts,
                top_k=top_k,
                team_handle=team,
            )

        def bwd(out):
            return tvm_ffi.moe_fused_bwd_bf16(
                dY,
                out[2],
                out[1],
                out[4],
                out[5],
                out[3],
                ei,
                ew,
                all_B,
                all_C,
                all_A,
                num_experts=num_experts,
                top_k=top_k,
                team_handle=team,
                fwd_tile_m=out[6],
            )

        os.environ["LIGER_MOE_FORCE_CONFIG"] = fwd_force
        os.environ["LIGER_MOE_BWD_FORCE_CONFIG"] = bwd_force
        forced_out = fwd()
        forced_grads = bwd(forced_out)
        torch.cuda.synchronize()
        forced_cpu = [forced_out[0].detach().cpu().clone()]
        forced_cpu.extend(tensor.detach().cpu().clone() for tensor in forced_grads)
        tvm_ffi.moe_pop_fwd()

        os.environ.pop("LIGER_MOE_FORCE_CONFIG")
        os.environ.pop("LIGER_MOE_BWD_FORCE_CONFIG")
        auto_out = fwd()
        auto_grads = bwd(auto_out)
        torch.cuda.synchronize()
        auto_cpu = [auto_out[0].detach().cpu().clone()]
        auto_cpu.extend(tensor.detach().cpu().clone() for tensor in auto_grads)
        tvm_ffi.moe_pop_fwd()

        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for name, auto, forced in zip(
        ("Y", "dX", "dB", "dC", "dA", "dW"),
        auto_cpu,
        forced_cpu,
    ):
        assert torch.isfinite(auto).all(), f"{name} has non-finite values"
        _check_close(auto, forced)


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_cuda_graph():
    """Fused MoE forward captures into a CUDA graph and replays correctly.

    Catches host-side syncs / non-capturable NVSHMEM calls in the kernel forward
    and verifies the symmetric-stack push/pop hits stable offsets across replays
    (the second replay would corrupt the first's NVSHMEM writes otherwise).
    """
    _run(_world_size(), _fwd_graph_worker)


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_packed_w13_strided_views():
    """Packed gate/up views avoid copies while matching contiguous weights."""
    _run(_world_size(), _strided_fwd_worker)


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_bwd_cuda_graph():
    """Fused MoE forward and backward each capture into a CUDA graph and replay.

    The backward runs under capture reading the forward's symmetric intermediates;
    grads from the replayed graphs must match the eager reference.
    """
    _run(_world_size(), _fwd_bwd_graph_worker)


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_unaligned_tokens():
    """A partial 32-token combine tile must produce every output row."""

    _run(_world_size(), _unaligned_fwd_worker)


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_bwd_unaligned_tokens_match_padded():
    """Unaligned forward/backward must match zero-padded execution."""

    _run(_world_size(), _unaligned_fwd_bwd_worker)


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_moe_fwd_bwd_single_token_auto_dispatch():
    """Automatic dispatch supports a rank with one token.

    The smallest rank has TKE=floor(T * top_k / experts_per_pe)=0 before
    clamping. Its automatic forward and backward results must match known
    compiled configurations while every rank participates in NVSHMEM.
    """
    _run(_world_size(), _single_token_auto_dispatch_worker)
