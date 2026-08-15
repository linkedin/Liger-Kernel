"""Tests for the ``cute`` backend's fused expert-parallel MoE autograd op.

Two layers of coverage:

  * ``test_cute_registration`` — pure-Python, always runs. Asserts the ``cute``
    implementation self-registers as an *opt-in* CUDA backend (no
    ``default_devices``, selected only via ``LIGER_KERNEL_IMPL=cute``). This
    needs neither a GPU nor the lck wheel, so it gives signal on plain CI too.

  * ``test_moe_fused_autograd`` — the functional test, **skipped unless the lck
    wheel is installed and >= 2 CUDA devices are present**. It drives the
    autograd wrapper ``liger_kernel.ops.cute.ops.moe_fused`` (not the raw TVM FFI
    bindings) through both paths: the no-grad fast path (which must pop the
    symmetric stack immediately) and the grad path (with-intermediates fwd + a
    backward that pops the stack after consuming the saved tensors). The
    expert-parallel ``ProcessGroup`` is passed through so the in-``forward``
    pg -> NVSHMEM team translation is exercised.

No mocks: like ``liger_cute_kernels/test/test_moe_cuda_graph.py`` it spawns one
real process per GPU, brings up a real NCCL group + NVSHMEM runtime via
``nvshmem.init_from_pg``, and gathers the per-rank experts for the torch
reference. Shapes (T=512, D=4096, I=2048, E=16, K=2) match a row the tuned
dispatch table covers so the auto path resolves a config.
"""

from __future__ import annotations

import os
import tempfile

from datetime import timedelta

import pytest

from liger_kernel.ops.backends.registry import IMPL_REGISTRY
from liger_kernel.ops.backends.registry import select_impl

# ── always-on registration check ─────────────────────────────────────────────


def test_cute_registration():
    """The ``cute`` backend registers itself as an opt-in CUDA implementation.

    Importing ``liger_kernel.ops`` runs impl discovery, which imports every
    backend's ``__init__`` (registration is metadata-only and must not need the
    lck wheel). ``cute`` declares no ``default_devices``, so it is never
    auto-applied — only an explicit ``LIGER_KERNEL_IMPL=cute`` selects it.
    """
    import liger_kernel.ops  # noqa: F401  -- triggers _discover_impls()

    info = IMPL_REGISTRY.get("cute")
    assert info is not None, "cute implementation did not self-register"
    assert info.devices == ("cuda",)
    assert info.default_devices == (), "cute must be opt-in only (no default_devices)"
    assert info.module_path == "liger_kernel.ops.cute.ops"

    # Opt-in semantics: not auto-applied on cuda, but explicit selection resolves.
    assert select_impl("cuda") is None, "cute must not be auto-applied on cuda"
    assert select_impl("cuda", explicit="cute") is info


# ── functional autograd test (needs the lck wheel + GPUs) ─────────────────────

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as Fnn

    from liger_kernel.ops import cute as _cute

    _LCK_AVAILABLE = _cute.is_available()
except ImportError:  # torch missing, or liger_kernel import failure
    torch = None
    dist = None
    mp = None
    Fnn = None
    _cute = None
    _LCK_AVAILABLE = False

_NDEV = torch.cuda.device_count() if (torch is not None and torch.cuda.is_available()) else 0

# MoE shape — matches a row the tuned table covers so the auto path resolves a
# config, and keeps E divisible by every world size we run.
_T, _D, _I, _E, _K = 512, 4096, 2048, 16, 2


def _world_size() -> int:
    """Largest power-of-two divisor of E that fits the available GPUs (>= 2)."""
    want = int(os.environ.get("LIGER_TEST_NPROC", "8"))
    for w in (8, 4, 2):
        if w <= min(want, _NDEV) and _E % w == 0:
            return w
    return 2


def _route(X, gate_W, top_k):
    """Standard top-K gate in plain torch (stands in for the native router).

    The same indices/weights feed the kernel and the reference, so absolute
    routing values are irrelevant — only consistency matters.
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


def _autograd_worker(rank, world_size, init_file):
    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi

    from liger_kernel.ops.cute.ops import LigerExpertParallelFusedMoEFunction
    from liger_kernel.ops.cute.ops import moe_fused

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

        # Reference uses the globally-gathered experts.
        all_B_g = _gather_experts(all_B_data, dist.group.WORLD)
        all_C_g = _gather_experts(all_C_data, dist.group.WORLD)
        all_A_g = _gather_experts(all_A_data, dist.group.WORLD)
        Y_ref = _torch_reference_moe(X_data, ei, ew, all_B_g, all_C_g, all_A_g, _K)

        experts_per_pe = all_B_data.size(0)
        packed_w13_data = torch.stack((all_B_data, all_C_data), dim=1).reshape(
            experts_per_pe,
            2 * _I,
            _D,
        )
        strided_B_data = packed_w13_data[:, :_I, :]
        strided_C_data = packed_w13_data[:, _I:, :]
        assert not strided_B_data.is_contiguous()
        assert not strided_C_data.is_contiguous()

        # ── No-grad fast path: must pop the symmetric stack immediately. ──
        with torch.no_grad():
            Y_ng = moe_fused(
                X_data,
                ei,
                ew,
                strided_B_data,
                strided_C_data,
                all_A_data,
                num_experts=_E,
                top_k=_K,
                pg=dist.group.WORLD,
            )
        assert Y_ng.shape == (_T, _D)
        assert not Y_ng.requires_grad, "no_grad path must not require grad"
        _check_close(Y_ng, Y_ref)

        # The Function class is reachable via the ops module (sanity on the export).
        assert LigerExpertParallelFusedMoEFunction is not None

        # Backward does not yet support expert-strided weights. Reject them
        # before forward allocates symmetric intermediates.
        packed_w13 = packed_w13_data.clone().detach().requires_grad_(True)
        strided_B = packed_w13[:, :_I, :]
        strided_C = packed_w13[:, _I:, :]
        with pytest.raises(ValueError, match="gradients are disabled"):
            moe_fused(
                X_data.clone().detach().requires_grad_(True),
                ei,
                ew,
                strided_B,
                strided_C,
                all_A_data.clone().detach().requires_grad_(True),
                num_experts=_E,
                top_k=_K,
                pg=dist.group.WORLD,
            )

        # ── Grad path: with-intermediates fwd, backward consumes + pops. ──
        X = X_data.clone().detach().requires_grad_(True)
        all_B = all_B_data.clone().detach().requires_grad_(True)
        all_C = all_C_data.clone().detach().requires_grad_(True)
        all_A = all_A_data.clone().detach().requires_grad_(True)

        Y = moe_fused(
            X,
            ei,
            ew,
            all_B,
            all_C,
            all_A,
            num_experts=_E,
            top_k=_K,
            pg=dist.group.WORLD,
        )
        assert Y.shape == (_T, _D)
        assert Y.requires_grad, "grad path must yield a tensor that requires grad"
        _check_close(Y, Y_ref)

        # Backward sanity: every grad-requiring input gets a finite grad of the
        # right shape. Numerical bwd correctness is covered by the kernel-level
        # CUDA-graph fwd+bwd test in the lck package.
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
def test_moe_fused_autograd():
    """``moe_fused`` runs its no-grad and grad paths and matches the reference.

    Drives the public autograd wrapper end to end across real PEs, passing the
    expert-parallel process group so the in-``forward`` pg -> team translation
    and both symmetric-stack pop sites (immediate under no_grad, deferred to
    backward otherwise) are exercised.
    """
    import shutil

    world_size = _world_size()
    rdzv = tempfile.mkdtemp(prefix="lck_autograd_")
    init_file = os.path.join(rdzv, "store")
    try:
        mp.spawn(_autograd_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        shutil.rmtree(rdzv, ignore_errors=True)
