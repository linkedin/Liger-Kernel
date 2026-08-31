"""Multi-GPU tests for the tensor-parallel scaled cross-entropy frontend."""

from __future__ import annotations

import os
import shutil
import tempfile

from datetime import timedelta

import pytest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyTPFunction
    from liger_kernel.ops.cute import fused_linear_scaled_cross_entropy_tp as lck_frontend

    _LCK_AVAILABLE = lck_frontend.is_available()
except ImportError:
    torch = None
    dist = None
    mp = None
    LigerFusedLinearScaledCrossEntropyTPFunction = None
    lck_frontend = None
    _LCK_AVAILABLE = False

_NDEV = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
_TOKENS = 521
_HIDDEN = 2048
_LOCAL_VOCAB = 320
_TEMPERATURE = 0.8
_IGNORE_INDEX = -100
_MOE_TOKENS = 512
_MOE_HIDDEN = 4096
_MOE_INTERMEDIATE = 2048
_MOE_EXPERTS = 16
_MOE_TOP_K = 2


def _group_layout(layout: str, world_size: int):
    if layout == "world":
        return [list(range(world_size))]
    if layout == "contiguous":
        return [[0, 1], [2, 3]]
    if layout == "strided":
        return [[0, 2], [1, 3]]
    raise ValueError(f"unknown process-group layout: {layout}")


def _create_tp_group(rank: int, world_size: int, layout: str):
    groups = _group_layout(layout, world_size)
    if layout == "world":
        return dist.group.WORLD, groups[0], []

    tp_group = None
    tp_ranks = None
    created_groups = []
    for ranks in groups:
        group = dist.new_group(ranks=ranks)
        created_groups.append(group)
        if rank in ranks:
            tp_group = group
            tp_ranks = ranks
    if tp_group is None or tp_ranks is None:
        raise RuntimeError(f"rank {rank} was not assigned to a tensor-parallel group")
    return tp_group, tp_ranks, created_groups


def _reference(x, global_weight, target, grad_nll, grad_entropy):
    inverse_temperature = 1.0 / _TEMPERATURE
    logits = x.float() @ global_weight.float().t()
    scaled_logits = logits * inverse_temperature
    lse = torch.logsumexp(scaled_logits, dim=-1)
    probabilities = torch.softmax(scaled_logits, dim=-1)
    entropy = lse - torch.sum(probabilities * scaled_logits, dim=-1)

    valid = target != _IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    rows = torch.arange(x.shape[0], device=x.device)
    nll = lse - scaled_logits[rows, safe_target]
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    entropy = torch.where(valid, entropy, torch.zeros_like(entropy))

    dz = probabilities * grad_nll[:, None]
    dz += probabilities * (lse[:, None] - entropy[:, None] - scaled_logits) * grad_entropy[:, None]
    dz[rows, safe_target] -= grad_nll
    dz *= valid[:, None] * inverse_temperature
    dz = dz.to(torch.bfloat16).float()
    return nll, entropy, dz @ global_weight.float(), dz


def _run_moe(rank: int, world_size: int):
    from liger_cute_kernels import tvm_ffi

    from liger_kernel.ops.cute.ops.moe import moe_fused

    tvm_ffi.moe_configure_symmetric(
        max_tokens=_MOE_TOKENS,
        hidden_dim=_MOE_HIDDEN,
        max_num_experts=_MOE_EXPERTS,
        max_top_k=_MOE_TOP_K,
        num_pes=world_size,
        num_hosts=1,
        gpus_per_host=world_size,
    )
    experts_per_rank = _MOE_EXPERTS // world_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4100 + rank)
    x = torch.randn(_MOE_TOKENS, _MOE_HIDDEN, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    all_b = torch.randn(
        experts_per_rank,
        _MOE_INTERMEDIATE,
        _MOE_HIDDEN,
        generator=generator,
    ).to(device="cuda", dtype=torch.bfloat16)
    all_c = torch.randn(
        experts_per_rank,
        _MOE_INTERMEDIATE,
        _MOE_HIDDEN,
        generator=generator,
    ).to(device="cuda", dtype=torch.bfloat16)
    all_a = torch.randn(
        experts_per_rank,
        _MOE_HIDDEN,
        _MOE_INTERMEDIATE,
        generator=generator,
    ).to(device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(_MOE_TOKENS, device="cuda", dtype=torch.int32)
    expert_indices = torch.stack((rows % _MOE_EXPERTS, (rows + 1) % _MOE_EXPERTS), dim=-1)
    expert_weights = torch.full(
        (_MOE_TOKENS, _MOE_TOP_K),
        1.0 / _MOE_TOP_K,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        output = moe_fused(
            x,
            expert_indices,
            expert_weights,
            all_b,
            all_c,
            all_a,
            _MOE_EXPERTS,
            _MOE_TOP_K,
            dist.group.WORLD,
        )
    torch.cuda.synchronize()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def _worker(rank: int, world_size: int, init_file: str, layout: str, implementation: str, run_moe: bool):
    if implementation == "fallback":
        import liger_kernel.ops.fused_linear_scaled_cross_entropy as frontend

        frontend._load_lck_tp_function = lambda group, device: None
        nvshmem = None
    else:
        from liger_cute_kernels import nvshmem

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )

    nvshmem_initialized = False
    team_handle = None
    try:
        tp_group, tp_ranks, _created_groups = _create_tp_group(rank, world_size, layout)
        tp_rank = tp_ranks.index(rank)
        tp_size = len(tp_ranks)
        group_index = _group_layout(layout, world_size).index(tp_ranks)
        if implementation == "lck":
            nvshmem.init_from_pg()
            nvshmem_initialized = True
            team_handle = nvshmem.resolve_team(tp_group)
            if run_moe:
                _run_moe(rank, world_size)

        x_generator = torch.Generator(device="cpu")
        x_generator.manual_seed(2027 + group_index)
        x = (
            torch.randn(_TOKENS, _HIDDEN, generator=x_generator)
            .mul_(0.05)
            .to(device="cuda", dtype=torch.bfloat16)
            .requires_grad_(True)
        )
        weight_generator = torch.Generator(device="cpu")
        weight_generator.manual_seed(3100 + 17 * group_index + tp_rank)
        weight = (
            torch.randn(_LOCAL_VOCAB, _HIDDEN, generator=weight_generator)
            .mul_(0.05)
            .to(device="cuda", dtype=torch.bfloat16)
            .requires_grad_(True)
        )

        gathered_weights = [torch.empty_like(weight) for _ in range(tp_size)]
        dist.all_gather(gathered_weights, weight.detach(), group=tp_group)
        global_weight = torch.cat(gathered_weights, dim=0)
        global_vocab = tp_size * _LOCAL_VOCAB

        target = (torch.arange(_TOKENS, device="cuda", dtype=torch.int64) * 17 + 3) % global_vocab
        target[::19] = _IGNORE_INDEX
        grad_nll = torch.linspace(0.25, 1.0, _TOKENS, device="cuda", dtype=torch.float32)
        grad_entropy = torch.linspace(-0.2, 0.3, _TOKENS, device="cuda", dtype=torch.float32)

        expected_nll, expected_entropy, expected_dx, dz = _reference(
            x.detach(),
            global_weight,
            target,
            grad_nll,
            grad_entropy,
        )
        local_start = tp_rank * _LOCAL_VOCAB
        expected_dw = dz[:, local_start : local_start + _LOCAL_VOCAB].t() @ x.detach().float()

        actual_nll, actual_entropy = LigerFusedLinearScaledCrossEntropyTPFunction.apply(
            x,
            weight,
            target,
            tp_group,
            _TEMPERATURE,
            _IGNORE_INDEX,
            1,
            True,
        )
        torch.autograd.backward((actual_nll, actual_entropy), (grad_nll, grad_entropy))
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_nll, expected_nll, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(actual_entropy, expected_entropy, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(x.grad.float(), expected_dx, atol=8e-3, rtol=4e-2)
        torch.testing.assert_close(weight.grad.float(), expected_dw, atol=8e-3, rtol=4e-2)

        torch.cuda.synchronize()
        if implementation == "lck":
            nvshmem.pool_clear_all()
            if team_handle != nvshmem.team_world():
                nvshmem.team_destroy(team_handle)
                team_handle = None
        dist.barrier()
        if implementation == "lck":
            nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        if implementation == "lck":
            try:
                if nvshmem_initialized:
                    nvshmem.finalize()
            except Exception:
                pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise


def _run(world_size: int, layout: str, implementation: str, run_moe: bool = False):
    rendezvous = tempfile.mkdtemp(prefix=f"liger_fslce_tp_{layout}_")
    init_file = os.path.join(rendezvous, "store")
    env = {
        "NVSHMEM_DISABLE_NCCL": "1",
        "NVSHMEM_REMOTE_TRANSPORT": "none",
        "NVSHMEM_SYMMETRIC_SIZE": "3G",
    }
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        mp.spawn(
            _worker,
            args=(world_size, init_file, layout, implementation, run_moe),
            nprocs=world_size,
            join=True,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(rendezvous, ignore_errors=True)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_tp_frontend_world_process_group(world_size):
    if not _LCK_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < world_size:
        pytest.skip(f"requires at least {world_size} CUDA devices")
    _run(world_size, "world", "lck")


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_tp_frontend_subgroups(layout):
    if not _LCK_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < 4:
        pytest.skip("requires at least four CUDA devices")
    _run(4, layout, "lck")


@pytest.mark.parametrize(
    ("world_size", "layout"),
    [
        (2, "world"),
        (4, "strided"),
    ],
)
def test_tp_frontend_liger_fallback(world_size, layout):
    if torch is None or _NDEV < world_size:
        pytest.skip(f"requires at least {world_size} CUDA devices")
    _run(world_size, layout, "fallback")


def test_lck_moe_and_tp_frontend_use_different_process_groups():
    if not _LCK_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < 4:
        pytest.skip("requires at least four CUDA devices")
    _run(4, "strided", "lck", run_moe=True)
