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
    from liger_kernel.ops.cute import fused_linear_scaled_cross_entropy_tp as native_frontend

    _NATIVE_AVAILABLE = native_frontend.is_available()
except ImportError:
    torch = None
    dist = None
    mp = None
    LigerFusedLinearScaledCrossEntropyTPFunction = None
    native_frontend = None
    _NATIVE_AVAILABLE = False

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
    from liger_kernel.ops.configure import MoEConfig
    from liger_kernel.ops.configure import configure
    from liger_kernel.ops.cute.ops.moe import moe_fused

    assert configure(
        process_groups={"ep": dist.group.WORLD},
        device=torch.device("cuda", rank),
        moe=MoEConfig(
            max_tokens=_MOE_TOKENS,
            hidden_size=_MOE_HIDDEN,
            num_experts=_MOE_EXPERTS,
            top_k=_MOE_TOP_K,
            num_hosts=1,
            gpus_per_host=world_size,
            group="ep",
        ),
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

        frontend._load_native_tp_function = lambda: None
        nvshmem = None
    else:
        from liger_cute_kernels import nvshmem

        from liger_kernel.ops.configure import FusedLinearCrossEntropyConfig
        from liger_kernel.ops.configure import configure

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
        if implementation == "native":
            assert configure(
                process_groups={"tp": tp_group},
                device=torch.device("cuda", rank),
                flsce=FusedLinearCrossEntropyConfig(
                    max_tokens=_TOKENS + 128,
                    hidden_size=_HIDDEN,
                    local_vocab_size=_LOCAL_VOCAB,
                    group="tp",
                ),
            )
            nvshmem_initialized = True
            team_handle = nvshmem.resolve_team(tp_group, create=False)
            assert configure(
                device=torch.device("cuda", rank),
                flsce=FusedLinearCrossEntropyConfig(
                    max_tokens=_TOKENS,
                    hidden_size=_HIDDEN,
                    local_vocab_size=_LOCAL_VOCAB,
                    group="tp",
                ),
            )
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

        if implementation == "native":
            with pytest.raises(ValueError, match="exceeds configured capacity"):
                LigerFusedLinearScaledCrossEntropyTPFunction.apply(
                    x.new_zeros((_TOKENS + 129, _HIDDEN)),
                    weight,
                    target.new_zeros((_TOKENS + 129,)),
                    tp_group,
                )

        torch.cuda.synchronize()
        if implementation == "native":
            nvshmem.pool_clear_all()
            if team_handle != nvshmem.team_world():
                nvshmem.team_destroy(team_handle)
                team_handle = None
        dist.barrier()
        if implementation == "native":
            nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        if implementation == "native":
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


def _run(
    world_size: int,
    layout: str,
    implementation: str,
    run_moe: bool = False,
    disable_nvls: bool = False,
):
    rendezvous = tempfile.mkdtemp(prefix=f"liger_fslce_tp_{layout}_")
    init_file = os.path.join(rendezvous, "store")
    env = {
        "NVSHMEM_DISABLE_NCCL": "1",
        "NVSHMEM_DISABLE_NVLS": "1" if disable_nvls else "0",
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


def _multi_context_worker(rank, world_size, init_file, bootstrap_subgroups, partial_overlap):
    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi

    from liger_kernel.ops.configure import FusedLinearCrossEntropyConfig
    from liger_kernel.ops.configure import configure

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    bootstrap = dist.group.WORLD
    bootstrap_partitions = [list(range(world_size))]
    if bootstrap_subgroups:
        bootstrap_partitions = [list(range(0, world_size, 2)), list(range(1, world_size, 2))]
        for ranks in bootstrap_partitions:
            group = dist.new_group(ranks)
            if rank in ranks:
                bootstrap = group

    # Every torch rank creates groups in the same order, even when NVSHMEM is
    # bootstrapped independently inside two non-WORLD groups.
    local_groups = []
    for layout in range(4):
        for members in bootstrap_partitions:
            if layout == 0:
                partitions = [members[i : i + 2] for i in range(0, len(members), 2)]
            elif layout == 1:
                if partial_overlap:
                    partitions = [members[:2], [members[2], members[4]], [members[3], members[5]]]
                else:
                    partitions = [members[::2], members[1::2]]
            elif layout == 2:
                partitions = [[member] for member in members]
            else:
                partitions = (
                    [members] if not partial_overlap else [members[i : i + 2] for i in range(0, len(members), 2)]
                )
            for ranks in partitions:
                group = dist.new_group(ranks)
                if rank in ranks:
                    local_groups.append((group, ranks))

    cases = []
    first_workspace_bytes = None
    previous_workspace_bytes = 0
    for index, (group, ranks) in enumerate(local_groups):
        name = f"tp{index}"
        assert configure(
            process_groups={name: group},
            bootstrap_group=bootstrap,
            device=f"cuda:{rank}",
            flsce=FusedLinearCrossEntropyConfig(_TOKENS + 128, _HIDDEN, _LOCAL_VOCAB, group=name),
        )
        workspace_bytes = tvm_ffi.fused_linear_scaled_cross_entropy_backward_workspace_bytes(
            _TOKENS + 128, _HIDDEN, _LOCAL_VOCAB, 1
        ) + tvm_ffi.fused_linear_scaled_cross_entropy_forward_workspace_bytes(_TOKENS + 128, _LOCAL_VOCAB)
        if first_workspace_bytes is None:
            first_workspace_bytes = workspace_bytes
        else:
            # Additional contexts must add only small signaling/mapping state,
            # not another copy of the bulk scratch for this fixed test shape.
            assert 0 <= workspace_bytes - previous_workspace_bytes < first_workspace_bytes // 16
        previous_workspace_bytes = workspace_bytes
        generator = torch.Generator().manual_seed(8011 + ranks[0])
        tokens = _TOKENS - index * 16
        x = (
            torch.randn(tokens, _HIDDEN, generator=generator)
            .mul_(0.05)
            .to(device="cuda", dtype=torch.bfloat16)
            .requires_grad_(True)
        )
        generator.manual_seed(9000 + rank)
        weight = (
            torch.randn(_LOCAL_VOCAB, _HIDDEN, generator=generator)
            .mul_(0.05)
            .to(device="cuda", dtype=torch.bfloat16)
            .requires_grad_(True)
        )
        weights = [torch.empty_like(weight) for _ in ranks]
        dist.all_gather(weights, weight.detach(), group=group)
        target = torch.arange(tokens, device="cuda", dtype=torch.int64) % (len(ranks) * _LOCAL_VOCAB)
        grad_nll = torch.linspace(0.25, 1.0, tokens, device="cuda")
        grad_entropy = torch.linspace(-0.2, 0.3, tokens, device="cuda")
        nll, entropy, dx, dz = _reference(x.detach(), torch.cat(weights), target, grad_nll, grad_entropy)
        start = ranks.index(rank) * _LOCAL_VOCAB
        dw = dz[:, start : start + _LOCAL_VOCAB].t() @ x.detach().float()
        case = (group, x, weight, target, grad_nll, grad_entropy, (nll, entropy, dx, dw))
        cases.append(case)
        # Unequal call counts on disjoint old teams must not contaminate the
        # epochs of a later overlapping partition.
        for _ in range(1 + ranks[0] % 3):
            with torch.no_grad():
                LigerFusedLinearScaledCrossEntropyTPFunction.apply(
                    x,
                    weight,
                    target,
                    group,
                    _TEMPERATURE,
                    _IGNORE_INDEX,
                    1,
                    True,
                )
        torch.cuda.synchronize()
        dist.barrier(group=bootstrap)

    def forward(case):
        group, x, weight, target, _, _, _ = case
        return LigerFusedLinearScaledCrossEntropyTPFunction.apply(
            x,
            weight,
            target,
            group,
            _TEMPERATURE,
            _IGNORE_INDEX,
            1,
            True,
        )

    for _ in range(2):
        outputs = [forward(case) for case in cases]
        for case, actual in reversed(list(zip(cases, outputs))):
            _, x, weight, _, grad_nll, grad_entropy, expected = case
            x.grad = weight.grad = None
            torch.autograd.backward(actual, (grad_nll, grad_entropy))
            torch.testing.assert_close(actual[0], expected[0], atol=3e-4, rtol=3e-4)
            torch.testing.assert_close(actual[1], expected[1], atol=3e-4, rtol=3e-4)
            torch.testing.assert_close(x.grad.float(), expected[2], atol=8e-3, rtol=4e-2)
            torch.testing.assert_close(weight.grad.float(), expected[3], atol=8e-3, rtol=4e-2)

    graphs = []
    for case in cases[:2]:
        group, x, weight, target, grad_nll, grad_entropy, expected = case
        team = nvshmem.resolve_team(group, create=False)
        vocab_start = dist.get_rank(group) * _LOCAL_VOCAB
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            nll, lse, entropy = tvm_ffi.fused_linear_scaled_cross_entropy_forward(
                x,
                weight,
                target,
                vocab_start,
                _IGNORE_INDEX,
                1.0 / _TEMPERATURE,
                team,
                True,
            )
            dx, dw = tvm_ffi.fused_linear_scaled_cross_entropy_backward(
                grad_nll,
                grad_entropy,
                x,
                weight,
                target,
                lse,
                entropy,
                vocab_start,
                _IGNORE_INDEX,
                1.0 / _TEMPERATURE,
                team,
                1,
                True,
            )
        graphs.append((graph, (nll, entropy, dx, dw), expected))
    for _ in range(3):
        with torch.no_grad():
            forward(cases[-1])
        for graph, actual, expected in reversed(graphs):
            graph.replay()
            torch.cuda.synchronize()
            for index, (result, reference) in enumerate(zip(actual, expected)):
                atol, rtol = (3e-4, 3e-4) if index < 2 else (8e-3, 4e-2)
                torch.testing.assert_close(result.float(), reference, atol=atol, rtol=rtol)
    del graphs, graph
    torch.cuda.synchronize()
    original_team = nvshmem.resolve_team(cases[0][0], create=False)
    nvshmem.team_destroy(original_team)
    with pytest.raises(RuntimeError, match="not initialized"):
        nvshmem.resolve_team(cases[0][0], create=False)
    assert configure(
        bootstrap_group=bootstrap,
        device=f"cuda:{rank}",
        flsce=FusedLinearCrossEntropyConfig(_TOKENS + 128, _HIDDEN, _LOCAL_VOCAB, group="tp0"),
    )
    with torch.no_grad():
        actual = forward(cases[0])
    torch.testing.assert_close(actual[0], cases[0][-1][0], atol=3e-4, rtol=3e-4)
    torch.cuda.synchronize()
    nvshmem.pool_clear_all()
    # Reusing the same team after pool clear must rebuild its native context.
    assert configure(
        bootstrap_group=bootstrap,
        device=f"cuda:{rank}",
        flsce=FusedLinearCrossEntropyConfig(_TOKENS + 128, _HIDDEN, _LOCAL_VOCAB, group="tp0"),
    )
    with torch.no_grad():
        actual = forward(cases[0])
    torch.testing.assert_close(actual[0], cases[0][-1][0], atol=3e-4, rtol=3e-4)
    torch.cuda.synchronize()
    nvshmem.finalize()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "bootstrap_subgroups", "partial_overlap", "disable_nvls"),
    [(4, False, False, False), (6, False, True, False), (8, True, False, False), (4, False, False, True)],
)
def test_native_multiple_tp_contexts(world_size, bootstrap_subgroups, partial_overlap, disable_nvls, monkeypatch):
    if not _NATIVE_AVAILABLE or _NDEV < world_size:
        pytest.skip(f"requires native LCK and at least {world_size} CUDA devices")
    monkeypatch.setenv("NVSHMEM_DISABLE_NCCL", "1")
    monkeypatch.setenv("NVSHMEM_DISABLE_NVLS", "1" if disable_nvls else "0")
    monkeypatch.setenv("NVSHMEM_REMOTE_TRANSPORT", "none")
    monkeypatch.setenv("NVSHMEM_SYMMETRIC_SIZE", "6G")
    with tempfile.TemporaryDirectory(prefix="liger_tp_contexts_") as rendezvous:
        mp.spawn(
            _multi_context_worker,
            args=(world_size, os.path.join(rendezvous, "store"), bootstrap_subgroups, partial_overlap),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_tp_frontend_world_process_group(world_size):
    if not _NATIVE_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < world_size:
        pytest.skip(f"requires at least {world_size} CUDA devices")
    _run(world_size, "world", "native")


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_tp_frontend_subgroups(layout):
    if not _NATIVE_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < 4:
        pytest.skip("requires at least four CUDA devices")
    _run(4, layout, "native")


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


def test_native_moe_and_tp_frontend_use_different_process_groups():
    if not _NATIVE_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < 4:
        pytest.skip("requires at least four CUDA devices")
    _run(4, "strided", "native", run_moe=True)


def test_tp_frontend_direct_peer_fallback():
    if not _NATIVE_AVAILABLE:
        pytest.skip("requires a matching liger_cute_kernels wheel")
    if _NDEV < 2:
        pytest.skip("requires at least two CUDA devices")
    _run(2, "world", "native", disable_nvls=True)
