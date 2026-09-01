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

    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi
except ImportError:
    torch = None
    dist = None
    mp = None
    nvshmem = None
    tvm_ffi = None

pytestmark = pytest.mark.skipif(
    nvshmem is None,
    reason="liger_cute_kernels dependencies are not installed",
)

_NATIVE_AVAILABLE = tvm_ffi is not None and tvm_ffi.is_available()
_NDEV = torch.cuda.device_count() if _NATIVE_AVAILABLE and torch is not None and torch.cuda.is_available() else 0


def test_fslce_forward_binding_is_exposed():
    if not _NATIVE_AVAILABLE:
        pytest.skip("requires the native core")
    module = tvm_ffi._load_module()
    for name in (
        "fused_linear_scaled_cross_entropy_configure_forward",
        "fused_linear_scaled_cross_entropy_forward",
    ):
        assert hasattr(module, name), f"missing native TVM FFI export: {name}"


def test_forward_returns_zero_entropy_when_disabled(monkeypatch):
    class FakeModule:
        @staticmethod
        def fused_linear_scaled_cross_entropy_forward(*args):
            del args

    monkeypatch.setattr(tvm_ffi, "_load_module", lambda: FakeModule())
    x = torch.empty((2, 8), dtype=torch.bfloat16)
    weight = torch.empty((4, 8), dtype=torch.bfloat16)
    target = torch.zeros(2, dtype=torch.int64)
    _, _, entropy = tvm_ffi.fused_linear_scaled_cross_entropy_forward(
        x,
        weight,
        target,
        vocab_start=0,
        ignore_index=-100,
        inverse_temperature=1.0,
        team_handle=1,
        return_entropy=False,
    )
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))


def _reference(x, weights, target, grad_output, temperature):
    logits = (x.float() @ weights.float().T) / temperature
    lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - lse[:, None]
    probabilities = torch.softmax(logits, dim=-1)
    rows = torch.arange(x.shape[0], device=x.device)
    nll = -log_probs[rows, target]
    entropy = -(probabilities * log_probs).sum(dim=-1)
    dz = probabilities
    dz[torch.arange(x.shape[0], device=x.device), target] -= 1.0
    dz *= grad_output[:, None] / temperature
    dz = dz.to(torch.bfloat16).float()
    return nll, lse, entropy, dz @ weights.float(), dz


def _worker(rank: int, world_size: int, init_file: str, subgroup: bool):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )

    team = None
    try:
        nvshmem.init_from_pg()
        pg = dist.group.WORLD
        group_ranks = list(range(world_size))
        if subgroup:
            groups = ([0, 2], [1, 3])
            for ranks in groups:
                candidate = dist.new_group(ranks=ranks)
                if rank in ranks:
                    pg = candidate
                    group_ranks = list(ranks)
            team = nvshmem.resolve_team(pg)
        else:
            team = nvshmem.resolve_team(None)

        local_rank = group_ranks.index(rank)
        team_size = len(group_ranks)
        tokens = 128
        hidden = 2048
        local_vocab = 320
        temperature = 0.8
        global_vocab = team_size * local_vocab

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2027 + (rank & 1 if subgroup else 0))
        x = torch.randn(tokens, hidden, generator=generator).mul_(0.05).to(device="cuda", dtype=torch.bfloat16)
        weight_generator = torch.Generator(device="cpu")
        weight_generator.manual_seed(3100 + local_rank)
        weight = (
            torch.randn(local_vocab, hidden, generator=weight_generator)
            .mul_(0.05)
            .to(device="cuda", dtype=torch.bfloat16)
        )
        gathered = [torch.empty_like(weight) for _ in range(team_size)]
        dist.all_gather(gathered, weight, group=pg)
        global_weight = torch.cat(gathered, dim=0)

        target = torch.arange(tokens, device="cuda", dtype=torch.int64) % global_vocab
        grad_output = torch.linspace(0.5, 1.0, tokens, device="cuda", dtype=torch.float32)
        entropy = torch.zeros(tokens, device="cuda", dtype=torch.float32)
        entropy_grad = torch.zeros_like(entropy)
        expected_nll, expected_lse, expected_entropy, expected_dx, dz = _reference(
            x, global_weight, target, grad_output, temperature
        )
        expected_dw = dz[:, local_rank * local_vocab : (local_rank + 1) * local_vocab].T @ x.float()

        tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(tokens, local_vocab)
        tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(tokens, hidden, local_vocab, 1, team)
        actual_nll, lse, entropy = tvm_ffi.fused_linear_scaled_cross_entropy_forward(
            x,
            weight,
            target,
            local_rank * local_vocab,
            -100,
            1.0 / temperature,
            team,
            True,
        )
        actual_dx, actual_dw = tvm_ffi.fused_linear_scaled_cross_entropy_backward(
            grad_output,
            entropy_grad,
            x,
            weight,
            target,
            lse,
            entropy,
            local_rank * local_vocab,
            -100,
            1.0 / temperature,
            team,
            1,
            False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_nll, expected_nll, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(entropy, expected_entropy, atol=2e-4, rtol=2e-4)

        replay_nll = torch.empty_like(actual_nll)
        replay_lse = torch.empty_like(lse)
        replay_entropy = torch.empty_like(entropy)
        module = tvm_ffi._load_module()
        forward_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(forward_graph):
            module.fused_linear_scaled_cross_entropy_forward(
                x,
                weight,
                target,
                local_rank * local_vocab,
                -100,
                1.0 / temperature,
                team,
                True,
                replay_nll,
                replay_lse,
                replay_entropy,
            )
        dist.barrier()
        for _ in range(2):
            replay_nll.fill_(float("nan"))
            replay_lse.fill_(float("nan"))
            replay_entropy.fill_(float("nan"))
            torch.cuda.synchronize()
            forward_graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replay_nll, expected_nll, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(replay_lse, expected_lse, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(replay_entropy, expected_entropy, atol=2e-4, rtol=2e-4)
        del forward_graph

        torch.testing.assert_close(actual_dx.float(), expected_dx, atol=8e-3, rtol=4e-2)
        torch.testing.assert_close(actual_dw.float(), expected_dw, atol=8e-3, rtol=4e-2)

        nvshmem.pool_clear_all()
        tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(tokens, hidden, local_vocab, 1, team)
        replay_dx = torch.empty_like(x)
        replay_dw = torch.empty_like(weight)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            module.fused_linear_scaled_cross_entropy_backward(
                grad_output,
                entropy_grad,
                x,
                weight,
                target,
                lse,
                entropy,
                local_rank * local_vocab,
                -100,
                1.0 / temperature,
                team,
                1,
                False,
                replay_dx,
                replay_dw,
            )
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replay_dx.float(), expected_dx, atol=8e-3, rtol=4e-2)
        torch.testing.assert_close(replay_dw.float(), expected_dw, atol=8e-3, rtol=4e-2)
        replay_dx.fill_(float("nan"))
        replay_dw.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replay_dx.float(), expected_dx, atol=8e-3, rtol=4e-2)
        torch.testing.assert_close(replay_dw.float(), expected_dw, atol=8e-3, rtol=4e-2)

        nvshmem.pool_clear_all()
        if subgroup and team != nvshmem.team_world():
            nvshmem.team_destroy(team)
            team = None
        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise


def _run(world_size: int, subgroup: bool):
    rendezvous = tempfile.mkdtemp(prefix="fslce_team_")
    init_file = os.path.join(rendezvous, "store")
    old_heap = os.environ.get("NVSHMEM_SYMMETRIC_SIZE")
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = "3G"
    try:
        mp.spawn(
            _worker,
            args=(world_size, init_file, subgroup),
            nprocs=world_size,
            join=True,
        )
    finally:
        if old_heap is None:
            os.environ.pop("NVSHMEM_SYMMETRIC_SIZE", None)
        else:
            os.environ["NVSHMEM_SYMMETRIC_SIZE"] = old_heap
        shutil.rmtree(rendezvous, ignore_errors=True)


@pytest.mark.skipif(
    not _NATIVE_AVAILABLE or _NDEV < 2,
    reason="requires the native core and at least two GPUs",
)
def test_fslce_forward_backward_world_team():
    _run(2, subgroup=False)


@pytest.mark.skipif(
    not _NATIVE_AVAILABLE or _NDEV < 4,
    reason="requires the native core and at least four GPUs",
)
def test_fslce_forward_backward_process_group_subteams():
    _run(4, subgroup=True)
