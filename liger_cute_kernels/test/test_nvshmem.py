"""Integration tests for the ``liger_cute_kernels`` NVSHMEM bootstrap / team layer.

No mocks. Every distributed test launches one real process per GPU with
``torch.multiprocessing``, initialises a real ``torch.distributed`` process group
(NCCL) and a real NVSHMEM runtime via :func:`liger_cute_kernels.nvshmem.init_from_pg`,
then exercises the actual compiled bindings:

  * bootstrap + per-PE identity (my_pe / n_pes),
  * process-group -> NVSHMEM-team translation (team_from_pg / resolve_team) for
    contiguous and strided sub-groups, including the team-local PE numbering and
    cross-team PE translation,
  * the WORLD short-circuit,
  * comm-schedule upload + pool clears (moe_configure_symmetric / pool_clear_all).

Each worker captures every result BEFORE the collective teardown
(team_destroy / nvshmem_finalize / destroy_process_group) and only asserts
afterwards, so a failing expectation on one rank cannot deadlock the others on a
later collective. ``torch.multiprocessing.spawn`` re-raises any worker exception
(failed assert included) in the parent, failing the test.

Requires real GPUs: the WORLD test needs >=2, the sub-group tests need >=4.
Skipped (not failed) when too few CUDA devices are present.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from datetime import timedelta

import pytest

try:
    import liger_cute_kernels.tvm_ffi as _ext_mod
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from liger_cute_kernels import nvshmem
except ImportError:
    torch = None
    dist = None
    mp = None
    _ext_mod = None
    nvshmem = None

pytestmark = pytest.mark.skipif(
    _ext_mod is None or not _ext_mod.is_available(),
    reason="liger_cute_kernels not built/installed; build it to run these.",
)

_NDEV = (
    torch.cuda.device_count()
    if (_ext_mod is not None and _ext_mod.is_available() and torch is not None and torch.cuda.is_available())
    else 0
)

_NVSHMEM_BINDINGS = (
    "uniqueid_nbytes",
    "get_uniqueid",
    "init_with_uniqueid",
    "init_pmi",
    "finalize",
    "my_pe",
    "n_pes",
    "team_world",
    "team_split_strided",
    "team_destroy",
    "team_my_pe",
    "team_n_pes",
    "team_translate_pe",
    "pool_clear_all",
    "pool_clear_buffers",
)


# ── Surface / pure-host (no runtime) ─────────────────────────────────────────


def test_extension_exposes_nvshmem_bindings():
    for name in _NVSHMEM_BINDINGS:
        assert hasattr(_ext_mod, name), f"missing binding: {name}"


def test_uniqueid_nbytes_positive():
    # sizeof(nvshmemx_uniqueid_t) — a compile-time constant; no runtime needed.
    n = _ext_mod.uniqueid_nbytes()
    assert isinstance(n, int) and n > 0


def test_ranks_to_strided_valid():
    assert nvshmem._ranks_to_strided([3]) == (3, 1, 1)
    assert nvshmem._ranks_to_strided([0, 1, 2, 3]) == (0, 1, 4)
    assert nvshmem._ranks_to_strided([2, 4, 6, 8]) == (2, 2, 4)


@pytest.mark.parametrize(
    "bad",
    [
        [],  # empty
        [4, 2, 0],  # non-positive stride
        [0, 1, 3],  # diffs vary (not arithmetic)
        [0, 2, 3],  # diffs vary
    ],
)
def test_ranks_to_strided_rejects(bad):
    with pytest.raises(ValueError):
        nvshmem._ranks_to_strided(bad)


# ── Distributed harness ──────────────────────────────────────────────────────


def _worker(rank: int, world_size: int, init_file: str, scenario: str, groups):
    """One real PE: init dist + NVSHMEM, exercise the bindings, tear down.

    Returns nothing; raises AssertionError on a mismatch (re-raised by spawn in
    the parent). All result capture happens before the collective teardown.
    """
    torch.cuda.set_device(rank)
    # File-based rendezvous (FileStore) rather than a TCP port: no free-port race
    # and no socket-bind flakiness. The parent supplies a fresh, non-existent path
    # that the store creates and shares across ranks.
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )

    checks = []  # (name, got, expected)

    def expect(name, got, exp):
        checks.append((name, got, exp))

    sub_team = None
    try:
        # Bootstrap NVSHMEM so NVSHMEM_TEAM_WORLD mirrors the WORLD process group.
        nvshmem.init_from_pg()

        world_team = nvshmem.team_world()
        expect("my_pe", nvshmem.my_pe(), rank)
        expect("n_pes", nvshmem.n_pes(), world_size)
        expect("team_world_n_pes", nvshmem.team_n_pes(world_team), world_size)
        expect("team_world_my_pe", nvshmem.team_my_pe(world_team), rank)
        expect("resolve_none_is_world", nvshmem.resolve_team(None), world_team)

        if scenario == "world":
            # A process group covering all of WORLD maps to NVSHMEM_TEAM_WORLD.
            expect("team_from_world_pg", nvshmem.team_from_pg(dist.group.WORLD), world_team)

            # Comm-schedule upload + symmetric sizing (local, per-PE) succeeds on
            # the live runtime, then drop the cached pools.
            _ext_mod.moe_configure_symmetric(
                max_tokens=64,
                hidden_dim=512,
                max_num_experts=world_size,
                max_top_k=1,
                num_pes=world_size,
                num_hosts=1,
                gpus_per_host=world_size,
            )
            _ext_mod.pool_clear_all()

        elif scenario == "subgroups":
            # Build every sub-group collectively (all ranks call new_group for
            # each group, in the same order), keep the one this rank belongs to.
            my_group = None
            my_pg = None
            for g in groups:
                pg = dist.new_group(ranks=g)
                if rank in g:
                    my_group = sorted(g)
                    my_pg = pg

            # resolve_team builds the NVSHMEM team on first call and caches it;
            # the second call must return the identical handle from cache.
            sub_team = nvshmem.resolve_team(my_pg)
            expect("resolve_caches", nvshmem.resolve_team(my_pg), sub_team)

            local_pe = my_group.index(rank)
            expect("sub_is_not_world", sub_team != world_team, True)
            expect("sub_n_pes", nvshmem.team_n_pes(sub_team), len(my_group))
            expect("sub_my_pe", nvshmem.team_my_pe(sub_team), local_pe)
            # team-local PE -> world PE is this rank's global id ...
            expect(
                "translate_sub_to_world",
                nvshmem.team_translate_pe(sub_team, local_pe, world_team),
                rank,
            )
            # ... and world PE -> team-local PE is its index within the group.
            expect(
                "translate_world_to_sub",
                nvshmem.team_translate_pe(world_team, rank, sub_team),
                local_pe,
            )
        else:  # pragma: no cover - guards against a bad scenario string
            raise ValueError(f"unknown scenario {scenario!r}")

        # ── Collective teardown (uniform across ranks; before any assert) ──────
        if sub_team is not None and sub_team != world_team:
            nvshmem.team_destroy(sub_team)  # collective within each sub-team
        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        # Best-effort cleanup so a failure doesn't wedge the process group.
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for name, got, exp in checks:
        assert got == exp, f"[rank {rank}] {name}: got {got!r}, expected {exp!r}"


def _run(world_size: int, scenario: str, groups=None):
    # A fresh temp dir per launch; the FileStore backing file lives inside it and
    # must not pre-exist, so hand the store a path it creates itself.
    rdzv_dir = tempfile.mkdtemp(prefix="lck_rdzv_")
    init_file = os.path.join(rdzv_dir, "store")
    try:
        mp.spawn(
            _worker,
            args=(world_size, init_file, scenario, groups),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(rdzv_dir, ignore_errors=True)


# ── Distributed tests (real PEs) ─────────────────────────────────────────────


@pytest.mark.skipif(_NDEV < 2, reason="needs >=2 CUDA devices")
def test_world_bootstrap_short_circuit_and_configure():
    # 2 PEs: bootstrap, per-PE identity, WORLD-group short-circuit, configure.
    _run(2, "world")


@pytest.mark.skipif(_NDEV < 4, reason="needs >=4 CUDA devices")
def test_team_from_pg_contiguous_subgroups():
    # WORLD={0,1,2,3} -> contiguous EP groups {0,1} and {2,3}.
    _run(4, "subgroups", groups=[[0, 1], [2, 3]])


@pytest.mark.skipif(_NDEV < 4, reason="needs >=4 CUDA devices")
def test_team_from_pg_strided_subgroups():
    # WORLD={0,1,2,3} -> strided EP groups {0,2} and {1,3}.
    _run(4, "subgroups", groups=[[0, 2], [1, 3]])
