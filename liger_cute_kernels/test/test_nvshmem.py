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
    import liger_cute_kernels.tvm_ffi as tvm_ffi
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from liger_cute_kernels import nvshmem
except ImportError:
    torch = None
    dist = None
    mp = None
    tvm_ffi = None
    nvshmem = None

pytestmark = pytest.mark.skipif(nvshmem is None, reason="liger_cute_kernels dependencies are not installed")

_NATIVE_AVAILABLE = tvm_ffi is not None and tvm_ffi.is_available()
_NDEV = torch.cuda.device_count() if (_NATIVE_AVAILABLE and torch is not None and torch.cuda.is_available()) else 0

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


@pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="liger_cute_kernels native core is not built")
def test_extension_exposes_nvshmem_bindings():
    for name in _NVSHMEM_BINDINGS:
        assert hasattr(tvm_ffi, name), f"missing binding: {name}"


@pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="liger_cute_kernels native core is not built")
def test_uniqueid_nbytes_positive():
    # sizeof(nvshmemx_uniqueid_t) — a compile-time constant; no runtime needed.
    n = tvm_ffi.uniqueid_nbytes()
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


@pytest.fixture
def isolated_team_state():
    nvshmem._reset_team_state()
    yield
    nvshmem._reset_team_state()


def test_team_from_pg_uses_bootstrap_pe_numbering(monkeypatch, isolated_team_state):
    bootstrap_pg = object()
    subgroup_pg = object()
    bootstrap_global_ranks = (1, 3, 5, 7)
    subgroup_global_ranks = (3, 7)
    nvshmem._BOOTSTRAP_PG = bootstrap_pg
    nvshmem._BOOTSTRAP_GLOBAL_RANKS = bootstrap_global_ranks

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda pg=None: 2 if pg is subgroup_pg else 4)
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda pg, rank: subgroup_global_ranks[rank] if pg is subgroup_pg else bootstrap_global_ranks[rank],
    )
    monkeypatch.setattr(dist, "get_rank", lambda pg=None: 1 if pg is bootstrap_pg else 3)

    def fake_all_gather_object(output, value, group=None):
        assert group is bootstrap_pg
        assert value == (1, 3)
        output[:] = [(0, 2), (1, 3), (0, 2), (1, 3)]

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(tvm_ffi, "team_world", lambda: 42)
    splits = []

    def fake_split(parent, start, stride, size):
        splits.append((parent, start, stride, size))
        return 100 + len(splits)

    monkeypatch.setattr(tvm_ffi, "team_split_strided", fake_split)

    handle = nvshmem.team_from_pg(subgroup_pg)

    assert handle == 102
    assert splits == [(42, 0, 2, 2), (42, 1, 2, 2)]


def test_uncached_subgroup_lookup_never_creates_team(monkeypatch, isolated_team_state):
    pg = object()
    monkeypatch.setattr(nvshmem, "_pg_parent_pes", lambda _: (0, 2))
    monkeypatch.setattr(nvshmem, "_bootstrap_context", lambda: (object(), (0, 1, 2, 3)))
    create_calls = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda _: create_calls.append(True) or 17)

    with pytest.raises(RuntimeError, match="collectively during distributed setup"):
        nvshmem.resolve_team(pg, create=False)
    assert create_calls == []

    assert nvshmem.resolve_team(pg) == 17
    assert nvshmem.resolve_team(pg, create=False) == 17
    assert create_calls == [True]


def test_init_pmi_records_aligned_torch_world(monkeypatch, isolated_team_state):
    world_pg = dist.group.WORLD
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda pg=None: 4)
    monkeypatch.setattr(dist, "get_rank", lambda pg=None: 2)
    monkeypatch.setattr(dist, "get_global_rank", lambda pg, rank: rank)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(slice(None), [value] * 4),
    )
    monkeypatch.setattr(tvm_ffi, "init_pmi", lambda: None)
    monkeypatch.setattr(tvm_ffi, "n_pes", lambda: 4)
    monkeypatch.setattr(tvm_ffi, "my_pe", lambda: 2)

    nvshmem.init_pmi()

    assert nvshmem._BOOTSTRAP_PG is world_pg
    assert nvshmem._BOOTSTRAP_GLOBAL_RANKS == (0, 1, 2, 3)


def test_init_pmi_without_matching_world_keeps_world_only_mode(monkeypatch, isolated_team_state):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda pg=None: 4)
    monkeypatch.setattr(tvm_ffi, "init_pmi", lambda: None)
    monkeypatch.setattr(tvm_ffi, "n_pes", lambda: 2)

    nvshmem.init_pmi()

    assert nvshmem._BOOTSTRAP_PG is None
    assert nvshmem._BOOTSTRAP_GLOBAL_RANKS is None


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
            tvm_ffi.moe_configure_symmetric(
                max_tokens=64,
                hidden_dim=512,
                max_num_experts=world_size,
                max_top_k=1,
                num_pes=world_size,
                num_hosts=1,
                gpus_per_host=world_size,
            )
            tvm_ffi.pool_clear_all()

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


def _nonworld_bootstrap_worker(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )

    bootstrap_groups = ([0, 2], [1, 3])
    bootstrap_pg = None
    singleton_pg = None
    for group in (*bootstrap_groups, [0], [1], [2], [3]):
        pg = dist.new_group(ranks=group)
        if rank in group:
            if len(group) == 2:
                bootstrap_pg = pg
                bootstrap_ranks = group
            else:
                singleton_pg = pg

    sub_team = None
    checks = []

    def expect(name, got, expected):
        checks.append((name, got, expected))

    try:
        nvshmem.init_from_pg(bootstrap_pg)
        world_team = nvshmem.team_world()
        expect("n_pes", nvshmem.n_pes(), 2)
        expect("my_pe", nvshmem.my_pe(), bootstrap_ranks.index(rank))

        try:
            nvshmem.resolve_team(singleton_pg, create=False)
        except RuntimeError as exc:
            expect("uncached_lookup_error", "collectively during distributed setup" in str(exc), True)
        else:
            expect("uncached_lookup_error", False, True)

        sub_team = nvshmem.resolve_team(singleton_pg)
        expect("cached_handle", nvshmem.resolve_team(singleton_pg, create=False), sub_team)
        expect("subteam_size", nvshmem.team_n_pes(sub_team), 1)
        expect("subteam_pe", nvshmem.team_my_pe(sub_team), 0)
        expect(
            "translated_pe",
            nvshmem.team_translate_pe(sub_team, 0, world_team),
            bootstrap_ranks.index(rank),
        )

        nvshmem.team_destroy(sub_team)
        sub_team = None
        dist.barrier()
        nvshmem.finalize()
        dist.destroy_process_group()
    except BaseException:
        try:
            if sub_team is not None:
                nvshmem.team_destroy(sub_team)
            nvshmem.finalize()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise

    for name, got, expected in checks:
        assert got == expected, f"[rank {rank}] {name}: got {got!r}, expected {expected!r}"


def _run_nonworld_bootstrap():
    rdzv_dir = tempfile.mkdtemp(prefix="lck_nonworld_rdzv_")
    init_file = os.path.join(rdzv_dir, "store")
    try:
        mp.spawn(
            _nonworld_bootstrap_worker,
            args=(4, init_file),
            nprocs=4,
            join=True,
        )
    finally:
        shutil.rmtree(rdzv_dir, ignore_errors=True)


# ── Distributed tests (real PEs) ─────────────────────────────────────────────


@pytest.mark.skipif(_NDEV < 1, reason="needs a CUDA device")
def test_single_pe_world_bootstrap_short_circuit_and_configure():
    _run(1, "world")


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


@pytest.mark.skipif(_NDEV < 4, reason="needs >=4 CUDA devices")
def test_nonworld_bootstrap_maps_global_ranks_to_parent_pes():
    _run_nonworld_bootstrap()
