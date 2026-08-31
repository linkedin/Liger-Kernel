"""NVSHMEM bootstrap + process-group → team translation (torch-facing layer).

These helpers sit on top of the low-level ``liger_cute_kernels.tvm_ffi`` primitives
(``team_world``, ``team_split_strided``, ``uniqueid_*``, ...). Unlike the core
``.so``, this module depends on ``torch`` / ``torch.distributed``: bootstrapping
NVSHMEM and translating a ``torch.distributed.ProcessGroup`` into an NVSHMEM team
needs distributed collectives (``broadcast``, ``all_gather_object``) and the
group's global-rank mapping, none of which can live in the torch-free core.

Typical use::

    from liger_cute_kernels import nvshmem
    nvshmem.init_from_pg()              # bootstrap NVSHMEM from WORLD
    team = nvshmem.resolve_team(ep_group)  # collective setup; caches the team
    ...                                 # MoE forward only reads the cached team
    nvshmem.finalize()

Ported from LigerCommKernels' ``liger_comm_kernels/__init__.py`` (bootstrap /
team section) and ``moe_ops._resolve_team`` (the per-ProcessGroup cache).
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_cute_kernels import tvm_ffi

__all__ = [
    "init_from_pg",
    "init_pmi",
    "finalize",
    "my_pe",
    "n_pes",
    "team_world",
    "team_split_strided",
    "team_from_pg",
    "team_destroy",
    "team_my_pe",
    "team_n_pes",
    "team_translate_pe",
    "resolve_team",
    "pool_clear_all",
    "pool_clear_buffers",
]

_BOOTSTRAP_PG: Optional["torch.distributed.ProcessGroup"] = None
_BOOTSTRAP_GLOBAL_RANKS: tuple[int, ...] | None = None
_PG_TEAM_CACHE: dict[int, int] = {}


def _reset_team_state() -> None:
    global _BOOTSTRAP_PG, _BOOTSTRAP_GLOBAL_RANKS
    _BOOTSTRAP_PG = None
    _BOOTSTRAP_GLOBAL_RANKS = None
    _PG_TEAM_CACHE.clear()


# ── Bootstrap ────────────────────────────────────────────────────────────────


def init_from_pg(pg: Optional["torch.distributed.ProcessGroup"] = None) -> None:
    """Bootstrap NVSHMEM using a torch.distributed process group.

    Rank 0 of ``pg`` generates a unique id; the id is broadcast through ``pg``
    and every rank calls ``nvshmem_init_with_uniqueid``. Afterwards
    ``NVSHMEM_TEAM_WORLD`` mirrors ``pg`` exactly.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialised")
    pg = pg if pg is not None else dist.group.WORLD

    rank = dist.get_rank(pg)
    world = dist.get_world_size(pg)
    global_ranks = tuple(dist.get_global_rank(pg, i) for i in range(world))

    # NVSHMEM reads the unique id from host memory, but NCCL-backed PGs can only
    # broadcast CUDA tensors. Fill on CPU, move to CUDA for the broadcast, then
    # copy back to a contiguous CPU buffer.
    uid_cpu = torch.empty(tvm_ffi.uniqueid_nbytes(), dtype=torch.uint8, device="cpu")
    if rank == 0:
        tvm_ffi.get_uniqueid(uid_cpu.data_ptr())
    uid_cuda = uid_cpu.to("cuda", non_blocking=False)
    dist.broadcast(uid_cuda, src=dist.get_global_rank(pg, 0), group=pg)
    uid_cpu = uid_cuda.cpu().contiguous()
    tvm_ffi.init_with_uniqueid(rank, world, uid_cpu.data_ptr())

    _reset_team_state()
    global _BOOTSTRAP_PG, _BOOTSTRAP_GLOBAL_RANKS
    _BOOTSTRAP_PG = pg
    _BOOTSTRAP_GLOBAL_RANKS = global_ranks


def init_pmi(pg: Optional["torch.distributed.ProcessGroup"] = None) -> None:
    """Bootstrap NVSHMEM under PMI and register an aligned torch rank mapping.

    When torch.distributed is initialized, ``pg`` identifies the torch group
    whose rank order must match NVSHMEM PE numbering. The default WORLD group is
    registered automatically when its size matches the NVSHMEM job. If no
    matching torch group is available, ``pg=None`` remains valid for callers
    that use ``NVSHMEM_TEAM_WORLD`` directly without process-group translation.
    """
    import torch.distributed as dist

    explicit_pg = pg is not None
    if pg is not None and not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialised before passing pg to init_pmi")

    tvm_ffi.init_pmi()
    _reset_team_state()

    if not dist.is_initialized():
        return

    pg = pg if pg is not None else dist.group.WORLD
    pg_size = dist.get_world_size(pg)
    nvshmem_size = tvm_ffi.n_pes()
    if pg_size != nvshmem_size:
        if not explicit_pg:
            return
        tvm_ffi.finalize()
        _reset_team_state()
        raise RuntimeError(
            f"PMI NVSHMEM job has {nvshmem_size} PEs, but the supplied torch process group has {pg_size} ranks"
        )

    rank_matches = dist.get_rank(pg) == tvm_ffi.my_pe()
    gathered_matches = [None] * pg_size
    dist.all_gather_object(gathered_matches, rank_matches, group=pg)
    if not all(gathered_matches):
        if not explicit_pg:
            return
        tvm_ffi.finalize()
        _reset_team_state()
        raise RuntimeError("the supplied torch process-group rank order does not match PMI NVSHMEM PE numbering")

    global _BOOTSTRAP_PG, _BOOTSTRAP_GLOBAL_RANKS
    _BOOTSTRAP_PG = pg
    _BOOTSTRAP_GLOBAL_RANKS = tuple(dist.get_global_rank(pg, i) for i in range(pg_size))


def finalize() -> None:
    try:
        tvm_ffi.finalize()
    finally:
        _reset_team_state()


# ── PE queries / team wrappers ───────────────────────────────────────────────


def my_pe() -> int:
    return tvm_ffi.my_pe()


def n_pes() -> int:
    return tvm_ffi.n_pes()


def team_world() -> int:
    return tvm_ffi.team_world()


def team_split_strided(parent: int, start: int, stride: int, size: int) -> int:
    """Split ``parent`` strided; returns a team handle or -1 on non-member ranks."""
    return tvm_ffi.team_split_strided(parent, start, stride, size)


def team_destroy(team_handle: int) -> None:
    tvm_ffi.team_destroy(team_handle)


def team_my_pe(team_handle: int) -> int:
    """NVSHMEM team-local PE for the calling rank."""
    return tvm_ffi.team_my_pe(team_handle)


def team_n_pes(team_handle: int) -> int:
    """Total PE count of the given team."""
    return tvm_ffi.team_n_pes(team_handle)


def team_translate_pe(src_team: int, src_pe: int, dst_team: int) -> int:
    """Translate ``src_pe`` from ``src_team``'s numbering to ``dst_team``'s.

    Returns -1 if the PE isn't a member of ``dst_team``.
    """
    return tvm_ffi.team_translate_pe(src_team, src_pe, dst_team)


# ── Memory pools ─────────────────────────────────────────────────────────────


def pool_clear_all() -> None:
    """Drain both the symmetric stack and the buffer pool (no NVSHMEM finalize)."""
    tvm_ffi.pool_clear_all()


def pool_clear_buffers() -> None:
    """Drain only the per-name buffer pool; keep the symmetric stack intact."""
    tvm_ffi.pool_clear_buffers()


# ── Process group → NVSHMEM team translation ─────────────────────────────────


def _bootstrap_context() -> tuple["torch.distributed.ProcessGroup", tuple[int, ...]]:
    """Return the torch group whose rank order defines NVSHMEM_TEAM_WORLD."""
    if _BOOTSTRAP_PG is not None and _BOOTSTRAP_GLOBAL_RANKS is not None:
        return _BOOTSTRAP_PG, _BOOTSTRAP_GLOBAL_RANKS

    raise RuntimeError(
        "cannot map torch process groups to NVSHMEM PEs: initialize NVSHMEM "
        "with init_from_pg(), or pass an aligned torch group to init_pmi()"
    )


def _pg_parent_pes(pg: "torch.distributed.ProcessGroup") -> tuple[int, ...]:
    """Map a torch process group's rank order into NVSHMEM parent-team PEs."""
    import torch.distributed as dist

    _, bootstrap_global_ranks = _bootstrap_context()
    global_to_parent = {global_rank: pe for pe, global_rank in enumerate(bootstrap_global_ranks)}
    group_global_ranks = tuple(dist.get_global_rank(pg, i) for i in range(dist.get_world_size(pg)))
    try:
        return tuple(global_to_parent[global_rank] for global_rank in group_global_ranks)
    except KeyError as exc:
        raise ValueError(
            f"process-group rank {exc.args[0]} is outside the NVSHMEM bootstrap group {bootstrap_global_ranks}"
        ) from exc


def _ranks_to_strided(ranks):
    """Return (start, stride, size) for an arithmetic rank list, or raise."""
    if len(ranks) == 0:
        raise ValueError("empty rank list")
    if len(ranks) == 1:
        return ranks[0], 1, 1
    stride = ranks[1] - ranks[0]
    if stride <= 0:
        raise ValueError(f"non-positive stride in {ranks}")
    for a, b in zip(ranks, ranks[1:]):
        if b - a != stride:
            raise ValueError(
                f"process-group parent PEs {list(ranks)} are not strided "
                f"(diffs vary); nvshmem team build requires arithmetic stride"
            )
    return ranks[0], stride, len(ranks)


def team_from_pg(pg: "torch.distributed.ProcessGroup") -> int:
    """Build an NVSHMEM team that mirrors a torch process group.

    Collective across the process group used by :func:`init_from_pg` — every
    NVSHMEM PE must call this with its own local ``pg``. The union of all
    callers' ``pg`` memberships must partition that bootstrap group.

    The group's ranks, after translation into bootstrap-group PE numbering, must
    form an arithmetic sequence. Returns the NVSHMEM team handle for the caller's
    subgroup.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialised")

    bootstrap_pg, bootstrap_global_ranks = _bootstrap_context()
    bootstrap_size = len(bootstrap_global_ranks)
    my_group_parent_pes = _pg_parent_pes(pg)
    my_parent_pe = dist.get_rank(bootstrap_pg)

    # Short-circuit: if the group already covers the bootstrap group, skip the
    # (collective) split — just return NVSHMEM_TEAM_WORLD. Avoids creating a team
    # that has to be destroyed before finalize.
    if my_group_parent_pes == tuple(range(bootstrap_size)):
        return tvm_ffi.team_world()

    gathered = [None] * bootstrap_size
    dist.all_gather_object(gathered, my_group_parent_pes, group=bootstrap_pg)

    unique_groups = sorted({tuple(g) for g in gathered})
    for caller_pe, group in enumerate(gathered):
        if caller_pe not in group or any(tuple(gathered[member]) != tuple(group) for member in group):
            raise ValueError(
                "callers' process groups must form a consistent partition of "
                f"the NVSHMEM bootstrap team; gathered parent-PE groups: {gathered}"
            )

    my_handle = -1
    for group in unique_groups:
        start, stride, size = _ranks_to_strided(group)
        # Every rank issues every split in the same order (the split is collective
        # across the parent team); each rank keeps the handle for the group it is
        # a member of.
        handle = tvm_ffi.team_split_strided(tvm_ffi.team_world(), start, stride, size)
        if my_parent_pe in group:
            my_handle = handle

    if my_handle == -1:
        raise RuntimeError(
            f"team_from_pg: parent PE {my_parent_pe} did not land in any group (gathered groups: {unique_groups})"
        )
    return my_handle


def resolve_team(pg: Optional["torch.distributed.ProcessGroup"], *, create: bool = True) -> int:
    """Return the NVSHMEM team handle for ``pg`` (cached per ProcessGroup).

    ``pg=None`` maps to ``NVSHMEM_TEAM_WORLD``. On a single-PE NVSHMEM job that
    means no remote work; on a multi-PE job the kernel routes across every PE in
    WORLD — pass an explicit ``pg`` to scope the routing.

    Creating a non-WORLD team is collective across the NVSHMEM bootstrap group.
    Call this function with its default ``create=True`` during distributed setup
    on every bootstrap PE. Runtime paths such as MoE forward use ``create=False``
    and fail rather than starting an unexpected collective.
    """
    if pg is None:
        return tvm_ffi.team_world()

    parent_pes = _pg_parent_pes(pg)
    if parent_pes == tuple(range(len(_bootstrap_context()[1]))):
        return tvm_ffi.team_world()

    key = id(pg)
    handle = _PG_TEAM_CACHE.get(key)
    if handle is None:
        if not create:
            raise RuntimeError(
                "NVSHMEM team is not initialized for this process group; call "
                "liger_cute_kernels.nvshmem.resolve_team(pg) collectively during "
                "distributed setup before invoking the MoE forward path"
            )
        handle = team_from_pg(pg)
        _PG_TEAM_CACHE[key] = handle
    return handle
