"""NVSHMEM bootstrap + process-group → team translation (torch-facing layer).

These helpers sit on top of the low-level ``liger_cute_kernels._C`` primitives
(``team_world``, ``team_split_strided``, ``uniqueid_*``, ...). Unlike the core
``.so``, this module depends on ``torch`` / ``torch.distributed``: bootstrapping
NVSHMEM and translating a ``torch.distributed.ProcessGroup`` into an NVSHMEM team
needs distributed collectives (``broadcast``, ``all_gather_object``) and the
group's global-rank mapping, none of which can live in the torch-free core.

Typical use::

    from liger_cute_kernels import nvshmem
    nvshmem.init_from_pg()                  # bootstrap NVSHMEM from WORLD
    team = nvshmem.team_from_pg(ep_group)   # NVSHMEM team mirroring the EP group
    ...                                     # pass `team` into the MoE entry points
    nvshmem.finalize()

Ported from LigerCommKernels' ``liger_comm_kernels/__init__.py`` (bootstrap /
team section) and ``moe_ops._resolve_team`` (the per-ProcessGroup cache).
"""

from __future__ import annotations

from typing import Optional

import torch

from liger_cute_kernels import tvm_ffi as _C

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

    # NVSHMEM reads the unique id from host memory, but NCCL-backed PGs can only
    # broadcast CUDA tensors. Fill on CPU, move to CUDA for the broadcast, then
    # copy back to a contiguous CPU buffer.
    uid_cpu = torch.empty(_C.uniqueid_nbytes(), dtype=torch.uint8, device="cpu")
    if rank == 0:
        _C.get_uniqueid(uid_cpu.data_ptr())
    uid_cuda = uid_cpu.to("cuda", non_blocking=False)
    dist.broadcast(uid_cuda, src=dist.get_global_rank(pg, 0), group=pg)
    uid_cpu = uid_cuda.cpu().contiguous()
    _C.init_with_uniqueid(rank, world, uid_cpu.data_ptr())


def init_pmi() -> None:
    """Bootstrap NVSHMEM under the launcher's PMI bootstrap (srun/mpirun)."""
    _C.init_pmi()


def finalize() -> None:
    _C.finalize()


# ── PE queries / team wrappers ───────────────────────────────────────────────


def my_pe() -> int:
    return _C.my_pe()


def n_pes() -> int:
    return _C.n_pes()


def team_world() -> int:
    return _C.team_world()


def team_split_strided(parent: int, start: int, stride: int, size: int) -> int:
    """Split ``parent`` strided; returns a team handle or -1 on non-member ranks."""
    return _C.team_split_strided(parent, start, stride, size)


def team_destroy(team_handle: int) -> None:
    _C.team_destroy(team_handle)


def team_my_pe(team_handle: int) -> int:
    """NVSHMEM team-local PE for the calling rank."""
    return _C.team_my_pe(team_handle)


def team_n_pes(team_handle: int) -> int:
    """Total PE count of the given team."""
    return _C.team_n_pes(team_handle)


def team_translate_pe(src_team: int, src_pe: int, dst_team: int) -> int:
    """Translate ``src_pe`` from ``src_team``'s numbering to ``dst_team``'s.

    Returns -1 if the PE isn't a member of ``dst_team``.
    """
    return _C.team_translate_pe(src_team, src_pe, dst_team)


# ── Memory pools ─────────────────────────────────────────────────────────────


def pool_clear_all() -> None:
    """Drain both the symmetric stack and the buffer pool (no NVSHMEM finalize)."""
    _C.pool_clear_all()


def pool_clear_buffers() -> None:
    """Drain only the per-name buffer pool; keep the symmetric stack intact."""
    _C.pool_clear_buffers()


# ── Process group → NVSHMEM team translation ─────────────────────────────────


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
                f"process-group ranks {list(ranks)} are not strided "
                f"(diffs vary); nvshmem team build requires arithmetic stride"
            )
    return ranks[0], stride, len(ranks)


def team_from_pg(pg: "torch.distributed.ProcessGroup") -> int:
    """Build an NVSHMEM team that mirrors a torch process group.

    Collective across ``dist.group.WORLD`` — every NVSHMEM PE must call this with
    its own local ``pg``. The union of all callers' ``pg`` memberships must
    partition WORLD.

    The group's global ranks must form an arithmetic sequence (Megatron's
    parallel_state always produces such ranks); otherwise ValueError is raised on
    all ranks. Returns the NVSHMEM team handle for the caller's subgroup.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialised")

    my_group_ranks = tuple(sorted(dist.get_global_rank(pg, i) for i in range(dist.get_world_size(pg))))

    world_size = dist.get_world_size()

    # Short-circuit: if the group already covers all of WORLD, skip the
    # (collective) split — just return NVSHMEM_TEAM_WORLD. Avoids creating a team
    # that has to be destroyed before finalize.
    if len(my_group_ranks) == world_size and tuple(my_group_ranks) == tuple(range(world_size)):
        return _C.team_world()

    gathered = [None] * world_size
    dist.all_gather_object(gathered, my_group_ranks)

    unique_groups = sorted({tuple(g) for g in gathered})

    my_global_rank = dist.get_rank()
    my_handle = -1
    for group in unique_groups:
        start, stride, size = _ranks_to_strided(group)
        # Every rank issues every split in the same order (the split is collective
        # across the parent team); each rank keeps the handle for the group it is
        # a member of.
        handle = _C.team_split_strided(_C.team_world(), start, stride, size)
        if my_global_rank in group:
            my_handle = handle

    if my_handle == -1:
        raise RuntimeError(
            f"team_from_pg: rank {my_global_rank} did not land in any group (gathered groups: {unique_groups})"
        )
    return my_handle


# Cache: id(pg) -> NVSHMEM team handle. team_from_pg is collective and kicks off
# an all_gather_object for non-WORLD groups, so we resolve once per ProcessGroup.
# id-based keying avoids requiring ProcessGroup to be hashable; entries become
# stale only if a pg is destroyed and a new pg happens to land at the same id,
# which is rare in practice.
_PG_TEAM_CACHE: dict[int, int] = {}


def resolve_team(pg: Optional["torch.distributed.ProcessGroup"]) -> int:
    """Return the NVSHMEM team handle for ``pg`` (cached per ProcessGroup).

    ``pg=None`` maps to ``NVSHMEM_TEAM_WORLD``. On a single-PE NVSHMEM job that
    means no remote work; on a multi-PE job the kernel routes across every PE in
    WORLD — pass an explicit ``pg`` to scope the routing.
    """
    if pg is None:
        return _C.team_world()

    key = id(pg)
    handle = _PG_TEAM_CACHE.get(key)
    if handle is None:
        handle = team_from_pg(pg)
        _PG_TEAM_CACHE[key] = handle
    return handle
