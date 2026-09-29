"""Collective setup for the shared LCK runtime and operator workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field

import torch
import torch.distributed as dist

_OperatorSpec = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class FusedLinearCrossEntropyConfig:
    """Maximum local FLSCE problem; ``group`` names a registered process group.

    ``group=None`` uses the bootstrap group.
    """

    max_tokens: int
    hidden_size: int
    local_vocab_size: int
    group: str | None = None
    tiles_per_reduce: int = 1


@dataclass(frozen=True)
class MoEConfig:
    """MoE capacity and EP-team topology, not the bootstrap-world topology.

    ``num_experts`` is the total expert count in the EP group. ``hidden_size``
    and ``num_experts`` must match subsequent launches. Team ranks must be
    host-major, with ``gpus_per_host`` consecutive members on each host.
    ``group`` names a registered process group; ``None`` uses the bootstrap group.
    """

    max_tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    num_hosts: int
    gpus_per_host: int
    group: str | None = None


@dataclass
class _Configuration:
    bootstrap_ranks: tuple[int, ...]
    device: torch.device
    process_groups: dict[str, dist.ProcessGroup] = field(default_factory=dict)
    flsce: _OperatorSpec | None = None
    moe: _OperatorSpec | None = None
    flsce_team: int | None = None
    moe_team: int | None = None
    preparing_flsce: bool = False


_configuration: _Configuration | None = None


def _reset_configuration() -> None:
    global _configuration
    _configuration = None


def _clear_capacities() -> None:
    if _configuration is not None:
        _configuration.flsce = None
        _configuration.moe = None
        _configuration.flsce_team = None
        _configuration.moe_team = None


def _ranks(group) -> tuple[int, ...]:
    size = dist.get_world_size(group)
    if size < 1 or dist.get_rank(group) < 0:
        raise ValueError("the calling rank must belong to each configured process group")
    return tuple(dist.get_global_rank(group, i) for i in range(size))


def _positive(**values) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2**31 - 1:
            raise ValueError(f"{name} must be a positive int32")


def _resolve_group(name, process_groups, bootstrap_group):
    if name is None:
        return bootstrap_group
    if not isinstance(name, str) or name not in process_groups:
        raise ValueError(f"unknown process group {name!r}; register it in process_groups")
    return process_groups[name]


def _describe_process_groups(process_groups):
    groups = dict(_configuration.process_groups) if _configuration is not None else {}
    if process_groups is not None:
        if not isinstance(process_groups, Mapping):
            raise TypeError("process_groups must be a mapping from names to process groups")
        for name, group in process_groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("process-group names must be non-empty strings")
            if group is None:
                raise ValueError(f"process_groups[{name!r}] must not be None")
            if name in groups and _ranks(groups[name]) != _ranks(group):
                raise ValueError(f"registered process group {name!r} cannot change membership")
            groups[name] = group
    return groups, tuple((name, _ranks(groups[name])) for name in sorted(groups))


def _describe_flsce(config, process_groups, bootstrap_group):
    if config is None:
        return None
    if not isinstance(config, FusedLinearCrossEntropyConfig):
        raise TypeError("flsce must be a FusedLinearCrossEntropyConfig")
    _positive(
        max_tokens=config.max_tokens,
        hidden_size=config.hidden_size,
        local_vocab_size=config.local_vocab_size,
        tiles_per_reduce=config.tiles_per_reduce,
    )
    if config.tiles_per_reduce not in (1, 2, 4):
        raise ValueError("tiles_per_reduce must be one of 1, 2, or 4")
    hidden = (config.hidden_size + 7) // 8 * 8
    _positive(padded_hidden_size=hidden)
    group = _resolve_group(config.group, process_groups, bootstrap_group)
    return _ranks(group), (config.max_tokens, hidden, config.local_vocab_size, config.tiles_per_reduce)


def _describe_moe(config, process_groups, bootstrap_group):
    if config is None:
        return None
    if not isinstance(config, MoEConfig):
        raise TypeError("moe must be a MoEConfig")
    _positive(
        max_tokens=config.max_tokens,
        hidden_size=config.hidden_size,
        num_experts=config.num_experts,
        top_k=config.top_k,
        num_hosts=config.num_hosts,
        gpus_per_host=config.gpus_per_host,
    )
    group = _resolve_group(config.group, process_groups, bootstrap_group)
    ranks = _ranks(group)
    if config.num_hosts * config.gpus_per_host != len(ranks):
        raise ValueError("MoE num_hosts * gpus_per_host must equal the EP group size")
    if config.num_experts % len(ranks) or config.top_k > config.num_experts:
        raise ValueError("MoE num_experts must be divisible by EP size and cover top_k")
    # The native slot count is int32 and includes expert tile padding.
    _positive(max_total_slots=config.max_tokens * config.top_k + config.num_experts * 128)
    return ranks, (
        config.max_tokens,
        config.hidden_size,
        config.num_experts,
        config.top_k,
        config.num_hosts,
        config.gpus_per_host,
    )


def _check_existing(name, requested, existing) -> None:
    if requested is None or existing is None:
        return
    if requested[0] != existing[0]:
        raise ValueError(f"{name} process-group replacement is unsupported; configure one partition per operator")
    if any(new > old for new, old in zip(requested[1], existing[1])):
        raise ValueError(f"{name} capacity growth is unsupported; configure the maximum across all consumers upfront")
    if name == "moe" and any(requested[1][i] != existing[1][i] for i in (1, 2, 4, 5)):
        raise ValueError("MoE hidden size, expert count, and topology must remain unchanged")


def _check_group_partition(groups, bootstrap_ranks) -> None:
    from .nvshmem import _ranks_to_strided

    parent_pe = {rank: i for i, rank in enumerate(bootstrap_ranks)}
    for rank, ranks in zip(bootstrap_ranks, groups):
        if rank not in ranks or any(member not in parent_pe for member in ranks):
            raise ValueError("process groups must partition the bootstrap group")
        if len(ranks) != len(groups[0]):
            raise ValueError("process groups must have uniform size across the bootstrap group")
        _ranks_to_strided(tuple(parent_pe[member] for member in ranks))
        if any(groups[parent_pe[member]] != ranks for member in ranks):
            raise ValueError("process groups must form a consistent partition of the bootstrap group")


def _check_partition(plans, index, bootstrap_ranks) -> None:
    sections = [plan[index] for plan in plans]
    if all(section is None for section in sections):
        return
    if any(section is None for section in sections):
        raise ValueError("every bootstrap rank must configure the same operator sections")
    if any(section[1] != sections[0][1] for section in sections):
        raise ValueError("operator capacities and topology must agree across the bootstrap group")
    _check_group_partition([section[0] for section in sections], bootstrap_ranks)


def configure(
    *,
    process_groups: Mapping[str, dist.ProcessGroup] | None = None,
    bootstrap_group: dist.ProcessGroup | None = None,
    device: torch.device | str | None = None,
    flsce: FusedLinearCrossEntropyConfig | None = None,
    moe: MoEConfig | None = None,
) -> None:
    """Configure the shared NVSHMEM runtime and optional FLSCE/MoE resources.

    Collective on ``bootstrap_group`` (default: torch WORLD), including on
    repeated calls. All ranks must supply the same sections and capacities;
    their local TP/EP groups must form uniform strided partitions of that world.
    Select the correct CUDA device before calling, as required by NCCL.

    ``process_groups`` registers named groups, for example
    ``{"tp": pg_collection.tp, "ep": pg_collection.ep}``. Operator sections
    select a name with ``group="tp"`` / ``group="ep"``. All declared groups
    are prepared in sorted-name order, including groups not yet used by an
    operator. Names must agree on all bootstrap ranks. Additional names can
    be registered later; existing names cannot change membership. Omitted
    names remain registered until finalization.

    A section can be added without clearing another operator's resources.
    Existing capacities may be reused by smaller requests, but cannot grow.
    Each process supports one CUDA device and one partition per operator.
    Configure at a coordinated setup boundary, never during graph capture.

    This API owns initialization, not third-party NVSHMEM interoperability.
    An already initialized unmanaged runtime (including DeepEP V1 RDMA) is
    rejected. Do not initialize another NVSHMEM owner afterwards. Environment
    settings must be established before the first call. Resources remain
    resident across model offload and inference-engine sleep/wake. Only call
    ``nvshmem.finalize()`` after all operators, backward passes, and graphs have
    finished; configuration never finalizes or clears another operator's state.
    """
    from . import nvshmem
    from . import tvm_ffi

    global _configuration
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before configure")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("LCK configure requires an NVIDIA CUDA device")
    bootstrap_group = bootstrap_group if bootstrap_group is not None else dist.group.WORLD
    bootstrap_ranks = _ranks(bootstrap_group)
    device = torch.device(device if device is not None else "cuda")
    if device.type != "cuda":
        raise ValueError("LCK configure requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("configure must run before CUDA graph capture")
        error = None
        flsce_spec = moe_spec = None
        groups = {}
        group_specs = ()
        hardware = None
        try:
            hardware = (
                torch.cuda.get_device_capability(),
                torch.cuda.get_device_properties(device).multi_processor_count,
            )
            if hardware[0][0] not in (9, 10):
                raise ValueError("LCK configure supports Hopper and Blackwell devices")
            groups, group_specs = _describe_process_groups(process_groups)
            flsce_spec = _describe_flsce(flsce, groups, bootstrap_group)
            moe_spec = _describe_moe(moe, groups, bootstrap_group)
            if _configuration is None:
                if tvm_ffi.nvshmem_is_initialized():
                    raise RuntimeError(
                        "NVSHMEM is already initialized outside LCK configure; sharing with an unmanaged "
                        "runtime such as DeepEP V1 RDMA is unsupported"
                    )
                tvm_ffi._load_module()
            else:
                if _configuration.bootstrap_ranks != bootstrap_ranks or _configuration.device != device:
                    raise ValueError("LCK bootstrap group and device cannot change after configuration")
                _check_existing("flsce", flsce_spec, _configuration.flsce)
                _check_existing("moe", moe_spec, _configuration.moe)
        except (TypeError, ValueError, RuntimeError, ImportError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        state = (
            _configuration is not None,
            _configuration is not None and _configuration.flsce is not None,
            _configuration is not None and _configuration.moe is not None,
            tuple(sorted(_configuration.process_groups)) if _configuration is not None else (),
        )
        plans = [None] * len(bootstrap_ranks)
        dist.all_gather_object(
            plans, (error, flsce_spec, moe_spec, hardware, state, group_specs), group=bootstrap_group
        )
        errors = [f"rank {rank}: {plan[0]}" for rank, plan in zip(bootstrap_ranks, plans) if plan[0] is not None]
        if errors:
            raise RuntimeError("LCK configure rejected: " + "; ".join(errors))
        if any(plan[3:5] != plans[0][3:5] for plan in plans):
            raise RuntimeError(
                "LCK configure requires matching hardware and configuration state on every bootstrap rank"
            )
        _check_partition(plans, 1, bootstrap_ranks)
        _check_partition(plans, 2, bootstrap_ranks)
        names = tuple(name for name, _ in group_specs)
        if any(tuple(name for name, _ in plan[5]) != names for plan in plans):
            raise ValueError("process-group names must agree across the bootstrap group")
        for index in range(len(names)):
            _check_group_partition([plan[5][index][1] for plan in plans], bootstrap_ranks)

        torch.cuda.synchronize(device)
        if _configuration is None:
            nvshmem.init_from_pg(bootstrap_group)
            _configuration = _Configuration(bootstrap_ranks, device)
        for name in names:
            if name not in _configuration.process_groups:
                nvshmem.team_from_pg(groups[name])
            _configuration.process_groups[name] = groups[name]
        if flsce_spec is not None:
            group = _resolve_group(flsce.group, groups, bootstrap_group)
            if _configuration.flsce is None:
                team = nvshmem.resolve_team(group, create=False)
                _configuration.preparing_flsce = True
                try:
                    tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(*flsce_spec[1], team)
                    tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(flsce_spec[1][0], flsce_spec[1][2])
                finally:
                    _configuration.preparing_flsce = False
                _configuration.flsce = flsce_spec
                _configuration.flsce_team = team
        if moe_spec is not None:
            group = _resolve_group(moe.group, groups, bootstrap_group)
            if _configuration.moe is None:
                team = nvshmem.resolve_team(group, create=False)
                tokens, hidden, experts, top_k, hosts, local_pes = moe_spec[1]
                tvm_ffi.moe_configure_symmetric(tokens, hidden, experts, top_k, len(moe_spec[0]), hosts, local_pes)
                _configuration.moe = moe_spec
                _configuration.moe_team = team


def _validate_flsce_call(tokens, hidden, vocab, tiles, team) -> None:
    if _configuration is None:
        return  # Preserve the existing low-level setup API.
    if torch.cuda.current_device() != _configuration.device.index:
        raise ValueError("FLSCE input device differs from the configured CUDA device")
    if _configuration.flsce is None:
        if not _configuration.preparing_flsce:
            raise RuntimeError("call configure(flsce=FusedLinearCrossEntropyConfig(...)) before launching FLSCE")
        return
    if any(actual > maximum for actual, maximum in zip((tokens, hidden, vocab, tiles), _configuration.flsce[1])):
        raise ValueError("FLSCE input exceeds configured capacity; configure the maximum upfront")
    if team != _configuration.flsce_team:
        raise ValueError("FLSCE process group differs from the configured TP group")


def _validate_moe_call(tokens, hidden, experts, top_k, team, device) -> None:
    if _configuration is None:
        return  # Preserve the existing low-level setup API.
    if _configuration.moe is None:
        raise RuntimeError("call configure(moe=MoEConfig(...)) before launching MoE")
    _, capacity = _configuration.moe
    if device != _configuration.device:
        raise ValueError("MoE input device differs from the configured CUDA device")
    if tokens > capacity[0] or top_k > capacity[3]:
        raise ValueError("MoE input exceeds configured capacity; configure the maximum upfront")
    if hidden != capacity[1] or experts != capacity[2]:
        raise ValueError("MoE hidden size and expert count must match the configured values")
    if team != _configuration.moe_team:
        raise ValueError("MoE process group differs from the configured EP group")
