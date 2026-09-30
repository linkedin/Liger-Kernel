"""Host-side contract tests; native numerical coverage lives in test/cute."""

from __future__ import annotations

import builtins
import importlib.util
import pkgutil
import sys
import types

from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from liger_cute_kernels import FusedLinearCrossEntropyConfig
from liger_cute_kernels import MoEConfig
from liger_cute_kernels import UnsupportedDeviceError
from liger_cute_kernels import configuration
from liger_cute_kernels import configure
from liger_cute_kernels import nvshmem
from liger_cute_kernels import tvm_ffi

FLSCE = FusedLinearCrossEntropyConfig(2048, 512, 1024)
MOE = MoEConfig(2048, 512, 8, 2, 1, 2)
_CONFIGURE_BACKWARD = tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward


@pytest.fixture
def runtime(monkeypatch):
    nvshmem._reset_team_state()
    world = object()
    members = {world: (0, 1)}
    calls = []
    monkeypatch.setattr(configuration.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(configuration.dist, "get_world_size", lambda group: len(members[group]))
    monkeypatch.setattr(configuration.dist, "get_global_rank", lambda group, rank: members[group][rank])
    monkeypatch.setattr(configuration.dist, "get_rank", lambda group: members[group].index(0))
    monkeypatch.setattr(
        configuration.dist,
        "all_gather_object",
        lambda output, value, group: output.__setitem__(slice(None), [value] * len(members[group])),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(multi_processor_count=132))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(tvm_ffi, "nvshmem_is_initialized", lambda: False)
    monkeypatch.setattr(tvm_ffi, "_load_module", lambda: object())
    monkeypatch.setattr(nvshmem, "init_from_pg", lambda pg: calls.append(("init", pg)))
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda pg: 19)
    monkeypatch.setattr(nvshmem, "resolve_team", lambda pg, *, create: 19)
    monkeypatch.setattr(
        tvm_ffi, "fused_linear_scaled_cross_entropy_configure_context", lambda *args: calls.append(("context", args))
    )
    monkeypatch.setattr(
        tvm_ffi, "fused_linear_scaled_cross_entropy_configure_forward", lambda *args: calls.append(("forward", args))
    )
    monkeypatch.setattr(tvm_ffi, "moe_configure_context", lambda *args: calls.append(("moe", args)))
    yield SimpleNamespace(
        world=world,
        members=members,
        calls=calls,
        configure=partial(configure, bootstrap_group=world, device="cuda:0"),
    )
    nvshmem._reset_team_state()


def test_shared_runtime_configures_both_operators_once(runtime):
    assert runtime.configure(flsce=FLSCE, moe=MOE) is None
    runtime.configure(flsce=FLSCE, moe=MOE)
    assert runtime.calls == [
        ("init", runtime.world),
        ("context", (2048, 512, 1024, 1, 19, 0)),
        ("moe", (2048, 512, 8, 2, 1, 2, 1, 19, 0)),
    ]


@pytest.mark.parametrize("first", ["flsce", "moe"])
def test_add_operator_preserves_existing_reservation(runtime, first):
    sections = {"flsce": FLSCE, "moe": MOE}
    runtime.configure(**{first: sections[first]})
    before = list(runtime.calls)
    runtime.configure(**sections)
    assert runtime.calls[: len(before)] == before
    assert [name for name, _ in runtime.calls].count("init") == 1
    assert [name for name, _ in runtime.calls].count("moe") == 1
    assert [name for name, _ in runtime.calls].count("context") == 1


def test_runtime_only_then_operator_setup(runtime):
    runtime.configure()
    runtime.configure(flsce=FLSCE)
    assert [name for name, _ in runtime.calls] == ["init", "context"]


def test_actor_and_reference_can_reuse_larger_capacity(runtime):
    runtime.configure(flsce=FLSCE)
    before = list(runtime.calls)
    runtime.configure(flsce=replace(FLSCE, max_tokens=512, local_vocab_size=512))
    assert runtime.calls == before
    assert configuration._configuration.flsce[1] == (2048, 512, 1024, 1)


def test_hidden_capacity_includes_native_padding(runtime):
    runtime.configure(flsce=replace(FLSCE, hidden_size=513))
    assert runtime.calls[1] == ("context", (2048, 520, 1024, 1, 19, 0))


@pytest.mark.parametrize("section", ["flsce", "moe"])
def test_capacity_growth_rejected_without_touching_buffers(runtime, section):
    config = FLSCE if section == "flsce" else MOE
    runtime.configure(**{section: config})
    before = list(runtime.calls)
    with pytest.raises(RuntimeError, match="capacity growth"):
        runtime.configure(**{section: replace(config, max_tokens=4096)})
    assert runtime.calls == before


def test_add_tp_partition_preserves_process_wide_capacity(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE)
    subgroup = object()
    runtime.members[subgroup] = (0,)
    monkeypatch.setattr(nvshmem, "resolve_team", lambda group, *, create: 20 if group is subgroup else 19)

    def gather(output, value, group):
        peer = list(value)
        peer[1] = ((1,), value[1][1])
        peer[5] = (("other", (1,)),)
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(process_groups={"other": subgroup}, flsce=replace(FLSCE, max_tokens=512, group="other"))
    assert runtime.calls[-1] == ("context", (2048, 512, 1024, 1, 20, 1))
    assert configuration._configuration.flsce_teams == {19: (0, 1), 20: (0,)}
    configuration._validate_flsce_call(2048, 512, 1024, 1, 19)
    configuration._validate_flsce_call(512, 512, 1024, 1, 20)
    before = list(runtime.calls)
    runtime.configure(flsce=replace(FLSCE, group="other"))
    assert runtime.calls == before


def test_add_ep_partition_preserves_process_wide_capacity(runtime, monkeypatch):
    runtime.configure(moe=replace(MOE, max_inflight=2))
    subgroup = object()
    runtime.members[subgroup] = (0,)
    monkeypatch.setattr(nvshmem, "resolve_team", lambda group, *, create: 20 if group is subgroup else 19)

    def gather(output, value, group):
        peer = list(value)
        peer[2] = ((1,), value[2][1])
        peer[5] = (("other", (1,)),)
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(
        process_groups={"other": subgroup},
        moe=replace(MOE, max_tokens=512, gpus_per_host=1, group="other"),
    )
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 1, 1, 2, 20, 1))
    assert set(configuration._configuration.moe_teams) == {19, 20}
    configuration._validate_moe_call(2048, 512, 8, 2, 19, torch.device("cuda:0"))
    configuration._validate_moe_call(512, 256, 4, 1, 20, torch.device("cuda:0"))
    before = list(runtime.calls)
    runtime.configure(moe=replace(MOE, group="other", gpus_per_host=1))
    assert runtime.calls == before


def test_partial_ep_cache_hits_reserve_collective_slot(runtime, monkeypatch):
    subgroup = object()
    runtime.members.update({runtime.world: tuple(range(6)), subgroup: (0, 1)})
    partitions = [(0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5)]
    hits = [False] * 6

    def gather(output, value, group):
        output[:] = []
        for rank in range(6):
            peer = list(value)
            peer[2] = (partitions[rank], value[2][1])
            peer[5] = (("ep", partitions[rank]),)
            peer[7] = hits[rank]
            output.append(tuple(peer))

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(process_groups={"ep": subgroup}, moe=replace(MOE, group="ep"))
    partitions[:] = [(0, 1), (0, 1), (2, 4), (3, 5), (2, 4), (3, 5)]
    hits[:] = [True, True, False, False, False, False]
    runtime.configure(moe=replace(MOE, max_tokens=512, group="ep"))
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 1, 2, 1, 19, 1))
    assert configuration._configuration.moe_context_slot == 2
    hits[:] = [True] * 6
    before = list(runtime.calls)
    runtime.configure(moe=replace(MOE, group="ep"))
    assert runtime.calls == before


def test_inconsistent_ep_context_cache_rejected_before_setup(runtime, monkeypatch):
    runtime.configure(moe=MOE)
    before = list(runtime.calls)

    def gather(output, value, group):
        peer = list(value)
        peer[7] = False
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="context cache differs"):
        runtime.configure(moe=MOE)
    assert runtime.calls == before


def test_destroyed_ep_team_requires_context_reconfiguration(runtime, monkeypatch):
    runtime.configure(moe=MOE)
    monkeypatch.setattr(tvm_ffi, "team_destroy", lambda team: None)
    nvshmem.team_destroy(19)
    with pytest.raises(ValueError, match="no configured EP context"):
        configuration._validate_moe_call(512, 512, 8, 2, 19, torch.device("cuda:0"))
    runtime.configure(moe=MOE)
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 1, 2, 1, 19, 1))


def test_distinct_tp_and_ep_partitions_share_bootstrap(runtime, monkeypatch):
    tp, ep = object(), object()
    runtime.members.update({runtime.world: (0, 1, 2, 3), tp: (0, 1), ep: (0, 2)})
    teams = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda group: teams.append(group) or len(teams))

    def gather(output, value, group):
        error, flsce, moe, hardware, state, group_specs, cached, moe_cached = value
        output[:] = [
            (
                error,
                ((rank // 2 * 2, rank // 2 * 2 + 1), flsce[1]),
                ((rank % 2, rank % 2 + 2), moe[1]),
                hardware,
                state,
                (("ep", (rank % 2, rank % 2 + 2)), ("tp", (rank // 2 * 2, rank // 2 * 2 + 1))),
                cached,
                moe_cached,
            )
            for rank in range(4)
        ]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(
        process_groups={"tp": tp, "ep": ep},
        flsce=replace(FLSCE, group="tp"),
        moe=replace(MOE, group="ep"),
    )
    assert runtime.calls[0] == ("init", runtime.world)
    assert teams == [ep, tp]
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 1, 2, 1, 19, 0))


def test_partial_context_cache_hits_still_reserve_collective_slot(runtime, monkeypatch):
    subgroup = object()
    runtime.members.update({runtime.world: tuple(range(6)), subgroup: (0, 1)})
    partitions = [(0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5)]
    peer_hits = [False] * 6

    def gather(output, value, group):
        output[:] = []
        for rank in range(6):
            peer = list(value)
            peer[1] = (partitions[rank], value[1][1])
            peer[5] = (("tp", partitions[rank]),)
            peer[6] = peer_hits[rank]
            output.append(tuple(peer))

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(process_groups={"tp": subgroup}, flsce=replace(FLSCE, group="tp"))
    partitions[:] = [(0, 1), (0, 1), (2, 4), (3, 5), (2, 4), (3, 5)]
    peer_hits[:] = [True, True, False, False, False, False]
    runtime.configure(flsce=replace(FLSCE, max_tokens=512, group="tp"))
    assert runtime.calls[-1] == ("context", (2048, 512, 1024, 1, 19, 1))
    assert configuration._configuration.flsce_context_slot == 2
    peer_hits[:] = [True] * 6
    before = list(runtime.calls)
    runtime.configure(flsce=replace(FLSCE, group="tp"))
    assert runtime.calls == before


def test_inconsistent_tp_context_cache_rejected_before_setup(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE)
    before = list(runtime.calls)

    def gather(output, value, group):
        peer = list(value)
        peer[6] = False
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="context cache differs"):
        runtime.configure(flsce=FLSCE)
    assert runtime.calls == before


def test_destroyed_tp_team_requires_context_reconfiguration(runtime, monkeypatch):
    prepared = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda pg: prepared.append(pg) or 19)
    runtime.configure(process_groups={"tp": runtime.world}, flsce=replace(FLSCE, group="tp"))
    monkeypatch.setattr(tvm_ffi, "team_destroy", lambda team: None)
    nvshmem.team_destroy(19)
    with pytest.raises(ValueError, match="no configured TP context"):
        configuration._validate_flsce_call(512, 512, 1024, 1, 19)
    assert configuration._configuration.flsce[1] == (2048, 512, 1024, 1)
    runtime.configure(flsce=replace(FLSCE, group="tp"))
    assert prepared == [runtime.world, runtime.world]
    assert runtime.calls[-1] == ("context", (2048, 512, 1024, 1, 19, 1))


def test_named_groups_are_prepared_in_sorted_order(runtime, monkeypatch):
    first, second, third = object(), object(), object()
    runtime.members.update({first: (0, 1), second: (0, 1), third: (0, 1)})
    prepared = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda group: prepared.append(group) or 19)
    runtime.configure(process_groups={"tensor": third, "data": first, "expert": second})
    assert prepared == [first, second, third]
    assert runtime.calls == [("init", runtime.world)]


def test_registered_names_persist_when_adding_groups_and_operators(runtime, monkeypatch):
    prepared = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda group: prepared.append(group) or 19)
    runtime.configure(process_groups={"tensor": runtime.world})
    runtime.configure(process_groups={"expert": runtime.world}, flsce=replace(FLSCE, group="tensor"))
    runtime.configure(moe=replace(MOE, group="expert"))
    runtime.configure(process_groups={"tensor": runtime.world}, flsce=replace(FLSCE, group="tensor"))
    assert prepared == [runtime.world] * 7
    assert set(configuration._configuration.process_groups) == {"tensor", "expert"}
    assert [name for name, _ in runtime.calls] == ["init", "context", "moe"]


def test_registered_name_cannot_change_membership(runtime):
    runtime.configure(process_groups={"tp": runtime.world})
    other = object()
    runtime.members[other] = (0,)
    with pytest.raises(RuntimeError, match="cannot change membership"):
        runtime.configure(process_groups={"tp": other})
    assert configuration._configuration.process_groups["tp"] is runtime.world


@pytest.mark.parametrize("groups", [[], {"tp": None}, {"": object()}, {1: object()}])
def test_invalid_process_group_mapping_fails_before_init(runtime, groups):
    with pytest.raises(RuntimeError, match="LCK configure rejected"):
        runtime.configure(process_groups=groups)
    assert runtime.calls == []


@pytest.mark.parametrize("section", ["flsce", "moe"])
def test_unknown_group_name_fails_before_init(runtime, section):
    config = FLSCE if section == "flsce" else MOE
    with pytest.raises(RuntimeError, match="unknown process group"):
        runtime.configure(**{section: replace(config, group="missing")})
    assert runtime.calls == []


def test_named_group_keys_must_match_on_all_ranks(runtime, monkeypatch):
    def gather(output, value, group):
        peer = list(value)
        peer[5] = (("different", (0, 1)),)
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="names must agree"):
        runtime.configure(process_groups={"tp": runtime.world})
    assert runtime.calls == []


def test_unused_named_group_still_requires_a_consistent_partition(runtime, monkeypatch):
    def gather(output, value, group):
        peer = list(value)
        peer[5] = (("unused", (1,)),)
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="consistent partition|uniform size"):
        runtime.configure(process_groups={"unused": runtime.world})
    assert runtime.calls == []


@pytest.mark.parametrize(
    "config",
    [
        replace(FLSCE, max_tokens=0),
        replace(FLSCE, hidden_size=True),
        replace(FLSCE, local_vocab_size=2**31),
        replace(FLSCE, tiles_per_reduce=3),
        replace(FLSCE, tiles_per_reduce=1.0),
    ],
)
def test_invalid_flsce_values_do_not_initialize(runtime, config):
    with pytest.raises(RuntimeError, match="LCK configure rejected"):
        runtime.configure(flsce=config)
    assert runtime.calls == []


@pytest.mark.parametrize(
    "config",
    [
        replace(MOE, num_hosts=2),
        replace(MOE, num_experts=7),
        replace(MOE, top_k=9),
        replace(MOE, max_tokens=2**31 - 1),
        replace(MOE, max_inflight=0),
    ],
)
def test_invalid_moe_values_do_not_initialize(runtime, config):
    with pytest.raises(RuntimeError, match="LCK configure rejected"):
        runtime.configure(moe=config)
    assert runtime.calls == []


@pytest.mark.parametrize("index", [0, 1, 3, 4])
def test_rank_disagreement_fails_before_initialization(runtime, monkeypatch, index):
    def gather(output, value, group):
        other = list(value)
        other[index] = {
            0: "ValueError: bad configuration on peer",
            1: (value[1][0], (4096, 512, 1024, 1)),
            3: ((10, 0), 148),
            4: (True, False, False),
        }[index]
        output[:] = [value, tuple(other)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.configure(flsce=FLSCE)
    assert runtime.calls == []


def test_unmanaged_nvshmem_is_rejected(runtime, monkeypatch):
    monkeypatch.setattr(tvm_ffi, "nvshmem_is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="DeepEP V1"):
        runtime.configure(flsce=FLSCE)
    assert runtime.calls == []


def test_capture_rejected_before_collectives(runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        configuration.dist,
        "all_gather_object",
        lambda *args, **kwargs: pytest.fail("must not communicate during capture"),
    )
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        runtime.configure(flsce=FLSCE)
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("tokens", "hidden", "experts", "top_k", "team"),
    [(2049, 512, 8, 2, 19), (512, 1024, 8, 2, 19), (512, 512, 16, 2, 19), (512, 512, 8, 3, 19), (512, 512, 8, 2, 20)],
)
def test_moe_launch_rejects_invalid_request_before_native_call(
    runtime, monkeypatch, tokens, hidden, experts, top_k, team
):
    runtime.configure(moe=MOE)
    monkeypatch.setattr(
        tvm_ffi, "_moe_symm_config", lambda *args: pytest.fail("must reject before querying native workspace")
    )
    x = SimpleNamespace(shape=(tokens, hidden), device=torch.device("cuda:0"))
    with pytest.raises(ValueError):
        tvm_ffi.moe_fused_fwd_bf16(x, None, None, None, None, None, experts, top_k, team)


def test_moe_launch_within_capacity(runtime):
    runtime.configure(moe=MOE)
    configuration._validate_moe_call(512, 512, 8, 1, 19, torch.device("cuda:0"))


def test_moe_retained_capacity_cannot_grow(runtime):
    runtime.configure(moe=MOE)
    before = list(runtime.calls)
    with pytest.raises(RuntimeError, match="capacity growth"):
        runtime.configure(moe=replace(MOE, max_inflight=2))
    assert runtime.calls == before


@pytest.mark.parametrize(
    ("tokens", "hidden", "vocab", "tiles", "team"),
    [
        (2049, 512, 1024, 1, 19),
        (512, 520, 1024, 1, 19),
        (512, 512, 1025, 1, 19),
        (512, 512, 1024, 2, 19),
        (512, 512, 1024, 1, 20),
    ],
)
def test_flsce_actual_request_is_checked(runtime, tokens, hidden, vocab, tiles, team):
    runtime.configure(flsce=FLSCE)
    with pytest.raises(ValueError):
        configuration._validate_flsce_call(tokens, hidden, vocab, tiles, team)


def test_flsce_launch_on_other_device_is_rejected(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    with pytest.raises(ValueError, match="device differs"):
        configuration._validate_flsce_call(512, 512, 1024, 1, 19)


def test_flsce_tvm_boundary_checks_capacity_before_native_call(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE)
    monkeypatch.setattr(tvm_ffi, "_load_module", lambda: pytest.fail("must reject before native call"))
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        _CONFIGURE_BACKWARD(2049, 512, 1024, 1, 19)


def test_flsce_cannot_allocate_implicitly_in_managed_runtime(runtime):
    runtime.configure()
    with pytest.raises(RuntimeError, match="before launching FLSCE"):
        _CONFIGURE_BACKWARD(512, 512, 1024, 1, 19)


def test_native_setup_failure_leaves_flsce_unconfigured(runtime, monkeypatch):
    def fail(*args):
        raise RuntimeError("native allocation failed")

    monkeypatch.setattr(tvm_ffi, "fused_linear_scaled_cross_entropy_configure_context", fail)
    with pytest.raises(RuntimeError, match="native allocation failed"):
        runtime.configure(flsce=FLSCE)
    assert configuration._configuration.flsce is None
    assert not configuration._configuration.preparing_flsce


def test_finalize_resets_managed_state(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE, moe=MOE)
    monkeypatch.setattr(tvm_ffi, "finalize", lambda: None)
    nvshmem.finalize()
    assert configuration._configuration is None


def test_pool_clear_requires_reconfiguration(runtime, monkeypatch):
    runtime.configure(flsce=FLSCE, moe=MOE)
    monkeypatch.setattr(tvm_ffi, "pool_clear_all", lambda: None)
    nvshmem.pool_clear_all()
    assert configuration._configuration.flsce is None
    with pytest.raises(RuntimeError, match="before launching MoE"):
        configuration._validate_moe_call(512, 512, 8, 2, 19, torch.device("cuda:0"))
    runtime.configure(flsce=FLSCE, moe=MOE)
    assert [name for name, _ in runtime.calls].count("init") == 1
    assert [name for name, _ in runtime.calls].count("context") == 2


@pytest.fixture
def private_frontend(monkeypatch):
    registry = types.ModuleType("liger_kernel.ops.backends.registry")
    registry.ImplInfo = SimpleNamespace
    registry.register_impl = lambda info: None
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    path = Path(__file__).resolve().parents[2] / "src/liger_kernel/ops/cute/__init__.py"
    spec = importlib.util.spec_from_file_location("lck_config_frontend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def public_ops(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "src/liger_kernel/ops/__init__.py"
    spec = importlib.util.spec_from_file_location("lck_public_ops", path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    # Isolate the public package boundary from unrelated Triton operator imports.
    def import_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("liger_cute_kernels"):
            pytest.fail("public ops import must not import optional LCK")
        if name.startswith("liger_kernel.ops.") and fromlist:
            symbols = {symbol: object() for symbol in fromlist}
            if name == "liger_kernel.ops.backends":
                symbols.update(LIGER_KERNEL_IMPL_ENV="LIGER_KERNEL_IMPL", select_impl=lambda *args, **kwargs: None)
            return SimpleNamespace(**symbols)
        if name == "liger_kernel.utils":
            return SimpleNamespace(infer_device=lambda: "cpu")
        return original_import(name, globals, locals, fromlist, level)

    def import_module(name, package=None):
        if name.startswith("liger_cute_kernels"):
            pytest.fail("public ops discovery must not import optional LCK")
        if name == "liger_kernel.ops.backends":
            return SimpleNamespace()
        return original_import_module(name, package)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_dependencies)
        context.setattr(importlib, "import_module", import_module)
        context.setattr(pkgutil, "iter_modules", lambda paths: ((None, "configure", False),))
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def public_configuration(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "src/liger_kernel/ops/configure.py"
    spec = importlib.util.spec_from_file_location("lck_public_configuration", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_public_configuration_data_matches_lck_schema(public_configuration):
    for public_type, native_type in (
        (public_configuration.FusedLinearCrossEntropyConfig, FusedLinearCrossEntropyConfig),
        (public_configuration.MoEConfig, MoEConfig),
    ):
        public_fields = [(field.name, field.default, field.default_factory) for field in fields(public_type)]
        native_fields = [(field.name, field.default, field.default_factory) for field in fields(native_type)]
        assert public_fields == native_fields


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3)])
def test_public_module_configures_both_operators(runtime, public_configuration, monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert (
        public_configuration.configure(
            process_groups={"tp": runtime.world, "ep": runtime.world},
            bootstrap_group=runtime.world,
            device="cuda:0",
            flsce=public_configuration.FusedLinearCrossEntropyConfig(**asdict(replace(FLSCE, group="tp"))),
            moe=public_configuration.MoEConfig(**asdict(replace(MOE, group="ep"))),
        )
        is True
    )
    assert runtime.calls[1] == ("context", (2048, 512, 1024, 1, 19, 0))
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 1, 2, 1, 19, 0))


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 1), (12, 0)])
def test_public_configuration_skips_unsupported_architecture(
    runtime, public_configuration, monkeypatch, capability, recwarn
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(tvm_ffi, "_load_module", lambda: pytest.fail("must not load native code"))
    monkeypatch.setattr(tvm_ffi, "nvshmem_is_initialized", lambda: pytest.fail("must not inspect native runtime"))
    with pytest.raises(UnsupportedDeviceError, match="Hopper and Blackwell"):
        runtime.configure(flsce=FLSCE, moe=MOE)
    assert not recwarn
    with pytest.warns(UserWarning, match="native configuration was skipped"):
        assert (
            public_configuration.configure(
                bootstrap_group=runtime.world,
                device="cuda:0",
                flsce=public_configuration.FusedLinearCrossEntropyConfig(**asdict(FLSCE)),
                moe=public_configuration.MoEConfig(**asdict(MOE)),
            )
            is False
        )
    assert runtime.calls == []
    assert configuration._configuration is None


@pytest.mark.parametrize(
    ("device", "cuda_available", "hip"),
    [("cpu", True, None), ("xpu:0", True, None), ("cuda:0", False, None), ("cuda:0", True, "6.3")],
)
def test_public_configuration_skips_non_nvidia_devices(
    runtime, public_configuration, monkeypatch, device, cuda_available, hip, recwarn
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "device", lambda device: pytest.fail("must not enter a CUDA context"))
    with pytest.raises(UnsupportedDeviceError, match="NVIDIA CUDA device"):
        configure(device=device)
    assert not recwarn
    with pytest.warns(UserWarning, match="native configuration was skipped"):
        assert public_configuration.configure(device=device) is False
    assert runtime.calls == []
    assert configuration._configuration is None


def test_unsupported_architecture_still_rejects_invalid_capacity(runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    with pytest.raises(RuntimeError, match="max_tokens must be a positive int32"):
        runtime.configure(flsce=replace(FLSCE, max_tokens=0))
    assert runtime.calls == []


def test_mixed_architectures_reject_collectively_without_initialization(runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    def gather(output, value, group):
        peer = list(value)
        peer[3] = ((9, 0), 132)
        output[:] = [value, tuple(peer)]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="matching hardware"):
        runtime.configure(flsce=FLSCE, moe=MOE)
    assert runtime.calls == []


def test_public_configuration_preserves_legacy_native_success(public_configuration, monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(configure=lambda **kwargs: None))
    assert public_configuration.configure() is True


def test_backend_discovery_does_not_import_optional_lck(public_ops):
    for name in ("configure", "FusedLinearCrossEntropyConfig", "MoEConfig"):
        assert not hasattr(public_ops, name)
    assert not hasattr(public_ops, "_CONFIGURATION_EXPORTS")


def test_private_implementation_has_no_configuration_exports(private_frontend):
    for name in ("configure", "FusedLinearCrossEntropyConfig", "MoEConfig"):
        assert not hasattr(private_frontend, name)
        assert name not in private_frontend.__all__


def test_configuration_function_is_importable_without_lck(public_configuration, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    original_import = builtins.__import__

    def without_lck(name, *args, **kwargs):
        if name.startswith("liger_cute_kernels"):
            return missing(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    monkeypatch.setattr(builtins, "__import__", without_lck)
    public_configuration.__spec__.loader.exec_module(public_configuration)
    assert callable(public_configuration.configure)
    flsce = public_configuration.FusedLinearCrossEntropyConfig(2048, 512, 1024, group="tp")
    moe = public_configuration.MoEConfig(2048, 512, 8, 2, 1, 2, group="ep")
    with pytest.warns(UserWarning, match="configuration was skipped"):
        assert (
            public_configuration.configure(
                process_groups={"tp": object(), "ep": object()},
                flsce=flsce,
                moe=moe,
            )
            is False
        )
    with pytest.raises(AttributeError, match="unknown"):
        _ = public_configuration.unknown


@pytest.mark.parametrize("dependency", ["torch", "tvm_ffi"])
def test_configuration_preserves_dependency_import_errors(public_configuration, monkeypatch, dependency):
    error = ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)

    def fail(name):
        raise error

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(ModuleNotFoundError) as caught:
        public_configuration.configure()
    assert caught.value is error


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
@pytest.mark.parametrize("typed_error_available", [False, True])
def test_configuration_preserves_native_setup_errors(
    public_configuration, monkeypatch, error_type, typed_error_available
):
    error = error_type("native setup failed")

    def fail(**kwargs):
        raise error

    lck = SimpleNamespace(configure=fail)
    if typed_error_available:
        lck.UnsupportedDeviceError = UnsupportedDeviceError
    monkeypatch.setattr(importlib, "import_module", lambda name: lck)
    with pytest.raises(error_type) as caught:
        public_configuration.configure()
    assert caught.value is error


def test_operator_class_has_no_configure_method():
    path = Path(__file__).resolve().parents[2] / "src/liger_kernel/ops/fused_linear_scaled_cross_entropy.py"
    spec = importlib.util.spec_from_file_location("lck_flsce_frontend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(module.LigerFusedLinearScaledCrossEntropyTPFunction, "configure")
