"""Host-side contract tests; native numerical coverage lives in test/cute."""

from __future__ import annotations

import importlib.util

from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from liger_cute_kernels import FusedLinearCrossEntropyConfig
from liger_cute_kernels import MoEConfig
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
    monkeypatch.setattr(
        tvm_ffi, "fused_linear_scaled_cross_entropy_configure_backward", lambda *args: calls.append(("backward", args))
    )
    monkeypatch.setattr(
        tvm_ffi, "fused_linear_scaled_cross_entropy_configure_forward", lambda *args: calls.append(("forward", args))
    )
    monkeypatch.setattr(tvm_ffi, "moe_configure_symmetric", lambda *args: calls.append(("moe", args)))
    yield SimpleNamespace(
        world=world,
        members=members,
        calls=calls,
        configure=partial(configure, bootstrap_group=world, device="cuda:0"),
    )
    nvshmem._reset_team_state()


def test_shared_runtime_configures_both_operators_once(runtime):
    runtime.configure(flsce=FLSCE, moe=MOE)
    runtime.configure(flsce=FLSCE, moe=MOE)
    assert runtime.calls == [
        ("init", runtime.world),
        ("backward", (2048, 512, 1024, 1, 19)),
        ("forward", (2048, 1024)),
        ("moe", (2048, 512, 8, 2, 2, 1, 2)),
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
    assert [name for name, _ in runtime.calls].count("backward") == 1


def test_runtime_only_then_operator_setup(runtime):
    runtime.configure()
    runtime.configure(flsce=FLSCE)
    assert [name for name, _ in runtime.calls] == ["init", "backward", "forward"]


def test_actor_and_reference_can_reuse_larger_capacity(runtime):
    runtime.configure(flsce=FLSCE)
    before = list(runtime.calls)
    runtime.configure(flsce=replace(FLSCE, max_tokens=512, local_vocab_size=512))
    assert runtime.calls == before
    assert configuration._configuration.flsce[1] == (2048, 512, 1024, 1)


def test_hidden_capacity_includes_native_padding(runtime):
    runtime.configure(flsce=replace(FLSCE, hidden_size=513))
    assert runtime.calls[1] == ("backward", (2048, 520, 1024, 1, 19))


@pytest.mark.parametrize("section", ["flsce", "moe"])
def test_capacity_growth_rejected_without_touching_buffers(runtime, section):
    config = FLSCE if section == "flsce" else MOE
    runtime.configure(**{section: config})
    before = list(runtime.calls)
    with pytest.raises(RuntimeError, match="capacity growth"):
        runtime.configure(**{section: replace(config, max_tokens=4096)})
    assert runtime.calls == before


def test_group_replacement_is_explicitly_unsupported(runtime):
    runtime.configure(flsce=FLSCE)
    subgroup = object()
    runtime.members[subgroup] = (0,)
    with pytest.raises(RuntimeError, match="process-group replacement"):
        runtime.configure(flsce=replace(FLSCE, tp_group=subgroup))


def test_distinct_tp_and_ep_partitions_share_bootstrap(runtime, monkeypatch):
    tp, ep = object(), object()
    runtime.members.update({runtime.world: (0, 1, 2, 3), tp: (0, 1), ep: (0, 2)})
    teams = []
    monkeypatch.setattr(nvshmem, "team_from_pg", lambda group: teams.append(group) or len(teams))

    def gather(output, value, group):
        error, flsce, moe, hardware, state = value
        output[:] = [
            (error, ((rank // 2 * 2, rank // 2 * 2 + 1), flsce[1]), ((rank % 2, rank % 2 + 2), moe[1]), hardware, state)
            for rank in range(4)
        ]

    monkeypatch.setattr(configuration.dist, "all_gather_object", gather)
    runtime.configure(flsce=replace(FLSCE, tp_group=tp), moe=replace(MOE, ep_group=ep))
    assert runtime.calls[0] == ("init", runtime.world)
    assert teams == [tp, ep]
    assert runtime.calls[-1] == ("moe", (2048, 512, 8, 2, 2, 1, 2))


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
    [(2049, 512, 8, 2, 19), (512, 256, 8, 2, 19), (512, 512, 4, 2, 19), (512, 512, 8, 3, 19), (512, 512, 8, 2, 20)],
)
def test_moe_launch_rejects_invalid_request_before_native_call(
    runtime, monkeypatch, tokens, hidden, experts, top_k, team
):
    runtime.configure(moe=MOE)
    monkeypatch.setattr(
        tvm_ffi, "_moe_symm_config", lambda: pytest.fail("must reject before querying native workspace")
    )
    x = SimpleNamespace(shape=(tokens, hidden), device=torch.device("cuda:0"))
    with pytest.raises(ValueError):
        tvm_ffi.moe_fused_fwd_bf16(x, None, None, None, None, None, experts, top_k, team)


def test_moe_launch_within_capacity(runtime):
    runtime.configure(moe=MOE)
    configuration._validate_moe_call(512, 512, 8, 1, 19, torch.device("cuda:0"))


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

    monkeypatch.setattr(tvm_ffi, "fused_linear_scaled_cross_entropy_configure_forward", fail)
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
    assert [name for name, _ in runtime.calls].count("backward") == 2


@pytest.fixture
def frontend():
    path = Path(__file__).resolve().parents[2] / "src/liger_kernel/ops/fused_linear_scaled_cross_entropy.py"
    spec = importlib.util.spec_from_file_location("lck_config_frontend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_flsce_configure_delegates_to_lck(runtime, frontend, monkeypatch):
    monkeypatch.setattr(frontend, "_load_native_tp_function", lambda: object())
    assert frontend.LigerFusedLinearScaledCrossEntropyTPFunction.configure(
        max_tokens=FLSCE.max_tokens,
        hidden_size=FLSCE.hidden_size,
        local_vocab_size=FLSCE.local_vocab_size,
        tp_group=runtime.world,
        bootstrap_group=runtime.world,
        device="cuda:0",
    )
    assert runtime.calls[1] == ("backward", (2048, 512, 1024, 1, 19))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_public_flsce_fallback_does_not_initialize(runtime, frontend, monkeypatch, dtype):
    monkeypatch.setattr(frontend, "_load_native_tp_function", lambda: pytest.fail("fallback must not load LCK"))
    assert not frontend.LigerFusedLinearScaledCrossEntropyTPFunction.configure(
        max_tokens=2048, hidden_size=512, local_vocab_size=1024, device="cuda:0", dtype=dtype
    )
    assert runtime.calls == []


def test_public_flsce_without_native_package_does_not_initialize(runtime, frontend, monkeypatch):
    monkeypatch.setattr(frontend, "_load_native_tp_function", lambda: None)
    assert not frontend.LigerFusedLinearScaledCrossEntropyTPFunction.configure(
        max_tokens=2048, hidden_size=512, local_vocab_size=1024, device="cuda:0"
    )
    assert runtime.calls == []
