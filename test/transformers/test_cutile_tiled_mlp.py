"""The cuTile tiled MLP is a second, independent copy of the tiling logic, so it needs the same
gradient-accumulation coverage as the default backend (huggingface/kernels-community#1081)."""

import copy

import pytest
import torch
import torch.nn as nn

try:
    from liger_kernel.ops.cutile.ops.tiled_mlp import apply_tiled_mlp as cutile_apply_tiled_mlp
except Exception:  # the cutile ops package needs cuda-tile at import time
    cutile_apply_tiled_mlp = None

pytestmark = pytest.mark.skipif(
    cutile_apply_tiled_mlp is None,
    reason="cuTile backend requires cuda-tile",
)


class _PlainMLP(nn.Module):
    """A SwiGLU MLP built from stock modules: the cuTile tiled MLP is pure torch, so the tiling can be
    exercised without a cuTile kernel."""

    def __init__(self, in_features=64, hidden_features=128, out_features=64):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return _plain_mlp_forward(self, x)


def _plain_mlp_forward(module, x):
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


def _tiled(mlp, x, num_shards):
    return cutile_apply_tiled_mlp(
        fn=_plain_mlp_forward,
        mlp_module=mlp,
        x=x,
        num_shards=num_shards,
        compute_params=[p for p in mlp.parameters() if p.requires_grad],
    )


@pytest.mark.parametrize("num_shards", [2, 4, 8])
def test_cutile_tiled_mlp_accumulates_each_weight_once(num_shards):
    """DDP marks a parameter ready on every AccumulateGrad firing and rejects a repeat, so accumulating
    once per shard breaks it."""
    mlp = _PlainMLP()
    counts = {name: 0 for name, _ in mlp.named_parameters()}
    for name, param in mlp.named_parameters():
        param.register_post_accumulate_grad_hook(lambda _param, name=name: counts.__setitem__(name, counts[name] + 1))

    x = torch.randn(2, 256, 64, requires_grad=True)
    _tiled(mlp, x, num_shards).sum().backward()

    assert counts == dict.fromkeys(counts, 1)


def test_cutile_tiled_mlp_zero3_accumulates_once_per_shard():
    """ZeRO-3 keeps the per-shard accumulation: its partitioned weights are 0-sized placeholders outside
    the owning module's forward, so autograd would record (0,) as the expected gradient shape."""
    num_shards = 4
    mlp = _PlainMLP()
    for param in mlp.parameters():
        param.ds_id = 0

    counts = {name: 0 for name, _ in mlp.named_parameters()}
    for name, param in mlp.named_parameters():
        param.register_post_accumulate_grad_hook(lambda _param, name=name: counts.__setitem__(name, counts[name] + 1))

    x = torch.randn(2, 256, 64, requires_grad=True)
    _tiled(mlp, x, num_shards).sum().backward()

    assert counts == dict.fromkeys(counts, num_shards)


@pytest.mark.parametrize("num_shards", [2, 4])
def test_cutile_tiled_mlp_matches_untiled_reference(num_shards):
    reference = _PlainMLP().double()
    tiled = copy.deepcopy(reference)

    x = torch.randn(2, 256, 64, dtype=torch.float64)
    x_reference = x.detach().clone().requires_grad_(True)
    x_tiled = x.detach().clone().requires_grad_(True)

    y_reference = reference(x_reference)
    y_tiled = _tiled(tiled, x_tiled, num_shards)
    torch.testing.assert_close(y_reference, y_tiled, msg="Forward outputs don't match")

    dy = torch.randn_like(y_reference)
    y_reference.backward(dy.clone())
    y_tiled.backward(dy.clone())

    for p_reference, p_tiled in zip(reference.parameters(), tiled.parameters()):
        torch.testing.assert_close(p_reference.grad, p_tiled.grad, msg="Weight gradients don't match")
    torch.testing.assert_close(x_reference.grad, x_tiled.grad, msg="Input gradients don't match")
