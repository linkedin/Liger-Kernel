from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

pytest.importorskip("cuda.tile")

from liger_kernel.ops.cutile.ops.megatron_fused_linear_cross_entropy import (  # noqa: E402
    liger_megatron_fused_linear_cross_entropy,
)


def _reference_loss(hidden, weight, target, bias=None, ignore_index=-100):
    logits = hidden.float() @ weight.float().t()
    if bias is not None:
        logits = logits + bias.float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(target.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CuTile FLCE requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cutile_megatron_flce_tp1_matches_pytorch(dtype, with_bias):
    torch.manual_seed(42)
    hidden_base = torch.randn(3, 2, 65, device="cuda", dtype=dtype)
    weight_base = torch.randn(33, 65, device="cuda", dtype=dtype) * 0.02
    bias_base = torch.randn(33, device="cuda", dtype=dtype) * 0.02 if with_bias else None
    target = torch.randint(0, 33, (3, 2), device="cuda")
    target[0, 0] = -100
    upstream = torch.randn(3, 2, device="cuda")

    hidden_ref = hidden_base.clone().requires_grad_(True)
    weight_ref = weight_base.clone().requires_grad_(True)
    bias_ref = bias_base.clone().requires_grad_(True) if bias_base is not None else None
    hidden_cutile = hidden_base.clone().requires_grad_(True)
    weight_cutile = weight_base.clone().requires_grad_(True)
    bias_cutile = bias_base.clone().requires_grad_(True) if bias_base is not None else None

    reference = _reference_loss(hidden_ref, weight_ref, target, bias_ref)
    actual = liger_megatron_fused_linear_cross_entropy(
        hidden_cutile,
        weight_cutile,
        target,
        bias=bias_cutile,
    )

    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)
    reference.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(hidden_cutile.grad, hidden_ref.grad, atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(weight_cutile.grad, weight_ref.grad, atol=5e-3, rtol=5e-2)
    if with_bias:
        torch.testing.assert_close(bias_cutile.grad, bias_ref.grad, atol=5e-3, rtol=5e-2)


def test_cutile_megatron_flce_backend_dispatch():
    env = os.environ.copy()
    env["LIGER_KERNEL_IMPL"] = "cutile"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from liger_kernel.megatron.fused_linear_cross_entropy import "
                "liger_megatron_fused_linear_cross_entropy as fn; print(fn.__module__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "liger_kernel.ops.cutile.ops.megatron_fused_linear_cross_entropy"


def _tp_worker(rank, world_size, file_name, dtype):
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    tp_group = dist.group.WORLD
    vocab_global = 64
    vocab_local = vocab_global // world_size

    torch.manual_seed(123)
    hidden_base = torch.randn(3, 2, 65, device=device, dtype=dtype)
    weight_global = torch.randn(vocab_global, 65, device=device, dtype=dtype) * 0.02
    bias_global = torch.randn(vocab_global, device=device, dtype=dtype) * 0.02
    target = torch.randint(0, vocab_global, (3, 2), device=device)
    upstream = torch.randn(3, 2, device=device)
    target[0, 0] = -100
    for tensor in (hidden_base, weight_global, bias_global, target, upstream):
        dist.broadcast(tensor, src=0, group=tp_group)

    start = rank * vocab_local
    end = start + vocab_local
    hidden_cutile = hidden_base.clone().requires_grad_(True)
    weight_local = weight_global[start:end].clone().requires_grad_(True)
    bias_local = bias_global[start:end].clone().requires_grad_(True)

    actual = liger_megatron_fused_linear_cross_entropy(
        hidden_cutile,
        weight_local,
        target,
        bias=bias_local,
        tp_group=tp_group,
    )

    hidden_ref = hidden_base.clone().requires_grad_(True)
    weight_ref = weight_global.clone().requires_grad_(True)
    bias_ref = bias_global.clone().requires_grad_(True)
    reference = _reference_loss(hidden_ref, weight_ref, target, bias_ref)

    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-2)
    actual.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(hidden_cutile.grad, hidden_ref.grad, atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(weight_local.grad, weight_ref.grad[start:end], atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(bias_local.grad, bias_ref.grad[start:end], atol=5e-3, rtol=5e-2)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA GPUs",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cutile_megatron_flce_tp2_matches_global_reference(dtype):
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(
            _tp_worker,
            args=(2, rendezvous.name, dtype),
            nprocs=2,
            join=True,
        )
