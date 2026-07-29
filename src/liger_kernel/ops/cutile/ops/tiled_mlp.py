# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tiled MLP (cuTile backend).

Pure Python implementation — no GPU kernel.
Shards input along sequence dimension (dim=-2), applies fn on each shard,
and concatenates. Backward re-computes forward per shard to save memory.
"""

import math

from typing import Callable
from typing import List
from typing import Optional

import torch


class LigerTiledMLPFunction(torch.autograd.Function):
    """Tiled MLP computation (no GPU kernel, memory-efficient via re-computation)."""

    @staticmethod
    def forward(ctx, fn, mlp_module, x, shards, compute_params=None):
        # compute_params is part of the upstream API (intended for DeepSpeed ZeRO
        # weight registration); we accept and forward it for parity but don't
        # consume it here — the autograd machinery already tracks fn's weights.
        del compute_params
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        return torch.cat(output_shards, dim=-2)

    @staticmethod
    def backward(ctx, *grads):
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad

        x_detached = x.detach()
        x_shards = list(torch.chunk(x_detached, chunks=shards, dim=-2))
        grad_shards = list(torch.chunk(grads[0], chunks=shards, dim=-2))

        if x_requires_grad:
            x_grad = torch.zeros_like(x_detached)
            # torch.chunk on dim=-2 of a contiguous tensor returns views that
            # share storage with x_grad. We assign the matching view as each
            # shard leaf's .grad so autograd writes the per-shard gradients
            # in-place into x_grad without an extra copy.
            assert x_detached.is_contiguous(), "x must be contiguous for x_grad view chunking"
            x_grad_shards = list(torch.chunk(x_grad, chunks=shards, dim=-2))
        else:
            x_grad = None

        for i, (x_shard, grad_shard) in enumerate(zip(x_shards, grad_shards)):
            x_shard_leaf = x_shard.detach().requires_grad_(x_requires_grad)
            if x_requires_grad:
                x_shard_leaf.grad = x_grad_shards[i]
            with torch.enable_grad():
                output = fn(mlp_module, x_shard_leaf)
            torch.autograd.backward(output, grad_shard)

        return None, None, x_grad, None, None


def apply_tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    if num_shards is None:
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)
    num_shards = max(1, num_shards)
    return LigerTiledMLPFunction.apply(fn, mlp_module, x, num_shards, compute_params)
