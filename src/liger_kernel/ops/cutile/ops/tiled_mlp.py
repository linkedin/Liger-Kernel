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

from liger_kernel.ops.tiled_mlp import _autograd_input_params


class LigerTiledMLPFunction(torch.autograd.Function):
    """Tiled MLP computation (no GPU kernel, memory-efficient via re-computation)."""

    @staticmethod
    def forward(ctx, fn, mlp_module, x, shards, compute_params=None, *params):
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        # the weights in `params`, resolved to the originals `fn` closes over, so their gradients can
        # be returned from backward instead of accumulated once per shard
        ctx.grad_params = _autograd_input_params(compute_params)
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
        grad_params = ctx.grad_params

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

        # Summed over shards and returned as this Function's gradients, so autograd accumulates each
        # weight exactly once. Empty under ZeRO-3, which keeps the per-shard accumulation below.
        param_grads = [None] * len(grad_params)

        for i, (x_shard, grad_shard) in enumerate(zip(x_shards, grad_shards)):
            x_shard_leaf = x_shard.detach().requires_grad_(x_requires_grad)

            if grad_params:
                with torch.enable_grad():
                    output = fn(mlp_module, x_shard_leaf)
                inputs = ([x_shard_leaf] if x_requires_grad else []) + grad_params
                # torch.autograd.grad returns the gradients instead of accumulating into .grad, so no
                # AccumulateGrad node fires here and DDP's per-parameter hook stays untouched
                shard_grads = torch.autograd.grad(
                    outputs=output,
                    inputs=inputs,
                    grad_outputs=grad_shard,
                    allow_unused=True,
                )
                if x_requires_grad:
                    x_grad_shards[i].copy_(shard_grads[0])
                    shard_grads = shard_grads[1:]
                for j, shard_grad in enumerate(shard_grads):
                    if shard_grad is None:
                        continue
                    # the first shard's gradient is freshly allocated, so accumulate into it in place
                    param_grads[j] = shard_grad if param_grads[j] is None else param_grads[j].add_(shard_grad)
            else:
                if x_requires_grad:
                    x_shard_leaf.grad = x_grad_shards[i]
                with torch.enable_grad():
                    output = fn(mlp_module, x_shard_leaf)
                torch.autograd.backward(output, grad_shard)

        return None, None, x_grad, None, None, *param_grads


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
    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
        # passed as explicit Function inputs so autograd accumulates each weight once per backward
        *_autograd_input_params(compute_params),
    )
