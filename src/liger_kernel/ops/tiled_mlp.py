import math

from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.distributed as dist

from liger_kernel.ops.utils import ensure_contiguous


def _autograd_input_params(
    compute_params: Optional[List[torch.nn.Parameter]],
) -> List[torch.nn.Parameter]:
    """The weights to pass to the autograd Function as explicit inputs.

    Doing so makes autograd accumulate each weight once, at the end of the backward, instead of once
    per shard. DDP registers one hook per parameter and marks it ready on every firing, so the
    per-shard accumulation would trip "Expected to mark a variable ready only once".

    Returns an empty list under ZeRO-3: partitioned weights are 0-sized placeholders outside their
    owning module's forward, so autograd would record ``(0,)`` as the expected gradient shape and the
    recompute would fail. Those stay on the accumulate-into-``.grad`` path, where the reduction is
    deferred to the last shard via ``ds_grad_is_ready``.
    """
    if not compute_params:
        return []
    if any(hasattr(p, "ds_id") for p in compute_params):
        return []
    return [p for p in compute_params if p.requires_grad]


class LigerTiledMLPFunction(torch.autograd.Function):
    """
    Based on DeepSpeed's TiledMLP:
    https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L838

    Perform a tiled MLP computation to massively reduce memory usage needed to compute MLP
    when using very long sequence lengths.

    This module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration.
    And if you're using activation checkpointing it then occurs thrice.

    Args:
        fn: the function to call on sharded inputs (e.g., mlp.forward)
        mlp_module: the MLP nn.Module object
        x: the input to MLP.forward (hidden_states)
        shards: how many shards to use
        compute_params: a list of weights engaged in the compute

    Returns:
        the computed hidden_states
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.compute_params = compute_params
        # `params` arrives via ensure_contiguous, which may hand us copies; resolve the originals so
        # torch.autograd.grad differentiates the tensors the recompute actually uses.
        ctx.grad_params = _autograd_input_params(compute_params)
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *grads) -> tuple:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards
        compute_params = ctx.compute_params
        grad_params = ctx.grad_params

        grad_output = grads[0]

        x_requires_grad = x.requires_grad
        x = x.detach()
        # detach() unsets x.requires_grad, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, in_features] or [seqlen, in_features] (moe experts)
        in_features = x.shape[-1]
        # fn may change the feature dim, e.g. Linear(in_features, out_features), so flatten x and
        # grad_output with their own last dims rather than assuming they are equal
        out_features = grad_output.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, in_features)
        incoming_grad = grad_output.view(-1, out_features)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))
        incoming_grad_shards = list(torch.chunk(incoming_grad, chunks=shards, dim=0))

        # ZeRO-3 partitioned parameters carry a ds_id; collect them once so the per-shard loop only
        # flips the ready flag. Parameters on other backends have no ds_id and are left untouched.
        ds_params = [p for p in compute_params if hasattr(p, "ds_id")] if compute_params else []

        # Summed over shards and returned as this Function's gradients, so autograd accumulates each
        # weight exactly once. Empty under ZeRO-3, which keeps the per-shard accumulation below.
        param_grads = [None] * len(grad_params)

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            incoming_grad_shard = incoming_grad_shards[i]

            if grad_params:
                with torch.enable_grad():
                    output = fn(mlp_module, x_shard)
                inputs = ([x_shard] if x_requires_grad else []) + grad_params
                # torch.autograd.grad returns the gradients instead of accumulating into .grad, so no
                # AccumulateGrad node fires here and DDP's per-parameter hook stays untouched
                shard_grads = torch.autograd.grad(
                    outputs=output,
                    inputs=inputs,
                    grad_outputs=incoming_grad_shard,
                    allow_unused=True,
                )
                if x_requires_grad:
                    x_grad.narrow(0, shard_offset, shard_step).copy_(shard_grads[0])
                    shard_grads = shard_grads[1:]
                for j, shard_grad in enumerate(shard_grads):
                    if shard_grad is None:
                        continue
                    # the first shard's gradient is freshly allocated, so accumulate into it in place
                    param_grads[j] = shard_grad if param_grads[j] is None else param_grads[j].add_(shard_grad)
            else:
                x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

                # Defer DeepSpeed's reduction until the last shard has accumulated into param.grad; the flag
                # is read by ZeRO's hook during each shard's backward, so it must be set per shard.
                for param in ds_params:
                    param.ds_grad_is_ready = i + 1 == len(x_shards)

                with torch.enable_grad():
                    output = fn(mlp_module, x_shard)
                torch.autograd.backward(output, incoming_grad_shard)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None, *param_grads)


def apply_tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    Apply tiled MLP computation for memory efficiency.

    Args:
        fn: the function to call on sharded inputs (e.g., lambda module, x: module(x))
        mlp_module: the MLP nn.Module object
        x: the input tensor with shape [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        num_shards: number of shards to use. If None, automatically calculated as ceil(seqlen / hidden_size)
        compute_params: list of parameters for DeepSpeed ZeRO optimization

    Returns:
        output tensor with the same shape as input
    """
    if num_shards is None:
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)

    # Ensure num_shards is at least 1
    num_shards = max(1, num_shards)

    # All ranks must run the same number of shards: a sharded-parameter backend (DeepSpeed ZeRO-3, FSDP)
    # gathers weights inside each shard's recompute, so a rank that runs fewer shards stops participating
    # in those collectives and deadlocks the others. Harmonize on the per-rank maximum.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        num_shards_tensor = torch.tensor(num_shards, device=x.device)
        dist.all_reduce(num_shards_tensor, op=dist.ReduceOp.MAX)
        num_shards = int(num_shards_tensor.item())

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
        # passed as explicit Function inputs so autograd accumulates each weight once per backward
        *_autograd_input_params(compute_params),
    )
