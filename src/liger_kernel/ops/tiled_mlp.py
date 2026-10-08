import math

from typing import Callable
from typing import List
from typing import Optional

import torch

from liger_kernel.ops.utils import ensure_contiguous


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
        compute_params: weights engaged in the compute

    Returns:
        the computed hidden_states
    """

    @staticmethod
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_param_names: tuple[str, ...],
        *compute_params: torch.nn.Parameter,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.compute_param_names = compute_param_names
        x = x.contiguous()
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
        compute_params = tuple(_resolve_parameter(mlp_module, name) for name in ctx.compute_param_names)

        x_requires_grad = ctx.needs_input_grad[2]
        x = x.detach()
        # detach() unsets x.requires_grad, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x) if x_requires_grad else None
        param_grads = [None for _ in compute_params]

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        shard_offset = 0
        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)
                grad_inputs = ((x_shard,) if x_requires_grad else ()) + compute_params
                local_grads = torch.autograd.grad(
                    outputs=output,
                    inputs=grad_inputs,
                    grad_outputs=incoming_grad_shard,
                    allow_unused=True,
                )

            if x_requires_grad and x_grad is not None:
                local_x_grad = local_grads[0]
                if local_x_grad is not None:
                    x_grad.narrow(0, shard_offset, shard_step).copy_(local_x_grad)
                local_param_grads = local_grads[1:]
            else:
                local_param_grads = local_grads

            for param_index, local_param_grad in enumerate(local_param_grads):
                if local_param_grad is None:
                    continue
                if param_grads[param_index] is None:
                    param_grads[param_index] = local_param_grad
                else:
                    param_grads[param_index].add_(local_param_grad)

            shard_offset += shard_step

        # unflatten
        if x_grad is not None:
            x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None, *param_grads)


def _resolve_parameter(module: torch.nn.Module, name: str) -> torch.Tensor:
    parameter = module
    for component in name.split("."):
        parameter = getattr(parameter, component)
    return parameter


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

    if compute_params is None:
        compute_params = [param for param in mlp_module.parameters() if param.requires_grad]
    else:
        compute_params = [param for param in compute_params if param.requires_grad]

    module_param_names = {id(param): name for name, param in mlp_module.named_parameters(remove_duplicate=False)}
    try:
        compute_param_names = tuple(module_param_names[id(param)] for param in compute_params)
    except KeyError as error:
        raise ValueError("compute_params must contain parameters from mlp_module") from error

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_param_names,
        *compute_params,
    )
