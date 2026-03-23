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
        *params: MLP parameters (passed as explicit inputs for FSDP compatibility)

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
        *params: torch.nn.Parameter,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.num_params = len(params)
        ctx.params = params  # Store params as tuple, don't save (they're in mlp_module)
        ctx.save_for_backward(x)  # Only save input tensor

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
        x = ctx.saved_tensors[0]  # Only x was saved
        params = ctx.params  # Get params from context (not saved_tensors)
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad
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

        # Initialize param grad accumulators as None for lazy allocation
        param_grads: List[Optional[torch.Tensor]] = [None for _ in params]

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        # Calculate cumulative offsets for correct gradient slicing when shards are uneven
        shard_offset = 0
        for i, x_shard in enumerate(x_shards):
            x_shard = x_shard.detach()
            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            # Build inputs list: x_shard + params that require grad
            inputs = [x_shard] if x_requires_grad else []
            inputs.extend([p for p in params if p.requires_grad])

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)
                if inputs:
                    # Use torch.autograd.grad for FSDP compatibility
                    # FSDP needs explicit gradient returns to manage sharded parameters
                    local_grads = torch.autograd.grad(
                        outputs=output,
                        inputs=inputs,
                        grad_outputs=incoming_grad_shard,
                    )
                else:
                    local_grads = []

            # Process gradients
            grad_idx = 0
            if x_requires_grad and x_grad is not None:
                x_grad.narrow(0, shard_offset, shard_step).copy_(local_grads[grad_idx])
                grad_idx += 1

            # Accumulate parameter gradients using in-place operations
            for param_idx, p in enumerate(params):
                if p.requires_grad:
                    grad = local_grads[grad_idx]
                    if param_grads[param_idx] is None:
                        # First shard: clone to avoid keeping local_grads alive
                        param_grads[param_idx] = grad.clone()
                    else:
                        # Subsequent shards: accumulate in-place
                        existing_grad = param_grads[param_idx]
                        assert existing_grad is not None
                        # Use add_ for true in-place accumulation
                        existing_grad.add_(grad)
                    grad_idx += 1

            # Update offset for next shard
            shard_offset += shard_step

            # CRITICAL: Explicitly delete local_grads to free memory immediately
            # Without this, the gradient tensors stay alive until loop completion
            del local_grads

        # unflatten x_grad if needed
        if x_grad is not None:
            x_grad = x_grad.view(x_shape_orig)

        # Return gradients: (fn, mlp_module, x, shards, *params)
        # Clone param_grads to ensure they're not views into local_grads
        final_param_grads = []
        for param_idx, p in enumerate(params):
            if param_grads[param_idx] is not None:
                final_param_grads.append(param_grads[param_idx].clone())
            else:
                final_param_grads.append(torch.zeros_like(p))

        #  (fn, mlp_module, x, shards, *params)
        return (None, None, x_grad, None, *final_param_grads)


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
        compute_params: list of parameters engaged in the computation (for FSDP compatibility)

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

    # Get all parameters from the module if compute_params not provided
    if compute_params is None:
        compute_params = list(mlp_module.parameters())

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        *compute_params,
    )
