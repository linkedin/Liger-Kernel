import math

from contextlib import nullcontext
from typing import Callable
from typing import List
from typing import Optional

import torch

from liger_kernel.ops.utils import ensure_contiguous

# Try to import FSDP at module level
try:
    from torch.distributed.fsdp import FullyShardedDataParallel

    FSDP_AVAILABLE = True
except ImportError:
    FullyShardedDataParallel = None
    FSDP_AVAILABLE = False


def _find_ddp_wrapper(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    """
    Find the DDP or FSDP wrapper for a given module by traversing up the module hierarchy.

    This function searches for a wrapper that has the no_sync() method, which is used
    to prevent gradient synchronization during tiled computation.

    Args:
        module: The module to find the wrapper for

    Returns:
        The DDP/FSDP wrapper if found, None otherwise
    """
    # First check if the module itself is a wrapper
    if hasattr(module, "no_sync"):
        return module

    # Check if there's a _ddp_wrapper attribute (custom tracking)
    if hasattr(module, "_ddp_wrapper") and module._ddp_wrapper is not None:
        return module._ddp_wrapper

    return None


def _detect_distributed_framework(mlp_module: torch.nn.Module) -> tuple:
    """
    Detect if the module is wrapped with DDP or FSDP.

    Returns:
        (is_ddp, is_fsdp, wrapper): tuple of (bool, bool, wrapper_or_None)
    """
    # Direct wrapper detection
    is_ddp = isinstance(mlp_module, torch.nn.parallel.DistributedDataParallel)
    is_fsdp = FSDP_AVAILABLE and isinstance(mlp_module, FullyShardedDataParallel)

    wrapper = None
    if is_ddp or is_fsdp:
        wrapper = mlp_module

    # If not directly wrapped, check if distributed training is active
    if not (is_ddp or is_fsdp):
        # Try to find wrapper through custom tracking
        wrapper = _find_ddp_wrapper(mlp_module)
        if wrapper is not None:
            is_ddp = isinstance(wrapper, torch.nn.parallel.DistributedDataParallel)
            is_fsdp = FSDP_AVAILABLE and isinstance(wrapper, FullyShardedDataParallel)

        # If still not found, check if distributed is initialized
        if not (is_ddp or is_fsdp):
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    # Distributed is active but no wrapper found
                    is_ddp = True
            except (ImportError, RuntimeError):
                pass

    return is_ddp, is_fsdp, wrapper


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
        compute_params: a list of weights engaged in the compute (only needed when using DeepSpeed ZeRO)

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
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad] if compute_params else []

        # Detect distributed training framework once in forward
        ctx.is_ddp, ctx.is_fsdp, ctx.ddp_wrapper = _detect_distributed_framework(mlp_module)

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
        is_ddp = ctx.is_ddp
        is_fsdp = ctx.is_fsdp
        ddp_wrapper = ctx.ddp_wrapper

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
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        for i, x_shard in enumerate(x_shards):
            is_last_shard = i + 1 >= shards

            # Handle gradient synchronization for different distributed frameworks
            if compute_params:
                # DeepSpeed: use ds_grad_is_ready flag
                if hasattr(compute_params[0], "ds_grad_is_ready"):
                    for param in compute_params:
                        param.ds_grad_is_ready = is_last_shard
                # DDP/FSDP: use no_sync() context manager for all but last shard
                elif is_ddp or is_fsdp:
                    pass  # Handled by context manager below

            # Use no_sync() context to prevent gradient reduction until last shard
            sync_context = nullcontext()
            if (is_ddp or is_fsdp) and not is_last_shard:
                # Use the DDP/FSDP wrapper's no_sync() if available
                if ddp_wrapper is not None and hasattr(ddp_wrapper, "no_sync"):
                    sync_context = ddp_wrapper.no_sync()
                # Fallback: check if mlp_module itself has no_sync() (direct wrapper case)
                elif hasattr(mlp_module, "no_sync"):
                    sync_context = mlp_module.no_sync()
                # Case: no wrapper found with no_sync() method
                # In this edge case, gradient synchronization will occur on every shard (inefficient),
                # but the final result remains correct.

            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with sync_context:
                with torch.enable_grad():
                    output = fn(mlp_module, x_shard)
                torch.autograd.backward(output, incoming_grad_shard)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)


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

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )
