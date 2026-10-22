import math

from typing import Callable
from typing import List
from typing import Optional

import torch

from torch.utils.checkpoint import checkpoint


class LigerTiledMLPFunction:
    """Compatibility entry point for checkpointed tiled MLP computation.

    Each shard is recomputed during backward to reduce activation memory.
    Non-reentrant checkpointing records the parameter dependencies even when
    the input does not require gradients, and supports ``torch.autograd.grad``.
    """

    @staticmethod
    def apply(
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        # Keep compute_params in the public API; autograd discovers the weights
        # used by fn without registering them through a separate parameter list.
        x = x.contiguous()
        output_shards = [
            checkpoint(fn, mlp_module, x_shard, use_reentrant=False)
            for x_shard in torch.chunk(x, chunks=shards, dim=-2)
        ]
        return torch.cat(output_shards, dim=-2)


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
        compute_params: optional parameter list retained for API compatibility

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
