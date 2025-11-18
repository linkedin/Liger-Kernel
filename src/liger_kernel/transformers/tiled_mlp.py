from typing import Optional
import sys
import torch
import torch.nn as nn

from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp


def _register_ddp_wrapper_hook(module: nn.Module) -> None:
    """
    Register a forward pre-hook to track the DDP/FSDP wrapper.

    This allows the tiled MLP to find the wrapper and use its no_sync() method
    for efficient gradient synchronization.
    """

    def _find_wrapper_hook(module, input):
        # Skip if already set
        if hasattr(module, "_ddp_wrapper") and module._ddp_wrapper is not None:
            return

        # Try to find wrapper by traversing the call stack
        # This is a heuristic approach since PyTorch doesn't track parent modules

        frame = sys._getframe()
        max_depth = 20  # Limit search depth

        for _ in range(max_depth):
            frame = frame.f_back
            if frame is None:
                break

            # Look for 'self' in the frame's locals
            if "self" in frame.f_locals:
                obj = frame.f_locals["self"]
                # Check if it's a DDP or FSDP wrapper
                if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
                    module._ddp_wrapper = obj
                    return
                # Check for FSDP
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel

                    if isinstance(obj, FullyShardedDataParallel):
                        module._ddp_wrapper = obj
                        return
                except ImportError:
                    pass

    module.register_forward_pre_hook(_find_wrapper_hook)


class LigerTiledGEGLUMLP(nn.Module):
    """
    Memory-efficient GEGLU MLP using tiled computation.

    This module combines GEGLU activation with tiled processing to handle
    very long sequences efficiently. The forward pass is recomputed during
    backward to save memory.

    Args:
        config: Model configuration with hidden_size and intermediate_size attributes
        num_shards: Number of shards to split the sequence. If None, automatically
                   calculated as ceil(seqlen / hidden_size)
    """

    def __init__(self, config, num_shards: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Initialize DDP wrapper tracking
        self._ddp_wrapper = None
        _register_ddp_wrapper_hook(self)

        # Validate activation function
        if hasattr(config, "hidden_act") and config.hidden_act not in [
            "gelu",
            "gelu_new",
            "gelu_pytorch_tanh",
        ]:
            raise ValueError(f"LigerTiledGEGLUMLP requires GELU activation, got {config.hidden_act}")

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(LigerGELUMulFunction.apply(gate, up))

    def forward(self, x):
        """
        Forward pass with tiled computation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
               or [seq_len, hidden_size]

        Returns:
            Output tensor of the same shape as input
        """
        compute_params = [
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        ]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


class LigerTiledSwiGLUMLP(nn.Module):
    """
    Memory-efficient SwiGLU MLP using tiled computation.

    This module combines SwiGLU activation with tiled processing to handle
    very long sequences efficiently. The forward pass is recomputed during
    backward to save memory.

    Args:
        config: Model configuration with hidden_size and intermediate_size attributes
        num_shards: Number of shards to split the sequence. If None, automatically
                   calculated as ceil(seqlen / hidden_size)
    """

    def __init__(self, config, num_shards: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_shards = num_shards

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Initialize DDP wrapper tracking
        self._ddp_wrapper = None
        _register_ddp_wrapper_hook(self)

        # Validate activation function
        if hasattr(config, "hidden_act") and config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"LigerTiledSwiGLUMLP requires SiLU/Swish activation, got {config.hidden_act}")

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        return module.down_proj(LigerSiLUMulFunction.apply(gate, up))

    def forward(self, x):
        """
        Forward pass with tiled computation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
               or [seq_len, hidden_size]

        Returns:
            Output tensor of the same shape as input
        """
        compute_params = [
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        ]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )
