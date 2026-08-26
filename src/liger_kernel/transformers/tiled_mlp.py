from typing import Optional

import torch.nn as nn

from liger_kernel.ops import LigerGELUMulFunction
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.ops import apply_tiled_mlp


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

        # Validate activation function
        # LigerGELUMulFunction is the tanh approximation, so exact (erf) gelu is not one of these;
        # LigerTiledGLUMLP handles it without approximating.
        if hasattr(config, "hidden_act") and config.hidden_act not in [
            "gelu_new",
            "gelu_pytorch_tanh",
        ]:
            raise ValueError(
                f"LigerTiledGEGLUMLP requires tanh-approximation GELU, got {config.hidden_act}. "
                "Use LigerTiledGLUMLP to keep the tiling with this activation."
            )

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
        compute_params = [p for p in self.parameters() if p.requires_grad]

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
        compute_params = [p for p in self.parameters() if p.requires_grad]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


# Activations with a fused Liger gate*activation kernel. Exact (erf) GELU is deliberately absent:
# LigerGELUMulFunction implements the tanh approximation, so routing erf GELU through it would change
# numerics. Anything unlisted falls back to act_fn(gate) * up, which is still tiled -- tiling is where
# the memory saving is, the fused activation adds only a few percent on top.
_FUSED_MUL_BY_ACTIVATION_NAME = {
    "silu": LigerSiLUMulFunction,
    "swish": LigerSiLUMulFunction,
    "gelu_pytorch_tanh": LigerGELUMulFunction,
    "gelu_new": LigerGELUMulFunction,
}

_fused_mul_by_activation_type = None


def _fused_mul_for(module: nn.Module):
    """The fused kernel matching this module's activation, or None to compute it eagerly.

    Keyed on the type transformers itself builds for each activation name rather than on a hardcoded
    class name, since those have been renamed across versions (PytorchGELUTanh -> GELUTanh). Reads the
    activation off the module rather than the config so this also works when the caller has bound these
    methods onto an already-built MLP; Llama4 names the attribute activation_fn.
    """
    global _fused_mul_by_activation_type
    if _fused_mul_by_activation_type is None:
        from transformers.activations import ACT2FN

        resolved = {}
        for name, fused_mul in _FUSED_MUL_BY_ACTIVATION_NAME.items():
            try:
                resolved[type(ACT2FN[name])] = fused_mul
            except KeyError:
                continue
        _fused_mul_by_activation_type = resolved

    act = getattr(module, "act_fn", None)
    if act is None:
        act = getattr(module, "activation_fn", None)
    if act is None:
        return None
    return _fused_mul_by_activation_type.get(type(act))


class LigerTiledGLUMLP(nn.Module):
    """
    Memory-efficient gated MLP using tiled computation, for any activation.

    Like LigerTiledSwiGLUMLP and LigerTiledGEGLUMLP this shards the sequence and recomputes the
    forward during the backward, but it does not require a particular activation: the fused Liger
    kernel is used when one matches, otherwise the activation is applied eagerly. Only the fused
    kernel is given up in that case, not the tiling, so nearly all of the memory saving remains.

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

        from transformers.activations import ACT2FN

        # gemma-style configs name this hidden_activation
        activation = getattr(config, "hidden_activation", None) or getattr(config, "hidden_act", None)
        if activation is None:
            raise ValueError("LigerTiledGLUMLP requires config.hidden_activation or config.hidden_act")
        self.act_fn = ACT2FN[activation]

    def _mlp_forward(self, module, x):
        """Internal MLP forward function for tiled computation."""
        gate = module.gate_proj(x)
        up = module.up_proj(x)
        fused_mul = _fused_mul_for(module)
        if fused_mul is not None:
            return module.down_proj(fused_mul.apply(gate, up))
        return module.down_proj(module.act_fn(gate) * up)

    def forward(self, x):
        """
        Forward pass with tiled computation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
               or [seq_len, hidden_size]

        Returns:
            Output tensor of the same shape as input
        """
        compute_params = [p for p in self.parameters() if p.requires_grad]

        return apply_tiled_mlp(
            fn=self._mlp_forward,
            mlp_module=self,
            x=x,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


# The fused kernel each activation-specific tiled module hardcodes. Callers use this to check whether a
# module's activation actually matches, without having to instantiate anything. LigerTiledGLUMLP is
# absent on purpose: it adapts to whatever activation the module carries.
_REQUIRED_FUSED_MUL = {
    LigerTiledSwiGLUMLP: LigerSiLUMulFunction,
    LigerTiledGEGLUMLP: LigerGELUMulFunction,
}
