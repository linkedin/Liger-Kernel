import torch
import torch.nn as nn

from liger_kernel.ops import LigerMLPFunction


class LigerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LigerMLPFunction.apply(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)


class LigerFalconH1MLP(nn.Module):
    """
    Patch FalconH1MLP to use LigerMLPFunction with gate / down multipliers.
    Falcon H1's MLP block pre-scales the gate pre-activation and post-scales the
    down projection output:
        y = down_proj(silu(gate_proj(x) * gate_mult) * up_proj(x)) * down_mult
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_h1/modeling_falcon_h1.py
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if getattr(config, "mlp_bias", False):
            raise ValueError("LigerFalconH1MLP does not support bias")
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
        gate_multiplier, down_multiplier = config.mlp_multipliers
        self.gate_multiplier = gate_multiplier
        self.down_multiplier = down_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LigerMLPFunction.apply(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            float(self.gate_multiplier),
            float(self.down_multiplier),
        )
