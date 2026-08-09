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
