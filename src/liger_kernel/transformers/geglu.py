import torch.nn as nn

from liger_kernel.ops.geglu import LigerGELUMulFunction


class LigerGEGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # TODO: support exact GELU
        if config.hidden_act not in ["gelu_pytorch_tanh"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):

        return self.down_proj(
            LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )
