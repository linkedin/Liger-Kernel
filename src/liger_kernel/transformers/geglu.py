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
        # Right now Gemma 1, 1.1 and 2 models are all using `gelu_pytorch_tanh`
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gemma/modeling_gemma.py#L175
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/activations.py#L46
        # So we can safely assume we use tanh approximation form all the time

    def forward(self, x):
        return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
