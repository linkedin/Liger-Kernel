import torch.nn as nn

from liger_kernel.ops import LigerGELUMulFunction


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


class LigerGEGLUMLPForGemma4(LigerGEGLUMLP):
    """GEGLU MLP wrapper that tolerates Gemma4TextMLP's two-arg constructor.

    HF's Gemma4TextMLP is instantiated as ``Gemma4TextMLP(config, layer_idx)``;
    swapping in plain LigerGEGLUMLP (single-arg) breaks model construction.
    This subclass accepts and ignores ``layer_idx`` — 31B has
    ``use_double_wide_mlp=false``, so the layer_idx never needed to feed the
    doubled intermediate_size path. Forward is inherited unchanged.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config)
