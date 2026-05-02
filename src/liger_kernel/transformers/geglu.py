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
    """GEGLU MLP wrapper matching Gemma4TextMLP's (config, layer_idx) constructor.

    HF's Gemma4TextMLP conditionally doubles intermediate_size for KV-shared layers
    when ``config.use_double_wide_mlp=True``. This subclass replicates that logic
    so the class-level swap works for all Gemma 4 variants (31B text, future MoE).

    See: https://github.com/huggingface/transformers/blob/74a2a4d0c/src/transformers/models/gemma4/modeling_gemma4.py#L1030-L1035
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        # Match HF's conditional doubling for KV-shared layers
        if layer_idx is not None and getattr(config, "use_double_wide_mlp", False):
            num_hidden = getattr(config, "num_hidden_layers", 0)
            num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
            first_kv_shared = num_hidden - num_kv_shared
            if num_kv_shared > 0 and layer_idx >= first_kv_shared:
                doubled = config.intermediate_size * 2
                self.intermediate_size = doubled
                self.gate_proj = nn.Linear(self.hidden_size, doubled, bias=False)
                self.up_proj = nn.Linear(self.hidden_size, doubled, bias=False)
                self.down_proj = nn.Linear(doubled, self.hidden_size, bias=False)
