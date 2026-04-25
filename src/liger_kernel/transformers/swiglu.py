import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.ops import LigerSiLUMulFunction


class LigerSwiGLUMLP(nn.Module):
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

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))


class LigerExperts(nn.Module):
    """
    Patch MixtralExperts for transformers v5 or later to use LigerSiLUMulFunction
    https://github.com/huggingface/transformers/blob/393b4b3d28e29b4b05b19b4b7f3242a7fc893637/src/transformers/models/mixtral/modeling_mixtral.py#L63
    """

    def __init__(self, config):
        super().__init__()
        if hasattr(config, "num_experts"):
            # qwen3_moe, qwen3_next uses num_experts
            self.num_experts = config.num_experts
        else:
            self.num_experts = config.num_local_experts
        if hasattr(config, "moe_intermediate_size"):
            # qwen3_moe, qwen3_next uses moe_intermediate_size
            self.intermediate_dim = config.moe_intermediate_size
        else:
            self.intermediate_dim = config.intermediate_size

        self.hidden_dim = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Reshape to 2D if needed (e.g. batch × seq → tokens)
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_dim)

        # top_k_index / top_k_weights may come in as (batch, seq, K) or (T, K)
        top_k_index_2d = top_k_index.view(-1, top_k_index.shape[-1]).to(torch.int32)
        top_k_weights_2d = top_k_weights.view(-1, top_k_weights.shape[-1])

        out = LigerFusedMoEFunction.apply(x, self.gate_up_proj, self.down_proj, top_k_index_2d, top_k_weights_2d)
        return out.view(orig_shape)


class LigerPhi3SwiGLUMLP(nn.Module):
    """
    Patch Phi3MLP to use LigerSiLUMulFunction
    https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(LigerSiLUMulFunction.apply(gate, up_states))


class LigerQwen3MoeSwiGLUMLP(nn.Module):
    """
    Patch Qwen3MoeMLP to use LigerSiLUMulFunction.
    https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L57
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerHunyuanV1SwiGLUMLP(nn.Module):
    def __init__(self, config, layer_idx=None, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.layer_idx = layer_idx
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerFalconH1SwiGLUMLP(nn.Module):
    """
    Patch FalconH1MLP to use LigerSiLUMulFunction with gate / down multipliers.

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
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
        gate_multiplier, down_multiplier = config.mlp_multipliers
        self.gate_multiplier = float(gate_multiplier)
        self.down_multiplier = float(down_multiplier)

    def forward(self, x):
        # When patched onto an already-instantiated HF FalconH1MLP via _patch_swiglu_module,
        # only `forward` is rebound — read multipliers from the instance if present, else config.
        gate_multiplier = getattr(self, "gate_multiplier", None)
        down_multiplier = getattr(self, "down_multiplier", None)
        if gate_multiplier is None or down_multiplier is None:
            gate_multiplier, down_multiplier = self.config.mlp_multipliers
        return self.down_proj(
            LigerSiLUMulFunction.apply(
                self.gate_proj(x),
                self.up_proj(x),
                float(gate_multiplier),
                float(down_multiplier),
            )
        )
