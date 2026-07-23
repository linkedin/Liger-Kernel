import torch
import torch.nn as nn

from liger_kernel.ops import LigerFusedMoEFunction
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.ops.utils import is_hip


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


class LigerLfm2SwiGLUMLP(nn.Module):
    """LFM2 SwiGLU MLP using the fused SiLU-multiply kernel.

    LFM2 names its projections w1, w3, and w2 and computes
    w2(silu(w1(x)) * w3(x)). Its configuration also adjusts the dense
    intermediate size before module construction, so that calculation must be
    preserved when the class is patched before model initialization.
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
            if getattr(config, "block_auto_adjust_ff_dim", False):
                intermediate_size = int(2 * intermediate_size / 3)
                if config.block_ffn_dim_multiplier is not None:
                    intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                    intermediate_size = config.block_multiple_of * (
                        (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of
                    )

        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))


# MI325X sweeps of the 8B-A1B and 24B-A2B shapes put the crossover
# near 256 routed rows per expert; below it, the fused Triton path is faster.
_ROCM_GROUPED_MM_MIN_ROWS_PER_EXPERT = 256


class LigerLfm2MoeExperts(LigerExperts):
    """LFM2-MoE experts with shape-aware ROCm grouped-MM dispatch."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.act_fn = torch.nn.functional.silu
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if is_hip():
            grouped_mm_available = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
            if grouped_mm_available:
                tokens = hidden_states.numel() // self.hidden_dim
                top_k = top_k_index.shape[-1]
                enough_work_per_expert = tokens * top_k >= self.num_experts * _ROCM_GROUPED_MM_MIN_ROWS_PER_EXPERT
                if enough_work_per_expert:
                    try:
                        from transformers.integrations.moe import grouped_mm_experts_forward
                    except ImportError:
                        pass
                    else:
                        orig_shape = hidden_states.shape
                        x = hidden_states.view(-1, self.hidden_dim)
                        out = grouped_mm_experts_forward(
                            self, x, top_k_index.view(x.shape[0], -1), top_k_weights.view(x.shape[0], -1)
                        )
                        return out.view(orig_shape)
        return LigerExperts.forward(self, hidden_states, top_k_index, top_k_weights)


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
        return self.down_proj(
            LigerSiLUMulFunction.apply(
                self.gate_proj(x),
                self.up_proj(x),
                float(self.gate_multiplier),
                float(self.down_multiplier),
            )
        )
