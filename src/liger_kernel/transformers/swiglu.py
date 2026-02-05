import torch.nn as nn
import torch
import torch.nn.functional as F

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


class LigerMoEExperts(nn.Module):
    """
    nn.Module for patching MoE Experts nn.Modules (e.g. Qwen3MoeExperts, Llama4TextExperts).
    Forward signature matches Hugging Face Experts so the patched module is interchangeable.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # self is the HF Qwen3MoeExperts module.
        num_experts = self.num_experts
        act_fn = getattr(self, "act_fn", None)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Building expert mask as in HF 
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)          
            expert_mask = expert_mask.permute(2, 1, 0)                            
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()            

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0]
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])          
            current_state = hidden_states[token_idx]                               
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)                                  

            if act_fn in (F.silu, torch.nn.functional.silu) or getattr(getattr(act_fn, "__name__", None), "lower", lambda: "")() in {"silu", "swish"}:
                activated = LigerSiLUMulFunction.apply(gate, up) # Qwen3MoE uses SiLU/Swish activat
            else:
                activated = act_fn(gate) * up

            #down projection
            current_hidden_states = nn.functional.linear(activated, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def extra_repr(self):
        return "LigerMoEExperts"
        
    def _get_name(self):
        return LigerMoEExperts.__name__
