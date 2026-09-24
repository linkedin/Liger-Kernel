import torch

from liger_kernel.ops import LigerDeepseekV4RopeFunction


def liger_deepseek_v4_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply DeepSeek-V4's interleaved RoPE to the trailing rotary slice."""
    return LigerDeepseekV4RopeFunction.apply(x, cos, sin, unsqueeze_dim)
