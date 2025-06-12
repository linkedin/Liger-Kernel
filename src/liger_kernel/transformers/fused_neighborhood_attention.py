import math

from typing import Optional

import torch
import torch.nn as nn

from liger_kernel.ops.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionFunction


class LigerFusedNeighborhoodAttention(nn.Module):
    """
    Liger Fused Neighborhood Attention Module.

    Paper: https://arxiv.org/pdf/2504.16922

    Fused Neighborhood attention restricts the attention mechanism to a local neighborhood
    around each position, reducing computational complexity from O(n²) to O(n*k)
    where k is the neighborhood size.

    Args:
        hidden_size (int): The hidden dimension size
        num_heads (int): Number of attention heads
        kernel_size (int): Size of the neighborhood window (default: 7)
        dilation (int): Dilation factor for the neighborhood (default: 1)
        bias (bool): Whether to use bias in linear projections (default: True)
        dropout (float): Dropout probability (default: 0.0)
        scale (Optional[float]): Scaling factor for attention scores.
                                If None, uses 1/sqrt(head_dim) (default: None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        if kernel_size <= 0:
            raise ValueError(f"kernel_size ({kernel_size}) must be positive")

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size ({kernel_size}) must be odd")

        if dilation < 1:
            raise ValueError(f"dilation ({dilation}) must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the fused neighborhood attention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask (currently not supported)

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        if attention_mask is not None:
            raise NotImplementedError("Attention mask is not yet supported in LigerFusedNeighborhoodAttention")

        batch_size, seq_len, hidden_size = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = LigerFusedNeighborhoodAttentionFunction.apply(
            query, key, value, self.kernel_size, self.dilation, self.scale
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        output = self.out_proj(attn_output)

        return output

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, scale={self.scale}, dropout={self.dropout_p}"
        )


class LigerFusedNeighborhoodAttentionLayer(nn.Module):
    """
    A complete neighborhood attention layer with layer norm and residual connection.

    Args:
        hidden_size (int): The hidden dimension size
        num_heads (int): Number of attention heads
        kernel_size (int): Size of the neighborhood window (default: 7)
        dilation (int): Dilation factor for the neighborhood (default: 1)
        bias (bool): Whether to use bias in linear projections (default: True)
        dropout (float): Dropout probability (default: 0.0)
        layer_norm_eps (float): Epsilon for layer normalization (default: 1e-5)
        scale (Optional[float]): Scaling factor for attention scores (default: None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        scale: Optional[float] = None,
    ):
        super().__init__()

        self.attention = LigerFusedNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            dropout=dropout,
            scale=scale,
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connection and layer normalization.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask (currently not supported)

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        normed_hidden_states = self.layer_norm(hidden_states)

        attn_output = self.attention(normed_hidden_states, attention_mask)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        output = hidden_states + attn_output

        return output


class LigerFusedNeighborhoodAttentionConfig:
    """
    Configuration class for Fused Neighborhood Attention.

    This can be used to easily configure neighborhood attention parameters
    for different model architectures.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        scale: Optional[float] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.scale = scale

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "bias": self.bias,
            "dropout": self.dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "scale": self.scale,
        }
