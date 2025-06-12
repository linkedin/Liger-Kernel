import math

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.transformers.functional import liger_fused_neighborhood_attention
from liger_kernel.transformers.fused_neighborhood_attention import LigerFusedNeighborhoodAttention
from liger_kernel.transformers.fused_neighborhood_attention import LigerFusedNeighborhoodAttentionLayer
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


class TorchNeighborhoodAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        scale: float = None,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def _create_neighborhood_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        half_kernel = self.kernel_size // 2

        for i in range(seq_len):
            start = max(0, i - half_kernel * self.dilation)
            end = min(seq_len, i + half_kernel * self.dilation + 1)

            for j in range(start, end):
                if self.dilation == 1 or (j - i) % self.dilation == 0:
                    mask[i, j] = True

        return mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        mask = self._create_neighborhood_mask(seq_len, hidden_states.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        output = self.out_proj(attn_output)

        return output


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_heads, kernel_size",
    [
        (2, 32, 128, 4, 7),
        (1, 32, 128, 8, 5),
        (2, 24, 96, 3, 9),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_fused_neighborhood_attention_correctness(
    batch_size, seq_len, hidden_size, num_heads, kernel_size, bias, dtype, atol, rtol
):
    set_seed(42)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    ref_attn = (
        TorchNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=1,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    liger_attn = (
        LigerFusedNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=1,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        liger_attn.q_proj.weight.copy_(ref_attn.q_proj.weight)
        liger_attn.k_proj.weight.copy_(ref_attn.k_proj.weight)
        liger_attn.v_proj.weight.copy_(ref_attn.v_proj.weight)
        liger_attn.out_proj.weight.copy_(ref_attn.out_proj.weight)

        if bias:
            liger_attn.q_proj.bias.copy_(ref_attn.q_proj.bias)
            liger_attn.k_proj.bias.copy_(ref_attn.k_proj.bias)
            liger_attn.v_proj.bias.copy_(ref_attn.v_proj.bias)
            liger_attn.out_proj.bias.copy_(ref_attn.out_proj.bias)

    hidden_states1 = hidden_states.detach().clone().requires_grad_(True)
    hidden_states2 = hidden_states.detach().clone().requires_grad_(True)

    out1 = liger_attn(hidden_states1)
    out2 = ref_attn(hidden_states2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    loss1 = out1.sum()
    loss2 = out2.sum()
    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(hidden_states1.grad, hidden_states2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(liger_attn.q_proj.weight.grad, ref_attn.q_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_attn.k_proj.weight.grad, ref_attn.k_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_attn.v_proj.weight.grad, ref_attn.v_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_attn.out_proj.weight.grad, ref_attn.out_proj.weight.grad, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(liger_attn.q_proj.bias.grad, ref_attn.q_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_attn.k_proj.bias.grad, ref_attn.k_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_attn.v_proj.bias.grad, ref_attn.v_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_attn.out_proj.bias.grad, ref_attn.out_proj.bias.grad, atol=atol, rtol=rtol)


class TorchNeighborhoodAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: int = 7,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        scale: float = None,
    ):
        super().__init__()

        self.attention = TorchNeighborhoodAttention(
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)

        attn_output = self.attention(normed_hidden_states)

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        output = hidden_states + attn_output

        return output


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_heads, kernel_size",
    [
        (2, 32, 128, 4, 7),
        (1, 32, 128, 8, 5),
        (2, 24, 96, 3, 9),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_fused_neighborhood_attention_layer_correctness(
    batch_size, seq_len, hidden_size, num_heads, kernel_size, bias, dtype, atol, rtol
):
    set_seed(42)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    ref_layer = (
        TorchNeighborhoodAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    liger_layer = (
        LigerFusedNeighborhoodAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        liger_layer.attention.q_proj.weight.copy_(ref_layer.attention.q_proj.weight)
        liger_layer.attention.k_proj.weight.copy_(ref_layer.attention.k_proj.weight)
        liger_layer.attention.v_proj.weight.copy_(ref_layer.attention.v_proj.weight)
        liger_layer.attention.out_proj.weight.copy_(ref_layer.attention.out_proj.weight)

        liger_layer.layer_norm.weight.copy_(ref_layer.layer_norm.weight)
        liger_layer.layer_norm.bias.copy_(ref_layer.layer_norm.bias)

        if bias:
            liger_layer.attention.q_proj.bias.copy_(ref_layer.attention.q_proj.bias)
            liger_layer.attention.k_proj.bias.copy_(ref_layer.attention.k_proj.bias)
            liger_layer.attention.v_proj.bias.copy_(ref_layer.attention.v_proj.bias)
            liger_layer.attention.out_proj.bias.copy_(ref_layer.attention.out_proj.bias)

    hidden_states1 = hidden_states.detach().clone().requires_grad_(True)
    hidden_states2 = hidden_states.detach().clone().requires_grad_(True)

    out1 = liger_layer(hidden_states1)
    out2 = ref_layer(hidden_states2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    loss1 = out1.sum()
    loss2 = out2.sum()
    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(hidden_states1.grad, hidden_states2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "hidden_size, num_heads, kernel_size",
    [
        (128, 8, 7),
        (256, 16, 5),
        (64, 4, 3),
    ],
)
def test_fused_neighborhood_attention_shapes(hidden_size, num_heads, kernel_size):
    batch_size, seq_len = 2, 32

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    attention = LigerFusedNeighborhoodAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_size=kernel_size,
    ).to(device)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape, f"Expected shape {hidden_states.shape}, got {output.shape}"

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_fused_neighborhood_attention_edge_cases():
    with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_heads"):
        LigerFusedNeighborhoodAttention(hidden_size=100, num_heads=7)

    with pytest.raises(ValueError, match="kernel_size .* must be odd"):
        LigerFusedNeighborhoodAttention(hidden_size=128, num_heads=8, kernel_size=6)

    with pytest.raises(ValueError, match="kernel_size .* must be positive"):
        LigerFusedNeighborhoodAttention(hidden_size=128, num_heads=8, kernel_size=0)

    with pytest.raises(ValueError, match="dilation .* must be positive"):
        LigerFusedNeighborhoodAttention(hidden_size=128, num_heads=8, dilation=0)

    attention = LigerFusedNeighborhoodAttention(hidden_size=64, num_heads=4).to(device)
    hidden_states = torch.randn(1, 16, 64, device=device)
    attention_mask = torch.ones(1, 16, device=device)

    with pytest.raises(NotImplementedError, match="Attention mask is not yet supported"):
        attention(hidden_states, attention_mask)


def test_fused_neighborhood_attention_deterministic():
    set_seed(42)

    batch_size, seq_len, hidden_size, num_heads = 2, 32, 128, 8

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    attention = LigerFusedNeighborhoodAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_size=7,
    ).to(device)

    output1 = attention(hidden_states)
    output2 = attention(hidden_states)

    assert torch.allclose(output1, output2, atol=1e-6, rtol=1e-6), "Results are not deterministic"


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_heads",
    [
        (1, 16, 64, 4),
        (2, 32, 128, 8),
        (1, 48, 192, 6),
    ],
)
def test_fused_neighborhood_attention_gradient_flow(batch_size, seq_len, hidden_size, num_heads):
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

    attention = LigerFusedNeighborhoodAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_size=7,
    ).to(device)

    output = attention(hidden_states)
    loss = output.sum()

    loss.backward()

    assert hidden_states.grad is not None, "Input gradients are None"
    assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad)), "Input gradients are zero"

    for name, param in attention.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Parameter {name} has zero gradient"


def torch_fused_neighborhood_attention(
    query,
    key,
    value,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
):
    batch_size, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    mask = torch.zeros(seq_len, seq_len, device=query.device, dtype=torch.bool)
    half_kernel = kernel_size // 2

    for i in range(seq_len):
        start = max(0, i - half_kernel * dilation)
        end = min(seq_len, i + half_kernel * dilation + 1)

        for j in range(start, end):
            if dilation == 1 or (j - i) % dilation == 0:
                mask[i, j] = True

    scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, value)

    return output


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim, kernel_size",
    [
        (2, 4, 32, 32, 7),
        (1, 8, 24, 16, 5),
        (2, 6, 16, 64, 9),
        (1, 2, 48, 128, 3),
    ],
)
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_liger_fused_neighborhood_attention_functional_correctness(
    batch_size, num_heads, seq_len, head_dim, kernel_size, dilation, dtype, atol, rtol
):
    set_seed(42)

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    query1 = query.detach().clone().requires_grad_(True)
    key1 = key.detach().clone().requires_grad_(True)
    value1 = value.detach().clone().requires_grad_(True)

    query2 = query.detach().clone().requires_grad_(True)
    key2 = key.detach().clone().requires_grad_(True)
    value2 = value.detach().clone().requires_grad_(True)

    liger_output = liger_fused_neighborhood_attention(query1, key1, value1, kernel_size=kernel_size, dilation=dilation)

    torch_output = torch_fused_neighborhood_attention(query2, key2, value2, kernel_size=kernel_size, dilation=dilation)

    assert_verbose_allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    liger_loss = liger_output.sum()
    torch_loss = torch_output.sum()

    liger_loss.backward()
    torch_loss.backward()

    assert_verbose_allclose(query1.grad, query2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(key1.grad, key2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(value1.grad, value2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 32, 32),
        (1, 8, 16, 64),
    ],
)
def test_liger_fused_neighborhood_attention_functional_custom_scale(batch_size, num_heads, seq_len, head_dim):
    set_seed(42)

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    custom_scale = 0.5

    query1 = query.detach().clone().requires_grad_(True)
    key1 = key.detach().clone().requires_grad_(True)
    value1 = value.detach().clone().requires_grad_(True)

    query2 = query.detach().clone().requires_grad_(True)
    key2 = key.detach().clone().requires_grad_(True)
    value2 = value.detach().clone().requires_grad_(True)

    liger_output = liger_fused_neighborhood_attention(
        query1, key1, value1, kernel_size=7, dilation=1, scale=custom_scale
    )

    torch_output = torch_fused_neighborhood_attention(
        query2, key2, value2, kernel_size=7, dilation=1, scale=custom_scale
    )

    assert_verbose_allclose(liger_output, torch_output, atol=5e-3, rtol=5e-3)


def test_liger_fused_neighborhood_attention_functional_shapes():
    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 32

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    output = liger_fused_neighborhood_attention(query, key, value)

    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_liger_fused_neighborhood_attention_functional_deterministic():
    """Test that the functional interface is deterministic."""
    set_seed(42)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 32

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    output1 = liger_fused_neighborhood_attention(query, key, value)
    output2 = liger_fused_neighborhood_attention(query, key, value)

    assert torch.allclose(output1, output2, atol=1e-6, rtol=1e-6), "Functional interface is not deterministic"


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_liger_fused_neighborhood_attention_functional_parameters(kernel_size, dilation):
    """Test the functional interface with different kernel sizes and dilations."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 24, 16

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    output = liger_fused_neighborhood_attention(query, key, value, kernel_size=kernel_size, dilation=dilation)

    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
