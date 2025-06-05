import math

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.fused_neighborhood_attention import LigerFusedNeighborhoodAttention
from liger_kernel.utils import infer_device

device = infer_device()


class TorchNeighborhoodAttention(torch.nn.Module):
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

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
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


def bench_speed_fused_neighborhood_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["batch_size"]
    hidden_size = extra_benchmark_config["hidden_size"]
    num_heads = extra_benchmark_config["num_heads"]
    kernel_size = extra_benchmark_config["kernel_size"]
    dilation = extra_benchmark_config["dilation"]
    bias = extra_benchmark_config["bias"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, seq_len, hidden_size)

    liger_attn = (
        LigerFusedNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = (
        TorchNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        torch_attn.q_proj.weight.copy_(liger_attn.q_proj.weight)
        torch_attn.k_proj.weight.copy_(liger_attn.k_proj.weight)
        torch_attn.v_proj.weight.copy_(liger_attn.v_proj.weight)
        torch_attn.out_proj.weight.copy_(liger_attn.out_proj.weight)

        if bias:
            torch_attn.q_proj.bias.copy_(liger_attn.q_proj.bias)
            torch_attn.k_proj.bias.copy_(liger_attn.k_proj.bias)
            torch_attn.v_proj.bias.copy_(liger_attn.v_proj.bias)
            torch_attn.out_proj.bias.copy_(liger_attn.out_proj.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    print(f"Starting Warmup for input size: {x_shape}")
    _ = fwd()
    if mode in ("backward", "full"):
        y = _
        y.backward(dy, retain_graph=True)
    print("Done Warmup")

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=100, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_neighborhood_attention(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["batch_size"]
    hidden_size = extra_benchmark_config["hidden_size"]
    num_heads = extra_benchmark_config["num_heads"]
    kernel_size = extra_benchmark_config["kernel_size"]
    dilation = extra_benchmark_config["dilation"]
    bias = extra_benchmark_config["bias"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, seq_len, hidden_size)

    liger_attn = (
        LigerFusedNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = (
        TorchNeighborhoodAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            dropout=0.0,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        torch_attn.q_proj.weight.copy_(liger_attn.q_proj.weight)
        torch_attn.k_proj.weight.copy_(liger_attn.k_proj.weight)
        torch_attn.v_proj.weight.copy_(liger_attn.v_proj.weight)
        torch_attn.out_proj.weight.copy_(liger_attn.out_proj.weight)

        if bias:
            torch_attn.q_proj.bias.copy_(liger_attn.q_proj.bias)
            torch_attn.k_proj.bias.copy_(liger_attn.k_proj.bias)
            torch_attn.v_proj.bias.copy_(liger_attn.v_proj.bias)
            torch_attn.out_proj.bias.copy_(liger_attn.out_proj.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    def full():
        y = fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "fused_neighborhood_attention",
        "x_name": "seq_len",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(6, 13)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_heads": 8,
                "kernel_size": 7,
                "dilation": 1,
                "bias": True,
                "dtype": torch.float32,
            },
            {
                "batch_size": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "kernel_size": 7,
                "dilation": 1,
                "bias": True,
                "dtype": torch.float32,
            },
            {
                "batch_size": 2,
                "hidden_size": 1024,
                "num_heads": 16,
                "kernel_size": 9,
                "dilation": 1,
                "bias": True,
                "dtype": torch.float32,
            },
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_heads": 8,
                "kernel_size": 7,
                "dilation": 2,
                "bias": True,
                "dtype": torch.float32,
            },
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_heads": 8,
                "kernel_size": 7,
                "dilation": 1,
                "bias": True,
                "dtype": torch.bfloat16,
            },
            {
                "batch_size": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "kernel_size": 7,
                "dilation": 1,
                "bias": True,
                "dtype": torch.bfloat16,
            },
            {
                "batch_size": 2,
                "hidden_size": 1024,
                "num_heads": 16,
                "kernel_size": 9,
                "dilation": 1,
                "bias": True,
                "dtype": torch.bfloat16,
            },
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_heads": 8,
                "kernel_size": 7,
                "dilation": 2,
                "bias": True,
                "dtype": torch.bfloat16,
            },
        ],
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_neighborhood_attention,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_fused_neighborhood_attention,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
