"""Benchmark for Native Sparse Attention (arXiv 2502.11089).

This measures the pure-PyTorch NSA reference (provider ``liger``) against dense
full causal attention (provider ``torch``) as sequence length grows. PR-1 ships
the reference implementation only, so this establishes the benchmark harness and
a baseline; the Triton kernel added in follow-up work is measured against these
same numbers to demonstrate the sparse-attention speed/memory win.
"""

import math

import torch
import torch.nn as nn

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.native_sparse_attention import LigerNativeSparseAttention
from liger_kernel.utils import infer_device

device = infer_device()


class DenseCausalAttention(nn.Module):
    """Standard dense causal GQA attention — the O(S^2) baseline NSA replaces."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        group = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)).float() * self.scale
        causal = torch.arange(s, device=x.device).view(s, 1) >= torch.arange(s, device=x.device).view(1, s)
        scores = scores.masked_fill(~causal, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


def setup_native_sparse_attention(input: SingleBenchmarkRunInput):
    """Create input tensor and attention layer from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        seq_len = cfg["seq_len"]
        hidden_size = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        seq_len = input.x
        hidden_size = cfg["hidden_size"]
        dtype = cfg["dtype"]

    head_dim = cfg["head_dim"]
    batch = cfg["batch"]
    num_heads = max(1, hidden_size // head_dim)
    num_kv_heads = max(1, num_heads // cfg["gqa_group"])

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)

    if input.kernel_provider == "liger":
        layer = LigerNativeSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compress_block_size=cfg["compress_block_size"],
            compress_stride=cfg["compress_stride"],
            selection_block_size=cfg["selection_block_size"],
            num_selected_blocks=cfg["num_selected_blocks"],
            sliding_window_size=cfg["sliding_window_size"],
        ).to(device)
    elif input.kernel_provider == "torch":
        layer = DenseCausalAttention(hidden_size, num_heads, num_kv_heads, head_dim).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for native_sparse_attention")
    return x, layer.to(dtype)


_NSA_EXTRA = {
    "head_dim": 64,
    "gqa_group": 4,
    "batch": 1,
    "compress_block_size": 32,
    "compress_stride": 16,
    "selection_block_size": 64,
    "num_selected_blocks": 16,
    "sliding_window_size": 512,
}


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="native_sparse_attention",
            setup_fn=setup_native_sparse_attention,
            model_keys=["hidden_size", "dtype"],
            extra_configs=dict(_NSA_EXTRA),
            probe_dim="T",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="native_sparse_attention",
            probe_x=1024,
            model=model,
            setup_fn=setup_native_sparse_attention,
            model_keys=["hidden_size", "dtype"],
            extra_configs=dict(_NSA_EXTRA),
            scale_dim="T",
            x_label="Sequence length",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_native_sparse_attention),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_native_sparse_attention),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
