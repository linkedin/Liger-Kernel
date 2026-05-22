"""
AttnRes Benchmark: Liger (Triton) vs PyTorch

Kimi Attention Residuals: softmax attention over depth blocks.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops import LigerAttnResFunction
from liger_kernel.utils import infer_device

device = infer_device()


def setup_attn_res(input: SingleBenchmarkRunInput):
    """Create input tensors for AttnRes from benchmark config."""
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

    # V: [N, B, T, D]
    V = torch.randn(
        cfg["N"],
        cfg["bsz"],
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    w_query = torch.randn(hidden_size, device=device, dtype=dtype) * 0.02
    w_norm = torch.ones(hidden_size, device=device, dtype=dtype)
    eps = cfg.get("eps", 1e-6)

    if input.kernel_provider == "liger":
        fn = lambda: LigerAttnResFunction.apply(V, w_query, w_norm, eps)
    elif input.kernel_provider == "pytorch":
        from test.transformers.test_attn_res import pytorch_attn_res

        fn = lambda: pytorch_attn_res(V, w_query, w_norm, eps)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider}")

    return V, lambda _: fn()


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="attn_res",
            setup_fn=setup_attn_res,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "N": 8,
                "bsz": 1,
                "eps": 1e-6,
            },
            probe_provider="pytorch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="attn_res",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_attn_res,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "N": 8,
                "bsz": 1,
                "eps": 1e-6,
            },
            probe_provider="pytorch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["pytorch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_attn_res),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_attn_res),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
