"""Compare repeated add and RMSNorm calls without a backward pass."""

import argparse

import torch

from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import run_benchmarks

from liger_kernel.transformers.functional import liger_fused_add_rms_norm
from liger_kernel.utils import infer_device


def torch_add_rms_norm(x, residual, weight):
    summed = x + residual
    summed_fp32 = summed.float()
    normalized = summed_fp32 * torch.rsqrt(summed_fp32.square().mean(-1, keepdim=True) + 1e-6)
    return normalized.to(x.dtype) * weight


def setup_inference(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    device = infer_device()
    x = torch.randn(cfg["tokens"], cfg["hidden_size"], dtype=cfg["dtype"], device=device)
    residual = torch.randn_like(x)
    weight = torch.ones(cfg["hidden_size"], dtype=x.dtype, device=device)
    steps = input.x

    if input.kernel_provider in ("torch", "torch_compile"):
        call = torch_add_rms_norm
    elif input.kernel_provider == "liger":

        def call(x, residual, weight):
            return liger_fused_add_rms_norm(x, residual, weight, 1e-6)[0]

    elif input.kernel_provider == "liger_output_only":

        def call(x, residual, weight):
            return liger_fused_add_rms_norm(x, residual, weight, 1e-6, return_residual=False)

    else:
        raise ValueError(f"Unknown provider: {input.kernel_provider}")

    def forward(x):
        for _ in range(steps):
            x = call(x, residual, weight)
        return x

    if input.kernel_provider == "torch_compile":
        forward = torch.compile(forward, fullgraph=True)

    with torch.no_grad():
        expected = x
        for _ in range(steps):
            expected = torch_add_rms_norm(expected, residual, weight)
        actual = forward(x)
        atol, rtol = (1e-4, 1e-5) if x.dtype == torch.float32 else (2e-2, 2e-2)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return x, forward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=5280)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or args.hidden_size < 1 or any(steps < 1 for steps in args.steps):
        parser.error("tokens, hidden-size, and steps must be positive")

    common = {
        "kernel_name": "fused_add_rms_norm_inference",
        "x_name": "steps",
        "x_label": "shared norm steps",
        "x_values": args.steps,
        "kernel_providers": ["torch", "torch_compile", "liger", "liger_output_only"],
        "kernel_operation_modes": ["no-grad-forward"],
        "extra_benchmark_configs": [
            {"tokens": args.tokens, "hidden_size": args.hidden_size, "dtype": getattr(torch, args.dtype)}
        ],
        "overwrite": args.overwrite,
    }
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_inference, warmup=100, rep=100),
        metric_name="speed",
        metric_unit="ms",
        **common,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_inference),
        metric_name="memory",
        metric_unit="MB",
        **common,
    )
