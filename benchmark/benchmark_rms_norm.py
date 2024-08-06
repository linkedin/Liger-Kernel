import os

import torch
import torch.nn as nn
import triton
from utils import _print_memory_banner, _print_speed_banner, _test_memory

from liger_kernel.transformers.rms_norm import LigerRMSNorm


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 16)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="rmsnorm-fwd-speed-benchmark",
            args={"M": 2048, "dtype": torch.bfloat16, "mode": "forward"},
        ),
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 16)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="rmsnorm-bwd-speed-benchmark",
            args={"M": 2048, "dtype": torch.bfloat16, "mode": "backward"},
        ),
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 16)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="rmsnorm-full-speed-benchmark",
            args={"M": 2048, "dtype": torch.bfloat16, "mode": "full"},
        ),
    ]
)
def bench_speed_rms_norm(M, N, dtype, provider, mode, eps=1e-5, device="cuda"):
    x_shape = (M, N)

    triton_rms = LigerRMSNorm(hidden_size=N).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=N).to("cuda")

    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    # utility functions

    def y_fwd():
        if provider == "liger":
            return triton_rms(x)

        if provider == "huggingface":

            return llama_rms(x)

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, grad_to_none=[x], rep=500
        )
    elif mode == "backward":
        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            full, quantiles=quantiles, grad_to_none=[x], rep=500
        )

    return ms, max_ms, min_ms


def benchmark_speed_rms_norm_wrapper():
    _print_speed_banner()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = "rms_norm_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_rms_norm.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 16)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="GPU memory usage (MB)",
            plot_name="rmsnorm-full-memory-benchmark",
            args={"M": 2048, "dtype": torch.bfloat16, "mode": "full"},
        )
    ]
)
def bench_memory_rms_norm(M, N, dtype, provider, mode, eps=1e-5, device="cuda"):
    x_shape = (M, N)

    triton_rms = LigerRMSNorm(hidden_size=N).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=N).to("cuda")

    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger":
            return triton_rms(x)
        if provider == "huggingface":
            return llama_rms(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem = _test_memory(full)

    return mem / 2**20


def benchmark_memory_rms_norm_wrapper():
    _print_memory_banner()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = "rms_norm_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # TODO: make precision configurable in generated csv
    bench_memory_rms_norm.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_rms_norm_wrapper()
    benchmark_memory_rms_norm_wrapper()
