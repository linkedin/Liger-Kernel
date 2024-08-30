import os

import torch
import triton
from utils import _print_memory_banner, _print_speed_banner, _test_memory

from liger_kernel.transformers.layer_norm import LigerLayerNorm


# NOTE: For torch compile, we will just use default inductor settings. No further customization
@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 15)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="layernorm-fwd-speed-benchmark",
            args={"M": 4096, "dtype": torch.float32, "mode": "forward"},
        ),
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 15)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="layernorm-full-speed-benchmark",
            args={"M": 4096, "dtype": torch.float32, "mode": "full"},
        ),
    ]
)
def bench_speed_layer_norm(M, N, dtype, provider, mode, eps=1e-6, device="cuda"):
    x_shape = (M, N)
    triton_ln = LigerLayerNorm(hidden_size=N).to("cuda")
    torch_ln = torch.nn.LayerNorm(N, eps=eps).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "liger":
            return triton_ln(x)
        if provider == "huggingface":
            return torch_ln(x)

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


def benchmark_speed_layer_norm_wrapper():
    _print_speed_banner()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = "layer_norm_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_layer_norm.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 15)],
            xlabel="hidden size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="GPU memory usage (MB)",
            plot_name="layernorm-full-memory-benchmark",
            args={"M": 4096, "dtype": torch.float32, "mode": "full"},
        )
    ]
)
def bench_memory_layer_norm(M, N, dtype, provider, mode, eps=1e-6, device="cuda"):
    x_shape = (M, N)

    triton_ln = LigerLayerNorm(hidden_size=N).to("cuda")
    torch_ln = torch.nn.LayerNorm(N, eps=eps).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_ln(x)
        if provider == "huggingface":
            return torch_ln(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem = _test_memory(full)

    return mem / 2**20


def benchmark_memory_layer_norm_wrapper():
    _print_memory_banner()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = "layer_norm_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_layer_norm.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_layer_norm_wrapper()
    benchmark_memory_layer_norm_wrapper()
