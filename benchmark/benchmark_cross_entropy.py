import os

import torch
import triton
from torch.nn import CrossEntropyLoss
from utils import QUANTILES, _test_memory, get_current_file_directory

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(12, 18)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=[
                "Liger",
                "Hugging Face",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="cross-entropy-fwd-speed-benchmark",
            args={"B": 8, "T": 2048, "mode": "forward", "dtype": torch.bfloat16},
        ),
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(12, 18)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="cross-entropy-full-speed-benchmark",
            args={"B": 8, "T": 2048, "mode": "full", "dtype": torch.bfloat16},
        ),
    ]
)
def bench_speed_cross_entropy(B, T, V, provider, mode, dtype, device="cuda"):
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "backward":
        y = fwd()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[_input],
            rep=100,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    return ms, min_ms, max_ms


def benchmark_speed_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "cross_entropy_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_speed_cross_entropy.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(12, 18)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=[
                "Liger",
                "Hugging Face",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="GPU memory usage (MB)",
            plot_name="cross-entropy-memory-benchmark",
            args={"B": 8, "T": 2048, "dtype": torch.bfloat16},
        )
    ]
)
def bench_memory_cross_entropy(B, T, V, provider, dtype, device="cuda"):
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem, min_mem, max_mem = _test_memory(full, quantiles=QUANTILES)
    return (mem / 2**20, min_mem / 2**20, max_mem / 2**20)


def benchmark_memory_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "cross_entropy_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_memory_cross_entropy.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_cross_entropy_wrapper()
    benchmark_memory_cross_entropy_wrapper()
