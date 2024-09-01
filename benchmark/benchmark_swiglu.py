import os
from typing import List

import torch
import triton
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import (
    QUANTILES,
    _print_memory_banner,
    _print_speed_banner,
    _test_memory,
    get_current_file_directory,
)

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="silu",
)
SLEEP_SECONDS = 0.1


def _get_perf_configs(target: str, ylabel: str, modes: List[str] = ["full"]):
    perf_configs = []
    for mode in modes:
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=[2**i for i in range(10, 14)],
                xlabel="Seq Length",
                line_arg="provider",
                line_vals=["liger", "huggingface"],
                line_names=["Liger", "Hugging Face"],
                styles=[("blue", "solid"), ("orange", "solid")],
                ylabel=ylabel,
                plot_name=f"swiglu-{mode}-{target}-benchmark",
                args={"dtype": torch.bfloat16, "mode": mode},
            )
        )
    return perf_configs


@triton.testing.perf_report(
    _get_perf_configs(
        target="speed", ylabel="time (ms)", modes=["forward", "backward", "full"]
    )
)
def bench_speed_swiglu(N, dtype, provider, mode="forward", device="cuda"):
    # llama 7b: (4096, 11008)
    bsz, seq_len, hidden_size = 4, N, 4096

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fwd, quantiles=QUANTILES, grad_to_none=[x], rep=10
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=10,
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            full, quantiles=QUANTILES, grad_to_none=[x], rep=10
        )

    return ms, min_ms, max_ms


def benchmark_speed_swiglu_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "swiglu_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_swiglu.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    benchmarks=_get_perf_configs(
        target="memory",
        ylabel="GPU memory usage (MB)",
        modes=["forward", "backward", "full"],
    )
)
def bench_memory_swiglu(N, dtype, provider, mode="forward", device="cuda"):
    # llama 7b: (4096, 11008)
    bsz, seq_len, hidden_size = 4, N, 4096

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem, min_mem, max_mem = _test_memory(fwd, quantiles=QUANTILES)
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        mem, min_mem, max_mem = _test_memory(
            lambda: y.backward(do, retain_graph=True), quantiles=QUANTILES
        )
    else:
        mem, min_mem, max_mem = _test_memory(full, quantiles=QUANTILES)

    return (mem / 2**20, min_mem / 2**20, max_mem / 2**20)


def benchmark_memory_swiglu_wrapper():
    _print_memory_banner()

    curr_dir = get_current_file_directory()
    output_dir = os.path.join(curr_dir, "swiglu_memory")
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_swiglu.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_swiglu_wrapper()
    benchmark_memory_swiglu_wrapper()
