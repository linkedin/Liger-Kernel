import os
from typing import List

import torch
import triton
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import (
    _print_memory_banner,
    _print_speed_banner,
    _test_memory,
    get_current_file_directory,
)

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="gelu_pytorch_tanh",
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
                plot_name=f"geglu-{mode}-{target}-benchmark",
                args={"dtype": torch.bfloat16, "mode": mode},
            )
        )
    return perf_configs


@triton.testing.perf_report(
    _get_perf_configs(
        target="speed", ylabel="time (ms)", modes=["forward", "backward", "full"]
    )
)
def bench_speed_geglu(N, dtype, provider, mode="forward", device="cuda"):
    # llama 7b: (4096, 11008)
    bsz, seq_len, hidden_size = 4, N, 4096

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "liger":
        layer = LigerGEGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for GEGLU")

    def fwd():
        return layer(x)

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fwd, quantiles=quantiles, grad_to_none=[x], rep=10
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            rep=10,
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            full, quantiles=quantiles, grad_to_none=[x], rep=10
        )

    return ms, max_ms, min_ms


def benchmark_speed_geglu_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "geglu_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_geglu.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    benchmarks=_get_perf_configs(
        target="memory",
        ylabel="GPU memory usage (MB)",
        modes=["forward", "backward", "full"],
    )
)
def bench_memory_geglu(N, dtype, provider, mode="forward", device="cuda"):
    # llama 7b: (4096, 11008)
    bsz, seq_len, hidden_size = 4, N, 4096

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerGEGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for GEGLU")

    def fwd():
        return layer(x)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem = _test_memory(fwd)
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        mem = _test_memory(lambda: y.backward(do, retain_graph=True))
    else:
        mem = _test_memory(full)

    return mem / 2**20


def benchmark_memory_geglu_wrapper():
    _print_memory_banner()

    curr_dir = get_current_file_directory()
    output_dir = os.path.join(curr_dir, "geglu_memory")
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_geglu.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_geglu_wrapper()
    benchmark_memory_geglu_wrapper()
