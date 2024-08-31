import os

import torch
import triton
from utils import _print_speed_banner, _test_memory, get_current_file_directory

from liger_kernel.ops.gemm_split_k_fp8_e4m3 import gemm_split_k


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["m", "k", "n"],
            x_vals=[
                (64, 64, 64),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
            ],
            xlabel="Matrix Size (m x k x n)",
            line_arg="provider",
            line_vals=["liger", "torch", "torch_compile"],
            line_names=["Liger", "PyTorch", "Torch Compile"],
            styles=[("blue", "solid"), ("orange", "solid"), ("green", "solid")],
            ylabel="time (ms)",
            plot_name="gemm-split-k-fp8-fwd-speed-benchmark",
            args={"mode": "forward", "dtype": torch.float32},
        ),
        triton.testing.Benchmark(
            x_names=["m", "k", "n"],
            x_vals=[
                (64, 64, 64),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
            ],
            xlabel="Matrix Size (m x k x n)",
            line_arg="provider",
            line_vals=["liger", "torch", "torch_compile"],
            line_names=["Liger", "PyTorch", "Torch Compile"],
            styles=[("blue", "solid"), ("orange", "solid"), ("green", "solid")],
            ylabel="time (ms)",
            plot_name="gemm-split-k-fp8-full-speed-benchmark",
            args={"mode": "full", "dtype": torch.float32},
        ),
    ]
)
def bench_speed_gemm_split_k_fp8(m, k, n, provider, mode, dtype, device="cuda"):
    a_fp8 = torch.randn((m, k), device=device, dtype=dtype).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn((k, n), device=device, dtype=dtype).to(torch.float8_e4m3fn)

    a_float = a_fp8.float().requires_grad_()
    b_float = b_fp8.float().requires_grad_()

    def fwd_liger():
        return gemm_split_k(a_fp8, b_fp8)

    def fwd_torch():
        return torch.matmul(a_float, b_float)

    fwd_torch_compiled = torch.compile(fwd_torch)

    if provider == "liger":
        fwd_fn = fwd_liger
    elif provider == "torch":
        fwd_fn = fwd_torch
    else:
        fwd_fn = fwd_torch_compiled

    quantiles = [0.5, 0.2, 0.8]

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(fwd_fn, quantiles=quantiles)
    elif mode == "full":

        def full():
            y = fwd_fn()
            if provider != "liger":
                y.backward(torch.ones_like(y))
            else:
                pass

        ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles)

    return ms, min_ms, max_ms


def benchmark_speed_gemm_split_k_fp8_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "gemm_split_k_fp8_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_gemm_split_k_fp8.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["m", "k", "n"],
            x_vals=[
                (64, 64, 64),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
            ],
            xlabel="Matrix Size (m x k x n)",
            line_arg="provider",
            line_vals=["liger", "torch", "torch_compile"],
            line_names=["Liger", "PyTorch", "Torch Compile"],
            styles=[("blue", "solid"), ("orange", "solid"), ("green", "solid")],
            ylabel="GPU memory usage (MB)",
            plot_name="gemm-split-k-fp8-memory-benchmark",
            args={"dtype": torch.float32},
        )
    ]
)
def bench_memory_gemm_split_k_fp8(m, k, n, provider, dtype, device="cuda"):
    a_fp8 = torch.randn((m, k), device=device, dtype=dtype).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn((k, n), device=device, dtype=dtype).to(torch.float8_e4m3fn)

    a_float = a_fp8.float().requires_grad_()
    b_float = b_fp8.float().requires_grad_()

    def full_liger():
        _ = gemm_split_k(a_fp8, b_fp8)

    def full_torch():
        y = torch.matmul(a_float, b_float)
        y.backward(torch.ones_like(y))

    full_torch_compiled = torch.compile(full_torch)

    if provider == "liger":
        full_fn = full_liger
    elif provider == "torch":
        full_fn = full_torch
    else:
        full_fn = full_torch_compiled

    mem = _test_memory(full_fn)
    return mem / 2**20


def benchmark_memory_gemm_split_k_fp8_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "gemm_split_k_fp8_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_gemm_split_k_fp8.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_gemm_split_k_fp8_wrapper()
    benchmark_memory_gemm_split_k_fp8_wrapper()
