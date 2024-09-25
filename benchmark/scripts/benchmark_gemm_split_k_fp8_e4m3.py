import torch
import triton
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.ops.experimental.gemm_split_k_fp8_e4m3 import (
    LigerFP8GemmSplitKFunction,
)


def bench_speed_gemm_split_k_fp8(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    m, k, n = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    dtype = input.extra_benchmark_config["dtype"]
    device = "cuda"

    a_fp8 = torch.randn((m, k), device=device, dtype=dtype).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn((k, n), device=device, dtype=dtype).to(torch.float8_e4m3fn)

    a_float = a_fp8.float().requires_grad_()
    b_float = b_fp8.float().requires_grad_()

    def fwd_liger():
        return LigerFP8GemmSplitKFunction.apply(a_fp8, b_fp8)

    def fwd_torch():
        return torch.matmul(a_float, b_float)

    fwd_torch_compiled = torch.compile(fwd_torch)

    if provider == "liger":
        fwd_fn = fwd_liger
    elif provider == "torch":
        fwd_fn = fwd_torch
    else:
        fwd_fn = fwd_torch_compiled

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, quantiles=QUANTILES)
    elif mode == "full":

        def full():
            y = fwd_fn()
            if provider == "liger":
                dc = torch.ones_like(y, dtype=torch.float8_e4m3fn)
                LigerFP8GemmSplitKFunction.apply(dc, b_fp8.t())
                LigerFP8GemmSplitKFunction.apply(a_fp8.t(), dc)
            else:
                torch.sum(y).backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(y_50=ms_50, y_20=ms_20, y_80=ms_80)


def bench_memory_gemm_split_k_fp8(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    m, k, n = input.x
    provider = input.kernel_provider
    dtype = input.extra_benchmark_config["dtype"]
    device = "cuda"

    a_fp8 = torch.randn((m, k), device=device, dtype=dtype).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn((k, n), device=device, dtype=dtype).to(torch.float8_e4m3fn)

    a_float = a_fp8.float().requires_grad_()
    b_float = b_fp8.float().requires_grad_()

    def full_liger():
        y = LigerFP8GemmSplitKFunction.apply(a_fp8, b_fp8)
        dc = torch.ones_like(y, dtype=torch.float8_e4m3fn)
        LigerFP8GemmSplitKFunction.apply(dc, b_fp8.t())
        LigerFP8GemmSplitKFunction.apply(a_fp8.t(), dc)

    def full_torch():
        y = torch.matmul(a_float, b_float)
        torch.sum(y).backward()

    full_torch_compiled = torch.compile(full_torch)

    if provider == "liger":
        full_fn = full_liger
    elif provider == "torch":
        full_fn = full_torch
    else:
        full_fn = full_torch_compiled

    mem_50, mem_20, mem_80 = _test_memory(full_fn, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_50=mem_50, y_20=mem_20, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "gemm_split_k_fp8",
        "x_name": "Matrix Size (m x k x n)",
        "x_label": "Matrix Size (m x k x n)",
        "x_values": [
            (64, 64, 64),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (64, 128, 64),
            (256, 512, 256),
            (512, 1024, 512),
            (1024, 2048, 1024),
        ],
        "kernel_providers": ["liger", "torch", "torch_compile"],
        "extra_benchmark_configs": [{"dtype": torch.float32}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_gemm_split_k_fp8,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_gemm_split_k_fp8,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
