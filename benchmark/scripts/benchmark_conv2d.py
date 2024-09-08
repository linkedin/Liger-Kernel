import torch
import triton
from torch.nn import Conv2d
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.conv2d import LigerConv2d


def warmup_liger_conv2d(liger_conv2d, x):
    for _ in range(10):
        out = liger_conv2d(x)
        out.sum().backward()


def bench_speed_conv2d(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    C = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    N = input.extra_benchmark_config["N"]
    H = input.extra_benchmark_config["H"]
    W = input.extra_benchmark_config["W"]
    K = input.extra_benchmark_config["K"]
    R, S = input.extra_benchmark_config["kernel_size"]
    stride = input.extra_benchmark_config["stride"]
    padding = input.extra_benchmark_config["padding"]
    dilation = input.extra_benchmark_config["dilation"]
    dtype = input.extra_benchmark_config["dtype"]

    device = "cuda"

    torch_conv2d = (
        Conv2d(
            C, K, (R, S), stride=stride, padding=padding, dilation=dilation, bias=False
        )
        .to(device)
        .to(dtype)
    )
    liger_conv2d = (
        LigerConv2d(
            C, K, (R, S), stride=stride, padding=padding, dilation=dilation, bias=False
        )
        .to(device)
        .to(dtype)
    )

    x = torch.randn(N, C, H, W, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(K, C, R, S, dtype=dtype, device=device)

    torch_conv2d.weight.data = w.clone()
    liger_conv2d.weight.data = w.clone()

    # warmup
    if provider == "liger":
        warmup_liger_conv2d(liger_conv2d, x)

    def fwd():
        if provider == "liger":
            return liger_conv2d(x)
        else:
            return torch_conv2d(x)

    def full():
        output = fwd()
        output.backward(torch.randn_like(output))

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "full":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full, quantiles=QUANTILES, rep=100
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_conv2d(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    C = input.x
    provider = input.kernel_provider

    N = input.extra_benchmark_config["N"]
    H = input.extra_benchmark_config["H"]
    W = input.extra_benchmark_config["W"]
    K = input.extra_benchmark_config["K"]
    R, S = input.extra_benchmark_config["kernel_size"]
    stride = input.extra_benchmark_config["stride"]
    padding = input.extra_benchmark_config["padding"]
    dilation = input.extra_benchmark_config["dilation"]
    dtype = input.extra_benchmark_config["dtype"]

    device = "cuda"

    torch_conv2d = (
        Conv2d(
            C, K, (R, S), stride=stride, padding=padding, dilation=dilation, bias=False
        )
        .to(device)
        .to(dtype)
    )
    liger_conv2d = (
        LigerConv2d(
            C, K, (R, S), stride=stride, padding=padding, dilation=dilation, bias=False
        )
        .to(device)
        .to(dtype)
    )

    x = torch.randn(N, C, H, W, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(K, C, R, S, dtype=dtype, device=device)

    torch_conv2d.weight.data = w.clone()
    liger_conv2d.weight.data = w.clone()

    # warmup
    if provider == "liger":
        warmup_liger_conv2d(liger_conv2d, x)

    def fwd():
        if provider == "liger":
            return liger_conv2d(x)
        else:
            return torch_conv2d(x)

    def full():
        output = fwd()
        output.backward(torch.randn_like(output))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "conv2d",
        "x_name": "C",
        "x_label": "input channels",
        "x_values": [64, 128, 256, 512],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "N": 1,
                "H": 56,
                "W": 56,
                "K": 64,
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "dtype": torch.float16,
            },
            {
                "N": 1,
                "H": 112,
                "W": 112,
                "K": 128,
                "kernel_size": (5, 5),
                "stride": (2, 2),
                "padding": (2, 2),
                "dilation": (1, 1),
                "dtype": torch.float16,
            },
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_conv2d,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_conv2d,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
