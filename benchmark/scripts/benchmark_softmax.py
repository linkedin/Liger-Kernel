import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.softmax import LigerSoftmax
from liger_kernel.utils import infer_device

device = infer_device()


def bench_speed_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)
    liger_softmax = LigerSoftmax().to(device).to(dtype)
    torch_softmax = torch.nn.Softmax(dim=-1).to(device).to(dtype)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return liger_softmax(x)
        if provider == "torch":
            return torch_softmax(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(y_fwd, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=[x], rep=500)

    if any(val is None for val in (ms_20, ms_50, ms_80)):
        raise RuntimeError(f"Benchmark speed result is None: ms_20={ms_20}, ms_50={ms_50}, ms_80={ms_80}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    shape = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    dtype = extra_benchmark_config.get("dtype", torch.float32)

    torch_softmax = torch.nn.Softmax(dim=-1)
    liger_softmax = LigerSoftmax().to(device).to(dtype)

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    def fwd():
        if provider == "liger":
            return liger_softmax(x)
        elif provider == "torch":
            return torch_softmax(x)
        else:
            raise ValueError(f"Invalid provider: {provider} for softmax")

    def full():
        y = fwd()
        y.backward(torch.ones_like(y), retain_graph=True)

    if mode == "forward":
        mem_50, mem_20, mem_80 = _test_memory(fwd, quantiles=QUANTILES)
    elif mode == "backward":
        do = torch.ones_like(x)
        y = fwd()
        mem_50, mem_20, mem_80 = _test_memory(lambda: y.backward(do, retain_graph=True), quantiles=QUANTILES)
    else:
        mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    if any(val is None for val in (mem_20, mem_50, mem_80)):
        raise RuntimeError(f"Benchmark memory result is None: mem_20={mem_20}, mem_50={mem_50}, mem_80={mem_80}")

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = dict(
        kernel_name="softmax",
        x_name="N",
        x_label="hidden size",
        x_values=[128, 256, 512, 1024, 2048, 4096],
        kernel_providers=["liger", "torch"],
        extra_benchmark_configs=[
            {"M": 2048, "dtype": torch.float32},
            {"M": 2048, "dtype": torch.bfloat16},
        ],
    )

    run_benchmarks(
        bench_test_fn=bench_speed_softmax,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        overwrite=args.overwrite,
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_softmax,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        overwrite=args.overwrite,
        **common_configs,
    )
