import os
import sys

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def bench_speed_dyt(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    from test.transformers.test_dyt import LigerDyT
    from test.transformers.test_dyt import TorchDyT

    BT = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (BT, hidden_size)
    torch_dyt = TorchDyT(hidden_size=hidden_size).to(device)
    torch_compile_dyt = torch.compile(TorchDyT(hidden_size=hidden_size).to(device))
    triton_dyt = LigerDyT(hidden_size=hidden_size).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return triton_dyt(x)
        elif provider == "torch":
            return torch_dyt(x)
        elif provider == "torch_compile":
            return torch_compile_dyt(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(dy)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=[x], rep=500)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_dyt(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    from test.transformers.test_dyt import LigerDyT
    from test.transformers.test_dyt import TorchDyT

    BT = input.x
    provider = input.kernel_provider
    extra_benchmark_config = input.extra_benchmark_config
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (BT, hidden_size)
    torch_dyt = TorchDyT(hidden_size=hidden_size).to(device)
    torch_compile_dyt = torch.compile(TorchDyT(hidden_size=hidden_size).to(device))
    triton_dyt = LigerDyT(hidden_size=hidden_size).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return triton_dyt(x)
        elif provider == "torch":
            return torch_dyt(x)
        elif provider == "torch_compile":
            return torch_compile_dyt(x)

    def full():
        y = fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "dyt",
        "x_name": "BT",
        "x_label": "batch_size * seq_len",
        "x_values": [2**i for i in range(10, 15)],
        "kernel_providers": ["liger", "torch", "torch_compile"],
        "extra_benchmark_configs": [{"hidden_size": 4096, "dtype": torch.float32}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_dyt,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_dyt,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
