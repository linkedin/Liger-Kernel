import os
import sys

import torch

from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _setup_dyt(input: SingleBenchmarkRunInput):
    """Create input tensor and DyT layer from benchmark config."""
    from test.transformers.test_dyt import LigerDyT
    from test.transformers.test_dyt import TorchDyT

    cfg = input.extra_benchmark_config
    hidden_size = input.x
    x = torch.randn(cfg["BT"], hidden_size, device=device, dtype=cfg["dtype"], requires_grad=True)
    if input.kernel_provider == "liger":
        layer = LigerDyT(hidden_size=hidden_size, beta=cfg["beta"]).to(device)
    elif input.kernel_provider == "torch":
        layer = TorchDyT(hidden_size=hidden_size, beta=cfg["beta"]).to(device)
    elif input.kernel_provider == "torch_compile":
        layer = torch.compile(TorchDyT(hidden_size=hidden_size, beta=cfg["beta"]).to(device))
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for DyT")
    return x, layer


def bench_speed_dyt(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_dyt(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_dyt(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_dyt(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    for beta in [False, True]:
        common_configs = {
            "kernel_name": f"dyt_beta={beta}",
            "x_name": "hidden_size",
            "x_label": "hidden_size",
            "x_values": [1024 * i for i in range(1, 17)],
            "kernel_providers": ["liger", "torch", "torch_compile"],
            "extra_benchmark_configs": [{"BT": 4096, "dtype": torch.bfloat16, "beta": beta}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_dyt,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_dyt,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
