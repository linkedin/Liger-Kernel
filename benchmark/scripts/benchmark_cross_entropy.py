import torch
import triton

from torch.nn import CrossEntropyLoss
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()


def bench_memory_cross_entropy(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.kernel_provider
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device=device)
    target = torch.randint(V, (B * T, 1), device=device).squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def bench_speed_cross_entropy(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device=device)
    target = torch.randint(V, (B * T, 1), device=device).squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "cross_entropy",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [{"B": 8, "T": 2048}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_cross_entropy,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_cross_entropy,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
