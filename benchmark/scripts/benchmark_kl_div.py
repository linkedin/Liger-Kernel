import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from liger_kernel.utils import infer_device

device = infer_device()

S, E = 12, 18


def bench_speed_kldiv(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    reduction = "batchmean"
    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]
    torch_kl_div = nn.KLDivLoss(reduction=reduction)
    liger_kl_div = LigerKLDIVLoss(reduction=reduction)

    _input = torch.randn(B * T, V, requires_grad=True, device=device).log_softmax(dim=-1)
    target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_kl_div(_input, target)
        else:
            return torch_kl_div(_input, target)

    if input.kernel_operation_mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif input.kernel_operation_mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[_input],
            rep=100,
        )
    elif input.kernel_operation_mode == "full":

        def full():
            y = fwd()
            y.backward(retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_kldiv(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    reduction = "batchmean"
    torch_kl_div = nn.KLDivLoss(reduction=reduction)
    liger_kl_div = LigerKLDIVLoss(reduction=reduction)

    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device=device).log_softmax(dim=-1)
    target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_kl_div(_input, target)
        else:
            return torch_kl_div(_input, target)

    def full():
        y = fwd()
        y.backward(retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    common_args = {
        "kernel_name": "kl_div",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [{"B": 8, "T": 512}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_memory_kldiv,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_args,
    )

    run_benchmarks(
        bench_test_fn=bench_speed_kldiv,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_args,
    )
