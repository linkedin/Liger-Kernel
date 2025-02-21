import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.tvd import LigerTVDLoss


class TorchTVDLoss(torch.nn.Module):
    def __init__(self, reduction="batchmean"):
        super(TorchTVDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        tvd = torch.abs(p - q) / 2.0
        if self.reduction == "mean":
            return torch.sum(tvd) / (p.size(0) * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / p.size(0)
        else:
            raise ValueError("Invalid reduction type.")


S, E = 12, 18


def bench_speed_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    reduction = "batchmean"
    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]
    torch_tvd = TorchTVDLoss(reduction=reduction)
    liger_tvd = LigerTVDLoss(reduction=reduction)

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda").softmax(dim=-1)
    target = torch.randn(B * T, V, device="cuda").softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_tvd(_input, target)
        else:
            return torch_tvd(_input, target)

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


def bench_memory_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    reduction = "batchmean"
    torch_tvd = TorchTVDLoss(reduction=reduction)
    liger_tvd = LigerTVDLoss(reduction=reduction)

    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda").softmax(dim=-1)
    target = torch.randn(B * T, V, device="cuda").softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_tvd(_input, target)
        else:
            return torch_tvd(_input, target)

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
        "kernel_name": "tvd",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [{"B": 8, "T": 2048}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_memory_tvd,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_args,
    )

    run_benchmarks(
        bench_test_fn=bench_speed_tvd,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_args,
    )
