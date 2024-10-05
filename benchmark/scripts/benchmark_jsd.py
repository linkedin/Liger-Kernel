import torch
import torch.nn as nn
import triton
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.jsd import LigerJSD


class TorchJSD(nn.Module):
    def __init__(self, beta: float = 0.5, dtype: torch.dtype = torch.float):
        super(TorchJSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.beta = beta
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.tensor,  # input
        log_p: torch.tensor,  # target
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_p), torch.exp(log_q), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p) + (1 - self.beta) * self.kl(
            torch.log(m), log_q
        )
        return loss.to(self.dtype)


def bench_speed_jsd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]
    torch_jsd = TorchJSD()
    liger_jsd = LigerJSD()

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda").log_softmax(
        dim=-1
    )
    target = torch.randn(B * T, V, device="cuda").log_softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_jsd(_input, target)
        else:
            return torch_jsd(_input, target)

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

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full, quantiles=QUANTILES, rep=100
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_jsd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    torch_jsd = TorchJSD()
    liger_jsd = LigerJSD()

    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda").log_softmax(
        dim=-1
    )
    target = torch.randn(B * T, V, device="cuda").log_softmax(dim=-1)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_jsd(_input, target)
        else:
            return torch_jsd(_input, target)

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
        "kernel_name": "jsd",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [{"B": 4, "T": 2048}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_memory_jsd,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_args,
    )

    run_benchmarks(
        bench_test_fn=bench_speed_jsd,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_args,
    )
