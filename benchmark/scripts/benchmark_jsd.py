import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.jsd import LigerJSD
from liger_kernel.utils import infer_device

device = infer_device()


class TorchJSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label=None,
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (1 - self.beta) * self.kl(
            torch.log(m), log_q
        ).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
        return loss.to(self.dtype)


def bench_speed_jsd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    B, T = input.extra_benchmark_config["B"], input.extra_benchmark_config["T"]
    torch_jsd = TorchJSD()
    liger_jsd = LigerJSD()

    _input = torch.randn(B * T, V, requires_grad=True, device=device).log_softmax(dim=-1)
    target = torch.randn(B * T, V, device=device).log_softmax(dim=-1)

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

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
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

    _input = torch.randn(B * T, V, requires_grad=True, device=device).log_softmax(dim=-1)
    target = torch.randn(B * T, V, device=device).log_softmax(dim=-1)

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
