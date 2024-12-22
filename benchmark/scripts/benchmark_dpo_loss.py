import torch
import triton

from test.chunked_loss.test_dpo_loss import HF_DPO_Loss
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
from liger_kernel.utils import infer_device

device = infer_device()


class TorchDPOLoss(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        beta: float = 0.1,
        ignore_index: int = -100,
        bias: bool = False,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.dpo_loss = HF_DPO_Loss(beta=beta, ignore_index=ignore_index)

    def forward(self, x, target):
        return self.dpo_loss.get_batch_loss_metrics(
            x,
            self.lin.weight,
            target,
            self.lin.bias if hasattr(self.lin, "bias") else None,
        )


class LigerDPOLoss(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        beta: float = 0.1,
        ignore_index: int = -100,
        bias: bool = False,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, x, target):
        return LigerFusedLinearDPOFunction.apply(
            x,
            self.lin.weight,
            target,
            self.lin.bias if hasattr(self.lin, "bias") else None,
            self.ignore_index,
            self.beta,
            True,
        )


def bench_memory_dpo_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    bias = input.extra_benchmark_config["bias"]
    beta = input.extra_benchmark_config["beta"]
    ignore_index = input.extra_benchmark_config["ignore_index"]
    provider = input.kernel_provider

    torch_dpo_loss = TorchDPOLoss(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)
    liger_dpo_loss = LigerDPOLoss(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    # Target shape: [B, T]
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    # Add ignore_index tokens to simulate padding
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    def fwd():
        if provider == "liger":
            return liger_dpo_loss(_input, target)
        elif provider == "huggingface":
            return torch_dpo_loss(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def bench_speed_dpo_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    bias = input.extra_benchmark_config["bias"]
    beta = input.extra_benchmark_config["beta"]
    ignore_index = input.extra_benchmark_config["ignore_index"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    torch_dpo_loss = TorchDPOLoss(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)
    liger_dpo_loss = LigerDPOLoss(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)

    # Target shape: [B, T]
    target = torch.randint(V, (B, T), device=device, dtype=torch.long)

    # Add ignore_index tokens
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    def fwd():
        if provider == "liger":
            return liger_dpo_loss(_input, target)
        elif provider == "huggingface":
            return torch_dpo_loss(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
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

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "dpo_loss",
        "x_name": "B",
        "x_label": "Batch Size (B)",
        "x_values": [2**i for i in range(1, 6)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "T": 512,
                "H": 1024,
                "V": 128256,
                "mode": "forward",
                "dtype": torch.bfloat16,
                "bias": True,
                "beta": 0.1,
                "ignore_index": 42,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_dpo_loss,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_dpo_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
