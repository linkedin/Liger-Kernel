import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.sparsemax import LigerSparsemax
from liger_kernel.utils import infer_device

device = infer_device()


def torch_sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dims = input_tensor.dim()
    if dim < 0:
        dim = input_dims + dim
    input_sorted, _ = torch.sort(input_tensor, dim=dim, descending=True)
    cumsum_input = torch.cumsum(input_sorted, dim=dim)
    input_size = input_tensor.size(dim)
    range_tensor = torch.arange(1, input_size + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [1] * input_dims
    shape[dim] = input_size
    range_tensor = range_tensor.view(shape)
    k_bound = 1 + range_tensor * input_sorted
    support = k_bound > cumsum_input
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (input_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(input_tensor - tau, min=0)


class TorchSparsemax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_sparsemax(x, dim=self.dim)


def bench_speed_sparsemax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    B = extra_benchmark_config["B"]
    T = extra_benchmark_config["T"]
    dim = extra_benchmark_config["dim"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (B * T, V)

    torch_sparsemax_module = TorchSparsemax(dim=dim).to(device)
    liger_sparsemax_module = LigerSparsemax(dim=dim).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger":
            return liger_sparsemax_module(x)
        elif provider == "torch":
            return torch_sparsemax_module(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_sparsemax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    B = extra_benchmark_config["B"]
    T = extra_benchmark_config["T"]
    dim = extra_benchmark_config["dim"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (B * T, V)

    torch_sparsemax_module = TorchSparsemax(dim=dim).to(device)
    liger_sparsemax_module = LigerSparsemax(dim=dim).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger":
            return liger_sparsemax_module(x)
        elif provider == "torch":
            return torch_sparsemax_module(x)

    def full():
        y = y_fwd()
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
        "kernel_name": "sparsemax",
        "x_name": "V",
        "x_label": "feature size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [{"B": 4, "T": 512, "dim": -1, "dtype": torch.float32}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_sparsemax,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_sparsemax,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
