import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.poly_norm import LigerPolyNorm
from liger_kernel.utils import infer_device

device = infer_device()


class NaivePolyNorm(nn.Module):
    """
    Naive PyTorch implementation of PolyNorm.

    Reference:
        https://github.com/BryceZhuo/PolyCom/

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        # Align with PolyCom reference: (1/3, 1/3, 1/3) and bias=1.0
        self.weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.variance_epsilon = eps

    def _norm(self, x):
        """RMSNorm operation"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, hidden_states):
        """
        Forward pass of PolyNorm

        Args:
            hidden_states: input tensor of shape (..., H)

        Returns:
            output tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute powers
        x_pow3 = hidden_states**3
        x_pow2 = hidden_states**2
        x_pow1 = hidden_states**1

        # Normalize each power
        norm_x3 = self._norm(x_pow3)
        norm_x2 = self._norm(x_pow2)
        norm_x1 = self._norm(x_pow1)

        # Weighted sum with bias
        output = self.weight[0] * norm_x3 + self.weight[1] * norm_x2 + self.weight[2] * norm_x1 + self.bias

        return output.to(input_dtype)


def bench_speed_poly_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    triton_poly = LigerPolyNorm(eps=eps).to(device)
    naive_poly = NaivePolyNorm(eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions

    def y_fwd():
        if provider == "liger":
            return triton_poly(x)

        if provider == "huggingface":
            return naive_poly(x)

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


def bench_memory_poly_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    triton_poly = LigerPolyNorm(eps=eps).to(device)
    naive_poly = NaivePolyNorm(eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger":
            return triton_poly(x)
        if provider == "huggingface":
            return naive_poly(x)

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
        "kernel_name": "poly_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [{"M": 2048, "dtype": torch.bfloat16, "eps": 1e-6}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_poly_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_poly_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
