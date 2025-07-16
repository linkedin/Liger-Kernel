import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()


class NaiveAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Naive implementation of the add residual rms norm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


class AddLigerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        AddLigerRMSNorm is equivalent to NaiveAddRMSNorm class above, but uses the LigerRMSNorm kernel.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.rms_norm = LigerRMSNorm(hidden_size, eps, in_place=False)

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


def bench_speed_fused_residual_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    # Fused Add RMS Norm
    fused_add_rms_norm = LigerFusedAddRMSNorm(hidden_size=N, eps=eps).to(device)
    # Naive implementation
    naive_rms_norm = NaiveAddRMSNorm(hidden_size=N, eps=eps).to(device)
    # LigerRMSNorm without fused residual addition
    liger_rms_norm = AddLigerRMSNorm(hidden_size=N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    r = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)
    x.requires_grad_(True)
    r.requires_grad_(True)
    # utility functions

    def y_fwd():
        if provider == "liger_fused_add_rms_norm":
            return fused_add_rms_norm(x, r)

        if provider == "huggingface":
            return naive_rms_norm(x, r)

        if provider == "liger_rms_norm":
            return liger_rms_norm(x, r)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y, s = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: (torch.autograd.backward((y, s), (dy, ds), retain_graph=True)),
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y, s = y_fwd()
            torch.autograd.backward((y, s), (dy, ds))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_residual_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    fused_add_rms_norm = LigerFusedAddRMSNorm(hidden_size=N, eps=eps).to(device)
    naive_rms_norm = NaiveAddRMSNorm(hidden_size=N, eps=eps).to(device)
    liger_rms_norm = AddLigerRMSNorm(hidden_size=N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    r = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)
    x.requires_grad_(True)
    r.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger_fused_add_rms_norm":
            return fused_add_rms_norm(x, r)
        if provider == "huggingface":
            return naive_rms_norm(x, r)
        if provider == "liger_rms_norm":
            return liger_rms_norm(x, r)

    def full():
        y, s = y_fwd()
        torch.autograd.backward((y, s), (dy, ds))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "fused_add_rms_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger_fused_add_rms_norm", "huggingface", "liger_rms_norm"],
        "extra_benchmark_configs": [{"M": 2048, "dtype": torch.float32, "eps": 1e-6}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_residual_rms_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_residual_rms_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
