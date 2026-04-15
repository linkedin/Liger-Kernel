import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()


def bench_speed_group_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    C = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    H = extra_benchmark_config["H"]
    channels_per_group = extra_benchmark_config["channels_per_group"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, C, H)
    triton_ln = LigerGroupNorm(num_channels=C, num_groups=C // channels_per_group, eps=eps).to(device)
    torch_ln = torch.nn.GroupNorm(num_groups=C // channels_per_group, num_channels=C, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_ln(x)
        if provider == "huggingface":
            return torch_ln(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(y_fwd, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=[x], rep=500)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_group_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    C = input.x
    provider = input.kernel_provider
    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    H = extra_benchmark_config["H"]
    channels_per_group = extra_benchmark_config["channels_per_group"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, C, H)
    triton_ln = LigerGroupNorm(num_channels=C, num_groups=C // channels_per_group, eps=eps).to(device)
    torch_ln = torch.nn.GroupNorm(num_groups=C // channels_per_group, num_channels=C, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_ln(x)
        if provider == "huggingface":
            return torch_ln(x)

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
        "kernel_name": "group_norm",
        "x_name": "C",
        "x_label": "num_channels",
        "x_values": [2**i for i in range(5, 12)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "M": 128,
                "H": 512,
                "channels_per_group": 4,
                "dtype": torch.float32,
                "eps": 1e-6,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_group_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_group_norm,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
