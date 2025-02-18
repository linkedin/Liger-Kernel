import os
import sys

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def bench_memory_dpo_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    from test.chunked_loss.test_dpo_loss import LigerLMHeadDPO
    from test.chunked_loss.test_dpo_loss import TorchLMHeadDPO

    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    bias = input.extra_benchmark_config["bias"]
    beta = input.extra_benchmark_config["beta"]
    ignore_index = input.extra_benchmark_config["ignore_index"]
    provider = input.kernel_provider

    torch_dpo_loss = lambda x, ref_x, target: TorchLMHeadDPO(
        H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias
    ).to(device)(x, ref_x, target)[0]
    liger_dpo_loss = lambda x, ref_x, target: LigerLMHeadDPO(
        H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias
    ).to(device)(x, ref_x, target)[0]

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False)
    # Target shape: [B, T]
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    # Add ignore_index tokens to simulate padding
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    def fwd():
        if provider == "liger":
            return liger_dpo_loss(_input, ref_input, target)
        elif provider == "huggingface":
            return torch_dpo_loss(_input, ref_input, target)

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
    from test.chunked_loss.test_dpo_loss import LigerLMHeadDPO
    from test.chunked_loss.test_dpo_loss import TorchLMHeadDPO

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

    torch_dpo_loss = lambda x, ref_x, target: TorchLMHeadDPO(
        H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias
    ).to(device)(x, ref_x, target)[0]
    liger_dpo_loss = lambda x, ref_x, target: LigerLMHeadDPO(
        H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias
    ).to(device)(x, ref_x, target)[0]

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False)
    # Target shape: [B, T]
    target = torch.randint(V, (B, T), device=device, dtype=torch.long)

    # Add ignore_index tokens
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    def fwd():
        if provider == "liger":
            return liger_dpo_loss(_input, ref_input, target)
        elif provider == "huggingface":
            return torch_dpo_loss(_input, ref_input, target)

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
