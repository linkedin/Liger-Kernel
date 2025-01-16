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


#############################################################################
# Test the memory consumption of the linear fused cross entropy loss
#############################################################################


def bench_memory_fused_linear_orpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    from test.chunked_loss.test_orpo_loss import LigerLMHeadORPO
    from test.chunked_loss.test_orpo_loss import TorchLMHeadORPO

    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    torch_lm_head_orpo = lambda x, target: TorchLMHeadORPO(H=H, V=V, dtype=dtype).to(device)(x, target)[0]
    liger_lm_head_orpo = lambda x, target: LigerLMHeadORPO(H=H, V=V, dtype=dtype).to(device)(x, target)[0]

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)
    nll_target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_orpo(_input, target, nll_target)
        elif provider == "huggingface":
            return torch_lm_head_orpo(_input, target, nll_target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


# #############################################################################
# # Test the speed of the fused linear cross entropy loss
# #############################################################################


def bench_speed_fused_linear_orpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    from test.chunked_loss.test_orpo_loss import LigerLMHeadORPO
    from test.chunked_loss.test_orpo_loss import TorchLMHeadORPO

    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    torch_lm_head_orpo = lambda x, target: TorchLMHeadORPO(H=H, V=V, dtype=dtype).to(device)(x, target)[0]
    liger_lm_head_orpo = lambda x, target: LigerLMHeadORPO(H=H, V=V, dtype=dtype).to(device)(x, target)[0]

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)
    nll_target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_orpo(_input, target, nll_target)
        elif provider == "huggingface":
            return torch_lm_head_orpo(_input, target, nll_target)

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
        "kernel_name": "fused_linear_orpo_loss",
        "x_name": "B",
        "x_label": "B",
        "x_values": [2**i for i in range(1, 5)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "T": 1024,
                "H": 4096,
                "V": 128256,
                "mode": "forward",
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_orpo_loss,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_orpo_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
