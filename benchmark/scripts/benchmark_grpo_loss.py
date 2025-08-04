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
# Test the memory consumption of the linear fused GRPO loss
#############################################################################


def bench_memory_fused_linear_grpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    from test.chunked_loss.test_grpo_loss import LigerLMHeadGRPO
    from test.chunked_loss.test_grpo_loss import TorchLMHeadGRPO

    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    importance_sampling_level = input.extra_benchmark_config["importance_sampling_level"]
    provider = input.kernel_provider

    # Instantiate once and retrieve the first output only
    torch_lm_head_grpo = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
        device
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
        device
    )

    # Create inputs
    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    selected_token_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    attention_mask = torch.ones(B, T, device=device)
    advantages = torch.randn(B, dtype=dtype, device=device)
    ref_input = torch.randn(B, T, H, dtype=dtype, device=device)

    torch_fwd = lambda: torch_lm_head_grpo(_input, selected_token_ids, attention_mask, advantages, ref_input=ref_input)[
        0
    ]
    liger_fwd = lambda: liger_lm_head_grpo(_input, selected_token_ids, attention_mask, advantages, ref_input=ref_input)[
        0
    ]

    def fwd():
        if provider == "liger":
            return liger_fwd()
        elif provider == "torch":
            return torch_fwd()

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


#############################################################################
# Test the speed of the fused linear GRPO loss
#############################################################################


def bench_speed_fused_linear_grpo_loss(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    from test.chunked_loss.test_grpo_loss import LigerLMHeadGRPO
    from test.chunked_loss.test_grpo_loss import TorchLMHeadGRPO

    B = input.x
    T = input.extra_benchmark_config["T"]
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    importance_sampling_level = input.extra_benchmark_config["importance_sampling_level"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    # Instantiate once and retrieve the first output only
    torch_lm_head_grpo = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
        device
    )
    liger_lm_head_grpo = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
        device
    )

    # Create inputs
    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    selected_token_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    attention_mask = torch.ones(B, T, device=device)
    advantages = torch.randn(B, dtype=dtype, device=device)
    ref_input = torch.randn(B, T, H, dtype=dtype, device=device)

    torch_fwd = lambda: torch_lm_head_grpo(_input, selected_token_ids, attention_mask, advantages, ref_input=ref_input)[
        0
    ]
    liger_fwd = lambda: liger_lm_head_grpo(_input, selected_token_ids, attention_mask, advantages, ref_input=ref_input)[
        0
    ]

    def fwd():
        if provider == "liger":
            return liger_fwd()
        elif provider == "torch":
            return torch_fwd()

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

    # Benchmark token-level importance sampling (original GRPO)
    token_configs = {
        "kernel_name": "fused_linear_grpo_loss_token",
        "x_name": "B",
        "x_label": "B",
        "x_values": [2**i for i in range(1, 5)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "T": 1024,
                "H": 4096,
                "V": 128256,
                "importance_sampling_level": "token",
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }

    # Benchmark sequence-level importance sampling (GSPO)
    sequence_configs = {
        "kernel_name": "fused_linear_grpo_loss_sequence",
        "x_name": "B",
        "x_label": "B",
        "x_values": [2**i for i in range(1, 5)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "T": 1024,
                "H": 4096,
                "V": 128256,
                "importance_sampling_level": "sequence",
                "dtype": torch.bfloat16,
            }
        ],
        "overwrite": args.overwrite,
    }

    # Run benchmarks for token-level (GRPO)
    print("Benchmarking GRPO (token-level importance sampling)...")
    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_grpo_loss,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **token_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_grpo_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **token_configs,
    )

    # Run benchmarks for sequence-level (GSPO)
    print("Benchmarking GSPO (sequence-level importance sampling)...")
    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_grpo_loss,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **sequence_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_grpo_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **sequence_configs,
    )
