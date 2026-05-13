import os
import sys

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def setup_grpo_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and GRPO loss from benchmark config."""
    from test.chunked_loss.test_grpo_loss import LigerLMHeadGRPO
    from test.chunked_loss.test_grpo_loss import TorchLMHeadGRPO

    cfg = input.extra_benchmark_config
    T = cfg["T"]
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        H = model_cfg.hidden_size
        V = model_cfg.vocab_size
        dtype = model_cfg.dtype
        B = cfg["bsz"]
    else:
        B = input.x
        H = cfg["hidden_size"]
        V = cfg["vocab_size"]
        dtype = cfg["dtype"]

    importance_sampling_level = cfg["importance_sampling_level"]

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    selected_token_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    attention_mask = torch.ones(B, T, device=device)
    advantages = torch.randn(B, dtype=dtype, device=device)
    ref_input = torch.randn(B, T, H, dtype=dtype, device=device)

    if input.kernel_provider == "liger":
        loss_module = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
            device
        )
    elif input.kernel_provider == "torch":
        loss_module = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=importance_sampling_level).to(
            device
        )
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for GRPOLoss")

    fwd = lambda x: loss_module(x, selected_token_ids, attention_mask, advantages, ref_input=ref_input)[0]
    return _input, fwd


def _run_grpo_benchmarks(args, importance_sampling_level, kernel_name_suffix):
    """Run D1 or D2 benchmarks for a given importance_sampling_level."""
    kernel_name = f"fused_linear_grpo_loss_{kernel_name_suffix}"

    T = 1024

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name=kernel_name,
            setup_fn=setup_grpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "importance_sampling_level": importance_sampling_level,
            },
            probe_dim="B",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)

        common_configs = build_token_length_sweep(
            kernel_name=kernel_name,
            probe_x=1,
            model=model,
            setup_fn=setup_grpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "importance_sampling_level": importance_sampling_level,
            },
            scale_dim="B",
            probe_provider="torch",
            x_label="batch size",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_grpo_loss),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_grpo_loss),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    # Benchmark token-level importance sampling (original GRPO)
    print("Benchmarking GRPO (token-level importance sampling)...")
    _run_grpo_benchmarks(args, importance_sampling_level="token", kernel_name_suffix="token")

    # Benchmark sequence-level importance sampling (GSPO)
    print("Benchmarking GSPO (sequence-level importance sampling)...")
    _run_grpo_benchmarks(args, importance_sampling_level="sequence", kernel_name_suffix="sequence")
