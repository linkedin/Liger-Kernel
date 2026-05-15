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


def setup_dpo_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and DPO loss from benchmark config."""
    from test.chunked_loss.test_dpo_loss import LigerLMHeadDPO
    from test.chunked_loss.test_dpo_loss import TorchLMHeadDPO

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

    bias = cfg["bias"]
    beta = cfg["beta"]
    ignore_index = cfg["ignore_index"]

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False)
    # Target shape: [B, T]
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    # Add ignore_index tokens to simulate padding
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    if input.kernel_provider == "liger":
        loss_module = LigerLMHeadDPO(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)
    elif input.kernel_provider == "torch":
        loss_module = TorchLMHeadDPO(H=H, V=V, dtype=dtype, beta=beta, ignore_index=ignore_index, bias=bias).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for DPOLoss")

    fwd = lambda x: loss_module(x, ref_input, target)[0]
    return _input, fwd


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    T = 512

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="dpo_loss",
            setup_fn=setup_dpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "bias": True,
                "beta": 0.1,
                "ignore_index": 42,
            },
            probe_dim="B",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)

        common_configs = build_token_length_sweep(
            kernel_name="dpo_loss",
            probe_x=1,
            model=model,
            setup_fn=setup_dpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "bias": True,
                "beta": 0.1,
                "ignore_index": 42,
            },
            scale_dim="B",
            probe_provider="torch",
            x_label="batch size",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_dpo_loss),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_dpo_loss),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
