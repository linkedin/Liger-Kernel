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


def setup_orpo_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and ORPO loss from benchmark config."""
    from test.chunked_loss.test_orpo_loss import LigerLMHeadORPO
    from test.chunked_loss.test_orpo_loss import TorchLMHeadORPO

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

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)
    nll_target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    if input.kernel_provider == "liger":
        loss_module = LigerLMHeadORPO(H=H, V=V, dtype=dtype).to(device)
    elif input.kernel_provider == "huggingface":
        loss_module = TorchLMHeadORPO(H=H, V=V, dtype=dtype).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for ORPOLoss")

    fwd = lambda x: loss_module(x, target, nll_target)[0]
    return _input, fwd


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    T = 1024

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="orpo_loss",
            setup_fn=setup_orpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={"T": T},
            probe_dim="B",
            probe_provider="huggingface",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)

        common_configs = build_token_length_sweep(
            kernel_name="orpo_loss",
            probe_x=1,
            model=model,
            setup_fn=setup_orpo_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={"T": T},
            scale_dim="B",
            x_label="batch size",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "huggingface"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_orpo_loss),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_orpo_loss),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
