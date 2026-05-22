import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from torch.nn import CrossEntropyLoss
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()


def setup_cross_entropy(input: SingleBenchmarkRunInput):
    """Create input tensor, target, and CE loss from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        V = model_cfg.vocab_size
        BT = cfg["bsz"] * cfg["seq_len"]
    else:
        BT = input.x
        V = cfg["vocab_size"]

    _input = torch.randn(BT, V, requires_grad=True, device=device)
    target = torch.randint(V, (BT,), device=device)

    if input.kernel_provider == "liger":
        loss_fn = LigerCrossEntropyLoss()
    elif input.kernel_provider == "torch":
        loss_fn = CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for CrossEntropy")

    return _input, lambda x: loss_fn(x, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="cross_entropy",
            setup_fn=setup_cross_entropy,
            model_keys=["vocab_size"],
            extra_configs={},
            probe_dim="BT",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)

        common_configs = build_token_length_sweep(
            kernel_name="cross_entropy",
            probe_x=1024,
            model=model,
            setup_fn=setup_cross_entropy,
            model_keys=["vocab_size"],
            extra_configs={},
            scale_dim="BT",
            probe_provider="torch",
            x_label="B * T",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_cross_entropy),
        kernel_operation_modes=["forward", "backward", "full", "no-grad-forward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_cross_entropy),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
