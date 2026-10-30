import argparse
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

# "none_sum" / "none_mean" reduce the per-token losses of reduction="none" outside the loss function.
# With reduction="none" the backward scales the saved gradient in place for either reduction.
REDUCTIONS = ("mean", "sum", "none_sum", "none_mean")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def add_cross_entropy_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=REDUCTIONS,
        help="Loss reduction to benchmark. 'none_*' uses reduction='none' followed by .sum()/.mean().",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=tuple(DTYPES),
        help="Dtype of the logits.",
    )


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

    reduction = cfg.get("reduction", "mean")
    dtype = DTYPES[cfg.get("dtype", "float32")]
    loss_reduction, _, post_reduction = reduction.partition("_")
    if post_reduction == "":
        post_reduction = None

    _input = torch.randn(BT, V, requires_grad=True, device=device, dtype=dtype)
    target = torch.randint(V, (BT,), device=device)

    if input.kernel_provider == "liger":
        loss_fn = LigerCrossEntropyLoss(reduction=loss_reduction)
    elif input.kernel_provider == "torch":
        loss_fn = CrossEntropyLoss(reduction=loss_reduction)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for CrossEntropy")

    if post_reduction is None:
        return _input, lambda x: loss_fn(x, target)
    return _input, lambda x: getattr(loss_fn(x, target), post_reduction)()


if __name__ == "__main__":
    args = parse_benchmark_script_args(add_cross_entropy_arguments)

    # Only non-default values are added so existing benchmark data keeps its config identity.
    extra_configs = {}
    if args.reduction != "mean":
        extra_configs["reduction"] = args.reduction
    if args.dtype != "float32":
        extra_configs["dtype"] = args.dtype

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="cross_entropy",
            setup_fn=setup_cross_entropy,
            model_keys=["vocab_size"],
            extra_configs=extra_configs,
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
            extra_configs=extra_configs,
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
