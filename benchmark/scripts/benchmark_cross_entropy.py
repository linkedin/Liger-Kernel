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
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import build_memory_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_speed_benchmark

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()

# "none_sum" / "none_mean" reduce the per-token losses of reduction="none" outside the loss function.
# --in-place makes the Liger reduction="none" backward scale the saved gradient in place.
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
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Pass in_place=True to LigerCrossEntropyLoss (Triton backend only). Requires --reduction none_sum or none_mean.",
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
        loss_fn = LigerCrossEntropyLoss(reduction=loss_reduction, in_place=cfg.get("in_place", False))
    elif input.kernel_provider == "torch":
        loss_fn = CrossEntropyLoss(reduction=loss_reduction)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for CrossEntropy")

    if post_reduction is None:
        return _input, lambda x: loss_fn(x, target)
    return _input, lambda x: getattr(loss_fn(x, target), post_reduction)()


def _time_fresh_backward(_input, fwd_fn, warmup, rep):
    """Times y.backward(do) in ms on a fresh graph per call, sized like triton.testing.do_bench."""
    torch_device = getattr(torch, device)
    cache = torch.empty(int(256e6), dtype=torch.int8, device=device)
    do = None

    def one_call():
        nonlocal do
        _input.grad = None
        y = fwd_fn()
        if do is None:
            do = torch.randn_like(y)
        cache.zero_()
        start, end = torch_device.Event(enable_timing=True), torch_device.Event(enable_timing=True)
        start.record()
        y.backward(do)
        end.record()
        torch_device.synchronize()
        return start.elapsed_time(end)

    one_call()  # discarded: includes the Triton compile
    estimate_ms = sum(one_call() for _ in range(5)) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    for _ in range(n_warmup):
        one_call()
    times = torch.tensor([one_call() for _ in range(n_repeat)], dtype=torch.float)
    return torch.quantile(times, torch.tensor(QUANTILES, dtype=torch.float)).tolist()


def bench_speed_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, fwd_fn = setup_cross_entropy(input)
    if input.kernel_operation_mode != "backward":
        return run_speed_benchmark(lambda: fwd_fn(_input), input.kernel_operation_mode, [_input], warmup=25, rep=10)
    # The backward scales the saved gradient in place (0-D grad other than 1.0, or in_place=True),
    # so one graph cannot be backpropagated repeatedly.
    ms_50, ms_20, ms_80 = _time_fresh_backward(_input, lambda: fwd_fn(_input), warmup=25, rep=10)
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args(add_cross_entropy_arguments)
    if args.in_place and not args.reduction.startswith("none_"):
        raise SystemExit("--in-place requires --reduction none_sum or none_mean")

    # Only non-default values are added so existing benchmark data keeps its config identity.
    extra_configs = {}
    if args.reduction != "mean":
        extra_configs["reduction"] = args.reduction
    if args.dtype != "float32":
        extra_configs["dtype"] = args.dtype
    if args.in_place:
        extra_configs["in_place"] = True

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
        bench_test_fn=bench_speed_cross_entropy,
        kernel_operation_modes=["forward", "backward", "full", "no-grad-forward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    # "full" builds a fresh graph per call, so it is unaffected
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_cross_entropy),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
