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

from liger_kernel.transformers.softmax import LigerSoftmax
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def setup_softmax(input: SingleBenchmarkRunInput):
    """Create input tensors and softmax module from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        bt = cfg["seq_len"] * cfg["bsz"]
        hidden_size = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        bt = input.x
        hidden_size = cfg["hidden_size"]
        dtype = cfg["dtype"]

    x = torch.randn(bt, hidden_size, dtype=dtype, device=device, requires_grad=True)

    if input.kernel_provider == "liger":
        softmax = LigerSoftmax().to(device).to(dtype)
    elif input.kernel_provider == "torch":
        softmax = torch.nn.Softmax(dim=-1).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for softmax")

    return x, softmax


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="softmax",
            setup_fn=setup_softmax,
            model_keys=["hidden_size", "dtype"],
            probe_provider="torch",
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 2048

        common_configs = build_token_length_sweep(
            kernel_name="softmax",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_softmax,
            model_keys=["hidden_size", "dtype"],
            scale_dim="BT",
            x_label="B*T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_softmax),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_softmax),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
