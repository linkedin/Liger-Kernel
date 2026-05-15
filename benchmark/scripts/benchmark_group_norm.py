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

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()


def setup_group_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and GroupNorm layer from benchmark config."""
    cfg = input.extra_benchmark_config
    H = cfg["H"]
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        num_channels = model_cfg.hidden_size
        dtype = model_cfg.dtype
        M = max(1, cfg["bsz"] * cfg["seq_len"] // H)
    else:
        # input.x is BT; derive M from BT / H
        M = max(1, input.x // H)
        num_channels = cfg["hidden_size"]
        dtype = cfg["dtype"]

    num_groups = num_channels // cfg["channels_per_group"]
    x = torch.randn(M, num_channels, H, device=device, dtype=dtype, requires_grad=True)

    if input.kernel_provider == "liger":
        layer = LigerGroupNorm(num_channels=num_channels, num_groups=num_groups, eps=cfg["eps"]).to(
            device=device, dtype=dtype
        )
    elif input.kernel_provider == "torch":
        layer = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=cfg["eps"]).to(
            device=device, dtype=dtype
        )
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for GroupNorm")

    return x, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    extra_configs = {
        "channels_per_group": 4,
        "eps": 1e-6,
        "H": 512,
    }

    probe_bt = 1024

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="group_norm",
            setup_fn=setup_group_norm,
            model_keys=["hidden_size", "dtype"],
            probe_provider="torch",
            extra_configs=extra_configs,
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="group_norm",
            probe_x=probe_bt,
            model=model,
            setup_fn=setup_group_norm,
            model_keys=["hidden_size", "dtype"],
            extra_configs=extra_configs,
            scale_dim="BT",
            x_label="B*T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_group_norm),
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_group_norm),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
