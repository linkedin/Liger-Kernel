import math

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()


def _setup_group_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and GroupNorm layer from benchmark config."""
    cfg = input.extra_benchmark_config
    num_channels = cfg["num_channels"]
    channels_per_group = cfg["channels_per_group"]
    H = cfg["H"]
    eps = cfg["eps"]
    num_groups = num_channels // channels_per_group
    x = torch.randn(
        input.x,
        num_channels,
        H,
        device=device,
        dtype=cfg["dtype"],
        requires_grad=True,
    )
    dtype = cfg["dtype"]
    if input.kernel_provider == "liger":
        layer = LigerGroupNorm(num_channels=num_channels, num_groups=num_groups, eps=eps).to(device=device, dtype=dtype)
    elif input.kernel_provider == "huggingface":
        layer = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps).to(
            device=device, dtype=dtype
        )
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for GroupNorm")
    return x, layer


def bench_speed_group_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_group_norm(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_group_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_group_norm(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


def _resolve_model_config_group_norm(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_group_norm(
        SingleBenchmarkRunInput(
            x=cfg["M"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "num_channels": model_info["hidden_size"],
                "channels_per_group": cfg["channels_per_group"],
                "H": cfg["H"],
                "dtype": model_info["dtype"],
                "eps": cfg["eps"],
            },
        )
    )


def bench_speed_group_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_group_norm(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_group_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_group_norm(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())
        channels_per_group = 4
        H = 512

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                M = max(1, probe_bt // H)
                probe_input = SingleBenchmarkRunInput(
                    x=M,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "num_channels": model_cfg.hidden_size,
                        "channels_per_group": channels_per_group,
                        "H": H,
                        "dtype": model_cfg.dtype,
                        "eps": 1e-6,
                    },
                )
                x, layer = _setup_group_norm(probe_input)
                return layer(x)

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)

        model_configs_info = {
            cfg.name: {
                "hidden_size": cfg.hidden_size,
                "dtype": cfg.dtype,
            }
            for cfg in sweep.model_configs
        }

        M = max(1, sweep.bt // H)

        common_configs = {
            "kernel_name": "group_norm",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "M": M,
                    "channels_per_group": channels_per_group,
                    "H": H,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_group_norm_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_group_norm_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        channels_per_group = 4
        H = 512
        probe_bt = 1024

        def _probe():
            M = max(1, probe_bt // H)
            probe_input = SingleBenchmarkRunInput(
                x=M,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "num_channels": model.hidden_size,
                    "channels_per_group": channels_per_group,
                    "H": H,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                },
            )
            x, layer = _setup_group_norm(probe_input)
            return layer(x)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "group_norm",
            "x_name": "M",
            "x_label": "batch size (M)",
            "x_values": [2**i for i in range(2, int(math.log2(config.batch_size * config.seq_len // H)) + 1)],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "num_channels": model.hidden_size,
                    "channels_per_group": channels_per_group,
                    "H": H,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_group_norm,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_group_norm,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
