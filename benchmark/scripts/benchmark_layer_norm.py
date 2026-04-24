import math

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.utils import infer_device

device = infer_device()


def _setup_layer_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and LayerNorm layer from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        seq_len = cfg["seq_len"]
        hidden_size = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        seq_len = input.x
        hidden_size = cfg["hidden_size"]
        dtype = cfg["dtype"]

    eps = cfg["eps"]
    x = torch.randn(
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = LigerLayerNorm(hidden_size=hidden_size, eps=eps).to(device).to(dtype)
    elif input.kernel_provider == "huggingface":
        layer = torch.nn.LayerNorm(hidden_size, eps=eps).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for LayerNorm")
    return x, layer


def bench_speed_layer_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_layer_norm(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_layer_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_layer_norm(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def probe_fn(model_cfg, probe_bt):
            probe_input = SingleBenchmarkRunInput(
                x=probe_bt,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model_cfg.hidden_size,
                    "dtype": model_cfg.dtype,
                    "eps": 1e-6,
                },
            )
            x, layer = _setup_layer_norm(probe_input)
            return layer(x)

        common_configs = build_model_config_sweep(
            kernel_name="layer_norm",
            all_model_configs=all_model_configs,
            probe_fn=probe_fn,
            extra_benchmark_config={
                "eps": 1e-6,
            },
            bt=args.bt,
            overwrite=args.overwrite,
        )

    else:
        model = get_benchmark_model_config(args.model)
        probe_bt = 1024

        def probe_fn():
            probe_input = SingleBenchmarkRunInput(
                x=probe_bt,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                },
            )
            x, layer = _setup_layer_norm(probe_input)
            return layer(x)

        def x_values_fn(config):
            return [2**i for i in range(10, int(math.log2(config.seq_len)) + 1)]

        common_configs = build_token_length_sweep(
            kernel_name="layer_norm",
            probe_seq_len=probe_bt,
            model=model,
            probe_fn=probe_fn,
            extra_config_fn={
                "hidden_size": model.hidden_size,
                "dtype": model.dtype,
                "eps": 1e-6,
            },
            x_values_fn=x_values_fn,
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "huggingface"]

    run_benchmarks(
        bench_test_fn=bench_speed_layer_norm,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_layer_norm,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
