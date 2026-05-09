import math

import torch
import torch.nn as nn

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

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _setup_rms_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and RMSNorm layer from benchmark config."""
    cfg = input.extra_benchmark_config
    hidden_size = cfg["hidden_size"]
    eps = cfg["eps"]
    x = torch.randn(
        input.x,
        hidden_size,
        device=device,
        dtype=cfg["dtype"],
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = LigerRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "huggingface":
        layer = LlamaRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for RMSNorm")
    return x, layer


def bench_speed_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_rms_norm(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_rms_norm(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


def _resolve_model_config_rms_norm(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_rms_norm(
        SingleBenchmarkRunInput(
            x=cfg["BT"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "eps": cfg["eps"],
            },
        )
    )


def bench_speed_rms_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_rms_norm(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_rms_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_rms_norm(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                probe_input = SingleBenchmarkRunInput(
                    x=probe_bt,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "dtype": model_cfg.dtype,
                        "eps": 1e-6,
                    },
                )
                x, layer = _setup_rms_norm(probe_input)
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

        common_configs = {
            "kernel_name": "rms_norm",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "BT": sweep.bt,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_rms_norm_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_rms_norm_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_bt = 1024

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=probe_bt,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                },
            )
            x, layer = _setup_rms_norm(probe_input)
            return layer(x)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "rms_norm",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_rms_norm,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_rms_norm,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
