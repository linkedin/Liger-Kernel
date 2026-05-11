import math

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


def _setup_geglu(input: SingleBenchmarkRunInput):
    """Create input tensor and GEGLU layer from benchmark config."""
    cfg = input.extra_benchmark_config
    llama_config = LlamaConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        hidden_act=cfg["hidden_act"],
    )
    x = torch.randn(
        cfg["bsz"],
        input.x,
        cfg["hidden_size"],
        device=device,
        dtype=cfg["dtype"],
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = LigerGEGLUMLP(config=llama_config).to(device).to(cfg["dtype"])
    elif input.kernel_provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(cfg["dtype"])
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for GEGLU")
    return x, layer


def bench_speed_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_geglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_geglu(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


def _resolve_model_config_geglu(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_geglu(
        SingleBenchmarkRunInput(
            x=cfg["seq_len"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "bsz": cfg["bsz"],
                "hidden_size": model_info["hidden_size"],
                "intermediate_size": model_info["intermediate_size"],
                "hidden_act": cfg["hidden_act"],
                "dtype": model_info["dtype"],
            },
        )
    )


def bench_speed_geglu_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_geglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_geglu_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _resolve_model_config_geglu(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def _probe_factory(model_cfg, probe_seq_len):
            def _probe():
                probe_input = SingleBenchmarkRunInput(
                    x=probe_seq_len,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "bsz": 1,
                        "hidden_size": model_cfg.hidden_size,
                        "intermediate_size": model_cfg.intermediate_size,
                        "hidden_act": "gelu_pytorch_tanh",
                        "dtype": model_cfg.dtype,
                    },
                )
                x, layer = _setup_geglu(probe_input)
                return layer(x)

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)

        model_configs_info = {
            cfg.name: {
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "dtype": cfg.dtype,
            }
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "geglu",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "bsz": sweep.batch_size,
                    "seq_len": sweep.seq_len,
                    "hidden_act": "gelu_pytorch_tanh",
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_geglu_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_geglu_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=probe_seq_len,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "bsz": 1,
                    "hidden_size": model.hidden_size,
                    "intermediate_size": model.intermediate_size,
                    "hidden_act": "gelu_pytorch_tanh",
                    "dtype": model.dtype,
                },
            )
            x, layer = _setup_geglu(probe_input)
            return layer(x)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_seq_len)

        common_configs = {
            "kernel_name": "geglu",
            "x_name": "T",
            "x_label": "sequence length",
            "x_values": [2**i for i in range(10, int(math.log2(config.seq_len)) + 1)],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {
                    "bsz": config.batch_size,
                    "hidden_size": model.hidden_size,
                    "intermediate_size": model.intermediate_size,
                    "hidden_act": "gelu_pytorch_tanh",
                    "dtype": model.dtype,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_geglu,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_geglu,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
