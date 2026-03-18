import math

import torch

from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import estimate_kernel_peak_memory
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


def _setup_swiglu(input: SingleBenchmarkRunInput):
    """Create input tensor and SwiGLU layer from benchmark config."""
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
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(cfg["dtype"])
    elif input.kernel_provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(cfg["dtype"])
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for SwiGLU")
    return x, layer


def bench_speed_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_swiglu(input)
    return run_speed_benchmark(lambda: layer(x), input.kernel_operation_mode, [x])


def bench_memory_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, layer = _setup_swiglu(input)
    return run_memory_benchmark(lambda: layer(x), input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

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
                "hidden_act": "silu",
                "dtype": model.dtype,
            },
        )
        x, layer = _setup_swiglu(probe_input)
        return layer(x)

    peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
    kernel_bpt = peak_bytes // probe_seq_len

    config = compute_seq_len_sweep_config(model, kernel_bytes_per_token=kernel_bpt)

    common_configs = {
        "kernel_name": "swiglu",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, int(math.log2(config.seq_len)) + 1)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "bsz": config.batch_size,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "hidden_act": "silu",
                "dtype": model.dtype,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_swiglu,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_swiglu,
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
