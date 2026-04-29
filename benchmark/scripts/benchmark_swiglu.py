import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


def setup_swiglu(input: SingleBenchmarkRunInput):
    """Create input tensor and SwiGLU layer from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        seq_len = cfg["seq_len"]
        hidden_size = model_cfg.hidden_size
        intermediate_size = model_cfg.intermediate_size
        dtype = model_cfg.dtype
    else:
        seq_len = input.x
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg["intermediate_size"]
        dtype = cfg["dtype"]

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=cfg["hidden_act"],
    )
    x = torch.randn(
        cfg["bsz"],
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
    elif input.kernel_provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for SwiGLU")
    return x, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        common_configs = build_model_config_sweep(
            kernel_name="swiglu",
            all_model_configs=all_model_configs,
            setup_fn=setup_swiglu,
            model_keys=["hidden_size", "intermediate_size", "dtype"],
            probe_provider="huggingface",
            extra_configs={
                "bsz": 1,
                "hidden_act": "silu",
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="swiglu",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_swiglu,
            model_keys=["hidden_size", "intermediate_size", "dtype"],
            extra_configs={"hidden_act": "silu", "bsz": 1},
            scale_dim="BT",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "huggingface"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_swiglu),
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_swiglu),
        kernel_operation_modes=["full", "forward", "backward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
