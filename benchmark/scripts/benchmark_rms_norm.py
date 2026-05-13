import torch
import torch.nn as nn

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

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


def setup_rms_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and RMSNorm layer from benchmark config."""
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
        layer = LigerRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "huggingface":
        layer = LlamaRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for RMSNorm")
    return x, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="rms_norm",
            setup_fn=setup_rms_norm,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            probe_dim="T",
            probe_provider="huggingface",
            bt=args.bt,
            overwrite=args.overwrite,
        )

    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="rms_norm",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_rms_norm,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            scale_dim="T",
            x_label="Sequence length",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["huggingface", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_rms_norm),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_rms_norm),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
