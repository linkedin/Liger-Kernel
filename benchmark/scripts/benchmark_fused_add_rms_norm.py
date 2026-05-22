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

from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.utils import infer_device

device = infer_device()


class NaiveAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Naive implementation of the add residual rms norm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


class AddLigerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        AddLigerRMSNorm is equivalent to NaiveAddRMSNorm class above, but uses the LigerRMSNorm kernel.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.rms_norm = LigerRMSNorm(hidden_size, eps, in_place=False)

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


def setup_fused_add_rms_norm(input: SingleBenchmarkRunInput):
    """Create input tensors and FusedAddRMSNorm layer from benchmark config."""
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

    eps = cfg["eps"]
    x_shape = (bt, hidden_size)
    x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    r = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)

    if input.kernel_provider == "liger_fused_add_rms_norm":
        layer = LigerFusedAddRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "torch":
        layer = NaiveAddRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "liger_rms_norm":
        layer = AddLigerRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for FusedAddRMSNorm")

    return x, lambda _: layer(x, r)[0]


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_add_rms_norm",
            setup_fn=setup_fused_add_rms_norm,
            model_keys=["hidden_size", "intermediate_size", "dtype"],
            probe_provider="torch",
            extra_configs={
                "eps": 1e-6,
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="fused_add_rms_norm",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_fused_add_rms_norm,
            model_keys=["hidden_size", "intermediate_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger_rms_norm", "liger_fused_add_rms_norm"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_add_rms_norm),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_add_rms_norm),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
