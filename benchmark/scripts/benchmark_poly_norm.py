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

from liger_kernel.transformers.poly_norm import LigerPolyNorm
from liger_kernel.utils import infer_device

device = infer_device()


class NaivePolyNorm(nn.Module):
    """
    Naive PyTorch implementation of PolyNorm.

    Reference:
        https://github.com/BryceZhuo/PolyCom/

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        # Align with PolyCom reference: (1/3, 1/3, 1/3) and bias=1.0
        self.weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.variance_epsilon = eps

    def _norm(self, x):
        """RMSNorm operation"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, hidden_states):
        """
        Forward pass of PolyNorm

        Args:
            hidden_states: input tensor of shape (..., H)

        Returns:
            output tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute powers
        x_pow3 = hidden_states**3
        x_pow2 = hidden_states**2
        x_pow1 = hidden_states**1

        # Normalize each power
        norm_x3 = self._norm(x_pow3)
        norm_x2 = self._norm(x_pow2)
        norm_x1 = self._norm(x_pow1)

        # Weighted sum with bias
        output = self.weight[0] * norm_x3 + self.weight[1] * norm_x2 + self.weight[2] * norm_x1 + self.bias

        return output.to(input_dtype)


def setup_poly_norm(input: SingleBenchmarkRunInput):
    """Create input tensor and PolyNorm layer from benchmark config."""
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
        layer = LigerPolyNorm(eps=eps).to(device)
    elif input.kernel_provider == "torch":
        layer = NaivePolyNorm(eps=eps).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for PolyNorm")
    return x, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="poly_norm",
            setup_fn=setup_poly_norm,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            probe_dim="T",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )

    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="poly_norm",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_poly_norm,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            scale_dim="T",
            x_label="Sequence length",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_poly_norm),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_poly_norm),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
