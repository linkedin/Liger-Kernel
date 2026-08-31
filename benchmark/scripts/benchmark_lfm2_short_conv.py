import torch
import torch.nn as nn

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from test.transformers.test_lfm2_short_conv import _reference
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops import LigerLfm2ShortConvFunction
from liger_kernel.utils import infer_device

device = infer_device()


class _ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, dtype, use_liger, bias_enabled):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, 1, kernel_size, device=device, dtype=dtype) * 0.02)
        self.bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype)) if bias_enabled else None
        self.use_liger = use_liger

    def forward(self, bcx):
        if self.use_liger:
            return LigerLfm2ShortConvFunction.apply(bcx, self.weight, self.bias)
        return _reference(bcx, self.weight, self.bias)


def setup_lfm2_short_conv(input: SingleBenchmarkRunInput):
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

    bcx = torch.randn(
        cfg["bsz"],
        seq_len,
        3 * hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = _ShortConv(hidden_size, cfg["kernel_size"], dtype, use_liger=True, bias_enabled=cfg["bias_enabled"])
    elif input.kernel_provider == "huggingface":
        layer = _ShortConv(hidden_size, cfg["kernel_size"], dtype, use_liger=False, bias_enabled=cfg["bias_enabled"])
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for LFM2 short convolution")
    return bcx, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="lfm2_short_conv",
            setup_fn=setup_lfm2_short_conv,
            model_keys=["hidden_size", "dtype"],
            probe_provider="huggingface",
            extra_configs={"bsz": 1, "kernel_size": 3, "bias_enabled": False},
            probe_dim="T",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model or "lfm2_1.2b")
        common_configs = build_token_length_sweep(
            kernel_name="lfm2_short_conv",
            probe_x=1024,
            model=model,
            setup_fn=setup_lfm2_short_conv,
            model_keys=["hidden_size", "dtype"],
            extra_configs={"bsz": 1, "kernel_size": 3, "bias_enabled": False},
            scale_dim="T",
            x_label="total tokens",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["huggingface", "liger"]
    for metric_name, metric_unit, bench_fn in (
        ("speed", "ms", build_speed_bench_fn(setup_lfm2_short_conv)),
        ("memory", "MB", build_memory_bench_fn(setup_lfm2_short_conv)),
    ):
        run_benchmarks(
            bench_test_fn=bench_fn,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name=metric_name,
            metric_unit=metric_unit,
            **common_configs,
        )
