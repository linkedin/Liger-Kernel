import os
import sys

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.fused_lora_linear import LigerFusedLoRALinear
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def setup_fused_lora_linear(input: SingleBenchmarkRunInput):
    """Create tokens and a LoRA-adapted square linear layer."""
    from test.transformers.test_fused_lora_linear import TorchFusedLoRALinear

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

    r = cfg.get("r", 16)
    lora_alpha = cfg.get("lora_alpha", 2 * r)
    x = torch.randn(bt, hidden_size, dtype=dtype, device=device, requires_grad=True)

    if input.kernel_provider == "liger":
        layer = LigerFusedLoRALinear(
            hidden_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            device=device,
            dtype=dtype,
        )
    elif input.kernel_provider == "torch":
        layer = TorchFusedLoRALinear(
            hidden_size,
            hidden_size,
            r=r,
            lora_alpha=lora_alpha,
            device=device,
            dtype=dtype,
        )
    elif input.kernel_provider == "torch_compile":
        layer = torch.compile(
            TorchFusedLoRALinear(
                hidden_size,
                hidden_size,
                r=r,
                lora_alpha=lora_alpha,
                device=device,
                dtype=dtype,
            ),
            dynamic=False,
        )
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for fused_lora_linear")

    return x, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_lora_linear",
            setup_fn=setup_fused_lora_linear,
            model_keys=["hidden_size", "dtype"],
            probe_provider="torch",
            probe_dim="BT",
            extra_configs={"r": 16},
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="fused_lora_linear",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_fused_lora_linear,
            model_keys=["hidden_size", "dtype"],
            extra_configs={"r": 16},
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch", "torch_compile"]
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_lora_linear),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_lora_linear),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
