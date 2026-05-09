import math

import torch
import torch.nn as nn
import triton

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
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


def _setup_fused_add_rms_norm(input: SingleBenchmarkRunInput):
    """Create input tensors and FusedAddRMSNorm layer from benchmark config."""
    cfg = input.extra_benchmark_config
    hidden_size = cfg["hidden_size"]
    eps = cfg["eps"]
    x_shape = (input.x, hidden_size)
    x = torch.randn(x_shape, dtype=cfg["dtype"], device=device, requires_grad=True)
    r = torch.randn(x_shape, dtype=cfg["dtype"], device=device, requires_grad=True)

    if input.kernel_provider == "liger_fused_add_rms_norm":
        layer = LigerFusedAddRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "huggingface":
        layer = NaiveAddRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    elif input.kernel_provider == "liger_rms_norm":
        layer = AddLigerRMSNorm(hidden_size=hidden_size, eps=eps).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for FusedAddRMSNorm")
    return x, r, layer


def bench_speed_fused_add_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, r, layer = _setup_fused_add_rms_norm(input)
    mode = input.kernel_operation_mode
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)

    def y_fwd():
        return layer(x, r)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y, s = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: torch.autograd.backward((y, s), (dy, ds), retain_graph=True),
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y, s = y_fwd()
            torch.autograd.backward((y, s), (dy, ds))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x, r],
            rep=500,
            quantiles=QUANTILES,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_add_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, r, layer = _setup_fused_add_rms_norm(input)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)

    def y_fwd():
        return layer(x, r)

    def full():
        y, s = y_fwd()
        torch.autograd.backward((y, s), (dy, ds))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


def _resolve_model_config_fused_add_rms_norm(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_fused_add_rms_norm(
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


def bench_speed_fused_add_rms_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, r, layer = _resolve_model_config_fused_add_rms_norm(input)
    mode = input.kernel_operation_mode
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)

    def y_fwd():
        return layer(x, r)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(y_fwd, grad_to_none=[x, r], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y, s = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: torch.autograd.backward((y, s), (dy, ds), retain_graph=True),
            grad_to_none=[x, r],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y, s = y_fwd()
            torch.autograd.backward((y, s), (dy, ds))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x, r], rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_fused_add_rms_norm_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, r, layer = _resolve_model_config_fused_add_rms_norm(input)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)

    def y_fwd():
        return layer(x, r)

    def full():
        y, s = y_fwd()
        torch.autograd.backward((y, s), (dy, ds))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


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
                x, r, layer = _setup_fused_add_rms_norm(probe_input)
                y, s = layer(x, r)
                return y + s  # combine for backward probe

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
            "kernel_name": "fused_add_rms_norm",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger_fused_add_rms_norm", "huggingface", "liger_rms_norm"],
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
            bench_test_fn=bench_speed_fused_add_rms_norm_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_add_rms_norm_model_config,
            kernel_operation_modes=["full"],
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
            x, r, layer = _setup_fused_add_rms_norm(probe_input)
            y, s = layer(x, r)
            return y + s

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "fused_add_rms_norm",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger_fused_add_rms_norm", "huggingface", "liger_rms_norm"],
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
            bench_test_fn=bench_speed_fused_add_rms_norm,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_add_rms_norm,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
