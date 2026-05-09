import math
import os
import sys

import torch
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

from liger_kernel.transformers.relu_squared import LigerReLUSquared
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TorchReLUSquared(torch.nn.Module):
    def forward(self, x):
        relu_applied = torch.nn.functional.relu(x)
        return torch.square(relu_applied)


def _setup_relu_squared(input: SingleBenchmarkRunInput):
    """Create input tensors and relu_squared module from benchmark config."""
    cfg = input.extra_benchmark_config
    hidden_size = cfg["hidden_size"]
    M = cfg.get("M", input.x)
    dtype = cfg["dtype"]

    x = torch.randn(M, hidden_size, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn_like(x)

    if input.kernel_provider == "liger":
        relu_sq = LigerReLUSquared().to(device)
    elif input.kernel_provider == "torch":
        relu_sq = TorchReLUSquared().to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for relu_squared")

    fwd_fn = lambda: relu_sq(x)
    return x, dy, fwd_fn


def bench_speed_relu_squared(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, y_fwd = _setup_relu_squared(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(y_fwd, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_relu_squared(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, y_fwd = _setup_relu_squared(input)

    def full():
        y = y_fwd()
        y.backward(torch.ones_like(y), retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_relu_squared(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_relu_squared(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "M": cfg["M"],
            },
        )
    )


def bench_speed_relu_squared_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _resolve_model_config_relu_squared(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd_fn, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    elif mode == "backward":
        y = fwd_fn()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":

        def full():
            y = fwd_fn()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, grad_to_none=[x], rep=500)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_relu_squared_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _resolve_model_config_relu_squared(input)

    def full():
        y = fwd_fn()
        y.backward(torch.ones_like(y), retain_graph=True)

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
                    x=0,
                    kernel_provider="torch",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "dtype": model_cfg.dtype,
                        "M": probe_bt,
                    },
                )
                _, _, fwd_fn = _setup_relu_squared(probe_input)
                return fwd_fn()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {"hidden_size": cfg.hidden_size, "dtype": cfg.dtype} for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "relu_squared",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [{"model_configs": model_configs_info, "M": sweep.bt}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_relu_squared_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_relu_squared_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_bt = 2048

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=0,
                kernel_provider="torch",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "M": probe_bt,
                },
            )
            _, _, fwd_fn = _setup_relu_squared(probe_input)
            return fwd_fn()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "relu_squared",
            "x_name": "BT",
            "x_label": "B x T",
            "x_values": [2**i for i in range(10, int(math.log2(max(1024, config.batch_size * config.seq_len))) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [{"hidden_size": model.hidden_size, "dtype": model.dtype}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_relu_squared,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_relu_squared,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
