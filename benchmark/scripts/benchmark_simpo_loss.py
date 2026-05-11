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

from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _setup_simpo_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and SimPO loss from benchmark config."""
    from test.chunked_loss.test_simpo_loss import LigerLMHeadSimPO
    from test.chunked_loss.test_simpo_loss import TorchLMHeadCPO

    cfg = input.extra_benchmark_config
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    dtype = cfg["dtype"]
    B = input.x
    T = cfg["T"]

    _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)

    if input.kernel_provider == "liger":
        loss_module = LigerLMHeadSimPO(H=H, V=V, dtype=dtype).to(device)
    elif input.kernel_provider == "huggingface":
        loss_module = TorchLMHeadCPO(H=H, V=V, dtype=dtype).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for SimPOLoss")

    fwd = lambda: loss_module(_input, target)[0]
    return _input, fwd


def bench_speed_simpo_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, fwd = _setup_simpo_loss(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_simpo_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, fwd = _setup_simpo_loss(input)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_simpo_loss(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_simpo_loss(
        SingleBenchmarkRunInput(
            x=cfg["B"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "vocab_size": model_info["vocab_size"],
                "dtype": model_info["dtype"],
                "T": cfg["T"],
            },
        )
    )


def bench_speed_simpo_loss_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, fwd = _resolve_model_config_simpo_loss(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            rep=100,
            quantiles=QUANTILES,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_simpo_loss_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, fwd = _resolve_model_config_simpo_loss(input)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())
        T = 1024

        def _probe_factory(model_cfg, probe_bt):
            def _probe():
                B = max(1, probe_bt // T)
                probe_input = SingleBenchmarkRunInput(
                    x=B,
                    kernel_provider="huggingface",
                    extra_benchmark_config={
                        "hidden_size": model_cfg.hidden_size,
                        "vocab_size": model_cfg.vocab_size,
                        "dtype": model_cfg.dtype,
                        "T": T,
                    },
                )
                _, fwd = _setup_simpo_loss(probe_input)
                return fwd()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {"hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size, "dtype": cfg.dtype}
            for cfg in sweep.model_configs
        }
        B = max(1, sweep.bt // T)

        common_configs = {
            "kernel_name": "fused_linear_simpo_loss",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [{"model_configs": model_configs_info, "B": B, "T": T}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_simpo_loss_model_config,
            kernel_operation_modes=["forward", "full", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_simpo_loss_model_config,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        T = 1024
        probe_bt = 1024

        def _probe():
            B = probe_bt // T
            probe_input = SingleBenchmarkRunInput(
                x=B,
                kernel_provider="huggingface",
                extra_benchmark_config={
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                    "T": T,
                },
            )
            _, fwd = _setup_simpo_loss(probe_input)
            return fwd()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "fused_linear_simpo_loss",
            "x_name": "B",
            "x_label": "Batch Size (B)",
            "x_values": [2**i for i in range(1, int(math.log2(max(2, config.batch_size * config.seq_len // T))) + 1)],
            "kernel_providers": ["liger", "huggingface"],
            "extra_benchmark_configs": [
                {"hidden_size": model.hidden_size, "vocab_size": model.vocab_size, "dtype": model.dtype, "T": T}
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_simpo_loss,
            kernel_operation_modes=["forward", "full", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_simpo_loss,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
