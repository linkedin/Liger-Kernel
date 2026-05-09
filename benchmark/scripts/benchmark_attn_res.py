"""
AttnRes Benchmark: Liger (Triton) vs PyTorch

Kimi Attention Residuals: softmax attention over depth blocks.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import compute_model_config_sweep_config
from benchmark_model_configs import compute_seq_len_sweep_config
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import parse_benchmark_script_args
from utils import run_benchmarks
from utils import run_memory_benchmark
from utils import run_speed_benchmark

from liger_kernel.ops import LigerAttnResFunction
from liger_kernel.utils import infer_device

device = infer_device()


def _setup_attn_res(input: SingleBenchmarkRunInput):
    """Create input tensors for AttnRes from benchmark config."""
    cfg = input.extra_benchmark_config
    seq_len = input.x

    # V: [N, B, T, D]
    V = torch.randn(
        cfg["N"],
        cfg["bsz"],
        seq_len,
        cfg["hidden_size"],
        device=device,
        dtype=cfg["dtype"],
        requires_grad=True,
    )
    w_query = torch.randn(cfg["hidden_size"], device=device, dtype=cfg["dtype"]) * 0.02
    w_norm = torch.ones(cfg["hidden_size"], device=device, dtype=cfg["dtype"])
    eps = cfg.get("eps", 1e-6)

    if input.kernel_provider == "liger":
        fn = lambda: LigerAttnResFunction.apply(V, w_query, w_norm, eps)
    elif input.kernel_provider == "pytorch":
        from test.transformers.test_attn_res import pytorch_attn_res

        fn = lambda: pytorch_attn_res(V, w_query, w_norm, eps)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider}")

    return V, fn


def bench_speed_attn_res(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V, fn = _setup_attn_res(input)
    return run_speed_benchmark(fn, input.kernel_operation_mode, [V])


def bench_memory_attn_res(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V, fn = _setup_attn_res(input)
    return run_memory_benchmark(fn, input.kernel_operation_mode)


def _resolve_model_config_attn_res(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_attn_res(
        SingleBenchmarkRunInput(
            x=cfg["seq_len"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "N": cfg["N"],
                "bsz": cfg["bsz"],
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "eps": cfg.get("eps", 1e-6),
            },
        )
    )


def bench_speed_attn_res_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V, fn = _resolve_model_config_attn_res(input)
    return run_speed_benchmark(fn, input.kernel_operation_mode, [V])


def bench_memory_attn_res_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    V, fn = _resolve_model_config_attn_res(input)
    return run_memory_benchmark(fn, input.kernel_operation_mode)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        all_model_configs = list(MODEL_REGISTRY.values())

        def _probe_factory(model_cfg, probe_seq_len):
            def _probe():
                probe_input = SingleBenchmarkRunInput(
                    x=probe_seq_len,
                    kernel_provider="pytorch",
                    extra_benchmark_config={
                        "N": 8,
                        "bsz": 1,
                        "hidden_size": model_cfg.hidden_size,
                        "dtype": model_cfg.dtype,
                        "eps": 1e-6,
                    },
                )
                V, fn = _setup_attn_res(probe_input)
                return fn()

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
            "kernel_name": "attn_res",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "pytorch"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "N": 8,
                    "bsz": sweep.batch_size,
                    "seq_len": sweep.seq_len,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_attn_res_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_attn_res_model_config,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        def _probe():
            probe_input = SingleBenchmarkRunInput(
                x=probe_seq_len,
                kernel_provider="pytorch",
                extra_benchmark_config={
                    "N": 8,
                    "bsz": 1,
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                },
            )
            V, fn = _setup_attn_res(probe_input)
            return fn()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_seq_len)

        common_configs = {
            "kernel_name": "attn_res",
            "x_name": "T",
            "x_label": "sequence length",
            "x_values": [2**i for i in range(10, int(math.log2(config.seq_len)) + 1)],
            "kernel_providers": ["liger", "pytorch"],
            "extra_benchmark_configs": [
                {
                    "N": 8,
                    "bsz": config.batch_size,
                    "hidden_size": model.hidden_size,
                    "dtype": model.dtype,
                    "eps": 1e-6,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_attn_res,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_attn_res,
            kernel_operation_modes=["full", "forward", "backward"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
