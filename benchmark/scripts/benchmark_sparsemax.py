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

from liger_kernel.transformers.sparsemax import LigerSparsemax
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def torch_sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dims = input_tensor.dim()
    if dim < 0:
        dim = input_dims + dim
    input_sorted, _ = torch.sort(input_tensor, dim=dim, descending=True)
    cumsum_input = torch.cumsum(input_sorted, dim=dim)
    input_size = input_tensor.size(dim)
    range_tensor = torch.arange(1, input_size + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [1] * input_dims
    shape[dim] = input_size
    range_tensor = range_tensor.view(shape)
    k_bound = 1 + range_tensor * input_sorted
    support = k_bound > cumsum_input
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (input_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(input_tensor - tau, min=0)


class TorchSparsemax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_sparsemax(x, dim=self.dim)


def _setup_sparsemax(input: SingleBenchmarkRunInput):
    """Create input tensors and sparsemax module from benchmark config."""
    cfg = input.extra_benchmark_config
    hidden_size = cfg.get("hidden_size", input.x)
    bt = cfg.get("bt", input.x)
    dtype = cfg["dtype"]
    dim = cfg.get("dim", -1)

    x = torch.randn(bt, hidden_size, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn_like(x)

    if input.kernel_provider == "liger":
        sparsemax_module = LigerSparsemax(dim=dim).to(device)
    elif input.kernel_provider == "torch":
        sparsemax_module = TorchSparsemax(dim=dim).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for sparsemax")

    fwd_fn = lambda: sparsemax_module(x)
    return x, dy, fwd_fn


def bench_speed_sparsemax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, y_fwd = _setup_sparsemax(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x],
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


def bench_memory_sparsemax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, y_fwd = _setup_sparsemax(input)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_sparsemax(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_sparsemax(
        SingleBenchmarkRunInput(
            x=input.x,
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "dtype": model_info["dtype"],
                "bt": cfg["bt"],
                "dim": cfg.get("dim", -1),
            },
        )
    )


def bench_speed_sparsemax_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, y_fwd = _resolve_model_config_sparsemax(input)
    mode = input.kernel_operation_mode

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(y_fwd, grad_to_none=[x], rep=500, quantiles=QUANTILES)
    elif mode == "backward":
        y = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=500, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_sparsemax_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    x, dy, fwd_fn = _resolve_model_config_sparsemax(input)

    def full():
        y = fwd_fn()
        y.backward(dy, retain_graph=True)

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
                        "bt": probe_bt,
                        "dim": -1,
                    },
                )
                _, _, fwd_fn = _setup_sparsemax(probe_input)
                return fwd_fn()

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)
        model_configs_info = {
            cfg.name: {"hidden_size": cfg.hidden_size, "dtype": cfg.dtype} for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "sparsemax",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [{"model_configs": model_configs_info, "bt": sweep.bt, "dim": -1}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_sparsemax_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_sparsemax_model_config,
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
                    "bt": probe_bt,
                    "dim": -1,
                },
            )
            _, _, fwd_fn = _setup_sparsemax(probe_input)
            return fwd_fn()

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "sparsemax",
            "x_name": "BT",
            "x_label": "B x T",
            "x_values": [2**i for i in range(10, int(math.log2(max(1024, config.batch_size * config.seq_len))) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [{"hidden_size": model.hidden_size, "dtype": model.dtype, "dim": -1}],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_sparsemax,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_sparsemax,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
