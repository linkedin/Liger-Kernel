import math

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

from liger_kernel.transformers.tvd import LigerTVDLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchTVDLoss(torch.nn.Module):
    def __init__(self, reduction="batchmean"):
        super(TorchTVDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        tvd = torch.abs(p - q) / 2.0
        if self.reduction == "mean":
            return torch.sum(tvd) / (p.size(0) * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / p.size(0)
        else:
            raise ValueError("Invalid reduction type.")


def _setup_tvd(input: SingleBenchmarkRunInput):
    """Create input tensors and TVD loss from benchmark config."""
    cfg = input.extra_benchmark_config
    V = cfg["vocab_size"]
    BT = input.x
    reduction = "batchmean"

    _input = torch.randn(BT, V, requires_grad=True, device=device).softmax(dim=-1)
    target = torch.randn(BT, V, device=device).softmax(dim=-1)

    if input.kernel_provider == "liger":
        loss_fn = LigerTVDLoss(reduction=reduction)
    elif input.kernel_provider == "torch":
        loss_fn = TorchTVDLoss(reduction=reduction)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for TVD")
    return _input, target, loss_fn


def bench_speed_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, loss_fn = _setup_tvd(input)
    mode = input.kernel_operation_mode

    def fwd():
        return loss_fn(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[_input],
            rep=100,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, loss_fn = _setup_tvd(input)

    def full():
        y = loss_fn(_input, target)
        y.backward(retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_tvd(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_tvd(
        SingleBenchmarkRunInput(
            x=cfg["BT"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "vocab_size": model_info["vocab_size"],
            },
        )
    )


def bench_speed_tvd_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, loss_fn = _resolve_model_config_tvd(input)
    mode = input.kernel_operation_mode

    def fwd():
        return loss_fn(_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[_input],
            rep=100,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_tvd_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    _input, target, loss_fn = _resolve_model_config_tvd(input)

    def full():
        y = loss_fn(_input, target)
        y.backward(retain_graph=True)

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
                    kernel_provider="torch",
                    extra_benchmark_config={
                        "vocab_size": model_cfg.vocab_size,
                    },
                )
                _input, target, loss_fn = _setup_tvd(probe_input)
                return loss_fn(_input, target)

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)

        model_configs_info = {
            cfg.name: {
                "vocab_size": cfg.vocab_size,
            }
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "tvd",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "BT": sweep.bt,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_tvd_model_config,
            kernel_operation_modes=["forward", "full", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_tvd_model_config,
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
                kernel_provider="torch",
                extra_benchmark_config={
                    "vocab_size": model.vocab_size,
                },
            )
            _input, target, loss_fn = _setup_tvd(probe_input)
            return loss_fn(_input, target)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "tvd",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "vocab_size": model.vocab_size,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_tvd,
            kernel_operation_modes=["forward", "full", "backward"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_tvd,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
