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

from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TorchJSDLoss(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        bias: bool = False,
    ):
        from test.chunked_loss.test_jsd_loss import HFJSDLoss

        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.jsd_loss = HFJSDLoss(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        ).get_batch_loss_metrics

    def forward(self, student, teacher, target):
        return self.jsd_loss(
            student,
            self.student_lin.weight,
            teacher,
            self.teacher_lin.weight,
            target,
        )


class LigerJSDLoss(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.jsd_loss = LigerFusedLinearJSDFunction.apply

    def forward(self, student, teacher, target):
        return self.jsd_loss(
            student,
            self.student_lin.weight,
            teacher,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
            self.weight_hard_loss,
            self.weight_soft_loss,
        )


def _setup_distill_jsd_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and JSD loss from benchmark config."""
    cfg = input.extra_benchmark_config
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    dtype = cfg["dtype"]
    bias = cfg["bias"]
    weight_hard_loss = cfg["weight_hard_loss"]
    weight_soft_loss = cfg["weight_soft_loss"]
    ignore_index = cfg["ignore_index"]
    BT = input.x

    _tensor = torch.rand(BT, H // 2, device=device, dtype=dtype)
    student_input = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(BT, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    if input.kernel_provider == "liger":
        loss_module = LigerJSDLoss(
            H=H,
            V=V,
            dtype=dtype,
            ignore_index=ignore_index,
            bias=bias,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
        ).to(device)
    elif input.kernel_provider == "torch":
        loss_module = TorchJSDLoss(
            H=H,
            V=V,
            dtype=dtype,
            ignore_index=ignore_index,
            bias=bias,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
        ).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for DistillJSDLoss")
    return student_input, teacher_input, target, loss_module


def bench_speed_distill_jsd_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, target, loss_module = _setup_distill_jsd_loss(input)
    mode = input.kernel_operation_mode

    def fwd():
        return loss_module(student_input, teacher_input, target)

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
            grad_to_none=[student_input],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, rep=100, quantiles=QUANTILES)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_distill_jsd_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, target, loss_module = _setup_distill_jsd_loss(input)

    def full():
        y = loss_module(student_input, teacher_input, target)
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_distill_jsd_loss(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_distill_jsd_loss(
        SingleBenchmarkRunInput(
            x=cfg["BT"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "vocab_size": model_info["vocab_size"],
                "dtype": model_info["dtype"],
                "bias": cfg["bias"],
                "weight_hard_loss": cfg["weight_hard_loss"],
                "weight_soft_loss": cfg["weight_soft_loss"],
                "ignore_index": cfg["ignore_index"],
            },
        )
    )


def bench_speed_distill_jsd_loss_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, target, loss_module = _resolve_model_config_distill_jsd_loss(input)
    mode = input.kernel_operation_mode

    def fwd():
        return loss_module(student_input, teacher_input, target)

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
            grad_to_none=[student_input],
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


def bench_memory_distill_jsd_loss_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, target, loss_module = _resolve_model_config_distill_jsd_loss(input)

    def full():
        y = loss_module(student_input, teacher_input, target)
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
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
                        "hidden_size": model_cfg.hidden_size,
                        "vocab_size": model_cfg.vocab_size,
                        "dtype": model_cfg.dtype,
                        "bias": False,
                        "weight_hard_loss": 0.5,
                        "weight_soft_loss": 0.5,
                        "ignore_index": -100,
                    },
                )
                student_input, teacher_input, target, loss_module = _setup_distill_jsd_loss(probe_input)
                return loss_module(student_input, teacher_input, target)

            return _probe

        sweep = compute_model_config_sweep_config(all_model_configs, probe_fn_factory=_probe_factory, bt=args.bt)

        model_configs_info = {
            cfg.name: {
                "hidden_size": cfg.hidden_size,
                "vocab_size": cfg.vocab_size,
                "dtype": cfg.dtype,
            }
            for cfg in sweep.model_configs
        }

        common_configs = {
            "kernel_name": "distill_jsd_loss",
            "x_name": "model_config",
            "x_label": "model configuration",
            "x_values": [cfg.name for cfg in sweep.model_configs],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "model_configs": model_configs_info,
                    "BT": sweep.bt,
                    "bias": False,
                    "weight_hard_loss": 0.5,
                    "weight_soft_loss": 0.5,
                    "ignore_index": -100,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_distill_jsd_loss_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_distill_jsd_loss_model_config,
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
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                    "bias": False,
                    "weight_hard_loss": 0.5,
                    "weight_soft_loss": 0.5,
                    "ignore_index": -100,
                },
            )
            student_input, teacher_input, target, loss_module = _setup_distill_jsd_loss(probe_input)
            return loss_module(student_input, teacher_input, target)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "distill_jsd_loss",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                    "bias": False,
                    "weight_hard_loss": 0.5,
                    "weight_soft_loss": 0.5,
                    "ignore_index": -100,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_distill_jsd_loss,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_distill_jsd_loss,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
