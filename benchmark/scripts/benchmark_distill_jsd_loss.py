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


def setup_distill_jsd_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and JSD loss from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        V = model_cfg.vocab_size
        H = model_cfg.hidden_size
        dtype = model_cfg.dtype
        BT = cfg["bsz"] * cfg["seq_len"]
    else:
        BT = input.x
        V = cfg["vocab_size"]
        H = cfg["hidden_size"]
        dtype = cfg["dtype"]

    bias = cfg["bias"]
    weight_hard_loss = cfg["weight_hard_loss"]
    weight_soft_loss = cfg["weight_soft_loss"]
    ignore_index = cfg["ignore_index"]

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
    return student_input, lambda _: loss_module(student_input, teacher_input, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="distill_jsd_loss",
            setup_fn=setup_distill_jsd_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "bias": False,
                "weight_hard_loss": 0.5,
                "weight_soft_loss": 0.5,
                "ignore_index": -100,
            },
            probe_dim="BT",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="distill_jsd_loss",
            probe_x=1024,
            model=model,
            setup_fn=setup_distill_jsd_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "bias": False,
                "weight_hard_loss": 0.5,
                "weight_soft_loss": 0.5,
                "ignore_index": -100,
            },
            scale_dim="BT",
            x_label="B * T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "torch"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_distill_jsd_loss),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_distill_jsd_loss),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
