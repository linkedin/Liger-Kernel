import os
import sys

import torch
import torch.nn as nn
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityFunction
from liger_kernel.utils import infer_device

device = infer_device()

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TorchCosineSimilarityLoss(nn.Module):
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
        from test.chunked_loss.test_cosine_loss import HFCosineLoss

        super().__init__()
        self.student_lin = nn.Linear(in_features=H // 2, out_features=V, bias=bias).to(dtype=dtype)
        self.teacher_lin = nn.Linear(in_features=H, out_features=V, bias=bias).to(dtype=dtype)
        self.cosine_loss = HFCosineLoss(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        ).get_batch_loss_metrics

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor):
        return self.cosine_loss(student, self.student_lin.weight, teacher, self.teacher_lin.weight, target)


class LigerCosineSimilarityLoss(nn.Module):
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
        self.student_lin = nn.Linear(in_features=H // 2, out_features=V, bias=bias).to(dtype=dtype)
        self.teacher_lin = nn.Linear(in_features=H, out_features=V, bias=bias).to(dtype=dtype)
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.cosine_loss = LigerFusedLinearCosineSimilarityFunction.apply

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor):
        return self.cosine_loss(
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


def bench_memory_cosine_similarity_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    bias = input.extra_benchmark_config["bias"]
    weight_hard_loss = input.extra_benchmark_config["weight_hard_loss"]
    weight_soft_loss = input.extra_benchmark_config["weight_soft_loss"]
    ignore_index = input.extra_benchmark_config["ignore_index"]
    provider = input.kernel_provider

    torch_cosine_loss = TorchCosineSimilarityLoss(
        H=H,
        V=V,
        dtype=dtype,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        bias=bias,
    ).to(device)
    liger_cosine_loss = LigerCosineSimilarityLoss(
        H=H,
        V=V,
        dtype=dtype,
        ignore_index=ignore_index,
        bias=bias,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
    ).to(device)

    _tensor = torch.rand(BT, H // 2, device=device, dtype=dtype)
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(BT, H, device=device, dtype=dtype)

    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    def fwd():
        if provider == "liger":
            return liger_cosine_loss(student_input1, teacher_input, target)
        elif provider == "torch":
            return torch_cosine_loss(student_input2, teacher_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def bench_speed_cosine_similarity_loss(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    bias = input.extra_benchmark_config["bias"]
    weight_hard_loss = input.extra_benchmark_config["weight_hard_loss"]
    weight_soft_loss = input.extra_benchmark_config["weight_soft_loss"]
    ignore_index = input.extra_benchmark_config["ignore_index"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    torch_cosine_loss = TorchCosineSimilarityLoss(
        H=H,
        V=V,
        dtype=dtype,
        ignore_index=ignore_index,
        bias=bias,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
    ).to(device)

    liger_cosine_loss = LigerCosineSimilarityLoss(
        H=H,
        V=V,
        dtype=dtype,
        ignore_index=ignore_index,
        bias=bias,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
    ).to(device)

    _tensor = torch.rand(BT, H // 2, device=device, dtype=dtype)
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(BT, H, device=device, dtype=dtype)

    target = torch.randint(0, V, (BT,), device=device, dtype=torch.long)

    def fwd():
        if provider == "liger":
            return liger_cosine_loss(student_input1, teacher_input, target)
        elif provider == "torch":
            return torch_cosine_loss(student_input2, teacher_input, target)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[student_input1, student_input2],
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

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "distill_cosine_loss",
        "x_name": "BT",
        "x_label": "B x T",
        "x_values": [2**i for i in range(10, 14)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {
                "H": 4096,
                "V": 128256,
                "mode": "forward",
                "dtype": torch.bfloat16,
                "bias": False,
                "weight_hard_loss": 0.5,
                "weight_soft_loss": 0.5,
                "ignore_index": -100,
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_cosine_similarity_loss,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_cosine_similarity_loss,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
