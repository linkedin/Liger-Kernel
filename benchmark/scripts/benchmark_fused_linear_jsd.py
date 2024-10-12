import torch
import triton
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD


class TorchJSD(torch.nn.Module):
    def __init__(self, beta: float = 0.5, dtype: torch.dtype = torch.float):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.beta = beta
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.tensor,  # input
        log_p: torch.tensor,  # target
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_p), torch.exp(log_q), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p) + (1 - self.beta) * self.kl(
            torch.log(m), log_q
        )
        return loss.to(self.dtype)


class TorchLMHeadJSD(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    :param beta: jsd beta
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.jsd = TorchJSD(beta, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        temperature: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.teacher_lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype, device=device
        )
        self.fused_jsd = LigerFusedLinearJSD(beta, temperature)

    def forward(self, student_input, teacher_input):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
        )


#############################################################################
# Test the memory consumption of the fused linear JSD
#############################################################################


def bench_memory_fused_linear_jsd(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    device = "cuda"
    torch_lm_head_jsd = TorchLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)

    # init the linear in all FusedLinearJSDs with the same weights
    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    student_input = torch.rand(BT, H, requires_grad=True, dtype=dtype, device=device)
    teacher_input = torch.rand(BT, H, dtype=dtype, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_jsd(student_input, teacher_input)
        elif provider == "torch":
            return torch_lm_head_jsd(student_input, teacher_input)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


# #############################################################################
# # Test the speed of the fused linear JSD
# #############################################################################


def bench_speed_fused_linear_jsd(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    mode = input.kernel_operation_mode

    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    device = "cuda"
    torch_lm_head_jsd = TorchLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)

    # init the linear in all FusedLinearJSDs with the same weights
    torch_lm_head_jsd.student_lin.weight.data = (
        liger_lm_head_jsd.student_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)
    torch_lm_head_jsd.teacher_lin.weight.data = (
        liger_lm_head_jsd.teacher_lin.weight.data
    ) = torch.rand(V, H, device=device, dtype=dtype)

    student_input = torch.rand(BT, H, requires_grad=True, dtype=dtype, device=device)
    teacher_input = torch.rand(BT, H, dtype=dtype, device=device)

    def fwd():
        if provider == "liger":
            return liger_lm_head_jsd(student_input, teacher_input)
        elif provider == "torch":
            return torch_lm_head_jsd(student_input, teacher_input)

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
            grad_to_none=[
                student_input,
                torch_lm_head_jsd.student_lin.weight,
                torch_lm_head_jsd.teacher_lin.weight,
            ],
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
        "kernel_name": "fused_linear_jsd",
        "x_name": "BT",
        "x_label": "B x T",
        "x_values": [2**i for i in range(10, 14)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {"H": 4096, "V": 128256, "mode": "forward", "dtype": torch.bfloat16}
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_jsd,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_jsd,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
