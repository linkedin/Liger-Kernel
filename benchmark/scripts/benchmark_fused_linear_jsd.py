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

from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD
from liger_kernel.utils import infer_device

device = infer_device()


class TorchJSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(TorchJSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label=None,
    ):
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (1 - self.beta) * self.kl(
            torch.log(m), log_q
        ).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = 0.0
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()
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
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.jsd = TorchJSD(beta=beta, ignore_index=ignore_index, dtype=dtype)
        self.temperature = temperature

    def forward(self, student_input, teacher_input, label=None):
        student_logits = self.student_lin(student_input)
        teacher_logits = self.teacher_lin(teacher_input)
        student_prob = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits / self.temperature, dim=-1)

        return self.jsd(student_prob, teacher_prob, label)


class LigerLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.fused_jsd = LigerFusedLinearJSD(jsd_beta=beta, ignore_index=ignore_index, temperature=temperature)

    def forward(self, student_input, teacher_input, label=None):
        return self.fused_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            label,
        )


def setup_fused_linear_jsd(input: SingleBenchmarkRunInput):
    """Create input tensors and fused linear JSD from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        BT = cfg["seq_len"] * cfg["bsz"]
        V = model_cfg.vocab_size
        H = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        BT = input.x
        V = cfg["vocab_size"]
        H = cfg["hidden_size"]
        dtype = cfg["dtype"]

    torch_lm_head_jsd = TorchLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)
    liger_lm_head_jsd = LigerLMHeadJSD(H=H, V=V, dtype=dtype, device=device).to(device)

    # init the linear in all FusedLinearJSDs with the same weights
    torch_lm_head_jsd.student_lin.weight.data = liger_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = liger_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    student_input = torch.rand(BT, H, requires_grad=True, dtype=dtype, device=device)
    teacher_input = torch.rand(BT, H, dtype=dtype, device=device)

    if input.kernel_provider == "liger":
        lm_head = liger_lm_head_jsd
    elif input.kernel_provider == "torch":
        lm_head = torch_lm_head_jsd
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for FusedLinearJSD")

    return student_input, lambda _: lm_head(student_input, teacher_input)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_linear_jsd",
            setup_fn=setup_fused_linear_jsd,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_dim="BT",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="fused_linear_jsd",
            probe_x=1024,
            model=model,
            setup_fn=setup_fused_linear_jsd,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            scale_dim="BT",
            x_label="B * T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_linear_jsd),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_linear_jsd),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
