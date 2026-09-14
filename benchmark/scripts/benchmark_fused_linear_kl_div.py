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

from liger_kernel.transformers.fused_linear_kl_div import LigerFusedLinearKLDivLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchLMHeadKLDiv(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based KL divergence loss.

    :param H: hidden size
    :param V: vocab size
    :param temperature: softmax temperature
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        ignore_index: int = -100,
        temperature: float = 1.0,
        eps: float = 1e-10,
    ):
        super(TorchLMHeadKLDiv, self).__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.eps = eps
        self.dtype = dtype

    def forward(self, student_input, target, label=None):
        logits = self.student_lin(student_input).to(torch.float32) / self.temperature
        log_p = torch.log_softmax(logits, dim=-1)
        q = target.to(torch.float32)
        # KL(q || p) = sum(q * (log q - log p)); 0 * log 0 is treated as 0 via the clamp
        loss_mat = q * (torch.log(q.clamp_min(self.eps)) - log_p)

        if label is not None:
            keep = (label != self.ignore_index).to(torch.float32).unsqueeze(-1)
            loss_mat = loss_mat * keep
            n_non_ignore = int(keep.sum().item())
        else:
            n_non_ignore = student_input.shape[0]

        total = loss_mat.sum()
        if n_non_ignore == 0:
            return (total * 0.0).to(self.dtype)
        return (total / n_non_ignore).to(self.dtype)


class LigerLMHeadKLDiv(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.student_lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.fused_kl = LigerFusedLinearKLDivLoss(
            ignore_index=ignore_index,
            temperature=temperature,
        )

    def forward(self, student_input, target, label=None):
        return self.fused_kl(student_input, self.student_lin.weight, target, label)


def setup_fused_linear_kl_div(input: SingleBenchmarkRunInput):
    """Create input tensors and fused linear KL divergence from benchmark config."""
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

    torch_lm_head_kl = TorchLMHeadKLDiv(H=H, V=V, dtype=dtype, device=device).to(device)
    liger_lm_head_kl = LigerLMHeadKLDiv(H=H, V=V, dtype=dtype, device=device).to(device)

    # init the linear in all FusedLinearKLDivs with the same weights
    torch_lm_head_kl.student_lin.weight.data = liger_lm_head_kl.student_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    student_input = torch.rand(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.rand(BT, V, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    if input.kernel_provider == "liger":
        lm_head = liger_lm_head_kl
    elif input.kernel_provider == "torch":
        lm_head = torch_lm_head_kl
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for FusedLinearKLDiv")

    return student_input, lambda _: lm_head(student_input, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_linear_kl_div",
            setup_fn=setup_fused_linear_kl_div,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_dim="BT",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="fused_linear_kl_div",
            probe_x=1024,
            model=model,
            setup_fn=setup_fused_linear_kl_div,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            scale_dim="BT",
            x_label="B * T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_linear_kl_div),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_linear_kl_div),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
