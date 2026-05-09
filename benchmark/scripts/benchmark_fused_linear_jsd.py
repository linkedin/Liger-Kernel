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


def _setup_fused_linear_jsd(input: SingleBenchmarkRunInput):
    """Create input tensors and fused linear JSD from benchmark config."""
    cfg = input.extra_benchmark_config
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    dtype = cfg["dtype"]
    BT = input.x

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

    return student_input, teacher_input, lm_head


def bench_speed_fused_linear_jsd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, lm_head = _setup_fused_linear_jsd(input)
    mode = input.kernel_operation_mode

    def fwd():
        return lm_head(student_input, teacher_input)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, rep=100, quantiles=QUANTILES)
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


def bench_memory_fused_linear_jsd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, lm_head = _setup_fused_linear_jsd(input)

    def full():
        y = lm_head(student_input, teacher_input)
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


def _resolve_model_config_fused_linear_jsd(input: SingleBenchmarkRunInput):
    """Resolve model-config-sweep input into standard setup args."""
    cfg = input.extra_benchmark_config
    model_info = cfg["model_configs"][input.x]
    return _setup_fused_linear_jsd(
        SingleBenchmarkRunInput(
            x=cfg["BT"],
            kernel_provider=input.kernel_provider,
            extra_benchmark_config={
                "hidden_size": model_info["hidden_size"],
                "vocab_size": model_info["vocab_size"],
                "dtype": model_info["dtype"],
            },
        )
    )


def bench_speed_fused_linear_jsd_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, lm_head = _resolve_model_config_fused_linear_jsd(input)
    mode = input.kernel_operation_mode

    def fwd():
        return lm_head(student_input, teacher_input)

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


def bench_memory_fused_linear_jsd_model_config(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    student_input, teacher_input, lm_head = _resolve_model_config_fused_linear_jsd(input)

    def full():
        y = lm_head(student_input, teacher_input)
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
                    },
                )
                student_input, teacher_input, lm_head = _setup_fused_linear_jsd(probe_input)
                return lm_head(student_input, teacher_input)

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
            "kernel_name": "fused_linear_jsd",
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
            bench_test_fn=bench_speed_fused_linear_jsd_model_config,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_linear_jsd_model_config,
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
                },
            )
            student_input, teacher_input, lm_head = _setup_fused_linear_jsd(probe_input)
            return lm_head(student_input, teacher_input)

        config = compute_seq_len_sweep_config(model, probe_fn=_probe, probe_seq_len=probe_bt)

        common_configs = {
            "kernel_name": "fused_linear_jsd",
            "x_name": "BT",
            "x_label": "B * T",
            "x_values": [2**i for i in range(10, int(math.log2(config.batch_size * config.seq_len)) + 1)],
            "kernel_providers": ["liger", "torch"],
            "extra_benchmark_configs": [
                {
                    "hidden_size": model.hidden_size,
                    "vocab_size": model.vocab_size,
                    "dtype": model.dtype,
                }
            ],
            "overwrite": args.overwrite,
        }

        run_benchmarks(
            bench_test_fn=bench_speed_fused_linear_jsd,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name="speed",
            metric_unit="ms",
            **common_configs,
        )
        run_benchmarks(
            bench_test_fn=bench_memory_fused_linear_jsd,
            kernel_operation_modes=["full"],
            metric_name="memory",
            metric_unit="MB",
            **common_configs,
        )
