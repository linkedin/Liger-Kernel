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

from liger_kernel.transformers.cce import LigerCCELoss
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, hidden, targets):
        return self.loss(self.linear(hidden).float(), targets)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, dtype: torch.dtype, provider: str):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
        self.loss = LigerCCELoss() if provider == "liger-cce" else LigerFusedLinearCrossEntropyLoss()

    def forward(self, hidden, targets):
        return self.loss(self.linear.weight, hidden, targets)


def setup_cce(input: SingleBenchmarkRunInput):
    config = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_config = MODEL_REGISTRY[input.x]
        token_count = config["seq_len"] * config["bsz"]
        vocab_size = model_config.vocab_size
        hidden_size = model_config.hidden_size
        dtype = model_config.dtype
    else:
        token_count = input.x
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        dtype = config["dtype"]

    hidden = torch.randn(token_count, hidden_size, requires_grad=True, dtype=dtype, device=device)
    targets = torch.randint(vocab_size, (token_count,), dtype=torch.long, device=device)
    if input.kernel_provider == "torch":
        module = TorchLMHeadCE(hidden_size, vocab_size, dtype).to(device)
    else:
        module = LigerLMHeadCE(hidden_size, vocab_size, dtype, input.kernel_provider).to(device)
    return hidden, lambda _: module(hidden, targets)


if __name__ == "__main__":
    args = parse_benchmark_script_args()
    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="cce",
            setup_fn=setup_cce,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_provider="torch",
            extra_configs={"eps": 1e-6},
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="cce",
            probe_x=1024,
            model=model,
            setup_fn=setup_cce,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={"eps": 1e-6},
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger", "liger-cce"]
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_cce),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_cce),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
