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

from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy_sm90 import LigerFusedLinearCrossEntropySM90Function
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

try:
    from quack.linear_cross_entropy import chunked_linear_cross_entropy
except ImportError:
    chunked_linear_cross_entropy = None

device = infer_device()


class TorchLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

    def forward(self, x, target):
        return torch.nn.functional.cross_entropy(self.linear(x), target)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)
        self.cross_entropy = LigerFusedLinearCrossEntropyLoss()

    def forward(self, x, target):
        return self.cross_entropy(self.linear.weight, x, target)


class CuteDSLHopperLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, dtype):
        super().__init__()
        if dtype != torch.bfloat16:
            raise TypeError("CuTe DSL SM90 FLCE requires bfloat16")
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype))
        torch.nn.init.normal_(self.weight, std=hidden_size**-0.5)

    def forward(self, x, target):
        return LigerFusedLinearCrossEntropySM90Function.apply(x, self.weight, target)


class QuackLMHeadCE(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, dtype, tokens):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype))
        self.tokens = tokens
        torch.nn.init.normal_(self.weight, std=hidden_size**-0.5)

    def forward(self, x, target):
        return chunked_linear_cross_entropy(
            x,
            self.weight,
            target,
            chunk_size=self.tokens,
        )


def setup_cutedsl_fused_linear_cross_entropy_sm90(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        tokens = cfg["seq_len"] * cfg["bsz"]
        vocab_size = model_cfg.vocab_size
        hidden_size = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        tokens = input.x
        vocab_size = cfg["vocab_size"]
        hidden_size = cfg["hidden_size"]
        dtype = cfg["dtype"]

    x = torch.randn(tokens, hidden_size, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(vocab_size, (tokens,), dtype=torch.long, device=device)

    if input.kernel_provider == "cutedsl-sm90":
        module = CuteDSLHopperLMHeadCE(hidden_size, vocab_size, dtype)
    elif input.kernel_provider == "liger":
        module = LigerLMHeadCE(hidden_size, vocab_size, dtype)
    elif input.kernel_provider == "quack":
        module = QuackLMHeadCE(hidden_size, vocab_size, dtype, tokens)
    else:
        module = TorchLMHeadCE(hidden_size, vocab_size, dtype)
    module = module.to(device)
    return x, lambda _: module(x, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="cutedsl_fused_linear_cross_entropy_sm90",
            setup_fn=setup_cutedsl_fused_linear_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_provider="torch",
            extra_configs={},
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="cutedsl_fused_linear_cross_entropy_sm90",
            probe_x=1024,
            model=model,
            setup_fn=setup_cutedsl_fused_linear_cross_entropy_sm90,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={},
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    providers = ["torch", "liger", "cutedsl-sm90"]
    if chunked_linear_cross_entropy is not None:
        providers.append("quack")
    common_configs["kernel_providers"] = providers

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_cutedsl_fused_linear_cross_entropy_sm90),
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_cutedsl_fused_linear_cross_entropy_sm90),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
