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

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, accum_dtype=None):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean", accum_dtype=accum_dtype
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


def setup_fused_linear_cross_entropy(input: SingleBenchmarkRunInput):
    """Create input tensor, target, and fused linear CE from benchmark config."""
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

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    if input.kernel_provider == "liger":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif input.kernel_provider == "liger-fp32-accum":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    else:
        lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    return _input, lambda _: lm_head_ce(_input, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_linear_cross_entropy",
            setup_fn=setup_fused_linear_cross_entropy,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_provider="torch",
            extra_configs={
                "eps": 1e-6,
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="fused_linear_cross_entropy",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_fused_linear_cross_entropy,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "eps": 1e-6,
            },
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger", "liger-fp32-accum"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_linear_cross_entropy),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_linear_cross_entropy),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
