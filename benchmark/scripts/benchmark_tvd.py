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

from liger_kernel.transformers.tvd import LigerTVDLoss
from liger_kernel.utils import infer_device

device = infer_device()


class TorchTVDLoss(torch.nn.Module):
    def __init__(self, reduction="batchmean"):
        super(TorchTVDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        tvd = torch.abs(p - q) / 2.0
        if self.reduction == "mean":
            return torch.sum(tvd) / (p.size(0) * p.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd)
        elif self.reduction == "none":
            return tvd
        elif self.reduction == "batchmean":
            return torch.sum(tvd) / p.size(0)
        else:
            raise ValueError("Invalid reduction type.")


def setup_tvd(input: SingleBenchmarkRunInput):
    """Create input tensors and TVD loss from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        BT = cfg["seq_len"] * cfg["bsz"]
        V = model_cfg.vocab_size
    else:
        BT = input.x
        V = cfg["vocab_size"]

    reduction = cfg.get("reduction", "batchmean")

    _input = torch.randn(BT, V, requires_grad=True, device=device).softmax(dim=-1)
    target = torch.randn(BT, V, device=device).softmax(dim=-1)

    if input.kernel_provider == "liger":
        loss_fn = LigerTVDLoss(reduction=reduction)
    elif input.kernel_provider == "torch":
        loss_fn = TorchTVDLoss(reduction=reduction)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for TVD")
    return _input, lambda x: loss_fn(x, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="tvd",
            setup_fn=setup_tvd,
            model_keys=["vocab_size"],
            probe_provider="torch",
            extra_configs={
                "reduction": "batchmean",
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="tvd",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_tvd,
            model_keys=["vocab_size"],
            extra_configs={"reduction": "batchmean"},
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_tvd),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_tvd),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
