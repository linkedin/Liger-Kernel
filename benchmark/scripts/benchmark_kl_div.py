import torch
import torch.nn as nn

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from liger_kernel.utils import infer_device

device = infer_device()


def setup_kl_div(input: SingleBenchmarkRunInput):
    """Create input tensors and KL div loss from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        BT = cfg["seq_len"] * cfg["bsz"]
        V = model_cfg.vocab_size
    else:
        BT = input.x
        V = cfg["vocab_size"]

    reduction = cfg.get("reduction", "batchmean")
    _input = torch.randn(BT, V, requires_grad=True, device=device).log_softmax(dim=-1)
    target = torch.randn(BT, V, device=device).softmax(dim=-1)

    if input.kernel_provider == "liger":
        loss_fn = LigerKLDIVLoss(reduction=reduction)
    elif input.kernel_provider == "torch":
        loss_fn = nn.KLDivLoss(reduction=reduction)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for KLDiv")
    return _input, lambda _: loss_fn(_input, target)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="kl_div",
            setup_fn=setup_kl_div,
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
            kernel_name="kl_div",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_kl_div,
            model_keys=["vocab_size"],
            extra_configs={
                "reduction": "batchmean",
            },
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_kl_div),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_kl_div),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
