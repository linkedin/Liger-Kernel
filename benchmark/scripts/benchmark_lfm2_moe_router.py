import torch
import torch.nn as nn

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from test.transformers.test_lfm2_moe_router import _reference
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops import LigerLfm2MoeRouterFunction
from liger_kernel.utils import infer_device

device = infer_device()


class _Router(nn.Module):
    def __init__(self, num_experts, top_k, dtype, use_liger):
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(num_experts, device=device, dtype=torch.float32))
        self.top_k = top_k
        self.use_liger = use_liger
        self.dtype = dtype

    def forward(self, router_logits):
        if self.use_liger:
            return LigerLfm2MoeRouterFunction.apply(
                router_logits,
                self.expert_bias,
                self.top_k,
                True,
                1.0,
            )[1]
        return _reference(router_logits, self.expert_bias, self.top_k, True, 1.0)[1]


def setup_lfm2_moe_router(input: SingleBenchmarkRunInput):
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        num_tokens = cfg["bsz"] * cfg["seq_len"]
        num_experts = model_cfg.num_experts
        top_k = model_cfg.topk
        dtype = model_cfg.dtype
    else:
        num_tokens = cfg["bsz"] * input.x
        num_experts = cfg["num_experts"]
        top_k = cfg["topk"]
        dtype = cfg["dtype"]

    if num_experts is None or top_k is None:
        raise ValueError("LFM2 MoE router benchmarks require an MoE model configuration")

    router_logits = torch.randn(
        num_tokens,
        num_experts,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if input.kernel_provider == "liger":
        layer = _Router(num_experts, top_k, dtype, use_liger=True)
    elif input.kernel_provider == "huggingface":
        layer = _Router(num_experts, top_k, dtype, use_liger=False)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for LFM2 MoE router")
    return router_logits, layer


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        moe_configs = [config for config in MODEL_REGISTRY.values() if config.is_moe]
        common_configs = build_model_config_sweep(
            kernel_name="lfm2_moe_router",
            all_model_configs=moe_configs,
            setup_fn=setup_lfm2_moe_router,
            model_keys=["num_experts", "topk", "dtype"],
            probe_provider="huggingface",
            extra_configs={"bsz": 1},
            probe_dim="T",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model or "lfm2_moe_8b_a1b")
        common_configs = build_token_length_sweep(
            kernel_name="lfm2_moe_router",
            probe_x=1024,
            model=model,
            setup_fn=setup_lfm2_moe_router,
            model_keys=["num_experts", "topk", "dtype"],
            extra_configs={"bsz": 1},
            scale_dim="T",
            x_label="total tokens",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["huggingface", "liger"]
    for metric_name, metric_unit, bench_fn in (
        ("speed", "ms", build_speed_bench_fn(setup_lfm2_moe_router)),
        ("memory", "MB", build_memory_bench_fn(setup_lfm2_moe_router)),
    ):
        run_benchmarks(
            bench_test_fn=bench_fn,
            kernel_operation_modes=["forward", "backward", "full"],
            metric_name=metric_name,
            metric_unit=metric_unit,
            **common_configs,
        )
