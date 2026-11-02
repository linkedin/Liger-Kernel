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

from liger_kernel.transformers import ISCorrection
from liger_kernel.transformers import LigerFusedLinearScoreCenteringLoss
from liger_kernel.utils import infer_device

device = infer_device()

TOPK = 128
TIS = ISCorrection(high=2.0, outside="clamp")


def torch_score_centering_loss(logits, targets, mask, advantages, samp_token_logp, topk_ids, topk_logp, eps=1e-6):
    """Score centering on top of TIS on full logits, as in the score centering README."""
    train_logp = logits.float().log_softmax(-1)
    head_logp = train_logp.gather(-1, topk_ids)
    token_logp = train_logp.gather(-1, targets[..., None])[..., 0]
    p_head, q_head = head_logp.exp(), topk_logp.exp()
    rho = (1 - q_head.sum(-1)).clamp_min(0) / (1 - p_head.sum(-1)).clamp_min(eps)
    alpha = rho * (1 / rho).clamp_max(TIS.high)
    head_weight = (head_logp - topk_logp).exp().clamp_max(TIS.high)
    residual = q_head * head_weight - alpha[..., None] * p_head
    plogp = (train_logp.exp() * train_logp).sum(-1)
    correction = (residual.detach() * head_logp).sum(-1) + (alpha * plogp).detach()
    token_weight = (token_logp - samp_token_logp).exp().clamp_max(TIS.high)
    token_loss = -advantages[:, None] * (token_weight.detach() * token_logp - correction)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


class TorchLMHeadScoreCentering(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)

    def forward(self, x, *args):
        return torch_score_centering_loss(self.lin(x), *args)


class LigerLMHeadScoreCentering(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype, device=device)
        self.loss = LigerFusedLinearScoreCenteringLoss(correction=TIS)

    def forward(self, x, targets, mask, advantages, samp_token_logp, topk_ids, topk_logp):
        return self.loss(x, self.lin.weight, targets, mask, advantages, samp_token_logp, topk_ids, topk_logp)[0]


def setup_fused_linear_score_centering(input: SingleBenchmarkRunInput):
    """Create input tensors and the fused linear score centering loss from benchmark config."""
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

    if input.kernel_provider == "liger":
        lm_head = LigerLMHeadScoreCentering(H=H, V=V, dtype=dtype)
    elif input.kernel_provider == "torch":
        lm_head = TorchLMHeadScoreCentering(H=H, V=V, dtype=dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for FusedLinearScoreCentering")

    # One sequence of BT completion tokens; only the sampler's top-k head and sampled-token logprob are stored.
    _input = torch.randn(1, BT, H, requires_grad=True, dtype=dtype, device=device)
    sampler_logp = torch.randn(1, BT, TOPK + 1, device=device).log_softmax(-1) - 1.0
    topk_ids = torch.randint(0, V, (1, BT, TOPK), device=device)
    args = (
        topk_ids[..., 0],
        torch.ones(1, BT, device=device),
        torch.randn(1, device=device),
        sampler_logp[..., 0],
        topk_ids,
        sampler_logp[..., 1:],
    )
    return _input, lambda _: lm_head(_input, *args)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="fused_linear_score_centering",
            setup_fn=setup_fused_linear_score_centering,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_dim="BT",
            probe_provider="torch",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        common_configs = build_token_length_sweep(
            kernel_name="fused_linear_score_centering",
            probe_x=1024,
            model=model,
            setup_fn=setup_fused_linear_score_centering,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            scale_dim="BT",
            x_label="B * T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_fused_linear_score_centering),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_fused_linear_score_centering),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
