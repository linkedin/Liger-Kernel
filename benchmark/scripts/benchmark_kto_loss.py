import os
import sys

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

from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss
from liger_kernel.utils import infer_device

device = infer_device()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TorchLMHeadKTO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        use_bias: bool = False,
        use_ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        from test.chunked_loss.test_kto_loss import HFKTOLoss

        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=use_bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=use_ref_bias, dtype=dtype)
        self.KTO_loss = HFKTOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y, preference_labels, kl=None):
        return self.KTO_loss(
            weight=self.lin.weight,
            _input=x,
            target=y,
            bias=self.lin.bias,
            ref_input=ref_x,
            ref_weight=self.ref_lin.weight,
            ref_bias=self.ref_lin.bias,
            preference_labels=preference_labels,
            kl=kl,
        )


class LigerLMHeadKTO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        use_bias: bool = False,
        use_ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=use_bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=use_ref_bias, dtype=dtype)
        self.KTO_loss = LigerFusedLinearKTOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
        )

    def forward(self, x, ref_x, y, preference_labels, kl=None):
        return self.KTO_loss(
            _input=x,
            lin_weight=self.lin.weight,
            target=y,
            preference_labels=preference_labels,
            bias=self.lin.bias,
            ref_input=ref_x,
            ref_weight=self.ref_lin.weight,
            ref_bias=self.ref_lin.bias,
            kl=kl,
        )


def setup_kto_loss(input: SingleBenchmarkRunInput):
    """Create input tensors and KTO loss from benchmark config."""
    cfg = input.extra_benchmark_config
    T = cfg["T"]
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        H = model_cfg.hidden_size
        V = model_cfg.vocab_size
        dtype = model_cfg.dtype
        B = cfg["bsz"]
    else:
        B = input.x
        H = cfg["hidden_size"]
        V = cfg["vocab_size"]
        dtype = cfg["dtype"]

    bias = cfg["bias"]
    beta = cfg["beta"]
    ignore_index = cfg["ignore_index"]

    # Input shape: [B, T, H]
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype)
    target = torch.randint(V, (B, T), dtype=torch.long, device=device)
    # Preference labels shape: [B]
    # Create binary preference labels (0 or 1) for each sequence in the batch
    # Used to indicate preferred sequences (1) vs non-preferred sequences (0)
    preference_labels = torch.randint(2, (B,), dtype=torch.bool, device=device)
    # Precomputed KL divergence between policy and reference distributions
    kl = torch.randn(1, device=device, dtype=dtype)

    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    if input.kernel_provider == "liger":
        loss_module = LigerLMHeadKTO(
            H=H, V=V, dtype=dtype, use_bias=bias, use_ref_bias=bias, ignore_index=ignore_index, beta=beta
        ).to(device)
    elif input.kernel_provider == "huggingface":
        loss_module = TorchLMHeadKTO(
            H=H, V=V, dtype=dtype, use_bias=bias, use_ref_bias=bias, ignore_index=ignore_index, beta=beta
        ).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for KTOLoss")

    fwd = lambda _input: loss_module(x=_input, ref_x=ref_input, y=target, preference_labels=preference_labels, kl=kl)[0]
    return _input, fwd


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    T = 1024

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="kto_loss",
            setup_fn=setup_kto_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "bias": True,
                "beta": 0.1,
                "ignore_index": 42,
            },
            probe_dim="B",
            probe_provider="huggingface",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)

        common_configs = build_token_length_sweep(
            kernel_name="kto_loss",
            probe_x=1,
            model=model,
            setup_fn=setup_kto_loss,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
                "bias": True,
                "beta": 0.1,
                "ignore_index": 42,
            },
            scale_dim="B",
            x_label="batch size",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["liger", "huggingface"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_kto_loss),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_kto_loss),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
