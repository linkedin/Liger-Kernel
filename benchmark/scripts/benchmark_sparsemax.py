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

from liger_kernel.transformers.sparsemax import LigerSparsemax
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def torch_sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_dims = input_tensor.dim()
    if dim < 0:
        dim = input_dims + dim
    input_sorted, _ = torch.sort(input_tensor, dim=dim, descending=True)
    cumsum_input = torch.cumsum(input_sorted, dim=dim)
    input_size = input_tensor.size(dim)
    range_tensor = torch.arange(1, input_size + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    shape = [1] * input_dims
    shape[dim] = input_size
    range_tensor = range_tensor.view(shape)
    k_bound = 1 + range_tensor * input_sorted
    support = k_bound > cumsum_input
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    support_sum = (input_sorted * support).sum(dim=dim, keepdim=True)
    tau = (support_sum - 1) / k
    return torch.clamp(input_tensor - tau, min=0)


class TorchSparsemax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_sparsemax(x, dim=self.dim)


def setup_sparsemax(input: SingleBenchmarkRunInput):
    """Create input tensors and sparsemax module from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        bt = cfg["seq_len"] * cfg["bsz"]
        hidden_size = model_cfg.hidden_size
        dtype = model_cfg.dtype
    else:
        bt = input.x
        hidden_size = cfg["hidden_size"]
        dtype = cfg["dtype"]

    dim = cfg.get("dim", -1)

    x = torch.randn(bt, hidden_size, dtype=dtype, device=device, requires_grad=True)

    if input.kernel_provider == "liger":
        sparsemax_module = LigerSparsemax(dim=dim).to(device)
    elif input.kernel_provider == "torch":
        sparsemax_module = TorchSparsemax(dim=dim).to(device)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for sparsemax")

    return x, sparsemax_module


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="sparsemax",
            setup_fn=setup_sparsemax,
            model_keys=["hidden_size", "dtype"],
            probe_provider="torch",
            extra_configs={
                "dim": -1,
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 2048

        common_configs = build_token_length_sweep(
            kernel_name="sparsemax",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_sparsemax,
            model_keys=["hidden_size", "dtype"],
            extra_configs={
                "dim": -1,
            },
            scale_dim="BT",
            x_label="B*T",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "liger"]
    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_sparsemax),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_sparsemax),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
