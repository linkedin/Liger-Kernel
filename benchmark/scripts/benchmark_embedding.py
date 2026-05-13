import os
import sys

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from torch.nn import Embedding
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.experimental.embedding import LigerEmbedding
from liger_kernel.utils import infer_device

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# NOTE: For torch compile, we will just use default inductor settings. No further customization
# is needed.


def setup_embedding(input: SingleBenchmarkRunInput):
    """Create input tensors and embedding module from benchmark config."""
    cfg = input.extra_benchmark_config
    T = cfg["T"]
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        BT = cfg["seq_len"] * cfg["bsz"]
        D = model_cfg.hidden_size
        V = model_cfg.vocab_size
        dtype = model_cfg.dtype
    else:
        BT = input.x
        D = cfg["hidden_size"]
        V = cfg["vocab_size"]
        dtype = cfg["dtype"]

    B = max(1, BT // T) if "BT" not in cfg else BT // T

    input_ids = torch.randint(0, V, (B, T), device=device)

    if input.kernel_provider == "liger":
        emb = LigerEmbedding(V, D).to(device).to(dtype)
    elif input.kernel_provider == "torch_compile":
        emb = torch.compile(Embedding(V, D).to(device).to(dtype))
    elif input.kernel_provider == "torch":
        emb = Embedding(V, D).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for embedding")

    return input_ids, emb


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    T = 1024
    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="embedding",
            setup_fn=setup_embedding,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            probe_provider="torch",
            extra_configs={
                "T": T,
            },
            probe_dim="BT",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 1024

        common_configs = build_token_length_sweep(
            kernel_name="embedding",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_embedding,
            model_keys=["hidden_size", "vocab_size", "dtype"],
            extra_configs={
                "T": T,
            },
            scale_dim="BT",
            x_label="total tokens",
            probe_provider="torch",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["torch", "torch_compile", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_embedding),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_embedding),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
