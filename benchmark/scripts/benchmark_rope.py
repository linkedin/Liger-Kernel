import os
import sys

import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def setup_rope(input: SingleBenchmarkRunInput):
    """Create input tensors and RoPE embedding from benchmark config."""
    cfg = input.extra_benchmark_config
    if isinstance(input.x, str):
        model_cfg = MODEL_REGISTRY[input.x]
        seq_len = cfg["seq_len"]
        hidden_size = model_cfg.hidden_size
        num_q_heads = model_cfg.num_attention_heads
        num_kv_heads = model_cfg.num_key_value_heads
        dtype = model_cfg.dtype
    else:
        seq_len = input.x
        hidden_size = cfg["hidden_size"]
        num_q_heads = cfg["num_attention_heads"]
        num_kv_heads = cfg["num_key_value_heads"]
        dtype = cfg["dtype"]

    head_dim = hidden_size // num_q_heads
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )
    q = torch.randn(
        (1, seq_len, num_q_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    k = torch.randn(
        (1, seq_len, num_kv_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k, pos_ids)

    if input.kernel_provider == "liger":
        fwd_fn = lambda: liger_rotary_pos_emb(q, k, cos, sin, pos_ids)
    elif input.kernel_provider == "huggingface":
        fwd_fn = lambda: apply_rotary_pos_emb(q, k, cos, sin)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for RoPE embedding")

    return q, lambda _: fwd_fn()[0]


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="rope",
            setup_fn=setup_rope,
            model_keys=["hidden_size", "num_attention_heads", "num_key_value_heads", "dtype"],
            probe_provider="huggingface",
            probe_dim="T",
            bt=args.bt,
            overwrite=args.overwrite,
        )
    else:
        model = get_benchmark_model_config(args.model)
        probe_seq_len = 2048

        common_configs = build_token_length_sweep(
            kernel_name="rope",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_rope,
            model_keys=["hidden_size", "num_attention_heads", "num_key_value_heads", "dtype"],
            scale_dim="T",
            x_label="sequence length",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["huggingface", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_rope),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_rope),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
