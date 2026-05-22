import torch

from benchmark_model_configs import MODEL_REGISTRY
from benchmark_model_configs import build_model_config_sweep
from benchmark_model_configs import build_token_length_sweep
from benchmark_model_configs import get_benchmark_model_config
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding
from transformers.models.llama4.modeling_llama4 import apply_rotary_emb
from utils import SingleBenchmarkRunInput
from utils import build_memory_bench_fn
from utils import build_speed_bench_fn
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()


def setup_llama4_rope(input: SingleBenchmarkRunInput):
    """Create input tensors and Llama4 RoPE embedding from benchmark config."""
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

    config = Llama4TextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
    )

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        Llama4TextRotaryEmbedding,
        Llama4TextRotaryEmbedding,
        before_kwargs={"config": config, "device": device},
        after_kwargs={"config": config, "device": device},
    )

    q = torch.randn(
        (1, seq_len, num_q_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    )
    k = torch.randn(
        (1, seq_len, num_kv_heads, head_dim),
        device=device,
        requires_grad=True,
        dtype=dtype,
    )
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    freqs_cis = rotary_emb(q, pos_ids)

    if input.kernel_provider == "liger":
        fwd_fn = lambda: liger_llama4_text_rotary_pos_emb(q, k, freqs_cis)
    elif input.kernel_provider == "huggingface":
        fwd_fn = lambda: apply_rotary_emb(q, k, freqs_cis)
    else:
        raise ValueError(f"Invalid provider: {input.kernel_provider} for Llama4 RoPE embedding")

    return q, lambda _: fwd_fn()[0]


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    if args.sweep_mode == "model_config":
        common_configs = build_model_config_sweep(
            kernel_name="llama4_rope",
            setup_fn=setup_llama4_rope,
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
            kernel_name="llama4_rope",
            probe_x=probe_seq_len,
            model=model,
            setup_fn=setup_llama4_rope,
            model_keys=["hidden_size", "num_attention_heads", "num_key_value_heads", "dtype"],
            scale_dim="T",
            x_label="sequence length",
            probe_provider="huggingface",
            overwrite=args.overwrite,
        )

    common_configs["kernel_providers"] = ["huggingface", "liger"]

    run_benchmarks(
        bench_test_fn=build_speed_bench_fn(setup_llama4_rope),
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=build_memory_bench_fn(setup_llama4_rope),
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
