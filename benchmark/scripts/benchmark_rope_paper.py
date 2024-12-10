import torch
import triton
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfAttention,
    RoFormerSinusoidalPositionalEmbedding,
)
from utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.rope_paper import liger_rotary_paper_pos_emb
from liger_kernel.utils import infer_device

device = infer_device()
apply_rotary_pos_emb = RoFormerSelfAttention.apply_rotary_position_embeddings


def bench_speed_rope_paper(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    num_q_heads = extra_benchmark_config["num_q_heads"]
    num_kv_heads = extra_benchmark_config["num_kv_heads"]
    dtype = extra_benchmark_config["dtype"]

    # x can be either hidden_size or seq_len
    hidden_size = (
        extra_benchmark_config["hidden_size"]
        if "hidden_size" in extra_benchmark_config
        else input.x
    )
    seq_len = (
        extra_benchmark_config["seq_len"]
        if "seq_len" in extra_benchmark_config
        else input.x
    )

    head_dim = hidden_size // num_q_heads
    rotary_emb = RoFormerSinusoidalPositionalEmbedding(
        num_positions=seq_len, embedding_dim=head_dim
    ).to(device)
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
    dq, dk = torch.randn_like(q, device=device, dtype=dtype), torch.randn_like(
        k, device=device
    )

    sinusoidal_pos = rotary_emb((1, seq_len))[None, :, :].to(dtype)
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    def fwd():
        if provider == "liger":
            return liger_rotary_paper_pos_emb(q, k, cos, sin)
        elif provider == "huggingface":
            return apply_rotary_pos_emb(sinusoidal_pos[None, :, :, :], q, k)
        else:
            raise ValueError(f"Invalid provider: {provider} for RoPE paper embedding")

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        q_out, k_out = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: torch.autograd.grad(
                (q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True
            ),
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            q_out, k_out = fwd()
            torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[q, k],
            rep=400,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_rope_paper(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    num_q_heads = extra_benchmark_config["num_q_heads"]
    num_kv_heads = extra_benchmark_config["num_kv_heads"]
    dtype = extra_benchmark_config["dtype"]

    # x can be either hidden_size or seq_len
    hidden_size = (
        extra_benchmark_config["hidden_size"]
        if "hidden_size" in extra_benchmark_config
        else input.x
    )
    seq_len = (
        extra_benchmark_config["seq_len"]
        if "seq_len" in extra_benchmark_config
        else input.x
    )

    head_dim = hidden_size // num_q_heads
    rotary_emb = RoFormerSinusoidalPositionalEmbedding(
        num_positions=seq_len, embedding_dim=head_dim
    ).to(device)
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
    dq, dk = torch.randn_like(q, device=device, dtype=dtype), torch.randn_like(
        k, device=device
    )

    sinusoidal_pos = rotary_emb((1, seq_len))[None, :, :].to(dtype)
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    def full():
        if provider == "liger":
            q_out, k_out = liger_rotary_paper_pos_emb(q, k, cos, sin)
        else:
            q_out, k_out = apply_rotary_pos_emb(sinusoidal_pos[None, :, :, :], q, k)
        torch.autograd.grad(
            (q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True
        )

    mem_50, mem_20, mem_80 = _test_memory(
        full,
        quantiles=QUANTILES,
    )
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs_varying_hidden_size = {
        "kernel_name": "rope_paper",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [32 * (2**i) for i in range(4, 10, 2)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "dtype": torch.bfloat16,
                "seq_len": 2048,
                "num_q_heads": 32,
                "num_kv_heads": 8,
            }
        ],
        "overwrite": args.overwrite,
    }
    run_benchmarks(
        bench_test_fn=bench_speed_rope_paper,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs_varying_hidden_size,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_rope_paper,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs_varying_hidden_size,
    )

    common_configs_varying_seq_len = {
        "kernel_name": "rope_paper",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 15)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "dtype": torch.bfloat16,
                "hidden_size": 8192,
                "num_q_heads": 32,
                "num_kv_heads": 8,
            }
        ],
        "overwrite": args.overwrite,
    }
    run_benchmarks(
        bench_test_fn=bench_speed_rope_paper,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs_varying_seq_len,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_rope_paper,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs_varying_seq_len,
    )
