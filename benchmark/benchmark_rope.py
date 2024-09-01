import os
from typing import List

import torch
import triton
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from utils import (
    QUANTILES,
    _print_memory_banner,
    _print_speed_banner,
    _test_memory,
    get_current_file_directory,
)

from liger_kernel.transformers.rope import liger_rotary_pos_emb


def _get_perf_configs(target: str, ylabel: str, modes: List[str] = ["full"]):
    perf_configs = []
    for mode in modes:
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=["total_hidden_size"],
                x_vals=[32 * (2**i) for i in range(4, 10, 2)],
                line_arg="provider",
                line_vals=["liger", "huggingface"],
                line_names=["Liger", "Hugging Face"],
                styles=[("blue", "solid"), ("orange", "solid")],
                ylabel=ylabel,
                plot_name=f"rope-{mode}-{target}-benchmark-seq-2048",
                args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
            )
        )
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=["seq_len"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["liger", "huggingface"],
                line_names=["Liger", "Hugging Face"],
                styles=[("blue", "solid"), ("orange", "solid")],
                ylabel=ylabel,
                plot_name=f"rope-{mode}-{target}-benchmark-total_dim_8192",
                args={"dtype": torch.bfloat16, "mode": mode, "total_hidden_size": 8192},
            )
        )
    return perf_configs


@triton.testing.perf_report(
    _get_perf_configs(
        target="speed", ylabel="time (ms)", modes=["forward", "backward", "full"]
    )
)
def bench_speed_rope(total_hidden_size, seq_len, provider, mode, dtype):
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = total_hidden_size // num_q_heads
    rotary_emb = LlamaRotaryEmbedding(head_dim, device="cuda")
    q = torch.randn(
        (1, seq_len, num_q_heads, head_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    k = torch.randn(
        (1, seq_len, num_kv_heads, head_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    dq, dk = torch.randn_like(q, device="cuda", dtype=dtype), torch.randn_like(
        k, device="cuda"
    )
    pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k, pos_ids)

    def fwd():
        if provider == "liger":
            return liger_rotary_pos_emb(q, k, cos, sin, pos_ids)
        elif provider == "huggingface":
            return apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        else:
            raise ValueError(f"Invalid provider: {provider} for RoPE embedding")

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fwd, quantiles=QUANTILES, grad_to_none=[q, k], rep=400
        )
    elif mode == "backward":
        q_out, k_out = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.autograd.grad(
                (q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True
            ),
            quantiles=QUANTILES,
            grad_to_none=[q, k],
            rep=400,
        )
    elif mode == "full":

        def full():
            q_out, k_out = fwd()
            torch.autograd.grad((q_out, k_out), (q, k), (dq, dk), allow_unused=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            full, quantiles=QUANTILES, grad_to_none=[q, k], rep=400
        )
    return ms, min_ms, max_ms


def benchmark_speed_rope_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "rope_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_rope.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    benchmarks=_get_perf_configs(target="memory", ylabel="GPU memory usage (MB)")
)
def bench_memory_rope(total_hidden_size, seq_len, provider, mode, dtype):
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = total_hidden_size // num_q_heads
    rotary_emb = LlamaRotaryEmbedding(head_dim, device="cuda")
    q = torch.randn(
        (1, seq_len, num_q_heads, head_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    k = torch.randn(
        (1, seq_len, num_kv_heads, head_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    ).transpose(1, 2)
    dq, dk = torch.randn_like(q, device="cuda", dtype=dtype), torch.randn_like(
        k, device="cuda"
    )
    pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k, pos_ids)

    def full():
        if provider == "liger":
            q_out, k_out = liger_rotary_pos_emb(q, k, cos, sin, pos_ids)
        else:
            q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        torch.autograd.grad(
            (q_out, k_out), (q, k), (dq, dk), allow_unused=True, retain_graph=True
        )

    mem, min_mem, max_mem = _test_memory(full, quantiles=QUANTILES)
    return (mem / 2**20, min_mem / 2**20, max_mem / 2**20)


def benchmark_memory_rope_wrapper():
    _print_memory_banner()

    curr_dir = get_current_file_directory()
    output_dir = os.path.join(curr_dir, "rope_memory")
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_rope.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_rope_wrapper()
    benchmark_memory_rope_wrapper()
