import os

import torch
import triton
from torch.nn import Embedding
from utils import _test_memory, get_current_file_directory

from liger_kernel.transformers.embedding import LigerEmbedding


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(10, 18)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=[
                "Liger",
                "Hugging Face",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="embedding-fwd-speed-benchmark",
            args={
                "B": 32, 
                "T": 512, 
                "D": 768, 
                "mode": "forward", 
                "dtype": torch.float32
            },
        ),
    ]
)
def bench_speed_embedding(B, T, V, D, provider, mode, dtype, device="cuda"):
    torch_emb = torch.nn.Embedding(V, D).to(device).to(dtype)
    liger_emb = LigerEmbedding(V, D).to(device).to(dtype)

    input_ids = torch.randint(0, V, (B, T), device=device)

    def fwd():
        if provider == "liger":
            return liger_emb(input_ids)
        else:
            return torch_emb(input_ids)

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
    return ms, min_ms, max_ms

@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(10, 18)], 
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=[
                "Liger",
                "Hugging Face",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="GPU memory usage (MB)",
            plot_name="embedding-memory-benchmark",
            args={
                "B": 32, 
                "T": 512, 
                "D": 768, 
                "mode": "full", 
                "dtype": torch.float32
            },
        ),
    ]
)
def bench_memory_embedding(B, T, V, D, provider, mode, dtype, device="cuda"):
    torch_emb = torch.nn.Embedding(V, D).to(device).to(dtype)
    liger_emb = LigerEmbedding(V, D).to(device).to(dtype)

    input_ids = torch.randint(0, V, (B, T), device=device)

    def fwd():
        if provider == "liger":
            return liger_emb(input_ids)
        else:
            return torch_emb(input_ids)

    def full():
        output = fwd()
        output.backward(torch.randn_like(output))

    mem = _test_memory(full)
    return mem / 2**20

def benchmark_speed_embedding_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "embedding_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_speed_embedding.run(save_path=output_dir, print_data=True)

def benchmark_memory_embedding_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "embedding_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_memory_embedding.run(save_path=output_dir, print_data=True)

if __name__ == "__main__":
    benchmark_speed_embedding_wrapper()
    benchmark_memory_embedding_wrapper()
