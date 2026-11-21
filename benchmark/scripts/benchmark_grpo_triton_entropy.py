#!/usr/bin/env python3
# BSD 2-Clause License
# Copyright (c) 2024-2026 LinkedIn Corporation. All rights reserved.
"""Micro-benchmark comparing GRPO + Token Entropy memory & latency across sequence lengths.

Follows `linkedin/Liger-Kernel/docs/contributing.md` (`benchmark/scripts/benchmark_grpo_loss.py`):
Compares three realistic implementations on large-vocabulary RLVR shapes (default `V = 151,936`):
1. `torch_materialized`: Materializes FP32 `log_softmax` + `exp` for both `log_probs` and `entropy`.
2. `two_pass_prepass` (`#1558` baseline): Runs an extra `no_grad` vocabulary pass for `entropy`
   before running selective `log_softmax` + backward.
3. `liger_triton_single_pass` (This PR): Computes `(log_probs, entropy)` and fused backward in
   a single streaming Triton vocabulary scan (`_selective_log_softmax_kernel`).

Usage on GPU machine (via Antigravity):
    python3 benchmark_grpo_triton_entropy.py --vocab-size 151936 --dtype bf16 --csv benchmark_grpo_entropy.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liger_kernel.ops._grpo_online_entropy import HAS_TORCH_TRITON

if HAS_TORCH_TRITON:
    import torch
    import torch.nn.functional as F
    import triton

    from liger_kernel.ops._grpo_online_entropy import selective_log_softmax_with_entropy


def _measure_peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_delta = max(0, torch.cuda.max_memory_allocated() - base_bytes)
    return peak_delta / (1024.0 * 1024.0)


def run_gpu_benchmark(
    seq_lens: list[int],
    batch_size: int,
    vocab_size: int,
    dtype_str: str,
    temperature: float,
    entropy_requires_grad: bool,
    csv_path: str | None,
) -> None:
    if not HAS_TORCH_TRITON or not torch.cuda.is_available():
        print(
            "CUDA + PyTorch + Triton required for live GPU timing. "
            "Run this script on your GPU host (or let Antigravity execute it) to generate the PR table."
        )
        return

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16 if dtype_str == "fp16" else torch.float32
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | Batch={batch_size}, Vocab={vocab_size}, Dtype={dtype_str}, T={temperature}")
    print(f"{'SeqLen':>8} | {'Provider':<26} | {'Fwd+Bwd (ms)':>13} | {'Extra Peak HBM (MB)':>20}")
    print("-" * 76)

    rows_out: list[dict[str, object]] = []

    for seq_len in seq_lens:
        logits_base = torch.randn(batch_size, seq_len, vocab_size, device="cuda", dtype=dtype)
        index = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)
        mask = torch.ones((batch_size, seq_len), device="cuda", dtype=torch.long)
        g_lp = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)
        g_ent = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)

        def run_torch_materialized():
            z = logits_base.detach().requires_grad_(True)
            lps = F.log_softmax(z.float() / temperature, dim=-1)
            ps = torch.exp(lps)
            sel_lp = lps.gather(-1, index.unsqueeze(-1)).squeeze(-1)
            ent = -(ps * lps).sum(dim=-1)
            loss = (sel_lp * g_lp).sum() + (ent * g_ent if entropy_requires_grad else 0.0).sum()
            loss.backward()

        def run_two_pass_prepass():
            z = logits_base.detach().requires_grad_(True)
            # Pass 1: Extra streaming pre-pass for entropy (#1558 workaround when Triton lacks entropy)
            with torch.set_grad_enabled(entropy_requires_grad):
                _, ent = selective_log_softmax_with_entropy(
                    z,
                    index,
                    temperature=temperature,
                    mask=mask,
                    return_entropy=True,
                    entropy_requires_grad=entropy_requires_grad,
                )
            # Pass 2: Separate selective_log_softmax pass for log_probs
            sel_lp, _ = selective_log_softmax_with_entropy(
                z, index, temperature=temperature, mask=mask, return_entropy=False
            )
            loss = (sel_lp * g_lp).sum() + (ent * g_ent if entropy_requires_grad else 0.0).sum()
            loss.backward()

        def run_liger_single_pass():
            z = logits_base.detach().requires_grad_(True)
            sel_lp, ent = selective_log_softmax_with_entropy(
                z,
                index,
                temperature=temperature,
                mask=mask,
                return_entropy=True,
                entropy_requires_grad=entropy_requires_grad,
            )
            loss = (sel_lp * g_lp).sum() + (ent * g_ent if entropy_requires_grad else 0.0).sum()
            loss.backward()

        for name, fn in (
            ("torch_materialized", run_torch_materialized),
            ("two_pass_entropy_prepass", run_two_pass_prepass),
            ("liger_triton_single_pass", run_liger_single_pass),
        ):
            try:
                peak_mb = _measure_peak_mb(fn)
                ms = triton.testing.do_bench(fn, warmup=10, rep=30)
                print(f"{seq_len:>8} | {name:<26} | {ms:>13.3f} | {peak_mb:>20.2f}")
                rows_out.append(
                    {
                        "kernel_name": "grpo_selective_log_softmax_entropy",
                        "kernel_provider": name,
                        "kernel_operation_mode": "full",
                        "x_name": "T",
                        "x_value": seq_len,
                        "latency_ms": round(float(ms), 4),
                        "extra_peak_mb": round(float(peak_mb), 2),
                        "vocab_size": vocab_size,
                        "dtype": dtype_str,
                        "gpu_name": gpu_name,
                    }
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    print(f"{seq_len:>8} | {name:<26} | {'OOM':>13} | {'OOM':>20}")
                    rows_out.append(
                        {
                            "kernel_name": "grpo_selective_log_softmax_entropy",
                            "kernel_provider": name,
                            "kernel_operation_mode": "full",
                            "x_name": "T",
                            "x_value": seq_len,
                            "latency_ms": "OOM",
                            "extra_peak_mb": "OOM",
                            "vocab_size": vocab_size,
                            "dtype": dtype_str,
                            "gpu_name": gpu_name,
                        }
                    )
                else:
                    raise

    if csv_path and rows_out:
        out_file = Path(csv_path)
        with out_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nSaved benchmark CSV to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Liger single-pass Triton GRPO entropy kernel")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--detached-entropy", action="store_true", help="Benchmark no-grad entropy (DAPO masking)")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    run_gpu_benchmark(
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        dtype_str=args.dtype,
        temperature=args.temperature,
        entropy_requires_grad=not args.detached_entropy,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
