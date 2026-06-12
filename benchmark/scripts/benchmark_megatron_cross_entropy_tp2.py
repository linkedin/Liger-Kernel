"""Benchmark Liger's Megatron CE at tensor_model_parallel_size=2.

Spawns 2 ranks via ``torch.multiprocessing.spawn``. Each rank holds a per-token
shape of ``[S, B, V_global/2]`` and runs forward / backward / full timing via
``triton.testing.do_bench``. AllReduces are real NCCL collectives across the
two ranks.

Compared providers (each rank executes its own slice):

  - **liger**: ``LigerMegatronCrossEntropy`` — Liger's vocab-parallel kernel
  - **megatron**: Megatron's *fused* ``fused_vocab_parallel_cross_entropy``
    (``cross_entropy_loss_fusion=True``, JIT-fused via TorchScript)
  - **megatron-unfused**: Megatron's *unfused* ``vocab_parallel_cross_entropy``
    (``cross_entropy_loss_fusion=False``, eager Python)

Rank 0 writes results to the shared ``benchmark/data/all_benchmark_data.csv``
tagged with ``kernel_name="megatron_cross_entropy_tp2"``; render via:

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_cross_entropy_tp2 --metric-name speed
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from utils import BenchmarkData
from utils import get_formatted_time
from utils import get_gpu_name
from utils import update_benchmark_data_csv

from liger_kernel.megatron import LigerMegatronCrossEntropy

try:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    _MEGATRON_AVAILABLE = True
except ImportError:
    fused_vocab_parallel_cross_entropy = None
    vocab_parallel_cross_entropy = None
    _MEGATRON_AVAILABLE = False


def _select_fwd(provider, tp_group):
    if provider == "liger":
        ce = LigerMegatronCrossEntropy()
        return lambda logits, target: ce(logits, target, tp_group=tp_group)
    if provider == "torch":

        def torch_ce(logits, target):
            s, b, v = logits.shape
            return F.cross_entropy(
                logits.reshape(-1, v).float(),
                target.reshape(-1),
                reduction="none",
            ).reshape(s, b)

        return torch_ce
    if provider == "megatron":
        return lambda logits, target: fused_vocab_parallel_cross_entropy(logits, target, tp_group)
    if provider == "megatron-unfused":
        return lambda logits, target: vocab_parallel_cross_entropy(logits, target, 0.0, tp_group)
    raise ValueError(f"unknown provider: {provider!r}")


_WARMUP_ITERS = 10
_MEASURE_ITERS = 50


def _timed_loop(fn, tp_group, n_iters: int):
    """Run ``fn`` ``n_iters`` times and return per-iter ms timings.

    CRITICAL: every rank MUST execute the same number of iterations because
    ``fn`` issues NCCL collectives. ``triton.testing.do_bench`` calibrates the
    iteration count from the first-call latency, which differs slightly between
    ranks and causes collective desync (NCCL watchdog timeout). This manual
    loop with a fixed ``n_iters`` keeps every rank in lockstep.

    We barrier + cuda-sync before and between events so the timing reflects
    only the iteration loop and not setup drift between ranks.
    """
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / n_iters


def _bench_one(provider, mode, s, b, v_global, tp_size, tp_group, device):
    """Time forward / backward / full for one provider on this rank.

    Returns the median per-iteration time in ms, plus 20th/80th percentiles
    (we re-run multiple short timed loops and take quantiles across them so
    the output schema matches the standard benchmark CSV).
    """
    v_local = v_global // tp_size
    logits = torch.randn(s, b, v_local, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, v_global, (s, b), device=device, dtype=torch.long)
    fwd_fn = _select_fwd(provider, tp_group)

    if mode == "forward":

        def step():
            fwd_fn(logits, target)

    elif mode == "backward":
        # Megatron's fused CE writes gradients in-place during backward; rerun fwd each
        # iteration so each backward sees a fresh graph. Measurement therefore includes
        # forward; subtract the "forward" row to derive backward-only timing.
        def step():
            if logits.grad is not None:
                logits.grad = None
            out = fwd_fn(logits, target)
            out.sum().backward()

    elif mode == "full":

        def step():
            if logits.grad is not None:
                logits.grad = None
            y = fwd_fn(logits, target)
            y.sum().backward()

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # Warmup
    for _ in range(_WARMUP_ITERS):
        step()
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    # 5 short timed loops, take quantiles for the CSV schema
    samples = []
    for _ in range(5):
        per_iter_ms = _timed_loop(step, tp_group, _MEASURE_ITERS)
        samples.append(per_iter_ms)
    samples_t = torch.tensor(samples)
    ms_50 = float(samples_t.quantile(0.5))
    ms_20 = float(samples_t.quantile(0.2))
    ms_80 = float(samples_t.quantile(0.8))
    return ms_50, ms_20, ms_80


def _worker(rank, tp_size, providers, vocab_sizes, modes, s, b, file_name, result_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=tp_size,
    )
    torch.cuda.set_device(rank)
    tp_group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    # Per-(provider, mode) lists of timing samples, keyed for grouping into
    # BenchmarkData objects at the end. Only rank 0 collects.
    grouped: dict = {} if rank == 0 else None
    if rank == 0:
        gpu_name = get_gpu_name()
        timestamp = get_formatted_time()

    for v_global in vocab_sizes:
        for provider in providers:
            for mode in modes:
                ms_50, ms_20, ms_80 = _bench_one(provider, mode, s, b, v_global, tp_size, tp_group, device)
                if rank == 0:
                    grouped.setdefault((provider, mode), []).append((v_global, ms_50, ms_20, ms_80))
                    print(
                        f"  [{provider:>17s}] V={v_global:>6d} {mode:>9s}: "
                        f"{ms_50:.4f} ms (20={ms_20:.4f}, 80={ms_80:.4f})"
                    )
                dist.barrier(group=tp_group)

    if rank == 0:
        # Group by (provider, mode) — one BenchmarkData per group with x_values as a list.
        bd_list = []
        for (provider, mode), samples in grouped.items():
            samples.sort()
            bd_list.append(
                BenchmarkData(
                    kernel_name="megatron_cross_entropy_tp2",
                    kernel_provider=provider,
                    metric_name="speed",
                    metric_unit="ms",
                    gpu_name=gpu_name,
                    x_name="V",
                    x_label="vocab size",
                    x_values=[r[0] for r in samples],
                    y_values_50=[r[1] for r in samples],
                    y_values_20=[r[2] for r in samples],
                    y_values_80=[r[3] for r in samples],
                    timestamp=timestamp,
                    kernel_operation_mode=mode,
                    extra_benchmark_config_str=f'{{"S": {s}, "B": {b}, "TP": {tp_size}}}',
                )
            )
        update_benchmark_data_csv(benchmark_data_list=bd_list, filename=result_path, overwrite=False)

    dist.destroy_process_group()


def main():
    if not _MEGATRON_AVAILABLE:
        print("megatron-core not installed — running with liger only.")
        providers = ["liger"]
    else:
        # "torch" (F.cross_entropy) is intentionally excluded at TP>1: it isn't
        # vocab-parallel and would assert on target values >= V_local.
        providers = ["liger", "megatron", "megatron-unfused"]

    vocab_sizes = [2**i for i in range(12, 18)]  # 4K to 128K
    modes = ["forward", "backward", "full"]
    s, b = 2048, 4
    tp_size = 2

    result_path = "all_benchmark_data.csv"

    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _worker,
            args=(tp_size, providers, vocab_sizes, modes, s, b, f.name, result_path),
            nprocs=tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
