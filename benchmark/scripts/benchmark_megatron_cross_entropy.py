"""Benchmark Liger's Megatron CE at any tensor_model_parallel_size.

Same implementation runs for TP=1 and TP>1 — the vocab-parallel kernel handles both
regimes (at TP=1 the AllReduces are skipped at the wrapper). One script, one CLI flag
(``--tp-size``), one CSV row format. Spawns ``tp_size`` ranks via ``mp.spawn`` and
uses a fixed-iteration manual timing loop so every rank executes the same number of
NCCL collectives (``triton.testing.do_bench`` calibrates adaptively per-rank and
desyncs the collectives — see the long comment in ``_timed_loop``).

Compared providers (each rank executes its own ``V/TP`` slice):

  - **liger**: ``LigerMegatronCrossEntropy`` — Liger's vocab-parallel kernel.
  - **torch**: vanilla ``F.cross_entropy``. Included only at ``--tp-size 1`` — it
    isn't vocab-parallel and would assert on target values >= V_local at TP>1.
  - **megatron**: Megatron's *fused* ``fused_vocab_parallel_cross_entropy``
    (``cross_entropy_loss_fusion=True``, JIT-fused via TorchScript).
  - **megatron-unfused**: Megatron's *unfused* ``vocab_parallel_cross_entropy``
    (``cross_entropy_loss_fusion=False``, eager Python — the path with runtime
    ``label_smoothing`` support).

Rows go to ``benchmark/data/all_benchmark_data.csv`` tagged with
``kernel_name="megatron_cross_entropy"``; ``TP`` lives in
``extra_benchmark_config_str``. Render plots per-TP via ``--extra-config-filter``:

    python benchmark/scripts/benchmark_megatron_cross_entropy.py --tp-size 1
    python benchmark/scripts/benchmark_megatron_cross_entropy.py --tp-size 2
    python benchmark/scripts/benchmark_megatron_cross_entropy.py --tp-size 4

    python benchmark/benchmarks_visualizer.py \\
        --kernel-name megatron_cross_entropy --metric-name speed \\
        --extra-config-filter "'TP': 2"
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


_WARMUP_ITERS = 10
_MEASURE_ITERS = 50
_MEMORY_ITERS = 5  # reps used for peak-memory measurement


def _select_fwd(provider, tp_group):
    if provider == "liger":
        ce = LigerMegatronCrossEntropy()
        return lambda logits, target: ce(logits, target, tp_group=tp_group)
    if provider == "megatron":
        return lambda logits, target: fused_vocab_parallel_cross_entropy(logits, target, tp_group)
    if provider == "megatron-unfused":
        return lambda logits, target: vocab_parallel_cross_entropy(logits, target, 0.0, tp_group)
    raise ValueError(f"unknown provider: {provider!r}")


def _timed_loop(fn, tp_group, n_iters: int):
    """Run ``fn`` ``n_iters`` times and return per-iter ms.

    CRITICAL for TP>1: every rank MUST execute the same number of iterations because
    ``fn`` issues NCCL collectives. ``triton.testing.do_bench`` calibrates the
    iteration count from the first-call latency, which differs slightly between
    ranks and causes collective desync (NCCL watchdog timeout). This manual loop
    with a fixed ``n_iters`` keeps every rank in lockstep. At TP=1 the same loop
    is used unchanged (no AllReduces, so no desync risk).
    """
    torch.cuda.synchronize()
    if tp_group is not None:
        dist.barrier(group=tp_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def _make_inputs(s, b, v_local, v_global, device):
    logits = torch.randn(s, b, v_local, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, v_global, (s, b), device=device, dtype=torch.long)
    return logits, target


def _bench_speed_one(provider, mode, s, b, v_global, tp_size, tp_group, device):
    """Time one (provider, mode, V) cell. Returns (ms_50, ms_20, ms_80) over 5 timed loops."""
    v_local = v_global // tp_size
    logits, target = _make_inputs(s, b, v_local, v_global, device)
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

    for _ in range(_WARMUP_ITERS):
        step()
    torch.cuda.synchronize()
    if tp_group is not None:
        dist.barrier(group=tp_group)

    samples = []
    for _ in range(5):
        samples.append(_timed_loop(step, tp_group, _MEASURE_ITERS))
    samples_t = torch.tensor(samples)
    return (
        float(samples_t.quantile(0.5)),
        float(samples_t.quantile(0.2)),
        float(samples_t.quantile(0.8)),
    )


def _bench_memory_one(provider, s, b, v_global, tp_size, tp_group, device):
    """Peak memory for one full forward+backward on this rank, in MB."""
    v_local = v_global // tp_size
    logits, target = _make_inputs(s, b, v_local, v_global, device)
    fwd_fn = _select_fwd(provider, tp_group)

    def step():
        if logits.grad is not None:
            logits.grad = None
        y = fwd_fn(logits, target)
        y.sum().backward()

    # Warmup once so initial kernel-compile allocations don't pollute the peak.
    step()
    if tp_group is not None:
        dist.barrier(group=tp_group)

    samples = []
    for _ in range(_MEMORY_ITERS):
        torch.cuda.reset_peak_memory_stats()
        step()
        samples.append(torch.cuda.max_memory_allocated() / 2**20)
    samples_t = torch.tensor(samples)
    return (
        float(samples_t.quantile(0.5)),
        float(samples_t.quantile(0.2)),
        float(samples_t.quantile(0.8)),
    )


def _worker(rank, tp_size, providers, vocab_sizes, s, b, file_name, result_path, overwrite):
    # Megatron's CE queries get_tensor_model_parallel_group() unconditionally — even
    # at TP=1 — so dist must be initialized. NCCL world_size=1 is fine; the
    # all-reduces become no-ops at runtime.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=tp_size,
    )
    # Always pass a real ProcessGroup (even at world_size=1) — Megatron's CE calls
    # ``.rank()`` / ``.size()`` on it unconditionally. NCCL no-ops the collectives.
    tp_group = dist.group.WORLD

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    grouped_speed: dict = {} if rank == 0 else None
    grouped_memory: dict = {} if rank == 0 else None
    if rank == 0:
        gpu_name = get_gpu_name()
        timestamp = get_formatted_time()

    speed_modes = ("forward", "backward", "full")
    for v_global in vocab_sizes:
        for provider in providers:
            for mode in speed_modes:
                ms_50, ms_20, ms_80 = _bench_speed_one(provider, mode, s, b, v_global, tp_size, tp_group, device)
                if rank == 0:
                    grouped_speed.setdefault((provider, mode), []).append((v_global, ms_50, ms_20, ms_80))
                    print(
                        f"  [speed]  [{provider:>17s}] V={v_global:>6d} {mode:>9s}: "
                        f"{ms_50:.4f} ms (20={ms_20:.4f}, 80={ms_80:.4f})"
                    )
                if tp_group is not None:
                    dist.barrier(group=tp_group)

            mb_50, mb_20, mb_80 = _bench_memory_one(provider, s, b, v_global, tp_size, tp_group, device)
            if rank == 0:
                grouped_memory.setdefault((provider, "full"), []).append((v_global, mb_50, mb_20, mb_80))
                print(f"  [memory] [{provider:>17s}] V={v_global:>6d}      full: {mb_50:.1f} MB")
            if tp_group is not None:
                dist.barrier(group=tp_group)

    if rank == 0:
        bd_list = []
        for (provider, mode), samples in grouped_speed.items():
            samples.sort()
            bd_list.append(
                BenchmarkData(
                    kernel_name="megatron_cross_entropy",
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
        for (provider, mode), samples in grouped_memory.items():
            samples.sort()
            bd_list.append(
                BenchmarkData(
                    kernel_name="megatron_cross_entropy",
                    kernel_provider=provider,
                    metric_name="memory",
                    metric_unit="MB",
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
        update_benchmark_data_csv(benchmark_data_list=bd_list, filename=result_path, overwrite=overwrite)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1, help="tensor-parallel world size (1, 2, 4, 8, ...)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing CSV rows with the same key")
    args = parser.parse_args()

    tp_size = args.tp_size
    have_gpus = torch.cuda.device_count()
    if tp_size > have_gpus:
        raise RuntimeError(f"--tp-size={tp_size} requires {tp_size} GPUs; have {have_gpus}.")

    if _MEGATRON_AVAILABLE:
        providers = ["liger", "megatron", "megatron-unfused"]
    else:
        providers = ["liger"]

    vocab_sizes = [2**i for i in range(12, 18)]  # 4K to 128K
    result_path = "all_benchmark_data.csv"

    with tempfile.NamedTemporaryFile() as f:
        mp.spawn(
            _worker,
            args=(
                tp_size,
                providers,
                vocab_sizes,
                args.seq_len,
                args.batch_size,
                f.name,
                result_path,
                args.overwrite,
            ),
            nprocs=tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
