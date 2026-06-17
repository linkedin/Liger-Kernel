"""Benchmark Liger's Megatron fused LM-head + vocab-parallel CE at any TP size.

Compares four providers, each takes ``(hidden_states, weight, target, bias)`` and
returns a [S, B] per-token loss — apples-to-apples for the matmul + loss cost:

  - **liger-vp-flce**: fused LM-head + VP-CE, chunked, no [BT, V_local] materialization.
  - **liger-vp-ce-plus-cpl**: existing Liger VP-CE composed with a manual
    ``F.linear(x, W_local)`` (the "current state of the art" baseline).
  - **megatron-fused-plus-cpl**: Megatron's fused jit_fuser path composed with
    ``F.linear(x, W_local)``.
  - **megatron-unfused-plus-cpl**: Megatron's unfused eager-Python path with
    the same linear.

Rows go to ``benchmark/data/all_benchmark_data.csv`` tagged with
``kernel_name="megatron_fused_linear_cross_entropy"``; ``TP`` lives in
``extra_benchmark_config_str``.

Same fixed-iter timing loop as ``benchmark_megatron_cross_entropy.py`` (no
``triton.testing.do_bench`` — its adaptive calibration desyncs NCCL).
"""

from __future__ import annotations

import argparse
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
from liger_kernel.megatron import LigerMegatronFusedLinearCrossEntropy

try:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    _MEGATRON_AVAILABLE = True
except ImportError:
    fused_vocab_parallel_cross_entropy = None
    vocab_parallel_cross_entropy = None
    _MEGATRON_AVAILABLE = False


_WARMUP_ITERS = 5
_MEASURE_ITERS = 20
_MEMORY_ITERS = 3


def _select_fwd(provider, tp_group, weight, bias):
    """Return a callable that consumes (hidden, target) and runs forward + returns loss [S, B]."""
    if provider == "liger-vp-flce":
        ce = LigerMegatronFusedLinearCrossEntropy()
        return lambda h, t: ce(h, weight, t, bias=bias, tp_group=tp_group)
    if provider == "liger-vp-ce-plus-cpl":
        ce = LigerMegatronCrossEntropy()

        def fwd(h, t):
            logits = F.linear(h, weight, bias)
            return ce(logits, t, tp_group=tp_group)

        return fwd
    if provider == "megatron-fused-plus-cpl":

        def fwd(h, t):
            logits = F.linear(h, weight, bias)
            return fused_vocab_parallel_cross_entropy(logits, t, tp_group)

        return fwd
    if provider == "megatron-unfused-plus-cpl":

        def fwd(h, t):
            logits = F.linear(h, weight, bias)
            return vocab_parallel_cross_entropy(logits, t, 0.0, tp_group)

        return fwd
    raise ValueError(f"unknown provider: {provider!r}")


def _timed_loop(fn, tp_group, n_iters):
    """Fixed-iter manual loop — same as VP-CE benchmark. Keeps NCCL ranks in lockstep."""
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


def _make_inputs(s, b, h_dim, v_local, v_global, device):
    hidden = torch.randn(s, b, h_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(v_local, h_dim, device=device, dtype=torch.bfloat16) * 0.5
    weight.requires_grad_(True)
    target = torch.randint(0, v_global, (s, b), device=device, dtype=torch.long)
    return hidden, weight, target


def _bench_speed_one(provider, mode, s, b, h_dim, v_global, tp_size, tp_group, device):
    """Time one (provider, mode, V) cell. Returns (ms_50, ms_20, ms_80)."""
    v_local = v_global // tp_size
    hidden, weight, target = _make_inputs(s, b, h_dim, v_local, v_global, device)
    fwd_fn = _select_fwd(provider, tp_group, weight, bias=None)

    if mode == "forward":

        def step():
            fwd_fn(hidden, target)

    elif mode == "backward":

        def step():
            if hidden.grad is not None:
                hidden.grad = None
            if weight.grad is not None:
                weight.grad = None
            out = fwd_fn(hidden, target)
            out.sum().backward()

    elif mode == "full":

        def step():
            if hidden.grad is not None:
                hidden.grad = None
            if weight.grad is not None:
                weight.grad = None
            y = fwd_fn(hidden, target)
            y.sum().backward()

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    for _ in range(_WARMUP_ITERS):
        step()
    torch.cuda.synchronize()
    if tp_group is not None:
        dist.barrier(group=tp_group)

    samples = []
    for _ in range(3):
        samples.append(_timed_loop(step, tp_group, _MEASURE_ITERS))
    samples_t = torch.tensor(samples)
    return (
        float(samples_t.quantile(0.5)),
        float(samples_t.quantile(0.2)),
        float(samples_t.quantile(0.8)),
    )


def _bench_memory_one(provider, s, b, h_dim, v_global, tp_size, tp_group, device):
    """Peak HBM for one full forward+backward, MB."""
    v_local = v_global // tp_size
    hidden, weight, target = _make_inputs(s, b, h_dim, v_local, v_global, device)
    fwd_fn = _select_fwd(provider, tp_group, weight, bias=None)

    def step():
        if hidden.grad is not None:
            hidden.grad = None
        if weight.grad is not None:
            weight.grad = None
        y = fwd_fn(hidden, target)
        y.sum().backward()

    # Warmup so kernel-compile allocations don't pollute the peak.
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


def _worker(rank, tp_size, providers, vocab_sizes, s, b, h_dim, file_name, result_path, overwrite):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{file_name}",
        rank=rank,
        world_size=tp_size,
    )
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
                ms_50, ms_20, ms_80 = _bench_speed_one(provider, mode, s, b, h_dim, v_global, tp_size, tp_group, device)
                if rank == 0:
                    grouped_speed.setdefault((provider, mode), []).append((v_global, ms_50, ms_20, ms_80))
                    print(
                        f"  [speed]  [{provider:>26s}] V={v_global:>6d} {mode:>9s}: "
                        f"{ms_50:.4f} ms (20={ms_20:.4f}, 80={ms_80:.4f})"
                    )
                if tp_group is not None:
                    dist.barrier(group=tp_group)

            mb_50, mb_20, mb_80 = _bench_memory_one(provider, s, b, h_dim, v_global, tp_size, tp_group, device)
            if rank == 0:
                grouped_memory.setdefault((provider, "full"), []).append((v_global, mb_50, mb_20, mb_80))
                print(f"  [memory] [{provider:>26s}] V={v_global:>6d}      full: {mb_50:.1f} MB")
            if tp_group is not None:
                dist.barrier(group=tp_group)

    if rank == 0:
        bd_list = []
        kernel_name = "megatron_fused_linear_cross_entropy"
        for (provider, mode), samples in grouped_speed.items():
            samples.sort()
            bd_list.append(
                BenchmarkData(
                    kernel_name=kernel_name,
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
                    extra_benchmark_config_str=f'{{"S": {s}, "B": {b}, "H": {h_dim}, "TP": {tp_size}}}',
                )
            )
        for (provider, mode), samples in grouped_memory.items():
            samples.sort()
            bd_list.append(
                BenchmarkData(
                    kernel_name=kernel_name,
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
                    extra_benchmark_config_str=f'{{"S": {s}, "B": {b}, "H": {h_dim}, "TP": {tp_size}}}',
                )
            )
        update_benchmark_data_csv(benchmark_data_list=bd_list, filename=result_path, overwrite=overwrite)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1, help="tensor-parallel world size")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tp_size = args.tp_size
    have_gpus = torch.cuda.device_count()
    if tp_size > have_gpus:
        raise RuntimeError(f"--tp-size={tp_size} requires {tp_size} GPUs; have {have_gpus}.")

    if _MEGATRON_AVAILABLE:
        providers = [
            "liger-vp-flce",
            "liger-vp-ce-plus-cpl",
            "megatron-fused-plus-cpl",
            "megatron-unfused-plus-cpl",
        ]
    else:
        providers = ["liger-vp-flce", "liger-vp-ce-plus-cpl"]

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
                args.hidden_dim,
                f.name,
                result_path,
                args.overwrite,
            ),
            nprocs=tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
