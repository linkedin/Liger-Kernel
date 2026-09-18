"""Distributed forward benchmark for native TP FSLCE and the Verl fallback.

Examples:

    torchrun --standalone --nproc_per_node=8 \
        benchmark/scripts/benchmark_fused_scaled_linear_cross_entropy_tp.py \
        --provider native

    torchrun --standalone --nproc_per_node=8 \
        benchmark/scripts/benchmark_fused_scaled_linear_cross_entropy_tp.py \
        --provider verl-fallback

For multi-host runs, pass the usual ``torchrun`` ``--nnodes``,
``--node_rank``, ``--master_addr``, and ``--master_port`` arguments.
"""

import argparse
import importlib.util
import json
import os
import statistics

from pathlib import Path

import torch
import torch.distributed as dist


def _load_fallback(path: Path):
    spec = importlib.util.spec_from_file_location("liger_fslce_verl_fallback", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Verl fallback from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reduce_max_float(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def _reduce_max_int(value: int, device: torch.device) -> int:
    result = torch.tensor(value, dtype=torch.int64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return int(result.item())


def _diagnostic_durations(values: list[int]) -> dict[str, float]:
    start = values[0]
    final_publish = values[1]
    local_complete = values[2]
    ring_complete = values[3]
    allgather_start = values[4]
    allgather_complete = values[5]
    output_complete = values[6]

    def elapsed(begin: int, end: int) -> float:
        return 0.0 if begin == 0 or end == 0 or end < begin else (end - begin) / 1.0e6

    ring_intervals = []
    for wave in range(16):
        ring_start = values[16 + wave * 2]
        ring_end = values[17 + wave * 2]
        if ring_start != 0 and ring_end >= ring_start:
            ring_intervals.append((ring_start, ring_end))

    comm_active_ticks = sum(end - begin for begin, end in ring_intervals)
    comm_overlap_ticks = sum(max(0, min(end, final_publish) - max(begin, start)) for begin, end in ring_intervals)
    comm_post_gemm_ticks = sum(max(0, end - max(begin, final_publish)) for begin, end in ring_intervals)
    comm_pipeline_begin = ring_intervals[0][0] if ring_intervals else 0
    comm_pipeline_end = ring_intervals[-1][1] if ring_intervals else 0

    metrics = {
        "kernel_span_ms": elapsed(start, output_complete),
        "gemm_epilogue_ms": elapsed(start, final_publish),
        "final_publish_offset_ms": elapsed(start, final_publish),
        "final_tail_ms": elapsed(final_publish, output_complete),
        "output_complete_offset_ms": elapsed(start, output_complete),
        "local_reduce_tail_ms": elapsed(final_publish, local_complete),
        "local_reduce_complete_offset_ms": elapsed(start, local_complete),
        "ring_tail_ms": values[10] / 1.0e6,
        "ring_complete_offset_ms": elapsed(start, ring_complete),
        "comm_active_ms": comm_active_ticks / 1.0e6,
        "comm_pipeline_start_offset_ms": elapsed(start, comm_pipeline_begin),
        "comm_pipeline_end_offset_ms": elapsed(start, comm_pipeline_end),
        "comm_pipeline_span_ms": elapsed(comm_pipeline_begin, comm_pipeline_end),
        "comm_overlap_gemm_ms": comm_overlap_ticks / 1.0e6,
        "comm_post_gemm_active_ms": comm_post_gemm_ticks / 1.0e6,
        "final_drain_after_pipelines_ms": elapsed(
            max(final_publish, comm_pipeline_end),
            output_complete,
        ),
        "allgather_finalize_ms": elapsed(allgather_start, output_complete),
        "allgather_start_offset_ms": elapsed(start, allgather_start),
        "producer_slot_wait_ms": values[7] / 1.0e6,
        "warp0_split_wait_ms": values[8] / 1.0e6,
        "warp1_source_wait_ms": values[9] / 1.0e6,
        "allgather_ms": elapsed(allgather_start, allgather_complete),
        "ring_to_allgather_ms": elapsed(ring_complete, allgather_start),
    }
    for wave in range(16):
        ring_start = values[16 + wave * 2]
        ring_end = values[17 + wave * 2]
        if ring_start == 0 or ring_end < ring_start:
            continue
        metrics[f"ring_wave{wave}_start_ms"] = elapsed(start, ring_start)
        metrics[f"ring_wave{wave}_ms"] = elapsed(ring_start, ring_end)
        metrics[f"ring_wave{wave}_overlap_gemm_ms"] = (
            max(
                0,
                min(ring_end, final_publish) - max(ring_start, start),
            )
            / 1.0e6
        )
        metrics[f"ring_wave{wave}_post_gemm_ms"] = max(0, ring_end - max(ring_start, final_publish)) / 1.0e6
    return metrics


def _collect_diagnostics(
    metrics: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], int]:
    names = tuple(metrics)
    values = torch.tensor(
        [metrics[name] for name in names],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    stacked = torch.stack(gathered)
    maximum = stacked.amax(dim=0).cpu().tolist()
    span_index = names.index("kernel_span_ms")
    critical_rank = int(stacked[:, span_index].argmax().item())
    critical = stacked[critical_rank].cpu().tolist()
    return (
        dict(zip(names, maximum)),
        dict(zip(names, critical)),
        critical_rank,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("native", "verl-fallback"), required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--global-vocab", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--no-entropy", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--fallback-source",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "src"
        / "liger_kernel"
        / "ops"
        / "fused_linear_scaled_cross_entropy.py",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.tokens <= 0 or args.hidden <= 0 or args.global_vocab <= 0:
        raise ValueError("tokens, hidden, and global vocabulary must be positive")
    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("warmups must be non-negative and iterations must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.global_vocab % world_size != 0:
        raise ValueError(f"global vocabulary {args.global_vocab} is not divisible by TP size {world_size}")

    native_initialized = False
    try:
        local_vocab = args.global_vocab // world_size
        return_entropy = not args.no_entropy
        inverse_temperature = 1.0 / args.temperature
        fallback_chunk_size = None
        team = None
        if args.provider == "native":
            from liger_cute_kernels import nvshmem
            from liger_cute_kernels import tvm_ffi

            nvshmem.init_from_pg()
            native_initialized = True
            team = nvshmem.team_world()

        x_generator = torch.Generator(device=device).manual_seed(2027)
        x = torch.randn(
            args.tokens,
            args.hidden,
            generator=x_generator,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.05)
        weight_generator = torch.Generator(device=device).manual_seed(3100 + rank)
        weight = torch.randn(
            local_vocab,
            args.hidden,
            generator=weight_generator,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.05)
        target = torch.arange(args.tokens, dtype=torch.int64, device=device)
        target.remainder_(args.global_vocab)

        if args.provider == "native":
            tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(
                args.tokens,
                local_vocab,
            )
            tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(
                args.tokens,
                args.hidden,
                local_vocab,
                1,
                team,
            )

            def forward():
                return tvm_ffi.fused_linear_scaled_cross_entropy_forward(
                    x,
                    weight,
                    target,
                    rank * local_vocab,
                    -100,
                    inverse_temperature,
                    team,
                    return_entropy,
                )

        else:
            fallback = _load_fallback(args.fallback_source)
            fallback_chunk_size = int(fallback._FALLBACK_CHUNK_SIZE)

            def forward():
                return fallback._apply_tp_fallback(
                    x,
                    weight,
                    target,
                    args.temperature,
                    -100,
                    return_entropy,
                    dist.group.WORLD,
                )

        for _ in range(args.warmups):
            dist.barrier()
            output = forward()
            torch.cuda.synchronize()
            del output

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        baseline_free, total_memory = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)

        dist.barrier()
        memory_output = forward()
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        observed_free, _ = torch.cuda.mem_get_info(device)
        finite = all(
            bool(torch.isfinite(tensor).all())
            for tensor in (memory_output if isinstance(memory_output, tuple) else (memory_output,))
        )
        del memory_output

        latencies = []
        diagnostic_max_samples: list[dict[str, float]] = []
        diagnostic_critical_samples: list[dict[str, float]] = []
        diagnostic_critical_ranks: list[int] = []
        for _ in range(args.iterations):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward()
            end.record()
            diagnostic_output = (
                tvm_ffi.fused_linear_scaled_cross_entropy_forward_diagnostics(device)
                if args.provider == "native" and args.diagnostics
                else None
            )
            torch.cuda.synchronize()
            latencies.append(_reduce_max_float(start.elapsed_time(end), device))
            if diagnostic_output is not None:
                maximum, critical, critical_rank = _collect_diagnostics(
                    _diagnostic_durations(diagnostic_output.cpu().tolist()),
                    device,
                )
                diagnostic_max_samples.append(maximum)
                diagnostic_critical_samples.append(critical)
                diagnostic_critical_ranks.append(critical_rank)
            finite = finite and all(
                bool(torch.isfinite(tensor).all()) for tensor in (output if isinstance(output, tuple) else (output,))
            )
            del output

        finite_value = torch.tensor(int(finite), dtype=torch.int32, device=device)
        dist.all_reduce(finite_value, op=dist.ReduceOp.MIN)

        median_ms = statistics.median(latencies)
        global_flops = 2.0 * args.tokens * args.global_vocab * args.hidden
        result = {
            "provider": args.provider,
            "world_size": world_size,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "global_vocab": args.global_vocab,
            "local_vocab": local_vocab,
            "return_entropy": return_entropy,
            "fallback_chunk_size": fallback_chunk_size,
            "fallback_chunks": (
                None if fallback_chunk_size is None else (args.tokens + fallback_chunk_size - 1) // fallback_chunk_size
            ),
            "all_outputs_finite": bool(finite_value.item()),
            "latency_mean_ms": statistics.mean(latencies),
            "latency_median_ms": median_ms,
            "latency_std_ms": statistics.pstdev(latencies),
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "effective_global_tflops": global_flops / (median_ms * 1.0e9),
            "max_rank_baseline_allocated_bytes": _reduce_max_int(baseline_allocated, device),
            "max_rank_baseline_reserved_bytes": _reduce_max_int(baseline_reserved, device),
            "max_rank_peak_allocated_bytes": _reduce_max_int(peak_allocated, device),
            "max_rank_peak_reserved_bytes": _reduce_max_int(peak_reserved, device),
            "max_rank_incremental_peak_allocated_bytes": _reduce_max_int(peak_allocated - baseline_allocated, device),
            "max_rank_incremental_peak_reserved_bytes": _reduce_max_int(peak_reserved - baseline_reserved, device),
            "max_rank_device_used_at_baseline_bytes": _reduce_max_int(total_memory - baseline_free, device),
            "max_rank_device_used_after_forward_bytes": _reduce_max_int(total_memory - observed_free, device),
        }
        if diagnostic_max_samples:
            for name in diagnostic_max_samples[0]:
                result[f"diagnostic_max_{name}"] = statistics.median(sample[name] for sample in diagnostic_max_samples)
                result[f"diagnostic_critical_{name}"] = statistics.median(
                    sample[name] for sample in diagnostic_critical_samples
                )
            gemm_ms = result["diagnostic_critical_gemm_epilogue_ms"]
            tail_ms = result["diagnostic_critical_final_tail_ms"]
            result["diagnostic_critical_tail_to_gemm_ratio"] = tail_ms / gemm_ms if gemm_ms > 0.0 else 0.0
            result["diagnostic_critical_rank_median"] = statistics.median(diagnostic_critical_ranks)
            representative = min(
                range(len(latencies)),
                key=lambda index: abs(latencies[index] - median_ms),
            )
            result["diagnostic_representative_event_latency_ms"] = latencies[representative]
            result["diagnostic_representative_rank"] = diagnostic_critical_ranks[representative]
            for name, value in diagnostic_critical_samples[representative].items():
                result[f"diagnostic_representative_{name}"] = value
        if rank == 0:
            print("FSLCE_TP_FORWARD_BENCHMARK " + json.dumps(result, sort_keys=True))
        if not result["all_outputs_finite"]:
            raise RuntimeError("non-finite FSLCE benchmark output")
        dist.barrier()
    finally:
        if native_initialized:
            from liger_cute_kernels import nvshmem

            nvshmem.finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
