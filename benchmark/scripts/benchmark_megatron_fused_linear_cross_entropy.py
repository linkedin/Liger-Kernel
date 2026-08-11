"""Benchmark hidden-to-loss FLCE against Megatron's materialized output-loss stack.

Megatron-Core does not provide a fused linear cross-entropy kernel. Its comparable
training path is:

    vocab-parallel linear -> materialized local logits -> fused vocab-parallel CE

This script compares that path with ``LigerMegatronFusedLinearCrossEntropy``,
which saves low-precision CE state to avoid projection recomputation. The
``liger-triton`` provider additionally replaces all three local GEMMs with
portable Triton kernels. When Megatron-Core is installed, the ``megatron-core``
provider uses its fused CE. The always-available ``megatron-compatible``
provider uses Liger's drop-in Megatron CE.

Backward timing creates a fresh graph outside each timed event pair, so only
backward execution is measured while respecting Megatron's single-use fused CE
graph. Fixed iteration counts keep all tensor-parallel ranks in collective
lockstep.

Examples:

    python benchmark_megatron_fused_linear_cross_entropy.py --tp-size 1
    torchrun --help  # not needed; the script spawns TP ranks itself
    python benchmark_megatron_fused_linear_cross_entropy.py --tp-size 4 \
        --token-counts 512 2048 --vocab-sizes 32000 128256
"""

from __future__ import annotations

import argparse
import os
import tempfile

from dataclasses import dataclass
from pathlib import Path

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
from liger_kernel.ops.triton.ops.megatron_fused_linear_cross_entropy import (
    liger_megatron_fused_linear_cross_entropy as triton_megatron_fused_linear_cross_entropy,
)

try:
    from liger_kernel.ops.cutile.ops.megatron_fused_linear_cross_entropy import (
        liger_megatron_fused_linear_cross_entropy as cutile_megatron_fused_linear_cross_entropy,
    )

    _CUTILE_AVAILABLE = True
except ImportError:
    cutile_megatron_fused_linear_cross_entropy = None
    _CUTILE_AVAILABLE = False

try:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    _MEGATRON_CORE_AVAILABLE = True
except ImportError:
    fused_vocab_parallel_cross_entropy = None
    _MEGATRON_CORE_AVAILABLE = False


_SPEED_SAMPLES = 5
_MEMORY_SAMPLES = 3
_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class _ProviderState:
    hidden: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    target: torch.Tensor
    forward: object

    def clear_grads(self):
        self.hidden.grad = None
        self.weight.grad = None
        if self.bias is not None:
            self.bias.grad = None


def _all_reduce_hidden_grad(grad: torch.Tensor, tp_group):
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)
    return grad


def _make_state(
    provider: str,
    hidden_master: torch.Tensor,
    weight_master: torch.Tensor,
    bias_master: torch.Tensor | None,
    target: torch.Tensor,
    tp_group,
    tp_size: int,
) -> _ProviderState:
    hidden = hidden_master.clone().requires_grad_(True)
    weight = weight_master.clone().requires_grad_(True)
    bias = bias_master.clone().requires_grad_(True) if bias_master is not None else None

    if provider == "liger":
        loss = LigerMegatronFusedLinearCrossEntropy()
        forward = lambda: loss(hidden, weight, target, bias=bias, tp_group=tp_group)
    elif provider == "liger-triton":
        forward = lambda: triton_megatron_fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            tp_group=tp_group,
        )
    elif provider == "liger-cutile":
        if not _CUTILE_AVAILABLE:
            raise RuntimeError("provider 'liger-cutile' requires the cuda-tile package.")
        forward = lambda: cutile_megatron_fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            tp_group=tp_group,
        )
    else:
        if tp_size > 1:
            hidden.register_hook(lambda grad: _all_reduce_hidden_grad(grad, tp_group))

        if provider == "megatron-core":
            if not _MEGATRON_CORE_AVAILABLE:
                raise RuntimeError("provider 'megatron-core' requires the megatron-core package.")
            ce_forward = lambda logits: fused_vocab_parallel_cross_entropy(logits, target, tp_group)
        elif provider == "megatron-compatible":
            ce = LigerMegatronCrossEntropy()
            ce_forward = lambda logits: ce(logits, target, tp_group=tp_group)
        else:
            raise ValueError(f"unknown provider: {provider!r}")

        forward = lambda: ce_forward(F.linear(hidden, weight, bias))

    return _ProviderState(hidden, weight, bias, target, forward)


def _synchronized_elapsed_ms(step, tp_group, iterations: int) -> float:
    dist.barrier(group=tp_group)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    torch.cuda.synchronize()
    elapsed = torch.tensor(start.elapsed_time(end) / iterations, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=tp_group)
    return float(elapsed)


def _synchronized_backward_ms(state: _ProviderState, tp_group, iterations: int) -> float:
    dist.barrier(group=tp_group)
    torch.cuda.synchronize()
    event_pairs = []
    for _ in range(iterations):
        state.clear_grads()
        loss = state.forward()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward(torch.ones_like(loss))
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize()
    elapsed = torch.tensor(
        sum(start.elapsed_time(end) for start, end in event_pairs) / iterations,
        device="cuda",
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=tp_group)
    return float(elapsed)


def _quantiles(samples: list[float]) -> tuple[float, float, float]:
    values = torch.tensor(samples)
    return (
        float(values.quantile(0.5)),
        float(values.quantile(0.2)),
        float(values.quantile(0.8)),
    )


def _speed(
    state: _ProviderState,
    tp_group,
    warmup_iterations: int,
    measure_iterations: int,
) -> dict[str, tuple[float, float, float]]:
    def forward_step():
        state.forward()

    def full_step():
        state.clear_grads()
        loss = state.forward()
        loss.backward(torch.ones_like(loss))

    for _ in range(warmup_iterations):
        full_step()
    torch.cuda.synchronize()

    forward_samples = [
        _synchronized_elapsed_ms(forward_step, tp_group, measure_iterations) for _ in range(_SPEED_SAMPLES)
    ]
    backward_samples = [_synchronized_backward_ms(state, tp_group, measure_iterations) for _ in range(_SPEED_SAMPLES)]
    full_samples = [_synchronized_elapsed_ms(full_step, tp_group, measure_iterations) for _ in range(_SPEED_SAMPLES)]
    return {
        "forward": _quantiles(forward_samples),
        "backward": _quantiles(backward_samples),
        "full": _quantiles(full_samples),
    }


def _memory(state: _ProviderState, tp_group) -> tuple[float, float, float]:
    def full_step():
        state.clear_grads()
        loss = state.forward()
        loss.backward(torch.ones_like(loss))

    full_step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(_MEMORY_SAMPLES):
        state.clear_grads()
        torch.cuda.reset_peak_memory_stats()
        full_step()
        torch.cuda.synchronize()
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**20, device="cuda")
        dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=tp_group)
        samples.append(float(peak))
    return _quantiles(samples)


def _make_masters(
    rank: int,
    tp_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_global: int,
    dtype: torch.dtype,
    with_bias: bool,
    device: torch.device,
):
    if vocab_global % tp_size:
        raise ValueError(f"vocab size {vocab_global} must be divisible by TP={tp_size}.")
    vocab_local = vocab_global // tp_size
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    hidden = torch.randn(
        num_tokens,
        1,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    target = torch.randint(
        vocab_global,
        (num_tokens, 1),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    dist.broadcast(hidden, src=0)
    dist.broadcast(target, src=0)

    generator.manual_seed(1000 + rank)
    weight = torch.randn(
        vocab_local,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    bias = torch.randn(vocab_local, device=device, dtype=dtype, generator=generator) if with_bias else None
    return hidden, weight, bias, target


def _check_correctness(
    rank: int,
    tp_size: int,
    tp_group,
    dtype: torch.dtype,
    device: torch.device,
    providers,
):
    hidden, weight, bias, target = _make_masters(
        rank,
        tp_size,
        num_tokens=32,
        hidden_size=256,
        vocab_global=1024,
        dtype=dtype,
        with_bias=True,
        device=device,
    )
    weight.mul_(0.02)
    bias.mul_(0.02)
    upstream = torch.randn_like(target, dtype=torch.float32)
    dist.broadcast(upstream, src=0)
    outputs = {}
    correctness_providers = ["megatron-compatible"]
    correctness_providers.extend(
        provider for provider in ("liger", "liger-triton", "liger-cutile") if provider in providers
    )
    for provider in correctness_providers:
        state = _make_state(provider, hidden, weight, bias, target, tp_group, tp_size)
        loss = state.forward()
        loss.backward(upstream)
        outputs[provider] = (
            loss.detach().float(),
            state.hidden.grad.detach().float(),
            state.weight.grad.detach().float(),
            state.bias.grad.detach().float(),
        )

    reference = outputs["megatron-compatible"]
    names = ("loss", "grad_hidden", "grad_weight", "grad_bias")
    for provider in correctness_providers[1:]:
        actual = outputs[provider]
        for name, actual_tensor, reference_tensor in zip(names, actual, reference):
            torch.testing.assert_close(
                actual_tensor,
                reference_tensor,
                atol=5e-3,
                rtol=5e-2,
                msg=f"{provider}: {name}",
            )


def _worker(
    rank,
    tp_size,
    providers,
    token_counts,
    vocab_sizes,
    hidden_size,
    dtype_name,
    with_bias,
    warmup_iterations,
    measure_iterations,
    rendezvous,
    result_path,
    overwrite,
):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=tp_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    tp_group = dist.group.WORLD
    dtype = _DTYPES[dtype_name]

    _check_correctness(rank, tp_size, tp_group, dtype, device, providers)
    if rank == 0:
        print("Correctness: loss and gradients match the materialized reference.", flush=True)

    grouped_speed = {} if rank == 0 else None
    grouped_memory = {} if rank == 0 else None
    for num_tokens in token_counts:
        for vocab_global in vocab_sizes:
            masters = _make_masters(
                rank,
                tp_size,
                num_tokens,
                hidden_size,
                vocab_global,
                dtype,
                with_bias,
                device,
            )
            for provider in providers:
                state = _make_state(provider, *masters, tp_group, tp_size)
                speed = _speed(state, tp_group, warmup_iterations, measure_iterations)
                memory = _memory(state, tp_group)
                if rank == 0:
                    for mode, (p50, p20, p80) in speed.items():
                        grouped_speed.setdefault((provider, mode, num_tokens), []).append((vocab_global, p50, p20, p80))
                        print(
                            f"[speed]  {provider:>21s} TP={tp_size} BT={num_tokens:>5d} "
                            f"V={vocab_global:>6d} {mode:>8s}: {p50:.4f} ms",
                            flush=True,
                        )
                    grouped_memory.setdefault((provider, "full", num_tokens), []).append((vocab_global, *memory))
                    print(
                        f"[memory] {provider:>21s} TP={tp_size} BT={num_tokens:>5d} "
                        f"V={vocab_global:>6d}: {memory[0]:.1f} MB",
                        flush=True,
                    )
                del state
                torch.cuda.empty_cache()
                dist.barrier(group=tp_group)
            del masters

    if rank == 0:
        timestamp = get_formatted_time()
        gpu_name = get_gpu_name()
        rows = []
        common = {
            "kernel_name": "megatron_fused_linear_cross_entropy",
            "gpu_name": gpu_name,
            "x_name": "V",
            "x_label": "global vocab size",
            "timestamp": timestamp,
        }
        for (provider, mode, num_tokens), samples in grouped_speed.items():
            samples.sort()
            rows.append(
                BenchmarkData(
                    kernel_provider=provider,
                    metric_name="speed",
                    metric_unit="ms",
                    x_values=[row[0] for row in samples],
                    y_values_50=[row[1] for row in samples],
                    y_values_20=[row[2] for row in samples],
                    y_values_80=[row[3] for row in samples],
                    kernel_operation_mode=mode,
                    extra_benchmark_config_str=(
                        f'{{"BT": {num_tokens}, "H": {hidden_size}, "TP": {tp_size}, '
                        f'"dtype": "{dtype_name}", "bias": {str(with_bias).lower()}}}'
                    ),
                    **common,
                )
            )
        for (provider, mode, num_tokens), samples in grouped_memory.items():
            samples.sort()
            rows.append(
                BenchmarkData(
                    kernel_provider=provider,
                    metric_name="memory",
                    metric_unit="MB",
                    x_values=[row[0] for row in samples],
                    y_values_50=[row[1] for row in samples],
                    y_values_20=[row[2] for row in samples],
                    y_values_80=[row[3] for row in samples],
                    kernel_operation_mode=mode,
                    extra_benchmark_config_str=(
                        f'{{"BT": {num_tokens}, "H": {hidden_size}, "TP": {tp_size}, '
                        f'"dtype": "{dtype_name}", "bias": {str(with_bias).lower()}}}'
                    ),
                    **common,
                )
            )
        update_benchmark_data_csv(rows, filename=result_path, overwrite=overwrite)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000, 128256])
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--with-bias", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measure-iterations", type=int, default=10)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["liger", "liger-triton", "liger-cutile", "megatron-compatible", "megatron-core"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "all_benchmark_data_megatron_flce.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.tp_size > torch.cuda.device_count():
        raise RuntimeError(f"--tp-size={args.tp_size} requires {args.tp_size} GPUs; found {torch.cuda.device_count()}.")
    providers = args.providers or ["megatron-compatible", "liger"]
    if _MEGATRON_CORE_AVAILABLE and args.providers is None:
        providers.insert(1, "megatron-core")
    if "megatron-core" in providers and not _MEGATRON_CORE_AVAILABLE:
        raise RuntimeError("provider 'megatron-core' requested, but megatron-core is not installed.")
    if "liger-cutile" in providers and not _CUTILE_AVAILABLE:
        raise RuntimeError("provider 'liger-cutile' requested, but cuda-tile is not installed.")
    if min(args.token_counts) <= 0 or args.hidden_size <= 0 or min(args.vocab_sizes) <= 0:
        raise ValueError("token counts, hidden size, and vocabulary sizes must be positive.")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile() as rendezvous:
        mp.spawn(
            _worker,
            args=(
                args.tp_size,
                providers,
                args.token_counts,
                args.vocab_sizes,
                args.hidden_size,
                args.dtype,
                args.with_bias,
                args.warmup_iterations,
                args.measure_iterations,
                rendezvous.name,
                str(output),
                args.overwrite,
            ),
            nprocs=args.tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
