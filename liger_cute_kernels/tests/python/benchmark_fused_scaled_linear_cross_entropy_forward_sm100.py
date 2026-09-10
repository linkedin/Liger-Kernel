"""Compare the native SM100 TP-FSLCE forward with the Verl-derived fallback."""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics

from pathlib import Path

import torch
import torch.distributed as dist

IGNORE_INDEX = -100
TEMPERATURE = 0.9


def _elapsed_ms(function, start: torch.cuda.Event, stop: torch.cuda.Event) -> float:
    start.record()
    function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop)


def _time_pair_ms(native, fallback, warmups: int, iterations: int, group):
    dist.barrier(group=group)
    for _ in range(warmups):
        native()
        fallback()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    native_samples = []
    fallback_samples = []
    for iteration in range(iterations):
        if iteration % 2 == 0:
            native_samples.append(_elapsed_ms(native, start, stop))
            fallback_samples.append(_elapsed_ms(fallback, start, stop))
        else:
            fallback_samples.append(_elapsed_ms(fallback, start, stop))
            native_samples.append(_elapsed_ms(native, start, stop))

    values = torch.tensor(
        [statistics.median(native_samples), statistics.median(fallback_samples)],
        dtype=torch.float64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    return float(values[0].item()), float(values[1].item())


def _peak_increment_bytes(function, group) -> int:
    dist.barrier(group=group)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    value = torch.tensor(peak, dtype=torch.int64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return int(value.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--local-vocab", type=int, default=65536)
    parser.add_argument(
        "--global-vocab",
        type=int,
        default=0,
        help="Use fixed-global-vocabulary scaling; overrides --local-vocab.",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD
    if world_size not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"benchmark supports TP1/2/4/8/16, got TP{world_size}")
    if args.global_vocab:
        if args.global_vocab % world_size != 0:
            raise ValueError(f"global vocabulary {args.global_vocab} is not divisible by TP{world_size}")
        args.local_vocab = args.global_vocab // world_size

    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi

    fallback_path = Path(__file__).resolve().parents[3] / "src/liger_kernel/ops/fused_linear_scaled_cross_entropy.py"
    fallback_spec = importlib.util.spec_from_file_location("liger_fslce_verl_fallback", fallback_path)
    if fallback_spec is None or fallback_spec.loader is None:
        raise ImportError(f"cannot load Verl-derived fallback from {fallback_path}")
    fallback_module = importlib.util.module_from_spec(fallback_spec)
    fallback_spec.loader.exec_module(fallback_module)
    apply_tp_fallback = fallback_module._apply_tp_fallback

    nvshmem.init_from_pg()
    team = nvshmem.team_world()
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(args.tokens, args.hidden, args.local_vocab, 1, team)
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(args.tokens, args.local_vocab)
    native_workspace_bytes = tvm_ffi.fused_linear_scaled_cross_entropy_forward_workspace_bytes(
        args.tokens, args.local_vocab
    ) + tvm_ffi.fused_linear_scaled_cross_entropy_backward_workspace_bytes(
        args.tokens, args.hidden, args.local_vocab, 1
    )

    x = torch.randn(
        args.tokens,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(123),
    ).mul_(0.03)
    weight = torch.randn(
        args.local_vocab,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(456 + rank),
    ).mul_(0.03)
    target = torch.randint(
        0,
        args.local_vocab * world_size,
        (args.tokens,),
        device=device,
        dtype=torch.int64,
        generator=torch.Generator(device=device).manual_seed(789),
    )
    vocab_start = rank * args.local_vocab
    inverse_temperature = 1.0 / TEMPERATURE
    nll = torch.empty(args.tokens, dtype=torch.float32, device=device)
    lse = torch.empty_like(nll)
    entropy = torch.zeros_like(nll)
    grad_output = torch.randn(
        args.tokens,
        device=device,
        dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(987),
    )
    entropy_grad = torch.zeros_like(grad_output)
    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    fallback_x = x.detach().requires_grad_(True)
    fallback_weight = weight.detach().requires_grad_(True)
    module = tvm_ffi._load_module()

    def native():
        module.fused_linear_scaled_cross_entropy_forward(
            x,
            weight,
            target,
            vocab_start,
            IGNORE_INDEX,
            inverse_temperature,
            team,
            False,
            nll,
            lse,
            entropy,
        )
        return nll

    def fallback():
        return apply_tp_fallback(
            x,
            weight,
            target,
            TEMPERATURE,
            IGNORE_INDEX,
            False,
            group,
        )

    def native_e2e():
        native()
        module.fused_linear_scaled_cross_entropy_backward(
            grad_output,
            entropy_grad,
            x,
            weight,
            target,
            lse,
            entropy,
            vocab_start,
            IGNORE_INDEX,
            inverse_temperature,
            team,
            1,
            False,
            grad_input,
            grad_weight,
        )
        return grad_input, grad_weight

    def fallback_e2e():
        fallback_nll = apply_tp_fallback(
            fallback_x,
            fallback_weight,
            target,
            TEMPERATURE,
            IGNORE_INDEX,
            False,
            group,
        )
        return torch.autograd.grad(
            fallback_nll,
            (fallback_x, fallback_weight),
            grad_output,
        )

    native()
    native_reference = nll.clone()
    fallback_reference = fallback()
    torch.cuda.synchronize()
    max_abs = (native_reference - fallback_reference).abs().max()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)

    native_ms, fallback_ms = _time_pair_ms(native, fallback, args.warmups, args.iterations, group)
    native_peak = _peak_increment_bytes(native, group)
    fallback_peak = _peak_increment_bytes(fallback, group)
    flops = 2.0 * args.tokens * args.hidden * args.local_vocab
    if not args.forward_only:
        native_e2e_ms, fallback_e2e_ms = _time_pair_ms(native_e2e, fallback_e2e, args.warmups, args.iterations, group)
        native_e2e_peak = _peak_increment_bytes(native_e2e, group)
        fallback_e2e_peak = _peak_increment_bytes(fallback_e2e, group)
        e2e_flops = 3.0 * flops

    if rank == 0:
        print(
            f"FORWARD_COMPARE TP={world_size} M={args.tokens} "
            f"N_local={args.local_vocab} K={args.hidden} stages=5 "
            f"native_ms={native_ms:.6f} "
            f"native_tflops={flops / (native_ms * 1.0e9):.2f} "
            f"fallback_ms={fallback_ms:.6f} "
            f"fallback_tflops={flops / (fallback_ms * 1.0e9):.2f} "
            f"speedup={fallback_ms / native_ms:.3f} "
            f"native_workspace_mib={native_workspace_bytes / (1024 * 1024):.2f} "
            f"native_peak_mib={native_peak / (1024 * 1024):.2f} "
            f"fallback_peak_mib={fallback_peak / (1024 * 1024):.2f} "
            f"max_abs={float(max_abs.item()):.6g}",
            flush=True,
        )
        if not args.forward_only:
            print(
                f"E2E_COMPARE TP={world_size} M={args.tokens} "
                f"N_local={args.local_vocab} K={args.hidden} stages=5 "
                f"native_ms={native_e2e_ms:.6f} "
                f"native_tflops={e2e_flops / (native_e2e_ms * 1.0e9):.2f} "
                f"fallback_ms={fallback_e2e_ms:.6f} "
                f"fallback_tflops={e2e_flops / (fallback_e2e_ms * 1.0e9):.2f} "
                f"speedup={fallback_e2e_ms / native_e2e_ms:.3f} "
                f"native_workspace_mib={native_workspace_bytes / (1024 * 1024):.2f} "
                f"native_peak_mib={native_e2e_peak / (1024 * 1024):.2f} "
                f"fallback_peak_mib={fallback_e2e_peak / (1024 * 1024):.2f}",
                flush=True,
            )

    nvshmem.finalize()
    dist.destroy_process_group()
    return int(float(max_abs.item()) > 2e-2)


if __name__ == "__main__":
    raise SystemExit(main())
