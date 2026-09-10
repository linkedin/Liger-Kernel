"""Time TP1/2/4/8/16 fused SM100 backward against isolated phase kernels."""

from __future__ import annotations

import argparse
import math
import os
import statistics

import torch
import torch.distributed as dist

INVERSE_TEMPERATURE = 1.0 / 0.9
IGNORE_INDEX = -100
WAVE_ROWS = 4096


def _time_ms(function, warmups: int, iterations: int, group) -> float:
    dist.barrier(group=group)
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    elapsed = torch.tensor(
        statistics.median(samples),
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
    return elapsed.item()


def _phase_flops(
    phase: int,
    tokens: int,
    hidden: int,
    local_vocab: int,
    wave_rows: int,
) -> float:
    waves = (tokens + wave_rows - 1) // wave_rows
    padded_vocab = (local_vocab + 63) // 64 * 64
    dz_n = (padded_vocab + 255) // 256 * 256
    dx_n = (hidden + 255) // 256 * 256
    dw_m = (padded_vocab + 255) // 256 * 256
    dw_n = (hidden + 255) // 256 * 256
    if phase == 1:
        return 2.0 * waves * wave_rows * dz_n * hidden
    if phase == 2:
        return 2.0 * waves * wave_rows * dx_n * padded_vocab
    if phase == 4:
        return 2.0 * waves * wave_rows * dw_m * dw_n
    raise ValueError(f"unknown phase {phase}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--local-vocab", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--phase-order",
        choices=("dz-dx-dw", "dw-dx-dz"),
        default="dz-dx-dw",
    )
    parser.add_argument(
        "--wave-rows",
        type=int,
        choices=(1024, 2048, 4096),
        default=WAVE_ROWS,
    )
    parser.add_argument("--sync-audit", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--standalone-inputs",
        action="store_true",
        help="Match the zero-input/uniform-metadata standalone GEMM benchmark.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD

    from liger_cute_kernels import nvshmem
    from liger_cute_kernels import tvm_ffi

    nvshmem.init_from_pg()
    team = nvshmem.team_world()
    if world_size not in (1, 2, 4, 8, 16):
        raise RuntimeError(f"benchmark supports TP1/2/4/8/16, got TP{world_size}")

    tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(args.tokens, args.local_vocab)
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(args.tokens, args.hidden, args.local_vocab, 1, team)

    input_generator = torch.Generator(device=device).manual_seed(123)
    weight_generator = torch.Generator(device=device).manual_seed(456 + rank)
    target_generator = torch.Generator(device=device).manual_seed(789)
    x = torch.randn(
        args.tokens,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=input_generator,
    ).mul_(0.03)
    weight = torch.randn(
        args.local_vocab,
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=weight_generator,
    ).mul_(0.03)
    target = torch.randint(
        0,
        args.local_vocab * world_size,
        (args.tokens,),
        device=device,
        dtype=torch.int64,
        generator=target_generator,
    )
    grad_output = torch.randn(
        args.tokens,
        device=device,
        dtype=torch.float32,
        generator=input_generator,
    )
    entropy_grad = torch.zeros_like(grad_output)
    lse = torch.zeros(args.tokens, device=device, dtype=torch.float32)
    entropy = torch.zeros_like(lse)
    if args.standalone_inputs:
        x.zero_()
        weight.zero_()
        target.copy_(torch.arange(args.tokens, device=device, dtype=torch.int64) % (args.local_vocab * world_size))
        target[::17] = IGNORE_INDEX
        grad_output.fill_(1.0)
        uniform_lse = math.log(float(args.local_vocab * world_size))
        lse.fill_(uniform_lse)
        entropy.fill_(uniform_lse)
        entropy_grad.fill_(0.125)
    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    module = tvm_ffi._load_module()
    common = (
        grad_output,
        entropy_grad,
        x,
        weight,
        target,
        lse,
        entropy,
        rank * args.local_vocab,
        IGNORE_INDEX,
        INVERSE_TEMPERATURE,
        team,
    )

    def run_phase(phase: int) -> None:
        module.fused_linear_scaled_cross_entropy_backward_phase_bench(*common, phase, False, grad_input, grad_weight)

    def run_fused() -> None:
        module.fused_linear_scaled_cross_entropy_backward(*common, 1, False, grad_input, grad_weight)

    def graph_callable(function):
        function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        return graph.replay

    run_phase(1)
    torch.cuda.synchronize()
    phase_functions = {
        phase: (
            graph_callable(lambda phase=phase: run_phase(phase))
            if args.cuda_graph
            else lambda phase=phase: run_phase(phase)
        )
        for phase in (1, 2, 4)
    }
    fused_function = graph_callable(run_fused) if args.cuda_graph else run_fused
    phase_order = (1, 2, 4) if args.phase_order == "dz-dx-dw" else (4, 2, 1)
    phase_ms = {
        phase: _time_ms(
            phase_functions[phase],
            args.warmups,
            args.iterations,
            group,
        )
        for phase in phase_order
    }
    fused_ms = _time_ms(
        fused_function,
        args.warmups,
        args.iterations,
        group,
    )
    sync_audit = None
    if args.sync_audit:
        empty_ms = _time_ms(
            lambda: run_phase(0),
            args.warmups,
            args.iterations,
            group,
        )
        pair_ms = {
            mask: _time_ms(
                lambda mask=mask: run_phase(mask),
                args.warmups,
                args.iterations,
                group,
            )
            for mask in (3, 5, 6)
        }
        no_dx_dw_barrier_ms = _time_ms(
            lambda: run_phase(14),
            args.warmups,
            args.iterations,
            group,
        )
        no_dz_dx_grid_ms = _time_ms(
            lambda: run_phase(19),
            args.warmups,
            args.iterations,
            group,
        )
        no_dz_dw_grid_ms = _time_ms(
            lambda: run_phase(21),
            args.warmups,
            args.iterations,
            group,
        )
        no_dz_drain_ms = _time_ms(
            lambda: run_phase(35),
            args.warmups,
            args.iterations,
            group,
        )
        forced_dz_grid_ms = _time_ms(
            lambda: run_phase(65),
            args.warmups,
            args.iterations,
            group,
        )
        forced_dx_dw_ms = _time_ms(
            lambda: run_phase(130),
            args.warmups,
            args.iterations,
            group,
        )
        sync_audit = {
            "empty": empty_ms,
            "dz_dx": pair_ms[3] - phase_ms[1] - phase_ms[2] + empty_ms,
            "dz_dw": pair_ms[5] - phase_ms[1] - phase_ms[4] + empty_ms,
            "dx_dw": pair_ms[6] - phase_ms[2] - phase_ms[4] + empty_ms,
            "total": fused_ms - sum(phase_ms.values()) + 2.0 * empty_ms,
            "pair_dz_dx": pair_ms[3],
            "pair_dz_dw": pair_ms[5],
            "pair_dx_dw": pair_ms[6],
            "pair_dx_dw_no_barrier": no_dx_dw_barrier_ms,
            "dx_dw_barrier_direct": pair_ms[6] - no_dx_dw_barrier_ms,
            "dz_dx_grid_direct": pair_ms[3] - no_dz_dx_grid_ms,
            "dz_dw_grid_direct": pair_ms[5] - no_dz_dw_grid_ms,
            "pair_dz_dx_no_grid": no_dz_dx_grid_ms,
            "pair_dz_dw_no_grid": no_dz_dw_grid_ms,
            "pair_dz_dx_no_drain": no_dz_drain_ms,
            "dz_drain_direct": pair_ms[3] - no_dz_drain_ms,
            "dz_grid_forced_direct": forced_dz_grid_ms - phase_ms[1],
            "dx_dw_forced_direct": forced_dx_dw_ms - phase_ms[2],
        }
    phase_names = {1: "dZ", 2: "dX", 4: "dW"}
    flops = {
        phase: _phase_flops(
            phase,
            args.tokens,
            args.hidden,
            args.local_vocab,
            args.wave_rows,
        )
        for phase in (1, 2, 4)
    }
    isolated_sum_ms = sum(phase_ms.values())
    total_flops = sum(flops.values())

    if rank == 0:
        print(
            f"TP{world_size} tokens={args.tokens} hidden={args.hidden} "
            f"local_vocab={args.local_vocab} waves="
            f"{(args.tokens + args.wave_rows - 1) // args.wave_rows} "
            f"wave_rows={args.wave_rows}"
        )
        for phase in (1, 2, 4):
            name = phase_names[phase]
            print(f"{name}: {phase_ms[phase]:.6f} ms, {flops[phase] / (phase_ms[phase] * 1e9):.2f} TFLOP/s")
        print(
            f"isolated_sum={isolated_sum_ms:.6f} ms "
            f"fused={fused_ms:.6f} ms "
            f"overhead={fused_ms - isolated_sum_ms:.6f} ms "
            f"({100.0 * (fused_ms - isolated_sum_ms) / isolated_sum_ms:.3f}%) "
            f"effective={total_flops / (fused_ms * 1e9):.2f} TFLOP/s/GPU"
        )
        if sync_audit is not None:
            print(
                "sync_audit "
                f"empty={sync_audit['empty']:.6f} ms "
                f"dZ_to_dX={sync_audit['dz_dx']:.6f} ms "
                f"dZ_to_dW={sync_audit['dz_dw']:.6f} ms "
                f"dX_to_dW={sync_audit['dx_dw']:.6f} ms "
                f"dX_to_dW_barrier_direct="
                f"{sync_audit['dx_dw_barrier_direct']:.6f} ms "
                f"dZ_to_dX_grid_direct="
                f"{sync_audit['dz_dx_grid_direct']:.6f} ms "
                f"dZ_to_dW_grid_direct="
                f"{sync_audit['dz_dw_grid_direct']:.6f} ms "
                f"dZ_drain_direct="
                f"{sync_audit['dz_drain_direct']:.6f} ms "
                f"dZ_grid_forced_direct="
                f"{sync_audit['dz_grid_forced_direct']:.6f} ms "
                f"dX_dW_forced_direct="
                f"{sync_audit['dx_dw_forced_direct']:.6f} ms "
                f"total_adjusted={sync_audit['total']:.6f} ms"
            )
            print(
                "pair_masks "
                f"dZ_dX={sync_audit['pair_dz_dx']:.6f} ms "
                f"dZ_dW={sync_audit['pair_dz_dw']:.6f} ms "
                f"dX_dW={sync_audit['pair_dx_dw']:.6f} ms "
                f"dX_dW_no_barrier="
                f"{sync_audit['pair_dx_dw_no_barrier']:.6f} ms "
                f"dZ_dX_no_grid="
                f"{sync_audit['pair_dz_dx_no_grid']:.6f} ms "
                f"dZ_dW_no_grid="
                f"{sync_audit['pair_dz_dw_no_grid']:.6f} ms "
                f"dZ_dX_no_drain="
                f"{sync_audit['pair_dz_dx_no_drain']:.6f} ms"
            )

    if args.timeline:
        diagnostics = tvm_ffi.fused_linear_scaled_cross_entropy_backward_diagnostics(device)
        torch.cuda.synchronize()
        gathered_diagnostics = [torch.empty_like(diagnostics) for _ in range(world_size)]
        dist.all_gather(
            gathered_diagnostics,
            diagnostics.contiguous(),
            group=group,
        )
        values = gathered_diagnostics[0].cpu().tolist()
        kernel_start = values[0]

        def relative(index: int) -> float:
            return (values[index] - kernel_start) / 1.0e6

        def duration(begin: int, end: int) -> float:
            return (values[end] - values[begin]) / 1.0e6

        if rank == 0:
            print(
                f"timeline kernel={duration(0, 1):.6f} ms "
                f"grid_wait_max={values[10] / 1.0e6:.6f} ms "
                f"dx_dw_wait_max={values[19] / 1.0e6:.6f} ms"
            )
            for name, indexes in {
                "dZ": (2, 3, 4, 5, 6, 7, 8, 9),
                "dX": (11, 12, 13, 14, 15, 16, 17, 18),
                "dW": (20, 21, 22, 23, 24, 25, 26, 27),
            }.items():
                (
                    tma_start,
                    tma_end,
                    mma_start,
                    mma_ready,
                    mma_end,
                    epi_start,
                    epi_ready,
                    epi_end,
                ) = indexes
                print(
                    f"timeline_{name} "
                    f"start={relative(tma_start):.6f} ms "
                    f"active={duration(tma_start, epi_end):.6f} ms "
                    f"tma={duration(tma_start, tma_end):.6f} ms "
                    f"mma={duration(mma_start, mma_end):.6f} ms "
                    f"epi={duration(epi_start, epi_end):.6f} ms "
                    f"mainloop_fill={duration(tma_start, mma_ready):.6f} ms "
                    f"mainloop_drain="
                    f"{max(0.0, duration(tma_end, mma_end)):.6f} ms "
                    f"acc_fill={duration(mma_start, epi_ready):.6f} ms "
                    f"epi_drain="
                    f"{max(0.0, duration(mma_end, epi_end)):.6f} ms"
                )
            for diagnostic_rank, rank_tensor in enumerate(gathered_diagnostics):
                rank_values = rank_tensor.cpu().tolist()
                rank_start = rank_values[0]

                def rank_relative(index: int) -> float:
                    return (rank_values[index] - rank_start) / 1.0e6

                local_start = rank_relative(28)
                local_reduce_end = rank_relative(29)
                local_store_end = rank_relative(30)
                dw_start = rank_relative(20)
                dw_end = max(
                    rank_relative(21),
                    rank_relative(24),
                    rank_relative(27),
                )
                overlap = max(
                    0.0,
                    min(local_store_end, dw_end) - max(local_start, dw_start),
                )
                tail = max(0.0, local_store_end - dw_end)
                print(
                    f"timeline_local rank={diagnostic_rank} "
                    f"start={local_start:.6f} ms "
                    f"reduce_end={local_reduce_end:.6f} ms "
                    f"store_end={local_store_end:.6f} ms "
                    f"dW_start={dw_start:.6f} ms "
                    f"dW_end={dw_end:.6f} ms "
                    f"overlap={overlap:.6f} ms "
                    f"tail={tail:.6f} ms"
                )
                if len(rank_values) > 32 and rank_values[31] != 0:
                    remote_start = rank_relative(31)
                    remote_end = rank_relative(32)
                    transport_end = rank_relative(33) if len(rank_values) > 33 and rank_values[33] != 0 else remote_end
                    remote_overlap = max(
                        0.0,
                        min(remote_end, dw_end) - max(remote_start, dw_start),
                    )
                    remote_tail = max(0.0, remote_end - dw_end)
                    print(
                        f"timeline_remote rank={diagnostic_rank} "
                        f"start={remote_start:.6f} ms "
                        f"end={remote_end:.6f} ms "
                        f"duration={remote_end - remote_start:.6f} ms "
                        f"transport={transport_end - remote_start:.6f} ms "
                        f"merge={remote_end - transport_end:.6f} ms "
                        f"dW_overlap={remote_overlap:.6f} ms "
                        f"tail={remote_tail:.6f} ms"
                    )

    nvshmem.finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
