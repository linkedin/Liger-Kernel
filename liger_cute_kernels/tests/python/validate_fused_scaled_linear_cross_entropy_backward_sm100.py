"""TP1/2/4/8/16 correctness checks for the persistent SM100 FSLCE backward."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

INVERSE_TEMPERATURE = 1.0 / 0.9
IGNORE_INDEX = -100
WAVE_ROWS = 4096


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().abs().max().clamp_min(1e-8)
    return ((actual.float() - expected.float()).abs().max() / denominator).item()


def _softmax_stats(
    x: torch.Tensor,
    weight: torch.Tensor,
    group,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_logits = (x.float() @ weight.float().t()) * INVERSE_TEMPERATURE
    maximum = scaled_logits.amax(dim=-1)
    if world_size > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    shifted = torch.exp(scaled_logits - maximum[:, None])
    stats = torch.stack([shifted.sum(dim=-1), (shifted * scaled_logits).sum(dim=-1)])
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    total_sum, total_weighted = stats
    lse = maximum + torch.log(total_sum)
    entropy = lse - total_weighted / total_sum
    return lse.contiguous(), entropy.contiguous()


def _reference_dz(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    entropy_grad: torch.Tensor,
    lse: torch.Tensor,
    entropy: torch.Tensor,
    vocab_start: int,
    return_entropy: bool,
) -> torch.Tensor:
    logits = x.float() @ weight.float().t()
    scaled_logits = logits * INVERSE_TEMPERATURE
    probability = torch.exp(scaled_logits - lse[:, None])
    if return_entropy:
        dz = probability * (grad_output[:, None] + entropy_grad[:, None] * ((lse - entropy)[:, None] - scaled_logits))
    else:
        dz = probability * grad_output[:, None]

    local_target = target - vocab_start
    valid = (target != IGNORE_INDEX) & (local_target >= 0) & (local_target < weight.shape[0])
    rows = torch.nonzero(valid).squeeze(-1)
    if rows.numel() > 0:
        dz[rows, local_target[rows]] -= grad_output[rows]
    dz[target == IGNORE_INDEX] = 0.0
    return (dz * INVERSE_TEMPERATURE).to(torch.bfloat16)


def _reference_dw(
    dz: torch.Tensor,
    x: torch.Tensor,
    wave_rows: int,
) -> torch.Tensor:
    result = None
    for begin in range(0, x.shape[0], wave_rows):
        end = min(begin + wave_rows, x.shape[0])
        contribution = (dz[begin:end].float().t() @ x[begin:end].float()).to(torch.bfloat16)
        result = contribution if result is None else (result.float() + contribution.float()).to(torch.bfloat16)
    assert result is not None
    return result


def _run_case(
    tvm_ffi,
    team: int,
    device: torch.device,
    rank: int,
    world_size: int,
    group,
    wave_rows: int,
    name: str,
    tokens: int,
    hidden: int,
    local_vocab: int,
    return_entropy: bool,
    ignore_period: int,
) -> bool:
    x_generator = torch.Generator(device=device).manual_seed(1234)
    weight_generator = torch.Generator(device=device).manual_seed(4321 + rank)
    target_generator = torch.Generator(device=device).manual_seed(99)
    grad_generator = torch.Generator(device=device).manual_seed(7)

    x = torch.randn(
        tokens,
        hidden,
        generator=x_generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.05)
    weight = torch.randn(
        local_vocab,
        hidden,
        generator=weight_generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.05)
    global_vocab = local_vocab * world_size
    vocab_start = rank * local_vocab
    target = torch.randint(
        0,
        global_vocab,
        (tokens,),
        generator=target_generator,
        device=device,
        dtype=torch.int64,
    )
    if ignore_period:
        ignored = torch.arange(tokens, device=device) % ignore_period == 0
        target = torch.where(ignored, torch.full_like(target, IGNORE_INDEX), target)

    grad_output = torch.randn(
        tokens,
        generator=grad_generator,
        device=device,
        dtype=torch.float32,
    ).mul_(0.5)
    entropy_grad = (
        torch.randn(
            tokens,
            generator=grad_generator,
            device=device,
            dtype=torch.float32,
        ).mul_(0.25)
        if return_entropy
        else torch.zeros(tokens, device=device, dtype=torch.float32)
    )
    lse, entropy = _softmax_stats(x, weight, group, world_size)

    grad_input, grad_weight = tvm_ffi.fused_linear_scaled_cross_entropy_backward(
        grad_output,
        entropy_grad,
        x,
        weight,
        target,
        lse,
        entropy,
        vocab_start,
        IGNORE_INDEX,
        INVERSE_TEMPERATURE,
        team,
        1,
        return_entropy,
    )
    torch.cuda.synchronize()

    dz = _reference_dz(
        x,
        weight,
        target,
        grad_output,
        entropy_grad,
        lse,
        entropy,
        vocab_start,
        return_entropy,
    )
    dx_reference = dz.float() @ weight.float()
    if world_size > 1:
        dist.all_reduce(dx_reference, op=dist.ReduceOp.SUM, group=group)
    dx_reference = dx_reference.to(torch.bfloat16)
    dw_reference = _reference_dw(dz, x, wave_rows)
    dx_error = _relative_error(grad_input, dx_reference)
    dw_error = _relative_error(grad_weight, dw_reference)
    cross_rank_error = 0.0
    if world_size > 1:
        gathered = [torch.empty_like(grad_input) for _ in range(world_size)]
        dist.all_gather(gathered, grad_input.contiguous(), group=group)
        cross_rank_error = max((other.float() - gathered[0].float()).abs().max().item() for other in gathered[1:])
    passed = dx_error < 2e-2 and dw_error < 2e-2 and cross_rank_error == 0.0
    if rank == 0:
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name:<30} "
            f"TP={world_size} tokens={tokens:<5} hidden={hidden:<5} "
            f"local_vocab={local_vocab:<5} entropy={int(return_entropy)} "
            f"dx_rel={dx_error:.3e} dw_rel={dw_error:.3e} "
            f"cross_rank={cross_rank_error:.3e}",
            flush=True,
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--local-vocab", type=int, default=4096)
    parser.add_argument("--case", default="")
    parser.add_argument(
        "--wave-rows",
        type=int,
        choices=(1024, 2048, 4096),
        default=WAVE_ROWS,
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
        raise RuntimeError(f"validation supports TP1/2/4/8/16, got TP{world_size}")

    ragged_vocab = max(256, args.local_vocab - 88)
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_forward(args.max_tokens, args.local_vocab)
    tvm_ffi.fused_linear_scaled_cross_entropy_configure_backward(
        args.max_tokens, args.hidden, args.local_vocab, 1, team
    )
    cases = [
        ("aligned", 1024, args.hidden, args.local_vocab, False, 0),
        ("entropy", 1024, args.hidden, args.local_vocab, True, 0),
        ("ragged ignored", 1500, args.hidden, args.local_vocab, False, 7),
        ("ragged entropy ignored", 1500, args.hidden, ragged_vocab, True, 5),
        ("full wave", 4096, args.hidden, args.local_vocab, False, 0),
        ("two-wave boundary", 4097, args.hidden, ragged_vocab, True, 13),
        ("two-wave tail", 5000, args.hidden, args.local_vocab, False, 17),
    ]
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            raise RuntimeError(f"unknown case: {args.case}")

    failures = 0
    for case in cases:
        dist.barrier()
        failures += not _run_case(
            tvm_ffi,
            team,
            device,
            rank,
            world_size,
            group,
            args.wave_rows,
            *case,
        )

    if rank == 0:
        print(
            f"TP{world_size} failures={failures}/{len(cases)}",
            flush=True,
        )
    nvshmem.finalize()
    dist.destroy_process_group()
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
