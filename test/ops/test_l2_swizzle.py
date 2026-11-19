import pytest
import torch
import triton
import triton.language as tl

from liger_kernel.ops.mlp import _l2_swizzle


@triton.jit
def _swizzle_test_kernel(pids, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    i = tl.program_id(0)
    pid_m, pid_n = _l2_swizzle(tl.load(pids + i), M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
    tl.store(out + 2 * i, pid_m)
    tl.store(out + 2 * i + 1, pid_n)


def _reference_swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    num_pid_m = -(-M // BLOCK_M)
    num_pid_n = -(-N // BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    base_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    effective_group_size_m = min(num_pid_m - base_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = (pid_in_group % effective_group_size_m) + base_pid_m
    pid_n = pid_in_group // effective_group_size_m
    return pid_m, pid_n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel execution")
@pytest.mark.parametrize(
    "M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M",
    [
        (1024, 2048, 64, 128, 8),
        (256, 512, 64, 64, 1),
        (2**31 - 1, 1024, 64, 128, 8),
        (1024, 2**31 - 1, 64, 128, 8),
        (2**31 - 64, 1024, 64, 128, 8),
        (1024, 2**31 - 128, 64, 128, 8),
    ],
)
def test_l2_swizzle_overflow_and_correctness(M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    grid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # Check head and tail pids of the launch grid
    pids = list(range(min(16, grid)))
    if grid > 32:
        pids += list(range(grid - 16, grid))
    pids = sorted(list(set(pids)))

    out = torch.empty(2 * len(pids), device="cuda", dtype=torch.int32)
    pids_tensor = torch.tensor(pids, device="cuda", dtype=torch.int32)

    _swizzle_test_kernel[(len(pids),)](
        pids_tensor,
        out,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    results = [tuple(t) for t in out.view(-1, 2).tolist()]
    for p, (got_m, got_n) in zip(pids, results):
        ref_m, ref_n = _reference_swizzle(p, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
        assert (got_m, got_n) == (ref_m, ref_n), (
            f"Mismatch at pid={p} for shape M={M}, N={N}: got ({got_m}, {got_n}), expected ({ref_m}, {ref_n})"
        )
