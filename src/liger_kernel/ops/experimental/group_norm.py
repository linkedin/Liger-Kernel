import triton
import triton.language as tl
import torch
from torch._prims_common import suggest_memory_format

@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    # w2_over_w = weight_2 / new_weight
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )

@eval('''triton.heuristics({
    'BLOCK_SIZE': lambda kwargs: min(4096, triton.next_power_of_2(kwargs['HW'])),
})''')
@eval('''triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, triton.next_power_of_2(kwargs['HW'] // 128))),
    'C_G': lambda kwargs: kwargs['C'] // kwargs['groups'],
    'GROUP_SIZE': lambda kwargs: kwargs['C'] // kwargs['groups'] * kwargs['HW'],
})''')
@triton.jit
def group_norm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    N,
    C,
    HW,
    groups,
    eps,
    C_G,
    GROUP_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offset = pid_batch * C * HW + group * GROUP_SIZE
    X = input_ptr + offset
    Y = output_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + r, mask = r < GROUP_SIZE).to(tl.float32)
        m2_ = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        weight_ = (r < GROUP_SIZE).to(tl.float32)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_, weight_)
    
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1. / tl.sqrt(var + eps)

    for c in range(0, C_G):
        gamma = tl.load(gamma_ptr + group * C_G + c).to(tl.float32)
        beta = tl.load(beta_ptr + group * C_G + c).to(tl.float32)
        a = rstd * gamma
        b = beta - a * mean
        for off in range(0, HW, BLOCK_SIZE):
            r = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + c * HW + r, mask = r < HW).to(tl.float32)
            x = a * x + b
            tl.store(Y + c * HW + r, x, mask = r < HW)

def group_norm(
    input,
    groups,
    gamma,
    beta,
    eps
):
    assert input.is_cuda and gamma.is_cuda and beta.is_cuda
    N, C, H, W = input.shape
    assert C % groups == 0
    assert gamma.shape == (C, )
    assert beta.shape == (C, )
    assert suggest_memory_format(input) != torch.channels_last
    input = input.contiguous()
    output = torch.empty_like(input)

    def grid(meta):
        return (groups, N)
    
    group_norm_kernel[grid](input, gamma, beta, output, N, C, H * W, groups, eps)
    
    return output