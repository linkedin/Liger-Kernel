import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import infer_device
from liger_kernel.utils import is_npu_available

# Feature widths whose block is at most this many columns get BLOCK_M rows per
# program (millions of rows of 16-128 channels, where one program per row
# starves the GPU). Wider rows keep one row per program.
_ROW_TILE_MAX_BLOCK_N = 256
# Elements per program on the row-tiled path, the block size used by
# `_block_rms_norm` for the same tradeoff.
_ROW_TILE_ELEMENTS = 4096
# Backward programs per SM on the row-tiled path.
_ROW_TILE_BWD_PROGRAMS_PER_SM = 4
# Smallest BLOCK_N in the autotune configs below. The backward buffers are
# sized for this worst case because the autotuned BLOCK_N is not known
# before launch (see liger_dyt_bwd).
_MIN_AUTOTUNE_BLOCK_N = 64


def _dyt_configs():
    configs = []
    for block_n in (64, 128, 256, 512, 1024, 2048, 4096):
        block_m = _ROW_TILE_ELEMENTS // block_n if block_n <= _ROW_TILE_MAX_BLOCK_N else 1
        warp_counts = (4, 8) if block_n < 1024 else (4, 8, 16)
        for num_stages in (1, 2):
            for num_warps in warp_counts:
                configs.append(
                    triton.Config({"BLOCK_N": block_n, "BLOCK_M": block_m}, num_stages=num_stages, num_warps=num_warps)
                )
    return configs


def _dyt_bwd_num_programs(meta, num_rows, num_sms):
    if meta["BLOCK_M"] > 1:
        return min(num_sms * _ROW_TILE_BWD_PROGRAMS_PER_SM, triton.cdiv(num_rows, meta["BLOCK_M"]))
    return num_sms


if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.autotune(
    configs=_dyt_configs(),
    key=["N"],
)
@triton.jit
def _dyt_fwd_kernel(
    X,
    Y,
    Alpha,
    Gamma,
    Beta,
    HAVE_BETA: tl.constexpr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # A 1D grid, which allows 2**31 - 1 programs (dimensions 1 and 2 allow
    # 65,535), with a row's column blocks adjacent.
    pid = tl.cast(tl.program_id(0), tl.int64)
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    row = (pid // num_col_blocks) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = (pid % num_col_blocks) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col < N
    mask = (row < M)[:, None] & col_mask[None, :]
    offsets = row[:, None] * N + col[None, :]

    alpha = tl.load(Alpha).to(tl.float32)
    gamma = tl.load(Gamma + col, mask=col_mask, other=0.0).to(tl.float32)
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)

    tanh_x = tanh(alpha * x)
    y = tanh_x * gamma[None, :]
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=col_mask, other=0.0).to(tl.float32)
        y += beta[None, :]
    tl.store(Y + offsets, y, mask=mask)


@triton.autotune(
    configs=_dyt_configs(),
    key=["N"],
    # DA, DG, and DB are sized for the worst-case config, so different
    # (BLOCK_M, BLOCK_N) configs write to different slot/row counts. Autotune
    # trials don't zero outputs between runs, so stale values from a prior
    # trial would leak into the sums in liger_dyt_bwd. Reset them between
    # trials to isolate each config's writes.
    reset_to_zero=["DA", "DG", "DB"],
)
@triton.jit
def _dyt_bwd_kernel(
    DY,
    DX,
    DA,
    DG,
    DB,
    X,
    Alpha,
    Gamma,
    HAVE_BETA: tl.constexpr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.cast(tl.program_id(0), tl.int64)
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    row_block_id = pid // num_col_blocks
    col_block_id = pid % num_col_blocks
    col = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col < N

    alpha = tl.load(Alpha).to(tl.float32)
    gamma = tl.load(Gamma + col, mask=col_mask, other=0.0).to(tl.float32)
    da = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if HAVE_BETA:
        db = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_row_programs = tl.num_programs(0) // num_col_blocks
    for start_row in range(row_block_id * BLOCK_M, M, num_row_programs * BLOCK_M):
        row = start_row + tl.arange(0, BLOCK_M)
        # Mask padded rows out after the arithmetic, not just on load:
        # with infinite alpha, tanh(inf * 0) is NaN and would poison the
        # parameter gradients if the padded rows were reduced.
        mask = (row < M)[:, None] & col_mask[None, :]
        offsets = row[:, None] * N + col[None, :]
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0.0).to(tl.float32)
        tanh_x = tanh(alpha * x)
        if HAVE_BETA:
            db += tl.where(mask, dy, 0.0)
        dg += tl.where(mask, dy * tanh_x, 0.0)
        tmp = (1 - tanh_x * tanh_x) * dy * gamma[None, :]
        da += tl.where(mask, x * tmp, 0.0)
        dx = alpha * tmp
        tl.store(DX + offsets, dx, mask=mask)

    tl.store(DG + row_block_id * N + col, tl.sum(dg, 0), mask=col_mask)
    if HAVE_BETA:
        tl.store(DB + row_block_id * N + col, tl.sum(db, 0), mask=col_mask)
    tl.store(DA + pid, tl.sum(tl.sum(da, 1), 0))


def liger_dyt_fwd(x, alpha, gamma, beta):
    assert x.is_contiguous()
    HAVE_BETA = True if beta is not None else False
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape

    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _dyt_fwd_kernel[grid](
        x,
        y,
        alpha,
        gamma,
        beta,
        HAVE_BETA,
        M,
        N,
    )
    return y.view(input_shape)


def liger_dyt_bwd(dy, x, alpha, gamma, beta):
    assert dy.is_contiguous()
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N = x.shape
    HAVE_BETA = True if beta is not None else False

    device = infer_device()
    if device == "cuda":
        NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    elif device == "xpu":
        NUM_SMS = torch.xpu.get_device_properties(x.device).gpu_subslice_count
    elif device == "npu":
        NUM_SMS = get_npu_core_count()
    # Sized for the worst-case autotuned config (most programs, smallest
    # BLOCK_N), and zeroed so programs/column blocks a config does not use
    # contribute nothing to the sums below.
    max_programs = NUM_SMS * _ROW_TILE_BWD_PROGRAMS_PER_SM
    da = torch.zeros(max_programs, triton.cdiv(N, _MIN_AUTOTUNE_BLOCK_N), dtype=torch.float32, device=x.device)
    dg = torch.zeros(max_programs, N, dtype=torch.float32, device=x.device)
    db = (
        torch.zeros(max_programs, N, dtype=torch.float32, device=x.device)
        if HAVE_BETA
        else torch.empty(0, dtype=torch.float32, device=x.device)
    )
    dx = torch.empty_like(dy)

    grid = lambda meta: (_dyt_bwd_num_programs(meta, M, NUM_SMS) * triton.cdiv(N, meta["BLOCK_N"]),)
    _dyt_bwd_kernel[grid](dy, dx, da, dg, db, x, alpha, gamma, HAVE_BETA, M, N)
    db = db.sum(0).to(beta.dtype) if HAVE_BETA else None
    dg = dg.sum(0).to(gamma.dtype)
    # dalpha takes alpha's dtype, like dgamma takes gamma's: rounding a
    # float32 parameter gradient through x.dtype (e.g. bfloat16) loses it.
    da = da.sum().to(alpha.dtype).unsqueeze(0)
    return dx.view(input_shape), da, dg, db


class LigerDyTFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, alpha, gamma, beta):
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy):
        x, alpha, gamma, beta = ctx.saved_tensors
        dx, dalpha, dgamma, dbeta = liger_dyt_bwd(dy, x, alpha, gamma, beta)
        return dx, dalpha, dgamma, dbeta
