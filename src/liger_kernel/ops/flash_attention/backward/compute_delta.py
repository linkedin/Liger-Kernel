import triton
import triton.language as tl
from triton import Config

MIN_B = 16


@triton.autotune(
    configs=[
        Config({"BLOCK_M": MIN_B}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 32}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 64}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 128}, num_warps=4, num_stages=0),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "DTYPE"],  # TODO: add dtype
)
@triton.jit
def _compute_delta(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    max_seqlen_q_rounded,
    cum_seqlens_q,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    DTYPE,
    VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # Locate kernel inside the grid
    start_m = tl.program_id(0)  # current block in the Q matrix
    off_head_and_batch = tl.program_id(1)
    off_batch = off_head_and_batch // nheads
    off_head = off_head_and_batch % nheads
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Infer actual sequence length of Q and the offset to the last sequence
    if VARLEN:
        actual_seqlen_q = tl.load(cum_seqlens_q + off_batch + 1) - tl.load(
            cum_seqlens_q + off_batch
        )
        cu_seq_start_q = tl.load(cum_seqlens_q + off_batch)
        off_batch = 0
    else:
        actual_seqlen_q = seqlen_q
        cu_seq_start_q = 0

    # Load the output tensor
    Out_offseted = (
        Out + off_batch * stride_ob + off_head * stride_oh + cu_seq_start_q * stride_om
    )
    o = tl.load(
        Out_offseted + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    # And its gradient
    DO_offseted = (
        DO
        + off_batch * stride_dob
        + off_head * stride_doh
        + cu_seq_start_q * stride_dom
    )
    do = tl.load(
        DO_offseted + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_head_and_batch * max_seqlen_q_rounded + offs_m, delta)
