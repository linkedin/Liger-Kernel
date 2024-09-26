import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple, Optional


def attention_pack(
    x: torch.Tensor,  # [batch_size, seqlen, num_heads, head_dim]
    attention_mask: torch.Tensor,  # [batch_size, seqlen]
) -> torch.Tensor:
    to_pack = []
    for i, attn_mask in enumerate(attention_mask):
        seqlen = attn_mask.sum().int().item()
        kept = x[i, :seqlen]  # [seqlen, num_heads, head_dim]
        to_pack.append(kept)
    return torch.concatenate(to_pack, dim=0).unsqueeze(0)


def attention_unpack(
    x: torch.Tensor,  # [1, sum_seqlens, num_heads, head_dim]
    cum_seqlens: torch.Tensor,  # [0, seqlen_1, seqlen_1+seqlen_2, ...]
    batch_size: int,
    goal_seqlen: int,
) -> torch.Tensor:
    unpacked = torch.zeros(size=(batch_size, goal_seqlen, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(cum_seqlens.size(0)-1):
        seq_start = cum_seqlens[i]
        seq_end = cum_seqlens[i+1]
        unpacked[i, :seq_end-seq_start] = x[0, seq_start:seq_end]
    return unpacked


@triton.jit
def load_fn(
    ptrs,
    offs_axis_0: tl.const_pointer_type,
    offs_axis_1: tl.const_pointer_type,
    PAD_AXIS_0: tl.constexpr,
    PAD_AXIS_1: tl.constexpr,
    LIM_AXIS_0: tl.constexpr,
    LIM_AXIS_1: tl.constexpr,
):
    if PAD_AXIS_0:
        if PAD_AXIS_1:
            x = tl.load(ptrs, mask=(offs_axis_0[:, None] < LIM_AXIS_0) & (offs_axis_1[None, :] < LIM_AXIS_1), other=0.0)
        else:
            x = tl.load(ptrs, mask=offs_axis_0[:, None] < LIM_AXIS_0, other=0.0)
    else:
        if PAD_AXIS_1:
            x = tl.load(ptrs, mask=offs_axis_1[None, :] < LIM_AXIS_1, other=0.0)
        else:
            x = tl.load(ptrs)
    return x


def infer_bias_strides(
    bias: Optional[Tensor], batch: int, nheads_q: int, seqlen_q: int, seqlen_k: int,
) -> Tuple[int, ...]:
    if bias is not None:
        assert (bias.size(2) == seqlen_q and bias.size(3) == seqlen_k), f"{bias.shape = }"
        if bias.size(0) == 1:
            stride_bb = 0
        elif bias.size(0) == batch:
            stride_bb = bias.stride(0)
        else:
            raise ValueError(f"Attention bias has {bias.size(0) = } while {batch = }")
        if bias.size(1) == 1:
            stride_bh = 0
        elif bias.stride(1) == nheads_q:
            stride_bh = bias.stride(1)
        else:
            raise ValueError(f"Attention bias has {bias.size(1) = } while {nheads_q = }")
        stride_bm = bias.stride(2)
    else:
        stride_bb, stride_bh, stride_bm = 0, 0, 0
    return stride_bb, stride_bh, stride_bm


def handle_dropout(dropout_p: float, dropout_seed: Optional[int], is_forward: bool) -> int:
    assert dropout_p >= 0, f"Dropout probability {dropout_p = } must be above 0."
    assert dropout_p < 1, f"Dropout probability {dropout_p = } must be strictly below 1."
    if dropout_p == 0:
        return 0
    elif is_forward:
        return torch.randint(low=0, high=2**32, size=(1,)).item() if dropout_seed is None else dropout_seed
    else:
        raise NotImplementedError("Backward pass does not yet support dropout.")


class torch_ignore_deterministic:
    def __enter__(self):
        self.previous_mode = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            raise exc_val
        torch.use_deterministic_algorithms(self.previous_mode)


def encode_dtype(x: Tensor) -> int:
    if x.dtype == torch.float16:
        return 16
    if x.dtype == torch.bfloat16:
        return 17
    if x.dtype == torch.float32:
        return 32
    raise ValueError(x.dtype)
