import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)

    # cos size: (1, seq_len, head_dim)
    # stride: (seq_len * head_dim, head_dim, 1)
    pid = tl.program_id(0)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    if not BACKWARD_PASS:
        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        # with some math, we can get:
        # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


@triton.jit
def _triton_rope_paper(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)

    # cos size: (1, seq_len, head_dim // 2)
    # stride: (seq_len * head_dim, head_dim, 1)
    pid = tl.program_id(0)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    # ####################################################################
    # Load the even-indexed and odd-indexed elements of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # even-indexed elements of the head
    even_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :] * 2
    )
    even_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :] * 2
    )
    even_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] * 2 < hd
    )
    even_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] * 2 < hd
    )
    q_tile_even = tl.load(q_ptr + even_q_offsets, mask=even_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_even = tl.load(k_ptr + even_k_offsets, mask=even_k_mask, other=0).to(
        sin_row.dtype
    )

    # odd-indexed elements of the head
    odd_q_offsets = even_q_offsets + 1
    odd_k_offsets = even_k_offsets + 1
    odd_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] * 2 + 1 < hd
    )
    odd_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] * 2 + 1 < hd
    )
    q_tile_odd = tl.load(q_ptr + odd_q_offsets, mask=odd_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_odd = tl.load(k_ptr + odd_k_offsets, mask=odd_k_mask, other=0).to(
        sin_row.dtype
    )

    if not BACKWARD_PASS:
        # y_even = x_even * cos - x_odd * sin
        # y_odd = x_odd * cos + x_even * sin
        new_q_tile_even = q_tile_even * cos_row - q_tile_odd * sin_row
        tl.store(q_ptr + even_q_offsets, new_q_tile_even, mask=even_q_mask)
        new_q_tile_odd = q_tile_odd * cos_row + q_tile_even * sin_row
        tl.store(q_ptr + odd_q_offsets, new_q_tile_odd, mask=odd_q_mask)

        new_k_tile_even = k_tile_even * cos_row - k_tile_odd * sin_row
        tl.store(k_ptr + even_k_offsets, new_k_tile_even, mask=even_k_mask)
        new_k_tile_odd = k_tile_odd * cos_row + k_tile_even * sin_row
        tl.store(k_ptr + odd_k_offsets, new_k_tile_odd, mask=odd_k_mask)
    else:
        # dy_even = dx_even * cos + dx_odd * sin
        # dy_odd = dx_odd * cos - dx_even * sin
        new_q_tile_even = q_tile_even * cos_row + q_tile_odd * sin_row
        tl.store(q_ptr + even_q_offsets, new_q_tile_even, mask=even_q_mask)
        new_q_tile_odd = q_tile_odd * cos_row - q_tile_even * sin_row
        tl.store(q_ptr + odd_q_offsets, new_q_tile_odd, mask=odd_q_mask)

        new_k_tile_even = k_tile_even * cos_row + k_tile_odd * sin_row
        tl.store(k_ptr + even_k_offsets, new_k_tile_even, mask=even_k_mask)
        new_k_tile_odd = k_tile_odd * cos_row - k_tile_even * sin_row
        tl.store(k_ptr + odd_k_offsets, new_k_tile_odd, mask=odd_k_mask)


def rope_forward(q, k, cos, sin, paper_form):

    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    if not paper_form:
        _triton_rope[(n_row,)](
            q,
            q.stride(1),
            k,
            k.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            pad_n_q_head,
            pad_n_kv_head,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=False,
        )
    else:
        _triton_rope_paper[(n_row,)](
            q,
            q.stride(1),
            k,
            k.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            pad_n_q_head,
            pad_n_kv_head,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=False,
        )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin, paper_form):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = dq.shape
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure dq and dk are contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()

    # backward is similar to forward except swapping few ops
    if not paper_form:
        _triton_rope[(n_row,)](
            dq,
            dq.stride(1),
            dk,
            dk.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            pad_n_q_head,
            pad_n_kv_head,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=True,
        )
    else:
        _triton_rope_paper[(n_row,)](
            dq,
            dq.stride(1),
            dk,
            dk.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            pad_n_q_head,
            pad_n_kv_head,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=True,
        )
    return dq.transpose(1, 2), dk.transpose(1, 2)


class LigerRopeFunction(torch.autograd.Function):
    """
    Triton implementation of the Rotary Positional Embedding (RoPE) operation.
    This implements both HuggingFace Llama & Mistral version and the original RoPE paper version.

    Please find the corresponding HuggingFace Llama & Mistral implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    Please find the corresponding HuggingFace paper-form implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/roformer/modeling_roformer.py#L309

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        cos,
        sin,
        position_ids=None,
        unsqueeze_dim=1,
        paper_form: bool = False,
    ):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim)
        sin size: (1, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin, paper_form)
        ctx.paper_form = paper_form
        ctx.save_for_backward(cos, sin)
        return q, k

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim)
        sin size: (1, seq_len, head_dim)
        """

        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin, ctx.paper_form)
        return dq, dk, None, None, None, None, None
