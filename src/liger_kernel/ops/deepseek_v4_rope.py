import torch
import triton
import triton.language as tl


@triton.jit
def _deepseek_v4_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    x_stride_d,
    cos_stride_b,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_s,
    sin_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
    total_rows,
    num_heads,
    seq_len,
    trig_batch_size,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    use_fp64_trig: tl.constexpr,
    sin_sign: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM
    dims = tl.arange(0, BLOCK_SIZE)
    valid_dim = dims < head_dim
    nope_dim = head_dim - rope_dim
    is_rope = dims >= nope_dim
    rope_dims = tl.maximum(dims - nope_dim, 0)
    pair_indices = rope_dims // 2
    partner_dims = tl.where((rope_dims & 1) == 0, dims + 1, dims - 1)
    is_even = (rope_dims & 1) == 0

    for row_offset in tl.range(0, ROWS_PER_PROGRAM, num_stages=1):
        row = row_start + row_offset
        valid = valid_dim & (row < total_rows)
        seq_idx = row % seq_len
        head_idx = (row // seq_len) % num_heads
        batch_idx = row // (seq_len * num_heads)
        trig_batch_idx = tl.where(trig_batch_size == 1, 0, batch_idx)

        x_base = batch_idx * x_stride_b + head_idx * x_stride_h + seq_idx * x_stride_s
        output_base = batch_idx * output_stride_b + head_idx * output_stride_h + seq_idx * output_stride_s
        cos_base = trig_batch_idx * cos_stride_b + seq_idx * cos_stride_s
        sin_base = trig_batch_idx * sin_stride_b + seq_idx * sin_stride_s

        x_native = tl.load(x_ptr + x_base + dims * x_stride_d, mask=valid, other=0.0)
        partner_native = tl.load(
            x_ptr + x_base + partner_dims * x_stride_d,
            mask=valid & is_rope,
            other=0.0,
        )
        cos = tl.load(
            cos_ptr + cos_base + pair_indices * cos_stride_d,
            mask=valid & is_rope,
            other=1.0,
        )
        sin = tl.load(
            sin_ptr + sin_base + pair_indices * sin_stride_d,
            mask=valid & is_rope,
            other=0.0,
        )

        x = x_native.to(tl.float32)
        partner = partner_native.to(tl.float32)
        if use_fp64_trig:
            x = x.to(tl.float64)
            partner = partner.to(tl.float64)
            cos = cos.to(tl.float64)
            sin = sin.to(tl.float64)
        else:
            cos = cos.to(tl.float32)
            sin = sin.to(tl.float32)
        sin *= sin_sign

        rotated = tl.where(is_even, x * cos - partner * sin, x * cos + partner * sin)
        output = tl.where(is_rope, rotated, x_native)
        tl.store(output_ptr + output_base + dims * output_stride_d, output, mask=valid)


def _normalize_unsqueeze_dim(unsqueeze_dim: int) -> int:
    if unsqueeze_dim < 0:
        unsqueeze_dim += 4
    if unsqueeze_dim not in (1, 2):
        raise ValueError(f"unsqueeze_dim must be 1 or 2, got {unsqueeze_dim}")
    return unsqueeze_dim


def _validate_inputs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int) -> int:
    unsqueeze_dim = _normalize_unsqueeze_dim(unsqueeze_dim)
    if x.ndim != 4:
        raise ValueError(f"x must be a 4D tensor, got shape {tuple(x.shape)}")
    if cos.ndim != 3 or sin.ndim != 3:
        raise ValueError(f"cos and sin must be 3D tensors, got shapes {tuple(cos.shape)} and {tuple(sin.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have the same shape, got {tuple(cos.shape)} and {tuple(sin.shape)}")
    if x.device != cos.device or x.device != sin.device:
        raise ValueError(f"x, cos, and sin must be on the same device, got {x.device}, {cos.device}, and {sin.device}")
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if x.dtype not in supported_dtypes or cos.dtype not in supported_dtypes or sin.dtype not in supported_dtypes:
        raise ValueError("x, cos, and sin must have float16, bfloat16, float32, or float64 dtype")

    batch_size = x.shape[0]
    seq_len = x.shape[3 - unsqueeze_dim]
    if cos.shape[0] not in (1, batch_size):
        raise ValueError(f"cos and sin batch size must be 1 or {batch_size}, got {cos.shape[0]}")
    if cos.shape[1] != seq_len:
        raise ValueError(f"cos and sin sequence length must be {seq_len}, got {cos.shape[1]}")

    head_dim = x.shape[-1]
    rope_dim = cos.shape[-1] * 2
    if rope_dim == 0 or rope_dim > head_dim:
        raise ValueError(f"rope_dim must be in [2, {head_dim}], got {rope_dim}")
    return unsqueeze_dim


def deepseek_v4_rope_forward(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
    sin_sign: float = 1.0,
) -> torch.Tensor:
    unsqueeze_dim = _validate_inputs(x, cos, sin, unsqueeze_dim)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if output.numel() == 0:
        return output

    batch_size = x.shape[0]
    num_heads = x.shape[unsqueeze_dim]
    seq_dim = 3 - unsqueeze_dim
    seq_len = x.shape[seq_dim]
    head_dim = x.shape[-1]
    rope_dim = cos.shape[-1] * 2
    total_rows = batch_size * num_heads * seq_len
    rows_per_program = 4
    block_size = triton.next_power_of_2(head_dim)
    num_warps = 8 if block_size >= 256 else 4

    _deepseek_v4_rope_kernel[(triton.cdiv(total_rows, rows_per_program),)](
        x,
        cos,
        sin,
        output,
        x.stride(0),
        x.stride(unsqueeze_dim),
        x.stride(seq_dim),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        output.stride(0),
        output.stride(unsqueeze_dim),
        output.stride(seq_dim),
        output.stride(3),
        total_rows,
        num_heads,
        seq_len,
        cos.shape[0],
        head_dim,
        rope_dim,
        cos.dtype == torch.float64 or sin.dtype == torch.float64,
        sin_sign,
        rows_per_program,
        block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


class LigerDeepseekV4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, unsqueeze_dim=1):
        unsqueeze_dim = _normalize_unsqueeze_dim(unsqueeze_dim)
        output = deepseek_v4_rope_forward(x, cos, sin, unsqueeze_dim)
        ctx.save_for_backward(cos, sin)
        ctx.unsqueeze_dim = unsqueeze_dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad_input = deepseek_v4_rope_forward(
            grad_output,
            cos,
            sin,
            ctx.unsqueeze_dim,
            sin_sign=-1.0,
        )
        return grad_input, None, None, None
