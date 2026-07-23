import torch
import triton
import triton.language as tl


@triton.jit
def _short_conv_forward(
    bcx,
    weight,
    bias,
    output,
    n_elements,
    seq_len,
    hidden_size,
    stride_b,
    stride_t,
    stride_h,
    stride_wh,
    stride_wk,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    hidden = offsets % hidden_size
    tokens = offsets // hidden_size
    time = tokens % seq_len
    batch = tokens // seq_len

    conv = tl.zeros((BLOCK,), dtype=tl.float32)
    for lag in range(K):
        source_time = time - lag
        source_mask = mask & (source_time >= 0)
        source = batch * stride_b + source_time * stride_t + hidden * stride_h
        gate_b = tl.load(bcx + source, mask=source_mask, other=0.0).to(tl.float32)
        value = tl.load(bcx + source + 2 * hidden_size * stride_h, mask=source_mask, other=0.0).to(tl.float32)
        kernel = tl.load(weight + hidden * stride_wh + (K - 1 - lag) * stride_wk, mask=mask, other=0.0).to(tl.float32)
        conv += gate_b * value * kernel

    if HAS_BIAS:
        conv += tl.load(bias + hidden, mask=mask, other=0.0).to(tl.float32)

    current = batch * stride_b + time * stride_t + hidden * stride_h
    gate_c = tl.load(bcx + current + hidden_size * stride_h, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offsets, gate_c * conv, mask=mask)


@triton.jit
def _short_conv_input_backward(
    grad_output,
    bcx,
    weight,
    bias,
    grad_bcx,
    n_elements,
    seq_len,
    hidden_size,
    stride_b,
    stride_t,
    stride_h,
    stride_wh,
    stride_wk,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    hidden = offsets % hidden_size
    tokens = offsets // hidden_size
    time = tokens % seq_len
    batch = tokens // seq_len
    current = batch * stride_b + time * stride_t + hidden * stride_h

    conv = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_product = tl.zeros((BLOCK,), dtype=tl.float32)
    for lag in range(K):
        source_time = time - lag
        source_mask = mask & (source_time >= 0)
        source = batch * stride_b + source_time * stride_t + hidden * stride_h
        source_b = tl.load(bcx + source, mask=source_mask, other=0.0).to(tl.float32)
        source_x = tl.load(bcx + source + 2 * hidden_size * stride_h, mask=source_mask, other=0.0).to(tl.float32)
        kernel = tl.load(weight + hidden * stride_wh + (K - 1 - lag) * stride_wk, mask=mask, other=0.0).to(tl.float32)
        conv += source_b * source_x * kernel

        output_time = time + lag
        output_mask = mask & (output_time < seq_len)
        output_offset = (batch * seq_len + output_time) * hidden_size + hidden
        output_base = batch * stride_b + output_time * stride_t + hidden * stride_h
        grad_y = tl.load(grad_output + output_offset, mask=output_mask, other=0.0).to(tl.float32)
        output_c = tl.load(bcx + output_base + hidden_size * stride_h, mask=output_mask, other=0.0).to(tl.float32)
        grad_product += grad_y * output_c * kernel

    if HAS_BIAS:
        conv += tl.load(bias + hidden, mask=mask, other=0.0).to(tl.float32)

    current_b = tl.load(bcx + current, mask=mask, other=0.0).to(tl.float32)
    current_x = tl.load(bcx + current + 2 * hidden_size * stride_h, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)

    grad_base = tokens * 3 * hidden_size + hidden
    tl.store(grad_bcx + grad_base, grad_product * current_x, mask=mask)
    tl.store(grad_bcx + grad_base + hidden_size, grad_y * conv, mask=mask)
    tl.store(grad_bcx + grad_base + 2 * hidden_size, grad_product * current_b, mask=mask)


@triton.jit
def _short_conv_weight_backward(
    grad_output,
    bcx,
    weight_partials,
    bias_partials,
    seq_len,
    hidden_size,
    batch_tokens,
    n_chunks,
    stride_b,
    stride_t,
    stride_h,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    hidden = tl.program_id(0)
    chunk = tl.program_id(1)
    batch_token = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = batch_token < batch_tokens
    batch = batch_token // seq_len
    output_time = batch_token % seq_len
    output_offset = batch_token * hidden_size + hidden
    output_base = batch * stride_b + output_time * stride_t + hidden * stride_h
    grad_y = tl.load(grad_output + output_offset, mask=mask, other=0.0).to(tl.float32)
    gate_c = tl.load(bcx + output_base + hidden_size * stride_h, mask=mask, other=0.0).to(tl.float32)
    grad_conv = grad_y * gate_c

    # One program handles all taps for a hidden channel, reusing grad_y and
    # gate_c and reducing the launch grid by K.
    for kernel_idx in range(K):
        source_time = output_time + kernel_idx - (K - 1)
        source_mask = mask & (source_time >= 0)
        source = batch * stride_b + source_time * stride_t + hidden * stride_h
        gate_b = tl.load(bcx + source, mask=source_mask, other=0.0).to(tl.float32)
        value = tl.load(bcx + source + 2 * hidden_size * stride_h, mask=source_mask, other=0.0).to(tl.float32)
        partial_offset = (hidden * K + kernel_idx) * n_chunks + chunk
        tl.store(weight_partials + partial_offset, tl.sum(grad_conv * gate_b * value, axis=0))

    if HAS_BIAS:
        tl.store(bias_partials + hidden * n_chunks + chunk, tl.sum(grad_conv, axis=0))


class LigerLfm2ShortConvFunction(torch.autograd.Function):
    """Fuses the B*x gate, causal depthwise convolution, and C gate."""

    @staticmethod
    def forward(ctx, bcx, weight, bias=None):
        if bcx.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError("expected bcx [batch, time, 3*hidden] and weight [hidden, 1, kernel]")
        batch_size, seq_len, three_hidden = bcx.shape
        hidden_size = weight.shape[0]
        if three_hidden != 3 * hidden_size:
            raise ValueError("bcx final dimension must equal three times the convolution hidden size")

        output = torch.empty((batch_size, seq_len, hidden_size), dtype=bcx.dtype, device=bcx.device)
        n_elements = output.numel()
        _short_conv_forward[(triton.cdiv(n_elements, 512),)](
            bcx,
            weight,
            bias,
            output,
            n_elements,
            seq_len,
            hidden_size,
            bcx.stride(0),
            bcx.stride(1),
            bcx.stride(2),
            weight.stride(0),
            weight.stride(2),
            K=weight.shape[2],
            HAS_BIAS=bias is not None,
            BLOCK=512,
        )
        saved_bias = bias if bias is not None else torch.empty(0, dtype=bcx.dtype, device=bcx.device)
        ctx.save_for_backward(bcx, weight, saved_bias)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bcx, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        batch_size, seq_len, three_hidden = bcx.shape
        hidden_size = three_hidden // 3
        kernel_size = weight.shape[2]
        n_elements = batch_size * seq_len * hidden_size

        grad_bcx = torch.empty_like(bcx)
        _short_conv_input_backward[(triton.cdiv(n_elements, 512),)](
            grad_output,
            bcx,
            weight,
            bias,
            grad_bcx,
            n_elements,
            seq_len,
            hidden_size,
            bcx.stride(0),
            bcx.stride(1),
            bcx.stride(2),
            weight.stride(0),
            weight.stride(2),
            K=kernel_size,
            HAS_BIAS=ctx.has_bias,
            BLOCK=512,
        )

        batch_tokens = batch_size * seq_len
        n_chunks = triton.cdiv(batch_tokens, 256)
        partial_shape = (hidden_size * kernel_size, n_chunks)
        weight_partials = torch.empty(partial_shape, dtype=torch.float32, device=bcx.device)
        bias_partials = (
            torch.empty((hidden_size, n_chunks), dtype=torch.float32, device=bcx.device)
            if ctx.has_bias
            else weight_partials
        )
        _short_conv_weight_backward[(hidden_size, n_chunks)](
            grad_output,
            bcx,
            weight_partials,
            bias_partials,
            seq_len,
            hidden_size,
            batch_tokens,
            n_chunks,
            bcx.stride(0),
            bcx.stride(1),
            bcx.stride(2),
            K=kernel_size,
            HAS_BIAS=ctx.has_bias,
            BLOCK=256,
        )
        grad_weight = weight_partials.sum(1).reshape(hidden_size, 1, kernel_size).to(weight.dtype)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = bias_partials.sum(1).to(bias.dtype)
        return grad_bcx, grad_weight, grad_bias
