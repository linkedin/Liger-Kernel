import triton
import triton.language as tl
import torch

# Naive 2D convolution kernel (NCHW, stride=1, padding=0)
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr, P: tl.constexpr,
):
    oc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    acc = 0.0
    for ic in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh + kh - P
                iw = ow + kw - P
                in_bounds_h = (ih >= 0) & (ih < H_in)
                in_bounds_w = (iw >= 0) & (iw < W_in)

                if in_bounds_h & in_bounds_w:
                    inp_idx = ic * H_in * W_in + ih * W_in + iw
                    w_idx   = oc * (C_in * KH * KW) + ic * (KH * KW) + kh * KW + kw
                    inp_val = tl.load(input_ptr + inp_idx)
                    w_val   = tl.load(weight_ptr + w_idx)
                    acc += inp_val * w_val
    out_idx = oc * (H_out * W_out) + oh * W_out + ow
    tl.store(output_ptr + out_idx, acc)


def conv2d_triton(x, w, padding=1):
    C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = w.shape

    H_out = H_in - KH + 1 + 2 * padding
    W_out = W_in - KW + 1 + 2 * padding

    y = torch.empty((C_out, H_out, W_out), device='cuda', dtype=x.dtype)

    grid = (C_out, H_out, W_out)
    conv2d_kernel[grid](
        x, w, y,
        C_in, H_in, W_in,
        C_out, KH, KW, H_out, W_out, padding
    )
    return y
