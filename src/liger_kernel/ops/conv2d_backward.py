import torch
import triton
import triton.language as tl

# Kernel: compute dW[oc, ic, kh, kw] for NCHW single-batch allowed (handles N>1).
@triton.jit
def conv2d_backward_weight_kernel(
    input_ptr,      # float*  (N, C_in, H_in, W_in)
    dout_ptr,       # float*  (N, C_out, H_out, W_out)
    dw_ptr,         # float*  (C_out, C_in, KH, KW)
    N: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, P: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
):
    """
    Grid: (C_out, C_in, KH*KW)
    Each program computes a single (oc, ic, kh, kw) element by reducing over N, H_out, W_out.
    """
    oc = tl.program_id(0)
    ic = tl.program_id(1)
    k  = tl.program_id(2)           # flattened kernel index
    kh = k // KW
    kw = k % KW

    acc = 0.0

    # Loop over batch and output spatial positions
    # For small tests this is fine; a tiled version would be faster.
    for n in range(N):
        base_in_n = n * (C_in * H_in * W_in)
        base_dout_n = n * (C_out * H_out * W_out)
        for oh in range(H_out):
            # compute input h coordinate
            ih = oh * stride_h + kh - P
            # ih may be OOB; we still loop but mask
            for ow in range(W_out):
                iw = ow * stride_w + kw - P
                # mask in bounds
                mask = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
                if mask:
                    inp_idx = base_in_n + ic * (H_in * W_in) + ih * W_in + iw
                    dout_idx = base_dout_n + oc * (H_out * W_out) + oh * W_out + ow
                    inp = tl.load(input_ptr + inp_idx)
                    dout = tl.load(dout_ptr + dout_idx)
                    acc += inp * dout

    # store the reduced gradient
    dw_idx = oc * (C_in * KH * KW) + ic * (KH * KW) + kh * KW + kw
    tl.store(dw_ptr + dw_idx, acc)


def conv2d_backward_weight_triton(x, dout, kernel_size=(3,3), padding=1, stride=(1,1)):
    assert x.is_cuda and dout.is_cuda
    N, C_in, H_in, W_in = x.shape
    N2, C_out, H_out, W_out = dout.shape
    assert N == N2

    KH, KW = kernel_size
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    P = padding

    dw = torch.empty((C_out, C_in, KH, KW), device=x.device, dtype=x.dtype)

    grid = (C_out, C_in, KH * KW)
    conv2d_backward_weight_kernel[grid](
        x, dout, dw,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW, P,
        stride_h, stride_w
    )
    return dw