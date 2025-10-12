import triton
import triton.language as tl
import torch

# 2D convolution kernel (NCHW) with stride support
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr, P: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
):
    oc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    acc = 0.0
    for ic in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride_h + kh - P
                iw = ow * stride_w + kw - P
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

def conv2d_triton(x, w, bias=None, padding=1, stride=1):
    C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = w.shape
    
    # Handle stride as tuple or int
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # Calculate output dimensions with stride
    H_out = (H_in + 2 * padding - KH) // stride_h + 1
    W_out = (W_in + 2 * padding - KW) // stride_w + 1

    y = torch.empty((C_out, H_out, W_out), device='cuda', dtype=x.dtype)

    grid = (C_out, H_out, W_out)
    conv2d_kernel[grid](
        x, w, y,
        C_in, H_in, W_in,
        C_out, KH, KW, H_out, W_out, padding,
        stride_h, stride_w
    )
    
    # Add bias if provided
    if bias is not None:
        y = y + bias.view(C_out, 1, 1)
    
    return y
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
    k  = tl.program_id(2)
    kh = k // KW
    kw = k % KW

    # Initialize accumulator with correct dtype by loading a dummy value
    # Load from a valid position to get the dtype
    dummy_inp = tl.load(input_ptr + 0)
    acc = dummy_inp * 0.0

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


# Kernel: compute dX[n, ic, ih, iw] - gradient with respect to input
@triton.jit
def conv2d_backward_input_kernel(
    dout_ptr,       # float*  (N, C_out, H_out, W_out)
    weight_ptr,     # float*  (C_out, C_in, KH, KW)
    dx_ptr,         # float*  (N, C_in, H_in, W_in)
    N: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, P: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
):
    """
    Grid: (N, C_in, H_in*W_in)
    Each program computes a single (n, ic, ih, iw) element by reducing over C_out, KH, KW.
    """
    n = tl.program_id(0)
    ic = tl.program_id(1)
    pos = tl.program_id(2)
    ih = pos // W_in
    iw = pos % W_in
    
    # Initialize accumulator with correct dtype
    dummy_dout = tl.load(dout_ptr + 0)
    acc = dummy_dout * 0.0
    
    # For each output channel and kernel position
    for oc in range(C_out):
        for kh in range(KH):
            for kw in range(KW):
                # Compute which output position(s) this input position contributes to
                # Forward: out[oh, ow] = sum over in[oh*stride+kh-P, ow*stride+kw-P]
                # Backward: in[ih, iw] receives gradients from out[oh, ow] where ih = oh*stride+kh-P
                # So: oh = (ih + P - kh) / stride (must be integer and in bounds)
                
                oh_num = ih + P - kh
                ow_num = iw + P - kw
                
                # Check if this maps to a valid output position
                if oh_num % stride_h == 0 and ow_num % stride_w == 0:
                    oh = oh_num // stride_h
                    ow = ow_num // stride_w
                    
                    if (oh >= 0) & (oh < H_out) & (ow >= 0) & (ow < W_out):
                        dout_idx = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow
                        w_idx = oc * (C_in * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        
                        dout_val = tl.load(dout_ptr + dout_idx)
                        w_val = tl.load(weight_ptr + w_idx)
                        
                        acc += dout_val * w_val
    
    # Store the gradient
    dx_idx = n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw
    tl.store(dx_ptr + dx_idx, acc)


def conv2d_backward_input_triton(dout, w, input_shape, padding=1, stride=(1,1)):
    assert dout.is_cuda and w.is_cuda
    N, C_out, H_out, W_out = dout.shape
    C_out2, C_in, KH, KW = w.shape
    assert C_out == C_out2
    
    N2, C_in2, H_in, W_in = input_shape
    assert N == N2 and C_in == C_in2
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    P = padding
    
    dx = torch.empty((N, C_in, H_in, W_in), device=dout.device, dtype=dout.dtype)
    
    grid = (N, C_in, H_in * W_in)
    conv2d_backward_input_kernel[grid](
        dout, w, dx,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW, P,
        stride_h, stride_w
    )
    return dx


def conv2d_backward_bias_triton(dout):
    """
    Compute bias gradient by summing over batch and spatial dimensions.
    dout: (N, C_out, H_out, W_out)
    Returns: (C_out,)
    """
    # Bias gradient is just the sum over N, H, W dimensions
    return dout.sum(dim=(0, 2, 3))


class TritonConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias=None, padding=1, stride=1):
        ctx.save_for_backward(x, w, bias)
        ctx.padding = padding
        ctx.stride = stride
        
        # Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
        if x.dim() == 3:
            # Single batch case
            return conv2d_triton(x, w, bias=bias, padding=padding, stride=stride)
        elif x.dim() == 4:
            # Batched case - process each sample independently
            N = x.shape[0]
            outputs = []
            for i in range(N):
                out = conv2d_triton(x[i], w, bias=bias, padding=padding, stride=stride)
                outputs.append(out)
            return torch.stack(outputs, dim=0)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, bias = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        
        # Store original dimensions
        grad_output_was_3d = grad_output.dim() == 3
        x_was_3d = x.dim() == 3
        
        # Ensure grad_output and x are 4D for backward kernel
        if grad_output_was_3d:
            grad_output = grad_output.unsqueeze(0)
        if x_was_3d:
            x = x.unsqueeze(0)
        
        # Ensure tensors are contiguous (Triton requires contiguous memory)
        x = x.contiguous()
        grad_output = grad_output.contiguous()
        
        # Compute weight gradient
        KH, KW = w.shape[2], w.shape[3]
        grad_weight = conv2d_backward_weight_triton(
            x, grad_output, 
            kernel_size=(KH, KW), 
            padding=padding, 
            stride=stride
        )
        
        # Compute input gradient
        grad_input = conv2d_backward_input_triton(
            grad_output, w,
            input_shape=x.shape,
            padding=padding,
            stride=stride
        )
        
        # Compute bias gradient if bias was provided
        grad_bias = None
        if bias is not None:
            grad_bias = conv2d_backward_bias_triton(grad_output)
        
        # Remove batch dimension if input was 3D
        if x_was_3d:
            grad_input = grad_input.squeeze(0)
        
        return grad_input, grad_weight, grad_bias, None, None
    