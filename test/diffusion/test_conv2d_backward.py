# test_conv2d_backward.py
import torch

from liger_kernel.ops.conv2d_backward import conv2d_backward_weight_triton

def run_case(N, C_in, C_out, H, W, KH, KW, padding, stride):
    print(f"\n=== Test Case ===")
    print(f"N={N}, C_in={C_in}, C_out={C_out}, H={H}, W={W}, "
          f"KH={KH}, KW={KW}, padding={padding}, stride={stride}")

    x = torch.randn(N, C_in, H, W, device="cuda", requires_grad=False)
    w = torch.randn(C_out, C_in, KH, KW, device="cuda", requires_grad=True)

    # Forward + autograd reference
    y = torch.nn.functional.conv2d(x, w, stride=stride, padding=padding)
    loss = y.sum()
    loss.backward()
    dw_ref = w.grad.detach().clone()

    # Triton backward
    dout = torch.ones_like(y)  # dL/dy = 1
    dw_triton = conv2d_backward_weight_triton(
        x, dout, kernel_size=(KH, KW), padding=padding, stride=stride
    )

    max_diff = (dw_triton - dw_ref).abs().max().item()
    print("dw_triton.shape =", tuple(dw_triton.shape))
    print("dw_ref.shape    =", tuple(dw_ref.shape))
    print("max abs diff    =", max_diff)
    print("allclose?       =", torch.allclose(dw_triton, dw_ref, atol=1e-4))


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Small sanity checks
    run_case(N=1, C_in=1, C_out=1, H=3, W=3, KH=3, KW=3, padding=0, stride=1)
    run_case(N=1, C_in=1, C_out=1, H=5, W=5, KH=3, KW=3, padding=1, stride=1)

    # 2. Multi-channel inputs
    run_case(N=1, C_in=2, C_out=1, H=5, W=5, KH=3, KW=3, padding=1, stride=1)
    run_case(N=1, C_in=3, C_out=2, H=5, W=5, KH=3, KW=3, padding=0, stride=1)

    # 3. Larger batch
    run_case(N=4, C_in=3, C_out=2, H=7, W=7, KH=3, KW=3, padding=1, stride=1)

    # 4. Strided convolutions
    run_case(N=2, C_in=2, C_out=2, H=8, W=8, KH=3, KW=3, padding=1, stride=2)
    run_case(N=2, C_in=2, C_out=2, H=9, W=9, KH=5, KW=5, padding=2, stride=2)

    # 5. Non-square kernels
    run_case(N=1, C_in=1, C_out=1, H=6, W=6, KH=3, KW=5, padding=0, stride=1)
    run_case(N=1, C_in=2, C_out=2, H=6, W=6, KH=5, KW=3, padding=1, stride=1)
