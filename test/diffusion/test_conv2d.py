import torch
from liger_kernel.ops.conv2d import conv2d_triton

x = torch.randn(3, 5, 5, device='cuda', dtype=torch.float32)
w = torch.randn(2, 3, 3, 3, device='cuda', dtype=torch.float32)

y = conv2d_triton(x, w, padding=1)
ref = torch.nn.functional.conv2d(x.unsqueeze(0), w, padding=1).squeeze(0)

print(torch.allclose(y, ref, atol=1e-4))