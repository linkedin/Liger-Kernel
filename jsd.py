from test.utils import assert_verbose_allclose, supports_bfloat16

import torch
from torch.nn import KLDivLoss


class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = 0.5 * (torch.exp(p) + torch.exp(q))
        return 0.5 * (self.kl(m.log(), p) + self.kl(m.log(), q))


torch.manual_seed(0)
torch_jsd = JSD()
device = "cuda"

(B, T, H, V) = (1, 2, 3, 6)

input = torch.randn(B * T, H, device=device, dtype=torch.float32, requires_grad=True)
# input = torch.tensor(
#     [[1, 2, 3, 4]], device=device, dtype=torch.float32, requires_grad=True
# )

x1 = input.detach().clone().requires_grad_(True)
x2 = input.detach().clone()
W = torch.nn.Linear(H, V, device=device, bias=False)
print(f"{x1=}")
print(f"{x2=}")
print(f"{W.weight=}")
Wx1 = W(x1)
Wx1.retain_grad()
Wx2 = x2 @ W.weight.t()
print(f"{Wx1=}")
print(f"{Wx2=}")
y1 = torch.log_softmax(Wx1, dim=-1)
print(f"{torch.log_softmax(Wx2, dim=-1)=}")
print(f"y1 = torch.log_softmax(Wx1, dim=-1).sum() = {y1}")
grad_output = torch.rand(y1.shape, device=device)
y1.backward(grad_output)
print(f"{Wx1.grad=}")
print(f"{torch.sum(grad_output, dim=-1, keepdim=True).broadcast_to(B*T, V)=}")
print(f"{grad_output=}")
Wx1_grad = grad_output - torch.softmax(Wx2, dim=-1) * torch.sum(
    grad_output, dim=-1, keepdim=True
).broadcast_to(B * T, V)
print(f"{Wx1_grad=}")
assert_verbose_allclose(Wx1.grad, Wx1_grad, rtol=1e-5, atol=1e-5)
print(f"{x1.grad=}")
x1_grad = Wx1_grad @ W.weight
print(f"{x1_grad=}")
assert_verbose_allclose(x1.grad, x1_grad, rtol=1e-5, atol=1e-5)
