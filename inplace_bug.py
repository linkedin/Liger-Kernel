from test.utils import set_seed

import torch
import torch.nn.functional as F

from liger_kernel.transformers.functional import liger_cross_entropy

set_seed(42)


def run_inplace_experiment(logits_p, logits_q, cross_entropy_fn):
    _p = logits_p.clone().detach().requires_grad_(True)
    _p.retain_grad()
    softmax = torch.nn.Softmax(dim=-1)
    p = softmax(_p)
    p.retain_grad()
    loss = cross_entropy_fn(p, logits_q)
    loss.backward(retain_graph=True)

    print(f"Cross Entropy Loss: {loss.item()}")
    print(f"Input _p: {_p}")
    print(f"Input logits_q: {logits_q}")
    print(f"Gradients of p (batch item 0): {p.grad[0]}")
    print(f"Gradients of _p (batch item 0): {_p.grad[0]}")


torch.manual_seed(0)
logits_p = torch.randn(8, 8, requires_grad=True, device="cuda")
logits_q = torch.randint(0, 8, (8,), device="cuda", dtype=torch.long)


run_inplace_experiment(logits_p, logits_q, cross_entropy_fn=F.cross_entropy)

print()
print("LIGER:")
run_inplace_experiment(logits_p, logits_q, cross_entropy_fn=liger_cross_entropy)
