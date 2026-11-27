"""
Muon optimizer for Liger-Kernel.

Muon (Momentum Orthogonalized by Newton-schulz) replaces standard AdamW for
2D parameter matrices (linear projection weights) with accelerated polar decomposition,
while retaining AdamW for 1D parameters (biases, normalization weights, embeddings).
"""

import math

import torch

from torch.optim.optimizer import Optimizer

from liger_kernel.ops.newton_schulz import liger_newton_schulz


class LigerMuon(Optimizer):
    """
    Muon optimizer accelerated by Liger-Kernel.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate for 2D weight matrices (default: 0.02).
        momentum: Momentum coefficient for Muon (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        adamw_lr: Learning rate for 1D parameters falling back to AdamW (default: 3e-4).
        adamw_betas: Beta coefficients for AdamW fallback (default: (0.9, 0.95)).
        adamw_eps: Epsilon for AdamW fallback (default: 1e-8).
        adamw_wd: Weight decay for AdamW fallback (default: 0.01).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_wd = group["adamw_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    if p.ndim != 2:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                # 2D weights: Muon Newton-Schulz update
                if p.ndim == 2:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        g = grad + momentum * buf
                    else:
                        g = buf

                    update = liger_newton_schulz(g, steps=ns_steps)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update, alpha=-lr * scale)

                # 1D weights: AdamW fallback
                else:
                    exp_avg = state["momentum_buffer"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]

                    if adamw_wd != 0:
                        p.mul_(1.0 - adamw_lr * adamw_wd)

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = adamw_lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(adamw_eps)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
