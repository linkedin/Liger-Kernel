import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from liger_kernel.transformers.functional import liger_mhc_coeffs
from liger_kernel.transformers.functional import liger_mhc_post_res
from liger_kernel.transformers.functional import liger_mhc_pre
from liger_kernel.utils import infer_device
from test.transformers.test_mhc import mhc_coeffs_ref
from test.utils import set_seed
from test.utils import supports_bfloat16


class MiniMHCLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hc: int,
        c: int,
        tmax: int,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        use_fast: bool,
        device: str,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hc = hc
        self.c = c
        self.tmax = tmax
        self.rms_eps = rms_eps
        self.pre_eps = pre_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult = post_mult
        self.use_fast = use_fast
        self.act_dtype = torch.bfloat16

        self.embed = nn.Embedding(vocab_size, hc * c, device=device)
        self.inner = nn.Linear(c, c, bias=False, device=device)
        self.head = nn.Linear(hc * c, vocab_size, bias=False, device=device)

        m = hc * hc + 2 * hc
        k = hc * c
        self.phi = nn.Parameter(torch.randn(k, m, device=device, dtype=self.act_dtype) * 0.02)
        self.b = nn.Parameter(torch.zeros(m, device=device, dtype=torch.float32))
        self.alpha_pre = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.tensor(1.0, device=device, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids).to(self.act_dtype)
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.hc, self.c)

        if self.use_fast:
            h_pre, h_post, h_res = liger_mhc_coeffs(
                x,
                self.phi,
                self.b,
                self.alpha_pre,
                self.alpha_post,
                self.alpha_res,
                tmax=self.tmax,
                rms_eps=self.rms_eps,
                pre_eps=self.pre_eps,
                sinkhorn_eps=self.sinkhorn_eps,
                post_mult=self.post_mult,
            )
            x_in = liger_mhc_pre(x, h_pre)
            f_out = self.inner(x_in.float())
            x_out = liger_mhc_post_res(x, f_out, h_post, h_res)
        else:
            h_pre, h_post, h_res = mhc_coeffs_ref(
                x,
                self.phi,
                self.b,
                self.alpha_pre,
                self.alpha_post,
                self.alpha_res,
                tmax=self.tmax,
                rms_eps=self.rms_eps,
                pre_eps=self.pre_eps,
                sinkhorn_eps=self.sinkhorn_eps,
                post_mult=self.post_mult,
            )
            x_in = (x.float() * h_pre.unsqueeze(-1)).sum(dim=-2)
            f_out = self.inner(x_in)
            x_out = torch.einsum("...oi,...ic->...oc", h_res, x.float()) + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)

        x_merge = x_out.float().view(bsz, seq_len, self.hc * self.c)
        return self.head(x_merge)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_mhc_mini_lm_convergence():
    set_seed(0)

    device = infer_device()
    vocab_size = 32
    hc = 2
    c = 16
    tmax = 4
    steps = 6

    model_fast = MiniMHCLM(
        vocab_size=vocab_size,
        hc=hc,
        c=c,
        tmax=tmax,
        rms_eps=1e-6,
        pre_eps=1e-4,
        sinkhorn_eps=1e-6,
        post_mult=2.0,
        use_fast=True,
        device=device,
    )
    model_ref = MiniMHCLM(
        vocab_size=vocab_size,
        hc=hc,
        c=c,
        tmax=tmax,
        rms_eps=1e-6,
        pre_eps=1e-4,
        sinkhorn_eps=1e-6,
        post_mult=2.0,
        use_fast=False,
        device=device,
    )
    model_ref.load_state_dict(model_fast.state_dict())

    input_ids = torch.randint(0, vocab_size, (2, 8), device=device)
    labels = torch.randint(0, vocab_size, (2, 8), device=device)

    opt_fast = torch.optim.SGD(model_fast.parameters(), lr=0.1)
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)

    # Align tolerance with other bf16 convergence tests.
    loss_atol = 5e-3
    loss_rtol = 2e-2

    losses_fast = []
    losses_ref = []
    for _ in range(steps):
        logits_fast = model_fast(input_ids)
        logits_ref = model_ref(input_ids)

        loss_fast = F.cross_entropy(logits_fast.view(-1, vocab_size), labels.view(-1))
        loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), labels.view(-1))

        losses_fast.append(loss_fast.item())
        losses_ref.append(loss_ref.item())

        assert torch.allclose(loss_fast, loss_ref, rtol=loss_rtol, atol=loss_atol)

        loss_fast.backward()
        loss_ref.backward()

        opt_fast.step()
        opt_ref.step()

        opt_fast.zero_grad(set_to_none=True)
        opt_ref.zero_grad(set_to_none=True)

    assert losses_fast[-1] < losses_fast[0]
    assert losses_ref[-1] < losses_ref[0]
