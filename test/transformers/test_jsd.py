from test.utils import assert_verbose_allclose

import pytest
import torch
from torch.nn import KLDivLoss

from liger_kernel.transformers.jsd import LigerJSD


class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, log_p: torch.tensor, log_q: torch.tensor):
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = 0.5 * (torch.exp(log_p) + torch.exp(log_q))
        log_m = torch.log(m + 1e-10)
        loss = 0.5 * (self.kl(log_m, log_p) + self.kl(log_m, log_q))
        print(f"torch's {loss=}")
        return loss


_SHAPE_PARAMS = (
    "B, T, V",
    [
        (1, 1, 4096),
        (1, 1, 32000),
        # (32, 4096, 1024),
        # # weird shape
        # (41, 401, 1271),
        # pytest.param(
        #     1,
        #     4096,
        #     128256,
        #     marks=pytest.mark.skipif(
        #         torch.cuda.get_device_properties(0).total_memory
        #         < 36 * 1000 * 1000 * 1000,
        #         reason="This test requires a GPU with at least 36GB of memory",
        #     ),
        # ),
        # (3, 423, 32000),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        # pytest.param(
        #     torch.bfloat16,
        #     1e-8,
        #     5e-2,
        #     marks=pytest.mark.skipif(
        #         not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
        #     ),
        # ),
        (torch.float32, 1e-8, 1e-6),
        # (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_correctness_once(
    target_jsd,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    is_last_layer=True,
    device="cuda",
):
    torch.manual_seed(0)
    torch_jsd = JSD()

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)
    x3 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).log_softmax(dim=-1)

    output = torch_jsd(x1, target)
    output2 = target_jsd(x2, target)
    print(f"liger's loss = {output2}")
    # output3 = target_jsd(target, x3)
    assert torch.allclose(output, output2, atol=atol, rtol=rtol)
    # assert_verbose_allclose(output3, output2, atol=atol, rtol=rtol)
    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness(B, T, V, dtype, atol, rtol):
    liger_jsd = LigerJSD()
    _test_correctness_once(liger_jsd, B, T, V, dtype, atol, rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness_not_last(B, T, V, dtype, atol, rtol):
    liger_jsd = LigerJSD()

    _test_correctness_once(liger_jsd, B, T, V, dtype, atol, rtol, is_last_layer=False)
