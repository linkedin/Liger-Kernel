from test.chunked_loss.test_cpo_loss import HFCPOLoss
from test.utils import assert_verbose_allclose, set_seed

import pytest
import torch

from liger_kernel.chunked_loss.simpo_loss import LigerFusedLinearSimPOFunction
from liger_kernel.utils import infer_device


device = infer_device()

# set random seed globally
set_seed()


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-3, 5e-3),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "ignore_index, beta, gamma", [(-100, 0.1, 0.5), (42, 0.2, 0.85)]
)
def test_correctness(
    B, T, H, V, scalar, dtype, atol, rtol, bias, ignore_index, beta, gamma
):
    B = 2 * B  # SimPO loss requires B to be even

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device=device,
        dtype=torch.long,
    )
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    loss1 = HFCPOLoss(
        ignore_index=ignore_index, beta=beta, simpo_gamma=gamma, loss_type="simpo"
    ).get_batch_loss_metrics(input1, weight1, target, bias1)
    loss2 = LigerFusedLinearSimPOFunction.apply(
        input2, weight2, target, bias2, ignore_index, beta, 1.0, True, True, gamma
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
