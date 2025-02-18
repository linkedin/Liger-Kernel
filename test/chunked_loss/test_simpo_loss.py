import pytest
import torch

from liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss
from liger_kernel.chunked_loss.functional import liger_fused_linear_simpo
from liger_kernel.chunked_loss.simpo_loss import LigerFusedLinearSimPOFunction
from liger_kernel.utils import infer_device
from test.chunked_loss.test_cpo_loss import TorchLMHeadCPO
from test.utils import assert_verbose_allclose
from test.utils import set_seed

device = infer_device()

# set random seed globally
set_seed()


class LigerLMHeadSimPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.simpo_loss = LigerFusedLinearSimPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    def forward(self, x, y):
        return self.simpo_loss(self.lin.weight, x, y, self.lin.bias)


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
@pytest.mark.parametrize("ignore_index, beta, gamma", [(-100, 0.1, 0.5), (42, 0.2, 0.85)])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    ignore_index,
    beta,
    gamma,
    label_smoothing,
):
    B = 2 * B  # SimPO loss requires B to be even

    torch_lm_head_simpo = TorchLMHeadCPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        loss_type="simpo",
        label_smoothing=label_smoothing,
        simpo_gamma=gamma,
    )
    liger_lm_head_simpo = LigerLMHeadSimPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ignore_index=ignore_index,
        beta=beta,
        label_smoothing=label_smoothing,
        gamma=gamma,
    )

    torch_lm_head_simpo.lin.weight.data = liger_lm_head_simpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_simpo.lin.bias.data = liger_lm_head_simpo.lin.bias.data = torch.randn(
            V, device=device, dtype=dtype
        )

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

    loss1, aggregated_aux_outputs1 = torch_lm_head_simpo(input1, target)
    loss2, aggregated_aux_outputs2 = liger_lm_head_simpo(input2, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        assert_verbose_allclose(
            aggregated_aux_outputs1[i],
            aggregated_aux_outputs2[i],
            atol=atol,
            rtol=rtol,
        )

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_simpo.lin.weight.grad,
        liger_lm_head_simpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_simpo.lin.bias.grad,
            liger_lm_head_simpo.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias):
    B = 2 * B

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

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    loss1, aggregated_aux_outputs1 = LigerFusedLinearSimPOFunction.apply(input1, weight1, target, bias1)
    loss2, aggregated_aux_outputs2 = liger_fused_linear_simpo(input2, weight2, target, bias2)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
