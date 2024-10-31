import pytest
import torch

from liger_kernel.ops.group_norm import LigerGroupNormFunction
from liger_kernel.transformers.functional import liger_group_norm
from liger_kernel.transformers.group_norm import LigerGroupNorm


@pytest.mark.parametrize(
    "batch_size, num_channels, num_groups, hidden_size",
    [
        (1, 3, 1, 128),
        (2, 4, 2, 128),
        (16, 12, 3, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_group_norm(batch_size, num_channels, num_groups, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)

    x = torch.randn(
        batch_size, num_channels, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )
    liger_ln = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).cuda()
    torch_ln = torch.nn.GroupNorm(num_channels, num_groups, hidden_size, eps=1e-6).to(dtype).cuda()

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(x)
    torch_output = torch_ln(x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    # grad_output = torch.randn_like(x)
    # liger_output.backward(grad_output, retain_graph=True)
    # torch_output.backward(grad_output, retain_graph=True)

    # assert torch.allclose(x.grad, x.grad, atol=atol, rtol=rtol)
    # assert torch.allclose(
    #     liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol
    # )
    # assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)


# @pytest.mark.parametrize(
#     "hidden_size",
#     [8, 41],
# )
# @pytest.mark.parametrize(
#     "batch_size, seq_len",
#     [
#         (2, 2),
#         (9, 7),
#     ],
# )
# @pytest.mark.parametrize(
#     "dtype, atol, rtol",
#     [
#         (torch.float32, 1e-5, 1e-5),
#     ],
# )
# def test_liger_group_norm_functional(
#     hidden_size, batch_size, seq_len, dtype, atol, rtol
# ):
#     torch.manual_seed(0)

#     input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")

#     x1 = input.clone().requires_grad_(True)
#     x2 = input.clone().requires_grad_(True)

#     w = torch.randn(hidden_size, device="cuda", dtype=dtype)

#     w1 = w.clone().requires_grad_(True)
#     w2 = w.clone().requires_grad_(True)

#     b = torch.randn(hidden_size, device="cuda", dtype=dtype)

#     b1 = b.clone().requires_grad_(True)
#     b2 = b.clone().requires_grad_(True)

#     y1 = liger_group_norm(x1, w1, b1, 1e-6)
#     y2 = LigergroupNormFunction.apply(x2, w2, b2, 1e-6)

#     assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

#     grad_output = torch.randn_like(y2)

#     y1.backward(grad_output, retain_graph=True)
#     y2.backward(grad_output, retain_graph=True)

#     assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
#     assert torch.allclose(w1.grad, w2.grad, atol=atol, rtol=rtol)
#     assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)
