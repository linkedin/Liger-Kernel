import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    beta,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        Q = tl.exp(X)
        P = tl.exp(Y)
        M = beta * P + (1 - beta) * Q
        log_M = tl.log(M)

        loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
        # reduction == "batchmean"
        loss = loss / n_rows
        tl.store(loss_ptr + offsets, loss, mask=mask)

        dX = (1 - beta) * Q * (X - log_M) / n_rows
        tl.store(dX_ptr + offsets, dX, mask=mask)


MAX_FUSED_SIZE = 65536


def jsd_forward(_input, target, beta):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    # non reduction loss
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)

    _jsd_kernel[(n_rows,)](
        X_ptr=_input,  # input in logspace, X = log Q
        X_stride=_input.stride(-2),
        Y_ptr=target,  # ground truth in logspace, Y = log P
        Y_stride=target.stride(-2),
        loss_ptr=loss,
        loss_stride=loss.stride(-2),
        dX_ptr=dX,
        dX_stride=dX.stride(-2),
        beta=beta,
        n_rows=n_rows,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    loss = torch.sum(loss)
    return loss.to(_input.dtype), dX


def jsd_backward(dX, grad_output):
    # If jsd is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dX
    else:
        return grad_output * dX


class LigerJSDFunction(torch.autograd.Function):
    r"""
    This class implements the forward and backward pass for the generalized Jensen-Shannon Divergence.
    .. math::
        JSD(\beta)(P || Q)
            = \beta * KLDiv(P || (\beta * P + (1 - \beta) * Q)) + (1 - \beta) * KLDiv(Q || (\beta * P + (1 - \beta) * Q))

    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`_input`, to be the predictions, the output of the student model, in log-space
        and the second, :attr:`target`, to be the observations, the output of the teacher model, in log-space.
        This differs from the standard mathematical notation :math:`JSD(P || Q)` where
        :math:`P` denotes the teacher model and :math:`Q` denotes the student model.
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            _input (torch.Tensor): predict values with shape (BT, V) in logspace
            target (torch.Tensor): ground truth values with shape (BT, V) in logspace
            beta (float): coefficient beta of generalized JSD in the open interval (0, 1)

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        loss, dX = jsd_forward(_input, target, beta)
        ctx.save_for_backward(dX)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (dX,) = ctx.saved_tensors
        dX = jsd_backward(dX, grad_output)
        return (
            dX,
            None,
            None,
        )
