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
        M = 0.5 * P + 0.5 * Q
        log_M = tl.log(M)

        loss = 0.5 * (P * Y + Q * X - 2 * M * log_M)
        tl.store(loss_ptr + offsets, loss, mask=mask)

        dX = 0.5 * Q * (X - log_M) / n_rows
        tl.store(dX_ptr + offsets, dX, mask=mask)


MAX_FUSED_SIZE = 65536


def jsd_forward(_input, target):
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
        n_rows=n_rows,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # reduction == "batchmean"
    loss = torch.sum(loss) / n_rows
    return loss.to(_input.dtype), dX


def jsd_backward(dX, grad_output):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dX
    else:
        return grad_output * dX


class LigerJSDFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the JS Divergence using Triton, as defined by the following formula:

    Parameters:
    _input (tensor): predict values with shape (BT, V) in logspace
    target (tensor): gournd truth values with shape (BT, V) in logspace

    Returns:
    loss (tensor): JSD
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss, dX = jsd_forward(_input, target)
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
        )
