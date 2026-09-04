from typing import Optional

import torch
import triton
import triton.language as tl

# Trigger declaration of the ``jsd_loss_and_grad`` op location so the
# dispatcher knows where to discover the Triton + CuTe DSL impls.
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends import dispatch
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import infer_device


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
    label_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
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
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        if beta == 0.0:  # forward KL
            Y_max = tl.max(Y, axis=0)
            Y_shifted = Y - Y_max
            Y_prob = tl.exp(Y_shifted) * tl.exp(Y_max)  # Compensate for the shift
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:  # reverse KL
            X_max = tl.max(X, axis=0)
            X_shifted = X - X_max
            X_prob = tl.exp(X_shifted) * tl.exp(X_max)  # Compensate for the shift
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            max_val = tl.maximum(X, Y)
            X_shifted = X - max_val
            Y_shifted = Y - max_val

            # Compute log(M) before rescaling to probability space. Computing M
            # directly can underflow to zero for low-probability vocabulary blocks.
            M_shifted = beta * tl.exp(Y_shifted) + (1 - beta) * tl.exp(X_shifted)
            log_M = max_val + tl.log(M_shifted)

            Q = tl.exp(X)
            P = tl.exp(Y)
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q

            loss = beta_P * (Y - log_M) + one_minus_beta_Q * (X - log_M)
            dX = one_minus_beta_Q * (X - log_M)

        # Pre-compute scaling factor
        scale = 1.0 / n_non_ignore
        loss = loss * scale
        dX = dX * scale

        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)


MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536


def jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label, jsd_impl=None, jsd_mode=None):
    BT, V = _input.shape

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    # Compute per-element loss + dx through the dispatcher so CuTe DSL is
    # picked up on Hopper+ (preference_rank=10 < Triton's 50). The primitive
    # writes dx in-place into the first argument, so clone _input to preserve
    # the original tensor for the autograd context.
    dX = _input.clone()
    jsd_args = (
        dX,
        target,
        shift_labels,
        float(beta),
        int(ignore_index),
        float(n_non_ignore),
    )
    loss_tile, dX = dispatch(
        "jsd_loss_and_grad",
        *jsd_args,
        impl=jsd_impl,
        mode=jsd_mode,
    )

    loss = torch.sum(loss_tile)
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
        shift_labels: Optional[torch.Tensor] = None,
        beta: float = 0.5,
        ignore_index: int = -100,
        jsd_impl=None,
        jsd_mode=None,
    ) -> torch.Tensor:
        """
        Args:
            _input (torch.Tensor): predict values with shape (BT, V) in logspace
            target (torch.Tensor): ground truth values with shape (BT, V) in logspace
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, dX = jsd_forward(
            _input,
            target,
            shift_labels,
            beta,
            ignore_index,
            has_label,
            jsd_impl,
            jsd_mode,
        )
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
            None,
            None,
            None,
            None,
        )
