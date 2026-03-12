from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


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
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop - each kernel processes multiple rows
    for row_idx in range(pid, n_rows, num_progs):
        X_row_ptr = X_ptr + row_idx * X_stride
        Y_row_ptr = Y_ptr + row_idx * Y_stride
        loss_row_ptr = loss_ptr + row_idx * loss_stride
        dX_row_ptr = dX_ptr + row_idx * dX_stride

        should_skip = False
        if HAS_LABEL:
            label = tl.load(label_ptr + row_idx)
            should_skip = label == ignore_index

        if should_skip:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                tl.store(dX_row_ptr + offsets, 0.0, mask=mask)
                tl.store(loss_row_ptr + offsets, 0.0, mask=mask)
        else:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                X = tl.load(X_row_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
                Y = tl.load(Y_row_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

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
                    max_val = tl.maximum(tl.max(X, axis=0), tl.max(Y, axis=0))
                    X_shifted = X - max_val
                    Y_shifted = Y - max_val

                    # Pre-compute exp(max_val) since it's used twice
                    exp_max = tl.exp(max_val)

                    # Compute exp terms with compensation
                    Q = tl.exp(X_shifted) * exp_max  # = exp(X)
                    P = tl.exp(Y_shifted) * exp_max  # = exp(Y)

                    # Pre-compute common terms
                    beta_P = beta * P
                    one_minus_beta_Q = (1 - beta) * Q
                    M = beta_P + one_minus_beta_Q
                    log_M = tl.log(M)

                    loss = beta_P * Y + one_minus_beta_Q * X - M * log_M
                    dX = one_minus_beta_Q * (X - log_M)

                # Pre-compute scaling factor
                scale = 1.0 / n_non_ignore
                loss = loss * scale
                dX = dX * scale

                tl.store(loss_row_ptr + offsets, loss, mask=mask)
                tl.store(dX_row_ptr + offsets, dX, mask=mask)


def get_optimal_block_size(total_elements):
    """
    Calculate optimal Block Size using compute_default_tiling_strategy
    """
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=8.0, shapes=((total_elements,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return 2048


def jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = get_optimal_block_size(V)

    # non reduction loss
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    # Use NPU core count for grid size
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, n_rows)

    _jsd_kernel[(grid_size,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-2),
        loss_ptr=loss,
        loss_stride=loss.stride(-2),
        dX_ptr=dX,
        dX_stride=dX.stride(-2),
        label_ptr=(shift_labels if has_label else torch.empty(1, device=_input.device)),
        beta=beta,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        n_rows=n_rows,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
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
        shift_labels: Optional[torch.Tensor] = None,
        beta: float = 0.5,
        ignore_index: int = -100,
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

        loss, dX = jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)
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
        )
