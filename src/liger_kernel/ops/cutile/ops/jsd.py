# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.utils import ensure_contiguous

ConstFloat = ct.Constant[float]
ConstInt = ct.Constant[int]
JSD_BLOCK_SIZE = 4096


@ct.kernel(occupancy=ct.ByTarget(sm_100=4))
def jsd_kernel_ct(
    x,  # (BT, V) log Q (student)
    y,  # (BT, V) log P (teacher)
    loss,  # (BT, V) float32 loss accumulator
    dx,  # (BT, V) gradient output
    label,  # (BT,) label tensor, or dummy tensor when HAS_LABEL=0
    beta: ConstFloat,
    inv_n_non_ignore: ConstFloat,
    ignore_index: ConstInt,
    n_cols: ConstInt,
    BLOCK_SIZE: ConstInt,
    HAS_LABEL: ConstInt,
):
    """
    cuTile kernel for generalized Jensen-Shannon Divergence.
    """
    row_idx = ct.bid(0)

    if HAS_LABEL:
        lbl = ct.load(label, row_idx, shape=())
        if lbl == ignore_index:
            num_chunks_early = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            for ci in range(num_chunks_early):
                col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + ci * BLOCK_SIZE
                ct.scatter(dx, (row_idx, col_indices), ct.full((BLOCK_SIZE,), 0.0, dtype=dx.dtype), check_bounds=True)
            return

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(num_chunks):
        col_indices = ct.arange(BLOCK_SIZE, dtype=ct.int32) + chunk_idx * BLOCK_SIZE

        x_tile = ct.gather(x, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
        y_tile = ct.gather(y, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)

        x_f32 = ct.astype(x_tile, ct.float32)
        y_f32 = ct.astype(y_tile, ct.float32)

        loss_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
        dx_tile = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)

        if beta == 0.0:
            y_max = ct.max(y_f32, 0, keepdims=True)
            y_prob = ct.exp(y_f32 - y_max) * ct.exp(y_max)
            loss_tile = y_prob * (y_f32 - x_f32)
            dx_tile = -y_prob
        elif beta == 1.0:
            x_max = ct.max(x_f32, 0, keepdims=True)
            x_prob = ct.exp(x_f32 - x_max) * ct.exp(x_max)
            loss_tile = x_prob * (x_f32 - y_f32)
            dx_tile = loss_tile + x_prob
        else:
            x_max = ct.max(x_f32, 0, keepdims=True)
            y_max = ct.max(y_f32, 0, keepdims=True)
            max_val = ct.maximum(x_max, y_max)
            exp_max = ct.exp(max_val)
            q_prob = ct.exp(x_f32 - max_val) * exp_max
            p_prob = ct.exp(y_f32 - max_val) * exp_max
            beta_p = beta * p_prob
            one_minus_beta_q = (1.0 - beta) * q_prob
            m_prob = beta_p + one_minus_beta_q
            log_m = ct.log(m_prob)
            loss_tile = beta_p * y_f32 + one_minus_beta_q * x_f32 - m_prob * log_m
            dx_tile = one_minus_beta_q * (x_f32 - log_m)

        loss_tile = loss_tile * inv_n_non_ignore
        dx_tile = dx_tile * inv_n_non_ignore

        ct.scatter(loss, (row_idx, col_indices), loss_tile, check_bounds=True)
        ct.scatter(dx, (row_idx, col_indices), ct.astype(dx_tile, dx.dtype), check_bounds=True)


def jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    num_rows, vocab_size = _input.shape
    BLOCK_SIZE = min(JSD_BLOCK_SIZE, _next_power_of_2(vocab_size))

    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dx = torch.empty_like(_input)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = num_rows

    if n_non_ignore == 0:
        return torch.tensor(0.0, device=_input.device, dtype=_input.dtype), torch.zeros_like(_input)

    inv_n_non_ignore = 1.0 / n_non_ignore
    label_tensor = shift_labels if has_label else torch.empty(1, device=_input.device, dtype=torch.int64)

    ct.launch(
        torch.cuda.current_stream(),
        (num_rows, 1, 1),
        jsd_kernel_ct,
        (
            _input,
            target,
            loss,
            dx,
            label_tensor,
            float(beta),
            float(inv_n_non_ignore),
            int(ignore_index),
            int(vocab_size),
            int(BLOCK_SIZE),
            int(has_label),
        ),
    )

    return torch.sum(loss).to(_input.dtype), dx


def jsd_backward(dx, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dx
    return grad_output * dx


class LigerJSDFunction(torch.autograd.Function):
    r"""
    cuTile autograd wrapper for the generalized Jensen-Shannon Divergence loss.
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.Tensor],
        beta: float,
        ignore_index: int,
    ) -> torch.Tensor:
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],), (
                f"shift_labels must have shape (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, dx = jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label)
        ctx.save_for_backward(dx)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        (dx,) = ctx.saved_tensors
        dx = jsd_backward(dx, grad_output)
        return (dx, None, None, None, None)
