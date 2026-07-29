# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import cuda.tile as ct
import torch

from liger_kernel.ops.cutile.ops.jsd import JSD_BLOCK_SIZE
from liger_kernel.ops.cutile.ops.jsd import jsd_kernel_ct
from liger_kernel.ops.cutile.ops.utils import _next_power_of_2
from liger_kernel.ops.cutile.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_SIZE = 65536 // 2


def fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels,
    jsd_beta,
    ignore_index,
    has_label,
    temperature,
):
    device = student_input.device
    dtype = student_input.dtype

    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(JSD_BLOCK_SIZE, _next_power_of_2(V))

    inc_factor = (V + H - 1) // H
    chunk_size = _next_power_of_2((BT + inc_factor - 1) // inc_factor)
    num_chunks = (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    grad_input = torch.zeros_like(student_input)
    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    inv_n_non_ignore = 1.0 / max(n_non_ignore, 1)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32)
        chunk_n_rows = student_logits_chunk.shape[0]

        loss_chunk = loss_1d[start_idx:end_idx]
        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        log_prob_s_chunk = torch.log_softmax(student_logits_chunk, dim=-1).contiguous()
        log_prob_t_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1).contiguous()

        label_chunk = shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device, dtype=torch.int64)

        ct.launch(
            torch.cuda.current_stream(),
            (chunk_n_rows, 1, 1),
            jsd_kernel_ct,
            (
                log_prob_s_chunk,
                log_prob_t_chunk,
                loss_chunk,
                log_prob_s_chunk,
                label_chunk,
                float(jsd_beta),
                float(inv_n_non_ignore),
                int(ignore_index),
                int(V),
                int(BLOCK_SIZE),
                int(has_label),
            ),
        )

        # log_prob_s_chunk now holds dx (gradient w.r.t. log_softmax output)
        student_logits_chunk = (
            log_prob_s_chunk
            - torch.softmax(student_logits_chunk, dim=-1)
            * log_prob_s_chunk.sum(dim=-1, keepdim=True).broadcast_to(log_prob_s_chunk.shape)
        ) / temperature
        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):
    # If JSD is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, _next_power_of_2(H))

        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            element_mul_kernel,
            (grad_input, grad_output, int(H), int(BLOCK_SIZE), H % BLOCK_SIZE != 0),
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            ct.launch(
                torch.cuda.current_stream(),
                (n_rows, 1, 1),
                element_mul_kernel,
                (grad_weight, grad_output, int(H), int(BLOCK_SIZE), H % BLOCK_SIZE != 0),
            )

    return grad_input, grad_weight


class LigerFusedLinearJSDFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        jsd_beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (teacher_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            jsd_beta,
            ignore_index,
            has_label,
            temperature,
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(grad_output, grad_input, grad_weight)
        return (grad_input, grad_weight, None, None, None, None, None, None)
