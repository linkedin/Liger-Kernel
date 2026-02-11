from typing import Literal
from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

MAX_FUSED_SIZE = 65536 // 4

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]


@triton.jit
def _tv_distance_kernel(
    p_ptr,
    p_stride,
    q_ptr,
    q_stride,
    loss_ptr,
    loss_stride,
    grads_ptr,
    grads_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,  # V
    total_rows: tl.constexpr,  # BT
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    reduction: tl.constexpr = "batchmean",
):
    thread_id = tl.program_id(0)
    num_threads = tl.num_programs(0)

    for pid in tl.range(thread_id, total_rows, num_threads, num_stages=NUM_STAGES):
        p_row_ptr = p_ptr + pid * p_stride
        q_row_ptr = q_ptr + pid * q_stride
        loss_row_ptr = loss_ptr + pid * loss_stride
        grads_row_ptr = grads_ptr + pid * grads_stride
        label_row_ptr = label_ptr + pid

        base_offsets = tl.arange(0, BLOCK_SIZE)

        should_skip = False
        if HAS_LABEL:
            label = tl.load(label_row_ptr)
            if label == ignore_index:
                should_skip = True

        if should_skip:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols
                tl.store(grads_row_ptr + offsets, 0.0, mask=mask)
                if reduction == "none":
                    tl.store(loss_row_ptr + offsets, 0.0, mask=mask)
        else:
            loss_sum = 0.0
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols

                p = tl.load(p_row_ptr + offsets, mask=mask, other=0.0)
                q = tl.load(q_row_ptr + offsets, mask=mask, other=0.0)

                # TVD(P || Q) = 0.5 * |P - Q|
                tv_loss = 0.5 * tl.abs(p - q)
                grad_res = tl.where(p > q, 0.5, -0.5)

                tl.store(grads_row_ptr + offsets, grad_res, mask=mask)

                if reduction == "none":
                    tl.store(loss_row_ptr + offsets, tv_loss, mask=mask)
                else:
                    loss_sum += tl.sum(tv_loss, axis=0)

            if reduction != "none":
                tl.store(loss_row_ptr, loss_sum)


def tv_distance_forward_triton(p, q, shift_labels, reduction, ignore_index, has_label):
    BT, V = p.shape

    # TVD forward tiling strategy
    # - In main loop (calculate loss and grad):
    #   * p: BLOCK_Q elements
    #   * q: BLOCK_Q elements
    #   * tv_loss: BLOCK_Q elements
    #   * grad_res: BLOCK_Q elements
    #   * loss_sum: BLOCK_Q elements (when reduction != "none")
    #   * Total: 4 * BLOCK_Q elements or 5 * BLOCK_Q elements when reduction != "none"
    # - Since loss_sum is not necessarily used in every calculation,
    # - and considering the consumption of other shared memory and the potential memory consumption of the HAS_LABEL loop.
    # - Conservative estimate: 5 * BLOCK_Q * dtype_size * 8 bits
    # - For safety, use: memory_multiplier=5.0 * BLOCK_SIZE * pad_hd * dtype_size * 8 bits
    # - shapes: ((V,),)
    # - tiling_dims: (0,) means first dimension of each shape can be tiled
    # - Returns: ((block_size,),
    shapes = ((V,),)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.80,
        # In the TVD calculation, many data are implicitly converted to f32, so the size of f32 can be directly used.
        dtype_size=4,
        memory_multiplier=5.0,
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes) > 0 and len(tile_shapes[0]) > 0:
        # Strategy returns ((block_size,),)
        BLOCK_SIZE = tile_shapes[0][0]
    else:
        # Fallback to desired block size if no best practice found (no tiling needed)
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    num_cores = get_npu_core_count()
    grid = (min(num_cores, BT),)

    out_size = (BT, V) if reduction == "none" else (BT,)

    # The loss and grid accumulation on BF16 platform of NPU will have precision errors.
    output_tensor = torch.zeros(out_size, device=p.device, dtype=torch.float32)
    grads = torch.empty_like(p, dtype=torch.float32)

    n_non_ignore = (shift_labels != ignore_index).sum().item() if has_label else BT

    _tv_distance_kernel[grid](
        p,
        p.stride(0),
        q,
        q.stride(0),
        output_tensor,
        output_tensor.stride(0),
        grads,
        grads.stride(0),
        shift_labels if has_label else torch.empty(1, device=p.device),
        ignore_index,
        V,
        BT,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_LABEL=has_label,
        NUM_STAGES=3 if BT < 4096 else 4,
        reduction=reduction,
    )

    if reduction == "batchmean":
        return output_tensor.sum() / n_non_ignore, grads / n_non_ignore
    elif reduction == "sum":
        return output_tensor.sum(dim=0), grads
    elif reduction == "mean":
        return output_tensor.sum() / (n_non_ignore * V), grads / (n_non_ignore * V)
    else:
        return output_tensor, grads


def tvd_backward_triton(grad_output, grads):
    # If this is the last layer, grad_output is 1.0. Skip the mul then.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grads

    return grads * grad_output


class LigerTVDLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the Total Variation Distance Loss using Triton.
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        p: torch.Tensor,
        q: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        reduction: REDUCTION_LITERAL = "batchmean",
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """A forward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            p (torch.Tensor): A tensor of shape (BT, V) containing the first distribution.
            q (torch.Tensor): A tensor of shape (BT, V) containing the second distribution.
            shift_labels (Optional[torch.Tensor]): A tensor of shape (BT,) containing the labels.
            reduction (REDUCTION_LITERAL, optional): The reduction method to be applied. Defaults to "batchmean".
            ignore_index (int, optional): The index to ignore during loss calculation. Defaults to -100.

        Returns:
            torch.Tensor: The computed Total Variation Distance Loss.
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (p.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grads = tv_distance_forward_triton(p, q, shift_labels, reduction, ignore_index, has_label)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """A backward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: The gradient of the loss with respect to the inputs.
        """
        (grads,) = ctx.saved_tensors
        grads = tvd_backward_triton(grad_output, grads)

        return grads, None, None, None, None
