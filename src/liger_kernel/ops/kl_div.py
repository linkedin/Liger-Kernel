import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


MAX_FUSED_SIZE = 65536 // 4  # 65536 // 4 or 8 works the best


@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B], output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = y_true * (tl.log(y_true) - y)
        else:
            res = y_true * (y_true - y)

        loss = tl.sum(res, axis=0)

        tl.store(loss_ptr, loss)
        loss_ptr += loss_stride


@triton.jit
def _kldiv_kernel_backward(
    input_ptr,
    input_stride,
    target_ptr,
    target_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    input_ptr += pid * input_stride
    target_ptr += pid * target_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -target / input

        tl.store(input_ptr + offsets, res, mask=mask)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction):  # [B, S]  # [B, S]
    B, S = y_pred.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    output_tensor = torch.zeros((B,), device=y_pred.device, dtype=torch.float32)

    grid = (B,)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    if reduction == "batchmean":
        return output_tensor.sum() / B
    elif reduction == "sum":
        return output_tensor.sum()
    elif reduction == "mean":
        return output_tensor.mean()
    else:
        return output_tensor


def kldiv_backward_triton(input, target, grad_output):
    B, S = input.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (B,)

    _kldiv_kernel_backward[grid](
        input,
        input.stride(0),
        target,
        target.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return input

    return input * grad_output


class LigerKLDivLossFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, y_pred, y_true, reduction="batchmean", log_target=False):
        ctx.save_for_backward(y_pred, y_true)
        return kldiv_forward_triton(
            y_pred, y_true, log_target=log_target, reduction=reduction
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors

        return (
            kldiv_backward_triton(y_pred, y_true, grad_output) / y_pred.shape[0],
            None,
            None,
            None,
        )
