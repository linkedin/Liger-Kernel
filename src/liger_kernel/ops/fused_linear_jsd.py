import os

from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2
DEFAULT_CHUNK_MEMORY_MB = 2048
DEFAULT_MIN_CHUNK_SIZE = 512
CHUNK_SIZE_ENV = "LIGER_FUSED_LINEAR_JSD_CHUNK_SIZE"
CHUNK_MEMORY_MB_ENV = "LIGER_FUSED_LINEAR_JSD_CHUNK_MEMORY_MB"
MIN_CHUNK_SIZE_ENV = "LIGER_FUSED_LINEAR_JSD_MIN_CHUNK_SIZE"


def _get_positive_int_env(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        value = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Got: {value}") from exc
    if value <= 0:
        return default
    return value


def _previous_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (int(n).bit_length() - 1)


def _calculate_adaptive_chunk_size(BT, H, V):
    fixed_chunk_size = _get_positive_int_env(CHUNK_SIZE_ENV)
    if fixed_chunk_size is not None:
        return min(BT, fixed_chunk_size)

    inc_factor = triton.cdiv(V, H)
    memory_efficient_chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))

    min_chunk_size = _get_positive_int_env(MIN_CHUNK_SIZE_ENV, DEFAULT_MIN_CHUNK_SIZE)
    chunk_size = max(memory_efficient_chunk_size, min_chunk_size)

    chunk_memory_mb = _get_positive_int_env(CHUNK_MEMORY_MB_ENV, DEFAULT_CHUNK_MEMORY_MB)
    if chunk_memory_mb is not None:
        # The memory budget is a hard cap. It may reduce the final chunk size
        # below MIN_CHUNK_SIZE_ENV to avoid exceeding the temporary-buffer budget.
        bytes_per_token = 4 * V * torch.float32.itemsize
        max_chunk_size = max(1, (chunk_memory_mb * 2**20) // bytes_per_token)
        max_chunk_size = _previous_power_of_2(max_chunk_size)
        chunk_size = min(chunk_size, max_chunk_size)

    return max(1, min(BT, chunk_size))


@triton.jit
def _jsd_lm_head_kernel(
    X_ptr,  # student log-probabilities, X = log Q
    X_stride,
    Y_ptr,  # teacher log-probabilities, Y = log P
    Y_stride,
    loss_ptr,
    dlogits_ptr,
    dlogits_stride,
    label_ptr,
    student_prob_ptr,  # student probabilities (softmax of logits), pre-computed for numerical stability
    student_prob_stride,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    temperature: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    dlogits_ptr += pid * dlogits_stride
    student_prob_ptr += pid * student_prob_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dlogits_ptr + offsets, 0.0, mask=offsets < n_cols)
            tl.store(loss_ptr + pid, 0.0)
            return

    loss_acc = 0.0
    dX_sum = 0.0
    scale = 1.0 / n_non_ignore

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Q = tl.load(student_prob_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if beta == 0.0:
            Y_max = tl.max(Y, axis=0)
            P = tl.exp(Y - Y_max) * tl.exp(Y_max)
            loss = P * (Y - X)
            dX = -P
        elif beta == 1.0:
            loss = Q * (X - Y)
            dX = loss + Q
        else:
            max_val = tl.maximum(tl.max(X, axis=0), tl.max(Y, axis=0))
            exp_max = tl.exp(max_val)
            P = tl.exp(Y - max_val) * exp_max
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q
            M = beta_P + one_minus_beta_Q
            log_M = tl.log(M)
            loss = beta_P * Y + one_minus_beta_Q * X - M * log_M
            dX = one_minus_beta_Q * (X - log_M)

        loss_acc += tl.sum(tl.where(mask, loss, 0.0), axis=0)
        dX_sum += tl.sum(tl.where(mask, dX, 0.0), axis=0)
        tl.store(dlogits_ptr + offsets, dX, mask=mask)

    loss_acc = loss_acc * scale
    dX_sum = dX_sum * scale
    tl.store(loss_ptr + pid, loss_acc)

    inv_temp = 1.0 / temperature
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        Q = tl.load(student_prob_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dX = tl.load(dlogits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dlogits = (dX * scale - Q * dX_sum) * inv_temp
        tl.store(dlogits_ptr + offsets, dlogits, mask=mask)


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
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    chunk_size = _calculate_adaptive_chunk_size(BT, H, V)
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    grad_input = torch.zeros_like(student_input)
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
        if n_non_ignore == 0:
            return loss_1d.sum(), grad_input, grad_weight
    else:
        n_non_ignore = BT

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32)
        chunk_n_rows = student_logits_chunk.shape[0]

        student_logprob_chunk = torch.log_softmax(student_logits_chunk / temperature, dim=-1)
        teacher_logprob_chunk = torch.log_softmax(teacher_logits_chunk / temperature, dim=-1)
        student_prob_chunk = torch.exp(student_logprob_chunk).to(student_logits_chunk.dtype).contiguous()

        _jsd_lm_head_kernel[(chunk_n_rows,)](
            X_ptr=student_logprob_chunk,
            X_stride=student_logprob_chunk.stride(-2),
            Y_ptr=teacher_logprob_chunk,
            Y_stride=teacher_logprob_chunk.stride(-2),
            loss_ptr=loss_1d[start_idx:end_idx],
            dlogits_ptr=student_logprob_chunk,
            dlogits_stride=student_logprob_chunk.stride(-2),
            label_ptr=shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device),
            student_prob_ptr=student_prob_chunk,
            student_prob_stride=student_prob_chunk.stride(-2),
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            n_cols=V,
            temperature=temperature,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
            num_warps=16,
        )

        grad_logits_chunk = student_logprob_chunk.to(dtype)
        grad_input[start_idx:end_idx] = grad_logits_chunk @ student_weight

        if grad_weight is not None:
            grad_weight.add_(grad_logits_chunk.t() @ student_input_chunk)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):
    # If JSD is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

    return grad_input, grad_weight


class LigerFusedLinearJSDFunction(torch.autograd.Function):
    """
    Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor. Since JSD is the last layer, we can
    compute the gradient at the forward pass.
    """

    @staticmethod
    @amp_custom_fwd
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
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
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
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(grad_output, grad_input, grad_weight)
        return (grad_input, grad_weight, None, None, None, None, None, None)
