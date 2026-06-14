import os

from typing import Optional

import torch
import triton
import triton.language as tl

from liger_kernel.ops.jsd import _jsd_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2
DEFAULT_CHUNK_MEMORY_MB = 1024
DEFAULT_MIN_CHUNK_SIZE = 256
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
        # The fast path keeps multiple fp32 (chunk, V) intermediates alive.
        # Budget for roughly four such tensors: student/teacher logits and
        # student/teacher log-probs. Use a power-of-two cap to avoid odd GEMMs.
        bytes_per_token = 4 * V * torch.float32.itemsize
        max_chunk_size = max(1, (chunk_memory_mb * 2**20) // bytes_per_token)
        chunk_size = min(chunk_size, _previous_power_of_2(max_chunk_size))

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
    label_ptr += pid

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            tl.store(loss_ptr + pid, 0.0)
            tl.store(dlogits_ptr + offsets, 0.0, mask=mask)
            return

    X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
    Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

    if beta == 0.0:  # forward KL
        Y_max = tl.max(Y, axis=0)
        P = tl.exp(Y - Y_max) * tl.exp(Y_max)
        loss = P * (Y - X)
        dX = -P
    elif beta == 1.0:  # reverse KL
        X_max = tl.max(X, axis=0)
        Q = tl.exp(X - X_max) * tl.exp(X_max)
        loss = Q * (X - Y)
        dX = loss + Q
    else:
        max_val = tl.maximum(tl.max(X, axis=0), tl.max(Y, axis=0))
        exp_max = tl.exp(max_val)
        Q = tl.exp(X - max_val) * exp_max
        P = tl.exp(Y - max_val) * exp_max
        beta_P = beta * P
        one_minus_beta_Q = (1 - beta) * Q
        M = beta_P + one_minus_beta_Q
        log_M = tl.log(M)
        loss = beta_P * Y + one_minus_beta_Q * X - M * log_M
        dX = one_minus_beta_Q * (X - log_M)

    scale = 1.0 / n_non_ignore
    loss = loss * scale
    dX = dX * scale
    Q = tl.exp(X)

    sum_dX = tl.sum(tl.where(mask, dX, 0.0), axis=0)
    dlogits = (dX - Q * sum_dX) / temperature

    tl.store(loss_ptr + pid, tl.sum(tl.where(mask, loss, 0.0), axis=0))
    tl.store(dlogits_ptr + offsets, dlogits, mask=mask)


def fused_linear_jsd_forward_original(
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

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    grad_input = torch.zeros_like(student_input)
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk both inputs, shape: chunk_size x H
        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        # shape: chunk_size x V
        # For anything starting from logits to the final JSD loss, we do computation
        # in FP32 to avoid losing numerical stability.
        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32)
        chunk_n_rows = student_logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size
        # log-softmax with temperature
        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        student_prob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
        teacher_prob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)

        # ensure _input and target are contiguous
        student_prob_chunk = student_prob_chunk.contiguous()
        teacher_prob_chunk = teacher_prob_chunk.contiguous()

        # Here we calculate the gradient of prob_chunk in place so we can save memory.
        _jsd_kernel[(chunk_n_rows,)](
            X_ptr=student_prob_chunk,
            X_stride=student_prob_chunk.stride(-2),
            Y_ptr=teacher_prob_chunk,
            Y_stride=teacher_prob_chunk.stride(-2),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-2),
            dX_ptr=student_prob_chunk,
            dX_stride=student_prob_chunk.stride(-2),
            label_ptr=(
                shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device)
            ),  # dummy ptr if no label
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
        )
        loss_1d[start_idx:end_idx] = loss_1d_slice
        # gradients of prob_chunk in place, shape: chunk_size x V
        # gradients of logits_chunk in place, shape: chunk_size x V
        student_logits_chunk = (
            student_prob_chunk
            - torch.softmax(student_logits_chunk, dim=-1)
            * student_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(student_prob_chunk.shape)
        ) / temperature
        # now we traverse back to grad w.r.t. input to `lm_head` and grad
        # w.r.t. `lm_head` which should be computed in original dtype
        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


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

        if temperature == 1.0:
            student_logprob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
            teacher_logprob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)
        else:
            student_logprob_chunk = torch.log_softmax(student_logits_chunk / temperature, dim=-1)
            teacher_logprob_chunk = torch.log_softmax(teacher_logits_chunk / temperature, dim=-1)

        _jsd_lm_head_kernel[(chunk_n_rows,)](
            X_ptr=student_logprob_chunk,
            X_stride=student_logprob_chunk.stride(-2),
            Y_ptr=teacher_logprob_chunk,
            Y_stride=teacher_logprob_chunk.stride(-2),
            loss_ptr=loss_1d[start_idx:end_idx],
            dlogits_ptr=student_logprob_chunk,
            dlogits_stride=student_logprob_chunk.stride(-2),
            label_ptr=shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device),
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
