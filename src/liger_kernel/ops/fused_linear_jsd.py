from typing import Optional

import torch
import triton
import triton.language as tl

from packaging.version import Version

from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2
_TORCH_VERSION = Version(torch.__version__.split("+")[0])
_ADDMM_SUPPORTS_OUT_DTYPE = _TORCH_VERSION >= Version("2.8.0")

# log2(e): lets us emit the hardware ex2.approx instruction via e^x = 2^(x * LOG2_E)
LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fused_linear_jsd_kernel(
    X_ptr,  # student logits, X = log Q is derived in-kernel, dX is written back in place
    X_stride,
    Y_ptr,  # teacher logits, Y = log P is derived in-kernel
    Y_stride,
    loss_ptr,
    loss_stride,
    label_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    temperature,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    # Same math as _jsd_kernel, but the log-probabilities are derived from the logits inside the
    # kernel, so JSD(P || Q) is fused with the two log-softmaxes and the gradient chain rule:
    # X = log Q = z_X / temperature - lse_X, Y = log P = z_Y / temperature - lse_Y
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)
    # Chaining through the log-softmax Jacobian gives the gradient w.r.t. the student logits z_X:
    # dz_i = (dX_i - Q_i * sum_j dX_j) / temperature
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    # 1. Load label first because if the label is ignore_index, we can return right away
    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(X_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    inv_temperature = 1.0 / temperature

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 2. [Online softmax] first pass: find max + sum for both X and Y
    m_X = float("-inf")  # m is the max value. use the notation from the paper
    d_X = 0.0  # d is the sum. use the notation from the paper
    m_Y = float("-inf")
    d_Y = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature

        block_max_X = tl.max(X, axis=0)
        m_new_X = tl.maximum(m_X, block_max_X)
        d_X = d_X * tl.exp2((m_X - m_new_X) * LOG2_E) + tl.sum(tl.exp2((X - m_new_X) * LOG2_E), axis=0)
        m_X = m_new_X

        block_max_Y = tl.max(Y, axis=0)
        m_new_Y = tl.maximum(m_Y, block_max_Y)
        d_Y = d_Y * tl.exp2((m_Y - m_new_Y) * LOG2_E) + tl.sum(tl.exp2((Y - m_new_Y) * LOG2_E), axis=0)
        m_Y = m_new_Y

    lse_X = m_X + tl.log(d_X)
    lse_Y = m_Y + tl.log(d_Y)

    # 3. [Online softmax] second pass: accumulate the loss and dX_sum = sum_j dX_j, the only
    # cross-column term the log-softmax Jacobian needs
    loss_sum = 0.0
    dX_sum = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature - lse_X
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature - lse_Y
        Q = tl.exp2(X * LOG2_E)  # = exp(X)
        P = tl.exp2(Y * LOG2_E)  # = exp(Y)

        if beta == 0.0:  # forward KL
            loss = P * (Y - X)
            dX = -P
        elif beta == 1.0:  # reverse KL
            loss = Q * (X - Y)
            dX = loss + Q
        else:
            M = beta * P + (1 - beta) * Q
            log_M = tl.log(M)

            loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
            dX = (1 - beta) * Q * (X - log_M)

        loss_sum += tl.sum(tl.where(mask, loss, 0.0), axis=0)
        dX_sum += tl.sum(tl.where(mask, dX, 0.0), axis=0)

    # Pre-compute scaling factor
    scale = 1.0 / n_non_ignore
    tl.store(loss_ptr, loss_sum * scale)

    # 4. [Online softmax] third pass: compute gradients w.r.t. the student logits. dX and Q are
    # recomputed rather than kept alive, since a full row of them does not fit in registers.
    # Here we use a trick to store X_ptr gradient in X_ptr so we can save memory
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature - lse_X
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32) * inv_temperature - lse_Y
        Q = tl.exp2(X * LOG2_E)
        P = tl.exp2(Y * LOG2_E)

        if beta == 0.0:  # forward KL
            dX = -P
        elif beta == 1.0:  # reverse KL
            dX = Q * (X - Y) + Q
        else:
            M = beta * P + (1 - beta) * Q
            log_M = tl.log(M)

            dX = (1 - beta) * Q * (X - log_M)

        dX = (dX - Q * dX_sum) * inv_temperature * scale
        tl.store(X_ptr + offsets, dX, mask=mask)


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
    accum_dtype=None,
):
    device = student_input.device

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

    if accum_dtype is None:
        grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    else:
        grad_weight = (
            torch.zeros_like(student_weight, dtype=accum_dtype, device=device) if student_weight.requires_grad else None
        )
    grad_input = torch.zeros_like(student_input)
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

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

        # when doing matmul, use the original precision
        student_logits_chunk = student_input_chunk @ student_weight.t()  # chunk_size x V
        teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()  # chunk_size x V
        chunk_n_rows = student_logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,

        # ensure student and teacher logits are contiguous
        student_logits_chunk = student_logits_chunk.contiguous()
        teacher_logits_chunk = teacher_logits_chunk.contiguous()

        # Here we calculate both the loss and the gradient of student_logits_chunk in place so we can save memory.
        _fused_linear_jsd_kernel[(chunk_n_rows,)](
            X_ptr=student_logits_chunk,
            X_stride=student_logits_chunk.stride(-2),
            Y_ptr=teacher_logits_chunk,
            Y_stride=teacher_logits_chunk.stride(-2),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            label_ptr=(
                shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device)
            ),  # dummy ptr if no label
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            temperature=temperature,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
            num_warps=32 if not is_hip() else 16,
        )
        loss_1d[start_idx:end_idx] = loss_1d_slice

        grad_logits_chunk = student_logits_chunk  # chunk_size x V
        grad_input[start_idx:end_idx] = grad_logits_chunk @ student_weight

        if grad_weight is not None:
            if accum_dtype is None:
                grad_weight.add_(grad_logits_chunk.t() @ student_input_chunk)
            else:
                grad_logits_t = grad_logits_chunk.t()
                if (
                    _ADDMM_SUPPORTS_OUT_DTYPE
                    and grad_weight.device.type == "cuda"
                    and torch.cuda.get_device_capability(grad_weight.device)[0] >= 8
                    and grad_weight.dtype == torch.float32
                    and grad_logits_t.dtype in (torch.float16, torch.bfloat16)
                ):
                    input_chunk = student_input_chunk
                    if input_chunk.dtype != grad_logits_t.dtype:
                        input_chunk = input_chunk.to(grad_logits_t.dtype)
                    torch.addmm(
                        grad_weight,
                        grad_logits_t,
                        input_chunk,
                        out_dtype=torch.float32,
                        out=grad_weight,
                    )
                else:
                    grad_weight.add_(torch.mm(grad_logits_t, student_input_chunk).float())

    loss = torch.sum(loss_1d)
    grad_weight = (
        grad_weight.to(student_weight.dtype) if grad_weight is not None and accum_dtype is not None else grad_weight
    )
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
        accum_dtype: Optional[torch.dtype] = None,
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
            accum_dtype,
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
        return (grad_input, grad_weight, None, None, None, None, None, None, None)
