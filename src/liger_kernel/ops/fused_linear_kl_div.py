from typing import Literal
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

MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2
_TORCH_VERSION = Version(torch.__version__.split("+")[0])
_ADDMM_SUPPORTS_OUT_DTYPE = _TORCH_VERSION >= Version("2.8.0")

REDUCTION_LITERAL = Literal["batchmean", "mean", "sum"]


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


@triton.jit
def _kl_div_kernel(
    Y_ptr,  # [BT, V], student log-probabilities, Y = log softmax(x / temperature)
    Y_stride,
    Q_ptr,  # [BT, V], target probabilities
    Q_stride,
    loss_ptr,  # [BT], per-row loss (already scaled by reduction)
    loss_stride,
    dY_ptr,  # [BT, V], gradient w.r.t. Y, written in-place over Y
    dY_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,
    eps,
    scale,  # pre-computed reduction scale fused into loss and gradients
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    Y_ptr += pid * Y_stride
    Q_ptr += pid * Q_stride
    loss_ptr += pid * loss_stride
    dY_ptr += pid * dY_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dY_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        Y = tl.load(Y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        Q = tl.load(Q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # KL(Q || P) = sum(Q * (log(Q) - log(P))) = sum(Q * (log(Q) - Y))
        # 0 * log(0) is treated as 0 by clamping Q to eps before the log
        loss = Q * (tl.log(tl.maximum(Q, eps)) - Y)
        dY = -Q

        loss_sum += tl.sum(loss, axis=0)
        tl.store(dY_ptr + offsets, dY * scale, mask=mask)

    tl.store(loss_ptr, loss_sum * scale)


def fused_linear_kl_div_forward(
    student_input,
    student_weight,
    target,
    shift_labels,
    reduction,
    ignore_index,
    has_label,
    temperature,
    eps,
    accum_dtype=None,
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

    if accum_dtype is None:
        grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    else:
        grad_weight = (
            torch.zeros_like(student_weight, dtype=accum_dtype, device=device) if student_weight.requires_grad else None
        )
    grad_input = torch.zeros_like(student_input)
    # we use fp32 for the per-row loss accumulator
    loss_1d = torch.zeros((BT,), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    # Pre-compute reduction scale (fused into the kernel; matches torch.nn.KLDivLoss semantics)
    if n_non_ignore == 0:
        scale = 0.0
    elif reduction == "batchmean":
        scale = 1.0 / n_non_ignore
    elif reduction == "mean":
        scale = 1.0 / (n_non_ignore * V)
    else:  # "sum"
        scale = 1.0

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk the inputs, shape: chunk_size x H / chunk_size x V
        input_chunk = student_input[start_idx:end_idx]
        target_chunk = target[start_idx:end_idx]

        # shape: chunk_size x V
        # For anything starting from logits to the final KL loss, we do computation
        # in FP32 to avoid losing numerical stability.
        logits_chunk = (input_chunk @ student_weight.t()).to(torch.float32) / temperature
        chunk_n_rows = logits_chunk.shape[0]

        # log-softmax with temperature
        log_prob_chunk = torch.log_softmax(logits_chunk, dim=-1).contiguous()

        # Here we calculate the gradient w.r.t. log_prob_chunk in place so we can save memory.
        _kl_div_kernel[(chunk_n_rows,)](
            Y_ptr=log_prob_chunk,
            Y_stride=log_prob_chunk.stride(-2),
            Q_ptr=target_chunk,
            Q_stride=target_chunk.stride(-2),
            loss_ptr=loss_1d[start_idx:end_idx],
            loss_stride=loss_1d[start_idx:end_idx].stride(0),
            dY_ptr=log_prob_chunk,
            dY_stride=log_prob_chunk.stride(-2),
            label_ptr=(
                shift_labels[start_idx:end_idx] if has_label else torch.empty(1, device=device)
            ),  # dummy ptr if no label
            ignore_index=ignore_index,
            n_cols=V,
            eps=eps,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
            num_warps=get_num_warps(BLOCK_SIZE),
        )
        # gradients of log_prob_chunk in place, shape: chunk_size x V
        # backprop through log-softmax:
        # dL/dlogits = g - softmax(logits) * sum(g), then divided by temperature
        # (log_prob_chunk now holds g = dL/dlog_prob)
        grad_logits_chunk = (
            log_prob_chunk
            - torch.softmax(logits_chunk, dim=-1)
            * log_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(log_prob_chunk.shape)
        ) / temperature
        # now we traverse back to grad w.r.t. input of `lm_head` and grad
        # w.r.t. `lm_head` which should be computed in original dtype
        grad_logits_chunk = grad_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = grad_logits_chunk @ student_weight

        if grad_weight is not None:
            if accum_dtype is None:
                grad_weight.add_(grad_logits_chunk.t() @ input_chunk)
            else:
                grad_logits_t = grad_logits_chunk.t()
                if (
                    _ADDMM_SUPPORTS_OUT_DTYPE
                    and grad_weight.device.type == "cuda"
                    and torch.cuda.get_device_capability(grad_weight.device)[0] >= 8
                    and grad_weight.dtype == torch.float32
                    and grad_logits_t.dtype in (torch.float16, torch.bfloat16)
                ):
                    chunk_input = input_chunk
                    if chunk_input.dtype != grad_logits_t.dtype:
                        chunk_input = chunk_input.to(grad_logits_t.dtype)
                    torch.addmm(
                        grad_weight,
                        grad_logits_t,
                        chunk_input,
                        out_dtype=torch.float32,
                        out=grad_weight,
                    )
                else:
                    grad_weight.add_(torch.mm(grad_logits_t, input_chunk).float())

    loss = loss_1d.sum()
    grad_weight = (
        grad_weight.to(student_weight.dtype) if grad_weight is not None and accum_dtype is not None else grad_weight
    )
    return loss, grad_input, grad_weight


def fused_linear_kl_div_backward(grad_output, grad_input, grad_weight):
    # If KL is the last layer, grad_output is 1.0. Skip the mul to save time
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


class LigerFusedLinearKLDivFunction(torch.autograd.Function):
    """
    Fusing the last linear layer with KL divergence against an explicit target distribution.

    Handle the forward and backward pass of the final linear layer via KL by avoiding
    the materialization of the large logits tensor. Since the KL loss is the last layer, we can
    compute the gradient at the forward pass.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        target: torch.Tensor,
        shift_labels: Optional[torch.LongTensor] = None,
        reduction: REDUCTION_LITERAL = "batchmean",
        ignore_index: int = -100,
        temperature: float = 1.0,
        eps: float = 1e-10,
        accum_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in the student model, with shape (BT, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in the student model, with shape (V, H), where V is vocab size.
            target (torch.tensor): target distribution with shape (BT, V), expected to be probabilities. Computes KL(target || softmax(student_input @ student_weight^T / temperature)).
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT,). Rows whose label equals ignore_index are excluded from the loss and receive zero gradients. Default: `None`
            reduction (REDUCTION_LITERAL, optional): reduction to be applied over the per-row KL losses: `"batchmean"`, `"mean"` or `"sum"` (same semantics as torch.nn.KLDivLoss). `"none"` is not supported because the gradient-in-forward scheme requires a scalar loss; use :class:`liger_kernel.transformers.kl_div.LigerKLDIVLoss` for element-wise losses. Default: `"batchmean"`
            ignore_index (int, optional): the index to ignore. Default: `-100`
            temperature (float, optional): temperature applied to the student logits before softmax. Default: `1.0`
            eps (float, optional): small value clamping the target before log so entries with q = 0 contribute 0 (0 * log 0 = 0). Default: `1e-10`
            accum_dtype (torch.dtype, optional): dtype of the weight gradient accumulator. Accumulating in `torch.float32` avoids precision loss from the chunked bf16/fp16 GEMM accumulation, at the cost of an extra fp32 weight-sized buffer. Default: `None` (same dtype as `student_weight`)

        Returns:
            loss (torch.Tensor): KL divergence, scaled by the reduction
        """
        if reduction not in ("batchmean", "mean", "sum"):
            raise ValueError(
                f"reduction must be one of ('batchmean', 'mean', 'sum'), got {reduction!r}. "
                "reduction='none' is not supported by the fused kernel; "
                "use LigerKLDIVLoss for element-wise losses."
            )
        assert temperature != 0, "temperature cannot be 0."
        if not target.is_contiguous():
            target = target.contiguous()

        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (student_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grad_input, grad_weight = fused_linear_kl_div_forward(
            student_input,
            student_weight,
            target,
            shift_labels,
            reduction,
            ignore_index,
            has_label,
            temperature,
            eps,
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
        grad_input, grad_weight = fused_linear_kl_div_backward(grad_output, grad_input, grad_weight)
        return (grad_input, grad_weight, None, None, None, None, None, None, None)
