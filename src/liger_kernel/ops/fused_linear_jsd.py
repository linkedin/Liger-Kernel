import torch
import triton

from liger_kernel.ops.jsd import _jsd_kernel
from liger_kernel.ops.utils import element_mul_kernel

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    jsd_beta,
    temperature,
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
    chunk_size = triton.next_power_of_2(
        triton.cdiv(BT, inc_factor)
    )  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_weight = (
        torch.zeros_like(student_weight, device=device)
        if student_weight.requires_grad
        else None
    )
    grad_input = torch.zeros_like(student_input)
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk both inputs, shape: chunk_size x H
        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        # when doing matmul, use the original precision, shape: chunk_size x V
        student_logits_chunk = student_input_chunk @ student_weight.t()
        teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()
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
            beta=jsd_beta,
            n_rows=BT,  # batchmean
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        loss_1d[start_idx:end_idx] = loss_1d_slice
        # gradients of prob_chunk in place, shape: chunk_size x V
        # gradients of logits_chunk in place, shape: chunk_size x V
        student_logits_chunk = (
            student_prob_chunk
            - torch.softmax(student_logits_chunk, dim=-1)
            * student_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(
                student_prob_chunk.shape
            )
        ) / temperature
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=student_logits_chunk.t(),  # gradients of logits_chunk
                mat2=student_input_chunk,
                out=grad_weight,
            )

    loss = torch.sum(loss_1d)
    return loss.to(student_input.dtype), grad_input, grad_weight


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
            num_warps=32,
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
                num_warps=32,
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
    def forward(
        ctx,
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        jsd_beta=0.5,
        temperature=1.0,
    ):
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            jsd_beta (float): coefficient beta of generalized JSD in the open interval (0, 1). Default: `0.5`
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            jsd_beta,
            temperature,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(
            grad_output, grad_input, grad_weight
        )
        return (grad_input, grad_weight, None, None, None, None)
