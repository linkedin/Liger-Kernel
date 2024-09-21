import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log P
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log Q
    Y_stride,
    loss_ptr,
    loss_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # JSD(P || Q) = (1/2) * (KL(P || M) + KL(Q || M)), M = (1/2) * (P + Q) = (1/2) * (e ^ X + e ^ Y)
    #             = log 2 + (1/2) * sum(X * e ^ X + Y * e ^ Y - 2 log M)
    # grad_x_i = (1/2) * (e ^ x_i + x_i * e ^ x_i - 2 * (1 / m_i) * (1 / 2) * e ^ x_i)
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride

    loss_sum = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        exp_X = tl.exp(X)
        exp_Y = tl.exp(Y)
        M = (1 / 2) * (exp_X + exp_Y)

        loss = tl.log(2.0) + (1 / 2) * (X * exp_X + Y * exp_Y - 2 * tl.log(M))
        loss_sum += tl.sum(loss)
        grad_X = (1 / 2 * n_rows) * (exp_X + exp_X * X - exp_X / M)

        tl.store(X_ptr + offsets, grad_X, mask=mask)

    tl.store(loss_ptr, loss_sum)


MAX_FUSED_SIZE = 65536


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def jsd_forward(_input, target):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)

    _jsd_kernel[(n_rows,)](
        X_ptr=_input,  # input in logspace, X = log P
        X_stride=_input.stride(-2),
        Y_ptr=target,  # ground truth in logspace, Y = log Q
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_rows=n_rows,
        n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # calculated according to the reduction mode same as in Pytorch. In the later versions, `mean` will be changed to the same behavior as `batchmean`
    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # https://github.com/pytorch/pytorch/blob/d7b57c4d63edb42e1deeeba9497fcb5f1f748ff2/torch/nn/functional.py#L3372
    loss = torch.sum(loss_1d) / n_rows
    return loss, _input


def jsd_backward(_input, grad_output):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass

    # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
    # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return _input


class LigerJSDFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the JS Divergence Loss using Triton, as defined by the following formula:
    ```python
    if log_target:
        loss = target * (target.log() - input)
    else:
        loss = target.exp() * (target - input)
    ```,
    then the loss is reduced according to the `reduction` parameter.
    as defined in the PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss, _input = jsd_forward(_input, target)
        ctx.save_for_backward(_input.detach())
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (_input,) = ctx.saved_tensors
        _input = jsd_backward(_input, grad_output)
        return (
            _input,
            None,
        )
