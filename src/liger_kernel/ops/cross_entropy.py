import torch
import triton
import triton.language as tl


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    ignore_index (int): The index to ignore in the target.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(
        X_ptr + y
    )  # we need to store the original value of X_y for the loss calculation

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 4. [Online softmax] second pass: calculate the gradients
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # N is the number of non ignored elements in the batch
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        X_block = (tl.exp(X_block - m) / d) / (n_non_ignore)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = -(ori_X_y - m - tl.log(d))

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - 1) / N`
    X_y = tl.load(X_ptr + y)
    X_y += -1 / (n_non_ignore)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2  # the best size we found by manually tuning


@triton.jit
def element_mul(
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


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger Cross Entropy loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
    """

    @staticmethod
    def forward(ctx, _input, target, ignore_index):
        """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        ignore_index (int): The index to ignore in the target.

        Returns:
        tensor: The computed loss.
        """
        BT, V = _input.shape
        n_rows = BT

        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        # unreduced loss
        loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

        n_non_ignore = (target != ignore_index).sum().item()

        # ensure _input and target are contiguous in the last dimension
        if _input.stride(-1) != 1:
            _input = _input.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()

        # Here we use a trick to store X_ptr gradient in X_ptr so we can save memory
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),  # always 1
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            # TODO: 32 seems to give the best performance
            # Performance is quite sensitive to num_warps
            num_warps=32,
        )

        loss = torch.sum(loss_1d) / n_non_ignore

        # TODO: investigation
        # If we don't detach the _input tensor, the memory will double
        # Not sure why but seems that there will be a time both grad and value exist but in different location
        ctx.save_for_backward(_input.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.

        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        (_input,) = ctx.saved_tensors
        # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
        if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            pass

        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        else:
            BT, V = _input.shape
            n_rows = BT
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

            element_mul[(n_rows,)](
                _input,
                _input.stride(-2),
                grad_output,
                V,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
            )

        return (
            _input,
            None,
            None,
        )
