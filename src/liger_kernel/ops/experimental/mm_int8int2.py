import torch
import triton
import triton.language as tl


def unpack_weights(packed: torch.Tensor, bits: int = 2) -> torch.Tensor:
    values_per_item = 8 // bits
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(values_per_item):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    unpacked = unpacked.to(torch.int32) - 1
    return unpacked


def pack_weights(intweights: torch.Tensor, bits: int = 2) -> torch.Tensor:
    intweights += 1
    original_shape = intweights.shape
    values_per_item = 8 // bits
    row_dim = (original_shape[0] + values_per_item - 1) // values_per_item

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(
        packed_tensor_shape, device=intweights.device, dtype=torch.uint8
    )
    unpacked = intweights.to(torch.uint8)

    def lshift(t: torch.Tensor, bits: int):
        return t << bits

    it = min(values_per_item, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= lshift(unpacked[start:end], bits * i)

    return packed


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # We want K / 4 to be divisible by BLOCK_SIZE_K so that the multiplication can be aligned
    tl.static_assert(
        K % (4 * BLOCK_SIZE_K) == 0,
        "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K",
    )
    # determine the block id in the 1D grid, pid <=> blockId in cuda
    pid = tl.program_id(axis=0)
    # number of blocks we would need in the M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # number of blocks we would need in the N dimension
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # blocks are grouped along the M dimension. num_pid_in_group computes how many blocks are grouped together,
    # and group_id calculates the group to which the current block (pid) belongs.
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group

    # pid of the first block in the group that the current block belongs too
    first_pid_m = group_id * GROUP_SIZE_M

    # pid_m : pid of the block along the M dimension of the output matrix, and pid_n : pid of the block along the N dimension of the output matrix
    # remember that the grid of blocks is 1D, but we calculate pid_m and pid_n to locate the block pid place in the output matrix
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offs_am represent the indices of elements within the block for matrices A with respect to the M dimension
    # offs_bn represent the indices of elements within the block for matrices B with respect to the N dimension
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    """
        This part of the code generates pointers to the specific blocks of matrices A and B that the current thread block will process.

        As described in the PyTorch documentation, a stride refers to the step size needed to move from one element to the next along a given dimension:

        For matrix A: stride_am = A.stride(0) = K (stride along the rows), and stride_ak = A.stride(1) = 1 (stride along the columns).
        For matrix B: stride_bk = B.stride(0) = N (stride along the rows), and stride_bn = B.stride(1) = 1 (stride along the columns).
        Now, let's break down the pointer generation:

        offs_am[:, None] creates a column of shape [BLOCK_SIZE_M, 1], which represents the row indices of matrix A that this block is processing. It is multiplied by K (the number of columns in matrix A) since A is stored in row-major order. So, the element at position (i, j) in A is located at index i*K + j in memory.
        offs_k[None, BLOCK_SIZE_K] creates a row vector representing the column indices of the block, i.e., a range from 0 to BLOCK_SIZE_K. This is used to compute the positions of the columns within the block.
        When combined, the result has the shape [BLOCK_SIZE_M, BLOCK_SIZE_K], where each entry (i, j) points to the element in matrix A at position (i, j) for the current block.

        The same logic is applied to matrix B, but the resulting shape is [BLOCK_SIZE_K, BLOCK_SIZE_N], representing the block of matrix B that the thread block will work on.
    """
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # An accumulator matrix is initialized with zeros. It stores the intermediate results of the block matrix multiplication.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    """
        We split the loop into two layers. The outer loop runs 4 times, and each iteration focuses on a specific portion of matrix A.

        For example, when i = 0, we’re only concerned with the blocks of matrix A that cover the range from 0 to K // (4 * BLOCK_SIZE_K).
        Since matrix B is packed, its first dimension is effectively divided by 4. So, while we process the first segment of matrix A,
        we still iterate over the entire first dimension of matrix B.

        In each of the 4 iterations of the outer loop, we go through the full blocks of matrix B, but what changes is the data we extract.
        Matrix B elements contain 4 weights, all packed into an int8 format, and during each iteration of the outer loop,
        we extract a different weight by using bitwise shifting operations. This way, we access a unique weight on each pass.
    """
    for i in range(4):
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
            # load the block of matrix A
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            # load the block of matrix B
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
            # when i = 0 for example, we only care about the first 2 bits of the elements of the matrix B, so we use the mask 00000011 to mask the other bits
            mask = 3 << (2 * i)
            # we shift the results after the mask
            b = (b_uint8 & mask) >> (2 * i)
            # During the packing of the weights, it's easier to pack 0, 1, 2, then -1, 0, 1, so we add 1 to the weight tensor, and we substract it here
            tensor_full = tl.full((1,), 1, dtype=tl.int8)
            # We accumulate the result of multiplication of the blocks along the K dimension on int32 to avoid any overflows or underflows.
            accumulator += tl.dot(a, (b.to(tl.int8) - tensor_full), out_dtype=tl.int32)
            # we move the pointers, for the a_ptrs we more in a horizontal way along the second dimension -> we use strid_ak=1
            # for b_ptrs we move in a vertical way, along the rows -> we use stride_bk=N
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator
    # These lines compute the offsets into matrix C where the result of this block’s computation should be stored.
    # stride_cm = N & stride_cn = 1
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # we do a boundary check to ensure only elements within matrix bounds are stored
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert (
        a.shape[1] == b.shape[0] * 4
    ), "Incompatible dimensions, the weight matrix need to be packed"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    # c is in int32 to avoid any overflows or underflows
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
