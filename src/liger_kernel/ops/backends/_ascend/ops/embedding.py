import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs, num_stages=NUM_STAGES):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements

        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim

        block_mask = mask_m[:, None] & mask_n[None, :]

        embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        embeddings = tl.load(
            embeddings_ptr + embedding_offsets,
            mask=block_mask,
            other=0.0,
        )

        output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        tl.store(
            output_ptr + output_offsets,
            embeddings,
            mask=block_mask,
        )


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs, num_stages=NUM_STAGES):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements

        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim

        block_mask = mask_m[:, None] & mask_n[None, :]

        grad_output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        grad_output = tl.load(
            grad_output_ptr + grad_output_offsets,
            mask=block_mask,
            other=0.0,
        )

        grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + grad_weight_offsets,
            grad_output,
            mask=block_mask,
        )


def get_optimal_block_size(total_elements, dtype_size, BLOCK_SIZE_N: tl.constexpr):
    # 1. Set Memory Multiplier
    # 3.0 are empirical values based on 910B UB (192KB)
    # embedding_offsets, embedding_offsets : BLOCK_SIZE_N * BLOCK_SIZE_M (total 2 * BLOCK_SIZE_N * BLOCK_SIZE_M)
    # Reserve a unit of space for the remaining one-dimensional ub to occupy.
    # A conservative estimate of the total space occupation is 3 * BLOCK_SIZE_N * BLOCK_SIZE_M
    multiplier = 3.0

    # 2. Call calculation function
    # Treat input as 1D (total_elements,), only tiling on dim 0
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=dtype_size,
        memory_multiplier=multiplier,
        shapes=((total_elements, BLOCK_SIZE_N),),
        tiling_dims=(0,),
    )

    # 3. Parse result
    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return triton.next_power_of_2(min(128, total_elements))


def embedding_forward(embeddings, indices):
    ori_shape = indices.shape
    indices = indices.view(-1)

    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]
    output = torch.empty(
        indices.shape[0],
        embeddings.shape[1],
        device=indices.device,
        dtype=embeddings.dtype,
    )

    # Due to the involvement of two-dimensional partitioning,
    # the sizes of block_m and block_n in the ub space will influence each other.
    # Considering that embedding_dim is usually relatively smaller in most cases,
    # a value is first assigned to block_n, and then the largest possible block_m is used.
    BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
    BLOCK_SIZE_M = get_optimal_block_size(n_elements, embeddings.element_size(), BLOCK_SIZE_N)
    num_cores = get_npu_core_count()
    total_blocks = triton.cdiv(n_elements, BLOCK_SIZE_M) * triton.cdiv(embedding_dim, BLOCK_SIZE_N)
    grid = min(num_cores, total_blocks)

    embedding_forward_kernel[(grid,)](
        embeddings,
        indices,
        output,
        n_elements,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=3,
    )

    return output.view(*ori_shape, -1)


def embedding_backward(embeddings, indices, grad_output):
    grad_output = grad_output.contiguous().view(-1, embeddings.shape[1])

    grad_weight = torch.zeros_like(embeddings)

    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]
    BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
    BLOCK_SIZE_M = get_optimal_block_size(n_elements, embeddings.element_size(), BLOCK_SIZE_N)
    num_cores = get_npu_core_count()
    total_blocks = triton.cdiv(n_elements, BLOCK_SIZE_M) * triton.cdiv(embedding_dim, BLOCK_SIZE_N)
    grid = min(num_cores, total_blocks)

    embedding_backward_kernel[(grid,)](
        grad_output,
        grad_weight,
        indices,
        n_elements,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=3,
    )

    return grad_weight


class LigerEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        output = embedding_forward(embeddings, indices)
        ctx.save_for_backward(indices, embeddings)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embeddings = ctx.saved_tensors
        grad_weight = embedding_backward(embeddings, indices, grad_output)

        return grad_weight, None
