import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_settings,
    ensure_contiguous
)

@triton.jit
def embedding_forward_kernel(
    embeddings_ptr, indices_ptr, output_ptr, n_elements, embedding_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    for i in range(0, embedding_dim, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < (embedding_dim - i)
        embedding_offsets = indices[:, None] * embedding_dim + (i + col_offsets[None, :])
        embeddings = tl.load(embeddings_ptr + embedding_offsets, mask=mask[:, None] & col_mask[None, :], other=0.0)
        tl.store(output_ptr + offsets[:, None] * embedding_dim + (i + col_offsets[None, :]), embeddings, mask=mask[:, None] & col_mask[None, :])

@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, grad_weight_ptr, indices_ptr, n_elements, embedding_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr 
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    for i in range(0, embedding_dim, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < (embedding_dim - i)
        grads = tl.load(grad_output_ptr + offsets[:, None] * embedding_dim + (i + col_offsets[None, :]), mask=mask[:, None] & col_mask[None, :])
        embedding_offsets = indices[:, None] * embedding_dim + (i + col_offsets[None, :])
        tl.atomic_add(grad_weight_ptr + embedding_offsets, grads, mask=mask[:, None] & col_mask[None, :])

class LigerEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ori_shape = indices.shape
        indices = indices.view(-1)
        output = torch.empty(indices.shape[0], embeddings.shape[1], device=indices.device, dtype=embeddings.dtype)

        n_elements = indices.numel()
        BLOCK_SIZE, num_warps = calculate_settings(embeddings.shape[1])

        embedding_forward_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
            embeddings,
            indices,
            output,
            n_elements,
            embedding_dim=embeddings.shape[1],
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(indices, embeddings)

        return output.view(*ori_shape, -1)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embedding_table = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, embedding_table.shape[1])
        grad_weight = torch.zeros_like(embedding_table)

        n_elements = indices.numel()
        BLOCK_SIZE, num_warps = calculate_settings(embedding_table.shape[1])

        embedding_backward_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_table.shape[1],
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return grad_weight, None
