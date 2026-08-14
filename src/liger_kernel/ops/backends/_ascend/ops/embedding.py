from functools import lru_cache

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# Ascend UB capacity is 192 KB.
_ASCEND_UB_CAPACITY_BITS = 1572864
_UB_MULTIPLIER = 3.2
_UB_SAFETY = 0.85

# Wide embeddings (embedding_dim >= threshold) use large N tiles tuned on Ascend910 bf16.
_WIDE_EMBEDDING_THRESHOLD = 2048
_FORWARD_MAX_BLOCK_N = 512
_FORWARD_WIDE_BLOCK_N_CANDIDATES = (4096, 2048, 1024, 512)

# Ascend910 bf16 2D forward compile limits: max BLOCK_SIZE_M per BLOCK_SIZE_N.
_FORWARD_COMPILED_MAX_M = {
    4096: 6,
    2048: 12,
    1024: 18,
    512: 37,
    256: 75,
    128: 126,
    64: 128,
}


@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim
        block_mask = mask_m[:, None] & mask_n[None, :]

        embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        embeddings = tl.load(
            embeddings_ptr + embedding_offsets,
            mask=block_mask,
            other=0.0,
        )

        output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)


@triton.jit
def embedding_forward_kernel_mouter(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)

    for block_m in tl.range(pid, grid_m, num_progs):
        start_m = block_m * BLOCK_SIZE_M

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        for block_n in tl.range(0, grid_n):
            start_n = block_n * BLOCK_SIZE_N

            offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
            mask_n = offsets_n < embedding_dim
            block_mask = mask_m[:, None] & mask_n[None, :]

            embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
            embeddings = tl.load(
                embeddings_ptr + embedding_offsets,
                mask=block_mask,
                other=0.0,
            )

            output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)

    for block_m in tl.range(pid, grid_m, num_progs):
        start_m = block_m * BLOCK_SIZE_M

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        for block_n in tl.range(0, grid_n):
            start_n = block_n * BLOCK_SIZE_N
            offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
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


@triton.jit
def embedding_backward_kernel_2d(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
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


def _max_block_m_for_ub(
    block_size_n: int,
    dtype_size: int,
    multiplier: float = _UB_MULTIPLIER,
    safety_margin: float = _UB_SAFETY,
) -> int:
    usable_bits = int(_ASCEND_UB_CAPACITY_BITS * safety_margin)
    bits_per_m = max(1, int(multiplier * block_size_n * dtype_size * 8))
    return max(1, min(usable_bits // bits_per_m, 128))


def _clamp_forward_block_m(block_m: int, block_n: int, dtype_size: int) -> int:
    compile_max_m = _FORWARD_COMPILED_MAX_M.get(block_n)
    if compile_max_m is None:
        compile_max_m = max(
            16,
            int(_ASCEND_UB_CAPACITY_BITS * 0.99 / (82.5 * block_n * max(dtype_size, 1) / 2)),
        )
    ub_max_m = _max_block_m_for_ub(block_n, dtype_size)
    return max(1, min(block_m, compile_max_m, ub_max_m))


def _largest_wide_forward_block_n(embedding_dim: int, dtype_size: int) -> int:
    for block_n in _FORWARD_WIDE_BLOCK_N_CANDIDATES:
        if block_n > embedding_dim:
            continue
        if _clamp_forward_block_m(4, block_n, dtype_size) >= 4:
            return block_n
    return _FORWARD_MAX_BLOCK_N


def _select_forward_tile_sizes(
    n_elements: int,
    embedding_dim: int,
    dtype_size: int,
) -> tuple[int, int]:
    if embedding_dim >= _WIDE_EMBEDDING_THRESHOLD:
        block_n = _largest_wide_forward_block_n(embedding_dim, dtype_size)
        max_m = _clamp_forward_block_m(128, block_n, dtype_size)
        block_m = min(6, max_m)
        return max(1, block_m), block_n

    best_key = None
    best_m = 64
    best_n = triton.next_power_of_2(min(128, embedding_dim))

    for block_n in (_FORWARD_MAX_BLOCK_N, 256, 128, 64):
        if block_n > embedding_dim:
            continue
        block_n = triton.next_power_of_2(block_n)
        max_m = _max_block_m_for_ub(block_n, dtype_size)
        if max_m < 16:
            continue

        block_m = min(max_m, triton.next_power_of_2(min(128, n_elements)))
        while block_m > max_m and block_m > 1:
            block_m //= 2
        block_m = _clamp_forward_block_m(block_m, block_n, dtype_size)
        if block_m < 1:
            continue

        grid_m = triton.cdiv(n_elements, block_m)
        grid_n = triton.cdiv(embedding_dim, block_n)
        n_bias = block_n if embedding_dim >= 1024 else 0
        key = (-(block_m * block_n), -n_bias, grid_m * grid_n)
        if best_key is None or key < best_key:
            best_key = key
            best_m, best_n = block_m, block_n

    return best_m, best_n


def _pick_forward_schedule(
    n_elements: int,
    embedding_dim: int,
    block_n: int,
) -> tuple[bool, int]:
    if embedding_dim < _WIDE_EMBEDDING_THRESHOLD:
        return False, 2 if n_elements >= 2048 else 1

    grid_n = triton.cdiv(embedding_dim, block_n)
    if grid_n == 1:
        if n_elements >= 8192:
            return False, 8
        if n_elements >= 4096:
            return False, 6
        if n_elements >= 2048:
            return False, 1
        return False, 2

    if n_elements <= 1024:
        return True, 3
    if n_elements <= 2048:
        return False, 1
    return False, 2


@lru_cache(maxsize=256)
def _get_optimal_block_m(n_elements: int, dtype_size: int, block_size_n: int) -> int:
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=_UB_SAFETY,
        dtype_size=dtype_size,
        memory_multiplier=_UB_MULTIPLIER,
        shapes=((n_elements, block_size_n),),
        tiling_dims=(0,),
    )
    block_m = tile_shapes[0][0] if tile_shapes else triton.next_power_of_2(min(128, n_elements))
    return min(block_m, _max_block_m_for_ub(block_size_n, dtype_size))


@lru_cache(maxsize=256)
def _select_backward_tile_sizes(
    n_elements: int,
    embedding_dim: int,
    dtype_size: int,
) -> tuple[int, int]:
    if embedding_dim >= _WIDE_EMBEDDING_THRESHOLD:
        block_n = min(_FORWARD_MAX_BLOCK_N, triton.next_power_of_2(embedding_dim))
        max_m = _max_block_m_for_ub(block_n, dtype_size)
        return _clamp_forward_block_m(max_m, block_n, dtype_size), block_n

    if n_elements >= 4096 and embedding_dim >= 256:
        block_n = 256
    else:
        block_n = triton.next_power_of_2(min(128, embedding_dim))
    return _get_optimal_block_m(n_elements, dtype_size, block_n), block_n


@lru_cache(maxsize=256)
def _get_forward_launch_config(n_elements: int, embedding_dim: int, dtype_size: int):
    block_m, block_n = _select_forward_tile_sizes(n_elements, embedding_dim, dtype_size)
    use_mouter, core_mult = _pick_forward_schedule(n_elements, embedding_dim, block_n)
    return block_m, block_n, use_mouter, core_mult


def _launch_grid(num_cores: int, total_blocks: int, core_multiplier: int = 1) -> int:
    return max(1, min(num_cores * core_multiplier, total_blocks))


def embedding_forward(embeddings, indices):
    ori_shape = indices.shape
    indices = indices.view(-1)

    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]
    if n_elements == 0:
        return torch.empty(*ori_shape, embedding_dim, device=indices.device, dtype=embeddings.dtype)

    output = torch.empty(
        n_elements,
        embedding_dim,
        device=indices.device,
        dtype=embeddings.dtype,
    )

    block_m, block_n, use_mouter, core_mult = _get_forward_launch_config(
        n_elements, embedding_dim, embeddings.element_size()
    )
    num_cores = get_npu_core_count()

    if use_mouter:
        total_blocks = triton.cdiv(n_elements, block_m)
        kernel = embedding_forward_kernel_mouter
    else:
        total_blocks = triton.cdiv(n_elements, block_m) * triton.cdiv(embedding_dim, block_n)
        kernel = embedding_forward_kernel

    kernel[_launch_grid(num_cores, total_blocks, core_mult),](
        embeddings,
        indices,
        output,
        n_elements,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
    )

    return output.view(*ori_shape, -1)


def embedding_backward(embeddings, indices, grad_output):
    indices = indices.contiguous().view(-1)
    grad_output = grad_output.contiguous().view(-1, embeddings.shape[1])

    grad_weight = torch.zeros_like(embeddings)

    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]
    if n_elements == 0:
        return grad_weight

    block_m, block_n = _select_backward_tile_sizes(n_elements, embedding_dim, embeddings.element_size())
    num_cores = get_npu_core_count()
    use_2d = embedding_dim >= _WIDE_EMBEDDING_THRESHOLD or n_elements >= 4096

    if use_2d:
        total_blocks = triton.cdiv(n_elements, block_m) * triton.cdiv(embedding_dim, block_n)
        core_mult = 1 if 4096 <= n_elements < 8192 else (2 if n_elements >= 2048 else 1)
        grid = _launch_grid(num_cores, total_blocks, core_mult)
        embedding_backward_kernel_2d[(grid,)](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
        )
    else:
        total_blocks = triton.cdiv(n_elements, block_m)
        grid = _launch_grid(num_cores, total_blocks)
        embedding_backward_kernel[(grid,)](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
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
