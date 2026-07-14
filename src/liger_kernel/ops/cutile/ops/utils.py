# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct

ConstBool = ct.Constant[bool]
ConstInt = ct.Constant[int]


def _next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


@ct.kernel(occupancy=1)
def element_mul_kernel(
    x,  # (n_rows, n_cols) updated in-place
    grad_output,  # 0-D tensor
    n_cols,
    BLOCK_SIZE: ConstInt,
    CHECK_BOUNDS: ConstBool,
):
    """
    Multiply each row of ``x`` by the scalar loss gradient in-place.

    ``grad_output`` must be a 0-D tensor (typical for scalar loss.backward()).
    Mirrors ``liger_kernel.ops.utils.element_mul_kernel``.
    """
    row_idx = ct.bid(0)
    grad = ct.astype(ct.gather(grad_output, (), check_bounds=False), ct.float32)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)
        x_tile = ct.astype(
            ct.gather(x, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0),
            ct.float32,
        )
        x_tile = x_tile * grad
        ct.scatter(x, (row_idx, col_idx), ct.astype(x_tile, x.dtype), check_bounds=CHECK_BOUNDS)
