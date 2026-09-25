# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""Minimal slice of ``quack.layout_utils`` — just ``expand``.

The upstream module also exposes ``transpose_view``, ``select``,
``permute_*``, MMA-accumulator reshapes and a zero-stride layout
helper, none of which appear in the forward/backward call graphs of
either ``rmsnorm_fwd`` or our inline backward kernels.
"""

import cutlass.cute as cute

from cutlass import Int32


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))
