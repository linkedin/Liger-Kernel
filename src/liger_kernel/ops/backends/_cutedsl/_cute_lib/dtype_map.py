# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""torch -> cute dtype mapping.

Trimmed slice of ``quack.cute_dsl_utils`` — only the
``torch2cute_dtype_map`` symbol is used by Liger. The bulk of that
module is the TVM-FFI converter patch, the arch-string parser, the
hardware-info LRU cache, and the ``ParamsBase`` dataclass — none of
which are reachable from our forward/backward call graph.
"""

import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import Int64

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}
