# This file contains code adapted from Quack
# (https://github.com/Dao-AILab/quack), Apache License 2.0.
# Copyright (c) 2025 Wentao Guo, Ted Zadouri, Tri Dao.
# Modifications by Liger-Kernel contributors.
"""Inlined CuTe DSL utilities for Liger's nvidia-cutedsl backends.

Adapted from `Quack <https://github.com/Dao-AILab/quack>`_ (Apache-2.0).
Trimmed to the minimum surface that Liger's RMSNorm/LayerNorm forward
+ inline backward kernels reach. Importing this package does **not**
require the upstream ``quack`` package to be installed.

See ``NOTICE.md`` in this directory for a per-symbol attribution map and
the upstream commit hash.
"""

from . import copy_utils
from . import layout_utils
from . import utils
from .compile_utils import make_fake_tensor
from .dtype_map import torch2cute_dtype_map
from .reduce import row_reduce
from .reduction_base import ReductionBase
from .rmsnorm_fwd import layernorm_fwd
from .rmsnorm_fwd import rmsnorm_fwd

__all__ = [
    "copy_utils",
    "layout_utils",
    "utils",
    "make_fake_tensor",
    "torch2cute_dtype_map",
    "row_reduce",
    "ReductionBase",
    "layernorm_fwd",
    "rmsnorm_fwd",
]
