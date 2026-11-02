# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tiled MLP (cuTile backend).

Pure Python implementation — no GPU kernel.
Shards input along sequence dimension (dim=-2), applies fn on each shard,
and concatenates. Backward re-computes forward per shard to save memory.
"""

# Tiling and checkpointing are backend independent; fn selects the MLP kernels.
from liger_kernel.ops.tiled_mlp import LigerTiledMLPFunction  # noqa: F401
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp  # noqa: F401
