"""Ascend TileLang backend registration for RoPE."""

from liger_kernel.backends import register_op
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_CAPABILITY
from liger_kernel.ops.tilelang.ascend.dispatch_common import ASCEND_TILELANG_RANK
from liger_kernel.ops.tilelang.ascend.dispatch_common import DEFAULT_TOLERANCES


@register_op(
    "rope",
    impl_name="ascend-tilelang",
    capability=ASCEND_TILELANG_CAPABILITY,
    modes=("default",),
    default_mode="default",
    preference_rank=ASCEND_TILELANG_RANK,
    tolerances=DEFAULT_TOLERANCES,
    notes="Ascend C RoPE with full and partial rotation for Ascend 950.",
)
def rope_ascend_tilelang(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, *, mode=None, rotary_dim=None):
    if mode not in (None, "default"):
        raise ValueError(f"ascend-tilelang RoPE only supports mode='default', got {mode!r}")
    from liger_kernel.ops.tilelang.ascend.ops.rope import LigerRopeFunction

    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim, rotary_dim)
