"""CuTile JSD operator adapters."""

from . import fused_linear_jsd as _fused_linear_jsd
from . import jsd as _jsd
from .fused_linear_jsd import LigerFusedLinearJSDFunction
from .fused_linear_jsd import TileGymLMHeadJSD
from .fused_linear_jsd import fused_linear_jsd_backward
from .fused_linear_jsd import fused_linear_jsd_forward
from .jsd import LigerJSDFunction
from .jsd import TileGymJSD
from .jsd import jsd_backward
from .jsd import jsd_forward

TILEGYM_AVAILABLE = _jsd._TILEGYM_AVAILABLE and _fused_linear_jsd._TILEGYM_AVAILABLE


def _require_tilegym() -> None:
    if TILEGYM_AVAILABLE:
        return

    import_error = _jsd._TILEGYM_IMPORT_ERROR or _fused_linear_jsd._TILEGYM_IMPORT_ERROR
    raise ImportError("tilegym cutile backend is not available. Install it from the ocean repo.") from import_error


__all__ = [
    "LigerFusedLinearJSDFunction",
    "fused_linear_jsd_forward",
    "fused_linear_jsd_backward",
    "LigerJSDFunction",
    "jsd_forward",
    "jsd_backward",
]
