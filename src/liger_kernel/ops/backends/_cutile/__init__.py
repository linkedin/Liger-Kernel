"""
cuTile backend for Liger-Kernel.

cuTile is an optional CUDA backend. It is opt-in only — users select it
explicitly via ``LIGER_KERNEL_BACKEND=cutile``. It is not auto-applied on
any device (note the empty ``default_devices`` on the registration below).
"""

from liger_kernel.ops.backends.registry import BackendInfo
from liger_kernel.ops.backends.registry import register_backend

register_backend(BackendInfo(name="cutile", devices=("cuda",)))
