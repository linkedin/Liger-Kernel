from liger_kernel.ops.backends.registry import VendorInfo
from liger_kernel.ops.backends.registry import register_vendor

register_vendor(
    VendorInfo(
        vendor="nvidia",
        device="cuda",
        module_path="src.liger_kernel.ops.backends._nvidia.ops",
    )
)
