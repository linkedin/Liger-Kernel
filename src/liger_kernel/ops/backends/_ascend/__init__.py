from liger_kernel.ops.backends.registry import VendorInfo
from liger_kernel.ops.backends.registry import register_vendor

register_vendor(
    VendorInfo(
        vendor="ascend",
        device="npu",
        module_path="src.liger_kernel.ops.backends._ascend.ops",
    )
)
