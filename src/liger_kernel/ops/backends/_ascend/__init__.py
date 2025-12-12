from liger_kernel.ops.backends.registry import VendorInfo
from liger_kernel.ops.backends.registry import register_vendor

# Register Ascend vendor for NPU device
register_vendor(VendorInfo(vendor="ascend", device="npu"))
