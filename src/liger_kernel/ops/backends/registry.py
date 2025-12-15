"""
Vendor registry for Liger-Kernel multi-backend support.

This module defines VendorInfo and the registry for vendor registration.
Each vendor registers itself by calling register_vendor() in its __init__.py.
"""

from dataclasses import dataclass
from typing import Optional

# Dynamically get backends package path to avoid hardcoding
_BACKENDS_PACKAGE = __name__.rsplit(".", 1)[0]  # "liger_kernel.ops.backends"


@dataclass
class VendorInfo:
    """
    Information about a chip vendor and its supported device.

    Attributes:
        vendor: Vendor name (e.g., "ascend", "intel", "nvidia")
        device: Device type this vendor supports (e.g., "npu", "xpu")
    """

    vendor: str
    device: str

    @property
    def module_path(self) -> str:
        """Auto-generated module path based on vendor name."""
        return f"{_BACKENDS_PACKAGE}._{self.vendor}.ops"


# Registry mapping device types to their vendor info
# Vendors register themselves via register_vendor()
VENDOR_REGISTRY: dict[str, VendorInfo] = {}


def register_vendor(vendor_info: VendorInfo) -> None:
    """
    Register a vendor's info in the global registry.

    This should be called in each vendor's __init__.py to register itself.

    Args:
        vendor_info: VendorInfo instance to register
    """
    VENDOR_REGISTRY[vendor_info.device] = vendor_info


def get_vendor_for_device(device: str) -> Optional[VendorInfo]:
    """
    Get the VendorInfo for a given device type.

    Args:
        device: Device type (e.g., "npu", "xpu")

    Returns:
        VendorInfo if found, None otherwise
    """
    return VENDOR_REGISTRY.get(device)
