from dataclasses import dataclass
from typing import Optional


@dataclass
class VendorInfo:
    vendor: str
    device: str
    module_path: str


VENDOR_REGISTRY: dict[str, VendorInfo] = {}


def register_vendor(vendor_info: VendorInfo):
    VENDOR_REGISTRY[vendor_info.device] = vendor_info


def get_vendor_for_device(device: str) -> Optional[VendorInfo]:
    return VENDOR_REGISTRY.get(device, None)
