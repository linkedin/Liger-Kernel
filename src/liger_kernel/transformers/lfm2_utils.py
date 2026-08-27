import torch

from liger_kernel.utils import infer_device_arch

_HOPPER_NATIVE_SEQUENCE_CUTOFF = 4096


def use_lfm2_native_forward(tensor: torch.Tensor) -> bool:
    """Select native ops for inference and short Hopper training sequences."""
    if not torch.is_grad_enabled():
        return True
    if tensor.device.type != "cuda" or torch.version.hip is not None:
        return False
    device_id = tensor.device.index if tensor.device.index is not None else 0
    return (
        infer_device_arch(device_id) == "hopper"
        and tensor.shape[-2] < _HOPPER_NATIVE_SEQUENCE_CUTOFF
    )
