import torch

from liger_kernel.utils import infer_device_arch

# Correctly-autocast H100 end-to-end sweeps show different crossovers for
# pointwise and sequence kernels. All Liger paths remain selected at 4K+.
_HOPPER_NATIVE_SEQUENCE_CUTOFFS = {
    "rms_norm": 1536,
    "rope": 1536,
    "short_conv": 4096,
    "swiglu": 4096,
}


def _lfm2_training_sequence_length(tensor: torch.Tensor, sequence_dim: int) -> int:
    """Return sequence length for an op-specific LFM2 tensor layout."""
    return tensor.shape[sequence_dim]


def use_lfm2_native_forward(tensor: torch.Tensor, operation: str, sequence_dim: int = -2) -> bool:
    """Select native ops for inference and operation-specific Hopper shapes."""
    if not torch.is_grad_enabled():
        return True
    if tensor.device.type != "cuda" or torch.version.hip is not None:
        return False
    device_id = tensor.device.index if tensor.device.index is not None else 0
    native_sequence_cutoff = _HOPPER_NATIVE_SEQUENCE_CUTOFFS[operation]
    return (
        infer_device_arch(device_id) == "hopper"
        and _lfm2_training_sequence_length(tensor, sequence_dim) < native_sequence_cutoff
    )
