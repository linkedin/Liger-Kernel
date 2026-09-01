import torch

from liger_kernel.utils import infer_device_arch

# Correctly-autocast H100 end-to-end sweeps show native pointwise ops winning
# below 4K for SwiGLU and short convolution. Liger RoPE and RMSNorm still
# improve this regime, while all Liger paths remain useful at 4K+.
_HOPPER_NATIVE_SEQUENCE_CUTOFF = 4096
_HOPPER_NATIVE_OPERATIONS = frozenset({"short_conv", "swiglu"})


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
    return (
        infer_device_arch(device_id) == "hopper"
        and operation in _HOPPER_NATIVE_OPERATIONS
        and _lfm2_training_sequence_length(tensor, sequence_dim) < _HOPPER_NATIVE_SEQUENCE_CUTOFF
    )
