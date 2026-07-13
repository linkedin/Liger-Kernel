try:
    import peft  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import functools

import torch


def is_peft_available():
    return PEFT_AVAILABLE


def infer_comm_backend():
    """
    Get communication backend name based on the environment.
    """
    if torch.distributed.is_nccl_available():
        # Works for Nvidia
        # TODO: nccl may not work for AMD decices that may require use of rccl.
        return "nccl"
    elif is_npu_available():
        # Use Ascend NPU if available (torch.npu)
        # Ascend is not standard torch backend and requires extension.
        # Assume that it is installed if NPUs are being used in
        # multi device environment.
        return "ascend"
    # XPU (Intel) if available
    elif torch.distributed.distributed_c10d.is_xccl_available():
        return "xccl"
    elif torch.distributed.is_mpi_available():
        # CPU backend, first option
        return "mpi"
    elif torch.distributed.is_gloo_available():
        # CPU backend, backup option
        return "gloo"
    else:
        raise RuntimeError("There is no distributed backend available.")


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    # Use Ascend NPU if available (torch.npu)
    elif is_npu_available():
        return "npu"
    # XPU (Intel) if available
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    try:
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        return False


# NVIDIA: CUDA compute capability (major, minor) -> coarse arch family
_NVIDIA_ARCH_BY_CC = {
    (7, 0): "volta_turing",  # Volta V100
    (7, 5): "volta_turing",  # Turing T4 / RTX 20xx
    (8, 0): "ampere_ada",  # Ampere A100
    (8, 6): "ampere_ada",  # Ampere RTX 30xx / A40
    (8, 9): "ampere_ada",  # Ada Lovelace RTX 40xx / L4 / L40
    (9, 0): "hopper",  # H100 / H200
    (10, 0): "blackwell",  # B100 / B200 / GB200 (sm_100)
    (10, 3): "blackwell_ultra",  # B300 / GB300 (sm_103)
    (12, 0): "blackwell_consumer",  # RTX 50xx (sm_120)
}

# AMD: gfx target (gcnArchName) -> coarse arch family
_AMD_ARCH_BY_GFX = {
    "gfx908": "cdna",  # MI100
    "gfx90a": "cdna2",  # MI200
    "gfx940": "cdna3",  # MI300
    "gfx941": "cdna3",
    "gfx942": "cdna3",  # MI300X/MI300A
    "gfx1100": "rdna3",  # RX 7900
    "gfx1101": "rdna3",
    "gfx1102": "rdna3",
}


def _infer_nvidia_arch(device_id: int) -> str:
    major, minor = torch.cuda.get_device_capability(device_id)
    return _NVIDIA_ARCH_BY_CC.get((major, minor), f"sm_{major}{minor}")


def _infer_amd_arch(device_id: int) -> str:
    # gcnArchName looks like "gfx942:sramecc+:xnack-"; keep the gfx target only.
    gfx = getattr(torch.cuda.get_device_properties(device_id), "gcnArchName", "").split(":")[0]
    return _AMD_ARCH_BY_GFX.get(gfx, gfx or "cuda")


def _infer_xpu_arch(device_id: int) -> str:
    name = torch.xpu.get_device_properties(device_id).name.lower()
    if any(tag in name for tag in ("max", "pvc", "ponte")):
        return "pvc"  # Ponte Vecchio / Data Center GPU Max
    if any(tag in name for tag in ("arc", "battlemage", "alchemist")):
        return "arc"
    return "xpu"


def _infer_npu_arch(device_id: int) -> str:
    name = torch.npu.get_device_properties(device_id).name.lower()
    if "910" in name:
        return "ascend910"
    if "310" in name:
        return "ascend310"
    return "npu"


@functools.lru_cache(maxsize=None)
def infer_device_arch(device_id: int = 0) -> str:
    """
    Get a coarse architecture/generation name for the current device.

    Returns a family name when detectable, falling back to the device type
    from ``infer_device()`` (e.g. ``"cpu"``) otherwise:

      - NVIDIA: ``"volta_turing"``, ``"ampere_ada"``, ``"hopper"``, ``"blackwell"``,
                ``"blackwell_ultra"``, ``"blackwell_consumer"`` (else ``"sm_<major><minor>"``)
      - AMD:    ``"cdna"``, ``"cdna2"``, ``"cdna3"``, ``"rdna3"`` (else the raw gfx target)
      - Intel:  ``"pvc"``, ``"arc"`` (else ``"xpu"``)
      - Ascend: ``"ascend910"``, ``"ascend310"`` (else ``"npu"``)

    The result is cached; call ``infer_device_arch.cache_clear()`` to reset.
    """
    device = infer_device()
    try:
        if device == "cuda":
            # ROCm reports as "cuda" in torch; torch.version.hip distinguishes AMD.
            return _infer_amd_arch(device_id) if torch.version.hip else _infer_nvidia_arch(device_id)
        if device == "xpu":
            return _infer_xpu_arch(device_id)
        if device == "npu":
            return _infer_npu_arch(device_id)
    except Exception:
        return device
    return device


def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        return before_fn(*before_args, **before_kwargs)
    else:
        return after_fn(*after_args, **after_kwargs)


def get_total_gpu_memory() -> int:
    """Returns total GPU memory in GBs."""
    device = infer_device()
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory // (1024**3)
    elif device == "xpu":
        return torch.xpu.get_device_properties(0).total_memory // (1024**3)
    elif device == "npu":
        return torch.npu.get_device_properties(0).total_memory // (1024**3)
    else:
        raise RuntimeError(f"Unsupported device: {device}")
