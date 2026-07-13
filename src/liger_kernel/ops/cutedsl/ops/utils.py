"""
Shared helpers for the CuTe DSL backend ops.
"""

import functools

import torch

from cutlass.cute.runtime import from_dlpack

from liger_kernel.utils import infer_device

# NVIDIA: CUDA compute capability (major, minor) -> coarse arch family (from
# github.com/linkedin/Liger-Kernel PR #1273 `infer_device_arch`).
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

# AMD: gfx target (gcnArchName) -> coarse arch family.
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
    """Get a coarse architecture/generation name for the current device.

    Returns a family name when detectable, falling back to the device type from ``infer_device()``
    (e.g. ``"cpu"``) otherwise:

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


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def to_cute_tensor(t, leading_dim=None, assumed_align=16):
    """torch.Tensor -> cute.Tensor via DLPack, with a dynamic (runtime) layout."""
    if t is None:
        return None
    ct = from_dlpack(t.detach(), assumed_align=assumed_align)
    ld = (t.ndim - 1) if leading_dim is None else leading_dim
    return ct.mark_layout_dynamic(leading_dim=ld)
