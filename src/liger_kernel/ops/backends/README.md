# Adding a New Backend

This directory contains backend-specific operator implementations that can replace the default Liger implementations.

A **backend** here is a named alternative implementation of Liger's operators. It may target a different hardware device (e.g., Ascend NPU vs. NVIDIA CUDA) or a different DSL on the same device (e.g., cuTile vs. Triton on CUDA), and may support one or more devices.

Each backend declares two device sets:

- **`devices`** — every device the backend supports.
- **`default_devices`** — the subset on which the backend is auto-applied at import time. On supported devices not listed here, the backend is opt-in only and must be requested explicitly via the `LIGER_KERNEL_BACKEND` environment variable.

## Concepts

- **Device**: PyTorch device type returned by `infer_device()` (e.g., `cuda`, `npu`, `xpu`)
- **BackendInfo**: Declarative description of a backend (name, supported devices, default devices)
- **Auto-applied backend**: A backend whose `default_devices` includes the current device — applied automatically (e.g., Ascend on NPU)
- **Opt-in backend**: A backend whose `default_devices` is empty (or excludes the current device) — applied only when `LIGER_KERNEL_BACKEND=<name>` is set (e.g., cuTile on CUDA)

## Directory Structure

```
backends/
├── README.md
├── __init__.py
├── registry.py          # BackendInfo, register_backend(), BACKEND_REGISTRY, select_backend()
├── _ascend/             # Ascend backend — auto-applied on NPU
│   ├── __init__.py      # register_backend(BackendInfo(name="ascend", devices=("npu",), default_devices=("npu",)))
│   └── ops/
│       ├── __init__.py  # Exports backend-specific implementations
│       └── geglu.py     # NPU-specific GEGLU implementation
├── _cutile/             # cuTile backend — opt-in on CUDA
│   ├── __init__.py      # register_backend(BackendInfo(name="cutile", devices=("cuda",)))
│   └── ops/
│       └── ...
└── _<name>/             # Your new backend
    └── ...
```

## How It Works

1. When `liger_kernel.ops.backends` is imported, every `_<name>/` subpackage is auto-imported.
2. Each backend's `__init__.py` calls `register_backend()` to register itself.
3. When `liger_kernel.ops` is imported, `_replace_with_backend_ops()` is called.
4. It detects the current device via `infer_device()` and reads `LIGER_KERNEL_BACKEND` from the environment.
5. It calls `select_backend(device, explicit=...)`:
   - If `LIGER_KERNEL_BACKEND` is **set**, the named backend is selected (validated against the current device).
   - If `LIGER_KERNEL_BACKEND` is **unset**, the first registered backend that lists the current device in its `default_devices` is selected; otherwise the defaults are kept.
6. The selected backend's operators replace/extend the symbols in the `liger_kernel.ops` namespace.

If an auto-selected backend fails to import (e.g., the vendor SDK isn't installed), the dispatcher silently falls back to defaults. An explicitly-requested backend that fails to import re-raises so the user sees the underlying error.

## Adding a New Backend

### Step 1: Create the directory structure

```bash
mkdir -p backends/_<name>/ops
touch backends/_<name>/__init__.py
touch backends/_<name>/ops/__init__.py
```

### Step 2: Register your backend

In `backends/_<name>/__init__.py`:

```python
"""
<Name> backend for Liger-Kernel.
"""

from liger_kernel.ops.backends.registry import BackendInfo
from liger_kernel.ops.backends.registry import register_backend

# Auto-applied on the listed devices:
register_backend(BackendInfo(name="<name>", devices=("<device>",), default_devices=("<device>",)))

# Or, opt-in only (selected via LIGER_KERNEL_BACKEND=<name>):
# register_backend(BackendInfo(name="<name>", devices=("<device>",)))
```

### Step 3: Ensure device detection works

Make sure `infer_device()` in `liger_kernel/utils.py` recognizes your device:

```python
def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    # Add your device detection here
    if is_<device>_available():
        return "<device>"
    return "cpu"
```

### Step 4: Implement backend-specific operators

Create operator files in `backends/_<name>/ops/`. For example, `geglu.py`:

```python
import torch

class LigerGELUMulFunction(torch.autograd.Function):
    """Backend-specific LigerGELUMulFunction implementation."""

    @staticmethod
    def forward(ctx, a, b):
        ...

    @staticmethod
    def backward(ctx, dc):
        ...

def geglu_forward_backend(a, b):
    ...

def geglu_backward_backend(a, b, dc):
    ...
```

### Step 5: Export in `ops/__init__.py`

In `backends/_<name>/ops/__init__.py`:

```python
"""<Name>-specific operator implementations."""

from .<module> import (
    LigerGELUMulFunction,
    geglu_forward_backend as geglu_forward,   # Rename to match default API
    geglu_backward_backend as geglu_backward,
)

# Explicitly declare what to export (recommended)
__all__ = [
    "LigerGELUMulFunction",
    "geglu_forward",
    "geglu_backward",
]
```

## Key Points

### Incremental Override

You **don't need to implement all operators**. Only implement the ones that require backend-specific adaptations. Unimplemented operators automatically fall back to the default implementation.

### Backend-Specific Additions

Backends can also **add new operators** that don't exist in the default implementation. These will be exported to the `liger_kernel.ops` namespace for users to import.

### Naming Convention

- Use the **same class/function names** as the default implementations for overrides
- This allows seamless replacement without changing user code
- Use `as` imports to rename if your internal naming differs

### Multi-Device Backends

A backend can support multiple devices by listing them all in `devices`. It can be the default on a subset (or none) of them. Examples:

```python
# Supports CUDA and XPU; default on neither (opt-in everywhere):
register_backend(BackendInfo(name="inductor", devices=("cuda", "xpu")))

# Supports CUDA and XPU; auto-applied on XPU only:
register_backend(BackendInfo(name="example", devices=("cuda", "xpu"), default_devices=("xpu",)))
```

## Example: Ascend NPU Backend

See `_ascend/` for a complete example of an auto-applied backend.

## Example: cuTile Backend

See `_cutile/` for a complete example of an opt-in backend.

Enable it on a CUDA device with:

```bash
LIGER_KERNEL_BACKEND=cutile python your_script.py
```

`select_backend()` validates the request: if the current device isn't in the backend's `devices`, or if `cuda-tile` isn't installed, the user gets a clear error instead of a silent fallback.
