# Adding a New Vendor Backend

This directory contains vendor-specific operator implementations that automatically replace the default (CUDA) implementations when running on the corresponding device.

## Concepts

- **Vendor**: Chip manufacturer (e.g., `ascend`, `intel`, `nvidia`)
- **Device**: Device type (e.g., `npu`, `xpu`, `cuda`)
- **VendorInfo**: Defines the mapping between vendor and device

## Directory Structure

```
backends/
├── README.md          
├── __init__.py         
├── registry.py         # VendorInfo, register_vendor(), VENDOR_REGISTRY
├── _ascend/            # Ascend (Huawei) vendor - supports NPU
│   ├── __init__.py     # Registers VendorInfo for NPU
│   └── ops/
│       ├── __init__.py # Exports vendor-specific implementations
│       └── geglu.py    # NPU-specific GEGLU implementation
└── _<vendor>/          # Your new vendor backend
    └── ...
```

## How It Works

1. When `liger_kernel.ops.backends` is imported, it imports all vendor packages (e.g., `_ascend`)
2. Each vendor's `__init__.py` calls `register_vendor()` to register itself
3. When `liger_kernel.ops` is imported, `_replace_with_vendor_ops()` is called
4. It detects the current device via `infer_device()` and looks up the vendor
5. Vendor implementations replace/add to the `liger_kernel.ops` namespace

## Adding a New Vendor

### Step 1: Create Directory Structure

```bash
mkdir -p backends/_<vendor>/ops
touch backends/_<vendor>/__init__.py
touch backends/_<vendor>/ops/__init__.py
```

### Step 2: Register Your Vendor

In `backends/_<vendor>/__init__.py`, register your vendor:

```python
"""
<Vendor> backend for Liger-Kernel.
"""

from liger_kernel.ops.backends.registry import VendorInfo, register_vendor

register_vendor(
    VendorInfo(
        vendor="<vendor>",
        device="<device>",
    )
)
```


### Step 3: Ensure Device Detection Works

Make sure `infer_device()` in `liger_kernel/utils.py` can detect your device:

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

### Step 4: Implement Vendor-Specific Operators

Create operator files in `backends/_<vendor>/ops/`. For example, `geglu.py`:

```python
import torch

class LigerGELUMulFunction(torch.autograd.Function):
    """
    Vendor-specific LigerGELUMulFunction implementation.
    """
    @staticmethod
    def forward(ctx, a, b):
        # Your vendor-specific forward implementation
        ...

    @staticmethod
    def backward(ctx, dc):
        # Your vendor-specific backward implementation
        ...

# Optional: vendor-specific kernel functions
def geglu_forward_vendor(a, b):
    ...

def geglu_backward_vendor(a, b, dc):
    ...
```

### Step 5: Export in `ops/__init__.py`

In `backends/_<vendor>/ops/__init__.py`, export your implementations:

```python
"""
<Vendor>-specific operator implementations.
"""

from .<module> import (
    LigerGELUMulFunction,
    geglu_forward_vendor as geglu_forward,   # Rename to match default API
    geglu_backward_vendor as geglu_backward,
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

You **don't need to implement all operators**. Only implement the ones that require vendor-specific adaptations. Unimplemented operators will automatically fall back to the default (CUDA) implementation.

### Vendor-Specific Additions

Vendors can also **add new operators** that don't exist in the default implementation. These will be exported to `liger_kernel.ops` namespace for users to import.

### Naming Convention

- Use the **same class/function names** as the default implementations for overrides
- This allows seamless replacement without changing user code
- Use `as` imports to rename if your internal naming differs

## Example: Ascend NPU Backend

See `_ascend/` directory for a complete example of the Ascend NPU backend implementation.
