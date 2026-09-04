# Adding a New Hardware Backend

This directory holds **alternative hardware backends** — operator implementations for devices other than the default (CUDA). Examples: Ascend NPU, future ROCm, future XPU.

DSL alternatives for the *default* hardware (CUDA / HIP) — cuTile, CuTe DSL,
FlyDSL, future TileLang — live at the top level of `src/liger_kernel/ops/`
(peers of this `backends/` directory), not inside it. The contract for
registering them is the same; only the on-disk location differs.

## Concepts

An **implementation** is a named alternative kernel set. Each implementation declares:

- **`name`** — identifier (e.g., `ascend`, `cutile`). Users select it via `LIGER_KERNEL_IMPL=<name>`.
- **`devices`** — every device the implementation supports.
- **`default_devices`** — the subset where it is auto-applied at import time. On supported devices not in this set, the implementation is opt-in only (requires `LIGER_KERNEL_IMPL=<name>`). Empty means opt-in only on every supported device.
- **`module_path`** — the Python module path where the kernels live (e.g., `liger_kernel.ops.cutile.ops`, `liger_kernel.ops.backends._ascend.ops`).

Two flavors fall out of the data:

- **Auto-applied** — `default_devices` includes the current device. Replaces defaults automatically (e.g., Ascend on NPU).
- **Opt-in** — only selected when the user sets `LIGER_KERNEL_IMPL=<name>` (e.g., cuTile on CUDA).

## Directory layout (full tree)

```
src/liger_kernel/ops/
├── jsd.py, rms_norm.py, ...            # default Triton-on-CUDA — the canonical kernels
├── cutile/                              # opt-in DSL on CUDA
│   ├── __init__.py                      # register_impl(ImplInfo(name="cutile", devices=("cuda",), module_path=...))
│   └── ops/
│       ├── __init__.py
│       └── jsd.py
├── backends/                            # alternative hardware backends (this directory)
│   ├── README.md
│   ├── __init__.py                      # auto-imports _<name>/ subpackages
│   ├── registry.py                      # ImplInfo, register_impl(), select_impl(), IMPL_REGISTRY
│   └── _ascend/                         # Ascend NPU — auto-applied on NPU
│       ├── __init__.py                  # register_impl(ImplInfo(name="ascend", ...))
│       └── ops/
│           ├── __init__.py
│           └── jsd.py, ...
└── __init__.py                          # imports defaults, runs _replace_with_impl_ops()
```

## How dispatch works

1. `liger_kernel.ops` is imported. Default top-level kernels (`ops/jsd.py`, etc.) load first.
2. `_discover_impls()` runs:
   - Imports `liger_kernel.ops.backends`, which auto-imports each `_<name>/` subpackage. Each subpackage's `__init__.py` calls `register_impl()`.
   - Iterates the top-level non-private subpackages of `ops/` (e.g., `cutile/`), excluding reserved dirs (`backends`, `experimental`), and imports each. Same self-registration pattern.
3. `_replace_with_impl_ops()` runs:
   - Detects the current device via `infer_device()`.
   - Reads `LIGER_KERNEL_IMPL` from the environment.
   - Calls `select_impl(device, explicit=<env var or None>)`:
     - If the env var is set, the named implementation is looked up and validated (device must be in `devices`).
     - If unset, the first registered implementation listing the current device in its `default_devices` is returned; otherwise no replacement happens.
   - If an implementation was selected, its operators replace/extend the `liger_kernel.ops` namespace.

If an auto-selected implementation fails to import (e.g., the vendor SDK isn't installed), the dispatcher silently falls back to defaults. An explicitly-requested implementation that fails to import re-raises so the user sees the underlying error.

## Adding a new hardware backend (lives in `backends/_<name>/`)

### Step 1: Create the directory

```bash
mkdir -p src/liger_kernel/ops/backends/_<name>/ops
touch src/liger_kernel/ops/backends/_<name>/__init__.py
touch src/liger_kernel/ops/backends/_<name>/ops/__init__.py
```

### Step 2: Register the implementation

In `backends/_<name>/__init__.py`:

```python
"""<Name> hardware backend for Liger-Kernel."""

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

# Auto-applied on the listed devices:
register_impl(ImplInfo(
    name="<name>",
    devices=("<device>",),
    default_devices=("<device>",),
    module_path=f"{__name__}.ops",
))
```

### Step 3: Ensure device detection works

Make sure `infer_device()` in `liger_kernel/utils.py` recognizes the device. Example:

```python
def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    if is_<device>_available():
        return "<device>"
    return "cpu"
```

### Step 4: Implement the operators

Create operator files in `backends/_<name>/ops/`. For example, `geglu.py`:

```python
import torch

class LigerGELUMulFunction(torch.autograd.Function):
    """Backend-specific LigerGELUMulFunction."""

    @staticmethod
    def forward(ctx, a, b):
        ...

    @staticmethod
    def backward(ctx, dc):
        ...
```

### Step 5: Export from `ops/__init__.py`

In `backends/_<name>/ops/__init__.py`:

```python
"""<Name>-specific operator implementations."""

from .geglu import LigerGELUMulFunction
from .geglu import geglu_backward
from .geglu import geglu_forward

__all__ = [
    "LigerGELUMulFunction",
    "geglu_forward",
    "geglu_backward",
]
```

## Adding a new DSL implementation (lives at top level of `ops/`)

The pattern is the same — only the on-disk location and the `module_path` differ:

```
src/liger_kernel/ops/<name>/
├── __init__.py                          # register_impl(...)
└── ops/
    ├── __init__.py                      # exports symbols
    └── jsd.py, ...                      # kernel files
```

```python
# ops/<name>/__init__.py

from liger_kernel.ops.backends.registry import ImplInfo
from liger_kernel.ops.backends.registry import register_impl

# Opt-in only (no `default_devices`):
register_impl(ImplInfo(
    name="<name>",
    devices=("cuda",),
    module_path=f"{__name__}.ops",  # liger_kernel.ops.<name>.ops
))
```

## Key points

### Incremental override

You **don't need to implement all operators**. Only implement the ones that need a different version. Unimplemented operators fall back to the defaults.

### Adding new operators

An implementation can also **add new operators** that don't exist in the defaults. They are exported to `liger_kernel.ops` for users to import.

### Naming convention

- Use the **same class/function names** as the defaults when overriding — lets user code stay unchanged.
- Use `as` imports to rename if your internal naming differs.

### Multi-device implementations

An implementation can support multiple devices by listing them all in `devices`. It can be the default on a subset (or none) of them.

```python
# Supports CUDA and XPU; default on neither (opt-in everywhere):
register_impl(ImplInfo(
    name="inductor",
    devices=("cuda", "xpu"),
    module_path="liger_kernel.ops.inductor.ops",
))

# Supports CUDA and XPU; auto-applied on XPU only:
register_impl(ImplInfo(
    name="example",
    devices=("cuda", "xpu"),
    default_devices=("xpu",),
    module_path="liger_kernel.ops.example.ops",
))
```

## Examples in this repo

- `backends/_ascend/` — auto-applied hardware backend (Ascend NPU).
- `../cutile/` — opt-in DSL implementation on CUDA. Enable with:
  ```bash
  LIGER_KERNEL_IMPL=cutile python your_script.py
  ```
  Ops include CE, fused linear CE (BT-chunked), JSD / fused linear JSD, GeGLU, LayerNorm.
- `../cutedsl/` — opt-in NVIDIA CuTe DSL implementation. Enable with:
  ```bash
  LIGER_KERNEL_IMPL=cutedsl python your_script.py
  ```
  Ops include CE and fused linear CE (BT-chunked; never materializes full `BT×V`).
- `../flydsl/` — opt-in ROCm FlyDSL implementation (AMD). Enable with:
  ```bash
  LIGER_KERNEL_IMPL=flydsl python your_script.py
  ```
  Ops include CE and fused linear CE (BT-chunked). Unimplemented ops fall back to Triton.

`select_impl()` validates the request: if the current device isn't in the implementation's `devices`, or its module fails to import, the user gets a clear error instead of a silent fallback.
