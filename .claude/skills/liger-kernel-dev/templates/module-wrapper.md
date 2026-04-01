# Module Wrapper Template

## File: `src/liger_kernel/transformers/{kernel}.py`

```python
import torch
import torch.nn as nn

from liger_kernel.ops import Liger{Kernel}Function


class Liger{Kernel}(nn.Module):
    def __init__(self, ...params):
        super().__init__()
        # Store config
        self.param = param
        # Create learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return Liger{Kernel}Function.apply(x, self.weight, ...self.params)

    def extra_repr(self):
        return f"param={self.param}, ..."
```

### Key Rules

- Import from `liger_kernel.ops` (vendor-replaceable), not from the submodule
- Match the PyTorch reference API as closely as possible (same __init__ params, same forward signature)
- Include `extra_repr` for debugging
- For loss modules, name the class `Liger{Kernel}Loss` and the forward takes `(input, target)`
- For model-specific variants, create subclasses (see `LigerRMSNormForGemma` in `rms_norm.py`)
