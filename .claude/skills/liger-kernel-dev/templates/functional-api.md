# Functional API Template

## File: `src/liger_kernel/transformers/functional.py` (MODIFY)

### Step 1: Add Import

Add at the top of the file, in alphabetical order among existing imports:

```python
from liger_kernel.ops import Liger{Kernel}Function
```

### Step 2: Add Function

Add in alphabetical order among existing functions:

```python
def liger_{kernel}(x, ...params):
    return Liger{Kernel}Function.apply(x, ...params)
```

For functions with complex return types (like CrossEntropy), wrap the output:

```python
def liger_{kernel}(input, target, ...params):
    result = Liger{Kernel}Function.apply(input, target, ...params)
    return result
```

### Key Rules

- Function signature should match the PyTorch equivalent where applicable
- Use positional args for tensor inputs, keyword args for config params
- Keep the function body minimal — just call `.apply()` and return
