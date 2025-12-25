# Ascend NPU UB Manager Design Document

## Overview

The UB Manager (Unified Buffer Manager) is a core component in **Liger-Kernel** responsible for managing the Unified Buffer (UB) capacity on Ascend NPUs. By automatically detecting UB capacity and providing best-practice-based tiling strategies, it helps Triton kernels avoid UB overflow errors while maintaining high performance.

## Design Goals

1. **Automated UB Management**: Automatically detect device UB capacity without manual configuration
2. **Best-Practice-Based**: Use proven tiling strategies to avoid UB overflow
3. **Flexible Strategy System**: Support both fixed strategies and conditional strategies to adapt to different scenarios
4. **Easy to Extend**: Simple interfaces for adding new kernel strategies
5. **Performance Optimization**: Maximize performance while ensuring UB safety through LRU caching of strategy results

## Architecture Design

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    UB Manager System                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────────┐          │
│  │  UBManager   │────────▶│ Strategy Registry│          │
│  │   (Singleton)│         │  (Best Practices)│          │
│  └──────────────┘         └──────────────────┘          │
│         │                            │                  │
│         │                            │                  │
│         │         ┌──────────────┐   │                  │
│         │         │  LRU Cache   │   │                  │
│         │         │  (Strategy   │   │                  │
│         │         │   Results)   │   │                  │
│         │         └──────────────┘   │                  │
│         │                            │                  │
│         ▼                            ▼                  │
│  ┌──────────────┐         ┌──────────────────┐          │
│  │   Capacity   │         │  Strategy        │          │
│  │  Detection   │         │  Functions       │          │
│  └──────────────┘         └──────────────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
┌──────────────┐         ┌──────────────────┐
│   GEGLU      │         │      ROPE        │
│   Kernel     │         │     Kernel       │
└──────────────┘         └──────────────────┘
```

### Class Diagram

```
┌─────────────────────────────────────┐
│          UBManager                  │
├─────────────────────────────────────┤
│ - _npu_model: str                   │
│ - _ub_capacity_bits: int            │
│ - _strategy_cache: OrderedDict      │
│ - _strategy_cache_size: int         │
├─────────────────────────────────────┤
│ + ub_capacity_bits: int             │
│ + ub_capacity_bytes: int            │
│ + npu_model: str                    │
│ + get_tiling_strategy()             │
│ - _detect_npu_model()               │
│ - _detect_ub_capacity()             │
│ - _normalize_cache_key()            │
└─────────────────────────────────────┘
```

## Core Functionality

### 1. UB Capacity Detection

The UB Manager detects UB capacity in the following priority order:

1. **Environment Variable**: `ASCEND_UB_CAPACITY_BITS`
2. **Device Properties**: Retrieved from `torch.npu.get_device_properties(0).ub_capacity_bits`
3. **Model Defaults**: Use predefined values based on the detected NPU model

```
# Default UB capacity configuration
_DEFAULT_UB_CAPACITIES = {
    "Ascend910B1": 2097152,  # ~256 KB
    "Ascend910B4": 1572864,  # ~192 KB
    "default": 2097152,       # ~256 KB
}
```

### 2. Strategy Registration System

Strategies are registered via the `_TILING_STRATEGY_BEST_PRACTICES` dictionary and support two formats:

#### Fixed Strategy

```
("kernel_name", ub_capacity_bits): (block_size, ...)
```

Returns fixed tiling parameters directly, suitable for simple scenarios.

#### Conditional Strategy

```
("kernel_name", ub_capacity_bits): strategy_function
```

The strategy function dynamically computes tiling parameters based on input arguments:

```
def strategy_function(key_params: Optional[Union[Tuple, Dict]]) -> Optional[Tuple]:
    # Compute the tiling strategy based on key_params and UB capacity
    # Return (block_size, ...) or None
```

### 3. LRU Cache System

The UB Manager implements an LRU (Least Recently Used) cache to avoid recomputing strategies for the same parameters. This significantly improves performance when the same kernel is called repeatedly with identical parameters.

**Cache Features:**
- **Default Size**: 128 entries (configurable via `strategy_cache_size` parameter)
- **Cache Key**: Normalized tuple of `(kernel_name, ub_capacity_bits, normalized_key_params)`
- **Eviction Policy**: LRU - oldest entries are evicted when cache is full
- **Key Normalization**: 
  - `None` parameters → `(kernel_name, ub_capacity_bits, None)`
  - `dict` parameters → converted to sorted tuple of items for consistent hashing
  - `tuple` parameters → used directly

**Cache Benefits:**
- Avoids redundant strategy computations
- Reduces overhead for frequently called kernels
- Maintains memory efficiency through size limits

### 4. Strategy Lookup Flow

```
User calls get_tiling_strategy()
         │
         ▼
Normalize cache key
         │
         ▼
Check LRU cache
         │
         ├─── Cache hit ────▶ Update LRU position ────▶ Return cached result
         │
         ▼
      Cache miss
         │
         ▼
Build lookup key: (kernel_name, ub_capacity_bits)
         │
         ▼
Look up in _TILING_STRATEGY_BEST_PRACTICES
         │
         ├─── Not found ────▶ Return None
         │
         ▼
      Strategy found
         │
         ├─── Fixed tuple ────▶ Cache result ────▶ Return directly
         │
         ▼
     Callable function
         │
         ▼
  Call strategy function (with key_params)
         │
         │ (Strategy function handles parameter format internally)
         │
         ▼
   Compute strategy result
         │
         ▼
Cache result (if not None)
         │
         ▼
   Return strategy result
```

## Usage Examples

### Basic Usage

```
from liger_kernel.ops.backends._ascend.ub_manager import get_tiling_strategy

# GEGLU forward with tuple parameters
strategy = get_tiling_strategy("geglu_forward", (4096, 2))
if strategy:
    block_size = strategy[0]
    # Call kernel with block_size

# GEGLU forward with dict parameters (also supported)
strategy = get_tiling_strategy("geglu_forward", {"n_cols": 4096, "dtype_size": 2})
if strategy:
    block_size = strategy[0]

# ROPE forward
strategy = get_tiling_strategy("rope_forward", (32, 32, 128, 4))
if strategy:
    BLOCK_Q, BLOCK_K = strategy
    # Call kernel with BLOCK_Q and BLOCK_K

# Subsequent calls with same parameters will use cached results
strategy = get_tiling_strategy("geglu_forward", (4096, 2))  # Cache hit!
```

### Usage Inside a Kernel

```
# GEGLU example
def geglu_forward(a, b):
    n_cols = a.shape[-1]
    dtype_size = a.element_size()
    
    # Get strategy (cached automatically)
    strategy = get_tiling_strategy("geglu_forward", (n_cols, dtype_size))
    
    if strategy is not None:
        block_size = strategy[0]
    else:
        block_size = triton.next_power_of_2(n_cols)  # Fallback
    
    # Call kernel
    kernel[(n_rows,)](a, b, c, BLOCK_SIZE=block_size)
```

### Customizing Cache Size

```
from liger_kernel.ops.backends._ascend.ub_manager import get_ub_manager, UBManager

# Create UBManager with custom cache size
ub_manager = UBManager(strategy_cache_size=256)  # Default is 128

# Or modify the global singleton
ub_manager = get_ub_manager()
# Note: Cache size is set at initialization and cannot be changed afterwards
```

## Extension Guide

### Adding a New Kernel Strategy

1. **Define the strategy function**:

```
def _my_kernel_strategy(key_params: Optional[Union[Tuple, Dict]]) -> Optional[Tuple]:
    """
    My kernel tiling strategy.
    
    Args:
        key_params: (param1, param2, dtype_size) or dict
    
    Returns:
        (block_size1, block_size2, ...) or None
    """
    if key_params is None:
        return None
    
    # Get UB capacity
    ub_manager = get_ub_manager()
    ub_capacity_bits = ub_manager.ub_capacity_bits
    
    # Extract parameters
    if isinstance(key_params, dict):
        param1 = key_params["param1"]
        param2 = key_params["param2"]
        dtype_size = key_params.get("dtype_size", 4)
    else:
        param1 = key_params[0]
        param2 = key_params[1]
        dtype_size = key_params[2] if len(key_params) > 2 else 4
    
    # Compute strategy
    SAFE_UB_CAPACITY = int(ub_capacity_bits * 0.80)
    # ... compute block_size based on memory estimation ...
    
    return (block_size1, block_size2)
```

**Register the strategy**:

```
_TILING_STRATEGY_BEST_PRACTICES = {
    # ... existing strategies ...
    
    # New strategies
    # Note: Strategies are keyed by (kernel_name, ub_capacity_bits)
    # The same strategy function can be used for different UB capacities
    ("my_kernel_forward", 1572864): _my_kernel_strategy,
    ("my_kernel_forward", 2097152): _my_kernel_strategy,  # Different UB capacity
    ("my_kernel_backward", 1572864): _my_kernel_strategy,
}
```

**Use the strategy in the kernel**:

```
def my_kernel_forward(input):
    # Prepare parameters
    param1 = input.shape[0]
    param2 = input.shape[1]
    dtype_size = 4 if input.dtype == torch.float32 else 2
    
    # Get strategy
    strategy = get_tiling_strategy("my_kernel_forward", (param1, param2, dtype_size))
    
    if strategy:
        block_size1, block_size2 = strategy
    else:
        # Fallback logic
        block_size1, block_size2 = default_sizes()
    
    # Call kernel
    kernel[(grid_size,)](
        input,
        BLOCK_SIZE1=block_size1,
        BLOCK_SIZE2=block_size2,
    )
```
