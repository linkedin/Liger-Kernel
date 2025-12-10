"""
Ascend NPU operator implementations.

This module exports Ascend NPU-optimized implementations that will automatically
replace the default implementations when running on NPU devices.

Both Function classes and kernel functions can be exported here.

To add a new operator:
1. Create the implementation file (e.g., rms_norm.py)
2. Import the Function class and/or kernel functions here
3. Optionally add to __all__ for explicit control

If __all__ is not defined, all public symbols will be auto-discovered.
"""
