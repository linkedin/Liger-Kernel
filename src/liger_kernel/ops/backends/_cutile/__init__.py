"""
CuTile backend adapters for Liger-Kernel.

This package intentionally does not register itself in the vendor registry.
CuTile and the default Triton implementation both run on CUDA, so backend
selection is controlled explicitly via the CUTILE_BACKEND environment variable.
"""
