"""Triton-DSL kernel implementations for Liger ops.

This package holds wrappers that register Liger's existing Triton kernels with
the multi-DSL backend registry (:mod:`liger_kernel.backends`). The actual
kernel source code still lives in ``liger_kernel.ops.<op>`` so that direct
imports of the historical paths keep working unchanged.

Files here are imported on demand by the dispatcher; nothing here is loaded
at ``import liger_kernel`` time.
"""
