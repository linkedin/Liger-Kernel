"""CuTe DSL (CUTLASS python) kernel implementations for Liger ops.

This package holds kernels written using NVIDIA's CUTLASS CuTe DSL
(``from cutlass import cute``). The DSL compiles via ``cute.compile(...)`` and
targets Hopper (sm_90) and newer.

Files here are imported on demand by the dispatcher (lazy discovery in
:mod:`liger_kernel.backends.registry`); nothing here is loaded at
``import liger_kernel`` time.

Preference ranking (lower wins) — based on measured perf on B200 sm_100:

- ``cutedsl`` (preference_rank=10) — wins on backward 1.4-2.2x across the
  shape sweep; ties on forward. Auto-pick when ``cutlass.cute`` is usable.
- ``triton`` (preference_rank=50) — universal fallback; ties cuTeDSL on fwd,
  loses on bwd. Always runs (no hardware/compiler gate).
- ``cutile`` (preference_rank=80) — Blackwell-only; loses to both Triton and
  CuTeDSL on every measured shape (matches NVIDIA's own pptx for LayerNorm).
  Kept because it still works for hidden_dim > 32K where cuTeDSL is range-
  capped. Users must request explicitly via ``impl="nvidia-cutile"``.

Reference implementations to draw from when porting an op:

- ``Dao-AILab/flash-attention`` (hopper/ directory) — high-perf Hopper kernels
  (FA fwd/bwd, layer_norm, softmax) implemented in CuTe (C++/CUDA in csrc/,
  Python wrappers via ``cute.compile``).
  https://github.com/Dao-AILab/flash-attention
- ``Dao-AILab/quack`` — small inline CuTe DSL kernels for rmsnorm, softmax,
  cross_entropy. Read these for ``ReductionBase``, ``TiledCopy`` + ``cp_async``
  patterns.
  https://github.com/Dao-AILab/quack
- ``NVIDIA/cutlass`` (examples/35_gemm_softmax and
  61_hopper_gemm_with_topk_and_softmax) — CUTLASS canonical patterns.
  https://github.com/NVIDIA/cutlass
"""
