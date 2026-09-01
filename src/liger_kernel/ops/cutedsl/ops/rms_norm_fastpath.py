"""Pure-Python eligibility helpers for the CuTe DSL RMSNorm vector path."""


def fast_path_vector_width(*element_sizes: int) -> int:
    """Return the common 16-byte vector width for all participating tensors."""
    largest = max(element_sizes)
    if largest <= 0 or 16 % largest:
        raise ValueError(f"element sizes must divide 16, got {element_sizes}")
    return 16 // largest


def backward_warp_count(n_cols: int) -> int:
    """Mirror calculate_settings() for hidden widths supported by the fast path."""
    if n_cols > 4096:
        return 16
    if n_cols > 1024:
        return 8
    return 4


def fwd_warp_count(n_cols: int, vec: int) -> int:
    """Width-aware warp count for the vector forward kernel.

    ``vec`` is the number of elements per 16-byte load, so ``n_cols // vec`` is the
    number of vectors in a row. Wide rows (>= 512 vectors, e.g. bf16 hidden >= 4096)
    profit from 8 warps: spreading the row over twice the threads halves the
    register-resident vector tiles per thread, which lifts occupancy from ~55% to
    ~88% and hides the DRAM-load latency that bounds this memory-bound kernel
    (measured 4-7% faster at bf16 hidden 4096/8192, and faster still on tall inputs).
    Narrow rows keep 4 warps: 8 warps would leave half the CTA idle (fewer vectors
    than threads) and make the cross-warp reduction the bottleneck (up to ~30% slower
    at bf16 hidden 1024/2048).
    """
    return 8 if (n_cols // vec) >= 512 else 4
