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


def fwd_warp_count(n_cols: int, vec: int, n_rows: int | None = None, sm90: bool = False) -> int:
    """Choose forward warps from width and, on SM90, available row parallelism."""
    n_vectors = n_cols // vec
    if not sm90 or n_rows is None:
        return 8 if n_vectors >= 512 else 4

    if n_rows < 2048:
        return 4 if n_vectors <= 128 else 8
    if n_rows >= 4096 and n_cols <= 1024:
        return 2
    if n_rows >= 8192 and n_vectors <= 256:
        return 2
    return 4 if n_vectors <= 512 else 8
