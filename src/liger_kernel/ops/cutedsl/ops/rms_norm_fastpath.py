"""Pure-Python eligibility helpers for the CuTe DSL RMSNorm vector path."""


def fast_path_vector_width(*element_sizes: int) -> int:
    """Return the common 16-byte vector width for all participating tensors."""
    largest = max(element_sizes)
    if largest <= 0 or 16 % largest:
        raise ValueError(f"element sizes must divide 16, got {element_sizes}")
    return 16 // largest


def triton_backward_warp_count(n_cols: int) -> int:
    """Mirror calculate_settings() for hidden widths supported by the fast path."""
    if n_cols > 4096:
        return 16
    if n_cols > 1024:
        return 8
    return 4
