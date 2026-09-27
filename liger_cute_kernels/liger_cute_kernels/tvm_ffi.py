"""TVM FFI facade for the LigerCute native core.

This module is the Python facade used by ``liger_kernel.ops.cute``. It loads the
torch-free native core through TVM FFI and passes tensors through DLPack views.
"""

from __future__ import annotations

import ctypes
import importlib.util

from pathlib import Path

import torch

_MOD = None
_NVSHMEM_LIBS_LOADED = False
_ARCH_CORE_NAMES = {
    9: "libliger_cute_kernels_sm90a.so",
    10: "libliger_cute_kernels_sm100f.so",
}


def is_available() -> bool:
    try:
        _load_module()
        return True
    except (FileNotFoundError, ImportError, RuntimeError, OSError):
        return False


def _load_module():
    global _MOD
    if _MOD is None:
        import tvm_ffi

        pkg_dir = Path(__file__).resolve().parent
        _load_nvshmem_libraries(pkg_dir)
        module = _select_core_module(pkg_dir)
        _MOD = tvm_ffi.load_module(str(module))
    return _MOD


def _select_core_module(pkg_dir: Path) -> Path:
    legacy = pkg_dir / "libliger_cute_kernels.so"
    architecture_cores = {
        major: pkg_dir / name for major, name in _ARCH_CORE_NAMES.items() if (pkg_dir / name).is_file()
    }
    if not architecture_cores:
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"Missing packaged TVM FFI modules under {pkg_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("selecting an architecture-specific Liger core requires a CUDA device")
    major = torch.cuda.get_device_capability()[0]
    module = architecture_cores.get(major)
    if module is None:
        supported = ", ".join(
            _ARCH_CORE_NAMES[value].removeprefix("libliger_cute_kernels_").removesuffix(".so")
            for value in sorted(architecture_cores)
        )
        raise RuntimeError(f"no Liger core for CUDA capability SM{major}; packaged capabilities: {supported}")
    return module


def _nvshmem_library_dirs(pkg_dir: Path) -> list[Path]:
    directories = [pkg_dir]
    try:
        spec = importlib.util.find_spec("nvidia.nvshmem")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        locations = list(spec.submodule_search_locations or [])
        if spec.origin:
            locations.append(str(Path(spec.origin).resolve().parent))
        directories.extend(Path(location).resolve() / "lib" for location in locations)
    return directories


def _load_nvshmem_libraries(pkg_dir: Path) -> None:
    global _NVSHMEM_LIBS_LOADED
    if _NVSHMEM_LIBS_LOADED:
        return
    directories = _nvshmem_library_dirs(pkg_dir)
    host = next(
        (
            directory / name
            for directory in directories
            for name in ("libnvshmem_host.so.3", "libnvshmem_host.so")
            if (directory / name).is_file()
        ),
        None,
    )
    if host is None:
        raise ImportError(
            "NVSHMEM runtime not found; install 'liger-cute-kernels[cu12]' or "
            "'liger-cute-kernels[cu13]' to match the CUDA version used to build the native wheel"
        )
    ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
    uid_bootstrap = host.parent / "nvshmem_bootstrap_uid.so.3"
    if uid_bootstrap.is_file():
        ctypes.CDLL(str(uid_bootstrap), mode=ctypes.RTLD_GLOBAL)
    _NVSHMEM_LIBS_LOADED = True


def _int64_out() -> torch.Tensor:
    return torch.empty(1, dtype=torch.int64, device="cpu")


def _int32_out() -> torch.Tensor:
    return torch.empty(1, dtype=torch.int32, device="cpu")


def uniqueid_nbytes() -> int:
    out = _int64_out()
    _load_module().uniqueid_nbytes(out)
    return int(out.item())


def get_uniqueid(buf_ptr: int) -> None:
    _load_module().get_uniqueid(int(buf_ptr))


def init_with_uniqueid(rank: int, nranks: int, buf_ptr: int) -> None:
    _load_module().init_with_uniqueid(int(rank), int(nranks), int(buf_ptr))


def init_pmi() -> None:
    _load_module().init_pmi()


def finalize() -> None:
    _load_module().finalize()


def my_pe() -> int:
    out = _int32_out()
    _load_module().my_pe(out)
    return int(out.item())


def n_pes() -> int:
    out = _int32_out()
    _load_module().n_pes(out)
    return int(out.item())


def team_world() -> int:
    out = _int64_out()
    _load_module().team_world(out)
    return int(out.item())


def team_split_strided(parent: int, start: int, stride: int, size: int) -> int:
    out = _int64_out()
    _load_module().team_split_strided(int(parent), int(start), int(stride), int(size), out)
    return int(out.item())


def team_destroy(team_handle: int) -> None:
    _load_module().team_destroy(int(team_handle))


def team_my_pe(team_handle: int) -> int:
    out = _int32_out()
    _load_module().team_my_pe(int(team_handle), out)
    return int(out.item())


def team_n_pes(team_handle: int) -> int:
    out = _int32_out()
    _load_module().team_n_pes(int(team_handle), out)
    return int(out.item())


def team_translate_pe(src_team: int, src_pe: int, dst_team: int) -> int:
    out = _int32_out()
    _load_module().team_translate_pe(int(src_team), int(src_pe), int(dst_team), out)
    return int(out.item())


def pool_clear_all() -> None:
    _load_module().pool_clear_all()


def pool_clear_buffers() -> None:
    _load_module().pool_clear_buffers()


def moe_configure_symmetric(
    max_tokens: int,
    hidden_dim: int,
    max_num_experts: int,
    max_top_k: int,
    num_pes: int,
    num_hosts: int,
    gpus_per_host: int,
) -> None:
    _load_module().moe_configure_symmetric(
        int(max_tokens),
        int(hidden_dim),
        int(max_num_experts),
        int(max_top_k),
        int(num_pes),
        int(num_hosts),
        int(gpus_per_host),
    )


def moe_pop_fwd() -> None:
    _load_module().moe_pop_fwd()


def _moe_symm_config() -> torch.Tensor:
    out = torch.empty(7, dtype=torch.int32, device="cpu")
    _load_module().moe_get_symm_config(out)
    if int(out[6].item()) == 0:
        raise RuntimeError("liger_cute: call moe_configure_symmetric before moe_fused_fwd_bf16")
    return out


def moe_fused_fwd_bf16(
    X: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    all_B: torch.Tensor,
    all_C: torch.Tensor,
    all_A: torch.Tensor,
    num_experts: int,
    top_k: int,
    team_handle: int,
):
    cfg = _moe_symm_config()
    num_tokens, hidden_dim = X.shape
    max_total_slots = int(cfg[0].item())
    max_m_tiles = (max_total_slots + 127) // 128
    Y = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device=X.device)
    token_expert_slots = torch.empty((max_total_slots,), dtype=torch.int32, device=X.device)
    tile_expert_ids = torch.empty((max_m_tiles,), dtype=torch.int32, device=X.device)
    symm_meta = torch.empty(17, dtype=torch.int64, device="cpu")
    _load_module().moe_fused_fwd_bf16(
        X,
        expert_indices,
        expert_weights,
        all_B,
        all_C,
        all_A,
        int(num_experts),
        int(top_k),
        int(team_handle),
        Y,
        token_expert_slots,
        tile_expert_ids,
        symm_meta,
    )
    chosen_tile_m = int(symm_meta[16].item())
    return Y, symm_meta, symm_meta, symm_meta, token_expert_slots, tile_expert_ids, chosen_tile_m


def moe_fused_bwd_bf16(
    dY: torch.Tensor,
    y_buf_meta: torch.Tensor,
    x_sorted_meta: torch.Tensor,
    token_expert_slots: torch.Tensor,
    tile_expert_ids: torch.Tensor,
    expert_offsets_meta: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    all_B: torch.Tensor,
    all_C: torch.Tensor,
    all_A: torch.Tensor,
    num_experts: int,
    top_k: int,
    team_handle: int,
    fwd_tile_m: int,
):
    del y_buf_meta, expert_offsets_meta, fwd_tile_m
    dX = torch.empty_like(dY)
    dB = torch.empty_like(all_B)
    dC = torch.empty_like(all_C)
    dA = torch.empty_like(all_A)
    dW = torch.empty_like(expert_weights)
    _load_module().moe_fused_bwd_bf16(
        dY,
        x_sorted_meta,
        token_expert_slots,
        tile_expert_ids,
        expert_indices,
        expert_weights,
        all_B,
        all_C,
        all_A,
        int(num_experts),
        int(top_k),
        int(team_handle),
        dX,
        dB,
        dC,
        dA,
        dW,
    )
    return dX, dB, dC, dA, dW


def fused_linear_scaled_cross_entropy_configure_forward(
    max_tokens: int,
    max_local_vocab: int,
) -> None:
    """Reserve reusable forward workspace for the maximum local problem.

    This is a collective configuration call across the NVSHMEM team used by
    subsequent forward launches. Capacities are immutable until the shared
    buffer pool is cleared, so every rank must pass identical values.
    """
    _load_module().fused_linear_scaled_cross_entropy_configure_forward(
        int(max_tokens),
        int(max_local_vocab),
    )


def fused_linear_scaled_cross_entropy_configure_backward(
    max_tokens: int,
    max_hidden: int,
    max_local_vocab: int,
    max_tiles_per_reduce: int,
    team_handle: int,
) -> None:
    """Reserve backward/NVLS/remote-ring workspace for an NVSHMEM TP team.

    All PEs in ``team_handle`` must call this with identical maxima before the
    first forward or backward launch. ``max_tiles_per_reduce`` must cover every
    later ``tiles_per_reduce`` request. Multi-host teams must have uniform
    per-host membership and host-major team-rank ordering.
    """
    _load_module().fused_linear_scaled_cross_entropy_configure_backward(
        int(max_tokens),
        int(max_hidden),
        int(max_local_vocab),
        int(max_tiles_per_reduce),
        int(team_handle),
    )


def fused_linear_scaled_cross_entropy_forward_workspace_bytes(
    max_tokens: int,
    max_local_vocab: int,
) -> int:
    """Return the reusable native forward workspace at the given capacity."""
    return int(
        _load_module().fused_linear_scaled_cross_entropy_forward_workspace_bytes(
            int(max_tokens),
            int(max_local_vocab),
        )
    )


def fused_linear_scaled_cross_entropy_backward_workspace_bytes(
    max_tokens: int,
    max_hidden: int,
    max_local_vocab: int,
    max_tiles_per_reduce: int,
) -> int:
    """Return total symmetric plus device-private backward pool bytes.

    Before configuration, this is a conservative topology-independent
    estimate. After configuration, all arguments must exactly match the
    immutable configured capacity and the exact allocated footprint is
    returned.
    """
    return int(
        _load_module().fused_linear_scaled_cross_entropy_backward_workspace_bytes(
            int(max_tokens),
            int(max_hidden),
            int(max_local_vocab),
            int(max_tiles_per_reduce),
        )
    )


def fused_linear_scaled_cross_entropy_forward_diagnostics(
    device: torch.device | str,
) -> torch.Tensor:
    """Return the device timestamp/counter block from the latest forward."""
    module = _load_module()
    entries = int(module.fused_linear_scaled_cross_entropy_forward_diagnostic_entries())
    output = torch.empty(entries, dtype=torch.int64, device=device)
    module.fused_linear_scaled_cross_entropy_forward_diagnostics(output)
    return output


def fused_linear_scaled_cross_entropy_backward_diagnostics(
    device: torch.device | str,
) -> torch.Tensor:
    """Return the device timestamp/counter block from the latest backward."""
    module = _load_module()
    entries = int(module.fused_linear_scaled_cross_entropy_backward_diagnostic_entries())
    output = torch.empty(entries, dtype=torch.int64, device=device)
    module.fused_linear_scaled_cross_entropy_backward_diagnostics(output)
    return output


def fused_linear_scaled_cross_entropy_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
    inverse_temperature: float,
    team_handle: int,
    return_entropy: bool,
):
    """Run tensor-parallel fused projection and scaled cross entropy.

    Args:
        x: Contiguous BF16 tensor with shape ``[tokens, hidden]``.
        weight: This rank's contiguous BF16 vocabulary shard with shape
            ``[local_vocab, hidden]``.
        target: Global int64 vocabulary indices with shape ``[tokens]``.
        vocab_start: Global vocabulary index represented by ``weight[0]``.
        ignore_index: Target value whose NLL and entropy outputs are zeroed.
        inverse_temperature: Positive multiplier applied to classifier logits.
        team_handle: Configured NVSHMEM tensor-parallel team.
        return_entropy: Whether to compute per-token entropy.

    Returns:
        ``(nll, lse, entropy)`` as FP32 tensors with shape ``[tokens]``.
        ``entropy`` is zero-filled when ``return_entropy`` is false.
    """
    tokens = x.shape[0]
    nll = torch.empty(tokens, dtype=torch.float32, device=x.device)
    lse = torch.empty(tokens, dtype=torch.float32, device=x.device)
    entropy = (
        torch.empty(tokens, dtype=torch.float32, device=x.device)
        if return_entropy
        else torch.zeros(tokens, dtype=torch.float32, device=x.device)
    )
    _load_module().fused_linear_scaled_cross_entropy_forward(
        x,
        weight,
        target,
        int(vocab_start),
        int(ignore_index),
        float(inverse_temperature),
        int(team_handle),
        bool(return_entropy),
        nll,
        lse,
        entropy,
    )
    return nll, lse, entropy


def fused_linear_scaled_cross_entropy_backward_phase_bench(
    grad_output: torch.Tensor,
    entropy_grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    entropy: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
    inverse_temperature: float,
    team_handle: int,
    phase: int,
    return_entropy: bool,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
):
    """Benchmark-only: run one backward GEMM phase (1=dZ, 2=dX, 4=dW).

    Uses the production kernel with a single phase bit set: identical tiles,
    pipelines, TMEM plan and epilogue, no reduction or communication.
    """
    _load_module().fused_linear_scaled_cross_entropy_backward_phase_bench(
        grad_output,
        entropy_grad,
        x,
        weight,
        target,
        lse,
        entropy,
        int(vocab_start),
        int(ignore_index),
        float(inverse_temperature),
        int(team_handle),
        int(phase),
        bool(return_entropy),
        grad_input,
        grad_weight,
    )


def fused_linear_scaled_cross_entropy_backward(
    grad_output: torch.Tensor,
    entropy_grad: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    entropy: torch.Tensor,
    vocab_start: int,
    ignore_index: int,
    inverse_temperature: float,
    team_handle: int,
    tiles_per_reduce: int,
    return_entropy: bool,
):
    """Run the persistent tensor-parallel fused backward.

    ``lse`` and ``entropy`` must be the outputs saved from the matching
    forward. The result is the globally reduced BF16 ``grad_input`` and this
    rank's local BF16 ``grad_weight``. On SM100, TP16 uses node-local NVLS,
    the matching-rank inter-host ring, and an all-CTA warp-1 FP32 merge.
    """
    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    _load_module().fused_linear_scaled_cross_entropy_backward(
        grad_output,
        entropy_grad,
        x,
        weight,
        target,
        lse,
        entropy,
        int(vocab_start),
        int(ignore_index),
        float(inverse_temperature),
        int(team_handle),
        int(tiles_per_reduce),
        bool(return_entropy),
        grad_input,
        grad_weight,
    )
    return grad_input, grad_weight
