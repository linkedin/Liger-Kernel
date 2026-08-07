"""TVM FFI facade for the LigerCute native core.

This module is the Python facade used by ``liger_kernel.ops.cute``. It loads the
torch-free native core through TVM FFI and passes tensors through DLPack views.
"""

from __future__ import annotations

import ctypes

from pathlib import Path

import torch

_MOD = None
_NVSHMEM_LIBS_LOADED = False


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
        module = pkg_dir / "libliger_cute_kernels.so"
        if not module.exists():
            raise FileNotFoundError(f"Missing packaged TVM FFI module: {module}")
        _MOD = tvm_ffi.load_module(str(module))
    return _MOD


def _load_nvshmem_libraries(pkg_dir: Path) -> None:
    global _NVSHMEM_LIBS_LOADED
    if _NVSHMEM_LIBS_LOADED:
        return
    for host in (pkg_dir / "libnvshmem_host.so.3", pkg_dir / "libnvshmem_host.so"):
        if host.exists():
            ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
            break
    uid_bootstrap = pkg_dir / "nvshmem_bootstrap_uid.so.3"
    if uid_bootstrap.exists():
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
    symm_meta = torch.empty(13, dtype=torch.int64, device="cpu")
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
    chosen_tile_m = int(symm_meta[12].item())
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
