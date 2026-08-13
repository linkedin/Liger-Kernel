"""Unit tests for the ``liger_cute_kernels`` TVM FFI MoE bindings.

These belong to the standalone ``liger_cute_kernels`` module — they import the
compiled package directly and do NOT depend on ``liger_kernel``. The native
kernels are not ported yet, so these do NOT check numerics; they exercise the
binding *surface* that already exists:

  * the ``liger_cute_kernels.tvm_ffi`` facade imports and exposes the MoE entry
    points,
  * symmetric configuration succeeds for a valid topology,
  * a failed ``LIGER_CHECK`` in the torch-free core propagates across the
    ``extern "C"`` boundary (status code + thread-local message) and surfaces as
    a Python ``RuntimeError`` carrying that message,
  * the TVM FFI TensorView validation runs (dtype validation fires).

The whole module is skipped unless the compiled ``liger_cute_kernels`` package
is importable (build it via this module's README.md). Tests that need device
tensors are additionally skipped without CUDA. The full valid forward/backward
needs an initialized NVSHMEM runtime (no bootstrap binding yet) and is left as
an explicit skip.
"""

import pytest

try:
    import liger_cute_kernels.tvm_ffi as tvm_ffi
    import torch
except ImportError:
    torch = None
    tvm_ffi = None

pytestmark = pytest.mark.skipif(
    tvm_ffi is None or not tvm_ffi.is_available(),
    reason="liger_cute_kernels not built/installed; build it to run these.",
)

# Safe even when torch failed to import (module is skipped in that case anyway).
_HAS_CUDA = tvm_ffi is not None and tvm_ffi.is_available() and torch is not None and torch.cuda.is_available()

# Valid symmetric-config topology reused across tests: num_hosts * gpus_per_host
# == num_pes, and max_num_experts divisible by num_pes.
_CFG = dict(
    max_tokens=128,
    hidden_dim=2048,
    max_num_experts=8,
    max_top_k=2,
    num_pes=2,
    num_hosts=1,
    gpus_per_host=2,
)


@pytest.fixture(scope="module")
def tvm_ffi_module():
    """The ``liger_cute_kernels.tvm_ffi`` facade module."""
    return tvm_ffi


def test_tvm_ffi_exposes_moe_bindings(tvm_ffi_module):
    for name in (
        "moe_configure_symmetric",
        "moe_pop_fwd",
        "moe_fused_fwd_bf16",
        "moe_fused_bwd_bf16",
    ):
        assert hasattr(tvm_ffi_module, name), f"missing binding: {name}"


@pytest.mark.skipif(not _HAS_CUDA, reason="configure uploads the comm schedule to device constant memory")
def test_configure_symmetric_valid(tvm_ffi_module):
    # Returns None and does not raise for a consistent topology. Needs CUDA: the
    # call now uploads the comm schedule (g_dest_table / g_rank_table) via
    # cudaMemcpyToSymbol, which requires a CUDA context.
    assert tvm_ffi_module.moe_configure_symmetric(**_CFG) is None


def test_configure_symmetric_topology_mismatch_raises(tvm_ffi_module):
    # num_hosts * gpus_per_host (1 * 2) != num_pes (4): LIGER_CHECK fails in the
    # core and the message crosses the boundary into the Python exception.
    bad = {**_CFG, "num_pes": 4}
    with pytest.raises(RuntimeError, match="must equal num_pes"):
        tvm_ffi_module.moe_configure_symmetric(**bad)


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA tensors")
def test_fwd_rejects_wrong_dtype(tvm_ffi_module):
    # Forward expects bf16 X; a float32 X must trip TVM FFI dtype validation,
    # which runs before any symmetric allocation (so no NVSHMEM runtime is required).
    tvm_ffi_module.moe_configure_symmetric(**_CFG)
    T, D, E, K = 16, _CFG["hidden_dim"], _CFG["max_num_experts"], _CFG["max_top_k"]
    X = torch.randn(T, D, dtype=torch.float32, device="cuda")  # wrong dtype on purpose
    expert_indices = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    expert_weights = torch.zeros(T, K, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(E, D, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="float32 vs\\. bfloat16.*X"):
        tvm_ffi_module.moe_fused_fwd_bf16(
            X, expert_indices, expert_weights, B, B, B, num_experts=E, top_k=K, team_handle=0
        )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA tensors")
def test_fwd_rejects_mismatched_gate_hidden_dim(tvm_ffi_module):
    tvm_ffi_module.moe_configure_symmetric(**_CFG)
    T, D, E, K = 16, _CFG["hidden_dim"], _CFG["max_num_experts"], _CFG["max_top_k"]
    experts_per_pe = E // _CFG["num_pes"]
    intermediate_dim = 16
    X = torch.zeros(T, D, dtype=torch.bfloat16, device="cuda")
    expert_indices = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    expert_weights = torch.zeros(T, K, dtype=torch.bfloat16, device="cuda")
    bad_B = torch.zeros(
        experts_per_pe,
        intermediate_dim,
        D // 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    C = torch.zeros(
        experts_per_pe,
        intermediate_dim,
        D,
        dtype=torch.bfloat16,
        device="cuda",
    )
    A = torch.zeros(
        experts_per_pe,
        D,
        intermediate_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="all_B hidden dimension must match X"):
        tvm_ffi_module.moe_fused_fwd_bf16(
            X,
            expert_indices,
            expert_weights,
            bad_B,
            C,
            A,
            num_experts=E,
            top_k=K,
            team_handle=0,
        )


@pytest.mark.skip(
    reason="full fwd/bwd allocates symmetric memory; needs an initialized NVSHMEM "
    "runtime — bootstrap bindings not ported yet."
)
def test_fused_fwd_bwd_roundtrip(tvm_ffi_module):
    # Placeholder for the end-to-end shape/lifetime contract once kernels and the
    # NVSHMEM bootstrap path are wired up: fwd returns
    # (Y, x_sorted, y_buf, all_expert_offsets, token_expert_slots,
    #  tile_expert_ids, chosen_tile_m); bwd consumes them and returns the grads.
    raise NotImplementedError
