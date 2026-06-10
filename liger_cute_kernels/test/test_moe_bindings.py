"""Unit tests for the ``liger_cute_kernels`` (``_C``) MoE bindings.

These belong to the standalone ``liger_cute_kernels`` module — they import the
compiled package directly and do NOT depend on ``liger_kernel``. The native
kernels are not ported yet, so these do NOT check numerics; they exercise the
binding *surface* that already exists:

  * the ``liger_cute_kernels._C`` extension imports and exposes the MoE entry
    points,
  * symmetric configuration succeeds for a valid topology,
  * a failed ``LIGER_CHECK`` in the torch-free core propagates across the
    ``extern "C"`` boundary (status code + thread-local message) and surfaces as
    a Python ``RuntimeError`` carrying that message,
  * the torch::Tensor -> TensorView marshalling runs (dtype validation fires).

The whole module is skipped unless the compiled ``liger_cute_kernels`` package
is importable (build it via this module's README.md). Tests that need device
tensors are additionally skipped without CUDA. The full valid forward/backward
needs an initialized NVSHMEM runtime (no bootstrap binding yet) and is left as
an explicit skip.
"""

import pytest

try:
    import liger_cute_kernels._C as _ext_mod
    import torch
except ImportError:
    torch = None
    _ext_mod = None

pytestmark = pytest.mark.skipif(
    _ext_mod is None,
    reason="liger_cute_kernels not built/installed; build it to run these.",
)

# Safe even when torch failed to import (module is skipped in that case anyway).
_HAS_CUDA = _ext_mod is not None and torch is not None and torch.cuda.is_available()

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
def ext():
    """The compiled ``liger_cute_kernels._C`` extension module."""
    return _ext_mod


def test_extension_exposes_moe_bindings(ext):
    for name in (
        "moe_configure_symmetric",
        "moe_pop_fwd",
        "moe_fused_fwd_bf16",
        "moe_fused_bwd_bf16",
    ):
        assert hasattr(ext, name), f"missing binding: {name}"


def test_configure_symmetric_valid(ext):
    # Returns None and does not raise for a consistent topology.
    assert ext.moe_configure_symmetric(**_CFG) is None


def test_configure_symmetric_topology_mismatch_raises(ext):
    # num_hosts * gpus_per_host (1 * 2) != num_pes (4): LIGER_CHECK fails in the
    # core and the message crosses the boundary into the Python exception.
    bad = {**_CFG, "num_pes": 4}
    with pytest.raises(RuntimeError, match="must equal num_pes"):
        ext.moe_configure_symmetric(**bad)


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA tensors")
def test_fwd_rejects_wrong_dtype(ext):
    # Forward expects bf16 X; a float32 X must trip the dtype LIGER_CHECK, which
    # runs before any symmetric allocation (so no NVSHMEM runtime is required).
    ext.moe_configure_symmetric(**_CFG)
    T, D, E, K = 16, _CFG["hidden_dim"], _CFG["max_num_experts"], _CFG["max_top_k"]
    X = torch.randn(T, D, dtype=torch.float32, device="cuda")  # wrong dtype on purpose
    expert_indices = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    expert_weights = torch.zeros(T, K, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(E, D, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="must be bfloat16"):
        ext.moe_fused_fwd_bf16(X, expert_indices, expert_weights, B, B, B, num_experts=E, top_k=K, team_handle=0)


@pytest.mark.skip(
    reason="full fwd/bwd allocates symmetric memory; needs an initialized NVSHMEM "
    "runtime — bootstrap bindings not ported yet."
)
def test_fused_fwd_bwd_roundtrip(ext):
    # Placeholder for the end-to-end shape/lifetime contract once kernels and the
    # NVSHMEM bootstrap path are wired up: fwd returns
    # (Y, x_sorted, y_buf, all_expert_offsets, token_expert_slots,
    #  tile_expert_ids, chosen_tile_m); bwd consumes them and returns the grads.
    raise NotImplementedError
