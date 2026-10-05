"""Unit tests for the ``liger_cute_kernels`` TVM FFI MoE bindings.

These belong to the standalone ``liger_cute_kernels`` module — they import the
compiled package directly and do NOT depend on ``liger_kernel``. They exercise
the binding surface and a real single-PE forward/backward roundtrip:

  * the ``liger_cute_kernels.tvm_ffi`` facade imports and exposes the MoE entry
    points,
  * symmetric configuration succeeds for a valid topology,
  * a failed ``LIGER_CHECK`` in the torch-free core propagates across the
    ``extern "C"`` boundary (status code + thread-local message) and surfaces as
    a Python ``RuntimeError`` carrying that message,
  * the TVM FFI TensorView validation runs (dtype validation fires).

The whole module is skipped unless the compiled ``liger_cute_kernels`` package
is importable (build it via this module's README.md). Tests that need device
tensors are additionally skipped without CUDA. The roundtrip initializes
NVSHMEM in an isolated worker and checks eager and captured execution.
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
    """Configure after NVSHMEM bootstrap, as required by the native API."""
    if not _HAS_CUDA:
        yield tvm_ffi
        return
    with pytest.MonkeyPatch.context() as env:
        env.setenv("NVSHMEM_DISABLE_NCCL", "1")
        env.setenv("NVSHMEM_REMOTE_TRANSPORT", "none")
        torch.cuda.set_device(0)
        uid = torch.empty(tvm_ffi.uniqueid_nbytes(), dtype=torch.uint8, device="cpu")
        tvm_ffi.get_uniqueid(uid.data_ptr())
        tvm_ffi.init_with_uniqueid(0, 1, uid.data_ptr())
        try:
            yield tvm_ffi
        finally:
            tvm_ffi.finalize()


def test_tvm_ffi_exposes_moe_bindings(tvm_ffi_module):
    for name in (
        "moe_configure_symmetric",
        "moe_pop_fwd",
        "moe_fused_fwd_bf16",
        "moe_fused_bwd_bf16",
    ):
        assert hasattr(tvm_ffi_module, name), f"missing binding: {name}"


@pytest.mark.skipif(not _HAS_CUDA, reason="native architecture selection requires CUDA")
def test_configure_symmetric_valid(tvm_ffi_module):
    assert tvm_ffi_module.moe_configure_symmetric(**_CFG) is None


@pytest.mark.skipif(not _HAS_CUDA, reason="native architecture selection requires CUDA")
def test_configure_symmetric_accepts_changed_topology(tvm_ffi_module):
    assert tvm_ffi_module.moe_configure_symmetric(**_CFG) is None
    changed = {**_CFG, "num_hosts": 2, "gpus_per_host": 1}

    assert tvm_ffi_module.moe_configure_symmetric(**changed) is None


def test_configure_symmetric_topology_mismatch_raises(tvm_ffi_module):
    # num_hosts * gpus_per_host (1 * 2) != num_pes (4): LIGER_CHECK fails in the
    # core and the message crosses the boundary into the Python exception.
    bad = {**_CFG, "num_pes": 4}
    with pytest.raises(RuntimeError, match="must equal num_pes"):
        tvm_ffi_module.moe_configure_symmetric(**bad)


@pytest.mark.parametrize("num_hosts,gpus_per_host,num_pes", [(0, 2, 0), (-1, -2, 2), (1, 0, 0), (1, 2, -2)])
def test_configure_symmetric_nonpositive_topology_raises(tvm_ffi_module, num_hosts, gpus_per_host, num_pes):
    bad = {**_CFG, "num_hosts": num_hosts, "gpus_per_host": gpus_per_host, "num_pes": num_pes}
    with pytest.raises(RuntimeError, match="must be positive"):
        tvm_ffi_module.moe_configure_symmetric(**bad)


def test_configure_symmetric_topology_product_does_not_overflow(tvm_ffi_module):
    bad = {**_CFG, "num_hosts": 65537, "gpus_per_host": 65537, "num_pes": 131073}
    with pytest.raises(RuntimeError, match="4295098369.*must equal num_pes"):
        tvm_ffi_module.moe_configure_symmetric(**bad)


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA tensors")
def test_fwd_rejects_wrong_dtype(tvm_ffi_module):
    # Forward expects bf16 X; a float32 X must trip TVM FFI dtype validation,
    # which runs before any symmetric allocation.
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


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA tensors")
def test_fused_fwd_bwd_roundtrip(monkeypatch):
    from test_moe_cuda_graph import _fwd_bwd_graph_worker
    from test_moe_cuda_graph import _run

    monkeypatch.setenv("NVSHMEM_DISABLE_NCCL", "1")
    monkeypatch.setenv("NVSHMEM_REMOTE_TRANSPORT", "none")
    monkeypatch.setenv("NVSHMEM_SYMMETRIC_SIZE", "6G")
    _run(1, _fwd_bwd_graph_worker)
