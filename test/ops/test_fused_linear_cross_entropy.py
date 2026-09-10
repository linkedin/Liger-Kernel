"""Cross-backend correctness tests for ``fused_linear_cross_entropy``.

Each backend registered for ``fused_linear_cross_entropy`` is exercised with
the same shapes and dtypes and compared (forward + backward) against a PyTorch
reference. Tolerances come from each ``OpImpl``'s registered ``tolerances``
table.

Mirrors ``test/ops/test_fused_linear_jsd.py`` in structure: collects cleanly
on a CPU-only box (the conftest ``autouse`` fixture skips when CUDA is
unavailable).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F
import triton

from torch.utils._python_dispatch import TorchDispatchMode

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401
import liger_kernel.ops.fused_linear_cross_entropy as flce_ops

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

from .conftest import get_available_backends_for_op

# (BT, V, H) — V must be a multiple of 8 for CuTe DSL 128-bit vectorized loads.
FLCE_TEST_SHAPES = [
    (32, 256, 1024),
    (64, 512, 768),
    (128, 256, 1024),
]
FLCE_TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_REGISTERED_BACKENDS = get_available_backends_for_op("fused_linear_cross_entropy")

_DEFAULT_TOLS = {
    torch.float16: {"atol_fwd": 5e-3, "rtol_fwd": 1e-3, "atol_bwd": 5e-2, "rtol_bwd": 1e-2},
    torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 1e-2, "atol_bwd": 1e-1, "rtol_bwd": 2e-2},
    torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
}


def _tolerances_for(backend: str, dtype: torch.dtype) -> dict:
    impl = get_registered("fused_linear_cross_entropy", backend)
    tols = dict((impl.tolerances if impl is not None else {}).get(dtype, {}))
    for k, v in _DEFAULT_TOLS.get(dtype, {}).items():
        tols.setdefault(k, v)
    return tols


def _flce_ref(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch reference: compute logits then cross-entropy loss."""
    logits = _input.float() @ weight.float().T
    if bias is not None:
        logits = logits + bias.float()
    loss = F.cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    return loss


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FLCE_TEST_SHAPES)
@pytest.mark.parametrize("dtype", FLCE_TEST_DTYPES)
def test_fused_linear_cross_entropy_correctness(backend, shape, dtype):
    """Forward + backward parity against the PyTorch FLCE reference."""
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    BT, V, H = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(42)

    inp_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02
    target_cpu = torch.randint(0, V, (BT,), generator=g)

    inp = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target = target_cpu.to(device=device)

    inp_ref = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w_ref = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target_ref = target_cpu.to(device=device)

    tols = _tolerances_for(backend, dtype)

    # Forward — dispatch returns (loss, z_loss, token_accuracy, predicted_tokens).
    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        w,
        target,
        None,  # bias
        None,  # ce_weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        None,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        backend=backend,
    )
    loss_ref = _flce_ref(inp_ref, w_ref, target_ref)

    torch.testing.assert_close(
        loss.to(torch.float32),
        loss_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] forward: {m}",
    )

    # Backward — scalar loss, so backward uses implicit grad=1.
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(
        inp.grad.to(torch.float32),
        inp_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] d_input: {m}",
    )
    torch.testing.assert_close(
        w.grad.to(torch.float32),
        w_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[fused_linear_cross_entropy/{backend} shape={shape} dtype={dtype}] d_weight: {m}",
    )


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
@pytest.mark.parametrize("shape", FLCE_TEST_SHAPES[:1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_linear_cross_entropy_with_bias(backend, shape, dtype):
    """FLCE with bias term."""
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    BT, V, H = shape
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(42)

    inp_cpu = torch.randn(BT, H, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(V, H, dtype=torch.float32, generator=g) * 0.02
    b_cpu = torch.randn(V, dtype=torch.float32, generator=g) * 0.01
    target_cpu = torch.randint(0, V, (BT,), generator=g)

    inp = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target = target_cpu.to(device=device)

    inp_ref = inp_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    w_ref = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    b_ref = b_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    target_ref = target_cpu.to(device=device)

    tols = _tolerances_for(backend, dtype)

    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        w,
        target,
        b,
        backend=backend,
    )
    loss_ref = _flce_ref(inp_ref, w_ref, target_ref, bias=b_ref)

    torch.testing.assert_close(
        loss.to(torch.float32),
        loss_ref.to(torch.float32),
        atol=tols["atol_fwd"],
        rtol=tols["rtol_fwd"],
        msg=lambda m: f"[FLCE+bias/{backend} shape={shape}] forward: {m}",
    )

    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(
        b.grad.to(torch.float32),
        b_ref.grad.to(torch.float32),
        atol=tols["atol_bwd"],
        rtol=tols["rtol_bwd"],
        msg=lambda m: f"[FLCE+bias/{backend} shape={shape}] d_bias: {m}",
    )


def test_fused_linear_cross_entropy_available_backends_includes_triton():
    """Sanity: the Triton implementation should always be available."""
    impls = available_backends("fused_linear_cross_entropy")
    assert any(b in ("triton", "nvidia-triton") for b in impls), f"expected 'triton' / 'nvidia-triton' in {impls}"


@pytest.mark.parametrize("backend", _REGISTERED_BACKENDS or ["__none__"])
def test_fused_linear_cross_entropy_propagates_backend_to_inner_ce(monkeypatch, backend):
    if backend == "__none__":
        pytest.skip("No fused_linear_cross_entropy backends registered in this environment")

    observed_impls = []
    real_dispatch = flce_ops.dispatch

    def tracking_dispatch(op_name, *args, **kwargs):
        if op_name == "cross_entropy_loss_and_grad":
            observed_impls.append(kwargs.get("impl"))
        return real_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(flce_ops, "dispatch", tracking_dispatch)

    inp = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, 256, (8,), device="cuda")

    loss, _, _, _ = dispatch(
        "fused_linear_cross_entropy",
        inp,
        weight,
        target,
        backend=backend,
    )
    loss.backward()

    assert observed_impls
    assert set(observed_impls) == {backend}


# ===========================================================================
# Regression tests for the weight-gradient ``addmm`` projection in
# ``fused_linear_cross_entropy_forward``.
#
# These lock in the low-precision ``grad_weight`` accumulation path: when
# ``accum_dtype=None`` and the parameters are bf16/fp16, the per-chunk weight
# gradient must be accumulated *directly* into ``grad_weight`` via
# ``torch.addmm(out=grad_weight)`` (mirroring the CuTe backend's direct bf16
# addmm), instead of the legacy ``torch.mm(...).float()`` which materialises a
# parameter-sized bf16->fp32 temporary + cast every chunk.
#
# The tests assert on the actual aten dispatch (``TorchDispatchMode``) *and* on
# numerical parity against a full-precision autograd reference. They are
# GPU-gated by ``test/ops/conftest.py``'s autouse ``_require_cuda_for_ops_tests``
# fixture, and the accumulation path itself requires a CUDA SM80+ device.
#
# The production token-chunk loop sizes chunks from the ``_CHUNK_MEM_CONST``
# memory budget (``inc_factor = cdiv(V, C * H)``). At the tiny shapes used here
# the default ``C=16`` collapses to a single chunk, which would never exercise
# the per-chunk grad_weight accumulation these tests target. The class-local
# autouse ``_force_multichunk`` fixture pins that budget knob to ``C=1`` *only
# for these migrated tests* so the real production loop splits these shapes into
# multiple chunks -- the cross-backend tests above keep the production ``C=16``.
# ===========================================================================

# ---------------------------------------------------------------------------
# Fast-path capability gates
# ---------------------------------------------------------------------------
#
# The whole directory is CUDA-gated by ``conftest.py``'s autouse fixture, but
# *being on CUDA* is not sufficient for the ``addmm(out=)`` / ``addmm.dtype_out``
# fast paths. Production (``fused_linear_cross_entropy_forward``) gates them on:
#
#   * an SM80+ device -- ``get_device_capability()[0] >= 8`` (SM80+; this is
#     also exactly where bf16 GEMM hardware becomes available), and
#   * ``_ADDMM_SUPPORTS_OUT_DTYPE`` -- torch >= 2.8, additionally required for
#     the fp32-accumulator ``addmm.dtype_out`` overload.
#
# On valid-but-older setups (real SM70/SM75, or the base ``torch>=2.1.2`` pin
# through torch<2.8) production correctly falls back to ``torch.mm(...).float()``.
# Only tests that *assert the fast path was taken* must be gated with the same
# predicate; parity/fallback tests stay ungated so they still run everywhere.


def _cuda_capability_major() -> int:
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.get_device_capability()[0]


# Reused across every fast-path-asserting test (mirrors production's SM80+ gate).
requires_sm80_or_higher = pytest.mark.skipif(
    _cuda_capability_major() < 8,
    reason="addmm(out=) fast path requires CUDA SM80+ (get_device_capability()[0] >= 8)",
)

# Additionally required by the fp32-accumulator addmm.dtype_out fast path.
requires_addmm_out_dtype = pytest.mark.skipif(
    not flce_ops._ADDMM_SUPPORTS_OUT_DTYPE,
    reason="addmm out_dtype overload (fp32 accumulator fast path) requires torch >= 2.8",
)


def _production_chunk_size(BT: int, V: int, H: int) -> int:
    """The chunk size the production loop will pick for these dims, using the
    *current* module-level ``_CHUNK_MEM_CONST`` (pinned to 1 by the class-local
    autouse fixture). Mirrors the arithmetic in
    ``fused_linear_cross_entropy_forward``."""
    inc_factor = triton.cdiv(V, flce_ops._CHUNK_MEM_CONST * H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    return min(chunk_size, BT)


# ---------------------------------------------------------------------------
# Dispatch recorder
# ---------------------------------------------------------------------------

_ADDMM_OPS = ("aten.addmm.out", "aten.addmm.dtype_out")


class _OpRecorder(TorchDispatchMode):
    """Records the aten ops we care about: addmm (with the out-tensor shape and
    whether the out_dtype overload was used) and ``_to_copy`` (with in/out dtype
    and shape so we can detect parameter-sized upcasts)."""

    def __init__(self):
        self.addmm_out_shapes = []  # list of torch.Size for aten.addmm.out
        self.addmm_dtype_out_shapes = []  # list of torch.Size for aten.addmm.dtype_out
        self.mm_default = 0  # count of aten.mm.default
        self.to_copies = []  # list of (in_dtype, out_dtype, shape)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in _ADDMM_OPS:
            out = kwargs.get("out")
            shape = tuple(out.shape) if out is not None else None
            if name == "aten.addmm.out":
                self.addmm_out_shapes.append(shape)
            else:
                self.addmm_dtype_out_shapes.append(shape)
        elif name == "aten.mm.default":
            self.mm_default += 1
        elif name == "aten._to_copy.default":
            src = args[0] if args else None
            if isinstance(src, torch.Tensor):
                self.to_copies.append((src.dtype, kwargs.get("dtype"), tuple(src.shape)))
        return func(*args, **kwargs)

    def has_addmm_out_into(self, shape) -> bool:
        return tuple(shape) in self.addmm_out_shapes

    def has_addmm_dtype_out_into(self, shape) -> bool:
        return tuple(shape) in self.addmm_dtype_out_shapes

    def count_addmm_out_into(self, shape) -> int:
        return self.addmm_out_shapes.count(tuple(shape))

    def count_addmm_dtype_out_into(self, shape) -> int:
        return self.addmm_dtype_out_shapes.count(tuple(shape))

    def has_param_upcast(self, shape, low_dtypes=(torch.bfloat16, torch.float16)) -> bool:
        """True iff some ``_to_copy`` upcast a *parameter-shaped* low-precision
        tensor to fp32 (the legacy ``.float()`` temporary we want to eliminate)."""
        tgt = tuple(shape)
        return any(src in low_dtypes and dst == torch.float32 and sh == tgt for (src, dst, sh) in self.to_copies)


# ---------------------------------------------------------------------------
# Reference + driver helpers
# ---------------------------------------------------------------------------

_ADDMM_DEVICE = "cuda"


def _addmm_reference_grads(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    token_grad_output: Optional[torch.Tensor] = None,
):
    """Full-precision autograd reference for (loss, grad_input, grad_weight, grad_bias)."""
    xi = _input.detach().float().requires_grad_(True)
    wi = weight.detach().float().requires_grad_(weight.requires_grad)
    bi = None
    if bias is not None:
        bi = bias.detach().float().requires_grad_(bias.requires_grad)
    logits = xi @ wi.t()
    if bi is not None:
        logits = logits + bi
    cw = ce_weight.detach().float() if ce_weight is not None else None
    loss = F.cross_entropy(
        logits,
        target,
        weight=cw,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    if reduction == "none":
        assert token_grad_output is not None
        loss.backward(token_grad_output.float())
    else:
        loss.backward()
    return (
        loss.detach(),
        xi.grad.detach(),
        wi.grad.detach() if wi.grad is not None else None,
        bi.grad.detach() if (bi is not None and bi.grad is not None) else None,
    )


def _run_flce(
    _input,
    weight,
    target,
    *,
    bias=None,
    ce_weight=None,
    ignore_index=-100,
    label_smoothing=0.0,
    reduction="mean",
    accum_dtype=None,
    use_token_scaling=False,
    token_grad_output=None,
    recorder: Optional[_OpRecorder] = None,
    autocast_dtype=None,
):
    """Invoke the forward the same way the autograd Function does: under
    ``no_grad`` (Function.forward runs with grad disabled), optionally under
    autocast, optionally under a dispatch recorder. The chunk geometry is chosen
    by production from ``_CHUNK_MEM_CONST`` -- never forwarded from the test."""

    def _call():
        return fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            ce_weight=ce_weight,
            bias=bias,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
            token_grad_output=token_grad_output,
        )

    import contextlib

    stack = contextlib.ExitStack()
    with stack:
        stack.enter_context(torch.no_grad())
        if autocast_dtype is not None:
            stack.enter_context(torch.autocast("cuda", dtype=autocast_dtype))
        if recorder is not None:
            stack.enter_context(recorder)
        loss, z_loss, token_acc, pred, grad_input, grad_weight, grad_bias = _call()
    return loss, grad_input, grad_weight, grad_bias


_ADDMM_TOLS = {
    torch.float16: dict(atol=5e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-1, rtol=2e-2),
    torch.float32: dict(atol=1e-4, rtol=1e-4),
}


def _mk_inputs(
    BT,
    V,
    H,
    dtype,
    *,
    bias,
    ce_weight,
    seed=0,
    input_dtype=None,
    weight_requires_grad=True,
    input_requires_grad=True,
    all_ignore=False,
    first_chunk_size=None,
):
    g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(seed)
    input_dtype = input_dtype or dtype
    _input = torch.randn(BT, H, device=_ADDMM_DEVICE, dtype=input_dtype, generator=g) * 0.1
    _input.requires_grad_(input_requires_grad)
    weight = torch.randn(V, H, device=_ADDMM_DEVICE, dtype=dtype, generator=g) * 0.1
    weight.requires_grad_(weight_requires_grad)
    b = None
    if bias:
        b = (torch.randn(V, device=_ADDMM_DEVICE, dtype=dtype, generator=g) * 0.1).requires_grad_(True)
    cw = None
    if ce_weight:
        cw = torch.rand(V, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) + 0.1
    target = torch.randint(0, V, (BT,), device=_ADDMM_DEVICE, generator=g, dtype=torch.long)
    if all_ignore:
        target[:] = -100
    elif first_chunk_size is not None:
        # Ignore exactly the first production chunk so its rows contribute no grad.
        target[:first_chunk_size] = -100
    return _input, weight, b, cw, target


class TestFusedLinearCrossEntropyAddmm:
    """Regression suite for the low-precision ``grad_weight`` addmm path.

    Grouped in a class so the ``_force_multichunk`` autouse fixture (which pins
    ``_CHUNK_MEM_CONST=1``) is scoped to *only* these migrated tests and never
    perturbs the production ``C=16`` used by the cross-backend tests above.
    """

    @pytest.fixture(autouse=True)
    def _force_multichunk(self, monkeypatch):
        """Pin the existing ``_CHUNK_MEM_CONST`` memory budget to 1 for each test.

        This is the *production* chunk-sizing knob (``inc_factor = cdiv(V, C*H)``),
        not a test-only chunk API. At ``C=1`` the small shapes below split into
        several token chunks, so the genuine per-chunk grad_weight accumulation
        loop runs more than once -- which is exactly what these regression tests
        cover. Being class-local, it does not affect any other test in the module.
        """
        monkeypatch.setattr(flce_ops, "_CHUNK_MEM_CONST", 1)

    # -----------------------------------------------------------------------
    # 1. Dispatch: the low-precision default path uses addmm(out=) and no param upcast
    # -----------------------------------------------------------------------

    @requires_sm80_or_higher
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("bias", [True, False])
    def test_lowprec_default_uses_addmm_out_no_param_upcast(self, dtype, bias):
        BT, V, H = 40, 128, 32  # C=1 -> chunk_size 16 -> 3 chunks (16,16,8): multichunk + tail
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, dtype, bias=bias, ce_weight=False)

        rec = _OpRecorder()
        loss, gi, gw, gb = _run_flce(_input, weight, target, bias=b, accum_dtype=None, recorder=rec)

        # The accumulator inherits the (low-precision) parameter dtype ...
        assert gw.dtype == weight.dtype
        # ... and is written by addmm(out=grad_weight) into a parameter-shaped buffer,
        # once per chunk -- proving the real multi-chunk accumulation loop ran.
        assert rec.count_addmm_out_into(weight.shape) >= 2, (
            f"expected >=2 aten.addmm.out into {tuple(weight.shape)} (multichunk), saw {rec.addmm_out_shapes}"
        )
        # The fp32 out_dtype overload must NOT be used for a low-precision accumulator.
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        # And crucially: no parameter-sized bf16/fp16 -> fp32 temporary (the legacy .float()).
        assert not rec.has_param_upcast(weight.shape), f"unexpected parameter-shaped upcast to fp32: {rec.to_copies}"

        # Parity vs full-precision autograd reference.
        tol = _ADDMM_TOLS[dtype]
        _, ref_gi, ref_gw, ref_gb = _addmm_reference_grads(_input, weight, target, bias=b)
        assert torch.isfinite(gw).all()
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)
        if bias:
            torch.testing.assert_close(gb.float(), ref_gb, **tol)

    # -----------------------------------------------------------------------
    # 2. Numerical parity across the combinatorial space (small shapes)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("ce_weight", [True, False])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_parity_combinations(self, dtype, reduction, bias, ce_weight, label_smoothing):
        BT, V, H = 40, 128, 32  # 3 chunks (16,16,8)
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, dtype, bias=bias, ce_weight=ce_weight, seed=1)
        _, gi, gw, gb = _run_flce(
            _input,
            weight,
            target,
            bias=b,
            ce_weight=cw,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        _, ref_gi, ref_gw, ref_gb = _addmm_reference_grads(
            _input,
            weight,
            target,
            bias=b,
            ce_weight=cw,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        tol = _ADDMM_TOLS[dtype]
        assert torch.isfinite(gw).all() and torch.isfinite(gi).all()
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)
        if bias:
            torch.testing.assert_close(gb.float(), ref_gb, **tol)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_parity_reduction_none_nonuniform_upstream(self, dtype):
        BT, V, H = 24, 96, 32  # C=1 -> chunk_size 8 -> 3 chunks (8,8,8)
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, dtype, bias=True, ce_weight=False, seed=2)
        g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(7)
        token_grad_output = torch.rand(BT, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) + 0.05
        _, gi, gw, gb = _run_flce(
            _input,
            weight,
            target,
            bias=b,
            reduction="none",
            token_grad_output=token_grad_output,
        )
        _, ref_gi, ref_gw, ref_gb = _addmm_reference_grads(
            _input,
            weight,
            target,
            bias=b,
            reduction="none",
            token_grad_output=token_grad_output,
        )
        tol = _ADDMM_TOLS[dtype]
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)
        torch.testing.assert_close(gb.float(), ref_gb, **tol)

    # -----------------------------------------------------------------------
    # 3. Ignore-index edge cases initialise correctly (zero-init preserved)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_first_chunk_all_ignored(self, dtype):
        BT, V, H = 40, 128, 32  # 3 chunks (16,16,8)
        cs = _production_chunk_size(BT, V, H)  # actual first-chunk size (16)
        _input, weight, b, cw, target = _mk_inputs(
            BT, V, H, dtype, bias=True, ce_weight=False, seed=3, first_chunk_size=cs
        )
        _, gi, gw, gb = _run_flce(_input, weight, target, bias=b)
        _, ref_gi, ref_gw, ref_gb = _addmm_reference_grads(_input, weight, target, bias=b)
        tol = _ADDMM_TOLS[dtype]
        assert torch.isfinite(gw).all()
        # First chunk was fully ignored -> its rows contribute zero grad_input.
        assert torch.count_nonzero(gi[:cs]) == 0
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_all_tokens_ignored_zero_grad(self, dtype):
        BT, V, H = 32, 128, 32
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, dtype, bias=True, ce_weight=False, seed=4, all_ignore=True)
        _, gi, gw, gb = _run_flce(_input, weight, target, bias=b)
        # Every token ignored -> all grads are exactly the zero-initialised buffers.
        assert torch.isfinite(gw).all()
        assert torch.count_nonzero(gw) == 0
        assert torch.count_nonzero(gi) == 0
        assert torch.count_nonzero(gb) == 0

    # -----------------------------------------------------------------------
    # 4. Frozen weight + trainable input -> no grad_weight, no addmm into weight
    # -----------------------------------------------------------------------

    def test_frozen_weight_trainable_input(self):
        BT, V, H = 40, 128, 32
        _input, weight, b, cw, target = _mk_inputs(
            BT,
            V,
            H,
            torch.bfloat16,
            bias=False,
            ce_weight=False,
            seed=5,
            weight_requires_grad=False,
            input_requires_grad=True,
        )
        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, recorder=rec)
        assert gw is None
        assert not rec.has_addmm_out_into(weight.shape)
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        # grad_input must still be produced and correct.
        _, ref_gi, _, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gi.float(), ref_gi, **_ADDMM_TOLS[torch.bfloat16])

    # -----------------------------------------------------------------------
    # 5. Non-contiguous input / weight
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_noncontiguous_input_and_weight(self, dtype):
        BT, V, H = 40, 128, 32
        g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(6)
        _input = (torch.randn(H, BT, device=_ADDMM_DEVICE, dtype=dtype, generator=g) * 0.1).t().requires_grad_(True)
        weight = (torch.randn(H, V, device=_ADDMM_DEVICE, dtype=dtype, generator=g) * 0.1).t().requires_grad_(True)
        assert not _input.is_contiguous() and not weight.is_contiguous()
        target = torch.randint(0, V, (BT,), device=_ADDMM_DEVICE, generator=g, dtype=torch.long)
        _, gi, gw, gb = _run_flce(_input, weight, target)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        tol = _ADDMM_TOLS[dtype]
        assert torch.isfinite(gw).all()
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)

    # -----------------------------------------------------------------------
    # 6. Explicit fp32 accumulator stays on the out_dtype fast path
    # -----------------------------------------------------------------------

    @requires_sm80_or_higher
    @requires_addmm_out_dtype
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fp32_accum_uses_dtype_out_fastpath(self, dtype):
        BT, V, H = 40, 128, 32  # 3 chunks (16,16,8)
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, dtype, bias=False, ce_weight=False, seed=8)
        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, accum_dtype=torch.float32, recorder=rec)
        # Accumulator is fp32; the returned grad is cast back to the weight dtype.
        assert gw.dtype == weight.dtype
        # The fp32 out_dtype overload must be used once per chunk (multichunk loop),
        # and the plain out overload must not.
        assert rec.count_addmm_dtype_out_into(weight.shape) >= 2, (
            f"expected >=2 aten.addmm.dtype_out into {tuple(weight.shape)} (multichunk), saw {rec.addmm_dtype_out_shapes}"
        )
        assert not rec.has_addmm_out_into(weight.shape)
        # No parameter-sized bf16/fp16 -> fp32 legacy temporary.
        assert not rec.has_param_upcast(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gw.float(), ref_gw, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gi.float(), ref_gi, **_ADDMM_TOLS[dtype])

    # -----------------------------------------------------------------------
    # 7. Pre-2.8 torch (no out_dtype support) -> legacy fallback, never out_dtype op
    # -----------------------------------------------------------------------

    @requires_sm80_or_higher
    @pytest.mark.parametrize("accum_dtype", [None, torch.float32])
    def test_pre28_falls_back_to_legacy_mm(self, monkeypatch, accum_dtype):
        BT, V, H = 40, 128, 32
        # Under bf16 params + fp32 accumulator, the fast path needs out_dtype support.
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, torch.bfloat16, bias=False, ce_weight=False, seed=9)
        monkeypatch.setattr(flce_ops, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, accum_dtype=accum_dtype, recorder=rec)
        # The out_dtype overload must never be reached when the flag is False.
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        if accum_dtype is torch.float32:
            # fp32 accumulator falls back to legacy torch.mm(...).float(): mm + a
            # parameter-sized fp32 upcast, and no addmm at all into the weight buffer.
            assert rec.mm_default >= 1
            assert rec.has_param_upcast(weight.shape)
            assert not rec.has_addmm_out_into(weight.shape)
        else:
            # bf16 accumulator does not depend on the out_dtype flag; the dtype-matched
            # low-precision addmm(out=) path is still taken (no legacy upcast).
            assert rec.has_addmm_out_into(weight.shape)
            assert not rec.has_param_upcast(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gw.float(), ref_gw, **_ADDMM_TOLS[torch.bfloat16])
        torch.testing.assert_close(gi.float(), ref_gi, **_ADDMM_TOLS[torch.bfloat16])

    # -----------------------------------------------------------------------
    # 8. AMP: in-place out ops are not autocast, so _input_chunk must be aligned
    # -----------------------------------------------------------------------

    @requires_sm80_or_higher
    def test_amp_bf16_params_fp32_input_uses_lowprec_addmm(self):
        """BF16 params + FP32 input under bf16 autocast: grad_weight is bf16 and
        grad_logits is bf16, so the dtype-matched low-precision addmm(out=) path runs.
        The fp32 _input_chunk must be cast to bf16 for the (non-autocast) out op."""
        BT, V, H = 40, 128, 32
        g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(10)
        _input = (torch.randn(BT, H, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        weight = (torch.randn(V, H, device=_ADDMM_DEVICE, dtype=torch.bfloat16, generator=g) * 0.1).requires_grad_(True)
        target = torch.randint(0, V, (BT,), device=_ADDMM_DEVICE, generator=g, dtype=torch.long)

        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, recorder=rec, autocast_dtype=torch.bfloat16)
        assert gw.dtype == torch.bfloat16
        assert rec.has_addmm_out_into(weight.shape)
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        assert not rec.has_param_upcast(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        tol = _ADDMM_TOLS[torch.bfloat16]
        torch.testing.assert_close(gw.float(), ref_gw, **tol)
        torch.testing.assert_close(gi.float(), ref_gi, **tol)

    @requires_sm80_or_higher
    @requires_addmm_out_dtype
    def test_amp_fp32_params_no_bias_uses_dtype_out_fastpath(self):
        """FP32 params, no bias, under bf16 autocast: grad_weight is fp32 and
        grad_logits is bf16, so the out_dtype fast path runs (with _input_chunk
        left fp32 and aligned to bf16 for addmm)."""
        BT, V, H = 40, 128, 32
        g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(11)
        _input = (torch.randn(BT, H, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        weight = (torch.randn(V, H, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        target = torch.randint(0, V, (BT,), device=_ADDMM_DEVICE, generator=g, dtype=torch.long)

        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, recorder=rec, autocast_dtype=torch.bfloat16)
        assert gw.dtype == torch.float32
        assert rec.has_addmm_dtype_out_into(weight.shape)
        assert not rec.has_addmm_out_into(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gw.float(), ref_gw, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gi.float(), ref_gi, atol=1e-2, rtol=1e-2)

    def test_amp_fp32_params_fp32_bias_promotes_and_falls_back(self):
        """FP32 params + FP32 bias under bf16 autocast: the (bf16) logits + (fp32)
        bias add promotes grad_logits to fp32, so neither addmm fast path applies and
        the legacy torch.mm(...).float() path must be used (preserving prior behavior).
        This guards against accidentally running an fp32 addmm for promoted logits."""
        BT, V, H = 40, 128, 32
        g = torch.Generator(device=_ADDMM_DEVICE).manual_seed(12)
        _input = (torch.randn(BT, H, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        weight = (torch.randn(V, H, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        bias = (torch.randn(V, device=_ADDMM_DEVICE, dtype=torch.float32, generator=g) * 0.1).requires_grad_(True)
        target = torch.randint(0, V, (BT,), device=_ADDMM_DEVICE, generator=g, dtype=torch.long)

        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, bias=bias, recorder=rec, autocast_dtype=torch.bfloat16)
        # grad_logits promoted to fp32 -> no fp32-out fast path, no low-prec out path.
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        assert not rec.has_addmm_out_into(weight.shape)
        assert rec.mm_default >= 1
        _, ref_gi, ref_gw, ref_gb = _addmm_reference_grads(_input, weight, target, bias=bias)
        torch.testing.assert_close(gw.float(), ref_gw, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gi.float(), ref_gi, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gb.float(), ref_gb, atol=1e-2, rtol=1e-2)

    @requires_sm80_or_higher
    def test_amp_use_token_scaling_lowprec(self):
        """Scaled-loss (use_token_scaling) under bf16 params still routes through the
        dtype-matched low-precision addmm(out=) path and stays numerically finite."""
        BT, V, H = 24, 96, 32
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, torch.bfloat16, bias=True, ce_weight=False, seed=13)
        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, bias=b, use_token_scaling=True, recorder=rec)
        assert gw.dtype == torch.bfloat16
        assert rec.has_addmm_out_into(weight.shape)
        assert not rec.has_param_upcast(weight.shape)
        assert torch.isfinite(gw).all() and torch.isfinite(gi).all()

    # -----------------------------------------------------------------------
    # 9. Capability guard: sub-SM80 reported capability forces the legacy fallback
    # -----------------------------------------------------------------------

    def test_sm_capability_guard_forces_legacy_fallback(self, monkeypatch):
        """Guard regression -- intentionally *not* SM80-gated. Even on real SM80+
        hardware, production gates the addmm fast paths on ``get_device_capability()
        [0] >= 8``; if the reported capability is < 8 the fast paths must not run.
        Monkeypatch the capability to (7, 0) and assert the legacy
        ``torch.mm(...).float()`` fallback is taken (bf16 params, which would
        otherwise hit the low-precision addmm(out=) path)."""
        BT, V, H = 40, 128, 32
        _input, weight, b, cw, target = _mk_inputs(BT, V, H, torch.bfloat16, bias=False, ce_weight=False, seed=14)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
        rec = _OpRecorder()
        _, gi, gw, gb = _run_flce(_input, weight, target, recorder=rec)
        # Sub-SM80 capability -> neither addmm fast path; legacy mm + fp32 upcast instead.
        assert not rec.has_addmm_out_into(weight.shape)
        assert not rec.has_addmm_dtype_out_into(weight.shape)
        assert rec.mm_default >= 1
        assert rec.has_param_upcast(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gw.float(), ref_gw, **_ADDMM_TOLS[torch.bfloat16])
        torch.testing.assert_close(gi.float(), ref_gi, **_ADDMM_TOLS[torch.bfloat16])
