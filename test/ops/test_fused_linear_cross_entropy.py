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

from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

import liger_kernel

# Importing functional registers the discovery map so available_backends works.
import liger_kernel.functional  # noqa: F401
import liger_kernel.ops.fused_linear_cross_entropy as flce_ops

from liger_kernel.backends.dispatch import available_backends
from liger_kernel.backends.dispatch import dispatch
from liger_kernel.backends.registry import get_registered
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

from .conftest import device
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

    inp = torch.randn(8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, 256, (8,), device=device)

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
# memory budget (``inc_factor = cdiv(V, C * H)``). The public default is C=1,
# so the small shapes below naturally exercise the real multi-chunk path.
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

requires_nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or flce_ops.is_hip(),
    reason="reusable mm(out=) buffers require NVIDIA CUDA",
)


def _production_chunk_size(BT: int, V: int, H: int, chunk_mem_const: int | None = None) -> int:
    """Mirror the production token-chunk geometry for the given dimensions."""
    chunk_mem_const = flce_ops._CHUNK_MEM_CONST if chunk_mem_const is None else chunk_mem_const
    inc_factor = triton.cdiv(V, chunk_mem_const * H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    return min(chunk_size, BT)


def test_triton_flce_chunk_memory_budget():
    assert flce_ops._CHUNK_MEM_CONST == 1
    assert _production_chunk_size(BT=8192, V=128256, H=4096) == 256
    assert _production_chunk_size(BT=8192, V=128256, H=4096, chunk_mem_const=8) == 2048


# ---------------------------------------------------------------------------
# Dispatch recorder
# ---------------------------------------------------------------------------

_ADDMM_OPS = ("aten.addmm.out", "aten.addmm.dtype_out")


class _OpRecorder(TorchDispatchMode):
    """Records the aten ops we care about: addmm (with the out-tensor shape and
    whether the out_dtype overload was used), mm outputs/copies, and dtype casts.
    Store metadata only, so the recorder cannot extend a buffer's lifetime."""

    def __init__(self):
        self.addmm_out_shapes = []  # list of torch.Size for aten.addmm.out
        self.addmm_dtype_out_shapes = []  # list of torch.Size for aten.addmm.dtype_out
        self.mm_default = 0  # count of aten.mm.default
        self.mm_default_shapes = []
        self.mm_out = []  # (shape, storage pointer, data pointer)
        self.copies = []  # (destination shape, destination storage pointer)
        self.empty_shapes = []
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
            self.mm_default_shapes.append((args[0].shape[0], args[1].shape[1]))
        elif name == "aten.mm.out":
            out = kwargs["out"]
            self.mm_out.append((tuple(out.shape), out.untyped_storage().data_ptr(), out.data_ptr()))
        elif name == "aten.copy_.default":
            dst = args[0]
            self.copies.append((tuple(dst.shape), dst.untyped_storage().data_ptr()))
        elif name == "aten.empty.memory_format":
            self.empty_shapes.append(tuple(args[0]))
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


class _BufferRecorder(_OpRecorder):
    """Check storage lifetime, including views that outlive their base tensor.

    StorageWeakRef observes the C++ storage without keeping it alive. Python
    tensor weakrefs alone would miss aliases such as grad_logits_t and ce_args.
    """

    def __init__(self, BT, V, H):
        super().__init__()
        self.V, self.H = V, H
        self.chunk_size = _production_chunk_size(BT, V, H)
        self.chunk_storages = []
        self.logits_gemms = 0
        self.final_casts = 0

    def assert_released(self, reusable_ptr=None):
        assert all(ref.expired() or ptr == reusable_ptr for ptr, ref in self.chunk_storages), (
            "a logits/probability/scaled-gradient storage is still live"
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in ("aten.mm.default", "aten.mm.out") and tuple(args[1].shape) == (self.H, self.V):
            out = kwargs.get("out")
            reusable_ptr = out.untyped_storage().data_ptr() if out is not None else None
            self.assert_released(reusable_ptr)
            self.logits_gemms += 1
        if (
            name == "aten._to_copy.default"
            and tuple(args[0].shape) == (self.V, self.H)
            and args[0].dtype == torch.float32
            and kwargs.get("dtype") in (torch.bfloat16, torch.float16)
        ):
            self.assert_released()
            self.final_casts += 1

        result = super().__torch_dispatch__(func, types, args, kwargs)
        if (
            isinstance(result, torch.Tensor)
            and result.ndim == 2
            and result.shape[1] == self.V
            and 0 < result.shape[0] <= self.chunk_size
        ):
            storage = result.untyped_storage()
            self.chunk_storages.append((storage.data_ptr(), StorageWeakRef(storage)))
        return result


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
    ce_impl=None,
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
            ce_impl=ce_impl,
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

_SHARED_CE_IMPLS = [None] + [b for b in _REGISTERED_BACKENDS if b in ("nvidia-triton", "nvidia-cutedsl")]


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
    """Regression suite for chunk buffers and dW accumulation."""

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
            assert rec.mm_default_shapes.count(tuple(weight.shape)) == triton.cdiv(BT, _production_chunk_size(BT, V, H))
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
    # Logits storage reuse and direct dX output, including ragged/strided views
    # -----------------------------------------------------------------------

    @requires_nvidia_cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
    def test_reuses_logits_and_writes_dx_out(self, monkeypatch, dtype, bias, layout):
        BT, V, H = 40, 128, 32
        _input, weight, b, _, target = _mk_inputs(BT, V, H, dtype, bias=bias, ce_weight=False, seed=15)
        if layout == "transposed":
            _input = _input.t().contiguous().t().detach().requires_grad_(True)
            weight = weight.t().contiguous().t().detach().requires_grad_(True)
        elif layout == "strided":
            _input = _input.repeat_interleave(2, dim=1)[:, ::2].detach().requires_grad_(True)
            weight = weight.repeat_interleave(2, dim=1)[:, ::2].detach().requires_grad_(True)
        if layout != "contiguous":
            assert not _input.is_contiguous() and not weight.is_contiguous()
        originals = tuple(t.detach().clone() if t is not None else None for t in (_input, weight, b))

        rec = _BufferRecorder(BT, V, H)
        actual = _run_flce(_input, weight, target, bias=b, recorder=rec)
        gi = actual[1]
        cs = _production_chunk_size(BT, V, H)
        rows = [min(cs, BT - start) for start in range(0, BT, cs)]
        assert len(rows) > 1 and rows[-1] < cs
        logits_out = [entry for entry in rec.mm_out if entry[0][1] == V]
        dx_out = [entry for entry in rec.mm_out if entry[0][1] == H]
        assert [shape for shape, _, _ in logits_out] == [(n, V) for n in rows]
        assert len({ptr for _, ptr, _ in logits_out}) == 1
        assert len({ptr for _, _, ptr in logits_out}) == 1
        assert rec.empty_shapes.count((cs, V)) == 1
        assert [shape for shape, _, _ in dx_out] == [(n, H) for n in rows]
        assert {ptr for _, ptr, _ in dx_out} == {gi.untyped_storage().data_ptr()}
        assert [ptr for _, _, ptr in dx_out] == [
            gi.data_ptr() + start * gi.stride(0) * gi.element_size() for start in range(0, BT, cs)
        ]
        assert all(ptr != gi.untyped_storage().data_ptr() for _, ptr in rec.copies)
        # Any remaining allocating mm must be the unchanged dW fallback, never logits/dX.
        assert all(shape == (V, H) for shape in rec.mm_default_shapes)
        assert rec.logits_gemms == len(rows)
        rec.assert_released()
        torch.testing.assert_close((_input, weight, b), originals, atol=0, rtol=0)

        monkeypatch.setattr(flce_ops, "_can_use_mm_out", lambda *args: False)
        expected = _run_flce(_input, weight, target, bias=b)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("ce_impl", _SHARED_CE_IMPLS)
    @pytest.mark.parametrize("buffer_path", ["reuse", "forced-fallback", "promoted-bias"])
    @pytest.mark.parametrize("use_token_scaling", [False, True])
    @pytest.mark.parametrize("forward_only", [False, True])
    def test_chunk_storage_released_before_next_chunk_and_final_cast(
        self, monkeypatch, ce_impl, buffer_path, use_token_scaling, forward_only
    ):
        BT, V, H = 40, 128, 32
        cs = _production_chunk_size(BT, V, H)
        _input, weight, b, cw, target = _mk_inputs(
            BT,
            V,
            H,
            torch.bfloat16,
            bias=True,
            ce_weight=True,
            seed=16,
            first_chunk_size=cs,
        )
        reuse = buffer_path == "reuse" and not flce_ops.is_hip()
        promoted_bias = buffer_path == "promoted-bias"
        if buffer_path == "forced-fallback":
            monkeypatch.setattr(flce_ops, "_can_use_mm_out", lambda *args: False)
        if promoted_bias:
            b = b.detach().float().requires_grad_(True)
        kwargs = dict(
            bias=b,
            ce_weight=cw,
            accum_dtype=torch.float32,
            use_token_scaling=use_token_scaling,
            compute_gradients=not forward_only,
            softcap=1.5,
            label_smoothing=0.1,
            lse_square_scale=1e-4,
            return_z_loss=True,
            return_token_accuracy=True,
            return_predicted_tokens=True,
            ce_impl=ce_impl,
        )
        rec = _BufferRecorder(BT, V, H)
        with torch.no_grad(), torch.autocast("cuda", enabled=promoted_bias, dtype=torch.bfloat16), rec:
            actual = fused_linear_cross_entropy_forward(_input, weight, target, **kwargs)
        assert rec.logits_gemms == triton.cdiv(BT, cs)
        assert rec.final_casts == (0 if forward_only else 1)
        assert bool(rec.mm_out) == reuse
        rec.assert_released()
        if forward_only:
            assert actual[5] is None and actual[6] is None
            assert torch.count_nonzero(actual[4]) == 0
            assert len(rec.mm_out) == (triton.cdiv(BT, cs) if reuse else 0)

        monkeypatch.setattr(flce_ops, "_can_use_mm_out", lambda *args: False)
        with torch.no_grad(), torch.autocast("cuda", enabled=promoted_bias, dtype=torch.bfloat16):
            expected = fused_linear_cross_entropy_forward(_input, weight, target, **kwargs)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "input_dtype,weight_dtype,bias_dtype,amp_dtype,reuse,dx_out",
        [
            (torch.bfloat16, torch.bfloat16, None, torch.bfloat16, True, True),
            (torch.float16, torch.float16, torch.float16, torch.float16, True, True),
            (torch.float32, torch.float32, None, torch.bfloat16, False, False),
            (torch.float32, torch.bfloat16, None, torch.bfloat16, False, False),
            (torch.bfloat16, torch.float32, None, torch.bfloat16, False, False),
            (torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, False, False),
            (torch.float32, torch.float32, torch.float32, torch.bfloat16, False, False),
            (torch.float16, torch.float16, None, torch.bfloat16, False, False),
            (torch.float32, torch.float32, torch.bfloat16, None, False, True),
        ],
    )
    def test_buffer_amp_gates_preserve_matmul_and_bias_semantics(
        self, monkeypatch, input_dtype, weight_dtype, bias_dtype, amp_dtype, reuse, dx_out
    ):
        reuse = reuse and not flce_ops.is_hip()
        dx_out = dx_out and not flce_ops.is_hip()
        BT, V, H = 40, 128, 32
        cs = _production_chunk_size(BT, V, H)
        _input, weight, b, _, target = _mk_inputs(
            BT, V, H, weight_dtype, input_dtype=input_dtype, bias=bias_dtype is not None, ce_weight=False, seed=17
        )
        if b is not None:
            b = b.detach().to(bias_dtype).requires_grad_(True)

        expected_logits = []
        with torch.no_grad(), torch.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            for start in range(0, BT, cs):
                logits = _input[start : start + cs] @ weight.t()
                expected_logits.append(logits if b is None else logits + b)

        grad_logits = []
        real_dispatch = flce_ops.dispatch

        def check_ce_logits(op, *args, **kwargs):
            torch.testing.assert_close(args[0], expected_logits[len(grad_logits)], atol=0, rtol=0)
            result = real_dispatch(op, *args, **kwargs)
            grad_logits.append(result[-1].clone())
            return result

        monkeypatch.setattr(flce_ops, "dispatch", check_ce_logits)
        rec = _OpRecorder()
        actual = _run_flce(
            _input, weight, target, bias=b, autocast_dtype=amp_dtype, ce_impl="nvidia-triton", recorder=rec
        )
        n_chunks = triton.cdiv(BT, cs)
        assert sum(shape[1] == V for shape, _, _ in rec.mm_out) == (n_chunks if reuse else 0)
        assert sum(shape[1] == H for shape, _, _ in rec.mm_out) == (n_chunks if dx_out else 0)
        assert rec.empty_shapes.count((cs, V)) == (1 if reuse else 0)
        assert len(grad_logits) == n_chunks
        with torch.no_grad(), torch.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            expected_dx = torch.cat([g @ weight for g in grad_logits]).to(_input.dtype)
        torch.testing.assert_close(actual[1], expected_dx, atol=0, rtol=0)

        monkeypatch.setattr(flce_ops, "dispatch", real_dispatch)
        monkeypatch.setattr(flce_ops, "_can_use_mm_out", lambda *args: False)
        expected = _run_flce(_input, weight, target, bias=b, autocast_dtype=amp_dtype, ce_impl="nvidia-triton")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @requires_nvidia_cuda
    def test_mm_out_gate_uses_legacy_autocast_query_on_old_torch(self, monkeypatch):
        _input = torch.empty(1, 1, device="cuda", dtype=torch.bfloat16)
        monkeypatch.setattr(flce_ops, "_TORCH_VERSION", flce_ops.Version("2.1.2"))
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
        monkeypatch.setattr(torch, "get_autocast_gpu_dtype", lambda: torch.bfloat16)

        def unexpected_modern_query(*args):
            pytest.fail("torch < 2.4 must not use get_autocast_dtype(device_type)")

        monkeypatch.setattr(torch, "get_autocast_dtype", unexpected_modern_query, raising=False)
        with torch.no_grad():
            assert flce_ops._can_use_mm_out(_input, _input)
            monkeypatch.setattr(torch, "get_autocast_gpu_dtype", lambda: torch.float16)
            assert not flce_ops._can_use_mm_out(_input, _input)
        assert not flce_ops._can_use_mm_out(_input, _input)

    def test_mm_out_gate_preserves_hip_fallback(self, monkeypatch):
        _input = torch.empty(1, 1, device="cuda", dtype=torch.float32)
        monkeypatch.setattr(flce_ops, "is_hip", lambda: True)
        with torch.no_grad():
            assert not flce_ops._can_use_mm_out(_input, _input)

    @pytest.mark.parametrize("ce_impl", _SHARED_CE_IMPLS)
    @pytest.mark.parametrize("use_token_scaling", [False, True])
    @pytest.mark.parametrize("weight_requires_grad", [False, True])
    def test_reduction_none_reenters_with_reusable_buffers(
        self, monkeypatch, ce_impl, use_token_scaling, weight_requires_grad
    ):
        BT, V, H = 40, 128, 32
        cs = _production_chunk_size(BT, V, H)
        _input, weight, b, cw, target = _mk_inputs(
            BT,
            V,
            H,
            torch.bfloat16,
            bias=True,
            ce_weight=True,
            seed=18,
            first_chunk_size=cs,
            weight_requires_grad=weight_requires_grad,
        )
        upstream = torch.linspace(-0.5, 1.5, BT, device="cuda")

        def run():
            outputs = flce_ops.LigerFusedLinearCrossEntropyFunction.apply(
                _input,
                weight,
                target,
                b,
                cw,
                -100,
                1e-4,
                0.1,
                "none",
                1.5,
                True,
                torch.float32,
                use_token_scaling,
                True,
                True,
                ce_impl,
            )
            outputs[0].backward(upstream)
            return outputs, _input.grad, weight.grad, b.grad

        rec = _BufferRecorder(BT, V, H)
        with rec:
            actual = run()
        assert rec.logits_gemms == 2 * triton.cdiv(BT, cs)
        assert rec.final_casts == int(weight_requires_grad)
        expected_mm_out = 0 if flce_ops.is_hip() else 3 * triton.cdiv(BT, cs)
        assert len(rec.mm_out) == expected_mm_out  # logits twice, dX only during backward
        rec.assert_released()
        assert torch.count_nonzero(actual[1][:cs]) == 0
        if not weight_requires_grad:
            assert actual[2] is None
            assert not rec.has_addmm_out_into(weight.shape)
            assert not rec.has_addmm_dtype_out_into(weight.shape)

        _input.grad = weight.grad = b.grad = None
        monkeypatch.setattr(flce_ops, "_can_use_mm_out", lambda *args: False)
        expected = run()
        torch.testing.assert_close(actual, expected)

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
        assert rec.mm_default_shapes.count(tuple(weight.shape)) == triton.cdiv(BT, _production_chunk_size(BT, V, H))
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
        assert rec.mm_default_shapes.count(tuple(weight.shape)) == triton.cdiv(BT, _production_chunk_size(BT, V, H))
        assert rec.has_param_upcast(weight.shape)
        _, ref_gi, ref_gw, _ = _addmm_reference_grads(_input, weight, target)
        torch.testing.assert_close(gw.float(), ref_gw, **_ADDMM_TOLS[torch.bfloat16])
        torch.testing.assert_close(gi.float(), ref_gi, **_ADDMM_TOLS[torch.bfloat16])
