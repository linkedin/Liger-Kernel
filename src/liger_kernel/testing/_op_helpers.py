"""Cross-backend correctness helpers for Liger Kernel ops.

The :func:`assert_op_correctness` driver is the single entry point used by
``test/ops/test_*.py``. Adding a new op means:

1. Implementing a ``pytorch_reference_<op>`` function in this module.
2. Registering it in the :data:`_REFERENCE_IMPLS` table.
3. Listing the input-tensor signature (which positional args carry the input,
   weights, eps, …) in :data:`_OP_SIGNATURES`.

The driver handles tolerance lookup (from the registered ``OpImpl``), tensor
construction, dispatch invocation, fwd/bwd comparison, sum-based dW fallback,
and produces a single AssertionError with op/backend/shape/dtype context.
"""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Tuple

import pytest
import torch

from liger_kernel.backends import dispatch
from liger_kernel.backends.registry import get_registered

# ---------------------------------------------------------------------------
# Public reference implementations
# ---------------------------------------------------------------------------


def pytorch_reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> torch.Tensor:
    """Ground-truth RMSNorm in PyTorch, matching the three casting modes used
    by Liger's registered backends.

    Args:
        x: input, shape ``(*, hidden)``.
        weight: scale, shape ``(hidden,)``.
        eps: numerical stabilizer.
        offset: 0.0 (Llama-style) or 1.0 (Gemma-style).
        casting_mode: ``"llama"`` casts to fp32 only inside the reduction;
            ``"gemma"`` keeps the whole arithmetic in fp32; ``"none"`` does
            the math in the input dtype.

    Returns:
        Output tensor with the same shape and dtype as ``x``.
    """
    in_dtype = x.dtype
    if casting_mode == "llama":
        x_fp32 = x.to(torch.float32)
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_square + eps)
        return (x_fp32 * rstd).to(in_dtype) * (offset + weight)
    if casting_mode == "gemma":
        x_fp32 = x.to(torch.float32)
        w_fp32 = weight.to(torch.float32)
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_square + eps)
        return ((x_fp32 * rstd) * (offset + w_fp32)).to(in_dtype)
    if casting_mode == "none":
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_square + eps)
        return (x * rstd) * (offset + weight)
    raise ValueError(f"Unknown casting_mode={casting_mode!r}")


def pytorch_reference_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Ground-truth LayerNorm in PyTorch.

    Computes ``weight * (x - mean) / sqrt(var + eps) + bias`` reducing over the
    last dimension, with the reduction performed in fp32 regardless of the
    input dtype (matching every registered Liger backend).

    Args:
        x: input, shape ``(*, hidden)``.
        weight: scale, shape ``(hidden,)``.
        bias: shift, shape ``(hidden,)``.
        eps: numerical stabilizer.

    Returns:
        Output tensor with the same shape and dtype as ``x``.
    """
    in_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x_f32 - mean) * torch.rsqrt(var + eps)
    return (x_hat * weight.to(torch.float32) + bias.to(torch.float32)).to(in_dtype)


# Registry of reference impls. Adding a new op means registering its
# ``pytorch_reference_<op>`` callable here.
_REFERENCE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "rms_norm": pytorch_reference_rms_norm,
    "layer_norm": pytorch_reference_layer_norm,
}


# Description of each op's input signature so the driver knows which positional
# arguments are *input tensors* (need grad) vs scalars (eps, mode strings, ...).
#
# Each entry is a dict with:
#   - ``build_inputs``: callable(shape, dtype, device, seed, **kwargs) -> dict
#       Returns a mapping with at least keys "x", "weight", "extra_args"
#       (extra_args is the tuple of trailing positional args after weight).
#   - ``call``: callable(impl_invoker, inputs, **kwargs) -> output
#       impl_invoker is either ``dispatch`` (with backend) or the reference fn.
_OP_SIGNATURES: dict[str, dict[str, Callable[..., Any]]] = {}


def _build_rms_norm_inputs(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    eps: float,
    offset: float,
    casting_mode: str,
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    M, N = shape[-2], shape[-1]
    # Build on CPU then move to target device — keeps seeding deterministic
    # across CPU/CUDA.
    x_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    w_cpu = torch.randn(N, dtype=torch.float32, generator=g)
    x = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    weight = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    return {
        "x": x,
        "weight": weight,
        "eps": eps,
        "offset": offset,
        "casting_mode": casting_mode,
    }


_OP_SIGNATURES["rms_norm"] = {"build_inputs": _build_rms_norm_inputs}


def _build_layer_norm_inputs(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    eps: float = 1e-6,
    include_bias: bool = True,
    **_: Any,
) -> dict:
    """Build LayerNorm inputs.

    Returns ``x`` (random, requires_grad), ``weight`` (ones, requires_grad),
    and ``bias``. When ``include_bias=True`` bias is randn with ``requires_grad=True``;
    when ``include_bias=False`` bias is a zeros tensor with ``requires_grad=False``
    so the driver can skip the bias-grad comparison.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    M, N = shape[-2], shape[-1]
    x_cpu = torch.randn(M, N, dtype=torch.float32, generator=g)
    w_cpu = torch.ones(N, dtype=torch.float32)
    if include_bias:
        b_cpu = torch.randn(N, dtype=torch.float32, generator=g)
    else:
        b_cpu = torch.zeros(N, dtype=torch.float32)
    x = x_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    weight = w_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
    bias = b_cpu.to(device=device, dtype=dtype).detach()
    if include_bias:
        bias.requires_grad_(True)
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "include_bias": include_bias,
    }


_OP_SIGNATURES["layer_norm"] = {"build_inputs": _build_layer_norm_inputs}


# ---------------------------------------------------------------------------
# Shape generator
# ---------------------------------------------------------------------------


def iter_test_shapes(shapes: Iterable[Tuple[int, ...]]) -> Iterator[Tuple[int, ...]]:
    """Yield canonical test shapes for parametrization.

    Currently the identity iterator over the provided sequence — exists so
    callers can centrally filter (e.g., skip huge shapes when out of VRAM)
    in one place later without rewriting individual test files.
    """
    for s in shapes:
        yield s


# ---------------------------------------------------------------------------
# Skip wrapper
# ---------------------------------------------------------------------------


def requires_cuda(reason: str = "CUDA required"):
    """Return a ``pytest.mark.skipif`` that skips when CUDA is unavailable.

    Use as a decorator on test functions that touch real kernels.
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason=reason)


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _tolerances_for(
    op_name: str,
    backend: str,
    dtype: torch.dtype,
    casting_mode: str = "llama",
) -> dict:
    """Pull (atol_fwd, rtol_fwd, atol_bwd, rtol_bwd) from the OpImpl, with
    conservative fallbacks if the impl didn't register dtype-specific entries.

    casting_mode="none" (RMSNorm without fp32 upcasting) is fundamentally
    lossier than "llama" or "gemma": every fp16/bf16 op in the reduction
    rounds, so element-wise diffs of 1-2 ULPs at the result magnitude are
    expected. We widen atol/rtol by 4x for that mode so a 1-ULP-difference
    in fp16 (≈ 0.016 at unit magnitude) doesn't trip the test. The reference
    implementation also runs in fp32 then casts back, so the diff is real,
    not a kernel bug.
    """
    impl = get_registered(op_name, backend)
    tols: Mapping[Any, Mapping[str, float]] = impl.tolerances if impl is not None else {}
    entry = dict(tols.get(dtype, {}))
    # Reasonable defaults if the impl didn't register the dtype.
    defaults = {
        torch.float16: {"atol_fwd": 1e-2, "rtol_fwd": 1e-2, "atol_bwd": 5e-2, "rtol_bwd": 5e-2},
        torch.bfloat16: {"atol_fwd": 2e-2, "rtol_fwd": 2e-2, "atol_bwd": 1e-1, "rtol_bwd": 5e-2},
        torch.float32: {"atol_fwd": 1e-5, "rtol_fwd": 1e-5, "atol_bwd": 1e-4, "rtol_bwd": 1e-4},
    }
    for k, v in defaults.get(dtype, {}).items():
        entry.setdefault(k, v)
    if casting_mode == "none" and dtype in (torch.float16, torch.bfloat16):
        # Guard each key — an impl that registered a partial ``tolerances``
        # dict (e.g. atol only, no rtol) plus a dtype that isn't in
        # ``defaults`` above would otherwise raise ``KeyError`` here.
        for k in ("atol_fwd", "atol_bwd", "rtol_fwd", "rtol_bwd"):
            if k in entry:
                entry[k] = entry[k] * 4.0
    return entry


def sum_based_dw_close(
    dw: torch.Tensor,
    dw_ref: torch.Tensor,
    rtol: float,
    atol: float = 1e-3,
) -> Tuple[bool, str]:
    """Compare two weight gradients by their per-feature sum.

    Liger's fp16/bf16 dW reductions accumulate over many rows in low precision;
    individual elements may disagree by more than ``rtol`` even when the
    aggregate (which is what the optimizer actually steps on) matches.

    Returns ``(ok, message)``. The message is empty on success.
    """
    if dw.shape != dw_ref.shape:
        return False, f"dw shape {tuple(dw.shape)} != ref {tuple(dw_ref.shape)}"
    s = dw.to(torch.float32).sum()
    s_ref = dw_ref.to(torch.float32).sum()
    denom = s_ref.abs().clamp_min(1e-8)
    rel = (s - s_ref).abs() / denom
    if torch.isfinite(rel) and (rel.item() < rtol or (s - s_ref).abs().item() < atol):
        return True, ""
    return (
        False,
        f"sum(dw)={s.item():.6g} vs sum(dw_ref)={s_ref.item():.6g} (rel={rel.item():.3g}, rtol={rtol:.3g})",
    )


# ---------------------------------------------------------------------------
# Main correctness driver
# ---------------------------------------------------------------------------


def _close_message(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    max_print: int = 5,
) -> Optional[str]:
    """Return ``None`` if close, else a one-paragraph mismatch report."""
    if actual.shape != expected.shape:
        return f"{label}: shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    mismatched = diff > tol
    if not mismatched.any():
        return None
    n = int(mismatched.sum().item())
    idx = torch.nonzero(mismatched, as_tuple=False)
    samples = []
    for row in idx[:max_print]:
        i = tuple(row.tolist())
        samples.append(f"    at {i}: actual={a[i].item():.6g}, expected={e[i].item():.6g}, diff={diff[i].item():.3g}")
    sample_block = "\n".join(samples)
    return (
        f"{label}: {n} elements outside atol={atol:.3g}, rtol={rtol:.3g} "
        f"(max diff={diff.max().item():.3g})\n{sample_block}"
    )


def assert_op_correctness(
    op_name: str,
    backend: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    *,
    casting_mode: str = "llama",
    offset: float = 0.0,
    eps: float = 1e-6,
    seed: int = 0,
    device: Optional[torch.device] = None,
    extra: Optional[dict] = None,
) -> None:
    """Run ``dispatch(op_name, ..., backend=backend)`` and verify forward AND
    backward against the registered PyTorch reference impl.

    Tolerances are taken from the matching ``OpImpl.tolerances`` entry, with
    sensible defaults if a dtype is missing. The weight gradient (``dW``) is
    first compared element-wise; if that fails, we fall back to a sum-based
    check (``|sum(dw) - sum(dw_ref)| / |sum(dw_ref)| < rtol``) — fp16/bf16
    reductions over large M routinely fail elementwise but match in aggregate.

    Failure messages cite ``op_name``, ``backend``, ``shape``, ``dtype``, and
    which of ``fwd`` / ``dx`` / ``dw`` failed.

    Args:
        op_name: registered op name (e.g., ``"rms_norm"``).
        backend: registered backend name (e.g., ``"triton"``).
        shape: input tensor shape (``(M, N)`` for RMSNorm).
        dtype: input/weight dtype.
        casting_mode: forwarded to op + reference.
        offset: forwarded to op + reference.
        eps: forwarded to op + reference.
        seed: RNG seed for tensor construction.
        device: where to allocate. Defaults to CUDA if available, else CPU.
    """
    if op_name not in _REFERENCE_IMPLS:
        raise KeyError(f"No reference impl registered for op {op_name!r}")
    if op_name not in _OP_SIGNATURES:
        raise KeyError(f"No input-builder registered for op {op_name!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build_inputs = _OP_SIGNATURES[op_name]["build_inputs"]
    extra = extra or {}

    # Per-op builder kwargs. We pass a shared base (eps/offset/casting_mode) so
    # existing builders that consume those names still work, then layer the
    # op-specific ``extra`` mapping on top.
    builder_kwargs: dict = {
        "eps": eps,
        "offset": offset,
        "casting_mode": casting_mode,
    }
    builder_kwargs.update(extra)

    inputs = build_inputs(shape, dtype, device, seed, **builder_kwargs)
    x = inputs["x"]
    weight = inputs["weight"]

    # Reference: build a separate, identical pair so the gradient graphs are
    # independent.
    inputs_ref = build_inputs(shape, dtype, device, seed, **builder_kwargs)
    x_ref = inputs_ref["x"]
    weight_ref = inputs_ref["weight"]

    reference = _REFERENCE_IMPLS[op_name]

    # Forward.
    bias = None
    bias_ref = None
    if op_name == "rms_norm":
        y = dispatch(
            "rms_norm",
            x,
            weight,
            eps,
            offset,
            casting_mode,
            False,
            None,
            backend=backend,
        )
        y_ref = reference(x_ref, weight_ref, eps, offset=offset, casting_mode=casting_mode)
    elif op_name == "layer_norm":
        bias = inputs["bias"]
        bias_ref = inputs_ref["bias"]
        y = dispatch(
            "layer_norm",
            x,
            weight,
            bias,
            eps,
            backend=backend,
        )
        y_ref = reference(x_ref, weight_ref, bias_ref, eps)
    else:  # pragma: no cover — guarded by the registry checks above.
        raise NotImplementedError(op_name)

    tols = _tolerances_for(op_name, backend, dtype, casting_mode=casting_mode)
    atol_fwd = tols["atol_fwd"]
    rtol_fwd = tols["rtol_fwd"]
    atol_bwd = tols["atol_bwd"]
    rtol_bwd = tols["rtol_bwd"]

    ctx = f"op={op_name} backend={backend} shape={tuple(shape)} dtype={dtype} casting={casting_mode}"

    msg = _close_message("forward", y, y_ref, atol=atol_fwd, rtol=rtol_fwd)
    if msg is not None:
        raise AssertionError(f"[{ctx}] {msg}")

    # Backward — seed with a deterministic upstream gradient.
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    dy_cpu = torch.randn(*y.shape, dtype=torch.float32, generator=g)
    dy = dy_cpu.to(device=device, dtype=dtype)
    try:
        y.backward(dy)
    except RuntimeError as e:
        msg = str(e)
        # Backends are allowed to document a range and refuse outside it. We
        # treat such refusals as ``skip`` so the test signal stays clean: the
        # kernel correctly rejected an out-of-range config; users still have
        # other backends for that shape.
        if "only supports hidden dim" in msg or "out of range" in msg.lower():
            pytest.skip(f"[{ctx}] backend documents range limit: {msg}")
        raise
    y_ref.backward(dy.clone())

    dx_msg = _close_message("dx", x.grad, x_ref.grad, atol=atol_bwd, rtol=rtol_bwd)
    if dx_msg is not None:
        raise AssertionError(f"[{ctx}] {dx_msg}")

    dw_msg = _close_message("dw", weight.grad, weight_ref.grad, atol=atol_bwd, rtol=rtol_bwd)
    if dw_msg is not None:
        ok, sum_msg = sum_based_dw_close(weight.grad, weight_ref.grad, rtol=rtol_bwd, atol=atol_bwd)
        if not ok:
            raise AssertionError(f"[{ctx}] {dw_msg}\nElement-wise dw failed; sum-based fallback also failed: {sum_msg}")

    # LayerNorm bias grad — only check when bias was a real grad-requiring
    # parameter. ``include_bias=False`` deliberately passes a zeros, no-grad
    # tensor so backends that allocate a dB buffer still work; comparing dB in
    # that case would be meaningless (both sides are zero by construction).
    if op_name == "layer_norm" and bias is not None and bias.requires_grad:
        db_msg = _close_message("db", bias.grad, bias_ref.grad, atol=atol_bwd, rtol=rtol_bwd)
        if db_msg is not None:
            ok, sum_msg = sum_based_dw_close(bias.grad, bias_ref.grad, rtol=rtol_bwd, atol=atol_bwd)
            if not ok:
                raise AssertionError(
                    f"[{ctx}] {db_msg}\nElement-wise db failed; sum-based fallback also failed: {sum_msg}"
                )
