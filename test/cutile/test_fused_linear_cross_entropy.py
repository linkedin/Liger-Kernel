import pytest
import torch
import torch.nn.functional as F

from torch.utils._python_dispatch import TorchDispatchMode

from test.utils import assert_verbose_allclose
from test.utils import set_seed

pytest.importorskip("cuda.tile")

import liger_kernel.ops.cutile.ops.fused_linear_cross_entropy as flce_mod
import liger_kernel.ops.fused_linear_cross_entropy as triton_flce_mod

from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import chunked_fused_linear_cross_entropy_backward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import chunked_fused_linear_cross_entropy_forward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
from liger_kernel.ops.fused_linear_cross_entropy import _get_chunk_size

# The cuTile FLCE kernels support both Hopper (SM90) and Blackwell (SM100).
_SUPPORTED_CAPABILITIES = ((9, 0), (10, 0))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuTile FLCE requires CUDA"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in _SUPPORTED_CAPABILITIES,
        reason="cuTile FLCE requires Hopper (SM90) or Blackwell (SM100)",
    ),
]

set_seed()


def _apply(x, weight, target, reduction="mean", ignore_index=-100, accum_dtype=None):
    """Invoke the Function with the 17-argument module-style contract.

    There is NO public ``chunk_size`` argument: the token-chunk geometry is the
    shared default Triton policy (:func:`_get_chunk_size`). ``ce_impl`` / ``ce_mode``
    are self-identity placeholders and default to ``None``. ``accum_dtype`` selects
    the dW accumulation buffer dtype (``None`` -> weight dtype/BF16,
    ``torch.float32`` -> FP32, explicit ``torch.bfloat16`` -> BF16).
    """
    return LigerFusedLinearCrossEntropyFunction.apply(
        x,
        weight,
        target,
        None,  # bias
        None,  # ce_weight
        ignore_index,
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        reduction,
        None,  # softcap
        False,  # return_z_loss
        accum_dtype,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        None,  # ce_impl
        None,  # ce_mode
    )


def _triton_apply(x, weight, target, reduction="mean", ignore_index=-100, accum_dtype=None):
    """Invoke the actual Triton FLCE autograd Function (the parity target)."""
    return triton_flce_mod.LigerFusedLinearCrossEntropyFunction.apply(
        x,
        weight,
        target,
        None,  # bias
        None,  # ce_weight
        ignore_index,
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        reduction,
        None,  # softcap
        False,  # return_z_loss
        accum_dtype,
    )


# Accumulation-storage policies exercised across the correctness/oracle tests:
# ``None`` and explicit BF16 both accumulate dW in the weight dtype (BF16) across
# chunks; ``torch.float32`` accumulates in FP32. Every one casts dW to the weight
# dtype exactly once at the end of the forward.
_ACCUM_MODES = [None, torch.bfloat16, torch.float32]


def _reference(x, weight, target, ignore_index, reduction):
    # Baseline: BF16 F.linear, then FP32 cross entropy (matches the kernel's math).
    return F.cross_entropy(
        F.linear(x, weight).float(),
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def _tolerances(reduction):
    atol = 5e-3 if reduction == "mean" else 5e-2
    return atol, 5e-2


def _relative_l2(actual, expected):
    return (torch.norm((actual - expected).float()) / torch.norm(expected.float()).clamp_min(1e-12)).item()


def _ref_dw_fp32(a, b):
    """Independent FP32 ``a @ b`` mirroring the production dW accumulation.

    Tracks ``flce_mod._ADDMM_SUPPORTS_OUT_DTYPE``: on torch>=2.8 the BF16 operands
    drive the ``out_dtype=torch.float32`` overload directly; on older torch (no
    overload) the bounded BF16 operands are upcast to FP32 and fed to the plain
    ``mm``. Both are numerically identical -- a BF16*BF16 product is exact in FP32
    -- so callers keep atol/rtol at 0.
    """
    if flce_mod._ADDMM_SUPPORTS_OUT_DTYPE:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.to(torch.float32), b.to(torch.float32))


def _expected_tiling(tokens, hidden_size, vocab_size):
    """Row counts of every token chunk under the shared default geometry."""
    chunk = _get_chunk_size(tokens, hidden_size, vocab_size)
    return [min(chunk, tokens - start) for start in range(0, tokens, chunk)]


# ---------------------------------------------------------------------------
# End-to-end correctness against the BF16-linear / FP32-CE reference on shapes
# that resolve to a SINGLE token chunk under the shared default geometry
# (V <= 16*H). One chunk means the projection GEMM tiling matches the reference,
# so the original tight elementwise tolerances apply. Shapes exercise a range of
# H (256-aligned and non-aligned), odd V, and > 4096 vocab (multiple partitions).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 256),  # H=128
        (129, 96, 384),  # H=96, tail token
        (32, 512, 8192),  # V > 4096 -> multiple vocab partitions, still one chunk
        (300, 256, 512),  # 256-aligned H
        (200, 128, 257),  # odd vocab size
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
def test_correctness(shape, reduction, accum_dtype):
    tokens, hidden_size, vocab_size = shape
    # These shapes must be single-chunk so the tight tolerances below are valid.
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) == 1
    ignore_index = -100
    upstream = torch.tensor(0.7, device="cuda")
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[: max(1, tokens // 8)] = ignore_index  # uneven ignored tokens

    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, reduction)
    loss_ref.backward(upstream)

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index, accum_dtype)
    loss.backward(upstream)

    atol, rtol = _tolerances(reduction)
    assert_verbose_allclose(loss_ref, loss, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=atol, rtol=rtol)
    # Gradients are always in the parameter dtype (BF16), even for FP32 accum.
    assert x.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Multi-chunk + tail correctness against the reference on shapes whose DEFAULT
# geometry (V > 16*H) is genuinely multi-chunk -- no chunk_size knob is used.
#
# Multi-chunk re-tiles the projection GEMM, so the chunked BF16 logits differ
# from the full-batch reference by cuBLAS tiling variance. Under `mean` this is
# normalized away (tight elementwise tolerance holds); under `sum` the
# unnormalized near-zero gradient elements can diverge by O(0.1) purely from GEMM
# tiling, so correctness is asserted as a small aggregate relative-L2 error (a
# structural chunking bug would blow the whole tensor up, not just near-zeros).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (513, 64, 4096),  # inc=4 -> chunk 256 -> [256, 256, 1]
        (200, 96, 8192),  # H=96 (non-aligned), multi vocab partition
        (250, 128, 8192),  # V > 4096 -> multiple vocab partitions
        (300, 64, 4096),  # 3 chunks
        (129, 16, 8192),  # inc=32 -> chunk 8 -> 17 tiny chunks + tail
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
def test_multichunk_correctness(shape, reduction, accum_dtype):
    tokens, hidden_size, vocab_size = shape
    # These shapes must be genuinely multi-chunk under the default geometry.
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) >= 2
    ignore_index = -100
    upstream = torch.tensor(0.7, device="cuda")
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[: max(1, tokens // 8)] = ignore_index

    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, reduction)
    loss_ref.backward(upstream)

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index, accum_dtype)
    loss.backward(upstream)

    if reduction == "mean":
        atol, rtol = _tolerances(reduction)
        assert_verbose_allclose(loss_ref, loss, atol=atol, rtol=rtol)
        assert_verbose_allclose(x_ref.grad, x.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(weight_ref.grad, weight.grad, atol=atol, rtol=rtol)
    else:
        assert abs(loss.item() - loss_ref.item()) / abs(loss_ref.item()) < 2e-2
        assert _relative_l2(x.grad, x_ref.grad) < 2e-2
        assert _relative_l2(weight.grad, weight_ref.grad) < 2e-2


# ---------------------------------------------------------------------------
# Chunking invariance: a forced multi-chunk run (the documented C=1 budget floor
# monkeypatched onto the shared _CHUNK_MEM_CONST) must match the single-chunk run
# the same shape gets at the default C=16 budget. Loss is partition-invariant;
# gradients differ only by GEMM/addmm reduction order (tight tolerance).
# ---------------------------------------------------------------------------
def test_chunking_matches_single_chunk(monkeypatch):
    tokens, hidden_size, vocab_size = 130, 128, 512
    ignore_index = -100
    upstream = torch.tensor(1.3, device="cuda")
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[:9] = ignore_index

    # Default budget is a single chunk for this shape...
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) == 1

    def run():
        x = x_data.clone().requires_grad_(True)
        weight = weight_data.clone().requires_grad_(True)
        loss = _apply(x, weight, target, "mean", ignore_index)[0]
        loss.backward(upstream)
        return loss.detach(), x.grad, weight.grad

    loss_full, gx_full, gw_full = run()

    # ...forcing the C=1 memory floor re-tiles the SAME shape into many chunks.
    monkeypatch.setattr(triton_flce_mod, "_CHUNK_MEM_CONST", 1)
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) >= 2
    loss_c, gx_c, gw_c = run()

    assert_verbose_allclose(loss_full, loss_c, atol=1e-4, rtol=1e-3)
    assert_verbose_allclose(gx_full, gx_c, atol=1e-3, rtol=1e-2)
    assert_verbose_allclose(gw_full, gw_c, atol=1e-3, rtol=1e-2)


# ---------------------------------------------------------------------------
# Global-mean normalization: a middle chunk that is entirely ignored still
# normalizes by the GLOBAL non-ignored count. Shape (192,64,4096) is 3 chunks of
# 64 under the default geometry; the middle chunk [64:128] is fully ignored.
# ---------------------------------------------------------------------------
def test_global_mean_with_all_ignored_middle_chunk():
    tokens, hidden_size, vocab_size = 192, 64, 4096
    ignore_index = -100
    assert _expected_tiling(tokens, hidden_size, vocab_size) == [64, 64, 64]
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[64:128] = ignore_index  # middle chunk fully ignored

    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, "mean")
    loss_ref.backward()

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, "mean", ignore_index)[0]
    loss.backward()

    assert_verbose_allclose(loss_ref, loss, atol=5e-3, rtol=5e-2)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=5e-3, rtol=5e-2)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=5e-3, rtol=5e-2)


def test_all_ignored_entire_batch():
    # Multi-chunk shape so the all-ignored path is exercised across chunks.
    tokens, hidden_size, vocab_size = 96, 64, 4096
    ignore_index = -100
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) >= 2
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.full((tokens,), ignore_index, device="cuda")

    loss = _apply(x, weight, target, "mean", ignore_index)[0]
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(x.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


# ---------------------------------------------------------------------------
# Independent gradient requirements: forward-only / input-only / weight-only /
# both, on a multi-chunk shape. Chunked forward must honor each combination.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "requires_input_grad, requires_weight_grad",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
def test_independent_gradient_requirements(requires_input_grad, requires_weight_grad, accum_dtype):
    torch.manual_seed(0)
    tokens, hidden_size, vocab_size = 200, 64, 4096
    ignore_index = -100
    upstream = torch.tensor(-0.7, device="cuda")
    tiling = _expected_tiling(tokens, hidden_size, vocab_size)
    assert len(tiling) >= 2 and tiling[-1] == 8  # multi-chunk with an 8-token tail

    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    # Ignored targets INSIDE a chunk exercise the global-mean normalizer.
    target[20:37] = ignore_index
    target[::13] = ignore_index

    x = x_data.clone().requires_grad_(requires_input_grad)
    weight = weight_data.clone().requires_grad_(requires_weight_grad)
    loss = _apply(x, weight, target, "mean", ignore_index, accum_dtype)[0]
    assert torch.isfinite(loss)

    # Same-chunk oracle on detached leaves -- the loss and each requested grad
    # match tightly regardless of which gradients were requested.
    loss_oracle, dx_oracle, dw_oracle = _same_chunk_oracle(
        x_data, weight_data, target, "mean", upstream, ignore_index, accum_dtype
    )
    assert_verbose_allclose(loss, loss_oracle, atol=1e-2, rtol=1e-3)

    if requires_input_grad or requires_weight_grad:
        loss.backward(upstream)
        assert (x.grad is not None) == requires_input_grad
        assert (weight.grad is not None) == requires_weight_grad
        if requires_input_grad:
            assert_verbose_allclose(x.grad, dx_oracle, atol=2e-3, rtol=1e-3)
        if requires_weight_grad:
            assert_verbose_allclose(weight.grad, dw_oracle, atol=2e-3, rtol=1e-3)
    else:
        assert loss.grad_fn is None  # no autograd graph when neither operand needs grad

    # Low-level chunked forward on leaves carrying the ORIGINAL requires flags:
    # only the requested gradient buffers are materialized. The new 3-tuple
    # contract returns the COMPLETED weight-dtype gradients (loss, gX, gW): dW is
    # cast to the weight dtype (BF16) once at the end of the forward, even for
    # FP32 accumulation. Run under no_grad -- the forward keys off requires_grad.
    xl = x_data.clone().requires_grad_(requires_input_grad)
    wl = weight_data.clone().requires_grad_(requires_weight_grad)
    with torch.no_grad():
        ll_loss, grad_input, grad_weight = chunked_fused_linear_cross_entropy_forward(
            xl, wl, target, reduction="mean", ignore_index=ignore_index, accum_dtype=accum_dtype
        )
    assert_verbose_allclose(ll_loss, loss_oracle, atol=1e-2, rtol=1e-3)
    if requires_input_grad:
        assert grad_input is not None
        assert grad_input.dtype == torch.bfloat16
        assert tuple(grad_input.shape) == (tokens, hidden_size)
    else:
        assert grad_input is None
    if requires_weight_grad:
        assert grad_weight is not None
        assert grad_weight.dtype == torch.bfloat16  # weight dtype, regardless of accum_dtype
        assert tuple(grad_weight.shape) == (vocab_size, hidden_size)
    else:
        assert grad_weight is None


# ---------------------------------------------------------------------------
# Repeated backward with a *different* non-unit upstream: the retained gradients
# must not be mutated, and gradients must scale linearly.
# ---------------------------------------------------------------------------
def test_repeated_backward_different_upstream():
    tokens, hidden_size, vocab_size = 200, 64, 4096
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) >= 2
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[:10] = -100
    loss = _apply(x, weight, target, "mean", -100)[0]

    loss.backward(torch.tensor(1.0, device="cuda"), retain_graph=True)
    gx1 = x.grad.clone()
    gw1 = weight.grad.clone()
    x.grad = None
    weight.grad = None
    loss.backward(torch.tensor(2.5, device="cuda"), retain_graph=True)

    assert_verbose_allclose(x.grad, 2.5 * gx1, atol=1e-3, rtol=1e-3)
    assert_verbose_allclose(weight.grad, 2.5 * gw1, atol=1e-3, rtol=1e-3)


def test_retained_backward_preserves_saved_gradient():
    tokens, hidden_size, vocab_size = 128, 64, 4096
    assert len(_expected_tiling(tokens, hidden_size, vocab_size)) >= 2
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    upstream = torch.tensor(0.7, device="cuda")
    loss = _apply(x, weight, target, "mean", -100)[0]

    loss.backward(upstream, retain_graph=True)
    first_x_grad = x.grad.clone()
    first_weight_grad = weight.grad.clone()
    x.grad = None
    weight.grad = None
    loss.backward(upstream, retain_graph=True)

    assert_verbose_allclose(first_x_grad, x.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(first_weight_grad, weight.grad, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Memory contract: the forward retains only [N, H] dX and [V, H] dW -- never a
# full [N, V] logits/dZ tensor -- and both are the COMPLETED weight-dtype
# gradients (dW cast once at the end of forward). Only one per-chunk logits/dZ
# tensor is live at a time, so peak memory stays below the full [N, V]
# materialization.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
def test_retains_only_final_gradients_no_full_logits(accum_dtype):
    tokens, hidden_size, vocab_size = 256, 128, 4096
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")

    with torch.no_grad():
        _, grad_input, grad_weight = chunked_fused_linear_cross_entropy_forward(
            x, weight, target, accum_dtype=accum_dtype
        )

    assert tuple(grad_input.shape) == (tokens, hidden_size)
    assert tuple(grad_weight.shape) == (vocab_size, hidden_size)
    # Both retained gradients are the completed weight dtype (BF16) -- the FP32
    # accumulator (when accum_dtype=fp32) is cast to BF16 at the end of forward,
    # so no parameter-sized FP32 tensor is retained for backward.
    assert grad_input.dtype == torch.bfloat16
    assert grad_weight.dtype == torch.bfloat16
    # No retained tensor has the full [N, V] footprint.
    assert grad_input.numel() < tokens * vocab_size
    assert grad_weight.numel() < tokens * vocab_size


class _AllocTracer(TorchDispatchMode):
    """Record the allocation-relevant matmul/copy ops of the FLCE forward.

    Captures the SEQUENCE of ``aten.mm``/``aten.addmm`` ops (so the Triton
    allocation style -- out-free ``aten.mm.default`` projection + dX, then an
    ``addmm`` dW accumulation -- can be compared to the Triton reference) and the
    output ``data_ptr`` of every projection GEMM (``arg1 == weight.t()``). Only
    integer ``data_ptr`` values are stored, never Tensor references, so the tracer
    cannot itself keep chunk storages alive and inflate peak memory.
    """

    _MM_FAMILY = (
        "aten.mm.default",
        "aten.mm.out",
        "aten.mm.dtype_out",
        "aten.addmm.default",
        "aten.addmm.out",
        "aten.addmm.dtype_out",
    )

    def __init__(self, hidden_size, vocab_size):
        self._hv = (hidden_size, vocab_size)
        self.mm_family = []
        self.proj_ptrs = []
        self.has_copy = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        out = func(*args, **kwargs)
        if name in self._MM_FAMILY:
            self.mm_family.append(name)
            if name == "aten.mm.default" and len(args) >= 2 and tuple(args[1].shape) == self._hv:
                result = out[0] if isinstance(out, (tuple, list)) else out
                self.proj_ptrs.append(result.data_ptr())  # int only -- no strong ref
        elif name.startswith("aten.copy"):
            self.has_copy = True
        return out


# ---------------------------------------------------------------------------
# Allocation-style regression: the chunked path must mirror the Triton
# reference's per-chunk allocations -- a fresh out-free ``aten.mm.default``
# projection and dX GEMM (NOT a recycled buffer or a direct GEMM-into-output
# ``aten.mm.out``), a ``copy_`` of the dX chunk result into grad_input, and an
# ``addmm`` dW accumulation. Traces both backends with both grads requested and
# asserts the matmul family matches. Adjacent-chunk projection storages must
# differ (the previous chunk's dZ view is still alive when the next projection is
# allocated), but not ALL storages -- the caching allocator may reuse freed ones.
# ---------------------------------------------------------------------------
def test_chunked_allocation_matches_triton_reference():
    # V > 16*H so the default geometry is genuinely multi-chunk.
    tokens, hidden_size, vocab_size = 4096, 128, 8192
    tiling = _expected_tiling(tokens, hidden_size, vocab_size)
    assert len(tiling) >= 2 and tiling[0] < tokens
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")

    # Peak-memory guard (measured WITHOUT the tracer so its bookkeeping cannot
    # skew the allocation): this 4-chunk shape allows up to two [buffer_rows, V]
    # logits/dZ chunk buffers to overlap, yet peak stays well below a full
    # [N, V] materialization.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    loss = _apply(x, weight, target, "mean", -100)[0]
    loss.backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base
    assert peak_delta < tokens * vocab_size * 2  # full bf16 [N, V]

    # Trace the cuTile chunked forward (both grads requested).
    with torch.no_grad(), _AllocTracer(hidden_size, vocab_size) as ct_trace:
        _, grad_input, grad_weight = chunked_fused_linear_cross_entropy_forward(x, weight, target, reduction="mean")
    # Trace the actual Triton low-level forward with the same both-grad request.
    with torch.no_grad(), _AllocTracer(hidden_size, vocab_size) as tr_trace:
        triton_flce_mod.fused_linear_cross_entropy_forward(x, weight, target, reduction="mean")

    n_chunks = len(tiling)
    # Triton allocation style: out-free mm.default for projection + dX, no
    # GEMM-into-output mm.out, and a copy_ of the dX chunk into grad_input.
    assert "aten.mm.out" not in ct_trace.mm_family
    assert ct_trace.mm_family.count("aten.mm.default") == 2 * n_chunks  # projection + dX per chunk
    assert ct_trace.mm_family.count("aten.addmm.out") == n_chunks  # dW addmm every chunk
    assert ct_trace.has_copy
    # The matmul family matches the Triton reference exactly (same op sequence).
    assert ct_trace.mm_family == tr_trace.mm_family
    assert tr_trace.has_copy

    # One projection storage recorded per chunk; adjacent chunks differ.
    assert len(ct_trace.proj_ptrs) == n_chunks
    assert ct_trace.proj_ptrs[0] != ct_trace.proj_ptrs[1]
    # No retained gradient carries the full [N, V] footprint.
    assert grad_input.numel() < tokens * vocab_size
    assert grad_weight.numel() < tokens * vocab_size


# ---------------------------------------------------------------------------
# Shared token-chunk geometry: cuTile reuses the EXACT Triton default policy.
# The imported helper is the same object the Triton backend uses, and it honors
# inc=cdiv(V, 16*H), chunk=next_pow2(cdiv(N, inc)) clamped to N (never a public
# chunk_size knob, never the removed CuTe <=1024 heuristic).
# ---------------------------------------------------------------------------
def test_cutile_reuses_triton_chunk_helper():
    # cuTile imports the Triton helper -- no second constant, no divergent policy.
    assert flce_mod._get_chunk_size is triton_flce_mod._get_chunk_size
    assert not hasattr(flce_mod, "_resolve_chunk_size")
    assert not hasattr(flce_mod, "_largest_power_of_2_le")


@pytest.mark.parametrize(
    "tokens, hidden_size, vocab_size, expected_chunk, expected_num_chunks",
    [
        (8192, 4096, 128256, 4096, 2),  # llama_3 reference: 4096 rows, 2 chunks (NOT 1024)
        (200, 512, 256, 200, 1),  # V <= 16*H -> single chunk == N
        (128, 128, 256, 128, 1),  # V <= 16*H -> single chunk == N
        (513, 64, 4096, 256, 3),  # odd N tail: [256, 256, 1]
        (129, 16, 8192, 8, 17),  # next_pow2 clamp -> tiny 8-row chunks + tail
        (1, 16, 8192, 1, 1),  # one token
    ],
)
def test_chunk_geometry_table(tokens, hidden_size, vocab_size, expected_chunk, expected_num_chunks):
    chunk = _get_chunk_size(tokens, hidden_size, vocab_size)
    assert chunk == expected_chunk
    tiling = _expected_tiling(tokens, hidden_size, vocab_size)
    assert len(tiling) == expected_num_chunks
    # Only the final chunk may be a short tail; all others are full width.
    assert tiling[:-1] == [expected_chunk] * (expected_num_chunks - 1)
    assert tiling[-1] == tokens - expected_chunk * (expected_num_chunks - 1)


class _ProjectionTracer(TorchDispatchMode):
    """Record the row count (M) of every projection GEMM (arg1 == weight.t())."""

    def __init__(self, hidden_size, vocab_size):
        self._hv = (hidden_size, vocab_size)
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if str(func).startswith("aten.mm") and len(args) >= 2 and tuple(args[1].shape) == self._hv:
            self.rows.append(int(args[0].shape[0]))
        return out


@pytest.mark.parametrize(
    "shape",
    [
        (250, 128, 8192),  # [64, 64, 64, 58]
        (513, 64, 4096),  # [256, 256, 1]
        (200, 512, 256),  # single chunk [200]
    ],
)
def test_projection_tiling_matches_triton(shape):
    """The cuTile projection GEMM tiling equals what the Triton backend uses.

    Both backends chunk tokens by the shared :func:`_get_chunk_size`. Trace the
    per-chunk projection ``x_chunk @ weight.t()`` M-dims from each backend and
    assert they are identical to each other and to the expected tiling.
    """
    tokens, hidden_size, vocab_size = shape
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    expected = _expected_tiling(tokens, hidden_size, vocab_size)

    with torch.no_grad(), _ProjectionTracer(hidden_size, vocab_size) as ct_trace:
        chunked_fused_linear_cross_entropy_forward(x, weight, target, reduction="mean")

    with torch.no_grad(), _ProjectionTracer(hidden_size, vocab_size) as tr_trace:
        triton_flce_mod.fused_linear_cross_entropy_forward(x, weight, target, reduction="mean")

    assert ct_trace.rows == expected
    assert tr_trace.rows == expected
    assert ct_trace.rows == tr_trace.rows


# ---------------------------------------------------------------------------
# Argument arities: the Function accepts the direct 3-arg legacy apply, the
# 15-arg functional contract, and the 17-arg module contract (ce_impl/ce_mode
# placeholders). There is no chunk_size parameter and no supports_chunk_size
# marker; ce_impl/ce_mode do not enable real inner dispatch.
# ---------------------------------------------------------------------------
def test_no_chunk_size_public_api():
    import inspect

    assert not hasattr(LigerFusedLinearCrossEntropyFunction, "supports_chunk_size")
    assert LigerFusedLinearCrossEntropyFunction.supports_inner_impl_dispatch is False
    params = list(inspect.signature(LigerFusedLinearCrossEntropyFunction.forward).parameters)
    assert "chunk_size" not in params
    # ctx + 17 declared args (15 base + ce_impl + ce_mode).
    assert len(params) == 18


def test_apply_arg_arities():
    x_data = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(256, (64,), device="cuda")

    # 3-arg legacy apply (minimal positional contract).
    loss3 = LigerFusedLinearCrossEntropyFunction.apply(
        x_data.clone().requires_grad_(True), weight_data.clone().requires_grad_(True), target
    )[0]
    assert torch.isfinite(loss3)

    # 15-arg functional contract (what liger_fused_linear_cross_entropy passes
    # when supports_inner_impl_dispatch is False).
    loss15 = LigerFusedLinearCrossEntropyFunction.apply(
        x_data.clone().requires_grad_(True),
        weight_data.clone().requires_grad_(True),
        target,
        None,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        None,
        False,
        False,
        False,
    )[0]
    assert torch.isfinite(loss15)

    # 17-arg module contract (what the nn.Module passes: + ce_impl=None, ce_mode=None).
    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss17 = _apply(x, weight, target, "mean", -100)[0]
    loss17.backward()
    assert torch.isfinite(loss17)
    assert x.grad is not None and weight.grad is not None


# ---------------------------------------------------------------------------
# Feature / dispatch rejection.
# ---------------------------------------------------------------------------
def test_rejects_unsupported_features():
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device="cuda")
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="bias"):
        LigerFusedLinearCrossEntropyFunction.apply(x, weight, target, bias)


def test_accum_dtype_supported_others_rejected():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (64,), device="cuda")

    # None (-> weight dtype/BF16), explicit BF16 (matching weight), and FP32 are
    # all accepted by the chunked/default path.
    for accum in (None, torch.bfloat16, torch.float32):
        loss = LigerFusedLinearCrossEntropyFunction.apply(
            x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, accum
        )[0]
        assert torch.isfinite(loss)

    # Any other accum dtype (e.g. fp16, fp64) is rejected clearly, not silently
    # downgraded.
    for bad in (torch.float16, torch.float64):
        with pytest.raises(NotImplementedError, match="accum"):
            LigerFusedLinearCrossEntropyFunction.apply(
                x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, bad
            )


def test_legacy_helper_rejects_low_precision_accum():
    # The legacy unchunked helper accumulates dW in FP32 only. None/FP32 are
    # accepted; an explicit low-precision accum it cannot honor is rejected
    # (never silently accepted then ignored).
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(256, (64,), device="cuda")
    with torch.no_grad():
        for accum in (None, torch.float32):
            loss, *_ = fused_linear_cross_entropy_forward(x, weight, target, accum_dtype=accum)
            assert torch.isfinite(loss.sum())
        with pytest.raises(NotImplementedError, match="accum"):
            fused_linear_cross_entropy_forward(x, weight, target, accum_dtype=torch.bfloat16)


def _apply_with_dispatch(x, weight, target, ce_impl, ce_mode):
    return LigerFusedLinearCrossEntropyFunction.apply(
        x,
        weight,
        target,
        None,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        None,
        False,
        False,
        False,
        ce_impl,
        ce_mode,
    )


@pytest.mark.parametrize("ce_impl", [None, "cutile", "nvidia-cutile"])
def test_ce_impl_self_identity_accepted(ce_impl):
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (64,), device="cuda")
    loss = _apply_with_dispatch(x, weight, target, ce_impl=ce_impl, ce_mode=None)[0]
    assert torch.isfinite(loss)


def test_ce_impl_and_mode_incompatible_rejected():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (64,), device="cuda")

    with pytest.raises(NotImplementedError):
        _apply_with_dispatch(x, weight, target, ce_impl="triton", ce_mode=None)
    with pytest.raises(NotImplementedError):
        _apply_with_dispatch(x, weight, target, ce_impl=None, ce_mode="chunked")


# ---------------------------------------------------------------------------
# Input validation: empty dims and CPU tensors are caught before any CUDA
# capability query.
# ---------------------------------------------------------------------------
def test_rejects_empty_tensors():
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    empty_x = torch.randn(0, 128, device="cuda", dtype=torch.bfloat16)
    empty_target = torch.randint(256, (0,), device="cuda")
    with pytest.raises(ValueError, match="empty"):
        with torch.no_grad():
            fused_linear_cross_entropy_forward(empty_x, weight, empty_target)


def test_rejects_cpu_tensors():
    x = torch.randn(64, 128, dtype=torch.bfloat16)
    weight = torch.randn(256, 128, dtype=torch.bfloat16)
    target = torch.randint(256, (64,))
    with pytest.raises(RuntimeError, match="CUDA"):
        with torch.no_grad():
            fused_linear_cross_entropy_forward(x, weight, target)


def test_large_token_grid():
    tokens = 65536
    x = torch.randn(tokens, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(128, (tokens,), device="cuda")

    loss = _apply(x, weight, target)[0]

    assert torch.isfinite(loss)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_noncurrent_cuda_device():
    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    x = torch.randn(128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device=device)

    loss = _apply(x, weight, target)[0]
    loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None
    assert weight.grad is not None


# ---------------------------------------------------------------------------
# dW cast order (main/chunked path): dW is cast to the weight dtype ONCE at the
# END of the forward, BEFORE the upstream gradient -- exactly like the Triton
# mean/sum path. A sub-BF16 dW element therefore rounds to zero at that cast and
# a later upstream multiply cannot recover it. This is the intended aligned
# contract (NOT the old "small gradient survives late cast" behavior); the legacy
# raw helper, which scales in FP32 before its single cast, still preserves it and
# is covered separately below.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_small_dw_gradient_rounds_to_zero_matches_triton(accum_dtype):
    # dW element magnitude ~ (1/V) * 1e-37 is far below the BF16 subnormal floor.
    x = torch.full((1, 64), 1e-37, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([0], device="cuda")
    upstream = torch.tensor(65536.0, device="cuda")

    loss = _apply(x, weight, target, "mean", -100, accum_dtype)[0]
    loss.backward(upstream)

    # The forward casts dW to BF16 before the upstream scale, so the sub-BF16
    # element flushes to zero and go=65536 cannot resurrect it -- matching Triton.
    assert weight.grad[1, 0].item() == 0.0

    x_tr = torch.full((1, 64), 1e-37, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight_tr = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    loss_tr = _triton_apply(x_tr, weight_tr, target, "mean", -100, accum_dtype)[0]
    loss_tr.backward(upstream)
    assert_verbose_allclose(weight.grad, weight_tr.grad, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Strict SAME-CHUNK oracle: reproduce the kernel math per chunk with the exact
# same GEMM tiling, DERIVED from the shared default geometry (not a chunk_size
# argument), then assert tight per-element tolerances. A structural chunking /
# scaling bug would move whole elements far beyond these tolerances. An
# independent full-batch PyTorch anchor is retained too.
# ---------------------------------------------------------------------------
def _same_chunk_oracle(x_data, weight_data, target, reduction, upstream, ignore_index=-100, accum_dtype=None):
    n_tokens, hidden = x_data.shape
    vocab = weight_data.shape[0]
    device = x_data.device
    # Derive the chunk size from the shared policy -- identical to the kernel.
    chunk_size = _get_chunk_size(n_tokens, hidden, vocab)
    valid = target != ignore_index
    global_scale = 1.0 / valid.sum().clamp_min(1).float() if reduction == "mean" else torch.tensor(1.0, device=device)
    # dW accumulation buffer dtype follows accum_dtype (None/BF16 -> weight dtype,
    # float32 -> FP32), mirroring the production path.
    accum_buf_dtype = torch.float32 if accum_dtype == torch.float32 else weight_data.dtype
    loss = torch.zeros((), dtype=torch.float32, device=device)
    dx = torch.zeros((n_tokens, hidden), dtype=torch.bfloat16, device=device)
    dw = torch.zeros((vocab, hidden), dtype=accum_buf_dtype, device=device)
    for start in range(0, n_tokens, chunk_size):
        stop = min(start + chunk_size, n_tokens)
        xc = x_data[start:stop]
        tc = target[start:stop]
        z = torch.mm(xc, weight_data.t())  # BF16 projection, same tiling as the kernel
        zf = z.float()
        ce = F.cross_entropy(zf, tc, ignore_index=ignore_index, reduction="none")
        loss += (ce * global_scale).sum()
        dz = torch.softmax(zf, dim=-1)
        rows = torch.arange(stop - start, device=device)
        valid_c = tc != ignore_index
        # Target subtract-1 in FP32, THEN fold the global normalizer in FP32,
        # THEN cast to BF16 -- normalization happens BEFORE the dX/dW GEMMs.
        dz[rows[valid_c], tc[valid_c]] -= 1.0
        dz[~valid_c] = 0.0
        dz = (dz * global_scale).to(torch.bfloat16)  # normalized dZ, BF16
        dx[start:stop] = torch.mm(dz, weight_data)  # dX from the normalized BF16 dZ
        dz_t = dz.t()
        if accum_buf_dtype == torch.float32:
            dw.add_(_ref_dw_fp32(dz_t, xc))  # FP32-accumulated dW
        else:
            torch.addmm(dw, dz_t, xc, out=dw)  # BF16 accumulate: addmm into zero-init buffer every chunk
    grad_weight = dw.to(torch.bfloat16)  # single cast to weight dtype at end of forward
    dx_final = (dx * upstream).to(torch.bfloat16)  # backward applies only upstream
    dw_final = (grad_weight * upstream).to(torch.bfloat16)
    return loss, dx_final, dw_final


@pytest.mark.parametrize("seed", [7, 314])
@pytest.mark.parametrize("hidden", [96, 256])  # non-aligned + 256-aligned H tails
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
# force_fallback drives the pre-2.8 dW path (plain mm/addmm on FP32-upcast BF16
# operands). It only affects the FP32-accumulator path; the None/BF16 matching
# path always uses a plain BF16 mm/addmm. The oracle tracks the same flag (via
# ``_ref_dw_fp32`` for FP32 accum), so both paths stay bit-identical and the SAME
# tight tolerances apply -- forced multi-chunk numerical coverage of the
# older-torch accumulation with no tolerance widening.
@pytest.mark.parametrize("force_fallback", [False, True])
def test_same_chunk_oracle_regression(seed, hidden, reduction, accum_dtype, force_fallback, monkeypatch):
    if force_fallback:
        monkeypatch.setattr(flce_mod, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
    torch.manual_seed(seed)
    n_tokens, vocab = 65, 4113  # odd V, tail chunk under the default geometry
    ignore_index = -100
    upstream = torch.tensor(-0.7, device="cuda")
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[17:34] = ignore_index
    target[::7] = ignore_index
    # These shapes are multi-chunk with a tail under the shared geometry.
    assert len(_expected_tiling(n_tokens, hidden, vocab)) >= 2

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, reduction, ignore_index, accum_dtype)[0]
    loss.backward(upstream)

    loss_oracle, dx_oracle, dw_oracle = _same_chunk_oracle(
        x_data, weight_data, target, reduction, upstream, ignore_index, accum_dtype
    )

    # Strict per-element match against the same-tiling oracle (dX is bit-exact;
    # dW differs only by FP32 accumulation order, far below these tolerances).
    assert_verbose_allclose(loss, loss_oracle, atol=1e-2, rtol=1e-3)
    assert_verbose_allclose(x.grad, dx_oracle, atol=2e-3, rtol=1e-3)
    assert_verbose_allclose(weight.grad, dw_oracle, atol=2e-3, rtol=1e-3)

    # Independent full-batch PyTorch anchor (aggregate relative-L2).
    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, reduction)
    loss_ref.backward(upstream)
    assert _relative_l2(x.grad, x_ref.grad) < 2e-2
    assert _relative_l2(weight.grad, weight_ref.grad) < 2e-2


# ---------------------------------------------------------------------------
# Persistent retained-graph alias safety: a first backward (and mutating +
# clearing the produced .grad) must not corrupt the saved FP32 dW accumulator or
# raw BF16 dX. A second backward with a different upstream must match a FRESH
# forward+backward at that upstream EXACTLY -- proving the scale/cast is
# recomputed from the retained FP32 dW (not a double-rounded rescale of the first
# BF16 result). Uses a multi-chunk shape so cross-chunk accumulation is retained.
# ---------------------------------------------------------------------------
def test_retained_backward_alias_safety():
    torch.manual_seed(11)
    n_tokens, hidden, vocab = 128, 64, 4096
    ignore_index = -100
    assert len(_expected_tiling(n_tokens, hidden, vocab)) >= 2
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[:9] = ignore_index

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, "mean", ignore_index)[0]

    loss.backward(torch.tensor(1.0, device="cuda"), retain_graph=True)
    # Mutate then clear the produced gradients (must not feed back into saved state).
    x.grad.fill_(123.0)
    weight.grad.fill_(-456.0)
    x.grad = None
    weight.grad = None
    loss.backward(torch.tensor(-1.3, device="cuda"), retain_graph=True)
    gx_second = x.grad.clone()
    gw_second = weight.grad.clone()

    # Fresh forward+backward at the same -1.3 upstream.
    x_fresh = x_data.clone().requires_grad_(True)
    weight_fresh = weight_data.clone().requires_grad_(True)
    loss_fresh = _apply(x_fresh, weight_fresh, target, "mean", ignore_index)[0]
    loss_fresh.backward(torch.tensor(-1.3, device="cuda"))

    assert_verbose_allclose(gx_second, x_fresh.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(gw_second, weight_fresh.grad, atol=0.0, rtol=0.0)

    # Directly snapshot the retained low-level tensors (the completed BF16 dX/dW)
    # and confirm two backward passes with different upstreams leave them
    # untouched -- the 3-arg backward multiplies out-of-place.
    xg = x_data.clone().requires_grad_(True)
    wg = weight_data.clone().requires_grad_(True)
    with torch.no_grad():
        _, grad_input, grad_weight = chunked_fused_linear_cross_entropy_forward(xg, wg, target)
    grad_input_snap = grad_input.clone()
    grad_weight_snap = grad_weight.clone()
    chunked_fused_linear_cross_entropy_backward(torch.tensor(1.0, device="cuda"), grad_input, grad_weight)
    chunked_fused_linear_cross_entropy_backward(torch.tensor(-1.3, device="cuda"), grad_input, grad_weight)
    assert_verbose_allclose(grad_input, grad_input_snap, atol=0.0, rtol=0.0)
    assert_verbose_allclose(grad_weight, grad_weight_snap, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Legacy low-level contract: the unchunked helper returns the full [N, V] dZ as
# its third element, and the paired 7-argument backward consumes it to run the
# dW matmul. This guards the original exported interface against silent contract
# drift (the chunked path uses the separate chunked_* helpers, no C argument).
# ---------------------------------------------------------------------------
def test_legacy_lowlevel_forward_backward_contract():
    torch.manual_seed(0)
    n_tokens, hidden, vocab = 128, 128, 257  # odd V exercises a non-256-aligned dW tail
    ignore_index = -100
    upstream = torch.tensor(-0.7, device="cuda")
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[:8] = ignore_index

    with torch.no_grad():
        loss, grad_input, grad_logits, gradient_scale = fused_linear_cross_entropy_forward(
            x, weight, target, reduction="mean"
        )

    # Third element is the full [N, V] dZ (legacy contract), not [V, H] dW.
    assert tuple(grad_logits.shape) == (n_tokens, vocab)
    assert grad_logits.dtype == torch.bfloat16
    assert gradient_scale.ndim == 0

    grad_x, grad_weight = fused_linear_cross_entropy_backward(
        upstream,
        grad_input,
        grad_logits,
        x.detach(),
        gradient_scale,
        weight.shape,
        weight.dtype,
    )
    assert tuple(grad_x.shape) == (n_tokens, hidden)
    assert grad_x.dtype == torch.bfloat16
    assert tuple(grad_weight.shape) == (vocab, hidden)
    assert grad_weight.dtype == weight.dtype

    # Cross-check the legacy gradients against the BF16-linear / FP32-CE anchor.
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, "mean")
    loss_ref.backward(upstream)
    assert abs(loss.sum().item() - loss_ref.item()) < 5e-3
    assert _relative_l2(grad_x, x_ref.grad) < 2e-2
    assert _relative_l2(grad_weight, weight_ref.grad) < 2e-2


# ---------------------------------------------------------------------------
# GEMM ownership trace: the dW gradient runs through PyTorch/cuBLAS, not a custom
# cuTile MMA kernel. For the FP32 accumulator the chunked path emits one FP32-out
# dW ``aten::addmm`` PER CHUNK into a zero-initialized buffer (no first-chunk
# ``mm`` init) -- via the ``out_dtype`` overload on torch>=2.8, or plain
# ``addmm.out`` writing into an FP32 buffer on the pre-2.8 fallback. Projection
# and dX use the out-free ``aten::mm.default`` overload (the Triton allocation
# style), never a GEMM-into-output ``aten::mm.out``. The removed cuTile matmul
# kernels must no longer exist on the module.
# ---------------------------------------------------------------------------
class _AtenOpRecorder(TorchDispatchMode):
    """Record every aten op name the path issues, plus dW mm/addmm out dtypes."""

    def __init__(self):
        self.ops = []
        self.dw_out_dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        out = func(*args, **kwargs)
        self.ops.append(name)
        # Record the dW GEMM output dtype regardless of which overload the path
        # took: the ``out_dtype`` overload (torch>=2.8) issues ``.dtype_out`` and
        # is always FP32; the pre-2.8 fallback issues plain ``.out`` writing into
        # an FP32 buffer. Filtering the plain overload on FP32 keeps the BF16-out
        # projection / dX GEMMs out of this list, so the BF16 accumulator path
        # still records ``[]``.
        if name in ("aten.mm.dtype_out", "aten.addmm.dtype_out"):
            result = out[0] if isinstance(out, (tuple, list)) else out
            self.dw_out_dtypes.append(result.dtype)
        elif name in ("aten.mm.out", "aten.addmm.out"):
            result = out[0] if isinstance(out, (tuple, list)) else out
            if result.dtype == torch.float32:
                self.dw_out_dtypes.append(result.dtype)
        return out


def test_no_custom_matmul_kernels_remain():
    # The dW GEMM is cuBLAS now; the old cuTile MMA kernels/launcher are gone.
    for removed in ("_matmul_tn_kernel", "_matmul_tn_n4_kernel", "_launch_matmul_tn"):
        assert not hasattr(flce_mod, removed), f"{removed} should have been removed"
    # The CE / scale-cast cuTile kernels are retained.
    assert hasattr(flce_mod, "_scale_cast_kernel")
    assert hasattr(flce_mod, "_fused_cross_entropy_dz_kernel")


@pytest.mark.parametrize("force_fallback", [False, True])
@pytest.mark.parametrize("accum_dtype", _ACCUM_MODES)
def test_chunked_dw_accumulator_dtype_and_gemms(accum_dtype, force_fallback, monkeypatch):
    torch.manual_seed(3)
    if force_fallback:
        # Drive the pre-2.8 path: plain mm.out/addmm.out instead of the
        # ``out_dtype`` overload. Only the FP32 accumulator is affected; the
        # None/BF16 modes never used the overload, so those cases are unchanged.
        monkeypatch.setattr(flce_mod, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
    # V > 16*H so the default geometry is multi-chunk: [16, 16, 16, 16, 6] -> 5 chunks.
    n_tokens, hidden, vocab = 70, 96, 8192
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab, (n_tokens,), device="cuda")

    n_chunks = len(_expected_tiling(n_tokens, hidden, vocab))
    assert n_chunks >= 2

    # Spy the actual dW accumulator: buffer dtype, dZ operand dtype, accumulate flag.
    accum_calls = []
    real_accum = flce_mod._accumulate_dw

    def spy(buf, dz_t, x_chunk, accumulate):
        # dz_t is the already-transposed [V, rows] operand; its dtype is preserved.
        accum_calls.append((buf.dtype, dz_t.dtype, accumulate))
        return real_accum(buf, dz_t, x_chunk, accumulate)

    monkeypatch.setattr(flce_mod, "_accumulate_dw", spy)

    with _AtenOpRecorder() as rec, torch.no_grad():
        _, _, grad_weight = chunked_fused_linear_cross_entropy_forward(x, weight, target, accum_dtype=accum_dtype)

    expected_buf_dtype = torch.float32 if accum_dtype == torch.float32 else torch.bfloat16
    # One accumulate call per chunk, all addmm (accumulate=True) into a
    # zero-initialized buffer -- no first-chunk mm-overwrite optimization.
    assert [c[2] for c in accum_calls] == [True] * n_chunks
    # The accumulator dtype is uniform across chunks and follows accum_dtype.
    assert all(c[0] == expected_buf_dtype for c in accum_calls)
    # dZ is always BF16 -- there is never a parameter-sized FP32 dZ tensor.
    assert all(c[1] == torch.bfloat16 for c in accum_calls)
    # The completed dW is always the weight dtype. The FP32 accumulator upcasts
    # once and casts back to BF16 at the end (a single end-cast); the BF16
    # accumulator holds a BF16 buffer that is re-cast per chunk with no distinct
    # FP32 end-cast. Either way the returned dW is BF16.
    assert grad_weight.dtype == torch.bfloat16

    # Projection + dX use the out-free mm.default overload; no GEMM-into-output.
    assert "aten.mm.out" not in rec.ops

    if expected_buf_dtype == torch.float32:
        if flce_mod._ADDMM_SUPPORTS_OUT_DTYPE:
            # FP32 accumulator drives the out_dtype overload: one FP32-out addmm
            # per chunk (including the first) into the zero-initialized buffer.
            assert "aten.mm.dtype_out" not in rec.ops
            assert rec.ops.count("aten.addmm.dtype_out") == n_chunks
            assert "aten.addmm.out" not in rec.ops
        else:
            # Pre-2.8 fallback: no out_dtype overload. dW is a plain addmm.out per
            # chunk (including the first), all writing into the FP32 buffer.
            # addmm.out is issued ONLY by the dW accumulation.
            assert "aten.mm.dtype_out" not in rec.ops
            assert "aten.addmm.dtype_out" not in rec.ops
            assert rec.ops.count("aten.addmm.out") == n_chunks
        # Both overloads emit exactly n_chunks FP32-out dW addmm GEMMs.
        assert rec.dw_out_dtypes == [torch.float32] * n_chunks
    else:
        # BF16 accumulator: plain BF16 addmm every chunk, no out_dtype overload.
        assert "aten.mm.dtype_out" not in rec.ops
        assert "aten.addmm.dtype_out" not in rec.ops
        # addmm.out is issued ONLY by the dW accumulation, one per chunk.
        assert rec.ops.count("aten.addmm.out") == n_chunks
        assert rec.dw_out_dtypes == []
    assert "aten.addmm.default" not in rec.ops


def test_legacy_backward_dw_uses_fp32_mm_nonunit_upstream():
    torch.manual_seed(5)
    n_tokens, hidden, vocab = 96, 128, 300
    upstream = torch.tensor(-1.7, device="cuda")  # non-unit upstream
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab, (n_tokens,), device="cuda")

    with torch.no_grad():
        _, grad_input, grad_logits, gradient_scale = fused_linear_cross_entropy_forward(
            x, weight, target, reduction="mean"
        )

    with _AtenOpRecorder() as rec:
        _, grad_weight = fused_linear_cross_entropy_backward(
            upstream, grad_input, grad_logits, x.detach(), gradient_scale, weight.shape, weight.dtype
        )

    # Legacy dW is a single FP32-out mm (unchunked); no addmm accumulation.
    if flce_mod._ADDMM_SUPPORTS_OUT_DTYPE:
        assert rec.ops.count("aten.mm.dtype_out") == 1
        assert "aten.addmm.dtype_out" not in rec.ops
        assert rec.dw_out_dtypes == [torch.float32]
    else:
        # Pre-2.8 fallback: the single dW GEMM is a plain mm.out into the FP32
        # buffer -- no ``.dtype_out`` overload and no addmm accumulation. The
        # recorder still captures its FP32 output.
        assert "aten.mm.dtype_out" not in rec.ops
        assert "aten.addmm.dtype_out" not in rec.ops
        assert rec.dw_out_dtypes == [torch.float32]
    assert grad_weight.dtype == weight.dtype


# ---------------------------------------------------------------------------
# Pre-2.8 fallback: when the ``out_dtype`` mm/addmm overload is unavailable the
# FP32-accumulator path must upcast the bounded operands to FP32 and use the
# plain mm/addmm (no ``.dtype_out`` op). The None/BF16 path is unaffected (it
# never used the overload), so this only exercises accum_dtype=torch.float32.
# ---------------------------------------------------------------------------
def test_pre_2_8_fallback_avoids_out_dtype_overload(monkeypatch):
    monkeypatch.setattr(flce_mod, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
    torch.manual_seed(9)
    # V > 16*H -> multi-chunk under the default geometry: [16, 16, 16, 2] -> 4 chunks.
    n_tokens, hidden, vocab = 50, 64, 4096
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab, (n_tokens,), device="cuda")

    n_chunks = len(_expected_tiling(n_tokens, hidden, vocab))
    assert n_chunks >= 2
    with _AtenOpRecorder() as rec, torch.no_grad():
        chunked_fused_linear_cross_entropy_forward(x, weight, target, accum_dtype=torch.float32)

    # No out_dtype overload is used on the fallback path.
    assert "aten.mm.dtype_out" not in rec.ops
    assert "aten.addmm.dtype_out" not in rec.ops
    # Projection + dX use out-free mm.default; no GEMM-into-output mm.out.
    assert "aten.mm.out" not in rec.ops
    # dW accumulates with a plain addmm.out every chunk (zero-init buffer, no
    # first-chunk mm). addmm.out is issued ONLY by the dW accumulation.
    assert rec.ops.count("aten.addmm.out") == n_chunks
    # Every dW GEMM writes into the FP32 buffer, so all n_chunks outputs are FP32.
    assert rec.dw_out_dtypes == [torch.float32] * n_chunks


# ---------------------------------------------------------------------------
# Legacy small-gradient preservation: the LEGACY raw helper retains the RAW dZ
# and applies the combined upstream/mean scale in FP32 BEFORE its single
# weight-dtype cast, so a sub-BF16 dW is recovered by a large upstream. (The
# aligned main path instead casts dW before the upstream and rounds it to zero,
# matching Triton -- see test_small_dw_gradient_rounds_to_zero_matches_triton.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("force_fallback", [False, True])
def test_legacy_small_dw_gradient_survives_fp32_scale(force_fallback, monkeypatch):
    if force_fallback:
        monkeypatch.setattr(flce_mod, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
    x = torch.full((1, 64), 1e-37, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([0], device="cuda")
    upstream = torch.tensor(65536.0, device="cuda")

    with torch.no_grad():
        _, grad_input, grad_logits, gradient_scale = fused_linear_cross_entropy_forward(
            x, weight, target, reduction="mean"
        )
    _, grad_weight = fused_linear_cross_entropy_backward(
        upstream, grad_input, grad_logits, x.detach(), gradient_scale, weight.shape, weight.dtype
    )

    # Independent FP32 reference: raw dZ (softmax with the target decremented),
    # FP32 dW = dZ.T @ X, then upstream*mean-normalizer (1/1) applied in FP32 and
    # cast to BF16 exactly once -- so the sub-BF16 element survives.
    dz = torch.full((1, 4096), 1.0 / 4096, dtype=torch.bfloat16, device="cuda")
    dz[0, 0] -= 1.0
    dw_fp32 = _ref_dw_fp32(dz.t(), x.detach())
    ref = (dw_fp32 * upstream.to(torch.float32)).to(torch.bfloat16)

    assert grad_weight[1, 0].item() != 0.0
    assert_verbose_allclose(grad_weight, ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Actual Triton FLCE Function parity: at the shared default geometry (both
# backends chunk by _get_chunk_size) with a non-unit upstream, the aligned
# cuTile path matches the real Triton autograd Function for BOTH accumulation
# modes (None -> BF16 accumulator, torch.float32 -> FP32) and both reductions.
# This is NOT a bitwise CE claim -- the exp2 softmax differs slightly across
# backends -- so a small relative-L2 tolerance is used; the dtype/cast ordering
# is asserted exactly (both produce BF16 gradients).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("accum_dtype", [None, torch.float32])
def test_triton_function_parity(reduction, accum_dtype):
    torch.manual_seed(21)
    n_tokens, hidden, vocab = 200, 64, 4096
    ignore_index = -100
    upstream = torch.tensor(1.3, device="cuda")
    assert len(_expected_tiling(n_tokens, hidden, vocab)) >= 2
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[:9] = ignore_index
    target[::11] = ignore_index

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, reduction, ignore_index, accum_dtype)[0]
    loss.backward(upstream)

    x_tr = x_data.clone().requires_grad_(True)
    weight_tr = weight_data.clone().requires_grad_(True)
    loss_tr = _triton_apply(x_tr, weight_tr, target, reduction, ignore_index, accum_dtype)[0]
    loss_tr.backward(upstream)

    # Loss and gradients match the real Triton Function within CE softmax noise.
    assert abs(loss.item() - loss_tr.item()) / max(abs(loss_tr.item()), 1e-6) < 5e-3
    assert _relative_l2(x.grad, x_tr.grad) < 1e-2
    assert _relative_l2(weight.grad, weight_tr.grad) < 1e-2
    # Cast ordering is identical: both backends emit BF16 gradients.
    assert x.grad.dtype == x_tr.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == weight_tr.grad.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Before-GEMM normalization regression: the dZ actually fed to the dX/dW GEMMs
# must be the global-mean-normalized (softmax - one-hot) in FP32 rounded to BF16
# -- NOT the raw unnormalized dZ that the old late-scaling contract produced.
# Uses non-power-of-two valid counts (::7 and ::13 strides) and an entirely
# ignored middle chunk; the dZ fed to the GEMMs is captured via _accumulate_dw
# (the same buffer dX projects from) and compared to the normalized reference.
# ---------------------------------------------------------------------------
def test_before_gemm_normalization_feeds_normalized_dz(monkeypatch):
    torch.manual_seed(2)
    n_tokens, hidden, vocab = 192, 64, 4096
    ignore_index = -100
    assert _expected_tiling(n_tokens, hidden, vocab) == [64, 64, 64]
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[64:128] = ignore_index  # middle chunk fully ignored
    target[::7] = ignore_index
    target[::13] = ignore_index

    captured = []
    real_accum = flce_mod._accumulate_dw

    def spy(buf, dz_t, x_chunk, accumulate):
        # dz_t is the transposed [V, rows] operand; transpose back to [rows, V]
        # (the logits buffer dX also projects from) for the normalized-dZ compare.
        captured.append(dz_t.t().detach().clone())
        return real_accum(buf, dz_t, x_chunk, accumulate)

    monkeypatch.setattr(flce_mod, "_accumulate_dw", spy)

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    with torch.no_grad():
        chunked_fused_linear_cross_entropy_forward(x, weight, target, reduction="mean")

    fed_dz = torch.cat(captured, dim=0)
    assert tuple(fed_dz.shape) == (n_tokens, vocab)

    # Build BOTH references: the NORMALIZED dZ (new/aligned contract) and the RAW
    # dZ (old late-scaling contract).
    chunk_size = _get_chunk_size(n_tokens, hidden, vocab)
    global_scale = 1.0 / (target != ignore_index).sum().clamp_min(1).float()
    norm_ref = torch.zeros_like(fed_dz)
    raw_ref = torch.zeros_like(fed_dz)
    for start in range(0, n_tokens, chunk_size):
        stop = min(start + chunk_size, n_tokens)
        tc = target[start:stop]
        zf = torch.mm(x_data[start:stop], weight_data.t()).float()
        dz = torch.softmax(zf, dim=-1)
        rows = torch.arange(stop - start, device=zf.device)
        valid_c = tc != ignore_index
        dz[rows[valid_c], tc[valid_c]] -= 1.0
        dz[~valid_c] = 0.0
        raw_ref[start:stop] = dz.to(torch.bfloat16)
        norm_ref[start:stop] = (dz * global_scale).to(torch.bfloat16)

    # The fed dZ matches the NORMALIZED reference (softmax noise tolerance)...
    assert _relative_l2(fed_dz, norm_ref) < 5e-2
    # ...and is decisively NOT the raw unnormalized dZ (normalizer ~1/165 here).
    assert _relative_l2(fed_dz, raw_ref) > 0.5
    # The entirely-ignored middle chunk contributes exactly zero dZ.
    assert torch.count_nonzero(fed_dz[64:128]) == 0


# ---------------------------------------------------------------------------
# Ambient FP16 autocast regression: the chunked backend allocates its per-chunk
# projection (``input_chunk @ weight.t()``) and dX (``dZ @ weight``) FRESH -- not
# through ``torch.mm(..., out=)`` like the legacy helper -- so those GEMMs are
# autocast-eligible ops. Under an ambient ``torch.autocast('cuda', float16)``
# (which ``amp_custom_fwd`` leaves enabled) a naive ``@`` would recast the BF16
# operands to FP16, producing an FP16 logits/dZ buffer that the BF16 dW/dX path
# cannot consume (NotImplementedError on the mixed-dtype GEMM), and would silently
# widen the kernel's supported-precision surface. The fix wraps only those two
# fresh GEMM sites in ``torch.autocast(enabled=False)``, so the fresh allocation
# and temp/copy semantics are preserved but the operands keep their BF16 dtype.
# This test reproduces the original B200 failure: end-to-end autograd under FP16
# autocast must (a) NOT raise, (b) keep the dZ operand fed to the GEMMs BF16 (no
# silent conversion), and (c) match -- bit-for-bit at the same geometry -- the
# identical call with autocast disabled. No tolerance widening.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("accum_dtype", [None, torch.bfloat16, torch.float32])
def test_fp16_autocast_preserves_bf16_gemm_operands(reduction, accum_dtype, monkeypatch):
    torch.manual_seed(50)
    # H64 / V4096 with 50 tokens is the shape Main reproduced the B200 bug on;
    # bump to a multi-chunk token count with a ragged tail so the per-chunk fresh
    # projection + dX GEMMs (and the final partial chunk) all run under autocast.
    n_tokens, hidden, vocab = 200, 64, 4096
    ignore_index = -100
    upstream = torch.tensor(0.7, device="cuda")  # non-unit upstream
    tiling = _expected_tiling(n_tokens, hidden, vocab)
    assert len(tiling) >= 2 and tiling[-1] != tiling[0]  # multichunk + ragged tail
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[:5] = ignore_index
    target[::13] = ignore_index

    # Spy the dW accumulation to prove the dZ operand fed to the GEMMs stays BF16
    # (an FP16 recast of the logits buffer would surface here as an FP16 dz_t).
    seen_dtypes = []
    real_accum = flce_mod._accumulate_dw

    def spy(buf, dz_t, x_chunk, accumulate):
        seen_dtypes.append((buf.dtype, dz_t.dtype, x_chunk.dtype))
        return real_accum(buf, dz_t, x_chunk, accumulate)

    monkeypatch.setattr(flce_mod, "_accumulate_dw", spy)

    # (a) End-to-end autograd under an ambient FP16 autocast must not raise.
    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = _apply(x, weight, target, reduction, ignore_index, accum_dtype)[0]
    loss.backward(upstream)

    # (b) No silent dtype conversion: every GEMM operand fed to the dW helper is
    # BF16 (the projection buffer the CE kernel wrote dZ into, and the input chunk).
    assert seen_dtypes, "dW accumulation never ran"
    assert all(dz_t == torch.bfloat16 and xc == torch.bfloat16 for _, dz_t, xc in seen_dtypes)
    # Gradients stay in the parameter dtype -- autocast did not widen precision.
    assert x.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16

    # (c) Identical call with autocast disabled -> bit-for-bit identical results
    # (same geometry, same operand dtypes; the guard makes autocast a no-op here).
    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", enabled=False):
        loss_ref = _apply(x_ref, weight_ref, target, reduction, ignore_index, accum_dtype)[0]
    loss_ref.backward(upstream)

    assert torch.equal(loss, loss_ref)
    assert torch.equal(x.grad, x_ref.grad)
    assert torch.equal(weight.grad, weight_ref.grad)
