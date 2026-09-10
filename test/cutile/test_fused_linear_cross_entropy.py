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


def _apply(x, weight, target, reduction="mean", ignore_index=-100):
    """Invoke the Function with the 17-argument module-style contract.

    There is NO public ``chunk_size`` argument: the token-chunk geometry is the
    shared default Triton policy (:func:`_get_chunk_size`). ``ce_impl`` / ``ce_mode``
    are self-identity placeholders and default to ``None``.
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
        None,  # accum_dtype
        False,  # use_token_scaling
        False,  # return_token_accuracy
        False,  # return_predicted_tokens
        None,  # ce_impl
        None,  # ce_mode
    )


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
def test_correctness(shape, reduction):
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
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index)
    loss.backward(upstream)

    atol, rtol = _tolerances(reduction)
    assert_verbose_allclose(loss_ref, loss, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=atol, rtol=rtol)


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
def test_multichunk_correctness(shape, reduction):
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
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index)
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
def test_independent_gradient_requirements(requires_input_grad, requires_weight_grad):
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
    loss = _apply(x, weight, target, "mean", ignore_index)[0]
    assert torch.isfinite(loss)

    # Same-chunk oracle on detached leaves -- the loss and each requested grad
    # match tightly regardless of which gradients were requested.
    loss_oracle, dx_oracle, dw_oracle = _same_chunk_oracle(x_data, weight_data, target, "mean", upstream, ignore_index)
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
    # only the requested gradient buffers are materialized (with the right
    # dtype/shape); the unrequested ones stay None. Run under no_grad on the
    # leaves themselves -- the forward keys off tensor.requires_grad directly.
    xl = x_data.clone().requires_grad_(requires_input_grad)
    wl = weight_data.clone().requires_grad_(requires_weight_grad)
    with torch.no_grad():
        ll_loss, grad_input, dweight_accum, gradient_scale = chunked_fused_linear_cross_entropy_forward(
            xl, wl, target, reduction="mean", ignore_index=ignore_index
        )
    assert_verbose_allclose(ll_loss, loss_oracle, atol=1e-2, rtol=1e-3)
    if requires_input_grad:
        assert grad_input is not None
        assert grad_input.dtype == torch.bfloat16
        assert tuple(grad_input.shape) == (tokens, hidden_size)
    else:
        assert grad_input is None
    if requires_weight_grad:
        assert dweight_accum is not None
        assert dweight_accum.dtype == torch.float32
        assert tuple(dweight_accum.shape) == (vocab_size, hidden_size)
    else:
        assert dweight_accum is None
    assert gradient_scale.ndim == 0


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
# Memory contract: the forward retains only [N, H] dX and [V, H] dW (plus the
# scalar scale) -- never a full [N, V] logits/dZ tensor -- and the reusable
# logits buffer keeps peak memory below the full [N, V] materialization.
# ---------------------------------------------------------------------------
def test_retains_only_final_gradients_no_full_logits():
    tokens, hidden_size, vocab_size = 256, 128, 4096
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")

    with torch.no_grad():
        _, grad_input, grad_weight, gradient_scale = chunked_fused_linear_cross_entropy_forward(x, weight, target)

    assert tuple(grad_input.shape) == (tokens, hidden_size)
    assert tuple(grad_weight.shape) == (vocab_size, hidden_size)
    # The retained dW accumulator is FP32 (scaled + cast to weight dtype in backward).
    assert grad_weight.dtype == torch.float32
    assert gradient_scale.ndim == 0
    # No retained tensor has the full [N, V] footprint.
    assert grad_input.numel() < tokens * vocab_size
    assert grad_weight.numel() < tokens * vocab_size


def test_reusable_logits_buffer_bounds_peak_memory():
    # V > 16*H so the default geometry is genuinely multi-chunk (buffer_rows < N).
    tokens, hidden_size, vocab_size = 4096, 128, 8192
    tiling = _expected_tiling(tokens, hidden_size, vocab_size)
    assert len(tiling) >= 2 and tiling[0] < tokens
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    loss = _apply(x, weight, target, "mean", -100)[0]
    loss.backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base

    full_logits_bytes = tokens * vocab_size * 2  # bf16 [N, V]
    # A full [N, V] materialization would dominate; the chunked buffer must stay
    # well under it (chunked logits is buffer_rows * V * 2 bytes).
    assert peak_delta < full_logits_bytes


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


def test_accum_dtype_fp32_accepted_others_rejected():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (64,), device="cuda")

    # FP32 accumulation is explicitly allowed (it is the default policy).
    loss = LigerFusedLinearCrossEntropyFunction.apply(
        x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, torch.float32
    )[0]
    assert torch.isfinite(loss)

    with pytest.raises(NotImplementedError, match="accum"):
        LigerFusedLinearCrossEntropyFunction.apply(
            x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, torch.float16
        )


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
# dW small-gradient preservation: the FP32 dW accumulator is retained and the
# combined upstream/mean scale is applied in FP32 BEFORE the single BF16 cast.
# An early cast in the forward (before the upstream scale) flushes recoverable
# small gradients to zero.
# ---------------------------------------------------------------------------
def test_small_dw_gradient_survives_bf16_cast():
    # dW element magnitude ~ (1/V) * 1e-37 is below the BF16 subnormal floor,
    # but after the go=65536 upstream scale it is representable again.
    x = torch.full((1, 64), 1e-37, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([0], device="cuda")
    upstream = torch.tensor(65536.0, device="cuda")

    loss = _apply(x, weight, target, "mean", -100)[0]
    loss.backward(upstream)

    # Independent FP32 Torch reference: dZ = softmax(0) with the target row
    # decremented; the raw dW is the FP32 cuBLAS product dZ.T @ X, then the
    # combined (upstream * mean-normalizer, here 1/1) scale is applied in FP32
    # and cast to BF16 exactly once -- matching the retained-accumulator path.
    dz = torch.full((1, 4096), 1.0 / 4096, dtype=torch.bfloat16, device="cuda")
    dz[0, 0] -= 1.0
    dw_fp32 = _ref_dw_fp32(dz.t(), x.detach())  # FP32 dW = dZ.T @ X
    ref = (dw_fp32 * upstream.to(torch.float32)).to(torch.bfloat16)  # FP32 scale, single BF16 cast

    assert weight.grad[1, 0].item() != 0.0
    assert_verbose_allclose(weight.grad, ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Strict SAME-CHUNK oracle: reproduce the kernel math per chunk with the exact
# same GEMM tiling, DERIVED from the shared default geometry (not a chunk_size
# argument), then assert tight per-element tolerances. A structural chunking /
# scaling bug would move whole elements far beyond these tolerances. An
# independent full-batch PyTorch anchor is retained too.
# ---------------------------------------------------------------------------
def _same_chunk_oracle(x_data, weight_data, target, reduction, upstream, ignore_index=-100):
    n_tokens, hidden = x_data.shape
    vocab = weight_data.shape[0]
    device = x_data.device
    # Derive the chunk size from the shared policy -- identical to the kernel.
    chunk_size = _get_chunk_size(n_tokens, hidden, vocab)
    valid = target != ignore_index
    global_scale = 1.0 / valid.sum().clamp_min(1).float() if reduction == "mean" else torch.tensor(1.0, device=device)
    loss = torch.zeros((), dtype=torch.float32, device=device)
    dx = torch.zeros((n_tokens, hidden), dtype=torch.bfloat16, device=device)
    dw_fp32 = torch.zeros((vocab, hidden), dtype=torch.float32, device=device)
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
        dz[rows[valid_c], tc[valid_c]] -= 1.0
        dz[~valid_c] = 0.0
        dz = dz.to(torch.bfloat16)
        dx[start:stop] = torch.mm(dz, weight_data)  # raw BF16 dX
        dw_fp32.add_(_ref_dw_fp32(dz.t(), xc))  # FP32-accumulated dW
    combined = upstream * global_scale
    dx_final = dx * combined  # BF16 raw dX scaled by upstream * mean-normalizer
    dw_final = (dw_fp32 * combined).to(torch.bfloat16)  # FP32 scale, single BF16 cast
    return loss, dx_final, dw_final


@pytest.mark.parametrize("seed", [7, 314])
@pytest.mark.parametrize("hidden", [96, 256])  # non-aligned + 256-aligned H tails
@pytest.mark.parametrize("reduction", ["mean", "sum"])
# force_fallback drives the pre-2.8 dW path (plain mm/addmm on FP32-upcast BF16
# operands). The oracle's ``_ref_dw_fp32`` tracks the same flag, so both paths
# stay bit-identical and the SAME tight tolerances apply -- forced multi-chunk
# numerical coverage of the older-torch accumulation with no tolerance widening.
@pytest.mark.parametrize("force_fallback", [False, True])
def test_same_chunk_oracle_regression(seed, hidden, reduction, force_fallback, monkeypatch):
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
    loss = _apply(x, weight, target, reduction, ignore_index)[0]
    loss.backward(upstream)

    loss_oracle, dx_oracle, dw_oracle = _same_chunk_oracle(
        x_data, weight_data, target, reduction, upstream, ignore_index
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

    # Directly snapshot the retained low-level tensors and confirm two backward
    # passes with different upstreams leave them untouched.
    xg = x_data.clone().requires_grad_(True)
    wg = weight_data.clone().requires_grad_(True)
    with torch.no_grad():
        _, grad_input, dweight_accum, gradient_scale = chunked_fused_linear_cross_entropy_forward(xg, wg, target)
    grad_input_snap = grad_input.clone()
    dweight_snap = dweight_accum.clone()
    chunked_fused_linear_cross_entropy_backward(
        torch.tensor(1.0, device="cuda"), grad_input, dweight_accum, gradient_scale, torch.bfloat16
    )
    chunked_fused_linear_cross_entropy_backward(
        torch.tensor(-1.3, device="cuda"), grad_input, dweight_accum, gradient_scale, torch.bfloat16
    )
    assert_verbose_allclose(grad_input, grad_input_snap, atol=0.0, rtol=0.0)
    assert_verbose_allclose(dweight_accum, dweight_snap, atol=0.0, rtol=0.0)


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
# cuTile MMA kernel. The chunked path must emit exactly one FP32-out ``aten::mm``
# (first chunk) and one FP32-out ``aten::addmm`` per subsequent chunk; projection
# and dX use the plain (BF16-out) ``aten::mm.out`` overload. The removed cuTile
# matmul kernels must no longer exist on the module.
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
        if name in ("aten.mm.dtype_out", "aten.addmm.dtype_out"):
            result = out[0] if isinstance(out, (tuple, list)) else out
            self.dw_out_dtypes.append(result.dtype)
        return out


def test_no_custom_matmul_kernels_remain():
    # The dW GEMM is cuBLAS now; the old cuTile MMA kernels/launcher are gone.
    for removed in ("_matmul_tn_kernel", "_matmul_tn_n4_kernel", "_launch_matmul_tn"):
        assert not hasattr(flce_mod, removed), f"{removed} should have been removed"
    # The CE / scale-cast cuTile kernels are retained.
    assert hasattr(flce_mod, "_scale_cast_kernel")
    assert hasattr(flce_mod, "_fused_cross_entropy_dz_kernel")


def test_chunked_dw_uses_mm_then_addmm_fp32_out():
    torch.manual_seed(3)
    # V > 16*H so the default geometry is multi-chunk: [16, 16, 16, 16, 6] -> 5 chunks.
    n_tokens, hidden, vocab = 70, 96, 8192
    x = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab, (n_tokens,), device="cuda")

    n_chunks = len(_expected_tiling(n_tokens, hidden, vocab))
    assert n_chunks >= 2
    with _AtenOpRecorder() as rec, torch.no_grad():
        chunked_fused_linear_cross_entropy_forward(x, weight, target)

    # First dW chunk -> one FP32-out mm; every subsequent chunk -> FP32-out addmm.
    if flce_mod._ADDMM_SUPPORTS_OUT_DTYPE:
        assert rec.ops.count("aten.mm.dtype_out") == 1
        assert rec.ops.count("aten.addmm.dtype_out") == n_chunks - 1
        assert rec.dw_out_dtypes == [torch.float32] * n_chunks
        # dW never accumulated straight into a BF16 tensor via a plain addmm.
        assert "aten.addmm.out" not in rec.ops
        assert "aten.addmm.default" not in rec.ops
    else:
        # Pre-2.8 fallback: no ``.dtype_out`` overload. dW overwrites once via a
        # plain mm.out then accumulates via plain addmm.out into the FP32 buffer.
        # addmm is issued ONLY by the dW accumulation, so its count is chunks - 1.
        assert "aten.mm.dtype_out" not in rec.ops
        assert "aten.addmm.dtype_out" not in rec.ops
        assert rec.ops.count("aten.addmm.out") == n_chunks - 1
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
        # buffer -- no ``.dtype_out`` overload and no addmm accumulation.
        assert "aten.mm.dtype_out" not in rec.ops
        assert "aten.addmm.dtype_out" not in rec.ops
        assert rec.dw_out_dtypes == []
    assert grad_weight.dtype == weight.dtype


# ---------------------------------------------------------------------------
# Pre-2.8 fallback: when the ``out_dtype`` mm/addmm overload is unavailable the
# helper must upcast the bounded operands to FP32 and use the plain mm/addmm (no
# ``.dtype_out`` op). Small gradients must still survive the deferred cast.
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
        chunked_fused_linear_cross_entropy_forward(x, weight, target)

    # No out_dtype overload is used on the fallback path.
    assert "aten.mm.dtype_out" not in rec.ops
    assert "aten.addmm.dtype_out" not in rec.ops
    # dW still overwrites once (plain mm.out) then accumulates (plain addmm.out).
    # addmm is issued ONLY by the dW accumulation, so its count is chunks - 1.
    assert rec.ops.count("aten.addmm.out") == n_chunks - 1


def test_pre_2_8_fallback_small_gradient_survives(monkeypatch):
    monkeypatch.setattr(flce_mod, "_ADDMM_SUPPORTS_OUT_DTYPE", False)
    x = torch.full((1, 64), 1e-37, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([0], device="cuda")
    upstream = torch.tensor(65536.0, device="cuda")

    loss = _apply(x, weight, target, "mean", -100)[0]
    loss.backward(upstream)

    # The FP32 fallback preserves the sub-BF16 gradient just like the overload path.
    dz = torch.full((1, 4096), 1.0 / 4096, dtype=torch.bfloat16, device="cuda")
    dz[0, 0] -= 1.0
    dw_fp32 = torch.mm(dz.t().to(torch.float32), x.detach().to(torch.float32))  # bounded-operand upcast
    ref = (dw_fp32 * upstream.to(torch.float32)).to(torch.bfloat16)

    assert weight.grad[1, 0].item() != 0.0
    assert_verbose_allclose(weight.grad, ref, atol=0.0, rtol=0.0)
