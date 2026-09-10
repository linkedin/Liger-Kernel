import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed

pytest.importorskip("cuda.tile")

from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import _largest_power_of_2_le
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import _launch_matmul_tn
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import _resolve_chunk_size
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import chunked_fused_linear_cross_entropy_backward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import chunked_fused_linear_cross_entropy_forward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward
from liger_kernel.ops.cutile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

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


def _apply(x, weight, target, reduction="mean", ignore_index=-100, chunk_size=None):
    """Invoke the Function with the full 18-argument contract, defaulting the middle."""
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
        chunk_size,
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


# ---------------------------------------------------------------------------
# End-to-end correctness against the BF16-linear / FP32-CE reference, single
# chunk (chunk_size=None resolves to N for these shapes, so the projection GEMM
# tiling matches the reference and the original tight tolerances apply).
# Shapes exercise both dW schedules: H % 256 == 0 -> n4 kernel, else general.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 256),  # general dW branch (H=128)
        (129, 96, 384),  # general dW branch (H=96), tail token
        (32, 64, 8192),  # V > 4096 -> multiple vocab partitions
        (300, 256, 512),  # n4 dW branch (H=256)
        (200, 128, 257),  # odd vocab size
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_correctness(shape, reduction):
    tokens, hidden_size, vocab_size = shape
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
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index, None)
    loss.backward(upstream)

    atol, rtol = _tolerances(reduction)
    assert_verbose_allclose(loss_ref, loss, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Forced multi-chunk + tail correctness against the reference.
#
# A small chunk_size re-tiles the projection GEMM, so the chunked BF16 logits
# differ from the full-batch reference by cuBLAS tiling variance. Under `mean`
# this is normalized away (tight elementwise tolerance holds); under `sum` the
# unnormalized near-zero gradient elements can diverge by O(0.1) purely from
# GEMM tiling, so correctness is asserted as a small aggregate relative-L2 error
# (a structural chunking bug would blow the whole tensor up, not just the
# near-zero elements).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (130, 128, 256),  # general dW branch, 130 % 64 -> tail
        (200, 96, 384),  # general dW branch (H=96)
        (250, 128, 8192),  # V > 4096 -> multiple vocab partitions
        (300, 256, 512),  # n4 dW branch (H=256)
        (200, 128, 257),  # odd vocab size
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_forced_multichunk_correctness(shape, reduction):
    tokens, hidden_size, vocab_size = shape
    ignore_index = -100
    chunk_size = 64  # forces >= 2 chunks + a partial tail for every shape
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
    loss, _, _, _ = _apply(x, weight, target, reduction, ignore_index, chunk_size)
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
# Token chunking: forced multi-chunk with a tail, chunk_size == 1, and a
# chunk_size larger than N (clamped) must all match a single-chunk run.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", [1, 7, 64, 100000])
def test_chunking_matches_single_chunk(chunk_size):
    tokens, hidden_size, vocab_size = 130, 128, 512  # 130 % 64 -> tail
    ignore_index = -100
    upstream = torch.tensor(1.3, device="cuda")
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[:9] = ignore_index

    def run(cs):
        x = x_data.clone().requires_grad_(True)
        weight = weight_data.clone().requires_grad_(True)
        loss = _apply(x, weight, target, "mean", ignore_index, cs)[0]
        loss.backward(upstream)
        return loss.detach(), x.grad, weight.grad

    loss_full, gx_full, gw_full = run(tokens)  # single chunk
    loss_c, gx_c, gw_c = run(chunk_size)

    assert_verbose_allclose(loss_full, loss_c, atol=1e-4, rtol=1e-3)
    assert_verbose_allclose(gx_full, gx_c, atol=1e-3, rtol=1e-2)
    assert_verbose_allclose(gw_full, gw_c, atol=1e-3, rtol=1e-2)


# ---------------------------------------------------------------------------
# Global-mean normalization: a middle chunk that is entirely ignored still
# normalizes by the GLOBAL non-ignored count.
# ---------------------------------------------------------------------------
def test_global_mean_with_all_ignored_middle_chunk():
    tokens, hidden_size, vocab_size = 192, 128, 256
    ignore_index = -100
    chunk_size = 64  # 3 chunks; middle chunk fully ignored
    x_data = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab_size, (tokens,), device="cuda")
    target[64:128] = ignore_index

    x_ref = x_data.clone().requires_grad_(True)
    weight_ref = weight_data.clone().requires_grad_(True)
    loss_ref = _reference(x_ref, weight_ref, target, ignore_index, "mean")
    loss_ref.backward()

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, "mean", ignore_index, chunk_size)[0]
    loss.backward()

    assert_verbose_allclose(loss_ref, loss, atol=5e-3, rtol=5e-2)
    assert_verbose_allclose(x_ref.grad, x.grad, atol=5e-3, rtol=5e-2)
    assert_verbose_allclose(weight_ref.grad, weight.grad, atol=5e-3, rtol=5e-2)


def test_all_ignored_entire_batch():
    tokens, hidden_size, vocab_size = 96, 128, 256
    ignore_index = -100
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.full((tokens,), ignore_index, device="cuda")

    loss = _apply(x, weight, target, "mean", ignore_index, 32)[0]
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(x.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


# ---------------------------------------------------------------------------
# Independent gradient requirements: forward-only / input-only / weight-only /
# both. Chunked forward must honor each combination.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "requires_input_grad, requires_weight_grad",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_independent_gradient_requirements(requires_input_grad, requires_weight_grad):
    x = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16, requires_grad=requires_input_grad)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=requires_weight_grad)
    target = torch.randint(256, (200,), device="cuda")

    loss = _apply(x, weight, target, "mean", -100, 64)[0]
    assert torch.isfinite(loss)

    if requires_input_grad or requires_weight_grad:
        loss.backward()
        assert (x.grad is not None) == requires_input_grad
        assert (weight.grad is not None) == requires_weight_grad


# ---------------------------------------------------------------------------
# Repeated backward with a *different* non-unit upstream: the retained
# gradients must not be mutated, and gradients must scale linearly.
# ---------------------------------------------------------------------------
def test_repeated_backward_different_upstream():
    x = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (200,), device="cuda")
    target[:10] = -100
    loss = _apply(x, weight, target, "mean", -100, 64)[0]

    loss.backward(torch.tensor(1.0, device="cuda"), retain_graph=True)
    gx1 = x.grad.clone()
    gw1 = weight.grad.clone()
    x.grad = None
    weight.grad = None
    loss.backward(torch.tensor(2.5, device="cuda"), retain_graph=True)

    assert_verbose_allclose(x.grad, 2.5 * gx1, atol=1e-3, rtol=1e-3)
    assert_verbose_allclose(weight.grad, 2.5 * gw1, atol=1e-3, rtol=1e-3)


def test_retained_backward_preserves_saved_gradient():
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(256, (128,), device="cuda")
    upstream = torch.tensor(0.7, device="cuda")
    loss = _apply(x, weight, target, "mean", -100, 32)[0]

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
        _, grad_input, grad_weight, gradient_scale = chunked_fused_linear_cross_entropy_forward(
            x, weight, target, chunk_size=64
        )

    assert tuple(grad_input.shape) == (tokens, hidden_size)
    assert tuple(grad_weight.shape) == (vocab_size, hidden_size)
    # The retained dW accumulator is FP32 (scaled + cast to weight dtype in backward).
    assert grad_weight.dtype == torch.float32
    assert gradient_scale.ndim == 0
    # No retained tensor has the full [N, V] footprint.
    assert grad_input.numel() < tokens * vocab_size
    assert grad_weight.numel() < tokens * vocab_size


def test_reusable_logits_buffer_bounds_peak_memory():
    tokens, hidden_size, vocab_size = 4096, 128, 8192
    chunk_size = 128
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(vocab_size, (tokens,), device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    loss = _apply(x, weight, target, "mean", -100, chunk_size)[0]
    loss.backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base

    full_logits_bytes = tokens * vocab_size * 2  # bf16 [N, V]
    # A full [N, V] materialization would dominate; the chunked buffer must stay
    # well under it (chunked logits is chunk_size * V * 2 bytes).
    assert peak_delta < full_logits_bytes


# ---------------------------------------------------------------------------
# chunk_size resolution: validation and the default CuTe-native heuristic.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.5, "64"])
def test_invalid_chunk_size_rejected(bad):
    with pytest.raises(ValueError):
        _resolve_chunk_size(bad, 128)


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (1, 1),
        (128, 128),
        (2000, 512),
        (3000, 512),
        (5000, 1024),
        (4096, 1024),
        (65536, 1024),
    ],
)
def test_default_chunk_size_heuristic(tokens, expected):
    assert _resolve_chunk_size(None, tokens) == expected
    # Explicit heuristic: min(N, 1024, largest_pow2 <= max(512, N // 4)).
    manual = min(tokens, 1024, _largest_power_of_2_le(max(512, tokens // 4)))
    assert _resolve_chunk_size(None, tokens) == manual


def test_explicit_chunk_size_clamped_not_rounded():
    # Clamped to N, but not rounded to a power of two.
    assert _resolve_chunk_size(300, 128) == 128
    assert _resolve_chunk_size(70, 128) == 70  # kept as-is (not rounded to 64/128)


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

    # FP32 accumulation is explicitly allowed (it is the default).
    loss = LigerFusedLinearCrossEntropyFunction.apply(
        x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, torch.float32
    )[0]
    assert torch.isfinite(loss)

    with pytest.raises(NotImplementedError, match="accum"):
        LigerFusedLinearCrossEntropyFunction.apply(
            x, weight, target, None, None, -100, 0.0, 0.0, "mean", None, False, torch.float16
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
        None,
    )


def test_function_advertises_chunk_size_support():
    assert LigerFusedLinearCrossEntropyFunction.supports_chunk_size is True
    # ce_impl / ce_mode are self-identity placeholders only -- no real dispatch.
    assert LigerFusedLinearCrossEntropyFunction.supports_inner_impl_dispatch is False


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

    loss = _apply(x, weight, target, "mean", -100, None)[0]
    loss.backward(upstream)

    # Reference: original fused scale-into-cast path (dz = softmax(0) with the
    # target row decremented, single non-accumulating matmul that folds the
    # scale in before the BF16 store).
    dz = torch.full((1, 4096), 1.0 / 4096, dtype=torch.bfloat16, device="cuda")
    dz[0, 0] -= 1.0
    ref = torch.empty((4096, 64), dtype=torch.bfloat16, device="cuda")
    _launch_matmul_tn(dz, x.detach(), ref, upstream, False)

    assert weight.grad[1, 0].item() != 0.0
    assert_verbose_allclose(weight.grad, ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Strict SAME-CHUNK oracle: reproduce the kernel math per chunk with the exact
# same GEMM tiling (chunk-sized projection), then assert tight per-element
# tolerances. This is a stronger guard than the aggregate relative-L2 test:
# a structural chunking / scaling bug would move whole elements far beyond
# these tolerances. An independent full-batch PyTorch anchor is retained too.
# ---------------------------------------------------------------------------
def _same_chunk_oracle(x_data, weight_data, target, reduction, chunk_size, upstream, ignore_index=-100):
    n_tokens, hidden = x_data.shape
    vocab = weight_data.shape[0]
    device = x_data.device
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
        dw_fp32.add_(torch.mm(dz.t(), xc, out_dtype=torch.float32))  # FP32-accumulated dW
    combined = upstream * global_scale
    dx_final = dx * combined  # BF16 raw dX scaled by upstream * mean-normalizer
    dw_final = (dw_fp32 * combined).to(torch.bfloat16)  # FP32 scale, single BF16 cast
    return loss, dx_final, dw_final


@pytest.mark.parametrize("seed", [7, 314])
@pytest.mark.parametrize("hidden", [96, 256])  # general + n4 dW branches
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_same_chunk_oracle_regression(seed, hidden, reduction):
    torch.manual_seed(seed)
    n_tokens, vocab, chunk_size = 65, 4113, 17  # odd V, tail chunk (65 % 17)
    ignore_index = -100
    upstream = torch.tensor(-0.7, device="cuda")
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[17:34] = ignore_index
    target[::7] = ignore_index

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, reduction, ignore_index, chunk_size)[0]
    loss.backward(upstream)

    loss_oracle, dx_oracle, dw_oracle = _same_chunk_oracle(
        x_data, weight_data, target, reduction, chunk_size, upstream, ignore_index
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
# clearing the produced .grad) must not corrupt the saved FP32 dW accumulator
# or raw BF16 dX. A second backward with a different upstream must match a
# FRESH forward+backward at that upstream EXACTLY -- proving the scale/cast is
# recomputed from the retained FP32 dW (not a double-rounded rescale of the
# first BF16 result).
# ---------------------------------------------------------------------------
def test_retained_backward_alias_safety():
    torch.manual_seed(11)
    n_tokens, hidden, vocab, chunk_size = 128, 128, 256, 32
    ignore_index = -100
    x_data = torch.randn(n_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_tokens,), device="cuda")
    target[:9] = ignore_index

    x = x_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    loss = _apply(x, weight, target, "mean", ignore_index, chunk_size)[0]

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
    loss_fresh = _apply(x_fresh, weight_fresh, target, "mean", ignore_index, chunk_size)[0]
    loss_fresh.backward(torch.tensor(-1.3, device="cuda"))

    assert_verbose_allclose(gx_second, x_fresh.grad, atol=0.0, rtol=0.0)
    assert_verbose_allclose(gw_second, weight_fresh.grad, atol=0.0, rtol=0.0)

    # Directly snapshot the retained low-level tensors and confirm two backward
    # passes with different upstreams leave them untouched.
    xg = x_data.clone().requires_grad_(True)
    wg = weight_data.clone().requires_grad_(True)
    with torch.no_grad():
        _, grad_input, dweight_accum, gradient_scale = chunked_fused_linear_cross_entropy_forward(
            xg, wg, target, chunk_size=chunk_size
        )
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
# dW matmul. This guards the original exported interface against silent
# contract drift (the chunked path uses the separate chunked_* helpers).
# ---------------------------------------------------------------------------
def test_legacy_lowlevel_forward_backward_contract():
    torch.manual_seed(0)
    n_tokens, hidden, vocab = 128, 128, 257  # odd V exercises the general dW branch
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
