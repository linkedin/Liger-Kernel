import operator

import torch
import triton

from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_gemm import K_ALIGNMENT
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_gemm import VOCAB_TILE_SIZE
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_gemm import fused_lse
from liger_kernel.ops.cutedsl.ops._fused_linear_cross_entropy_gemm import recompute_softmax
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import compare_version

_TILE_N = VOCAB_TILE_SIZE
_SUPPORTS_OUT_DTYPE = compare_version("torch", operator.ge, "2.8.0")


# Backward loop orientation for the core path. Small token chunks make the
# token-oriented dW update repeatedly read and write the entire (V, H) gradient,
# while vocabulary chunking writes each dW slice once and accumulates only the
# smaller fp32 dX. Keep the measured B200 crossover from the existing backend.
VOCAB_BWD = True
_VOCAB_BWD_MAX_TOKEN_CHUNK = 512


def _bwd_chunk_size(BT, H, V):
    """Backward chunk size == Liger/Triton FLCE formula (memory-parity)."""
    inc = triton.cdiv(V, H)
    return max(1, min(triton.next_power_of_2(triton.cdiv(BT, inc)), BT))


def _target_logit(X, W, target, chunk=8192):
    """x_tgt[i] = X[i] · W[target[i]], computed in row-chunks so the (chunk, H) gather of W[target]
    never materializes the full (BT, H) temp. At BT=65536 the un-chunked gather is ~1 GB of transient
    (the whole reason the 2-CTA forward's peak memory exceeded the 1-CTA kernel, which folds the
    target logit into its fused kernel). bf16 operands, fp32 accumulate."""
    BT = X.shape[0]
    x_tgt = torch.empty(BT, device=X.device, dtype=torch.float32)
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        x_tgt[s:e] = (X[s:e] * W[target[s:e]]).sum(-1, dtype=torch.float32)
    return x_tgt


def _recompute_softmax(Xc, W, colvec_bias, rowvec_bias=None):
    """Recompute and exponentiate logits in the native GEMM epilogue.

    ``colvec_bias`` is the per-token ``-LSE`` or ignored-row sentinel, while
    ``rowvec_bias`` is the optional vocabulary bias. The kernel writes only the
    input-dtype softmax buffer.
    """
    vocab_bias = None if rowvec_bias is None else rowvec_bias.reshape(-1)
    return recompute_softmax(
        Xc,
        W,
        colvec_bias,
        vocab_bias,
    )


def _validate_inputs(_input, weight, target, bias, ce_weight, reduction):
    if _input.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"_input and weight must be 2D, got {_input.shape} and {weight.shape}.")
    if target.ndim != 1 or target.shape[0] != _input.shape[0]:
        raise ValueError(f"target must have shape ({_input.shape[0]},), got {target.shape}.")
    if _input.shape[1] != weight.shape[1]:
        raise ValueError(f"Input and weight hidden dimensions must match, got {_input.shape[1]} and {weight.shape[1]}.")
    if _input.shape[0] == 0 or _input.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("FLCE requires non-empty token, hidden, and vocabulary dimensions.")
    if _input.device != weight.device or _input.device != target.device:
        raise ValueError(
            f"_input, weight, and target must share a device, got {_input.device}, {weight.device}, and {target.device}."
        )
    if _input.dtype != weight.dtype:
        raise TypeError(f"_input and weight must share a dtype, got {_input.dtype} and {weight.dtype}.")
    if not torch.is_floating_point(_input):
        raise TypeError(f"_input and weight must have a floating-point dtype, got {_input.dtype}.")
    if target.dtype != torch.long:
        raise TypeError(f"target must have dtype torch.long, got {target.dtype}.")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"bias must have shape ({weight.shape[0]},), got {bias.shape}.")
        if bias.device != _input.device or not torch.is_floating_point(bias):
            raise TypeError("bias must be a floating-point tensor on the same device as _input.")
    if ce_weight is not None:
        if ce_weight.ndim != 1 or ce_weight.shape[0] != weight.shape[0]:
            raise ValueError(f"ce_weight must have shape ({weight.shape[0]},), got {ce_weight.shape}.")
        if ce_weight.device != _input.device or not torch.is_floating_point(ce_weight):
            raise TypeError("ce_weight must be a floating-point tensor on the same device as _input.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be one of 'mean', 'sum', or 'none', got {reduction!r}.")


def _native_sm100_supported(tensor):
    return tensor.device.type == "cuda" and torch.cuda.get_device_capability(tensor.device) == (10, 0)


def _addmm_fp32_out(out, lhs, rhs):
    if _SUPPORTS_OUT_DTYPE:
        torch.addmm(out, lhs, rhs, out_dtype=torch.float32, out=out)
    else:
        torch.addmm(out, lhs.float(), rhs.float(), out=out)


def _mm_out(out, lhs, rhs):
    if out.dtype == lhs.dtype:
        torch.mm(lhs, rhs, out=out)
    elif _SUPPORTS_OUT_DTYPE:
        torch.mm(lhs, rhs, out_dtype=out.dtype, out=out)
    else:
        torch.mm(lhs.float(), rhs.float(), out=out)


@torch.compile(dynamic=True)
def _dx_correct(dx_main, w_tgt, valid, go, out_dtype):
    """grad_input chunk = go * (sm@W − valid·W[target]). ``dx_main`` = sm@W (chunk,H); ``w_tgt`` =
    W[target] (chunk,H) gather. The −onehot term of dX lives here in (chunk,H) space (H ≪ V), not as a
    (chunk,V) pass. Ignored rows: dx_main is already 0 (sm row = 0) and the correction is masked → 0.
    go is passed as a 0-d TENSOR so torch.compile doesn't specialize on the scalar and recompile."""
    corr = torch.where(valid[:, None], w_tgt, torch.zeros_like(w_tgt))
    return (go * (dx_main.float() - corr.float())).to(out_dtype)


@torch.compile(dynamic=True)
def _dx_correct_z(dx_main, w_tgt, valid, go, row_scale, out_dtype):
    """grad_input chunk with the z-loss factor: go * (row_scale·(sm@W) − valid·W[target]). ``row_scale``
    = (1 + 2·lse_square_scale·lse) is the per-row z-loss scalar (it scales the softmax term of dlogits
    but NOT the −one-hot label term)."""
    corr = torch.where(valid[:, None], w_tgt, torch.zeros_like(w_tgt))
    return (go * (row_scale[:, None] * dx_main.float() - corr.float())).to(out_dtype)


@torch.compile(dynamic=True)
def _dx_correct_ls(dx_main, w_tgt, valid, go, row_scale, onehot_scale, smooth_vec, out_dtype):
    """grad_input chunk with label smoothing (no ce_weight):
    dx = go·(row_scale·(sm@W) − eps·W_sum − (1−ls)·valid·W[target]).
    ``row_scale`` = c = 1+2·lss·lse (per-row; all-ones when z-loss is off). ``smooth_vec`` = eps·W_sum
    (H,), the −eps broadcast term folded out of vocab space (Σ_i eps·W_i). ``onehot_scale`` = (1−ls).
    Masked to valid rows (ignored rows: sm=0, corrections masked → 0)."""
    main = row_scale[:, None] * dx_main.float()
    smooth = torch.where(valid[:, None], smooth_vec.float()[None, :], torch.zeros_like(main))
    corr = torch.where(valid[:, None], onehot_scale * w_tgt.float(), torch.zeros_like(main))
    return (go * (main - smooth - corr)).to(out_dtype)


def _weighted_col_sum(W, cw, vchunk=16384):
    """Σ_i cw[i]·W[i] = (H,) fp32, chunked over vocab so the fp32 copy of W is bounded to (vchunk,H).
    Used by the ce_weight+label_smoothing forward for cwW_raw (the weighted column combination that the
    dX broadcast and scaled_x_sum both reuse). cw is already fp32; W is the (possibly bf16) weight."""
    V, H = W.shape
    out = torch.zeros(H, device=W.device, dtype=torch.float32)
    for vs in range(0, V, vchunk):
        ve = min(vs + vchunk, V)
        out += cw[vs:ve] @ W[vs:ve].float()
    return out


def _chunked_matvec(X, vec, chunk):
    """Σ over columns of (X @ Wᵀ) weighted by a per-vocab vector, collapsed to X @ vec where
    vec = Σ_i coeff_i·W_i is a precomputed (H,) vector. Computed chunked in fp32 (bounds the fp32 copy
    of X to (chunk,H)). Used by label smoothing (vec = W.sum(0) or ce_weight@W) — no (BT,V) logits."""
    BT = X.shape[0]
    out = torch.empty(BT, device=X.device, dtype=torch.float32)
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        out[s:e] = X[s:e].float() @ vec
    return out


@torch.compile(dynamic=True)
def _dx_correct_weighted(dx_main, w_tgt, valid, go, a_row, onehot_scale, cw_col, out_dtype):
    """grad_input chunk for the ce_weight case (subsumes label smoothing + z-loss when weighted):
    dx = go·(a_row·(sm@W) + onehot_scale·W[target] + cw_col), all masked to valid rows.
    ``a_row`` (per-row) = softmax coefficient (weighted CE + smooth + z), ``onehot_scale`` (per-row) =
    −weight_y·(1−ls)/swnw label term, ``cw_col`` = Σ_i(−eps·weight_i/swnw)·W_i (H,) or None (ls==0)."""
    out = a_row[:, None] * dx_main.float() + onehot_scale[:, None] * w_tgt.float()
    if cw_col is not None:
        out = out + cw_col.float()[None, :]
    out = torch.where(valid[:, None], out, torch.zeros_like(out))
    return (go * out).to(out_dtype)


def _scatter_target_grad_rowscaled(grad_weight, xc, tclamp, valid, row_scale, alpha):
    """Per-row-scaled one-hot scatter: grad_weight[target[i]] += alpha·row_scale[i]·xc[i] for valid i.
    Like _scatter_target_grad but the scatter magnitude varies per row (the weighted one-hot term
    −weight_y·(1−ls)/swnw). Ignored rows masked to 0."""
    xc_scaled = torch.where(valid[:, None], row_scale[:, None] * xc.float(), torch.zeros_like(xc, dtype=torch.float32))
    grad_weight.index_add_(0, tclamp, xc_scaled.to(grad_weight.dtype), alpha=alpha)


def _accum_grad_weight(grad_weight, dlogits_t, xc, alpha):
    """grad_weight += ``alpha`` * (dlogits_tᵀ-view @ xc), accumulated IN PLACE with a fused cuBLAS
    addmm. ``alpha`` = grad_output/n folded into the GEMM (no separate scale pass).

    Both branches use ``torch.addmm(out=grad_weight)`` so the (V, H) accumulate is a single cuBLAS
    call (``grad_weight + alpha·dlogitsᵀ@xc``) with NO transient (V, H) temp and an fp32-internal
    accumulate — 1.5-1.8x faster than ``grad_weight += (dlogitsᵀ@xc).to(dtype)`` (which allocates a
    (V, H) temp and does a separate read-modify-write add) on the dW-GEMM, the single biggest cost in
    training. Neither the Triton FLCE nor the 1-CTA cutedsl path fuses the *default* (bf16 grad_weight)
    accumulate — they only fuse the fp32 (accum_dtype) case — so this is a net win over both.

      * grad_weight fp32 (accum_dtype=fp32): addmm(out_dtype=fp32) folds the bf16 operands into the
        fp32 buffer (exact accumulation, no bf16-rounding-per-step). Needs torch>=2.8; falls back to
        ``+= alpha·(dlogitsᵀ@xc).float()``.
      * grad_weight bf16/fp16 (the default, memory-parity with Triton): plain fused addmm.
    """
    if grad_weight.dtype == torch.float32 and dlogits_t.dtype in (torch.float16, torch.bfloat16):
        if _SUPPORTS_OUT_DTYPE:
            torch.addmm(grad_weight, dlogits_t, xc, alpha=alpha, out_dtype=torch.float32, out=grad_weight)
        else:
            grad_weight += alpha * (dlogits_t.float() @ xc.float())
    else:
        torch.addmm(grad_weight, dlogits_t, xc, alpha=alpha, out=grad_weight)


def _scatter_target_grad(grad_weight, xc, tclamp, valid, alpha):
    """The −one-hot term of grad_weight: grad_weight[target[i]] += ``alpha``·xc[i] for valid rows i.
    A (chunk,H) scatter (index_add) into (V,H) — the one-hot moved OUT of vocab space. Ignored rows are
    masked to 0 so they contribute nothing. ``xc`` is cast to grad_weight's dtype (index_add requires
    matching dtypes; fp32 for the accum_dtype path, else the param dtype)."""
    xc_masked = torch.where(valid[:, None], xc, torch.zeros_like(xc)).to(grad_weight.dtype)
    grad_weight.index_add_(0, tclamp, xc_masked, alpha=alpha)


def _vocab_chunk_size(V, BT, token_chunk):
    """Vocab-tile width for the vocab-oriented core backward, sized so the transient softmax buffer
    ``(BT, vc)`` is at most 1.75 times the token scheme's ``(token_chunk, V)``. The larger slice amortizes
    the persistent GEMM epilogue at small token chunks; dropping the final slice before label
    corrections keeps measured full-pass peak memory within 1% of the previous parity target.
    Floored to a multiple of ``_TILE_N`` so each native GEMM N-tile is full."""
    budget = 7 * token_chunk * V // 4
    vc = budget // BT
    vc = (vc // _TILE_N) * _TILE_N  # floor to a full N-tile -> buffer <= parity
    return max(_TILE_N, min(V, vc))


def _core_vocab_backward(ctx, grad_output):
    """Fused-fast-path backward for every case EXCEPT class-weighted CE (core / z-loss /
    label-smoothing / bias / token-scaling), chunked over the VOCAB dimension instead of tokens.
    Mathematically identical to the token path (dlogits = (per-row)·softmax − (1−ls)·onehot − ε, then
    dX = go·dlogits@W, dW = go·dlogitsᵀ@X) — only the loop orientation changes, so the dW GEMM
    contracts over K=BT (saturated) and writes each gW slice once (no per-chunk read-modify-write of
    the whole grad). Returns the 15-tuple of grads.

    Per vocab slice v: sm_v = exp(X@W_vᵀ − lse) masked (0 for ignored rows via the −1e30 colvec bias,
    the same fused native GEMM used by the token path). ``gX += (c·sm_v) @ W_v`` accumulates the
    softmax part of dX into the small (BT,H) fp32 buffer; ``gW[v] = (c·sm_v)ᵀ @ X`` is a single write.
    The label terms (one-hot, nonzero at one column per row; and the label-smoothing −ε broadcast) are
    applied ONCE outside the loop as cheap (BT,H) gather/scatter + (H,) rank-1 corrections — no (BT,V)
    pass — exactly as the token path does. ``c`` = z-loss coefficient (1 when off)."""
    X, W, target, valid, n, lse = ctx.saved_tensors
    BT, H = X.shape
    V = W.shape[0]
    token_chunk = _bwd_chunk_size(BT, H, V)
    vc = _vocab_chunk_size(V, BT, token_chunk)
    gw_dtype = ctx.accum_dtype if ctx.accum_dtype is not None else W.dtype
    colvec_bias = torch.where(valid, -lse, lse.new_full((), -1e30))  # −lse valid / −1e30 ignored
    tclamp = torch.where(valid, target, torch.zeros_like(target))

    # ---- ce_weight (class-weighted CE, subsuming label-smoothing + z-loss) --------------------------
    # The per-CLASS weights put a genuine per-COLUMN term c_col in dlogits
    # (d_i = a_row·sm_i + c_col[i] + [i==y]·onehot_scale). It still tiles over vocab: a_row (per-row
    # softmax coeff) pre-scales sm; the −one-hot uses the per-row onehot_scale; and c_col adds a rank-1
    # (c_col ⊗ Σ_valid X) dW update + a cw_col (H,) dX broadcast. ``go`` is the RAW upstream grad — the
    # swnw (weighted-count) normalizer is folded into a_row / onehot_scale / c_col.
    if ctx.cw is not None:
        lss = getattr(ctx, "lse_square_scale", 0.0)
        ls = getattr(ctx, "label_smoothing", 0.0)
        has_bias = getattr(ctx, "has_bias", False)
        bias_f = ctx.bias_f if has_bias else None
        scale = getattr(ctx, "scale", None)  # (BT,) token-scaling per-row weight (0 on ignored) or None
        cw = ctx.cw  # (V,) fp32 class weights
        eps = ls / V
        wy = cw[tclamp]  # (BT,)
        a_row = ((wy * (1.0 - ls) + eps * ctx.weight_sum) / ctx.swnw + (2.0 * lss) * lse / n)[:, None]  # (BT,1)
        onehot_scale = -(wy * (1.0 - ls)) / ctx.swnw  # (BT,) signed −one-hot magnitude
        go_wf = grad_output.detach().reshape(()).item()  # RAW upstream grad (swnw is inside a_row/…)
        # token-scaling multiplies EVERY per-row dlogits term by scale_r: fold it into a_row (→ sm),
        # the per-row one-hot (oh_coef), the cw_col dX broadcast, and the c_col / grad_bias count terms
        # (Σ_valid → Σ scale). ``arow_m`` pre-scales sm; ``cnt_w`` is the per-row count weight (0 ignored).
        arow_m = a_row if scale is None else a_row * scale[:, None]
        oh_coef = onehot_scale if scale is None else onehot_scale * scale  # (BT,) per-row one-hot magnitude
        cnt_w = valid.float() if scale is None else scale  # (BT,) per-row count weight (0 on ignored)
        gb_sm = torch.zeros(V, device=W.device, dtype=torch.float32) if has_bias else None

        gXacc = torch.zeros(BT, H, device=X.device, dtype=torch.float32)
        gW = torch.empty(V, H, device=W.device, dtype=gw_dtype)
        for s in range(0, V, vc):
            e = min(s + vc, V)
            Wv = W[s:e]
            rvb = bias_f[s:e].reshape(1, -1) if has_bias else None
            sm = _recompute_softmax(X, Wv, colvec_bias, rvb)
            sm.mul_(arow_m)  # (a_row·scale)⊙sm (per-row softmax coefficient × token-scaling)
            _addmm_fp32_out(gXacc, sm, Wv)  # dX += (arow_m·sm)@W
            _mm_out(gW[s:e], sm.t(), X)  # dW[v] = (arow_m·sm)ᵀ@X, write-once
            if has_bias:
                gb_sm[s:e] = sm.sum(0, dtype=torch.float32)  # Σ_r arow_m·sm colsum

        del sm
        gW.mul_(go_wf)
        oh_x = torch.where(valid[:, None], oh_coef.to(X.dtype)[:, None] * X, torch.zeros_like(X)).to(gw_dtype)
        gW.index_add_(0, tclamp, oh_x, alpha=go_wf)  # + go·oh_coef·X @ targets (bf16 (BT,H) temp)
        if ls > 0.0:
            c_col = (-eps / ctx.swnw) * cw  # (V,) per-column smooth weight
            x_valid_sum = (X * cnt_w[:, None]).sum(0, dtype=torch.float32)  # (H,) Σ_r w_r·X_r
            gW.add_((go_wf * c_col).to(gW.dtype)[:, None] * x_valid_sum.to(gW.dtype)[None, :])  # rank-1 outer
        gW = gW.to(W.dtype)

        # dX = go·(arow_m·sm@W + oh_coef·W[target] + [scale·]cw_col), masked to valid rows.
        gXacc.mul_(go_wf)
        if ls > 0.0:
            cw_col = ((-eps / ctx.swnw) * ctx.cwW_raw).float()  # (H,)
            if scale is None:
                gXacc.add_(cw_col, alpha=go_wf)  # + go·cw_col (H,) broadcast
            else:
                gXacc.add_((go_wf * scale)[:, None] * cw_col)  # + go·scale·cw_col (per-row)
        gX = gXacc.to(X.dtype)
        gX.add_((go_wf * oh_coef).to(X.dtype)[:, None] * W[tclamp])  # + go·oh_coef·W[target] (bf16)
        gX.mul_(valid[:, None])

        grad_bias = None
        if has_bias:
            grad_bias = gb_sm
            if ls > 0.0:
                grad_bias = grad_bias + ((-eps / ctx.swnw) * cw) * cnt_w.sum()  # + c_col·Σw
            grad_bias.index_add_(0, tclamp, torch.where(valid, oh_coef, torch.zeros_like(oh_coef)))
            grad_bias = (go_wf * grad_bias).to(W.dtype)
        return gX, gW, None, grad_bias, None, None, None, None, None, None, None, None, None, None, None

    go_f = grad_output.detach().reshape(()).item() / float(n.item())  # grad_output/n (one host sync)

    # z-loss and label-smoothing both fold into the vocab path with only per-row / (H,) corrections
    # (no (BT,V) pass, write-once dW intact):
    #   • z-loss  → per-row softmax coefficient c = 1 + 2·λ·lse; scale each sm slice by c IN PLACE so
    #     gXacc = c·(sm@W) for dX and (c·sm)ᵀ@X for dW fall out together (c≈1, bf16-safe, no buffer).
    #   • label-smoothing (ε = ls/V) → the one-hot is scaled by (1−ls); the −ε broadcast over ALL V
    #     columns folds OUT of vocab space into two (H,) vectors: −ε·W.sum(0) off every valid dX row,
    #     and −ε·Σ_valid(X) added to every dW row (one rank-1 update after the loop).
    lss = getattr(ctx, "lse_square_scale", 0.0)
    ls = getattr(ctx, "label_smoothing", 0.0)
    scale = getattr(ctx, "scale", None)  # (BT,) token-scaling per-row grad weight (stop-grad) or None
    z_factor = (1.0 + 2.0 * lss * lse)[:, None] if lss != 0.0 else None  # (BT,1) c (softmax term only)
    onehot_scale = 1.0 - ls  # one-hot magnitude (1.0 when no label smoothing)
    if ls != 0.0:
        eps = ls / V
        smooth_vec = (eps * W.sum(0, dtype=torch.float32)).to(X.dtype)  # (H,) −dX broadcast (valid rows)
    # token-scaling: a per-row weight on the WHOLE grad. It multiplies the softmax term (folded into
    # the sm pre-scale alongside c) AND the one-hot / smooth terms (applied per-row below). It never
    # co-occurs with bias/ce_weight (the forward falls back there), so those paths stay scalar-go.
    row_mul = z_factor  # sm pre-scale: c (softmax term only) …
    if scale is not None:
        sc = scale[:, None]
        row_mul = sc if row_mul is None else row_mul * sc  # … × token-scaling scale (whole per-row grad)
    # bias: the softmax gets the per-column bias slice; grad_bias = go·(colsum of c·sm − label terms),
    # accumulated write-once per vocab slice (disjoint columns) — no (BT,V) pass.
    has_bias = getattr(ctx, "has_bias", False)
    bias_f = ctx.bias_f if has_bias else None  # (V,) fp32 per-column FLCE bias
    gb_sm = torch.zeros(V, device=W.device, dtype=torch.float32) if has_bias else None  # Σ_r sm colsum

    gXacc = torch.zeros(BT, H, device=X.device, dtype=torch.float32)  # dX softmax part (accumulated)
    gW = torch.empty(V, H, device=W.device, dtype=gw_dtype)
    for s in range(0, V, vc):
        e = min(s + vc, V)
        Wv = W[s:e]
        rvb = bias_f[s:e].reshape(1, -1) if has_bias else None  # (1, e−s) bias slice for this vocab tile
        sm = _recompute_softmax(X, Wv, colvec_bias, rvb)  # (BT, e−s) masked softmax slice
        if row_mul is not None:
            sm.mul_(row_mul)  # (c·scale)⊙sm: scales the softmax term of BOTH dX and dW (labels added later)
        _addmm_fp32_out(gXacc, sm, Wv)  # dX += (·sm_v)@W_v (no transient)
        _mm_out(gW[s:e], sm.t(), X)  # dW[v] = (·sm_v)ᵀ@X, K=BT, write-once
        if has_bias:
            gb_sm[s:e] = sm.sum(0, dtype=torch.float32)  # softmax colsum (write-once per slice)

    del sm
    # The one-hot / label-smoothing terms carry the token-scaling `scale` (per-row) but NOT c: replace
    # X by scale·X for the dW label terms, and scale W[target]/W.sum(0) per-row for dX (no-op if scale None).
    # bf16 throughout (scale ∈ [0,1] softmax prob) so the per-row (BT,H) temps stay at core-case parity.
    x_row = X if scale is None else (scale.to(X.dtype)[:, None] * X)

    # dW = go·(·smᵀ@X − (1−ls)·onehot·(scale·X) − ε·Σ_valid(scale·X) [broadcast to ALL V rows]).
    gW.mul_(go_f)
    x_masked = torch.where(valid[:, None], x_row, torch.zeros_like(x_row)).to(gw_dtype)
    gW.index_add_(0, tclamp, x_masked, alpha=-go_f * onehot_scale)  # gW[target[i]] -= go·(1−ls)·(scale·X)[i]
    if ls != 0.0:
        gW.sub_((go_f * eps) * x_masked.sum(0, dtype=torch.float32).to(gw_dtype))  # −go·ε·Σ_valid(scale·X)
    gW = gW.to(W.dtype)

    # dX = go·(·sm@W − (1−ls)·[scale·]W[target] − ε·[scale·]W.sum(0)) for valid rows, 0 for ignored.
    gXacc.mul_(go_f)
    gX = gXacc.to(X.dtype)
    if scale is None:
        gX.add_(W[tclamp], alpha=-go_f * onehot_scale)  # −go·(1−ls)·W[target]
        if ls != 0.0:
            gX.sub_(smooth_vec, alpha=go_f)  # −go·ε·W.sum(0) (broadcast; ignored rows masked to 0 next)
    else:
        oh_row = ((go_f * onehot_scale) * scale).to(X.dtype)[:, None]  # (BT,1) per-row one-hot weight
        gX.sub_(oh_row * W[tclamp])  # −go·(1−ls)·scale·W[target] (bf16 (BT,H) temp — core-parity)
        if ls != 0.0:
            sv_row = (go_f * scale).to(X.dtype)[:, None]  # (BT,1)
            gX.sub_(sv_row * smooth_vec)  # −go·ε·scale·W.sum(0) (smooth_vec already bf16)
    gX.mul_(valid[:, None])

    # grad_bias[i] = go·(Σ_r w_r·c·sm[r,i] − ε·Σ_r w_r − (1−ls)·Σ_{r:y_r==i} w_r), where the per-row
    # weight w_r is the token-scaling `scale` (0 on ignored rows) or plain `valid` when unscaled. gb_sm
    # is already w-weighted (sm was pre-scaled by c·scale); the ε and one-hot count terms weight by w too.
    grad_bias = None
    if has_bias:
        bias_row_w = scale if scale is not None else valid.float()  # (BT,) per-row count weight (0 if ignored)
        grad_bias = gb_sm
        if ls != 0.0:
            grad_bias = grad_bias - eps * bias_row_w.sum()
        onehot_col = torch.zeros(V, device=W.device, dtype=torch.float32)
        onehot_col.index_add_(0, tclamp, bias_row_w)
        grad_bias = grad_bias - onehot_scale * onehot_col
        grad_bias = (go_f * grad_bias).to(W.dtype)
    return gX, gW, None, grad_bias, None, None, None, None, None, None, None, None, None, None, None


class _FlceState:
    """Plain state carrier standing in for autograd's ctx so the forward/backward compute is a
    pair of standalone module functions. ``fused_linear_cross_entropy_forward`` fills one; the
    autograd Function copies its fields onto the real ctx, and a direct functional caller passes
    the returned carrier straight to ``fused_linear_cross_entropy_backward``."""

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    bias=None,
    ce_weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """Fused linear + cross-entropy forward (loss only; logits never materialized). Returns
    ``(loss, z_loss, token_accuracy, predicted_tokens, state)`` where ``state`` carries everything
    the backward needs — pass it straight to ``fused_linear_cross_entropy_backward``. The autograd
    Function copies it onto its ctx; a direct functional caller uses it as-is."""
    ctx = _FlceState()
    _validate_inputs(_input, weight, target, bias, ce_weight, reduction)
    needs_grad = _input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    # Host-side target bounds check (matches the 1-CTA): zero out ignored tokens so they don't trip
    # it, then assert the valid targets are in [0, V). A device-side gather OOB would corrupt the
    # CUDA context; this raises a clean host AssertionError instead.
    _V = weight.shape[0]
    valid_targets = target[target != ignore_index]
    if valid_targets.numel() > 0:
        if valid_targets.max() >= _V:
            raise AssertionError(f"Target out of bounds. Expected < {_V}")
        if valid_targets.min() < 0:
            raise AssertionError("Target out of bounds. Expected >= 0")
    fully_trainable = _input.requires_grad and weight.requires_grad and (bias is None or bias.requires_grad)
    # Route to the general feature path when a feature the native 2-CTA path cannot
    # fuse is requested — softcap (its (1−tanh²) chain rule is per-element, so it breaks the
    # one-hot-as-correction backward), token accuracy / predicted tokens (need an argmax over V, and
    # the fused GEMM emits only the max VALUE, not its index — so they'd need a (chunk,V) pass with
    # no speed gain) — when the dtype is fp32 (the native GEMM is 16-bit only), or when the GPU is not
    # exactly SM100 (the kernel's tcgen05 launch and cluster policy are SM100-specific; SM103 and
    # SM120 use the general path), or when only a subset of operands requires gradients (the Triton
    # path avoids allocating unused large gradients). H need not be a multiple of the mainloop K tile:
    # it is zero-padded to the required alignment on the fast path
    # (the padded columns contribute 0 to every dot product → logits/grads exact; the all-zero padded
    # grad rows are sliced off in the backward). z_loss, label_smoothing, ce_weight, bias, and
    # token_scaling are all handled on the fast path. reduction="none" delegates to Triton because
    # its per-token upstream gradient requires deferred recomputation in backward.
    if (
        not _native_sm100_supported(_input)
        or reduction == "none"
        or (needs_grad and not fully_trainable)
        or (softcap is not None and softcap != 0.0)
        or return_token_accuracy
        or return_predicted_tokens
        or _input.dtype not in (torch.bfloat16, torch.float16)
    ):
        # fp32, non-SM100, reduction="none", softcap, and the argmax
        # metrics (token-accuracy / predicted-tokens) can't take the fused fast path. For these we
        # DELEGATE to the upstream Triton FLCE rather than run a memory-lean recompute fallback: on
        # these cases the GEMM has no tensor-core advantage (fp32 uses the same cutlass SIMT sgemm as
        # Triton), so an extra recompute GEMM in the backward is pure overhead — the recompute path
        # measured ~1.3× SLOWER than Triton for fp32. Delegating gives byte-identical numerics AND
        # exact speed/peak-memory parity with Triton (it IS the reference kernel). ``no_grad`` stops
        # a functional caller from building an autograd graph. Per-token gradients are deferred and
        # recomputed in backward; scalar reductions stash their gradients directly.
        from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward as _triton_flce_fwd

        defer_none = reduction == "none" and needs_grad
        with torch.no_grad():
            loss, z_out, acc_out, pred_out, gX, gW, gb = _triton_flce_fwd(
                _input=_input,
                weight=weight,
                target=target,
                ce_weight=ce_weight,
                bias=bias,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
                compute_gradients=False if defer_none else needs_grad,
                weight_requires_grad=weight.requires_grad,
            )
        if defer_none:
            ctx._triton_deferred_none = True
            ctx.save_for_backward(_input.detach(), weight.detach(), target, bias.detach() if bias is not None else None)
            ctx.fwd_kwargs = {
                "ce_weight": ce_weight,
                "ignore_index": ignore_index,
                "lse_square_scale": lse_square_scale,
                "label_smoothing": label_smoothing,
                "reduction": reduction,
                "softcap": softcap,
                "accum_dtype": accum_dtype,
                "use_token_scaling": use_token_scaling,
            }
            ctx.weight_requires_grad = weight.requires_grad
        else:
            ctx._triton_delegated = True
            ctx.grad_input = gX
            ctx.grad_weight = gW
            ctx.grad_bias = gb
        return loss, z_out, acc_out, pred_out, ctx
    # The native mainloop consumes full 64-element K tiles. Zero padding preserves
    # every dot product and produces padded gradient columns that are sliced below.
    h_orig = None
    _H = weight.shape[1]
    if _H % K_ALIGNMENT != 0:
        _pad = (-_H) % K_ALIGNMENT
        _input = torch.nn.functional.pad(_input, (0, _pad))
        weight = torch.nn.functional.pad(weight, (0, _pad))
        h_orig = _H
    ctx.h_orig = h_orig

    X = _input.detach().contiguous()
    W = weight.detach().contiguous()
    bias_f = bias.detach().float().contiguous() if bias is not None else None

    lse = fused_lse(X, W, bias_f)  # fused, logits (incl. per-column bias) never materialized
    valid = target != ignore_index
    tclamp_f = torch.where(valid, target, torch.zeros_like(target))  # ignored -> row 0 (gather-safe)
    x_tgt = _target_logit(X, W, tclamp_f)  # x_tgt[i] = X[i]·W[target[i]], chunked; clamped so the
    #                                       # ignore_index (-100) rows don't gather OOB at tiny V (their
    #                                       # x_tgt is discarded by ``valid`` below). Bounded mem.
    if bias_f is not None:
        x_tgt = x_tgt + bias_f[tclamp_f]  # target logit includes its bias term
    n = valid.sum().clamp(min=1).float() if reduction == "mean" else torch.ones((), device=X.device)
    V = W.shape[0]
    mean = reduction == "mean"
    eps = label_smoothing / V
    ctx.cw = None
    ctx.cwW_raw = None
    if ce_weight is not None:
        # ce_weight (class weights) — subsumes label smoothing + z-loss when weighted, mirroring the
        # Triton HAS_WEIGHT branch. loss_r = weight_y·(lse−x_tgt); with smoothing the soft term is
        # weighted too. Normalizers differ from the unweighted case: the CE/smooth part is divided by
        # sum_non_ignore_weight (Σ_valid weight_y), z-loss by n_non_ignore. All of this factors
        # through the backward GEMMs (see _dx_correct_weighted) — still NO (BT,V) logits.
        cw = ce_weight.detach().float()  # (V,)
        weight_sum = cw.sum()
        tcl = tclamp_f
        wy = cw[tcl]  # (BT,) weight at each row's target
        if mean:
            swnw = torch.where(valid, wy, torch.zeros_like(wy)).sum()
            if not valid.any():
                swnw = swnw.new_ones(())
        else:
            swnw = torch.ones((), device=X.device)
        ce_r = wy * (lse - x_tgt)
        if label_smoothing > 0.0:
            cwW_raw = _weighted_col_sum(W, cw)  # (H,) = Σ_i weight_i·W_i ; reused in backward for dX broadcast
            x_cwW = _chunked_matvec(X, cwW_raw, _bwd_chunk_size(X.shape[0], X.shape[1], V))
            # Σ_i logits_i·weight_i = X·cwW_raw + Σ_i bias_i·weight_i (the bias term is a scalar shift).
            if bias_f is not None:
                x_cwW = x_cwW + (bias_f * cw).sum()
            smooth = -eps * x_cwW + eps * lse * weight_sum
            per_row = ce_r * (1.0 - label_smoothing) + smooth
            ctx.cwW_raw = cwW_raw
        else:
            per_row = ce_r
        if mean:
            per_row = per_row / swnw
        per_row = torch.where(valid, per_row, torch.zeros_like(per_row))
        ctx.cw = cw
        ctx.weight_sum = weight_sum
        ctx.swnw = swnw
    else:
        ce_raw = lse - x_tgt
        # label smoothing (unweighted). Adds a soft cross-entropy vs the uniform distribution:
        # smooth = −eps·Σ_i logits_i + ls·lse (eps = ls/V). Σ_i logits_i = X @ W_sum (chunked fp32
        # matvec) — no (BT,V) logits. loss = ce·(1−ls) + smooth. Gradient stays on the FAST path via
        # (H,) broadcast corrections (no vocab-space pass).
        if label_smoothing > 0.0:
            W_sum = W.sum(0, dtype=torch.float32)  # (H,)
            x_logit_sum = _chunked_matvec(X, W_sum, _bwd_chunk_size(X.shape[0], X.shape[1], V))
            # Σ_i logits_i = X·W_sum + Σ_i bias_i (per-column bias adds a constant per-row shift).
            if bias_f is not None:
                x_logit_sum = x_logit_sum + bias_f.sum()
            smooth = -eps * x_logit_sum + label_smoothing * lse
            per_row = ce_raw * (1.0 - label_smoothing) + smooth
        else:
            per_row = ce_raw
        per_row = torch.where(valid, per_row, torch.zeros_like(per_row))
        if mean:
            per_row = per_row / n  # unweighted CE/smooth normalizer is n_non_ignore
    # z-loss (PaLM) — an auxiliary lse_square_scale·lse² term. The per-row scalar it contributes to
    # the gradient (1 + 2·lse_square_scale·lse) factors cleanly through the backward's dX/dW GEMMs,
    # so z-loss stays on the FAST path (no fallback). ignored rows contribute 0. z-loss is normalized
    # by n_non_ignore (NOT weight) — matches Triton — so it's divided by n even in the weighted case.
    if lse_square_scale != 0.0 or return_z_loss:
        z_row = torch.where(valid, lse_square_scale * lse * lse, torch.zeros_like(lse))
        z_scaled = z_row / n if mean else z_row  # (BT,)
    else:
        z_scaled = None
    per_row_total = per_row if z_scaled is None else per_row + z_scaled  # (BT,) per-token loss
    # token scaling (LinkedIn focal-style): multiply each token's loss by its target softmax prob
    # scale = softmax(logits)[target] = exp(x_tgt − lse), treated as a STOP-GRADIENT constant (the
    # backward just scales that row's dlogits by it). Cheap here (x_tgt, lse already in hand). x_tgt
    # already includes the per-column bias term (added above), so scale is the true softmax-at-target
    # even WITH bias / ce_weight — the backward threads it through all their dgrad terms.
    scale = None
    if use_token_scaling:
        scale = torch.where(valid, torch.exp(x_tgt - lse), torch.zeros_like(lse))
        per_row_total = per_row_total * scale
        if z_scaled is not None:
            z_scaled = z_scaled * scale
    if reduction == "none":
        loss = per_row_total  # (BT,) per-token loss (grad_output will be a (BT,) vector)
        z_total = z_scaled if (return_z_loss and z_scaled is not None) else None
    else:
        loss = per_row_total.sum()
        z_total = z_scaled.sum() if (return_z_loss and z_scaled is not None) else None
    if z_total is not None:
        z_total = z_total.to(X.dtype)

    # Save the forward LSE — the backward reuses it (as the −lse colvec bias in _recompute_exp) so
    # its softmax needs no per-chunk logsumexp reduction. It's just (BT,) fp32 = negligible mem.
    ctx.save_for_backward(X, W, target, valid, n, lse)
    ctx.ignore_index = ignore_index
    ctx.accum_dtype = accum_dtype
    ctx.lse_square_scale = lse_square_scale
    ctx.label_smoothing = label_smoothing
    ctx.reduction = reduction
    ctx.bias_f = bias_f  # (V,) fp32 per-column bias (None if no bias) — backward needs grad_bias
    ctx.has_bias = bias is not None
    ctx.scale = scale  # (BT,) token-scaling per-row factor (None if off) — backward folds it into go
    return loss, z_total, None, None, ctx  # (loss, z_loss, token_acc, pred, state)


def fused_linear_cross_entropy_backward(ctx, grad_output):
    """Fused linear + cross-entropy backward. ``ctx`` is the forward's state carrier (or the autograd
    ctx). Returns the grad tuple aligned to the forward args (grad_input, grad_weight, None, grad_bias,
    None...). When the forward zero-padded H to the native K-tile alignment (``ctx.h_orig`` set),
    the all-zero padded columns of grad_input/grad_weight are sliced back to the original H."""
    grads = _flce_backward(ctx, grad_output)
    h_orig = getattr(ctx, "h_orig", None)
    if h_orig is not None:
        grads = (grads[0][:, :h_orig].contiguous(), grads[1][:, :h_orig].contiguous(), *grads[2:])
    return grads


def _flce_backward(ctx, grad_output):
    """Dispatch the actual backward compute (general fallback vs the fused fast paths). Operates on
    whatever H it is given (possibly the TMA-padded H); the public wrapper slices padded grads."""
    # z_loss / token_accuracy / predicted_tokens are metrics — no gradient flows from them.
    if getattr(ctx, "_triton_deferred_none", False):
        from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward as _triton_flce_fwd

        _input, weight, target, bias = ctx.saved_tensors
        _, _, _, _, gX, gW, gb = _triton_flce_fwd(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            token_grad_output=grad_output,
            compute_gradients=True,
            weight_requires_grad=ctx.weight_requires_grad,
            **ctx.fwd_kwargs,
        )
        return (gX, gW, None, gb, None, None, None, None, None, None, None, None, None, None, None)
    # Triton-delegated fallback (fp32 / non-Blackwell / softcap / argmax-metrics): the grads were
    # already computed in the delegated forward; apply the scalar/per-row grad_output scale exactly
    # as Triton's own backward does (element_mul kernel; no-op when grad_output == 1).
    if getattr(ctx, "_triton_delegated", False):
        from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_backward as _triton_flce_bwd

        gX, gW, gb = _triton_flce_bwd(grad_output, ctx.grad_input, ctx.grad_weight, ctx.grad_bias)
        return (gX, gW, None, gb, None, None, None, None, None, None, None, None, None, None, None)
    X, W, target, valid, n, lse = ctx.saved_tensors
    BT, H = X.shape
    V = W.shape[0]
    chunk = _bwd_chunk_size(BT, H, V)
    colvec_bias = torch.where(valid, -lse, lse.new_full((), -1e30))  # −lse valid / −1e30 ignored
    tclamp = torch.where(valid, target, torch.zeros_like(target))
    lss = getattr(ctx, "lse_square_scale", 0.0)
    ls = getattr(ctx, "label_smoothing", 0.0)
    has_bias = getattr(ctx, "has_bias", False)
    # grad_output is applied as a FINAL scalar multiply in every fast dgrad helper (_dx_correct*),
    # so the bf16-softmax path's RELATIVE error is grad_output-independent (measured constant from
    # go=1 to go=65536, and ≤ the old fp32-residual "accurate" path at every logit scale). Hence no
    # grad_output≠1 fallback is needed — the fast path runs for the last-layer (go==1) case AND the
    # rarer not-last-layer / loss-scaled / grad-accum (go≠1) cases alike.

    # Vocab-oriented backward (opt-in, gated). EVERY case — core, z-loss (per-row softmax coefficient
    # c), label-smoothing (one-hot ×(1−ls) + (H,) broadcasts), bias (per-column softmax term +
    # write-once grad_bias), token-scaling (per-row grad weight), and ce_weight (per-row a_row + a
    # per-column c_col rank-1 update) — now tiles over vocab, so the dW GEMM contracts over K=BT and
    # writes each gW slice once (no per-chunk read-modify-write of the whole grad). Gated to small
    # token chunks, where the token dW GEMM starves; above the threshold the token path below already
    # saturates and the vocab reorientation would cost memory for no speed.
    # CHECKED BEFORE the token-path gX/gW allocations below — the vocab path allocates its own
    # (V,H) grad, so pre-allocating the token gW here would double the (V,H) footprint and erase the
    # memory parity that is the whole point.
    # token_scaling combined with ce_weight/bias is expressible ONLY on the vocab path (its per-row
    # scale threads through a_row / one-hot / c_col / grad_bias); the token path below wires
    # token_scaling only through the unweighted/no-bias branches. FORCE vocab for that combo so a
    # chunk>256 (large-batch) shape stays correct — it's a rare combo, so the perf crossover is moot.
    force_vocab = getattr(ctx, "scale", None) is not None and (has_bias or ctx.cw is not None)
    if force_vocab or (VOCAB_BWD and chunk <= _VOCAB_BWD_MAX_TOKEN_CHUNK):
        return _core_vocab_backward(ctx, grad_output)

    gX = torch.empty_like(X)
    # grad_weight dtype: accum_dtype when given, else the weight dtype — identical to Triton/1-CTA.
    # The weight-dtype default (bf16/fp16) keeps peak memory on par with Triton; accum_dtype=fp32
    # opts into exact accumulation (nearly doubles the (V, H) grad footprint).
    gw_dtype = ctx.accum_dtype if ctx.accum_dtype is not None else W.dtype
    gW = torch.zeros(V, H, device=W.device, dtype=gw_dtype)
    bias_rowvec = ctx.bias_f.reshape(1, -1) if has_bias else None
    n_valid = valid.sum().float()  # actual non-ignored count (grad_bias column terms use this, not n)

    if ctx.cw is not None:
        # ---- ce_weight (class-weight) backward, subsuming label smoothing + z-loss ----------------
        # Triton HAS_WEIGHT dlogits: d_i = a_row·sm_i + c_col[i] + [i==y]·onehot_scale, where
        #   a_row (per row)   = (weight_y·(1−ls) + eps·weight_sum)/swnw + 2·lss·lse/n   (softmax coeff)
        #   c_col[i]          = −eps·weight_i/swnw                                       (per column)
        #   onehot_scale      = −weight_y·(1−ls)/swnw                                    (label term)
        # These factor through the GEMMs: dX = go·(a_row·(sm@W) + onehot·W[y] + Σ_i c_col[i]W[i]),
        # dW = go·(smᵀ@(a_row·Xc) + scatter(onehot·Xc) + outer(c_col, Σ_valid Xc)). No (chunk,V) pass.
        # go here is the RAW upstream grad (normalizers are already inside a_row/onehot/c_col).
        cw = ctx.cw  # (V,) fp32
        weight_sum = ctx.weight_sum
        swnw = ctx.swnw
        cwW_raw = ctx.cwW_raw  # (H,) or None
        eps = ls / V
        wy = cw[tclamp]  # (BT,)
        a_row = (wy * (1.0 - ls) + eps * weight_sum) / swnw + (2.0 * lss) * lse / n
        onehot_scale = -(wy * (1.0 - ls)) / swnw
        cw_col = (-eps / swnw) * cwW_raw if ls > 0.0 else None  # (H,) dX broadcast
        go_w = grad_output.detach().float().reshape(()) if grad_output.numel() == 1 else lse.new_ones(())
        go_wf = go_w.item()
        gb_sm = torch.zeros(V, device=W.device, dtype=torch.float32) if has_bias else None  # Σ_r a_row·sm
        for s in range(0, BT, chunk):
            e = min(s + chunk, BT)
            Xc, tcl, vm, cb = X[s:e], tclamp[s:e], valid[s:e], colvec_bias[s:e]
            sm = _recompute_softmax(Xc, W, cb, bias_rowvec)
            gX[s:e] = _dx_correct_weighted(
                sm @ W,
                W[tcl],
                vm,
                go_w,
                a_row[s:e],
                onehot_scale[s:e],
                cw_col,
                X.dtype,
            )
            xc_scaled = (a_row[s:e, None] * Xc.float()).to(X.dtype)
            _accum_grad_weight(gW, sm.t(), xc_scaled, go_wf)
            _scatter_target_grad_rowscaled(gW, Xc, tcl, vm, onehot_scale[s:e], go_wf)
            if has_bias:
                gb_sm += (a_row[s:e, None] * sm.float()).sum(0)  # softmax part of grad_bias colsum
        if ls > 0.0:
            # dW += go·outer(c_col, Σ_valid Xc): the −eps·weight_i smooth term (per column, constant
            # across rows) → one rank-1 (V,H) update. Only paid when ce_weight AND smoothing are both on.
            c_col = (-eps / swnw) * cw  # (V,)
            x_valid_sum = (X * valid[:, None]).sum(0, dtype=torch.float32)  # (H,)
            gW.add_((go_wf * c_col).to(gW.dtype)[:, None] * x_valid_sum.to(gW.dtype)[None, :])
        gW = gW.to(W.dtype)
        grad_bias = None
        if has_bias:
            # grad_bias[i] = go·(Σ_r a_row·sm[r,i] + c_col[i]·n_valid + Σ_{r:y_r==i} onehot_scale_r).
            grad_bias = gb_sm
            if ls > 0.0:
                grad_bias = grad_bias + ((-eps / swnw) * cw) * n_valid
            grad_bias.index_add_(0, tclamp, torch.where(valid, onehot_scale, torch.zeros_like(onehot_scale)))
            grad_bias = (go_wf * grad_bias).to(W.dtype)
        return gX, gW, None, grad_bias, None, None, None, None, None, None, None, None, None, None, None

    # ---- unweighted backward (core / z-loss / label-smoothing) ------------------------------------
    # The dX / dW GEMMs use cuBLAS (torch.mm / addmm). The logit recompute + exp is FUSED into one
    # native GEMM whose epilogue emits the masked softmax sm = exp(Xc@W.T - lse) directly (0 for
    # ignored rows via the −1e30 bias), so no fp32 (chunk,V) logits are ever materialized AND the
    # ignore-mask is free. The one-hot subtract is applied OUT of vocab space as (chunk,H)
    # gather/scatter corrections (H ≪ V) — no (chunk,V) elementwise pass.
    # go = grad_output / n. A 0-d TENSOR feeds the compiled _dx_correct (so torch.compile doesn't
    # specialize on the scalar value); a Python float feeds the addmm / index_add ``alpha`` args.
    # TOKEN SCALING wants a PER-ROW grad weight: each row's dlogits is scaled by its detached
    # softmax-at-target ``scale``. Fold (grad_output/n)·scale into a (BT,) go_row and take the
    # per-row backward path (scale Xc/onehot rows by go_row, addmm alpha=1). Wired only through the
    # unweighted/no-bias branches — the gate sends token_scaling + ce_weight/bias to fallback.
    reduction = ctx.reduction
    mean = reduction == "mean"
    scale = getattr(ctx, "scale", None)  # (BT,) token-scaling factor or None
    per_row_grad = scale is not None
    go = go_f = go_row = None
    if per_row_grad:
        gob = grad_output.detach().float().reshape(())
        base = gob / n if mean else gob  # scalar upstream grad (÷ n for mean)
        go_row = base * scale  # (BT,) per-row gradient weight
    else:
        if grad_output.numel() == 1:
            go = grad_output.detach().float().reshape(()) / n
        else:
            go = 1.0 / n
        go_f = go.item()  # one host sync per backward (Triton FLCE does the same for n_non_ignore)
    # z-loss per-row factor c = 1 + 2·lse_square_scale·lse (1.0 when z-loss is off). It scales the
    # softmax term of dlogits but not the −one-hot label term; being per-row, it factors through
    # both the dX (row-scale of sm@W) and dW (row-scale of Xc) GEMMs — no (chunk,V) materialization.
    z_factor = (1.0 + 2.0 * lss * lse) if lss != 0.0 else None
    # label-smoothing backward terms (only when ls>0): the −eps constant (broadcast over ALL V
    # columns of dlogits) folds OUT of vocab space into two (H,) corrections — eps·W_sum subtracted
    # from every valid dX row, and eps·(Σ_valid Xc) subtracted from every dW row (one rank-1 update).
    # The one-hot label term is scaled by (1−ls). No (chunk,V) pass.
    if ls > 0.0:
        eps = ls / V
        smooth_vec = eps * W.sum(0, dtype=torch.float32)  # (H,) -> dX broadcast
        x_valid_sum = (X * valid[:, None]).sum(0, dtype=torch.float32)  # (H,) -> dW broadcast
    gb_sm = torch.zeros(V, device=W.device, dtype=torch.float32) if has_bias else None  # Σ_r rs·sm colsum
    for s in range(0, BT, chunk):
        e = min(s + chunk, BT)
        Xc, tcl, vm, cb = X[s:e], tclamp[s:e], valid[s:e], colvec_bias[s:e]
        gr = go_row[s:e] if per_row_grad else None  # (chunk,) per-row grad weight (none/token_scaling)
        sm = _recompute_softmax(Xc, W, cb, bias_rowvec)  # masked softmax, fused in one 2-SM GEMM (bf16)
        if ls > 0.0:
            # label smoothing (± z-loss): row_scale c folds the z-factor (ones if z off).
            rs = z_factor[s:e] if z_factor is not None else lse.new_ones(e - s)
            if per_row_grad:
                gX[s:e] = _dx_correct_ls(sm @ W, W[tcl], vm, gr[:, None], rs, 1.0 - ls, smooth_vec, X.dtype)
            else:
                gX[s:e] = _dx_correct_ls(sm @ W, W[tcl], vm, go, rs, 1.0 - ls, smooth_vec, X.dtype)
            xc_scaled = (rs[:, None] * Xc.float()).to(X.dtype) if z_factor is not None else Xc
            if per_row_grad:
                _accum_grad_weight(gW, sm.t(), (gr[:, None] * xc_scaled.float()).to(X.dtype), 1.0)
                _scatter_target_grad_rowscaled(gW, Xc, tcl, vm, -(1.0 - ls) * gr, 1.0)
            else:
                _accum_grad_weight(gW, sm.t(), xc_scaled, go_f)
                _scatter_target_grad(gW, Xc, tcl, vm, -go_f * (1.0 - ls))  # onehot scaled by (1−ls)
            if has_bias:
                gb_sm += (rs[:, None] * sm.float()).sum(0)
        elif z_factor is None:
            # grad_input = go·(sm@W − W[target]); the −one-hot is a (chunk,H) gather correction.
            if per_row_grad:
                gX[s:e] = _dx_correct(sm @ W, W[tcl], vm, gr[:, None], X.dtype)
            else:
                gX[s:e] = _dx_correct(sm @ W, W[tcl], vm, go, X.dtype)
            # grad_weight += go·(smᵀ@Xc); the −one-hot is a (chunk,H) scatter correction.
            if per_row_grad:
                _accum_grad_weight(gW, sm.t(), (gr[:, None] * Xc.float()).to(X.dtype), 1.0)
                _scatter_target_grad_rowscaled(gW, Xc, tcl, vm, -gr, 1.0)
            else:
                _accum_grad_weight(gW, sm.t(), Xc, go_f)
                _scatter_target_grad(gW, Xc, tcl, vm, -go_f)
            if has_bias:
                gb_sm += sm.float().sum(0)
        else:
            zc = z_factor[s:e]
            if per_row_grad:
                gX[s:e] = _dx_correct_z(sm @ W, W[tcl], vm, gr[:, None], zc, X.dtype)
            else:
                gX[s:e] = _dx_correct_z(sm @ W, W[tcl], vm, go, zc, X.dtype)
            # dW: smᵀ @ (c·Xc) − scatter(Xc). Scale Xc rows by c (cheap (chunk,H)); onehot is unscaled.
            xc_scaled = (zc[:, None] * Xc.float()).to(X.dtype)
            if per_row_grad:
                _accum_grad_weight(gW, sm.t(), (gr[:, None] * xc_scaled.float()).to(X.dtype), 1.0)
                _scatter_target_grad_rowscaled(gW, Xc, tcl, vm, -gr, 1.0)
            else:
                _accum_grad_weight(gW, sm.t(), xc_scaled, go_f)
                _scatter_target_grad(gW, Xc, tcl, vm, -go_f)
            if has_bias:
                gb_sm += (zc[:, None] * sm.float()).sum(0)
    if ls > 0.0:
        # dW −= go·eps·(Σ_valid Xc) broadcast over all V rows (the −eps smooth term's weight grad).
        if per_row_grad:
            xvs = (go_row[:, None] * X.float() * valid[:, None]).sum(0)  # (H,) per-row-weighted
            gW.sub_(eps * xvs.to(gW.dtype))
        else:
            gW.sub_((go_f * eps) * x_valid_sum.to(gW.dtype))
    gW = gW.to(W.dtype)  # cast back to the param dtype (no-op unless accum_dtype was fp32)
    grad_bias = None
    if has_bias:
        # grad_bias[i] = go·(Σ_r rs·sm[r,i] − eps·n_valid − (1−ls)·#{valid r: y_r==i}). The softmax
        # colsum is gb_sm; the −eps constant is broadcast over all V; the one-hot is a valid-target
        # bincount scaled by (1−ls).
        grad_bias = gb_sm
        if ls > 0.0:
            grad_bias = grad_bias - eps * n_valid
        onehot_col = torch.zeros(V, device=W.device, dtype=torch.float32)
        onehot_col.index_add_(0, tclamp, valid.float())
        grad_bias = grad_bias - (1.0 - ls) * onehot_col
        grad_bias = (go_f * grad_bias).to(W.dtype)
    # grads align with forward args (_input, weight, target, bias, ce_weight, ignore_index,
    # lse_square_scale, label_smoothing, reduction, softcap, return_z_loss, accum_dtype,
    # use_token_scaling, return_token_accuracy, return_predicted_tokens)
    return gX, gW, None, grad_bias, None, None, None, None, None, None, None, None, None, None, None


# =============================================================================
class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Fused linear + cross-entropy on the 2-SM GEMM. A thin autograd wrapper over
    ``fused_linear_cross_entropy_forward`` / ``fused_linear_cross_entropy_backward``. Same
    ``.apply`` signature and 4-tuple return as the Triton/cutedsl providers."""

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    ):
        loss, z_loss, token_accuracy, predicted_tokens, state = fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            bias,
            ce_weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            accum_dtype,
            use_token_scaling,
            return_token_accuracy,
            return_predicted_tokens,
        )
        # Transfer the standalone state carrier onto the autograd ctx — tensors via save_for_backward
        # (proper autograd bookkeeping), the rest as plain attrs. No copies (same tensor refs).
        if getattr(state, "saved_tensors", None) is not None:
            ctx.save_for_backward(*state.saved_tensors)
        for _k, _v in vars(state).items():
            if _k != "saved_tensors":
                setattr(ctx, _k, _v)
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2=None, grad_output3=None, grad_output4=None):
        # z_loss / token_accuracy / predicted_tokens are metrics — no gradient flows from them.
        return fused_linear_cross_entropy_backward(ctx, grad_output)
