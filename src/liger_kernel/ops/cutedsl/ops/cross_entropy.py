import inspect

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
import triton

from cutlass import Float32
from cutlass import Int32
from cutlass import const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device_arch

# Matches the Triton CE backward cap (NVIDIA-only path, so the NPU 2048 cap is irrelevant).
_MAX_FUSED_SIZE = 65536 // 2

# log2(e) and ln(2): the online-softmax math is done in base-2 (hardware ex2.approx)
# then converted, exactly mirroring the Triton kernel for numerical parity.
LOG2_E = 1.4426950408889634
NEG_LOG2_E = -1.4426950408889634  # for FMA-folding exp2 args: exp2(x*LOG2_E + (-m*LOG2_E))
LN2 = 0.6931471805599453


# Pick the nvvm.fmax calling convention from the installed binding itself rather
# than from a CUDA/torch/cutlass version (which can disagree with the wheel): the
# CUDA 12.9-era binding takes the result type as its first positional arg
# (res, a, b, ...); newer bindings infer it (a, b, ...).
# NOTE: single `except Exception` (not a tuple of types) on purpose — this module is
# AST-preprocessed by cute.compile, whose import scanner crashes on a tuple-form
# `except (A, B):` handler at module scope.
try:
    _FMAX_NEEDS_RESULT_TYPE = next(iter(inspect.signature(nvvm.fmax).parameters)) == "res"
except Exception:
    _FMAX_NEEDS_RESULT_TYPE = False  # assume the newer binding (matches FA4)


@dsl_user_op
def fmax(a, b, c=None, *, loc=None, ip=None) -> Float32:
    """Hardware fp32 max via the NVVM ``fmax`` intrinsic.

    Adapted from FlashAttention-4 (flash_attn/cute/utils.py), which calls
    ``nvvm.fmax`` directly since cutlass.cute exposes no high-level scalar fmax.
    FA4 targets the newer binding (result type inferred); we also support the
    older CUDA 12.9 binding, which needs the result type as the first argument.
    """
    av = Float32(a).ir_value(loc=loc, ip=ip)
    bv = Float32(b).ir_value(loc=loc, ip=ip)
    cv = Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None
    if _FMAX_NEEDS_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), av, bv, c=cv, loc=loc, ip=ip))
    return Float32(nvvm.fmax(av, bv, c=cv, loc=loc, ip=ip))


# A very negative fp32 sentinel used as -inf for the running max (avoids relying
# on a float("-inf") literal surviving the DSL trace).
NEG_INF_F32 = -1.0e38


# cp.async pipeline depth: prefetch this many row tiles gmem->smem (cache_mode=GLOBAL bypasses
# L1) while computing the current one. A "tile" = one 128-bit vector per thread = THREADS*16
# bytes, dtype-independent. 4 measured best on B200 (fp32 -5% vs 3; bf16 flat). 5 crushes
# occupancy at 32 warps (80 KB smem/CTA) and regresses fp32-large, so 4 is the sweet spot.
_NUM_STAGES = 4

# Compiled-kernel cache keyed on (dtypes, has_grad, feature flags, num_warps) — everything the
# kernel bakes. V/BT are dynamic so one compile serves all shapes. REQUIRED: without it the
# @cute.jit host fn recompiles on every call (~30 ms that dwarfs the kernel).
_compile_cache = {}

# Per-call host overhead is constant (~25 us): it doesn't scale with BT/V, so it dominates small
# shapes and vanishes at scale. Cache the CUstream wrapper keyed on torch's raw stream handle so
# we don't reconstruct the cuda.CUstream object every launch (general — no shape dependence).
_stream_cache = {}


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    s = _stream_cache.get(raw)
    if s is None:
        s = cuda.CUstream(raw)
        _stream_cache[raw] = s
    return s


# =============================================================================
# Device-side helpers
# =============================================================================
@cute.jit
def _warp_online_combine(m: Float32, d: Float32):
    """Full-warp online-softmax reduction via butterfly shuffle.

    After this every lane holds the row's final (max, normalizer). Combine rule:
        m' = max(m1, m2);  d' = d1*2^((m1-m')*log2e) + d2*2^((m2-m')*log2e)
    """
    for i in cutlass.range_constexpr(5):  # log2(32) = 5 butterfly steps
        offset = 1 << i
        m_o = cute.arch.shuffle_sync_bfly(m, offset=offset)
        d_o = cute.arch.shuffle_sync_bfly(d, offset=offset)
        m_new = fmax(m, m_o)
        d = d * cute.math.exp2((m - m_new) * LOG2_E, fastmath=True) + d_o * cute.math.exp2(
            (m_o - m_new) * LOG2_E, fastmath=True
        )
        m = m_new
    return m, d


@cute.jit
def _warp_argmax_combine(am: Float32, acol: Float32):
    """Full-warp argmax reduction via butterfly shuffle.

    Reduces (max logit value, smallest column achieving it) — the smallest-index
    tie-break matches Triton's argmax. ``acol`` is the column carried as fp32 (exact
    for V < 2^24). After this every lane holds the warp's (max, argmax-col).
    """
    for i in cutlass.range_constexpr(5):  # log2(32) = 5 butterfly steps
        offset = 1 << i
        om = cute.arch.shuffle_sync_bfly(am, offset=offset)
        oc = cute.arch.shuffle_sync_bfly(acol, offset=offset)
        # strictly-greater wins; exact tie keeps the smaller column. (Both compares use
        # the pre-update `am`, and the two conditions are mutually exclusive.)
        if om > am:
            acol = oc
        if om == am:
            if oc < acol:
                acol = oc
        am = fmax(am, om)
    return am, acol


@cute.jit
def _advance(idx, n: cutlass.Constexpr):
    """Circular increment of a ring index. For a power-of-2 ring (NUM_STAGES=4) this is a
    single AND mask (`(idx+1) & (n-1)`) instead of the ISETP+SEL+IADD a compare-select emits
    — one fewer ALU instruction per call, and the steady-state loop calls it twice per tile.
    Falls back to the branchless compare-select for non-power-of-2 n (resolved at trace time)."""
    if const_expr(n & (n - 1) == 0):
        return (idx + 1) & (n - 1)
    return idx + 1 if idx < n - 1 else 0


# =============================================================================
# Device kernel
# =============================================================================
@cute.kernel
def _ce_fwd_kernel(
    mX: cute.Tensor,  # (BT, V) logits in/out (gradient written in-place)
    mY: cute.Tensor,  # (BT,) int64 targets
    mLoss: cute.Tensor,  # (BT,) per-row loss (input dtype, matches Triton)
    mZLoss: cute.Tensor,  # (BT,) per-row z_loss out; written only if RETURN_Z_LOSS
    mTokenAcc: cute.Tensor,  # (BT,) fp32 per-row accuracy out; written only if RETURN_TOKEN_ACCURACY
    mPredTok: cute.Tensor,  # (BT,) int64 per-row argmax out; written only if RETURN_PREDICTED_TOKENS
    mWeight: cute.Tensor,  # (V,) fp32 class weights; read only if HAS_WEIGHT
    inv_n_loss: Float32,  # main-loss/grad normalizer: 1/sum_non_ignore_weight (mean+weight), 1/n (mean), else 1.0
    inv_n_z: Float32,  # z_loss normalizer: 1/n_non_ignore (mean), else 1.0 (z_loss is never weight-scaled)
    lse_sq_scale: Float32,  # lse_square_scale (z_loss coefficient); unused if not HAS_ZLOSS
    softcap: Float32,  # logit soft-cap threshold; unused if not HAS_SOFTCAP
    label_smoothing: Float32,  # smoothing amount; unused if not HAS_SMOOTHING
    weight_sum: Float32,  # sum of the full weight vector; used only by weighted smoothing
    ignore_index: Int32,
    HAS_GRAD: cutlass.Constexpr,
    HAS_ZLOSS: cutlass.Constexpr,  # lse_square_scale != 0 or return_z_loss
    RETURN_Z_LOSS: cutlass.Constexpr,  # write per-row z_loss to mZLoss
    HAS_SOFTCAP: cutlass.Constexpr,  # apply softcap*tanh(x/softcap) to logits
    RETURN_TOKEN_ACCURACY: cutlass.Constexpr,  # write per-row (argmax == y) to mTokenAcc
    RETURN_PREDICTED_TOKENS: cutlass.Constexpr,  # write per-row argmax column to mPredTok
    HAS_WEIGHT: cutlass.Constexpr,  # scale loss/grad by per-class weight
    HAS_SMOOTHING: cutlass.Constexpr,  # label_smoothing != 0
    NUM_WARPS: cutlass.Constexpr,  # warps/CTA cooperating on one row (8 for 2-byte, 32 for fp32)
):
    THREADS = const_expr(32 * NUM_WARPS)
    tid, _, _ = cute.arch.thread_idx()  # 0..THREADS-1
    lane = tid % 32
    warp = tid // 32
    row, _, _ = cute.arch.block_idx()

    # token_accuracy and predicted_tokens both reduce to the row's argmax column.
    NEED_ARGMAX = const_expr(RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS)

    # Cross-warp reduction scratch (one (m, d)[, argmax][, scaled_x_sum] per warp), carved from
    # one smem pool sized for NUM_WARPS; the cp.async tile buffer comes from the same pool below.
    # NUM_WARPS is compile-time, so the smem footprint shrinks for the 8-warp bf16 kernel and
    # grows only for the 32-warp fp32 kernel — keeping occupancy high where it matters.
    _smem = cutlass.utils.SmemAllocator()
    sm_m = _smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_d = _smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_argm = _smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_argcol = _smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)
    sm_sxs = _smem.allocate_tensor(Float32, cute.make_layout(NUM_WARPS), byte_alignment=4)

    gX = mX[row, None]  # 1D (V,) view of this row; grad written in-place here
    V = gX.shape[0]
    # cp.async's 128-bit atom needs a 16-byte-aligned gmem source, but the dynamic
    # row slice drops the static alignment to element size. Rebuild the row pointer
    # with an explicit 16-byte alignment — valid because each row starts 16-aligned
    # (V*sizeof is a multiple of 16 given V % VEC == 0).
    gX = cute.make_tensor(
        cute.make_ptr(mX.element_type, gX.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((V,)),
    )

    y = mY[row]
    is_ignored = y == ignore_index
    # Clamp the target index for ignored rows so the gX[y] load can't go OOB
    # (ignore_index is typically negative). `y * 0` keeps y's dtype.
    y_safe = y
    if is_ignored:
        y_safe = y * 0

    ori_xy = gX[y_safe].to(Float32)  # logit at the target, for the loss
    if const_expr(HAS_SOFTCAP):
        # cap the target logit too — the loss uses the capped value, like Triton.
        ori_xy = softcap * cute.math.tanh(ori_xy / softcap)

    # per-row class weight (stays 1.0 when unweighted, so it can multiply unconditionally).
    w_eff = Float32(1.0)
    if const_expr(HAS_WEIGHT):
        w_eff = mWeight[y_safe].to(Float32)
    # label-smoothing per-class mass eps = label_smoothing / V (matches Triton).
    eps = Float32(0.0)
    if const_expr(HAS_SMOOTHING):
        eps = label_smoothing / Float32(V)

    # 128-bit vectorization + cp.async pipeline. gXv: (VEC, V//VEC). 256 threads
    # cooperate; each loads its 128-bit vector per tile. Tail predicated -> V % VEC.
    VEC = const_expr(128 // gX.element_type.width)
    gXv = cute.tiled_divide(gX, (VEC,))
    num_vec = V // VEC
    num_tiles = (num_vec + THREADS - 1) // THREADS
    x_frag = cute.make_rmem_tensor((VEC,), gX.element_type)

    # Weighted label smoothing is the only path that needs per-column weights (the smoothing
    # loss/grad use weight_block); load them as a plain vectorized gmem read (the (V,) weight
    # vector is shared by every row, so it stays L2-resident — no cp.async pipeline needed).
    NEED_WBLOCK = const_expr(HAS_WEIGHT and HAS_SMOOTHING)
    if const_expr(NEED_WBLOCK):
        gW = cute.make_tensor(
            cute.make_ptr(mWeight.element_type, mWeight.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((V,)),
        )
        gWv = cute.tiled_divide(gW, (VEC,))
        w_frag = cute.make_rmem_tensor((VEC,), Float32)

    # cp.async (L1-bypassing) atom + multi-stage smem tile view (VEC, 256, NUM_STAGES).
    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), gX.element_type, num_bits_per_copy=128
    )
    sTiles = _smem.allocate_tensor(gX.element_type, cute.make_layout((THREADS * VEC, _NUM_STAGES)), byte_alignment=16)
    sTilesV = cute.tiled_divide(sTiles, (VEC,))  # (VEC, THREADS, NUM_STAGES)

    # --- pass 1: [Online softmax] first pass — find the running max m and normalizer d
    # (Algorithm 3, https://arxiv.org/pdf/1805.02867): 2 loads + 1 store vs 3+1 for safe
    # softmax. cp.async-pipelined; each thread keeps its own (m, d) then we reduce across the CTA.
    m = Float32(NEG_INF_F32)
    d = Float32(0.0)
    # per-thread argmax: max raw logit seen + smallest column achieving it (V = sentinel).
    t_am = Float32(NEG_INF_F32)
    t_acol = Float32(V)
    # per-thread label-smoothing partial: sum of (-eps * x_capped [* weight]) over its cols.
    t_sxs = Float32(0.0)
    # prologue: prefetch the first NUM_STAGES-1 tiles
    for s in cutlass.range_constexpr(_NUM_STAGES - 1):
        p_vidx = s * THREADS + tid
        if p_vidx < num_vec:
            cute.copy(cp_atom, gXv[None, p_vidx], sTilesV[None, tid, s])
        cute.arch.cp_async_commit_group()
    # steady state: rotating stage indices (no % modulo) + incremental vidx (ALU win).
    read_stage = Int32(0)
    write_stage = Int32(_NUM_STAGES - 1)
    r_vidx = Int32(tid)
    w_vidx = Int32((_NUM_STAGES - 1) * THREADS + tid)
    for _t in cutlass.range(0, num_tiles):
        if w_vidx < num_vec:
            cute.copy(cp_atom, gXv[None, w_vidx], sTilesV[None, tid, write_stage])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(_NUM_STAGES - 1)
        if r_vidx < num_vec:
            cute.autovec_copy(sTilesV[None, tid, read_stage], x_frag)  # smem -> reg
            if const_expr(NEED_ARGMAX):
                # argmax on the RAW logits: softcap is monotonic, so argmax(softcap(x)) ==
                # argmax(x) and the column is identical. Smaller j (and earlier, smaller-
                # column tiles) win ties via the strict `>`.
                base = r_vidx * VEC
                for j in cutlass.range_constexpr(VEC):
                    xj = x_frag[j].to(Float32)
                    if xj > t_am:
                        t_am = xj
                        t_acol = Float32(base + j)
            x_ssa = x_frag.load().to(Float32)  # (VEC,) fp32 TensorSSA
            if const_expr(HAS_SOFTCAP):
                x_ssa = softcap * cute.math.tanh(x_ssa / softcap)  # cap before max/sum
            if const_expr(HAS_SMOOTHING):
                # scaled_x_sum partial: sum of -eps * x_capped, * weight_block when weighted.
                neg_eps = Float32(0.0) - eps
                if const_expr(NEED_WBLOCK):
                    cute.autovec_copy(gWv[None, r_vidx], w_frag)
                    w_ssa = w_frag.load()
                    t_sxs = t_sxs + (x_ssa * w_ssa * neg_eps).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                else:
                    t_sxs = t_sxs + (x_ssa * neg_eps).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            local_max = x_ssa.reduce(cute.ReductionOp.MAX, Float32(NEG_INF_F32), 0)
            m_new = fmax(m, local_max)
            # FMA-fold: exp2((x - m_new)*LOG2_E) == exp2(x*LOG2_E + (-m_new*LOG2_E)).
            # Hoisting the per-tile scalar neg_m2 lets the per-element arg compile to one
            # FFMA instead of FADD+FMUL — fewer issued instructions on the issue-bound P1.
            neg_m2 = m_new * NEG_LOG2_E
            x_exp = cute.math.exp2(x_ssa * LOG2_E + neg_m2, fastmath=True)
            local_sum = x_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            d = d * cute.math.exp2((m - m_new) * LOG2_E, fastmath=True) + local_sum
            m = m_new
        read_stage = _advance(read_stage, _NUM_STAGES)
        write_stage = _advance(write_stage, _NUM_STAGES)
        r_vidx = r_vidx + THREADS
        w_vidx = w_vidx + THREADS
    cute.arch.cp_async_wait_group(0)  # drain remaining prefetches

    # warp-level reduce (collective: all 32 lanes) -> each lane has the warp's (m, d)
    m, d = _warp_online_combine(m, d)
    if const_expr(NEED_ARGMAX):
        t_am, t_acol = _warp_argmax_combine(t_am, t_acol)
    if const_expr(HAS_SMOOTHING):
        # plain warp sum-reduce of the scaled_x_sum partial (butterfly add).
        for _i in cutlass.range_constexpr(5):
            t_sxs = t_sxs + cute.arch.shuffle_sync_bfly(t_sxs, offset=1 << _i)
    # cross-warp: each warp's lane 0 publishes its (m, d)[, argmax][, scaled_x_sum]; then combine.
    if lane == 0:
        sm_m[warp] = m
        sm_d[warp] = d
        if const_expr(NEED_ARGMAX):
            sm_argm[warp] = t_am
            sm_argcol[warp] = t_acol
        if const_expr(HAS_SMOOTHING):
            sm_sxs[warp] = t_sxs
    cute.arch.barrier()
    m = Float32(NEG_INF_F32)
    d = Float32(0.0)
    if const_expr(NEED_ARGMAX):
        g_am = Float32(NEG_INF_F32)
        g_acol = Float32(V)
    if const_expr(HAS_SMOOTHING):
        g_sxs = Float32(0.0)
    for w in cutlass.range_constexpr(NUM_WARPS):
        mw = sm_m[w]
        dw = sm_d[w]
        m_new = fmax(m, mw)
        d = d * cute.math.exp2((m - m_new) * LOG2_E, fastmath=True) + dw * cute.math.exp2(
            (mw - m_new) * LOG2_E, fastmath=True
        )
        m = m_new
        if const_expr(NEED_ARGMAX):
            aw = sm_argm[w]
            cw = sm_argcol[w]
            if aw > g_am:
                g_acol = cw
            if aw == g_am:
                if cw < g_acol:
                    g_acol = cw
            g_am = fmax(g_am, aw)
        if const_expr(HAS_SMOOTHING):
            g_sxs = g_sxs + sm_sxs[w]

    # lse = m + ln(d) = m + log2(d)*ln2
    lse = m + cute.math.log2(d, fastmath=True) * LN2

    # main loss = weight_y * (lse - x_y) (weight_y == 1 when unweighted), then normalize by
    # inv_n_loss (1/sum_non_ignore_weight when weighted+mean, else 1/n or 1.0). The label-
    # smoothing mix is spliced in here under HAS_SMOOTHING.
    # loss = log(softmax(X_y)) = (X_y - m) - log d = X_y - lse, weighted by weight_y (== 1
    # when unweighted). sum(e^(X-m)) >= 1 (the max term is e^0 = 1) so this can't overflow.
    main = (lse - ori_xy) * w_eff
    if const_expr(HAS_SMOOTHING):
        # Label smoothing regularizes H(q, p) -> H(q', p), with eps = label_smoothing / V:
        #   H(q', p) = (1 - ls) * H(q, p) + ls * H(u, p)
        #            = (1 - ls) * H(q, p) + (sum(-eps * x_i) + ls * (m + log d))
        # g_sxs is the reduced sum(-eps * x_capped [* weight_block]); inv_n_loss normalizes
        # the whole. Refer to H(q', p) in section 7 of https://arxiv.org/pdf/1512.00567 and the
        # full derivation in https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
        if const_expr(HAS_WEIGHT):
            smooth_loss = g_sxs + eps * lse * weight_sum
        else:
            smooth_loss = g_sxs + label_smoothing * lse
        main = main * (Float32(1.0) - label_smoothing) + smooth_loss
    loss = main * inv_n_loss
    # An auxiliary z_loss = lse_square_scale * lse^2 (PaLM, page 14:
    # https://www.jmlr.org/papers/v24/22-1144.html), normalized by inv_n_z (always
    # /n_non_ignore for mean, never weight-scaled — matches Triton) and added to the per-row loss.
    zl = Float32(0.0)
    if const_expr(HAS_ZLOSS):
        zl = lse_sq_scale * lse * lse * inv_n_z
        loss = loss + zl
    if is_ignored:
        loss = Float32(0.0)
        zl = Float32(0.0)
    if tid == 0:
        mLoss[row] = loss.to(mLoss.element_type)
        if const_expr(RETURN_Z_LOSS):
            mZLoss[row] = zl.to(mZLoss.element_type)

    # token_accuracy / predicted_tokens: ignored rows get 0.0 / -1 (matches Triton).
    # The ignored sentinel is folded in at fp32 (col_out) before the single int cast — the
    # DSL forbids reassigning an int-typed var inside a dynamic `if`, but fp32 is fine
    # (same pattern as `loss = Float32(0.0)` above).
    if const_expr(NEED_ARGMAX):
        argcol = Int32(g_acol)  # global argmax column (g_acol is an integer-valued fp32)
        if tid == 0:
            if const_expr(RETURN_TOKEN_ACCURACY):
                acc = Float32(0.0)
                if Int32(y) == argcol:
                    acc = Float32(1.0)
                if is_ignored:
                    acc = Float32(0.0)
                mTokenAcc[row] = acc.to(mTokenAcc.element_type)
            if const_expr(RETURN_PREDICTED_TOKENS):
                col_out = g_acol
                if is_ignored:
                    col_out = Float32(-1.0)
                mPredTok[row] = Int32(col_out).to(mPredTok.element_type)

    # --- pass 2: [Online softmax] second pass — gradient, written in-place over the logits.
    # For 'mean' reduction (normalized by N = number of non-ignored elements):
    #   dx_i = softmax(x_i) / N,                                      i != y
    #   dx_y = (softmax(x_y) - 1) / N
    # With label smoothing (eps = label_smoothing / V):
    #   dx_i = (softmax(x_i) - eps) / N ;   dx_y = dx_i - (1 - ls) / N
    # With z_loss: every softmax(x_i) picks up a (1 + 2*lse_square_scale*lse) factor.
    # For 'sum'/'none' there is no /N. (Weighted variants scale by weight_y / sum_w; see below.)
    if const_expr(HAS_GRAD):
        # per-element grad = softmax_i * (coef / d). coef folds the dloss_ori coefficient and the
        # dz_loss coefficient 2*s*lse*inv_n_z. The two use DIFFERENT normalizers when weighted
        # (sum_w vs n), so they can't share one scale; the additive smoothing term (not
        # proportional to softmax) is applied per element below. is_ignored zeros the whole row.
        # dloss_ori coefficient: the weighted branch scales softmax by (1-ls)*weight_y; the
        # unweighted branch uses the equivalent simplified form (coefficient 1, with smoothing
        # handled by the additive -eps term applied per element below). Both match Triton.
        if const_expr(HAS_WEIGHT):
            coef = (Float32(1.0) - label_smoothing) * w_eff * inv_n_loss
        else:
            coef = inv_n_loss
        if const_expr(HAS_ZLOSS):
            coef = coef + (Float32(2.0) * lse_sq_scale * lse) * inv_n_z
        if is_ignored:
            coef = Float32(0.0)
        recip = coef / d
        # smoothing additive-term coefficient (eps * inv_n_loss), zeroed on ignored rows so an
        # ignored row's gradient stays entirely 0 — coef above only zeros the softmax part, but
        # the additive smoothing term must vanish too (Triton zeros ignored rows wholesale).
        if const_expr(HAS_SMOOTHING):
            eps_g = eps * inv_n_loss
            if is_ignored:
                eps_g = Float32(0.0)
        g_frag = cute.make_rmem_tensor((VEC,), gX.element_type)
        # P2's max `m` is loop-invariant, so hoist the FMA-fold scalar once:
        # exp2((x - m)*LOG2_E) == exp2(x*LOG2_E + neg_m2_p2). Per-element arg becomes one FFMA.
        neg_m2_p2 = m * NEG_LOG2_E
        # prologue (smem tile buffer free to reuse: the cross-warp barrier above
        # synced pass 1's drains before any pass-2 prefetch overwrites it)
        for s in cutlass.range_constexpr(_NUM_STAGES - 1):
            p_vidx = s * THREADS + tid
            if p_vidx < num_vec:
                cute.copy(cp_atom, gXv[None, p_vidx], sTilesV[None, tid, s])
            cute.arch.cp_async_commit_group()
        read_stage = Int32(0)
        write_stage = Int32(_NUM_STAGES - 1)
        r_vidx = Int32(tid)
        w_vidx = Int32((_NUM_STAGES - 1) * THREADS + tid)
        for _t in cutlass.range(0, num_tiles):
            if w_vidx < num_vec:
                cute.copy(cp_atom, gXv[None, w_vidx], sTilesV[None, tid, write_stage])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(_NUM_STAGES - 1)
            if r_vidx < num_vec:
                cute.autovec_copy(sTilesV[None, tid, read_stage], x_frag)  # smem -> reg
                x_ssa = x_frag.load().to(Float32)
                if const_expr(HAS_SOFTCAP):
                    t_ssa = cute.math.tanh(x_ssa / softcap)
                    x_ssa = softcap * t_ssa  # capped logits feed the softmax
                g_ssa = cute.math.exp2(x_ssa * LOG2_E + neg_m2_p2, fastmath=True) * recip
                if const_expr(HAS_SMOOTHING):
                    # additive smoothing term (not proportional to softmax), via the ignore-gated
                    # eps_g. Unweighted: -eps_g. Weighted: (softmax*weight_sum - weight_block)*eps_g.
                    if const_expr(NEED_WBLOCK):
                        cute.autovec_copy(gWv[None, r_vidx], w_frag)
                        w_ssa = w_frag.load()
                        softmax_ssa = cute.math.exp2(x_ssa * LOG2_E + neg_m2_p2, fastmath=True) / d
                        g_ssa = g_ssa + (softmax_ssa * weight_sum - w_ssa) * eps_g
                    else:
                        g_ssa = g_ssa + (Float32(0.0) - eps_g)
                if const_expr(HAS_SOFTCAP):
                    # chain rule applies to the FULL gradient incl. smoothing:
                    # d(softcap*tanh(x/softcap))/dx = 1 - tanh^2(x/softcap).
                    g_ssa = g_ssa * (1.0 - t_ssa * t_ssa)
                g_frag.store(g_ssa.to(gX.element_type))
                cute.autovec_copy(g_frag, gXv[None, r_vidx])  # 128-bit store to gmem
            read_stage = _advance(read_stage, _NUM_STAGES)
            write_stage = _advance(write_stage, _NUM_STAGES)
            r_vidx = r_vidx + THREADS
            w_vidx = w_vidx + THREADS
        cute.arch.cp_async_wait_group(0)  # drain remaining prefetches
        cute.arch.barrier()  # all grad writes visible before the target correction
        # -(1 - ls) * weight_y / N_loss correction at the (non-ignored) target index, one thread.
        do_correction = y != ignore_index
        if tid == 0:
            if do_correction:
                dxy = (Float32(0.0) - (Float32(1.0) - label_smoothing)) * w_eff * inv_n_loss
                if const_expr(HAS_SOFTCAP):
                    # same chain factor at y: t_y = tanh(x_y/softcap) = ori_xy_capped/softcap.
                    t_y = ori_xy / softcap
                    dxy = dxy * (1.0 - t_y * t_y)
                corr = gX[y].to(Float32) + dxy
                gX[y] = corr.to(gX.element_type)


# =============================================================================
# Host launch
# =============================================================================
@cute.jit
def _ce_fwd_host(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mLoss: cute.Tensor,
    mZLoss: cute.Tensor,
    mTokenAcc: cute.Tensor,
    mPredTok: cute.Tensor,
    mWeight: cute.Tensor,
    inv_n_loss: Float32,
    inv_n_z: Float32,
    lse_sq_scale: Float32,
    softcap: Float32,
    label_smoothing: Float32,
    weight_sum: Float32,
    ignore_index: Int32,
    HAS_GRAD: cutlass.Constexpr,
    HAS_ZLOSS: cutlass.Constexpr,
    RETURN_Z_LOSS: cutlass.Constexpr,
    HAS_SOFTCAP: cutlass.Constexpr,
    RETURN_TOKEN_ACCURACY: cutlass.Constexpr,
    RETURN_PREDICTED_TOKENS: cutlass.Constexpr,
    HAS_WEIGHT: cutlass.Constexpr,
    HAS_SMOOTHING: cutlass.Constexpr,
    NUM_WARPS: cutlass.Constexpr,
    stream: cuda.CUstream = None,
):
    BT = mX.shape[0]
    threads = 32 * NUM_WARPS
    # smem = the cp.async tile ring (threads * 16 bytes/thread * NUM_STAGES) + the 5 per-warp
    # reduction arrays (fp32), 16-byte rounded. Sized from NUM_WARPS so the 8-warp bf16 kernel
    # uses ~1/4 the smem of the 32-warp fp32 kernel and keeps more CTAs resident per SM.
    smem_bytes = threads * 16 * _NUM_STAGES + ((5 * NUM_WARPS * 4 + 15) // 16) * 16
    _ce_fwd_kernel(
        mX,
        mY,
        mLoss,
        mZLoss,
        mTokenAcc,
        mPredTok,
        mWeight,
        inv_n_loss,
        inv_n_z,
        lse_sq_scale,
        softcap,
        label_smoothing,
        weight_sum,
        ignore_index,
        HAS_GRAD,
        HAS_ZLOSS,
        RETURN_Z_LOSS,
        HAS_SOFTCAP,
        RETURN_TOKEN_ACCURACY,
        RETURN_PREDICTED_TOKENS,
        HAS_WEIGHT,
        HAS_SMOOTHING,
        NUM_WARPS,
    ).launch(
        grid=[BT, 1, 1],
        block=[threads, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


def _launch_ce_fwd(
    x,
    y,
    loss,
    inv_n_loss,
    ignore_index,
    has_grad,
    lse_sq_scale=0.0,
    z_loss_out=None,
    return_z_loss=False,
    softcap=None,
    label_smoothing=0.0,
    weight=None,
    weight_sum=0.0,
    return_token_accuracy=False,
    return_predicted_tokens=False,
    token_acc_out=None,
    pred_tok_out=None,
    inv_n_z=None,
):
    vec = 16 // x.element_size()  # 128-bit vectorization width: 8 bf16 / 4 fp32
    assert x.shape[-1] % vec == 0, (
        f"cutedsl CE needs V % {vec} == 0 for {x.dtype} (128-bit vectorized loads); "
        f"got V={x.shape[-1]}. The 256-thread tail is predicated, so only V % VEC is required."
    )
    # inv_n_z defaults to inv_n_loss: on the core / no-class-weight path the main loss and the
    # z_loss share one normalizer. Keeping inv_n_z a trailing keyword (not a 5th positional)
    # leaves the 6-arg call other cutedsl ops use unchanged: (x, y, loss, inv_n, ignore_index,
    # has_grad) — so this stays a drop-in for FLCE and any other caller.
    if inv_n_z is None:
        inv_n_z = inv_n_loss
    # Compile ONCE per (dtype, has_grad, feature flags) and cache the compiled callable;
    # invoke it each call. (Eager-calling the @cute.jit fn recompiled on every invocation
    # -> ~30 ms fixed overhead.) Launch on torch's current CUDA stream so the in-place
    # grad write is ordered w.r.t. the caller.
    has_zloss = bool(lse_sq_scale != 0.0 or return_z_loss)
    has_softcap = softcap is not None
    has_weight = weight is not None
    has_smoothing = bool(label_smoothing != 0.0)
    softcap_val = float(softcap) if has_softcap else 0.0
    x_ct = to_cute_tensor(x)
    y_ct = to_cute_tensor(y, assumed_align=8)  # int64
    loss_ct = to_cute_tensor(loss, assumed_align=2)  # bf16/fp16/fp32 scalar
    stream = _cute_stream()
    # Key on EVERY dtype the kernel bakes at compile time, not just x.dtype:
    #   mX.element_type (x), mY.element_type (y), mLoss.element_type (loss, via
    #   `loss.to(mLoss.element_type)`). Missing loss.dtype let two callers with the
    #   same (x.dtype, has_grad) but different loss-buffer widths reuse each other's
    #   kernel and write wrong-width values into the loss buffer — e.g. CE (loss =
    #   input dtype) vs FLCE (loss = fp32) collided on bf16.
    # When an optional output isn't requested, pass a same-shape dummy
    # of any dtype (mZLoss reuses `loss`; mTokenAcc reuses `loss`; mPredTok reuses the
    # int64 target `y`) — the kernel never touches it because its RETURN_* flag bakes
    # False, and the compile key carries that flag so a real-output compile can't reuse it.
    # Reuse the already-marshalled loss_ct/y_ct handles for the dummies (no extra from_dlpack
    # on the common path — one fewer DLPack capsule per call).
    z_ct = to_cute_tensor(z_loss_out, assumed_align=2) if return_z_loss else loss_ct
    ta_ct = to_cute_tensor(token_acc_out, assumed_align=4) if return_token_accuracy else loss_ct
    pt_ct = to_cute_tensor(pred_tok_out, assumed_align=8) if return_predicted_tokens else y_ct
    # weight is a fp32 (V,) vector when present (caller upcasts); dummy reuses int64 `y`.
    w_ct = to_cute_tensor(weight, assumed_align=4) if has_weight else y_ct
    # warps/CTA: mirror the Triton CE convention exactly (arch- and dtype-dependent):
    #   Blackwell (B200, sm_100+) bf16/fp16 -> 8 (instruction-issue-bound); fp32 -> 32
    #   Hopper (H100, sm_90) and earlier    -> 32 for all dtypes (bandwidth-bound)
    #   AMD (ROCm)                          -> 16
    # On Hopper the 8-warp bf16 kernel underfills the SMs and loses to the 32-warp Triton
    # forward, so we gate the 8-warp choice on Blackwell only (matches ops/cross_entropy.py).
    # Baked into the kernel, so it's part of the compile key.
    if is_hip():
        num_warps = 16
    else:
        is_blackwell = infer_device_arch().startswith("blackwell")
        num_warps = 8 if (x.element_size() == 2 and is_blackwell) else 32
    key = (
        x.dtype,
        y.dtype,
        loss.dtype,
        has_grad,
        has_zloss,
        bool(return_z_loss),
        has_softcap,
        bool(return_token_accuracy),
        bool(return_predicted_tokens),
        has_weight,
        has_smoothing,
        num_warps,
    )
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            _ce_fwd_host,
            x_ct,
            y_ct,
            loss_ct,
            z_ct,
            ta_ct,
            pt_ct,
            w_ct,
            float(inv_n_loss),
            float(inv_n_z),
            float(lse_sq_scale),
            float(softcap_val),
            float(label_smoothing),
            float(weight_sum),
            int(ignore_index),
            has_grad,
            has_zloss,
            bool(return_z_loss),
            has_softcap,
            bool(return_token_accuracy),
            bool(return_predicted_tokens),
            has_weight,
            has_smoothing,
            num_warps,
            stream,
        )
    # The constexpr flags are baked at compile; pass runtime tensors/scalars/stream only.
    _compile_cache[key](
        x_ct,
        y_ct,
        loss_ct,
        z_ct,
        ta_ct,
        pt_ct,
        w_ct,
        float(inv_n_loss),
        float(inv_n_z),
        float(lse_sq_scale),
        float(softcap_val),
        float(label_smoothing),
        float(weight_sum),
        int(ignore_index),
        stream,
    )


# =============================================================================
# Public host API (matches liger_kernel.ops.cross_entropy)
# =============================================================================
def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    """CuTe DSL CE forward. Returns (loss, z_loss, token_accuracy, predicted_tokens, _input)."""
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"

    BT, V = _input.shape
    _vec = 16 // _input.element_size()  # 128-bit vectorization width (8 bf16 / 4 fp32)
    assert V % _vec == 0, f"cutedsl CE needs V % {_vec} == 0 for {_input.dtype} (128-bit vectorized loads); got V={V}."
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    target_mask = target != ignore_index
    # Batch the count + bounds checks into ONE D2H sync (was three: sum().item(), max(), min()).
    # Each .item() is a ~15-20 us host stall independent of BT/V, so collapsing 3->1 cuts a
    # constant per-call cost that hurts small shapes most — fully general, no shape branching.
    _mt = target * target_mask
    _stats = torch.stack((target_mask.sum(), _mt.max(), _mt.min())).tolist()
    n_non_ignore = int(_stats[0])
    assert _stats[1] < V, f"Target out of bounds. Expected < {V}"
    assert _stats[2] >= 0, "Target out of bounds. Expected >= 0"

    # Class weight: sum_non_ignore_weight (the mean denominator for the weighted loss/grad)
    # replaces n_non_ignore; weight_sum (the full-vector sum) is used by weighted smoothing.
    # The kernel reads weight as fp32, so upcast here (exact parity for fp32 weights).
    sum_non_ignore_weight = float(n_non_ignore)
    weight_sum = 0.0
    if weight is not None:
        assert weight.shape[0] == V, f"weight must be a Tensor of size V={V}. Got: {tuple(weight.shape)}"
        assert torch.is_floating_point(weight), f"weight must be floating point. Got: {weight.dtype}"
        weight = weight.to(torch.float32)
        if weight.stride(-1) != 1:
            weight = weight.contiguous()
        sum_non_ignore_weight = torch.gather(weight, 0, target.masked_select(target_mask)).sum().item()
        weight_sum = weight.sum().item()

    loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device)
    # z_loss buffer: input dtype, zero-init so ignored rows stay 0 (matches Triton).
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    # token_accuracy is fp32 (1.0/0.0 per row); predicted_tokens is int64 (argmax column,
    # -1 for ignored rows). Zero/-1 init so ignored rows are correct even though the kernel
    # also writes them. Matches Triton's buffer dtypes/fills.
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    predicted_tokens_1d = (
        torch.full((BT,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None
    )

    # Normalizers (mean -> 1/N applied per-row in-kernel; sum/none -> 1.0; 1.0 when all
    # rows are ignored). The main loss/grad use sum_non_ignore_weight when weighted, else
    # n_non_ignore; z_loss is never weight-scaled so it always uses n_non_ignore.
    if reduction == "mean" and n_non_ignore > 0:
        if weight is not None and sum_non_ignore_weight > 0:
            inv_n_loss = 1.0 / sum_non_ignore_weight
        else:
            inv_n_loss = 1.0 / n_non_ignore
        inv_n_z = 1.0 / n_non_ignore
    else:
        inv_n_loss = 1.0
        inv_n_z = 1.0
    has_grad = bool(_input.requires_grad)

    _launch_ce_fwd(
        _input,
        target,
        loss_1d,
        inv_n_loss,
        ignore_index,
        has_grad,
        lse_square_scale,
        z_loss_1d,
        return_z_loss,
        softcap,
        label_smoothing=label_smoothing,
        weight=weight,
        weight_sum=weight_sum,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        token_acc_out=token_accuracy_1d,
        pred_tok_out=predicted_tokens_1d,
        inv_n_z=inv_n_z,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # accuracy reduces to the mean over non-ignored tokens (matches Triton).
        token_accuracy = torch.sum(token_accuracy_1d) / n_non_ignore if return_token_accuracy else None
    # predicted_tokens is always the per-row vector, regardless of reduction (matches Triton).
    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None
    return loss, z_loss, token_accuracy, predicted_tokens, _input


def cross_entropy_backward(_input, grad_output):
    """Scale the saved in-place gradient by grad_output (chain rule from upstream)."""
    # CE is usually the last layer -> grad_output == 1.0; skip the mul.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return _input
    # reduction="none": per-row upstream grad.
    if grad_output.ndim > 0:
        return _input * grad_output.unsqueeze(dim=1)
    # reduction in {mean, sum}: scalar upstream grad. Scale the saved gradient IN PLACE so we
    # never materialize a second BT×V buffer (Triton-parity peak memory: 1x logits, not 2x).
    # A raw Triton element-wise kernel is used instead of `_input *= grad_output` because an
    # in-place torch mul on the tensor returned from forward bumps its autograd version counter
    # and trips backward-through-backward anomalies; the raw kernel writes through the pointer
    # without that bookkeeping — exactly how the Triton CE backward dodges the same issue.
    BT, V = _input.shape
    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))
    element_mul_kernel[(BT,)](
        _input,
        _input.stride(-2),
        grad_output,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return _input


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    CuTe DSL autograd wrapper for Cross Entropy loss.

    Signature-compatible with ``liger_kernel.ops.cross_entropy.LigerCrossEntropyFunction``.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        input_requires_grad = _input.requires_grad

        loss, z_loss, token_accuracy, predicted_tokens, _input = cross_entropy_forward(
            _input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            return_token_accuracy,
            return_predicted_tokens,
        )
        if input_requires_grad:
            ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics

        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return (
            _input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
