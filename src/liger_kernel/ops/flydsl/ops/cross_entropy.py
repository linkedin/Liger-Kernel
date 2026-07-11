"""
FlyDSL fused cross-entropy (online logsumexp + optional in-place gradient).

Signature-compatible with ``liger_kernel.ops.cross_entropy``. Initial support
covers the common training path: mean/sum/none reduction, ignore_index,
label_smoothing, softcap, z_loss, and in-place gradients. Class weights and
token-accuracy / predicted-token extras are not implemented yet.

Mirrors the Triton reference (``liger_kernel.ops.cross_entropy``):

* **Online softmax** (Algorithm 3 of https://arxiv.org/pdf/1805.02867) fuses the
  row-max and the sum-of-exp into a single pass, so the row is read twice
  (once to build the LSE, once to write the gradient) rather than three times.
* **Normalizers live on the device.** ``mNorm`` is a 2-element fp32 tensor
  holding ``(inv_n_loss, inv_n_z)``; the kernel reads it and folds the mean
  normalization into the loss and the gradient. This keeps the host free of any
  ``.item()`` / D2H sync *and* avoids rescaling the whole (BT, V) tensor
  afterwards. It also keeps the stored per-row loss at ~L/N instead of ~L, which
  is what stops ``sum(loss_1d)`` from overflowing fp16 at large BT.
* The bulk of the row is always read with 128-bit vector buffer copies; only the
  final ``V % tile_cols`` columns take the masked scalar path.
"""

import math

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from flydsl.expr import arith
from flydsl.expr import const_expr
from flydsl.expr import gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.vector import ReductionOp
from flydsl.expr.vector import full as vec_full

from liger_kernel.ops.flydsl.ops.utils import dtype_to_flydsl_str
from liger_kernel.ops.flydsl.ops.utils import warp_size as _host_warp_size

LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG_INF_F32 = -1.0e38

BLOCK_THREADS = 256  # default / lower bound; the real block is picked per (V, dtype)
MAX_BLOCK_THREADS = 1024

# fp32 registers per thread we are willing to spend holding the row. Staying under
# this lets the gradient pass reuse the logits from pass 1 instead of re-reading
# them from HBM (the trick FlyDSL's own softmax_kernel.py uses).
REG_BUDGET = 64

# Compiled launchers keyed by (V, dtype_str, warp_size, feature flags).
_compile_cache: dict[tuple, object] = {}


def _elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def _pick_block(n_vec: int, vec_width: int) -> int:
    """Block size that keeps the per-thread row slice inside REG_BUDGET when possible."""
    max_tiles = max(1, REG_BUDGET // vec_width)
    target = max(1, -(-n_vec // max_tiles))
    blk = 1 << (target - 1).bit_length()  # next power of two
    return min(MAX_BLOCK_THREADS, max(BLOCK_THREADS, blk))


def _elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(dtype_str)


def _build_ce_launcher(
    V: int,
    dtype_str: str,
    warp_size: int,
    HAS_GRAD: bool,
    HAS_ZLOSS: bool,
    RETURN_Z_LOSS: bool,
    HAS_SOFTCAP: bool,
    HAS_SMOOTHING: bool,
    WRITE_LSE: bool = False,
):
    """Build a specialized FlyDSL CE launcher for fixed V / dtype / flags."""
    elem_bits = _elem_bits(dtype_str)
    elem_dtype = _elem_type(dtype_str)

    # Index the row in units of 128-bit vectors. Every real vocab (32000, 32768,
    # 128256, 152064, ...) is a multiple of the vector width, so the whole row is
    # vectorized and at most ONE trailing vector iteration needs masking. Gating on
    # `V % (BLOCK*vec) == 0` instead would push every production shape onto a
    # scalar, 1-element-per-lane path.
    vec_width = 128 // elem_bits
    VECTORIZED = V % vec_width == 0
    if not VECTORIZED:  # rare; fall back to scalar element indexing
        vec_width = 1
    n_vec = V // vec_width

    block_threads = _pick_block(n_vec, vec_width)
    RED_SLOTS = max(1, (block_threads + warp_size - 1) // warp_size)

    num_full = n_vec // block_threads  # unmasked iterations
    tail_vecs = n_vec % block_threads  # 0 or 1 masked iteration
    HAS_TAIL = tail_vecs > 0
    n_iters = num_full + (1 if HAS_TAIL else 0)

    # Hold the row in registers across the two passes when it fits, so the gradient
    # pass does not re-read the logits from HBM (2 loads + 1 store -> 1 load + 1
    # store). Same technique as FlyDSL's kernels/norm/softmax_kernel.py.
    REG_BUFFER = HAS_GRAD and (n_iters * vec_width <= REG_BUDGET)

    fm_fast = arith.FastMathFlags.fast

    @fx.struct
    class SharedStorage:
        # Separate buffers so consecutive block_reduces (max -> sum -> smooth)
        # cannot race on the same shared memory slot.
        s_red_max: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_red_sum: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_red_smooth: fx.Array[fx.Float32, RED_SLOTS, 16]

    # known_block_size is required above the AMDGPU default max_flat_workgroup_size
    # of 256, and lets the backend size the register budget for the real block.
    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def ce_kernel(
        mX: fx.Tensor,  # (BT, V) logits; grad written in-place when HAS_GRAD
        mY: fx.Tensor,  # (BT,) int32 targets
        mLoss: fx.Tensor,  # (BT,) fp32 per-row loss
        mZLoss: fx.Tensor,  # (BT,) fp32 per-row z_loss; written only if RETURN_Z_LOSS
        mNorm: fx.Tensor,  # (2,) fp32: [inv_n_loss, inv_n_z] -- device-side, no D2H sync
        mLse: fx.Tensor,  # (BT,) fp32 per-row logsumexp; lets backward rebuild softmax
        lse_sq_scale: fx.Float32,
        softcap: fx.Float32,
        label_smoothing: fx.Float32,
        ignore_index: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red_max = lds.s_red_max.view(fx.make_layout(RED_SLOTS, 1))
        s_red_sum = lds.s_red_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_red_smooth = lds.s_red_smooth.view(fx.make_layout(RED_SLOTS, 1))

        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        c_neg_inf = fx.Float32(NEG_INF_F32)
        c_log2e = fx.Float32(LOG2_E)
        c_ln2 = fx.Float32(LN2)

        def softcap_act(x):
            """softcap * tanh(x/softcap) via exp2 (math.tanh lacks a libcall on some AMD targets)."""
            z = x / softcap
            # tanh(z) = 2/(1+exp(-2z)) - 1; exp(-2z) = exp2(-2z * log2(e))
            e = fmath.exp2(z * fx.Float32(-2.0 * LOG2_E), fastmath=fm_fast)
            t = fx.Float32(2.0) / (c_one + e) - c_one
            return softcap * t, t

        def wave_reduce(x, mode):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(warp_size))):
                off = warp_size // (2 << _sh_exp)
                peer = w.shuffle_xor(off, warp_size)
                if const_expr(mode == "max"):
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce(val, mode, s_red_buffer):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce(val, mode)

            lane = tid % warp_size
            wave = tid // warp_size
            neutral = c_neg_inf if mode == "max" else c_zero

            w = wave_reduce(val, mode)
            if lane == 0:
                fx.memref_store(w, s_red_buffer, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red_buffer, lane_safe)
                ww = in_range.select(v, neutral)
                ww = wave_reduce(ww, mode)
                if lane == 0:
                    fx.memref_store(ww, s_red_buffer, 0)
            gpu.barrier()
            # No trailing barrier: each block_reduce owns a distinct LDS buffer, so
            # nothing can overwrite slot 0 before every thread has read it. Barriers
            # are expensive at 512-1024 threads and this kernel is latency-bound.
            return fx.memref_load(s_red_buffer, 0)

        # ------------------------------------------------------------------
        # Device-side normalizers: mNorm = [inv_n_loss, inv_n_z].
        # Loading them here (rather than taking host floats) is what lets the
        # caller keep n_non_ignore on the GPU without a D2H sync, while still
        # folding the mean into the loss/grad *inside* the kernel.
        # ------------------------------------------------------------------
        f32_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        norm_div = fx.logical_divide(mNorm, fx.make_layout(1, 1))
        n_rmem = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy_atom_call(f32_atom, fx.slice(norm_div, (None, 0)), n_rmem)
        inv_n_loss = fx.memref_load_vec(n_rmem)[0]
        fx.copy_atom_call(f32_atom, fx.slice(norm_div, (None, 1)), n_rmem)
        inv_n_z = fx.memref_load_vec(n_rmem)[0]

        # Target for this row (int32).
        y_div = fx.logical_divide(mY, fx.make_layout(1, 1))
        y_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        y_rmem = fx.make_rmem_tensor(1, fx.Int32)
        fx.copy_atom_call(y_atom, fx.slice(y_div, (None, bid)), y_rmem)
        y = fx.memref_load_vec(y_rmem)[0]
        is_ignored = y == ignore_index
        # Clamp into [0, V) for *addressing* only: make_buffer_tensor uses the
        # default num_records=0xFFFFFFFF, so hardware OOB clamping is off and an
        # out-of-range label would otherwise corrupt global memory.
        y_safe = is_ignored.select(fx.Int32(0), y)
        y_safe = (y_safe >= fx.Int32(0)).select(y_safe, fx.Int32(0))
        y_safe = (y_safe < fx.Int32(V)).select(y_safe, fx.Int32(0))

        X_buf = fx.rocdl.make_buffer_tensor(mX)
        row_x = fx.slice(X_buf, (bid, None))

        # Scalar accessor (target logit + ragged tail).
        x_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        x_div_s = fx.logical_divide(row_x, fx.make_layout(1, 1))

        # Vector accessor (the whole row, one 128-bit buffer copy per lane).
        a_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
        copy_atom = (
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if VECTORIZED
            else x_atom_s  # V not a multiple of the vector width: scalar fallback
        )

        def _load_vec(idx):
            r = fx.make_rmem_tensor(vec_width, elem_dtype)
            fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx)), r)
            return fx.memref_load_vec(r)

        def _load_scalar(idx):
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy_atom_call(x_atom_s, fx.slice(x_div_s, (None, idx)), r)
            return fx.memref_load_vec(r)[0]

        # Load target logit (scalar).
        ori_xy_e = _load_scalar(y_safe)
        ori_xy = ori_xy_e if dtype_str == "f32" else ori_xy_e.to(fx.Float32)
        if const_expr(HAS_SOFTCAP):
            ori_xy, _ = softcap_act(ori_xy)

        eps = fx.Float32(0.0)
        if const_expr(HAS_SMOOTHING):
            eps = label_smoothing / fx.Float32(float(V))

        # ------------------------------------------------------------------
        # Pass 1: online softmax -- running (m, d) in a single sweep over V.
        #   m_new = max(m, x); d = d * exp2((m - m_new) * log2e) + sum(exp2(x - m_new))
        # Matches liger_kernel/ops/cross_entropy.py:176-178.
        #
        # A thread's whole vector is either in range or out (masking is at vector
        # granularity), so out-of-range lanes can be neutralized on the *reduced*
        # scalars rather than on the vector itself.
        # ------------------------------------------------------------------
        m = c_neg_inf
        d = c_zero
        t_sxs = c_zero
        row_buf = []  # register-buffered fp32 row (post-softcap), when REG_BUFFER

        def _masked_idx(i):
            # `i` is a Python int (range_constexpr unrolls), so `masked` is a
            # trace-time constant: only the last iteration carries a mask. It must
            # go through const_expr -- a plain `if` is rewritten into an scf.if
            # region, and values first bound inside it would not escape.
            idx = tid + i * block_threads
            is_valid = idx < n_vec
            idx_safe = idx
            if const_expr(i >= num_full):
                idx_safe = is_valid.select(idx, 0)
            return idx, idx_safe, is_valid

        def _read_row(i):
            _, idx_safe, _ = _masked_idx(i)
            vec = _load_vec(idx_safe)
            x = vec.to(fx.Float32) if dtype_str != "f32" else vec
            if const_expr(HAS_SOFTCAP):
                x, _ = softcap_act(x)
            return x

        # The online-softmax update carries (m, d) across iterations, so fusing the
        # load into it serializes load -> compute -> load. This kernel is
        # latency-bound (register-buffering cut HBM traffic 33% for no speedup), so
        # when the row fits in registers we issue every buffer_load first -- they
        # are mutually independent and all go in flight at once -- and only then run
        # the dependent ALU chain. Cf. the tuning guide's software-prefetch section.
        if const_expr(REG_BUFFER):
            for i in range_constexpr(n_iters):
                row_buf.append(_read_row(i))

        for i in range_constexpr(n_iters):
            _, _, is_valid = _masked_idx(i)
            x = row_buf[i] if REG_BUFFER else _read_row(i)

            tile_max = x.reduce(ReductionOp.MAX)
            p = fmath.exp2(x * c_log2e + m.maximumf(tile_max) * fx.Float32(-LOG2_E), fastmath=fm_fast)
            p_sum = p.reduce(ReductionOp.ADD, fastmath=fm_fast)
            tile_sxs = c_zero
            if const_expr(HAS_SMOOTHING):
                tile_sxs = (x * (c_zero - eps)).reduce(ReductionOp.ADD, fastmath=fm_fast)

            if const_expr(i >= num_full):
                # -1e38 is a *finite* sentinel: m - m_new == 0 -> rescale == 1.
                tile_max = is_valid.select(tile_max, c_neg_inf)
                p_sum = is_valid.select(p_sum, c_zero)
                if const_expr(HAS_SMOOTHING):
                    tile_sxs = is_valid.select(tile_sxs, c_zero)

            m_new = m.maximumf(tile_max)
            rescale = fmath.exp2((m - m_new) * c_log2e, fastmath=fm_fast)
            d = d * rescale + p_sum
            m = m_new
            if const_expr(HAS_SMOOTHING):
                t_sxs = t_sxs + tile_sxs

        # Two-level online softmax: reduce m across the block, then rescale each
        # thread's partial d into the block-wide max before summing.
        m_thread = m
        m = block_reduce(m_thread, "max", s_red_max)
        d = d * fmath.exp2((m_thread - m) * c_log2e, fastmath=fm_fast)
        d = block_reduce(d, "sum", s_red_sum)
        if const_expr(HAS_SMOOTHING):
            t_sxs = block_reduce(t_sxs, "sum", s_red_smooth)

        # lse = m + ln(d) = m + log2(d) * ln2
        lse = m + fmath.log2(d, fastmath=fm_fast) * c_ln2

        main = lse - ori_xy
        if const_expr(HAS_SMOOTHING):
            smooth_loss = t_sxs + label_smoothing * lse
            main = main * (c_one - label_smoothing) + smooth_loss
        loss = main * inv_n_loss

        zl = c_zero
        if const_expr(HAS_ZLOSS):
            zl = lse_sq_scale * lse * lse * inv_n_z
            loss = loss + zl
        if is_ignored:
            loss = c_zero
            zl = c_zero

        # Store per-row loss (and optional z_loss) from thread 0, always fp32.
        if tid == 0:
            loss_div = fx.logical_divide(mLoss, fx.make_layout(1, 1))
            loss_r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store_vec(vec_full(1, loss, fx.Float32), loss_r)
            fx.copy_atom_call(f32_atom, loss_r, fx.slice(loss_div, (None, bid)))

            # lse is all backward needs to rebuild softmax(x) = exp2((x - lse)*log2e),
            # so the gradient can be produced later with grad_output folded in --
            # no unscaled gradient, hence no extra (BT, V) rescale pass.
            if const_expr(WRITE_LSE):
                lse_div = fx.logical_divide(mLse, fx.make_layout(1, 1))
                lse_r = fx.make_rmem_tensor(1, fx.Float32)
                fx.memref_store_vec(vec_full(1, lse, fx.Float32), lse_r)
                fx.copy_atom_call(f32_atom, lse_r, fx.slice(lse_div, (None, bid)))

            if const_expr(RETURN_Z_LOSS):
                z_div = fx.logical_divide(mZLoss, fx.make_layout(1, 1))
                z_r = fx.make_rmem_tensor(1, fx.Float32)
                fx.memref_store_vec(vec_full(1, zl, fx.Float32), z_r)
                fx.copy_atom_call(f32_atom, z_r, fx.slice(z_div, (None, bid)))

        # ------------------------------------------------------------------
        # Pass 2: in-place gradient (already normalized by inv_n_loss/inv_n_z).
        # ------------------------------------------------------------------
        if const_expr(HAS_GRAD):
            coef = inv_n_loss
            if const_expr(HAS_ZLOSS):
                coef = coef + (fx.Float32(2.0) * lse_sq_scale * lse) * inv_n_z
            if is_ignored:
                coef = c_zero
            recip = coef / d
            eps_g = c_zero
            if const_expr(HAS_SMOOTHING):
                eps_g = eps * inv_n_loss
                if is_ignored:
                    eps_g = c_zero
            neg_m2 = m * fx.Float32(-LOG2_E)

            def _grad_of(x, t_ssa):
                g = fmath.exp2(x * c_log2e + neg_m2, fastmath=fm_fast) * recip
                if const_expr(HAS_SMOOTHING):
                    g = g + (c_zero - eps_g)
                if const_expr(HAS_SOFTCAP):
                    g = g * (c_one - t_ssa * t_ssa)
                return g

            for i in range_constexpr(n_iters):
                masked = i >= num_full
                idx = tid + i * block_threads
                is_valid = idx < n_vec
                idx_safe = idx
                if const_expr(masked):
                    idx_safe = is_valid.select(idx, 0)

                r = fx.make_rmem_tensor(vec_width, elem_dtype)
                if const_expr(REG_BUFFER):
                    # Row still live in registers from pass 1 -- no HBM re-read.
                    x = row_buf[i]
                else:
                    fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx_safe)), r)
                    vec = fx.memref_load_vec(r)
                    x = vec.to(fx.Float32) if dtype_str != "f32" else vec
                    if const_expr(HAS_SOFTCAP):
                        x, _ = softcap_act(x)

                t_ssa = None
                if const_expr(HAS_SOFTCAP):
                    # x is already soft-capped, so tanh(x/softcap) == x/softcap.
                    t_ssa = x / softcap
                g = _grad_of(x, t_ssa)
                out_e = g if dtype_str == "f32" else g.to(elem_dtype)
                fx.memref_store_vec(out_e, r)
                if const_expr(masked):
                    # Runtime guard: out-of-range lanes must not write.
                    if is_valid:
                        fx.copy_atom_call(copy_atom, r, fx.slice(a_div, (None, idx_safe)))
                else:
                    fx.copy_atom_call(copy_atom, r, fx.slice(a_div, (None, idx_safe)))

            gpu.barrier()
            # Target correction: -(1 - ls) / N at column y (non-ignored).
            if tid == 0:
                if y != ignore_index:
                    dxy = (c_zero - (c_one - label_smoothing)) * inv_n_loss
                    view = fx.slice(x_div_s, (None, y_safe))
                    r = fx.make_rmem_tensor(1, elem_dtype)
                    fx.copy_atom_call(x_atom_s, view, r)
                    cur_e = fx.memref_load_vec(r)[0]
                    cur = cur_e if dtype_str == "f32" else cur_e.to(fx.Float32)
                    if const_expr(HAS_SOFTCAP):
                        # Chain rule at y: ori_xy is already soft-capped, so
                        # tanh(x/softcap) = ori_xy / softcap.
                        t_y = ori_xy / softcap
                        dxy = dxy * (c_one - t_y * t_y)
                    corr = cur + dxy
                    out_e = corr if dtype_str == "f32" else corr.to(elem_dtype)
                    fx.memref_store_vec(vec_full(1, out_e, elem_dtype), r)
                    fx.copy_atom_call(x_atom_s, r, view)

    @flyc.jit
    def launch_ce(
        mX: fx.Tensor,
        mY: fx.Tensor,
        mLoss: fx.Tensor,
        mZLoss: fx.Tensor,
        mNorm: fx.Tensor,
        mLse: fx.Tensor,
        lse_sq_scale: fx.Float32,
        softcap: fx.Float32,
        label_smoothing: fx.Float32,
        ignore_index: fx.Int32,
        bt: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ce_kernel(
            mX,
            mY,
            mLoss,
            mZLoss,
            mNorm,
            mLse,
            lse_sq_scale,
            softcap,
            label_smoothing,
            ignore_index,
        ).launch(
            grid=(bt, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_ce


def _build_ce_grad_launcher(
    V: int,
    dtype_str: str,
    warp_size: int,
    HAS_ZLOSS: bool,
    HAS_SOFTCAP: bool,
    HAS_SMOOTHING: bool,
    GO_PER_ROW: bool,
):
    """Gradient-only kernel: rebuilds softmax from the saved per-row ``lse``.

    Run from ``backward``, so the upstream ``grad_output`` is known and folds
    directly into the coefficient. That means we never materialize an *unscaled*
    gradient, and therefore never pay the extra (BT, V) rescale pass that the
    Triton path dodges only by taking a ``torch.equal`` D2H sync.

    Traffic: 1 read + 1 write of the logits. Combined with the forward's single
    read, the op moves 2 reads + 1 write in total -- the same as Triton's online
    softmax, with no host sync anywhere.
    """
    elem_bits = _elem_bits(dtype_str)
    elem_dtype = _elem_type(dtype_str)

    vec_width = 128 // elem_bits
    VECTORIZED = V % vec_width == 0
    if not VECTORIZED:
        vec_width = 1
    n_vec = V // vec_width

    block_threads = _pick_block(n_vec, vec_width)
    num_full = n_vec // block_threads
    tail_vecs = n_vec % block_threads
    n_iters = num_full + (1 if tail_vecs > 0 else 0)

    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def ce_grad_kernel(
        mX: fx.Tensor,  # (BT, V) logits in; gradients written in-place
        mY: fx.Tensor,  # (BT,) int32 targets
        mLse: fx.Tensor,  # (BT,) fp32 logsumexp from forward
        mNorm: fx.Tensor,  # (2,) fp32 [inv_n_loss, inv_n_z]
        mGo: fx.Tensor,  # (1,) or (BT,) fp32 upstream grad
        lse_sq_scale: fx.Float32,
        softcap: fx.Float32,
        label_smoothing: fx.Float32,
        ignore_index: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        c_log2e = fx.Float32(LOG2_E)

        def softcap_act(x):
            z = x / softcap
            e = fmath.exp2(z * fx.Float32(-2.0 * LOG2_E), fastmath=fm_fast)
            t = fx.Float32(2.0) / (c_one + e) - c_one
            return softcap * t, t

        f32_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        def _load_f32(tensor, idx):
            div = fx.logical_divide(tensor, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(f32_atom, fx.slice(div, (None, idx)), r)
            return fx.memref_load_vec(r)[0]

        inv_n_loss = _load_f32(mNorm, 0)
        inv_n_z = _load_f32(mNorm, 1)
        lse = _load_f32(mLse, bid)
        go = _load_f32(mGo, bid) if GO_PER_ROW else _load_f32(mGo, 0)

        y_div = fx.logical_divide(mY, fx.make_layout(1, 1))
        y_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        y_rmem = fx.make_rmem_tensor(1, fx.Int32)
        fx.copy_atom_call(y_atom, fx.slice(y_div, (None, bid)), y_rmem)
        y = fx.memref_load_vec(y_rmem)[0]
        is_ignored = y == ignore_index
        y_safe = is_ignored.select(fx.Int32(0), y)
        y_safe = (y_safe >= fx.Int32(0)).select(y_safe, fx.Int32(0))
        y_safe = (y_safe < fx.Int32(V)).select(y_safe, fx.Int32(0))

        X_buf = fx.rocdl.make_buffer_tensor(mX)
        row_x = fx.slice(X_buf, (bid, None))
        x_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        x_div_s = fx.logical_divide(row_x, fx.make_layout(1, 1))
        a_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits) if VECTORIZED else x_atom_s

        eps = fx.Float32(0.0)
        if const_expr(HAS_SMOOTHING):
            eps = label_smoothing / fx.Float32(float(V))

        # Capture the soft-capped logit at column y BEFORE the loop overwrites it
        # with the gradient; the target correction needs its tanh derivative.
        # EVERY thread reads X[y] here, while the loop below has some *other* thread
        # store the gradient to that same address -- a cross-thread RAW on global
        # memory. The barrier orders them (the AMDGPU backend emits s_waitcnt
        # vmcnt(0) before s_barrier, so the loads land first). Without it this races,
        # and only at high block counts.
        ty_at_y = c_zero
        if const_expr(HAS_SOFTCAP):
            r_y = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy_atom_call(x_atom_s, fx.slice(x_div_s, (None, y_safe)), r_y)
            xy_e = fx.memref_load_vec(r_y)[0]
            xy = xy_e if dtype_str == "f32" else xy_e.to(fx.Float32)
            _, ty_at_y = softcap_act(xy)
            gpu.barrier()

        # grad_output folds into the coefficients here -- the whole point.
        coef = inv_n_loss * go
        if const_expr(HAS_ZLOSS):
            coef = coef + (fx.Float32(2.0) * lse_sq_scale * lse) * inv_n_z * go
        if is_ignored:
            coef = c_zero
        eps_g = c_zero
        if const_expr(HAS_SMOOTHING):
            eps_g = eps * inv_n_loss * go
            if is_ignored:
                eps_g = c_zero
        neg_lse2 = lse * fx.Float32(-LOG2_E)

        for i in range_constexpr(n_iters):
            idx = tid + i * block_threads
            is_valid = idx < n_vec
            idx_safe = idx
            if const_expr(i >= num_full):
                idx_safe = is_valid.select(idx, 0)

            r = fx.make_rmem_tensor(vec_width, elem_dtype)
            fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx_safe)), r)
            vec = fx.memref_load_vec(r)
            x = vec.to(fx.Float32) if dtype_str != "f32" else vec
            t_ssa = None
            if const_expr(HAS_SOFTCAP):
                x, t_ssa = softcap_act(x)

            # softmax(x) = exp(x - lse) = exp2((x - lse) * log2e)
            g = fmath.exp2(x * c_log2e + neg_lse2, fastmath=fm_fast) * coef
            if const_expr(HAS_SMOOTHING):
                g = g + (c_zero - eps_g)
            if const_expr(HAS_SOFTCAP):
                g = g * (c_one - t_ssa * t_ssa)

            out_e = g if dtype_str == "f32" else g.to(elem_dtype)
            fx.memref_store_vec(out_e, r)
            if const_expr(i >= num_full):
                if is_valid:
                    fx.copy_atom_call(copy_atom, r, fx.slice(a_div, (None, idx_safe)))
            else:
                fx.copy_atom_call(copy_atom, r, fx.slice(a_div, (None, idx_safe)))

        gpu.barrier()
        # Target correction: -(1 - ls) * inv_n * grad_output at column y.
        if tid == 0:
            if y != ignore_index:
                dxy = (c_zero - (c_one - label_smoothing)) * inv_n_loss * go
                view = fx.slice(x_div_s, (None, y_safe))
                r = fx.make_rmem_tensor(1, elem_dtype)
                fx.copy_atom_call(x_atom_s, view, r)
                cur_e = fx.memref_load_vec(r)[0]
                cur = cur_e if dtype_str == "f32" else cur_e.to(fx.Float32)
                if const_expr(HAS_SOFTCAP):
                    dxy = dxy * (c_one - ty_at_y * ty_at_y)
                corr = cur + dxy
                out_e = corr if dtype_str == "f32" else corr.to(elem_dtype)
                fx.memref_store_vec(vec_full(1, out_e, elem_dtype), r)
                fx.copy_atom_call(x_atom_s, r, view)

    @flyc.jit
    def launch_ce_grad(
        mX: fx.Tensor,
        mY: fx.Tensor,
        mLse: fx.Tensor,
        mNorm: fx.Tensor,
        mGo: fx.Tensor,
        lse_sq_scale: fx.Float32,
        softcap: fx.Float32,
        label_smoothing: fx.Float32,
        ignore_index: fx.Int32,
        bt: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ce_grad_kernel(
            mX,
            mY,
            mLse,
            mNorm,
            mGo,
            lse_sq_scale,
            softcap,
            label_smoothing,
            ignore_index,
        ).launch(
            grid=(bt, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_ce_grad


def _get_launcher(
    V: int,
    dtype: torch.dtype,
    has_grad: bool,
    has_zloss: bool,
    return_z_loss: bool,
    has_softcap: bool,
    has_smoothing: bool,
    write_lse: bool = False,
    device: Optional[torch.device] = None,
):
    dtype_str = dtype_to_flydsl_str(dtype)
    ws = _host_warp_size(device)
    key = (V, dtype_str, ws, has_grad, has_zloss, return_z_loss, has_softcap, has_smoothing, write_lse)
    launcher = _compile_cache.get(key)
    if launcher is None:
        launcher = _build_ce_launcher(
            V,
            dtype_str,
            ws,
            has_grad,
            has_zloss,
            return_z_loss,
            has_softcap,
            has_smoothing,
            write_lse,
        )
        _compile_cache[key] = launcher
    return launcher


def make_norm(n_non_ignore: torch.Tensor, reduction: str, device: torch.device) -> torch.Tensor:
    """Build the device-side ``(2,)`` fp32 ``[inv_n_loss, inv_n_z]`` normalizer.

    ``n_non_ignore`` stays on the GPU: no ``.item()``, no D2H sync. For ``sum`` /
    ``none`` the normalizers are 1.0; for ``mean`` they are ``1 / n_non_ignore``
    (clamped so an all-ignored batch yields 0 rather than NaN).
    """
    if reduction == "mean":
        inv = torch.reciprocal(n_non_ignore.to(torch.float32).clamp(min=1.0))
        return inv.expand(2).contiguous()
    return torch.ones(2, dtype=torch.float32, device=device)


_grad_compile_cache: dict[tuple, object] = {}


def _get_grad_launcher(
    V: int,
    dtype: torch.dtype,
    has_zloss: bool,
    has_softcap: bool,
    has_smoothing: bool,
    go_per_row: bool,
    device: Optional[torch.device] = None,
):
    dtype_str = dtype_to_flydsl_str(dtype)
    ws = _host_warp_size(device)
    key = (V, dtype_str, ws, has_zloss, has_softcap, has_smoothing, go_per_row)
    launcher = _grad_compile_cache.get(key)
    if launcher is None:
        launcher = _build_ce_grad_launcher(V, dtype_str, ws, has_zloss, has_softcap, has_smoothing, go_per_row)
        _grad_compile_cache[key] = launcher
    return launcher


def launch_ce_grad_on_logits(
    logits: torch.Tensor,
    target_i32: torch.Tensor,
    lse: torch.Tensor,
    norm: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    ignore_index: int,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    label_smoothing: float = 0.0,
    return_z_loss: bool = False,
):
    """Write CE gradients in-place over ``logits``, with ``grad_output`` folded in."""
    n_rows, V = logits.shape
    go_per_row = grad_output.ndim > 0 and grad_output.numel() > 1
    go = grad_output.reshape(-1).to(torch.float32).contiguous()

    with torch.cuda.device(logits.device):
        stream = torch.cuda.current_stream()
        launcher = _get_grad_launcher(
            V,
            logits.dtype,
            bool(lse_square_scale != 0.0 or return_z_loss),
            softcap is not None,
            bool(label_smoothing != 0.0),
            bool(go_per_row),
            device=logits.device,
        )
        launcher(
            logits,
            target_i32,
            lse,
            norm,
            go,
            float(lse_square_scale),
            float(softcap) if softcap is not None else 0.0,
            float(label_smoothing),
            int(ignore_index),
            int(n_rows),
            stream=stream,
        )
    return logits


def launch_ce_on_logits(
    logits: torch.Tensor,
    target_i32: torch.Tensor,
    loss_1d: torch.Tensor,
    norm: torch.Tensor,
    *,
    ignore_index: int,
    has_grad: bool,
    lse_1d: Optional[torch.Tensor] = None,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    label_smoothing: float = 0.0,
    return_z_loss: bool = False,
    z_loss_1d: Optional[torch.Tensor] = None,
):
    """Launch FlyDSL CE on a (possibly chunked) logits tile.

    When ``has_grad`` is True, gradients are written in-place over ``logits``,
    already scaled by ``norm[0]`` / ``norm[1]`` -- no post-hoc rescale needed.

    ``norm`` is the ``(2,)`` fp32 device tensor from :func:`make_norm`. It must be
    computed over the *full* BT, not per chunk. ``loss_1d`` (and ``z_loss_1d``)
    must be fp32.
    """
    if logits.stride(-1) != 1:
        raise ValueError("launch_ce_on_logits requires contiguous logits along V")
    if target_i32.dtype != torch.int32:
        raise TypeError(f"target_i32 must be int32, got {target_i32.dtype}")
    if loss_1d.dtype != torch.float32:
        raise TypeError(f"loss_1d must be float32, got {loss_1d.dtype}")

    n_rows, V = logits.shape
    has_zloss = bool(lse_square_scale != 0.0 or return_z_loss)
    has_softcap = softcap is not None
    has_smoothing = bool(label_smoothing != 0.0)
    softcap_val = float(softcap) if has_softcap else 0.0
    z_buf = z_loss_1d if return_z_loss else loss_1d
    if return_z_loss and z_loss_1d is None:
        raise ValueError("z_loss_1d is required when return_z_loss=True")
    lse_buf = lse_1d if lse_1d is not None else loss_1d

    # Bind compile + launch to the tensors' device (FlyDSL rmsnorm convention;
    # multi-GPU / stream correctness).
    with torch.cuda.device(logits.device):
        stream = torch.cuda.current_stream()
        launcher = _get_launcher(
            V,
            logits.dtype,
            bool(has_grad),
            has_zloss,
            bool(return_z_loss),
            has_softcap,
            has_smoothing,
            write_lse=lse_1d is not None,
            device=logits.device,
        )
        launcher(
            logits,
            target_i32,
            loss_1d,
            z_buf,
            norm,
            lse_buf,
            float(lse_square_scale),
            float(softcap_val),
            float(label_smoothing),
            int(ignore_index),
            int(n_rows),
            stream=stream,
        )


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
    """FlyDSL CE forward. Returns (loss, z_loss, token_accuracy, predicted_tokens, _input)."""
    assert reduction in ("mean", "sum", "none"), f"Unsupported reduction: {reduction}"

    if weight is not None:
        raise NotImplementedError("flydsl CE does not yet support class weights")
    if return_token_accuracy:
        raise NotImplementedError("flydsl CE does not yet support return_token_accuracy")
    if return_predicted_tokens:
        raise NotImplementedError("flydsl CE does not yet support return_predicted_tokens")

    BT = _input.shape[0]
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # The non-ignore count stays on-device -- no .item()/.tolist() D2H sync. The
    # reciprocal is handed to the kernel as a tensor, so the mean is folded into
    # the loss and the gradient *inside* the kernel rather than by rescaling the
    # whole (BT, V) tensor afterwards.
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum()
    norm = make_norm(n_non_ignore, reduction, _input.device)

    # fp32 accumulator: the per-row losses are summed below, and a 16-bit buffer
    # overflows once BT * loss exceeds the dtype range (fp16 caps at 65504).
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device)
    z_loss_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device) if return_z_loss else None

    # The gradient is NOT produced here. We only save the per-row lse (BT floats),
    # and backward rebuilds softmax from it with grad_output folded in. That keeps
    # the logits intact through forward and avoids the extra (BT, V) rescale pass a
    # pre-scaled gradient would need.
    input_requires_grad = bool(_input.requires_grad)
    target_i32 = target.to(torch.int32)
    lse_1d = torch.empty(BT, dtype=torch.float32, device=_input.device) if input_requires_grad else None

    launch_ce_on_logits(
        _input,
        target_i32,
        loss_1d,
        norm,
        ignore_index=ignore_index,
        has_grad=False,
        lse_1d=lse_1d,
        lse_square_scale=lse_square_scale,
        softcap=softcap,
        label_smoothing=label_smoothing,
        return_z_loss=return_z_loss,
        z_loss_1d=z_loss_1d,
    )

    # The kernel already applied inv_n, so there is nothing to rescale here.
    if reduction == "none":
        loss = loss_1d.to(_input.dtype)
        z_loss = z_loss_1d.to(_input.dtype) if return_z_loss else None
    else:
        loss = torch.sum(loss_1d).to(_input.dtype)
        z_loss = torch.sum(z_loss_1d).to(_input.dtype) if return_z_loss else None

    return loss, z_loss, None, None, _input, target_i32, lse_1d, norm


def cross_entropy_backward(
    _input,
    grad_output,
    target_i32,
    lse_1d,
    norm,
    *,
    ignore_index,
    lse_square_scale=0.0,
    softcap=None,
    label_smoothing=0.0,
    return_z_loss=False,
):
    """Produce the gradient in-place over the logits, with grad_output folded in.

    No D2H sync (grad_output stays a device tensor) and no extra (BT, V) pass:
    the upstream gradient scales the coefficient rather than a materialized,
    unscaled gradient.
    """
    return launch_ce_grad_on_logits(
        _input,
        target_i32,
        lse_1d,
        norm,
        grad_output,
        ignore_index=ignore_index,
        lse_square_scale=lse_square_scale,
        softcap=softcap,
        label_smoothing=label_smoothing,
        return_z_loss=return_z_loss,
    )


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    FlyDSL autograd wrapper for Cross Entropy loss.

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

        loss, z_loss, token_accuracy, predicted_tokens, _input, target_i32, lse_1d, norm = cross_entropy_forward(
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
            # The logits are saved *unmodified*; backward turns them into the
            # gradient in place.
            ctx.save_for_backward(_input.detach(), target_i32, lse_1d, norm)
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        ctx.ignore_index = ignore_index
        ctx.lse_square_scale = lse_square_scale
        ctx.softcap = softcap
        ctx.label_smoothing = label_smoothing

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

        _input, target_i32, lse_1d, norm = ctx.saved_tensors
        _input = cross_entropy_backward(
            _input,
            grad_output,
            target_i32,
            lse_1d,
            norm,
            ignore_index=ctx.ignore_index,
            lse_square_scale=ctx.lse_square_scale,
            softcap=ctx.softcap,
            label_smoothing=ctx.label_smoothing,
            return_z_loss=ctx.return_z_loss,
        )
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
