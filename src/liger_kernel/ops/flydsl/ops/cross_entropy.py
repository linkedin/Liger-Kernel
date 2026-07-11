"""
FlyDSL fused cross-entropy (online logsumexp + optional in-place gradient).

Signature-compatible with ``liger_kernel.ops.cross_entropy``. Initial support
covers the common training path: mean/sum/none reduction, ignore_index,
label_smoothing, softcap, z_loss, and in-place gradients. Class weights and
token-accuracy / predicted-token extras are not implemented yet.
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

BLOCK_THREADS = 256

# Compiled launchers keyed by (V, dtype_str, warp_size, feature flags).
_compile_cache: dict[tuple, object] = {}


def _elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


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
):
    """Build a specialized FlyDSL CE launcher for fixed V / dtype / flags."""
    elem_bits = _elem_bits(dtype_str)
    elem_dtype = _elem_type(dtype_str)
    vec_width = 128 // elem_bits
    tile_cols = BLOCK_THREADS * vec_width
    RED_SLOTS = max(1, (BLOCK_THREADS + warp_size - 1) // warp_size)
    use_fast = V >= tile_cols and V % tile_cols == 0
    num_tiles = V // tile_cols if use_fast else 0
    fm_fast = arith.FastMathFlags.fast

    @fx.struct
    class SharedStorage:
        # Separate buffers so consecutive block_reduces (max -> sum -> smooth)
        # cannot race on the same shared memory slot.
        s_red_max: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_red_sum: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_red_smooth: fx.Array[fx.Float32, RED_SLOTS, 16]

    @flyc.kernel
    def ce_kernel(
        mX: fx.Tensor,  # (BT, V) logits; grad written in-place when HAS_GRAD
        mY: fx.Tensor,  # (BT,) int32 targets
        mLoss: fx.Tensor,  # (BT,) per-row loss (input dtype)
        mZLoss: fx.Tensor,  # (BT,) per-row z_loss; written only if RETURN_Z_LOSS
        inv_n_loss: fx.Float32,
        inv_n_z: fx.Float32,
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
            # Load into registers, then barrier so later consumers of this buffer
            # (or a subsequent reuse) cannot overwrite before every thread reads.
            out = fx.memref_load(s_red_buffer, 0)
            gpu.barrier()
            return out

        # Target for this row (int32).
        y_div = fx.logical_divide(mY, fx.make_layout(1, 1))
        y_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        y_rmem = fx.make_rmem_tensor(1, fx.Int32)
        fx.copy_atom_call(y_atom, fx.slice(y_div, (None, bid)), y_rmem)
        y = fx.memref_load_vec(y_rmem)[0]
        is_ignored = y == ignore_index
        y_safe = is_ignored.select(fx.Int32(0), y)

        X_buf = fx.rocdl.make_buffer_tensor(mX)
        row_x = fx.slice(X_buf, (bid, None))

        # Load target logit (scalar).
        x_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        x_div_s = fx.logical_divide(row_x, fx.make_layout(1, 1))
        y_view = fx.slice(x_div_s, (None, y_safe))
        y_rm = fx.make_rmem_tensor(1, elem_dtype)
        fx.copy_atom_call(x_atom_s, y_view, y_rm)
        ori_xy_e = fx.memref_load_vec(y_rm)[0]
        ori_xy = ori_xy_e if dtype_str == "f32" else ori_xy_e.to(fx.Float32)
        if const_expr(HAS_SOFTCAP):
            ori_xy, _ = softcap_act(ori_xy)

        eps = fx.Float32(0.0)
        if const_expr(HAS_SMOOTHING):
            eps = label_smoothing / fx.Float32(float(V))

        # ------------------------------------------------------------------
        # Pass 1a: row max  |  Pass 1b: sum(exp(x - m)) (+ smoothing partial)
        # ------------------------------------------------------------------
        m = c_neg_inf
        if const_expr(use_fast):
            a_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            def _load_vec(div_tensor, idx):
                r = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return fx.memref_load_vec(r)

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(a_div, idx)
                x = vec.to(fx.Float32) if dtype_str != "f32" else vec
                if const_expr(HAS_SOFTCAP):
                    x, _ = softcap_act(x)
                m = m.maximumf(x.reduce(ReductionOp.MAX))
        else:
            for base in range_constexpr(0, V, BLOCK_THREADS):
                idx = tid + base
                is_valid = idx < V
                idx_safe = is_valid.select(idx, 0)
                view = fx.slice(x_div_s, (None, idx_safe))
                r = fx.make_rmem_tensor(1, elem_dtype)
                fx.copy_atom_call(x_atom_s, view, r)
                val_e = fx.memref_load_vec(r)[0]
                val = val_e if dtype_str == "f32" else val_e.to(fx.Float32)
                if const_expr(HAS_SOFTCAP):
                    val, _ = softcap_act(val)
                safe_val = is_valid.select(val, c_neg_inf)
                m = m.maximumf(safe_val)

        m = block_reduce(m, "max", s_red_max)

        d = c_zero
        t_sxs = c_zero
        neg_m2 = m * fx.Float32(-LOG2_E)
        if const_expr(use_fast):
            a_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            def _load_vec2(div_tensor, idx):
                r = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return fx.memref_load_vec(r)

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec2(a_div, idx)
                x = vec.to(fx.Float32) if dtype_str != "f32" else vec
                if const_expr(HAS_SOFTCAP):
                    x, _ = softcap_act(x)
                if const_expr(HAS_SMOOTHING):
                    t_sxs = t_sxs + (x * (fx.Float32(0.0) - eps)).reduce(ReductionOp.ADD, c_zero, 0)
                x_exp = fmath.exp2(x * c_log2e + neg_m2, fastmath=fm_fast)
                d = d + x_exp.reduce(ReductionOp.ADD, fastmath=fm_fast)
        else:
            for base in range_constexpr(0, V, BLOCK_THREADS):
                idx = tid + base
                is_valid = idx < V
                idx_safe = is_valid.select(idx, 0)
                view = fx.slice(x_div_s, (None, idx_safe))
                r = fx.make_rmem_tensor(1, elem_dtype)
                fx.copy_atom_call(x_atom_s, view, r)
                val_e = fx.memref_load_vec(r)[0]
                val = val_e if dtype_str == "f32" else val_e.to(fx.Float32)
                if const_expr(HAS_SOFTCAP):
                    val, _ = softcap_act(val)
                contrib = fmath.exp2(val * c_log2e + neg_m2, fastmath=fm_fast)
                contrib = is_valid.select(contrib, c_zero)
                d = d + contrib
                if const_expr(HAS_SMOOTHING):
                    smooth_term = is_valid.select(val * (fx.Float32(0.0) - eps), c_zero)
                    t_sxs = t_sxs + smooth_term

        d = block_reduce(d, "sum", s_red_sum)
        if const_expr(HAS_SMOOTHING):
            t_sxs = block_reduce(t_sxs, "sum", s_red_smooth)

        # lse = m + ln(d) = m + log2(d) * ln2
        lse = m + fmath.log2(d, fastmath=fm_fast) * c_ln2

        main = (lse - ori_xy) * c_one
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

        # Store per-row loss (and optional z_loss) from thread 0.
        if tid == 0:
            loss_div = fx.logical_divide(mLoss, fx.make_layout(1, 1))
            loss_atom = fx.make_copy_atom(
                fx.UniversalCopy16b() if elem_bits <= 16 else fx.UniversalCopy32b(),
                elem_dtype,
            )
            loss_r = fx.make_rmem_tensor(1, elem_dtype)
            loss_out = loss if dtype_str == "f32" else loss.to(elem_dtype)
            fx.memref_store_vec(vec_full(1, loss_out, elem_dtype), loss_r)
            fx.copy_atom_call(loss_atom, loss_r, fx.slice(loss_div, (None, bid)))

            if const_expr(RETURN_Z_LOSS):
                z_div = fx.logical_divide(mZLoss, fx.make_layout(1, 1))
                z_r = fx.make_rmem_tensor(1, elem_dtype)
                z_out = zl if dtype_str == "f32" else zl.to(elem_dtype)
                fx.memref_store_vec(vec_full(1, z_out, elem_dtype), z_r)
                fx.copy_atom_call(loss_atom, z_r, fx.slice(z_div, (None, bid)))

        # ------------------------------------------------------------------
        # Pass 2: in-place gradient
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

            if const_expr(use_fast):
                a_div = fx.logical_divide(row_x, fx.make_layout(vec_width, 1))
                copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

                for tile_i in range_constexpr(num_tiles):
                    idx = tid + tile_i * BLOCK_THREADS
                    r = fx.make_rmem_tensor(vec_width, elem_dtype)
                    fx.copy_atom_call(copy_atom, fx.slice(a_div, (None, idx)), r)
                    vec = fx.memref_load_vec(r)
                    x = vec.to(fx.Float32) if dtype_str != "f32" else vec
                    t_ssa = None
                    if const_expr(HAS_SOFTCAP):
                        x, t_ssa = softcap_act(x)
                    g = fmath.exp2(x * c_log2e + neg_m2, fastmath=fm_fast) * recip
                    if const_expr(HAS_SMOOTHING):
                        g = g + (fx.Float32(0.0) - eps_g)
                    if const_expr(HAS_SOFTCAP):
                        g = g * (c_one - t_ssa * t_ssa)
                    out_e = g if dtype_str == "f32" else g.to(elem_dtype)
                    fx.memref_store_vec(out_e, r)
                    fx.copy_atom_call(copy_atom, r, fx.slice(a_div, (None, idx)))
            else:
                for base in range_constexpr(0, V, BLOCK_THREADS):
                    idx = tid + base
                    if idx < V:
                        view = fx.slice(x_div_s, (None, idx))
                        r = fx.make_rmem_tensor(1, elem_dtype)
                        fx.copy_atom_call(x_atom_s, view, r)
                        val_e = fx.memref_load_vec(r)[0]
                        val = val_e if dtype_str == "f32" else val_e.to(fx.Float32)
                        t_y = None
                        if const_expr(HAS_SOFTCAP):
                            val, t_y = softcap_act(val)
                        g = fmath.exp2(val * c_log2e + neg_m2, fastmath=fm_fast) * recip
                        if const_expr(HAS_SMOOTHING):
                            g = g + (fx.Float32(0.0) - eps_g)
                        if const_expr(HAS_SOFTCAP):
                            g = g * (c_one - t_y * t_y)
                        out_e = g if dtype_str == "f32" else g.to(elem_dtype)
                        fx.memref_store_vec(vec_full(1, out_e, elem_dtype), r)
                        fx.copy_atom_call(x_atom_s, r, view)

            gpu.barrier()
            # Target correction: -(1 - ls) / N at column y (non-ignored).
            if tid == 0:
                if y != ignore_index:
                    dxy = (fx.Float32(0.0) - (c_one - label_smoothing)) * inv_n_loss
                    view = fx.slice(x_div_s, (None, y))
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
        inv_n_loss: fx.Float32,
        inv_n_z: fx.Float32,
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
            inv_n_loss,
            inv_n_z,
            lse_sq_scale,
            softcap,
            label_smoothing,
            ignore_index,
        ).launch(
            grid=(bt, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_ce


def _get_launcher(
    V: int,
    dtype: torch.dtype,
    has_grad: bool,
    has_zloss: bool,
    return_z_loss: bool,
    has_softcap: bool,
    has_smoothing: bool,
    device: Optional[torch.device] = None,
):
    dtype_str = dtype_to_flydsl_str(dtype)
    ws = _host_warp_size(device)
    key = (V, dtype_str, ws, has_grad, has_zloss, return_z_loss, has_softcap, has_smoothing)
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
        )
        _compile_cache[key] = launcher
    return launcher


def launch_ce_on_logits(
    logits: torch.Tensor,
    target_i32: torch.Tensor,
    loss_1d: torch.Tensor,
    *,
    inv_n_loss: float,
    inv_n_z: float,
    ignore_index: int,
    has_grad: bool,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    label_smoothing: float = 0.0,
    return_z_loss: bool = False,
    z_loss_1d: Optional[torch.Tensor] = None,
):
    """Launch FlyDSL CE on a (possibly chunked) logits tile.

    When ``has_grad`` is True, gradients are written in-place over ``logits``.

    ``inv_n_loss`` / ``inv_n_z`` are host floats passed into the kernel. Callers
    that want to avoid a D2H sync for mean reduction should pass ``1.0`` here and
    scale the summed loss / grads on-device afterward (see ``cross_entropy_forward``
    and the FlyDSL FLCE path). If the caller does pass a mean normalizer, it must
    be computed over the *full* BT (not per chunk).
    """
    if logits.stride(-1) != 1:
        raise ValueError("launch_ce_on_logits requires contiguous logits along V")
    if target_i32.dtype != torch.int32:
        raise TypeError(f"target_i32 must be int32, got {target_i32.dtype}")

    n_rows, V = logits.shape
    has_zloss = bool(lse_square_scale != 0.0 or return_z_loss)
    has_softcap = softcap is not None
    has_smoothing = bool(label_smoothing != 0.0)
    softcap_val = float(softcap) if has_softcap else 0.0
    z_buf = z_loss_1d if return_z_loss else loss_1d
    if return_z_loss and z_loss_1d is None:
        raise ValueError("z_loss_1d is required when return_z_loss=True")

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
            device=logits.device,
        )
        launcher(
            logits,
            target_i32,
            loss_1d,
            z_buf,
            float(inv_n_loss),
            float(inv_n_z),
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

    # Keep the non-ignore count on-device — no .item()/.tolist() D2H sync. The CE
    # kernel always runs with inv_n=1 (sum-style); mean reduction is applied below
    # by scaling loss/grads with a device reciprocal.
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum()

    loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    input_requires_grad = bool(_input.requires_grad)
    launch_ce_on_logits(
        _input,
        target.to(torch.int32),
        loss_1d,
        inv_n_loss=1.0,
        inv_n_z=1.0,
        ignore_index=ignore_index,
        has_grad=input_requires_grad,
        lse_square_scale=lse_square_scale,
        softcap=softcap,
        label_smoothing=label_smoothing,
        return_z_loss=return_z_loss,
        z_loss_1d=z_loss_1d,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        if reduction == "mean":
            inv = torch.reciprocal(n_non_ignore.to(torch.float32).clamp(min=1.0))
            loss = loss * inv.to(dtype=loss.dtype)
            if z_loss is not None:
                z_loss = z_loss * inv.to(dtype=z_loss.dtype)
            if input_requires_grad:
                _input.mul_(inv.to(dtype=_input.dtype))

    return loss, z_loss, None, None, _input


def cross_entropy_backward(_input, grad_output):
    """Scale the saved in-place gradient by grad_output (chain rule from upstream).

    Always multiply — skipping via ``torch.equal`` would force a D2H sync.
    """
    if grad_output.ndim > 0:
        return _input * grad_output.unsqueeze(dim=1)
    return _input * grad_output


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
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

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
