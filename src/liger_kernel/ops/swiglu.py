import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import device_context
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.utils import infer_device_arch

# Blackwell (B200) column-tiling parameters. The original one-row layout
# (BLOCK_SIZE = next_pow2(n_cols)) is H100-tuned and leaves the backward kernel
# occupancy-starved on Blackwell for wide rows. Splitting each row into fixed
# 1024-wide column tiles over a 2D grid raises occupancy so HBM saturates
# (~1.63x backward at n_cols=14336, bit-exact). Tile size 1024 chosen by sweep.
_SWIGLU_TILE = 1024

# Only tile when the original one-row block is register-heavy: next_pow2(n_cols)
# >= this. Narrower rows gain nothing and keep the original one-row layout.
_SWIGLU_TILE_MIN_BLOCK = 16384


def _should_tile(n_cols):
    """Use the Blackwell column-tiled path only for wide-enough rows."""
    return infer_device_arch().startswith("blackwell") and triton.next_power_of_2(n_cols) >= _SWIGLU_TILE_MIN_BLOCK


def _swiglu_tile_settings(n_cols):
    """Pick the column-tile BLOCK_SIZE and num_warps for the Blackwell 2D-grid path."""
    BLOCK_SIZE = min(_SWIGLU_TILE, triton.next_power_of_2(n_cols))
    num_warps = 4
    return BLOCK_SIZE, num_warps


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


# NOTE on `gate_multiplier.to(tl.float32)` below: scalar kernel params are
# specialized to fp32 by eager Triton but to fp64 by Inductor when these kernels
# are launched from inside torch.compile. An fp64 scalar silently promotes the
# whole silu/sigmoid chain to float64, which halves throughput (108us vs 56us at
# 8192x3072 on H100). The explicit cast is a no-op in eager and keeps the
# compiled path identical.


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, gate_multiplier, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, gate_multiplier, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory. a_row already holds a * gate_multiplier.
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    # chain rule pulls an extra factor of gate_multiplier through the pre-activation scaling
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row * gate_multiplier.to(tl.float32)

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


@triton.jit
def _swiglu_forward_kernel_tiled(a_ptr, b_ptr, c_ptr, stride, gate_multiplier, n_cols, BLOCK_SIZE: tl.constexpr):
    # Blackwell path: 2D grid -- axis 0 selects the row, axis 1 the column tile.
    program_id = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1)

    # locate start index (row base; column offset added below)
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel_tiled(dc_ptr, a_ptr, b_ptr, stride, gate_multiplier, n_cols, BLOCK_SIZE: tl.constexpr):
    # Blackwell path: 2D grid -- axis 0 selects the row, axis 1 the column tile.
    program_id = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1)

    # locate start index (row base; column offset added below)
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory. a_row already holds a * gate_multiplier.
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    # chain rule pulls an extra factor of gate_multiplier through the pre-activation scaling
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row * gate_multiplier.to(tl.float32)

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b, gate_multiplier: float = 1.0):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    if _should_tile(n_cols):
        # Blackwell (B200), wide rows: column-tiled 2D grid for higher SM occupancy.
        BLOCK_SIZE, num_warps = _swiglu_tile_settings(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
        with device_context(a.device):
            _swiglu_forward_kernel_tiled[grid](
                a,
                b,
                c,
                c.stride(-2),
                float(gate_multiplier),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return a, b, c.view(*ori_shape)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with device_context(a.device):
        _swiglu_forward_kernel[(n_rows,)](
            a,
            b,
            c,
            c.stride(-2),
            float(gate_multiplier),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc, gate_multiplier: float = 1.0):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    if _should_tile(n_cols):
        # Blackwell (B200), wide rows: column-tiled 2D grid for higher SM occupancy.
        BLOCK_SIZE, num_warps = _swiglu_tile_settings(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
        with device_context(a.device):
            _swiglu_backward_kernel_tiled[grid](
                dc,
                a,
                b,
                dc.stride(-2),
                float(gate_multiplier),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return a.view(*ori_shape), b.view(*ori_shape)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with device_context(a.device):
        _swiglu_backward_kernel[(n_rows,)](
            dc,
            a,
            b,
            dc.stride(-2),
            float(gate_multiplier),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return a.view(*ori_shape), b.view(*ori_shape)


class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier

        if isinstance(a, torch.distributed.tensor.DTensor) or isinstance(b, torch.distributed.tensor.DTensor):
            device_mesh, placements = (
                (a.device_mesh, a.placements)
                if isinstance(a, torch.distributed.tensor.DTensor)
                else (b.device_mesh, b.placements)
            )

            # Assume that full tensors are gathered before and identical across
            # the associated process groups.
            if not isinstance(a, torch.distributed.tensor.DTensor):
                a = torch.distributed.tensor.distribute_tensor(a, device_mesh=device_mesh, placements=placements)
            if not isinstance(b, torch.distributed.tensor.DTensor):
                b = torch.distributed.tensor.distribute_tensor(b, device_mesh=device_mesh, placements=placements)
            a_local, b_local, c_local = swiglu_forward(a.to_local(), b.to_local(), gate_multiplier)
            if down_multiplier != 1.0:
                c_local = c_local * down_multiplier
            ctx.save_for_backward(a_local, b_local)
            ctx.dtensor_metadata = (device_mesh, placements)
            return torch.distributed.tensor.DTensor.from_local(c_local, device_mesh, placements)
        else:
            a, b, c = swiglu_forward(a, b, gate_multiplier)
            if down_multiplier != 1.0:
                c = c * down_multiplier
            ctx.save_for_backward(a, b)
            ctx.dtensor_metadata = None
            return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier
        down_multiplier = ctx.down_multiplier

        if ctx.dtensor_metadata is not None:
            device_mesh, placements = ctx.dtensor_metadata

            # Assume that full tensors are gathered before and identical across
            # the associated process groups.
            dc_local = (
                dc.to_local()
                if isinstance(dc, torch.distributed.tensor.DTensor)
                else torch.distributed.tensor.distribute_tensor(dc, device_mesh=device_mesh, placements=placements)
            )
            if down_multiplier != 1.0:
                dc_local = dc_local * down_multiplier
            a_local, b_local = swiglu_backward(a, b, dc_local, gate_multiplier)
            return (
                torch.distributed.tensor.DTensor.from_local(a_local, device_mesh, placements),
                torch.distributed.tensor.DTensor.from_local(b_local, device_mesh, placements),
                None,
                None,
            )

        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        a, b = swiglu_backward(a, b, dc, gate_multiplier)
        return a, b, None, None


# ---------------------------------------------------------------------------
# Fused gate-up variant
# ---------------------------------------------------------------------------
# For fused ``[tokens, 2 * ffn_size]`` gate-up tensors (Megatron, HF ``gate_up_proj``):
# the kernels read both halves via a column offset into the single buffer -- no copies,
# no cat. Input row stride is ``2 * ffn_size``, output is ``ffn_size``.


@triton.jit
def _swiglu_fused_gate_up_forward_kernel(
    y_ptr, c_ptr, in_stride, out_stride, ffn_size: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    y_ptr += program_id * in_stride
    c_ptr += program_id * out_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < ffn_size

    # Gate occupies columns [0, ffn_size), up occupies [ffn_size, 2 * ffn_size) of the same row.
    gate = tl.load(y_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(y_ptr + ffn_size + col_offsets, mask=mask, other=0)
    tl.store(c_ptr + col_offsets, silu(gate).cast(up.dtype) * up, mask=mask)


@triton.jit
def _swiglu_fused_gate_up_backward_kernel(
    dc_ptr, y_ptr, dy_ptr, in_stride, out_stride, ffn_size: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    dc_ptr += program_id * out_stride
    y_ptr += program_id * in_stride
    dy_ptr += program_id * in_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < ffn_size

    dc = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    gate = tl.load(y_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(y_ptr + ffn_size + col_offsets, mask=mask, other=0)

    # Recompute silu from saved input. When dy_ptr aliases y_ptr (in_place=True), all
    # loads precede all stores and each program owns one row, so aliasing is safe here.
    # Caller-level safety is enforced in swiglu_fused_gate_up_backward.
    sig = tl.sigmoid(gate)
    silu_gate = gate * sig
    d_gate = dc * (silu_gate * (1 - sig) + sig) * up
    d_up = dc * silu_gate

    tl.store(dy_ptr + col_offsets, d_gate, mask=mask)
    tl.store(dy_ptr + ffn_size + col_offsets, d_up, mask=mask)


def swiglu_fused_gate_up_forward(y):
    """SwiGLU over a fused ``[..., 2 * ffn_size]`` gate-up tensor. Returns ``(y, c)``."""
    ori_shape = y.shape
    fused_size = ori_shape[-1]
    if fused_size % 2 != 0:
        raise ValueError(f"fused gate-up input must have an even trailing dim; got {fused_size}.")
    ffn_size = fused_size // 2

    y = y.view(-1, fused_size)
    n_rows = y.shape[0]
    c = torch.empty(n_rows, ffn_size, dtype=y.dtype, device=y.device)

    BLOCK_SIZE, num_warps = calculate_settings(ffn_size)
    _swiglu_fused_gate_up_forward_kernel[(n_rows,)](
        y,
        c,
        y.stride(-2),
        c.stride(-2),
        ffn_size=ffn_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y, c.view(*ori_shape[:-1], ffn_size)


def swiglu_fused_gate_up_backward(y, dc, in_place=False):
    """Gradient w.r.t. the fused ``[..., 2 * ffn_size]`` gate-up tensor.

    Args:
        in_place: Overwrite ``y`` with the gradient in place instead of allocating a new
            ``[..., 2 * ffn_size]`` buffer. Saves ~1 GB peak on H100 at ``[2048*4, 2*32768]`` bf16.
            Default False -- Megatron may hold references to the fc1 output for activation
            recompute or CUDA-graph capture, so clobbering it requires care.
    """
    fused_size = y.shape[-1]
    ffn_size = fused_size // 2

    y = y.view(-1, fused_size)
    dc = dc.view(-1, ffn_size)
    n_rows = dc.shape[0]
    dy = y if in_place else torch.empty_like(y)

    BLOCK_SIZE, num_warps = calculate_settings(ffn_size)
    _swiglu_fused_gate_up_backward_kernel[(n_rows,)](
        dc,
        y,
        dy,
        y.stride(-2),
        dc.stride(-2),
        ffn_size=ffn_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dy


class LigerFusedGateUpSiLUMulFunction(torch.autograd.Function):
    """SwiGLU for a single tensor holding gate and up concatenated on the last dim.

    Equivalent to ``LigerSiLUMulFunction.apply(*torch.chunk(y, 2, -1))`` but without the
    contiguous copies that chunking forces, and without concatenating the gradient. Used by
    the Megatron integration, where ``linear_fc1`` always emits this layout.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, y, in_place=False):
        y, c = swiglu_fused_gate_up_forward(y)
        ctx.save_for_backward(y)
        ctx.in_place = in_place
        ctx.already_backward = False
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        (y,) = ctx.saved_tensors
        if ctx.in_place:
            # Triton writes bypass autograd's version counter, so a second backward would
            # silently read gradients instead of activations and produce wrong results.
            if ctx.already_backward:
                raise RuntimeError(
                    "LigerFusedGateUpSiLUMulFunction(in_place=True) can only run backward "
                    "once -- the saved activation is overwritten with the gradient. "
                    "Use in_place=False for retain_graph=True or double-backward."
                )
            ctx.already_backward = True
        return swiglu_fused_gate_up_backward(y, dc, ctx.in_place), None
