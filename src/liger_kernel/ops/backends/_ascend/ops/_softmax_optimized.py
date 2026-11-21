import os

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_COLUMNS = 8192
FORWARD_TILE_ELEMENTS = 16384
BACKWARD_TILE_ELEMENTS = 8192
MAX_TILE_ROWS = 16
LONG_TILE_COLUMNS = 2048
MIN_SPLIT_COLUMNS = 2048
SPLIT_TILE_ROWS = 2
COMPILED_CACHE_SIZE = 128
A3_DEVICE_PREFIX = "Ascend910_93"
MAX_INT32_OFFSET: tl.constexpr = (1 << 31) - 1


@triton.jit(do_not_specialize_on_alignment=[0, 1])
def _softmax_ascend_forward(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    x = tl.load(X + offsets, mask, other=-float("inf")).to(tl.float32)
    exp = tl.exp(x - tl.max(x, 1)[:, None])
    result = exp / tl.sum(exp, 1)[:, None]
    tl.store(Y + offsets, result, mask)


@triton.jit(do_not_specialize_on_alignment=[0, 1])
def _softmax_a2_forward(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    # Include masked tile tails in the bound; large tensors retain int64 indexing.
    if ((ROWS + R - 1) // R) * R * COLS + C > MAX_INT32_OFFSET:
        program = tl.program_id(0).to(tl.int64)
    else:
        program = tl.program_id(0)
    rows = program * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    raw = tl.load(X + offsets, mask, other=-float("inf"))
    maximum = tl.max(raw, 1).to(tl.float32)
    exp = tl.exp(raw.to(tl.float32) - maximum[:, None])
    inverse = 1.0 / tl.sum(exp, 1)
    tl.store(Y + offsets, exp * inverse[:, None], mask)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def _softmax_ascend_backward(DY, Y, DX, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
    y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
    dot = tl.sum(dy * y, 1)
    tl.store(DX + offsets, y * (dy - dot[:, None]), mask)


@triton.jit
def _softmax_long_forward(X, Y, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    maximum = tl.full((), -float("inf"), tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(x, 0))
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        total += tl.sum(tl.exp(x - maximum), 0)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        tl.store(Y + row * COLS + cols, tl.exp(x - maximum) / total, cols < COLS)


@triton.jit
def _softmax_long_backward(DY, Y, DX, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        dy = tl.load(DY + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        y = tl.load(Y + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        total += tl.sum(dy * y, 0)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        dy = tl.load(DY + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        y = tl.load(Y + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        tl.store(DX + row * COLS + cols, y * (dy - total), cols < COLS)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def split_backward(
    DY, Y, DX, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr, T: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    ia = row[:, None] * COLS + a[None, :]
    ib = row[:, None] * COLS + b[None, :]
    ma = row[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    ya = tl.load(Y + ia, ma, other=0).to(tl.float32)
    yb = tl.load(Y + ib, mb, other=0).to(tl.float32)
    da = tl.load(DY + ia, ma, other=0).to(tl.float32)
    db = tl.load(DY + ib, mb, other=0).to(tl.float32)
    dot = tl.sum(ya * da, 1) + tl.sum(yb * db, 1)
    tl.store(DX + ia, ya * (da - dot[:, None]), ma)
    tl.store(DX + ib, yb * (db - dot[:, None]), mb)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def triple_backward(
    DY,
    Y,
    DX,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    U: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    c = C + T + tl.arange(0, U)
    ic = row[:, None] * COLS + c[None, :]
    ia = row[:, None] * COLS + a[None, :]
    ib = row[:, None] * COLS + b[None, :]
    ma = row[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    mc = ma & (c[None, :] < COLS)
    ya = tl.load(Y + ia, ma, other=0).to(tl.float32)
    yb = tl.load(Y + ib, mb, other=0).to(tl.float32)
    da = tl.load(DY + ia, ma, other=0).to(tl.float32)
    db = tl.load(DY + ib, mb, other=0).to(tl.float32)
    yc = tl.load(Y + ic, mc, other=0).to(tl.float32)
    dc = tl.load(DY + ic, mc, other=0).to(tl.float32)
    dot = tl.sum(ya * da, 1) + tl.sum(yb * db, 1) + tl.sum(yc * dc, 1)
    tl.store(DX + ia, ya * (da - dot[:, None]), ma)
    tl.store(DX + ib, yb * (db - dot[:, None]), mb)
    tl.store(DX + ic, yc * (dc - dot[:, None]), mc)


A3_BACKWARD_TILE_ELEMENTS = 4096
A3_FUSED_BACKWARD_TILE_ROWS = 2
A3_FUSED_ROW_MIN = 2048
A2_DEVICE_PREFIX = "Ascend910B"
SPLIT_FORWARD_TILE_ROWS = 4
SPLIT_BACKWARD_TILE_ELEMENTS = 16384
A3_SHORT_ROW_LIMIT = 4096
A3_SHORT_BACKWARD_TILE_ELEMENTS = 8192
A2_TRIPLE_TILE_COLUMNS = 3584
A2_TRIPLE_TILE_ROWS = 4
A2_TRIPLE_FORWARD_TILE_ELEMENTS = 32768


def _ascend_tile(cols):
    columns = triton.next_power_of_2(cols)
    rows = max(1, min(MAX_TILE_ROWS, BACKWARD_TILE_ELEMENTS // columns))
    return rows, columns


@lru_cache(maxsize=COMPILED_CACHE_SIZE)
def _launcher(rows, cols, dtype, device, backward, tile_override=None, multibuffer_override=None):
    row_tile, col_tile = _ascend_tile(cols)
    if not backward:
        row_tile = max(1, min(MAX_TILE_ROWS, FORWARD_TILE_ELEMENTS // col_tile))
    with torch.npu.device(device):
        short_rows = (
            backward
            and cols == col_tile
            and MIN_SPLIT_COLUMNS < cols < MAX_FUSED_COLUMNS
            and rows < A3_SHORT_ROW_LIMIT
            and torch.npu.get_device_name(device).startswith(A3_DEVICE_PREFIX)
        )
        narrow_fused = (
            backward and cols == MIN_SPLIT_COLUMNS and torch.npu.get_device_name(device).startswith(A3_DEVICE_PREFIX)
        )
        prefix = col_tile // 2
        split_forward = (
            not backward
            and MIN_SPLIT_COLUMNS < cols <= MAX_FUSED_COLUMNS
            and cols != col_tile
            and cols - prefix <= prefix // 2
            and torch.npu.get_device_name(device).startswith(A2_DEVICE_PREFIX)
        )
        triple_forward = (
            not backward
            and MIN_SPLIT_COLUMNS < cols < MAX_FUSED_COLUMNS
            and cols == col_tile // 2 + col_tile // 4 + col_tile // 8
            and torch.npu.get_device_name(device).startswith(A2_DEVICE_PREFIX)
        )
        medium_fused = (
            backward
            and cols == col_tile
            and MIN_SPLIT_COLUMNS < cols < MAX_FUSED_COLUMNS
            and A3_FUSED_ROW_MIN <= rows < A3_SHORT_ROW_LIMIT
            and torch.npu.get_device_name(device).startswith(A3_DEVICE_PREFIX)
        )
        persistent = (
            backward
            and not medium_fused
            and cols == col_tile
            and MIN_SPLIT_COLUMNS < cols < MAX_FUSED_COLUMNS
            and torch.npu.get_device_name(device).startswith(A3_DEVICE_PREFIX)
        )
        if triple_forward:
            row_tile = min(MAX_TILE_ROWS, A2_TRIPLE_FORWARD_TILE_ELEMENTS // col_tile)
            middle = triton.next_power_of_2(cols - prefix) // 2
            tail = triton.next_power_of_2(cols - prefix - middle)
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            kernel = _softmax_triple_forward
            constants = (rows, cols, row_tile, prefix, middle, tail)
        elif split_forward:
            row_tile = SPLIT_FORWARD_TILE_ROWS
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            kernel = _softmax_split_forward
            constants = (rows, cols, row_tile, prefix, triton.next_power_of_2(cols - prefix))
        elif narrow_fused or medium_fused:
            row_tile = A3_FUSED_BACKWARD_TILE_ROWS
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            kernel = _softmax_ascend_backward
            constants = (rows, cols, row_tile, col_tile)
        elif persistent:
            tile_elements = A3_SHORT_BACKWARD_TILE_ELEMENTS if short_rows else A3_BACKWARD_TILE_ELEMENTS
            row_tile = max(1, tile_elements // col_tile)
            cores = int(triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"])
            grid = (min(cores, triton.cdiv(rows, row_tile)), 1, 1)
            kernel = _softmax_persistent_backward
            constants = (rows, cols, row_tile, col_tile)
        elif cols > MAX_FUSED_COLUMNS:
            kernel = _softmax_long_backward if backward else _softmax_long_forward
            grid = (rows, 1, 1)
            constants = (cols, LONG_TILE_COLUMNS)
        elif (
            backward
            and MIN_SPLIT_COLUMNS <= cols <= MAX_FUSED_COLUMNS
            and (cols != col_tile or torch.npu.get_device_name(device).startswith(A2_DEVICE_PREFIX))
        ):
            prefix = col_tile // 2
            remainder = cols - prefix
            row_tile = SPLIT_TILE_ROWS
            if cols == col_tile:
                row_tile = SPLIT_BACKWARD_TILE_ELEMENTS // cols
            if cols == A2_TRIPLE_TILE_COLUMNS and torch.npu.get_device_name(device).startswith(A2_DEVICE_PREFIX):
                row_tile = A2_TRIPLE_TILE_ROWS
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            if remainder == triton.next_power_of_2(remainder):
                kernel = split_backward
                constants = (rows, cols, row_tile, prefix, remainder)
            else:
                middle = triton.next_power_of_2(remainder) // 2
                tail = triton.next_power_of_2(remainder - middle)
                kernel = triple_backward
                constants = (rows, cols, row_tile, prefix, middle, tail)
        else:
            kernel = _softmax_ascend_backward if backward else _softmax_ascend_forward
            if not backward and torch.npu.get_device_name(device).startswith(A2_DEVICE_PREFIX):
                kernel = _softmax_a2_forward
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            constants = (rows, cols, row_tile, col_tile)
        if tile_override is not None and cols <= MAX_FUSED_COLUMNS:
            row_tile = tile_override
            programs = triton.cdiv(rows, row_tile)
            grid = (min(cores, programs) if persistent else programs, 1, 1)
            constants = (*constants[:2], row_tile, *constants[3:])
        multibuffer = not (persistent and not short_rows)
        if multibuffer_override is not None:
            multibuffer = multibuffer_override
        compiled = kernel.warmup(
            *([dtype] * (3 if backward else 2)),
            *constants,
            grid=grid,
            multibuffer=multibuffer,
        )
        compiled._init_handles()
        run = compiled.run
        function, metadata = compiled.function, compiled.packed_metadata
        stream_fn = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        if stream_fn is None:
            stream_fn = _native_module().current_stream

        def launch(*args):
            stream = stream_fn(device)
            run(
                *grid,
                stream,
                function,
                metadata,
                compiled.launch_metadata(grid, stream, *args),
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                *args,
            )

        from liger_kernel.ops.backends._ascend.ops._softmax_launcher import native_entry

        library, entry = (
            (None, 0) if run.compile_only or run.enable_msprof_register_tensor else native_entry(compiled, grid)
        )
        launch.native_spec = (compiled, grid, stream_fn, device, library, entry)
        launch.tile_rows = row_tile
        launch.persistent = persistent
        launch.multibuffer = multibuffer
        return launch


def _softmax_forward(x):
    cols = x.shape[-1]
    x = x.contiguous()
    rows = x.numel() // cols
    y = torch.empty_like(x)
    _launcher(rows, cols, x.dtype, x.device.index, False)(x, y)
    return y, cols, rows, False


def _softmax_backward(dy, y, col_tile, row_tile, multi_block_launch):
    dy, y = dy.contiguous(), y.contiguous()
    cols = y.shape[-1]
    rows = y.numel() // cols
    dx = torch.empty_like(y)
    _launcher(rows, cols, y.dtype, y.device.index, True)(dy, y, dx)
    return dx


@lru_cache(maxsize=1)
def _native_module():
    from torch.utils.cpp_extension import load
    from torch_npu.utils.cpp_extension import NpuExtension

    package = Path(torch_npu.__file__).resolve().parent
    source = Path(__file__).with_name("_softmax_native.cpp")
    includes = NpuExtension("liger_softmax", sources=[str(source)]).include_dirs
    includes.append(str(Path(os.environ["ASCEND_HOME_PATH"]) / "include"))
    signature = repr((torch.__version__, torch_npu.__version__, includes)).encode()
    digest = sha256(source.read_bytes() + signature).hexdigest()[:16]
    return load(
        name=f"liger_softmax_ascend_autograd_{digest}",
        sources=[str(source)],
        extra_cflags=["-O2", "-fvisibility=hidden"],
        extra_include_paths=includes,
        extra_ldflags=[f"-L{package / 'lib'}", "-ltorch_npu"],
        verbose=False,
    )


def _native_specs(x):
    cols = x.shape[-1]
    rows = x.numel() // cols
    device = x.device.index
    forward = _launcher(rows, cols, x.dtype, device, False)
    backward = _launcher(rows, cols, x.dtype, device, True)
    return forward.native_spec, backward.native_spec


@lru_cache(maxsize=1)
def _native_apply():
    return _native_module().CachedOp(_native_specs).apply


class LigerSoftmaxFunction(torch.autograd.Function):
    @classmethod
    def apply(cls, x, dim=-1):
        if torch._C._are_functorch_transforms_active() or not x.numel():
            return super().apply(x, dim)
        return _native_apply()(x, dim)

    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor, dim: int = -1):
        rank = input_.ndim
        if dim < -max(rank, 1) or dim >= max(rank, 1):
            raise IndexError(f"Dimension out of range for tensor with {rank} dimensions: {dim}")
        ctx.dim = dim % max(rank, 1)
        ctx.scalar = rank == 0
        ctx.empty = input_.numel() == 0
        if ctx.empty:
            return torch.empty_like(input_)
        if ctx.scalar:
            input_ = input_.reshape(1)
        elif ctx.dim != rank - 1:
            input_ = input_.movedim(ctx.dim, -1)
        y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.multi_block_launch = multi_block_launch
        if ctx.scalar:
            return y.reshape(())
        return y.movedim(-1, ctx.dim).contiguous()

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        if ctx.empty:
            result = torch.empty_like(grad_output)
            return (result, None)[: len(ctx.needs_input_grad)]
        if ctx.scalar:
            grad_output = grad_output.reshape(1)
        else:
            grad_output = grad_output.movedim(ctx.dim, -1)
        (y,) = ctx.saved_tensors
        dx = _softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.multi_block_launch,
        )
        dx = dx.reshape(()) if ctx.scalar else dx.movedim(-1, ctx.dim)
        return (dx, None)[: len(ctx.needs_input_grad)]


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def _softmax_persistent_backward(DY, Y, DX, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    # Keep one program per vector core and distribute masked row tiles across it.
    for start in range(tl.program_id(0) * R, ROWS, tl.num_programs(0) * R):
        rows = start.to(tl.int64) + tl.arange(0, R)
        cols = tl.arange(0, C)
        offsets = rows[:, None] * COLS + cols[None, :]
        mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
        dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
        y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
        dot = tl.sum(dy * y, 1)
        tl.store(DX + offsets, y * (dy - dot[:, None]), mask)


@triton.jit(do_not_specialize_on_alignment=[0, 1])
def _softmax_split_forward(
    X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr, T: tl.constexpr
):
    if ((ROWS + R - 1) // R) * R * COLS + C + T > MAX_INT32_OFFSET:
        program = tl.program_id(0).to(tl.int64)
    else:
        program = tl.program_id(0)
    row = program * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    ia = row[:, None] * COLS + a[None, :]
    ib = row[:, None] * COLS + b[None, :]
    ma = row[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    xa = tl.load(X + ia, ma, other=-float("inf"))
    xb = tl.load(X + ib, mb, other=-float("inf"))
    maximum = tl.maximum(tl.max(xa, 1), tl.max(xb, 1)).to(tl.float32)
    ea = tl.exp(xa.to(tl.float32) - maximum[:, None])
    eb = tl.exp(xb.to(tl.float32) - maximum[:, None])
    inverse = 1.0 / (tl.sum(ea, 1) + tl.sum(eb, 1))
    tl.store(Y + ia, ea * inverse[:, None], ma)
    tl.store(Y + ib, eb * inverse[:, None], mb)


@triton.jit(do_not_specialize_on_alignment=[0, 1])
def _softmax_triple_forward(
    X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr, T: tl.constexpr, U: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    c = C + T + tl.arange(0, U)
    ia = rows[:, None] * COLS + a[None, :]
    ib = rows[:, None] * COLS + b[None, :]
    ic = rows[:, None] * COLS + c[None, :]
    ma = rows[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    mc = ma & (c[None, :] < COLS)
    xa = tl.load(X + ia, ma, other=-float("inf")).to(tl.float32)
    xb = tl.load(X + ib, mb, other=-float("inf")).to(tl.float32)
    xc = tl.load(X + ic, mc, other=-float("inf")).to(tl.float32)
    maximum = tl.maximum(tl.maximum(tl.max(xa, 1), tl.max(xb, 1)), tl.max(xc, 1))
    ea = tl.exp(xa - maximum[:, None])
    eb = tl.exp(xb - maximum[:, None])
    ec = tl.exp(xc - maximum[:, None])
    inverse = 1.0 / (tl.sum(ea, 1) + tl.sum(eb, 1) + tl.sum(ec, 1))
    tl.store(Y + ia, ea * inverse[:, None], ma)
    tl.store(Y + ib, eb * inverse[:, None], mb)
    tl.store(Y + ic, ec * inverse[:, None], mc)
