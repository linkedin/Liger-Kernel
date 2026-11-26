from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

import torch
import triton
import triton.language as tl

from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def _relu_squared_forward_kernel(
    Y_ptr,
    Y_stride,
    X_ptr,
    X_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_stride
    Y_ptr += row_idx * Y_stride

    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    # relu(x) = max(0, x), then square
    relu_x = tl.maximum(x_row, 0)
    y_row = relu_x * relu_x

    tl.store(Y_ptr + col_offsets, y_row, mask=mask)


@triton.jit
def _relu_squared_backward_kernel(
    dX_ptr,
    dX_stride,
    dY_ptr,
    dY_stride,
    X_ptr,
    X_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dX_ptr += row_idx * dX_stride
    dY_ptr += row_idx * dY_stride
    X_ptr += row_idx * X_stride

    dy_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0)
    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)

    # d/dx[relu(x)^2] = 2 * relu(x) = 2 * x * (x > 0)
    relu_x = tl.maximum(x_row, 0)
    dx_row = dy_row * 2 * relu_x

    tl.store(dX_ptr + col_offsets, dx_row, mask=mask)


@lru_cache(maxsize=1)
def _ascend_relu_squared_native_supported():
    # The generated launcher's private ABI is validated only for this stack.
    # Keep the existing implementation for newer/default upstream NPU stacks.
    try:
        return version("triton-ascend") == "3.2.2" and version("torch-npu").split("+")[0] == "2.9.0"
    except PackageNotFoundError:
        return False


def relu_squared_forward(X):
    if X.device.type == "npu" and _ascend_relu_squared_native_supported():
        return _ascend_relu_squared_forward(X)
    ori_shape = X.shape
    n_cols = ori_shape[-1]
    X_2d = X.view(-1, n_cols)
    n_rows = X_2d.shape[0]

    Y = torch.empty_like(X_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_squared_forward_kernel[(n_rows,)](
        Y,
        Y.stride(-2),
        X_2d,
        X_2d.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*ori_shape)


def relu_squared_backward(X, dY):
    if X.device.type == "npu" and _ascend_relu_squared_native_supported():
        return _ascend_relu_squared_backward(X, dY)
    ori_shape = dY.shape
    n_cols = ori_shape[-1]
    X_2d = X.view(-1, n_cols)
    dY_2d = dY.view(-1, n_cols)
    n_rows = dY_2d.shape[0]

    dX = torch.empty_like(dY_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_squared_backward_kernel[(n_rows,)](
        dX,
        dX.stride(-2),
        dY_2d,
        dY_2d.stride(-2),
        X_2d,
        X_2d.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dX.view(*ori_shape)


@triton.jit(do_not_specialize=[0, 1])
def _relu_squared_ascend_forward_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    # Preserve NaN propagation; the Ascend maximum lowering uses max-number semantics.
    positive = tl.where(x <= 0, 0.0, x)
    tl.store(Y + offsets, positive * positive, offsets < N)


@triton.jit(do_not_specialize=[0, 1, 2])
def _relu_squared_ascend_backward_kernel(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    dy = tl.load(DY + offsets, offsets < N, other=0).to(tl.float32)
    # Preserve NaN propagation; the Ascend maximum lowering uses max-number semantics.
    positive = tl.where(x <= 0, 0.0, x)
    # Match torch.square backward's dtype rounding before ReLU's gradient gate.
    doubled = (positive * 2).to(X.dtype.element_ty).to(tl.float32)
    gradient = tl.where(x <= 0, 0.0, dy * doubled)
    tl.store(DX + offsets, gradient, offsets < N)


@lru_cache(maxsize=128)
def _ascend_relu_squared_launcher(n, dtype, device_index, backward, block=8192):
    # Cache compiled launchers, never tensors, pointers or streams. The runner
    # resolves the current stream on every invocation, including graph capture.
    grid = (triton.cdiv(n, block), 1, 1)
    with torch.npu.device(device_index):
        if backward:
            compiled = _relu_squared_ascend_backward_kernel.warmup(dtype, dtype, dtype, n, block, grid=grid)
        else:
            compiled = _relu_squared_ascend_forward_kernel.warmup(dtype, dtype, n, block, grid=grid)
        compiled._init_handles()
        run = compiled.run
        function = compiled.function
        metadata = compiled.packed_metadata
        # Match Triton-Ascend 3.2.2's version-dependent stream getter.
        # torch_npu 2.9.0 does not expose the NoWait variant.
        import torch_npu

        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            get_stream = torch_npu._C._npu_getCurrentRawStreamNoWait
        else:
            get_stream = _ascend_relu_squared_autograd_module().current_stream
        get_launch_metadata = compiled.launch_metadata

        def launch(*args):
            stream = get_stream(device_index)
            launch_metadata = get_launch_metadata(grid, stream, *args)
            run(
                *grid,
                stream,
                function,
                metadata,
                launch_metadata,
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                *args,
            )

        from liger_kernel.ops._relu_squared_launcher import native_entry

        if run.compile_only or run.enable_msprof_register_tensor:
            native_library, entry = None, 0
        else:
            native_library, entry = native_entry(compiled, grid)
        launch.native_spec = (compiled, grid, get_stream, device_index, native_library, entry)
        launch.requires_tensor_args = run.enable_msprof_register_tensor
        return launch


def _ascend_relu_squared_forward(x):
    if torch.npu.current_device() != x.device.index:
        with torch.npu.device(x.device):
            return _ascend_relu_squared_forward(x)
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _ascend_relu_squared_launcher(n, x.dtype, x.device.index, False)(x, out)
    return out


def _ascend_relu_squared_backward(x, dy):
    if torch.npu.current_device() != x.device.index:
        with torch.npu.device(x.device):
            return _ascend_relu_squared_backward(x, dy)
    dx = torch.empty_like(dy)
    n = x.numel()
    if n:
        _ascend_relu_squared_launcher(n, x.dtype, x.device.index, True)(x, dy, dx)
    return dx


@lru_cache(maxsize=1)
def _ascend_relu_squared_autograd_module():
    # Load only for NPU. The bridge handles autograd and allocation; arithmetic
    # remains in the Triton kernels above.
    import torch_npu

    from torch.utils.cpp_extension import load

    torch_npu_path = Path(torch_npu.__file__).parent

    return load(
        name="liger_relu_squared_ascend_autograd",
        sources=[str(Path(__file__).with_name("_relu_squared_ascend.cpp"))],
        extra_cflags=["-O2", "-fvisibility=hidden"],
        extra_include_paths=[str(torch_npu_path / "include")],
        extra_ldflags=[f"-L{torch_npu_path / 'lib'}", "-ltorch_npu"],
        verbose=False,
    )


def _ascend_relu_squared_specs(x):
    n, dtype, device = x.numel(), x.dtype, x.device.index
    return (
        _ascend_relu_squared_launcher(n, dtype, device, False).native_spec,
        _ascend_relu_squared_launcher(n, dtype, device, True).native_spec,
    )


@lru_cache(maxsize=1)
def _ascend_relu_squared_cached_op():
    return _ascend_relu_squared_autograd_module().CachedOp(_ascend_relu_squared_specs)


@lru_cache(maxsize=1)
def _ascend_relu_squared_apply():
    return _ascend_relu_squared_cached_op().apply


class LigerReLUSquaredFunction(torch.autograd.Function):
    @classmethod
    def apply(cls, X):
        if (
            X.device.type == "npu"
            and _ascend_relu_squared_native_supported()
            and not torch._C._are_functorch_transforms_active()
        ):
            return _ascend_relu_squared_apply()(X)
        return super().apply(X)

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X):
        Y = relu_squared_forward(X)
        ctx.save_for_backward(X)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        (X,) = ctx.saved_tensors
        dX = relu_squared_backward(X, dY)
        return dX
