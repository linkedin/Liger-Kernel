"""TileLang SIMD kernels, launch helpers and autograd for SwiGLU on Ascend 950."""

from math import prod

import tilelang
import tilelang.ascend.language as T
import torch

from torch.autograd.function import once_differentiable

from liger_kernel.ops.tilelang.ascend.utils import ASCEND_950_UB_SIZE_BYTES
from liger_kernel.ops.tilelang.ascend.utils import DTYPE_MAP
from liger_kernel.ops.tilelang.ascend.utils import load_float
from liger_kernel.ops.tilelang.ascend.utils import store_float


@tilelang.jit(target="ascend", out_idx=[])
def swiglu_forward_kernel(
    dtype: str,
    block: int = 16384,
    cores: int = 56,
    unit: bool = False,
    aligned: bool = False,
    ndim: int = 1,
):
    shape = tuple(T.dynamic(f"s{i}", "int64") for i in range(ndim))
    n = prod(shape)

    @T.prim_func
    def main(
        A: T.Tensor(shape, dtype),
        B: T.Tensor(shape, dtype),
        C: T.Tensor(shape, dtype),
        gate: T.float32,
        down: T.float32,
    ):
        # Flatten GM addressing without creating Torch views at each launch.
        A_flat = T.Tensor((n,), dtype, A.data)
        B_flat = T.Tensor((n,), dtype, B.data)
        C_flat = T.Tensor((n,), dtype, C.data)
        with T.Kernel(cores) as bx:
            a = T.alloc_shared((block,), dtype)
            b = T.alloc_shared((block,), dtype)
            c = T.alloc_shared((block,), dtype)
            T.annotate_buffer_versions({a: 2, b: 2, c: 2})
            for step in T.Pipelined(T.ceildiv(T.ceildiv(n, block) - bx, cores), num_stages=2):
                start = (T.cast(step, "int64") * cores + bx) * block
                valid = block if aligned else T.min(block, n - start)
                T.copy(A_flat[start : start + valid], a[:valid], l2_cache_ctrl=4)
                T.copy(B_flat[start : start + valid], b[:valid], l2_cache_ctrl=4)
                with T.SimdVF():
                    one = T.simd.vdup(1.0, "float32")
                    zero = T.simd.vdup(0.0, "float32")
                    for i in T.serial(T.ceildiv(valid, 64)):
                        av = load_float(a, i * 64, dtype)
                        if dtype != "float32":
                            bv = T.simd.vld(b[i * 64], dist="UNPK_B16")
                        else:
                            bv = load_float(b, i * 64, dtype)
                        if unit:
                            x = av
                        else:
                            x = T.simd.vmuls(av, gate)
                        ex = T.simd.vexpdif(zero, x)
                        denom = T.simd.vadds(ex, 1.0)
                        # The hardware divide is sufficient for SwiGLU's tested
                        # FP32 tolerance; avoid the costly exact-division emulation.
                        sigmoid = T.simd.vdiv(one, denom, precision="ftz_true")
                        silu = T.simd.vmul(x, sigmoid)
                        if dtype != "float32":
                            # Multiply in the input dtype to preserve SiLU rounding
                            # without converting both operands back to FP32.
                            narrow_silu = T.simd.vcvt(silu, dtype, sat=False)
                            narrow_product = T.simd.vmul(narrow_silu, bv)
                            if unit:
                                T.simd.vsts(c[i * 64], narrow_product, dist="PK_B32")
                            else:
                                product_float = T.simd.vcvt(narrow_product, "float32")
                                result = T.simd.vmuls(product_float, down)
                                store_float(c, i * 64, result, dtype)
                        else:
                            product = T.simd.vmul(silu, bv)
                            if unit:
                                result = product
                            else:
                                result = T.simd.vmuls(product, down)
                            store_float(c, i * 64, result, dtype)
                T.copy(c[:valid], C_flat[start : start + valid], l2_cache_ctrl=0)

    return main


@tilelang.jit(target="ascend", out_idx=[])
def swiglu_backward_kernel(
    dtype: str,
    block: int = 8192,
    cores: int = 56,
    unit: bool = False,
    aligned: bool = False,
    ndim: int = 1,
):
    shape = tuple(T.dynamic(f"s{i}", "int64") for i in range(ndim))
    n = prod(shape)

    @T.prim_func
    def main(
        A: T.Tensor(shape, dtype),
        B: T.Tensor(shape, dtype),
        DC: T.Tensor(shape, dtype),
        DA: T.Tensor(shape, dtype),
        DB: T.Tensor(shape, dtype),
        gate: T.float32,
        down: T.float32,
    ):
        # Flatten GM addressing without creating Torch views at each launch.
        A_flat = T.Tensor((n,), dtype, A.data)
        B_flat = T.Tensor((n,), dtype, B.data)
        DC_flat = T.Tensor((n,), dtype, DC.data)
        DA_flat = T.Tensor((n,), dtype, DA.data)
        DB_flat = T.Tensor((n,), dtype, DB.data)
        with T.Kernel(cores) as bx:
            a = T.alloc_shared((block,), dtype)
            b = T.alloc_shared((block,), dtype)
            dc = T.alloc_shared((block,), dtype)
            da = T.alloc_shared((block,), dtype)
            db = T.alloc_shared((block,), dtype)
            T.annotate_buffer_versions({a: 2, b: 2, dc: 2, da: 2, db: 2})
            for step in T.Pipelined(T.ceildiv(T.ceildiv(n, block) - bx, cores), num_stages=2):
                start = (T.cast(step, "int64") * cores + bx) * block
                valid = block if aligned else T.min(block, n - start)
                T.copy(A_flat[start : start + valid], a[:valid], l2_cache_ctrl=4)
                T.copy(B_flat[start : start + valid], b[:valid], l2_cache_ctrl=4)
                T.copy(DC_flat[start : start + valid], dc[:valid], l2_cache_ctrl=4)
                with T.SimdVF():
                    one = T.simd.vdup(1.0, "float32")
                    zero = T.simd.vdup(0.0, "float32")
                    for i in T.serial(T.ceildiv(valid, 64)):
                        av = load_float(a, i * 64, dtype)
                        bv = load_float(b, i * 64, dtype)
                        gv = load_float(dc, i * 64, dtype)
                        if unit:
                            x = av
                        else:
                            x = T.simd.vmuls(av, gate)
                        ex = T.simd.vexpdif(zero, x)
                        denom = T.simd.vadds(ex, 1.0)
                        # The hardware divide is sufficient for SwiGLU's tested
                        # FP32 tolerance; avoid the costly exact-division emulation.
                        sigmoid = T.simd.vdiv(one, denom, precision="ftz_true")
                        silu = T.simd.vmul(x, sigmoid)
                        if unit:
                            grad = gv
                        else:
                            grad = T.simd.vmuls(gv, down)
                        complement = T.simd.vsub(one, sigmoid)
                        ds = T.simd.vmul(silu, complement)
                        deriv = T.simd.vadd(ds, sigmoid)
                        gs = T.simd.vmul(grad, deriv)
                        gsb = T.simd.vmul(gs, bv)
                        if unit:
                            ga = gsb
                        else:
                            ga = T.simd.vmuls(gsb, gate)
                        gb = T.simd.vmul(grad, silu)
                        store_float(da, i * 64, ga, dtype)
                        store_float(db, i * 64, gb, dtype)
                T.copy(da[:valid], DA_flat[start : start + valid], l2_cache_ctrl=0)
                T.copy(db[:valid], DB_flat[start : start + valid], l2_cache_ctrl=0)

    return main


def get_max_block_size(
    ub_capacity_bytes: int,
    dtype: torch.dtype,
    buffer_count: int,
    *,
    buffer_versions: int = 1,
    reserve_bytes: int = 8192,
    alignment: int = 64,
) -> int:
    """Bound equal-sized UB tiles, reserving space for padding and compiler temporaries.

    Kernels supply their buffer count and versions; reserve_bytes also covers
    any fixed buffers. The compiler still determines the final UB allocation.
    """
    if buffer_count <= 0 or buffer_versions <= 0 or alignment <= 0 or reserve_bytes < 0:
        raise ValueError("Buffer count, versions and alignment must be positive; reserve_bytes must be nonnegative")
    bytes_per_element = dtype.itemsize * buffer_count * buffer_versions
    block = (ub_capacity_bytes - reserve_bytes) // (bytes_per_element * alignment) * alignment
    if block < alignment:
        raise ValueError("UB budget cannot fit one aligned tile after reserving space")
    return block


def _launch_config(n, dtype, device, backward=False):
    cores = max(1, torch.npu.get_device_properties(device).vector_core_num)
    limit = get_max_block_size(ASCEND_950_UB_SIZE_BYTES, dtype, 5 if backward else 3, buffer_versions=2)
    per_core = max(1, (n + cores - 1) // cores)
    block = max(64, min(1 << (limit.bit_length() - 1), 1 << (per_core - 1).bit_length()))
    return block, max(1, min(cores, (n + block - 1) // block))


def swiglu_forward(a, b, gate_multiplier=1.0, down_multiplier=1.0):
    a, b = a.contiguous(), b.contiguous()
    n = a.numel()
    if n == 0:
        return a, b, torch.empty_like(a)

    with torch.npu.device(a.device):
        block, cores = _launch_config(n, a.dtype, a.device)
        kernel = swiglu_forward_kernel(
            DTYPE_MAP[a.dtype],
            block=block,
            cores=cores,
            unit=gate_multiplier == down_multiplier == 1.0,
            aligned=n % block == 0,
            ndim=a.ndim,
        )
        result = torch.empty_like(a)
        kernel(a, b, result, gate_multiplier, down_multiplier)
    return a, b, result


def swiglu_backward(a, b, dc, gate_multiplier=1.0, down_multiplier=1.0):
    a, b, dc = a.contiguous(), b.contiguous(), dc.contiguous()
    n = a.numel()
    if n == 0:
        return torch.empty_like(a), torch.empty_like(b)

    with torch.npu.device(a.device):
        block, cores = _launch_config(n, a.dtype, a.device, backward=True)
        kernel = swiglu_backward_kernel(
            DTYPE_MAP[a.dtype],
            block=block,
            cores=cores,
            unit=gate_multiplier == down_multiplier == 1.0,
            aligned=n % block == 0,
            ndim=a.ndim,
        )
        da, db = torch.empty_like(a), torch.empty_like(b)
        kernel(a, b, dc, da, db, gate_multiplier, down_multiplier)
    return da, db


class LigerSiLUMulFunction(torch.autograd.Function):
    """First-order SwiGLU autograd; saved inputs are never overwritten."""

    @staticmethod
    def forward(ctx, a, b, gate_multiplier=1.0, down_multiplier=1.0):
        ctx.gate_multiplier = float(gate_multiplier)
        ctx.down_multiplier = float(down_multiplier)
        a, b, c = swiglu_forward(a, b, ctx.gate_multiplier, ctx.down_multiplier)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @once_differentiable
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = swiglu_backward(a, b, dc, ctx.gate_multiplier, ctx.down_multiplier)
        # Both .apply(a, b) and .apply(a, b, gate, down) are public conventions.
        return (da, db, None, None)[: len(ctx.needs_input_grad)]
