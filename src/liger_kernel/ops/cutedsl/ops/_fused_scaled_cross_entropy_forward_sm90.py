"""SM90 fused scaled cross entropy **forward** kernel: producer-less
"MMA-warp issues TMA" mainloop + dedicated SMEM-logits epilogue warp group.

This is the default forward used by
:mod:`liger_kernel.ops.cutedsl.ops.fused_scaled_cross_entropy_sm90`
(``m_tiles_per_cluster == 1``).  It computes the per-token negative
log-likelihood ``nll[M]`` and the log-sum-exp ``lse[M]`` of ``X @ W.T`` without
ever writing the logits to HBM.  The module is self-contained: apart from
:func:`to_cute_tensor` it imports nothing from the rest of the backend.

Architecture
============

384 threads / 3 warp groups, but the split is **not** the production
producer/consumer split:

===========  =========================================================
warp group   role
===========  =========================================================
WG0 (0-127)  WGMMA, rows ``[0, 64)`` of the M128 tile.  **Warp 0 of WG0
             is additionally the *TMA main warp*: it issues every
             global->SMEM ``cp.async.bulk.tensor`` for X and W.**
WG1 (128-255) WGMMA, rows ``[64, 128)`` of the M128 tile.
WG2 (256-383) Epilogue: online softmax max/exp/sum + target logit read
             from a staged SMEM logits tile, one row per thread, plus
             the CTA/cluster DSMEM reduction and the loss/lse stores.
===========  =========================================================

There is **no dedicated producer warp group**, so the whole register file is
split two ways instead of three and 2/3 of the threads do WGMMA instead of 1/2.
The MMA topology itself is unchanged from production: a cooperative
``atom_layout = (2, 1, 1)`` tiled MMA over a ``(64, tile_n)`` WGMMA atom, i.e.
``M128 x N256 x K64`` split along M between the two MMA warp groups, 3 stages,
``cluster_n = 16``, no multicast anywhere.

Mainloop TMA software pipeline
------------------------------

The interesting part is that the TMA-issuing warp is a *member of a WGMMA warp
group*, so it must be back with its warp group before every collective
(``.aligned``) WGMMA instruction or the warp group hangs.  The loop is arranged
so that this is automatic and so that a WGMMA group is always in flight while
the main warp refills a stage::

    prologue (main warp only):  issue S complete TMA stages   [k = 0 .. S-1]

    per output tile:
      k = 0:  consumer_wait(stage 0) ; fence ; WGMMA ; commit
      k = 1 .. K-1:
              consumer_wait(stage k) ; fence ; WGMMA ; commit
              wait_group(1)            <- group k-1 retired, group k IN FLIGHT
              consumer_release(stage k-1)
              if warp == 0: producer_acquire(stage k-1) ; TMA for k-1+S
                            (issued with group k still executing)
              -> all four warps meet again at the next wgmma.fence
      tile tail:
              wait_group(0)            <- only here is the WGMMA pipe drained
              consumer_release(last stage) ; refill it

Warps 1-3 of WG0 simply wait on the *stage* mbarrier while warp 0 refills; they
never wait on warp 0 through a barrier that warp 0 can only reach after a WGMMA,
so the cycle that would deadlock a naive implementation does not exist.  The
load stream is **linear across output tiles** (load ``n`` always goes to stage
``n % S``), so a tile boundary costs no pipeline drain and no TMA bubble: the
main warp is already prefetching the next output tile's W while the current
tile's last WGMMAs run.

Every stage refill is paired 1:1 with the release that freed it, which keeps
``producer_index == release_index`` at every refill without any modular
arithmetic.  The last ``S`` refills of an m-tile are skipped (their data would
be past the end of the stream); those ``S`` unconsumed empty-credits are exactly
the ``S`` credits the next m-tile's prologue consumes, so the pipeline stays
balanced across the persistent m-tile loop.

SMEM logits staging and the register->(row, col) map
----------------------------------------------------

After ``wait_group(0)`` the MMA warp groups convert their FP32 accumulator
straight into a **single** 16-bit SMEM logits tile (no second FP32 tensor).  The
register-to-tile map cannot be taken from ``thr_mma.partition_C(identity)``:
with the cooperative ``atom_layout = (2, 1, 1)`` slicing the production kernel
uses (``get_slice`` of a *constant* warp-group base), ``coords[i]`` is identical
for all 128 threads of a warp group -- it carries no lane term at all, so an
explicit map is required.  The map used here was validated against a real
WGMMA on an H200 (0 / 32768 mismatches)::

    row = wg * 64 + local_warp * 16 + ((i % 4) // 2) * 8 + lane // 4
    col = (i // 4) * 8 + (lane % 4) * 2 + (i % 2)

for ``i`` in ``[0, 128)``, ``wg`` in ``{0, 1}``.

Bank conflicts are the whole ball game here.  The production 2-idle-warp
epilogue ablation was correct but ran at 10.73 ms / 410 TFLOPS, and the reason
is the un-padded ``[128, 256]`` 16-bit tile: with one row per thread, every
thread of a warp lands on the *same* bank (row stride 512 B = 128 words ≡ 0 mod
32), i.e. a 32-way conflict on every access, twice (max pass + sum pass).  This
module pads the row stride to ``tile_n + 8`` 16-bit elements = 528 B = 132
words, which makes **both** directions conflict-free:

* MMA-side stores: element ``i`` and ``i+1`` of the accumulator are the same row
  and adjacent columns, so they pack into one 32-bit ``STS``.  Its word index is
  ``row * 132 + (i // 4) * 4 + lane % 4``; across a warp ``row`` varies as
  ``lane // 4`` so the bank is ``4 * (lane // 4) + lane % 4 (mod 32)`` -- a
  bijection onto the 32 banks.
* epilogue-side loads: 8 consecutive 16-bit logits are 16 B and 16 B-aligned
  (``528 = 33 * 16``), so they issue as ``LDS.128``; the hardware splits that
  into phases of 8 lanes and lanes ``t .. t+7`` cover banks
  ``4t .. 4t+3 (mod 32)`` -- again exactly the 32 banks, once each.

The epilogue also does a **single** pass over the tile instead of production's
max-pass-then-sum-pass: it walks the row in ``chunk_n``-column chunks and folds
each chunk into the running online-softmax state, so the SMEM traffic is halved
and the cross-tile accumulation comes for free.

Because there is exactly one epilogue warp group and it owns a whole row per
thread, there is no per-warp-group partial to combine: thread ``t`` of WG2 holds
the final CTA statistics for row ``t`` and drives the DSMEM cluster reduction
and the ``lse`` / ``token_loss`` stores directly.  No ``wg_max/sum/target``
staging array exists in this kernel.

Masking sentinels
-----------------

Columns past ``V`` read whatever the out-of-range TMA left in SMEM (zeros), so
they must be masked.  They are replaced by ``MASK_F32 = -3e38`` while the online
state initialises to ``NEG_INF_F32 = -1e38``.  The two-sentinel trick removes
the second predicate from the exp pass: ``exp2((-3e38 - m) * log2 e)`` flushes to
zero for every reachable ``m >= -1e38``, including a fully-masked tile where
``m`` is still ``-1e38``, while ``exp2((-1e38 - (-1e38)) * log2 e) == 1`` would
have silently added one per masked column had a single sentinel been used.

No multicast is used anywhere.  There is no HBM logits store in the forward
pass; the only asynchronous stores in this kernel are the TMA *loads*, and the
register->SMEM logits staging is a plain generic ``STS``.

Accepted configuration
----------------------
``M=4096 H=4096 V=131072`` BF16: cooperative ``tile_n = 256`` with a 3-deep AB
pipeline and ``cluster_n = 2``.  ``tile_n = 128``/``192``, ``stages = 2`` and
``cluster_n >= 8`` were all measurably worse; ``stages = 2`` is catastrophic
(no prefetch depth).  NCU on the accepted config: 168 registers/thread, zero
local ld/st (no accumulator spill), tensor pipe ~97 % active.

Two gotchas worth recording
---------------------------

1. **Register-pool deadlock.**  ``2*128*mma_registers + 128*epi_registers`` must
   be ``<= 64512``, *not* ``<= 65536``.  At exactly 65536 (216/80) the hardware
   granted ``setmaxnreg.inc`` to only one MMA warp per SM sub-partition and the
   other four spun forever in ``USETMAXREG.TRY_ALLOC.CTAPOOL``; both this kernel
   (232/40) and the fragment forward (240/24) also total 64512.  Diagnosed by
   attaching ``cuda-gdb -p <pid> -ex "info cuda warps"`` to the hung process.
2. **``llvm.nvvm.mapa`` ICEs** this toolchain ("only supported when pointer size
   is >= 64 bits"), independent of the SMEM offset.  The DSMEM helpers below
   therefore use inline PTX ``mapa.shared::cluster.u32`` with 32-bit
   ``.shared::cluster`` loads/atomics, exactly as
   ``cutlass/cute/arch/smem.py::map_dsmem_ptr`` does for the same reason.  Also
   note ``red.max.f32`` does not exist in PTX -- the cluster max reduction uses
   the monotonic-bit-pattern trick (``atom.max.s32`` / ``atom.min.u32``).
"""

import inspect

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import cutlass_dsl
from cutlass import pipeline
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline.helpers import pipeline_init_arrive
from cutlass.pipeline.helpers import pipeline_init_wait
from cutlass.utils import LayoutEnum
from cutlass.utils import hopper_helpers

from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

THREADS_PER_CTA = 384
NUM_MMA_WARP_GROUPS = 2
WARPS_PER_WARP_GROUP = 4
EPILOGUE_WARP_GROUP = 2

# Named barriers.  0 is reserved for __syncthreads().
MMA_BARRIER_ID = 4  # both MMA warp groups, 256 threads
EPI_BARRIER_ID = 5  # epilogue warp group, 128 threads

LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG_INF_F32 = -1.0e38
# Distinct, *more* negative sentinel for out-of-vocabulary columns; see module
# docstring ("Masking sentinels").
MASK_F32 = -3.0e38
HOPPER_MAX_SMEM_BYTES = 227 * 1024

_compile_cache = {}
_stream_cache = {}
_max_active_clusters_cache = {}

try:
    _FMAX_NEEDS_RESULT_TYPE = next(iter(inspect.signature(nvvm.fmax).parameters)) == "res"
except Exception:
    _FMAX_NEEDS_RESULT_TYPE = False


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None) -> Float32:
    av = Float32(a).ir_value(loc=loc, ip=ip)
    bv = Float32(b).ir_value(loc=loc, ip=ip)
    if _FMAX_NEEDS_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), av, bv, loc=loc, ip=ip))
    return Float32(nvvm.fmax(av, bv, loc=loc, ip=ip))


@dsl_user_op
def _map_dsmem_addr(local_ptr, peer_rank, *, loc=None, ip=None) -> Int32:
    """Map a local SMEM pointer to the same object in a peer CTA's SMEM.

    Returns a **32-bit** ``shared::cluster`` address.  The production module
    uses the ``nvvm.mapa`` intrinsic on a generic pointer, which libNVVM in this
    toolchain rejects for this kernel ("this intrinsic is only supported when
    pointer size is >= 64 bits") once the pointer folds into a constant
    ``addrspacecast`` expression.  ``cutlass.cute.arch.map_dsmem_ptr`` avoids the
    intrinsic entirely by emitting the PTX directly; the same trick is used here
    so every DSMEM access below is a plain 32-bit shared-window address.
    """
    addr = llvm.ptrtoint(T.i32(), local_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [addr, Int32(peer_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _asm(res_ty, args, tmpl, cons, side=True, *, loc=None, ip=None):
    return llvm.inline_asm(
        res_ty,
        args,
        tmpl,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _dsmem_load_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_add_f32(addr, value, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Float32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.add.f32 $0, [$1], $2;",
            "=f,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_max_s32(addr, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.max.s32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_min_u32(addr, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared::cluster.min.u32 $0, [$1], $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsmem_atomic_fmax(addr, value, *, loc=None, ip=None) -> Float32:
    """FP32 max via the monotonic-bit-pattern trick (identical to production).

    ``red/atom.max.f32`` does not exist in PTX, so positive floats are max'd as
    signed ints and negative floats are min'd as unsigned ints.
    """
    bits = llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    old = cutlass_dsl.if_generate(
        Int32(bits) < 0,
        lambda: _dsmem_atomic_min_u32(addr, Int32(bits), loc=loc, ip=ip),
        lambda: _dsmem_atomic_max_s32(addr, Int32(bits), loc=loc, ip=ip),
        [],
        [Int32],
        loc=loc,
        ip=ip,
    )
    return Float32(llvm.bitcast(T.f32(), old.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    stream = _stream_cache.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _stream_cache[raw] = stream
    return stream


def _max_active_clusters(cluster_n):
    key = (torch.cuda.current_device(), cluster_n)
    value = _max_active_clusters_cache.get(key)
    if value is None:
        torch.cuda.current_stream()
        value = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_n)
        _max_active_clusters_cache[key] = value
    return value


@dataclass(frozen=True)
class ScaledCEForwardConfig:
    """Tuning knobs for the MMA-issues-TMA + SMEM-epilogue forward.

    ``tile_m`` is fixed at 128 (one row per epilogue thread) and ``tile_n`` at
    256 (the production cooperative WGMMA topology).  ``logit_pad`` is the extra
    16-bit columns appended to every SMEM logits row; 8 makes the row stride
    528 B and both the MMA-side 32-bit stores and the epilogue-side 128-bit
    loads bank-conflict-free (see the module docstring).  ``chunk_n`` is how many
    columns the epilogue folds into the online-softmax state at a time; it is a
    pure register/rescale trade (``tile_n / chunk_n`` extra rescales per tile).

    Register budget: ``2 * 128 * mma_registers + 128 * epi_registers`` must not
    exceed **64512** registers.  The SM register file holds 65536 registers and
    the CTA is resident alone, but requesting the full file deadlocks: the
    hardware could only satisfy one ``setmaxnreg.inc`` per sub-partition and the
    remaining four MMA warps spun forever in ``USETMAXREG.TRY_ALLOC.CTAPOOL``
    (observed with cuda-gdb at 216/80 = exactly 65536).  This kernel (232/40)
    and the fragment forward (240/24) both stop at 64512, i.e. they leave one
    1024-register (8 regs/thread x 128 threads) reserve.
    """

    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    stages: int = 3
    cluster_n: int = 16
    fast_math: bool = True
    logit_dtype: str = "f16"
    logit_pad: int = 8
    chunk_n: int = 32
    mma_registers: int = 208
    epi_registers: int = 88
    return_entropy: bool = False

    @property
    def logit_stride(self):
        return self.tile_n + self.logit_pad

    def smem_bytes(self):
        ab = (self.tile_m + self.tile_n) * self.tile_k * 2 * self.stages
        logits = self.tile_m * self.logit_stride * 2
        reduction = (3 + int(self.return_entropy)) * self.tile_m * 4
        barriers = 16 * self.stages + 16
        return ab + logits + reduction + barriers + 3 * 1024

    def register_total(self):
        return 2 * 128 * self.mma_registers + 128 * self.epi_registers


class _ScaledCEForwardSM90:
    """MMA-warp-issues-TMA forward with a dedicated SMEM-logits epilogue WG."""

    def __init__(self, config: ScaledCEForwardConfig):
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        self.tile_k = config.tile_k
        self.stages = config.stages
        self.cluster_n = config.cluster_n
        self.fast_math = config.fast_math
        self.logit_pad = config.logit_pad
        self.logit_stride = config.tile_n + config.logit_pad
        self.chunk_n = config.chunk_n
        self.mma_registers = config.mma_registers
        self.epi_registers = config.epi_registers
        self.return_entropy = config.return_entropy
        self.logit_dtype = Float16 if config.logit_dtype == "f16" else BFloat16
        self.tile_shape = (self.tile_m, self.tile_n, self.tile_k)
        self.buffer_align = 1024
        self.mma_barrier = pipeline.NamedBarrier(
            barrier_id=MMA_BARRIER_ID,
            num_threads=NUM_MMA_WARP_GROUPS * 128,
        )
        self.epi_barrier = pipeline.NamedBarrier(
            barrier_id=EPI_BARRIER_ID,
            num_threads=128,
        )

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=1,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        inverse_temperature: Float32,
        ignore_index: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        x_layout = LayoutEnum.from_tensor(x)
        weight_layout = LayoutEnum.from_tensor(weight)
        # Production's cooperative split-M topology, unchanged.
        tiled_mma = hopper_helpers.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            x_layout.sm90_mma_major_mode(),
            weight_layout.sm90_mma_major_mode(),
            Float32,
            (NUM_MMA_WARP_GROUPS, 1, 1),
            (64, self.tile_n),
        )
        a_smem_layout = hopper_helpers.make_smem_layout_a(
            x_layout,
            self.tile_shape,
            BFloat16,
            self.stages,
        )
        b_smem_layout = hopper_helpers.make_smem_layout_b(
            weight_layout,
            self.tile_shape,
            BFloat16,
            self.stages,
        )
        tma_x, tma_tensor_x = self._make_tma(x, a_smem_layout, (self.tile_m, self.tile_k))
        tma_w, tma_tensor_w = self._make_tma(weight, b_smem_layout, (self.tile_n, self.tile_k))

        logit_dtype = self.logit_dtype

        @cute.struct
        class SharedStorage:
            pipeline: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            logits_pipeline: cute.struct.MemRange[cutlass.Int64, 2]
            sX: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(a_smem_layout)],
                self.buffer_align,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(b_smem_layout)],
                self.buffer_align,
            ]
            sLogits: cute.struct.Align[
                cute.struct.MemRange[
                    logit_dtype,
                    self.tile_m * self.logit_stride,
                ],
                self.buffer_align,
            ]
            cluster_max: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_sum: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_target: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16]
            cluster_weighted: cute.struct.Align[
                cute.struct.MemRange[Float32, self.tile_m if self.return_entropy else 1],
                16,
            ]

        self.shared_storage = SharedStorage
        cluster_shape = (1, self.cluster_n, 1)
        num_m_tiles = cute.ceil_div(x.shape[0], self.tile_m)
        num_persistent_clusters = cutlass.min(num_m_tiles, max_active_clusters)
        grid = (1, self.cluster_n, num_persistent_clusters)
        self.kernel(
            tma_x,
            tma_tensor_x,
            tma_w,
            tma_tensor_w,
            target,
            lse,
            token_loss,
            entropy,
            inverse_temperature,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
            ignore_index,
        ).launch(
            grid=grid,
            block=(THREADS_PER_CTA, 1, 1),
            cluster=cluster_shape,
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        x: cute.Tensor,
        tma_w: cute.CopyAtom,
        weight: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        token_loss: cute.Tensor,
        entropy: cute.Tensor,
        inverse_temperature: Float32,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        ignore_index: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        local_tid = tid % 128
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        _, _, pid_m = cute.arch.block_idx()
        _, _, m_tile_stride = cute.arch.grid_dim()
        num_m_tiles = cute.ceil_div(target.shape[0], self.tile_m)

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sX = storage.sX.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sW = storage.sW.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sLogits = storage.sLogits.get_tensor(
            cute.make_layout(
                (self.tile_m, self.tile_n),
                stride=(self.logit_stride, 1),
            )
        )
        cluster_max = storage.cluster_max.get_tensor(cute.make_layout(self.tile_m))
        cluster_sum = storage.cluster_sum.get_tensor(cute.make_layout(self.tile_m))
        cluster_target = storage.cluster_target.get_tensor(cute.make_layout(self.tile_m))
        cluster_weighted = storage.cluster_weighted.get_tensor(
            cute.make_layout(self.tile_m if cutlass.const_expr(self.return_entropy) else 1)
        )

        cta_layout = cute.make_layout((1, self.cluster_n, 1))
        cluster_coord = cta_layout.get_flat_coord(cluster_rank)

        gX = cute.local_tile(
            x,
            cute.slice_(self.tile_shape, (None, 0, None)),
            (None, None, None),
        )
        gW = cute.local_tile(
            weight,
            cute.slice_(self.tile_shape, (0, None, None)),
            (None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        w_cta_layout = cute.make_layout(cute.slice_(cta_layout, (None, 0, 0)).shape)
        tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
            tma_w,
            cluster_coord[0],
            w_cta_layout,
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )

        tma_bytes = cute.size_in_bytes(BFloat16, cute.slice_(a_smem_layout, (None, None, 0))) + cute.size_in_bytes(
            BFloat16, cute.slice_(b_smem_layout, (None, None, 0))
        )

        # Both MMA warp groups (8 warps) consume and release every AB stage.
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipeline.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                # ``PipelineTmaAsync.consumer_release`` is guarded by
                # ``is_signalling_thread`` (= lane 0 of each warp when the CTA
                # layout has size 1), so the empty barrier receives exactly one
                # arrival per consumer WARP, not per thread.
                NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP,
            ),
            tx_count=tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        # One-stage hand-off of the staged SMEM logits tile: the MMA warp groups
        # are the producer, the epilogue warp group the consumer.  Single-thread
        # commit / release on both sides, everybody waits.
        logits_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.logits_pipeline.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, self.cluster_n), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, self.cluster_n))

        num_k_tiles = cute.size(gX, mode=[3])
        num_k_blocks = self.tile_k // 16
        num_pair_tiles = cute.ceil_div(weight.shape[0], self.cluster_n * self.tile_n)

        root_max_addr = _map_dsmem_addr(storage.cluster_max.data_ptr(), 0)
        root_sum_addr = _map_dsmem_addr(storage.cluster_sum.data_ptr(), 0)
        root_target_addr = _map_dsmem_addr(storage.cluster_target.data_ptr(), 0)
        root_weighted_addr = _map_dsmem_addr(storage.cluster_weighted.data_ptr(), 0)

        if warp_group < NUM_MMA_WARP_GROUPS:
            # ================= MMA warp groups (WG0 + WG1) =================
            # Register budget is raised BEFORE the accumulator is materialised
            # so the epilogue warp group never carries its live range.
            cute.arch.setmaxregister_increase(self.mma_registers)

            mma_wg_layout = cute.make_layout(NUM_MMA_WARP_GROUPS, stride=128)
            thr_mma = tiled_mma.get_slice(mma_wg_layout(Int32(warp_group)))
            tCrX = tiled_mma.make_fragment_A(thr_mma.partition_A(sX))
            tCrW = tiled_mma.make_fragment_B(thr_mma.partition_B(sW))
            accum = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.tile_m, self.tile_n)),
                Float32,
            )
            n_accum = cute.size(accum)
            row_base_in_tile = warp_group * 64 + local_warp * 16 + lane // 4
            col_base_in_tile = (lane % 4) * 2

            read_index = Int32(0)
            read_phase = Int32(0)
            rel_index = Int32(0)
            rel_phase = Int32(0)
            load_index = Int32(0)
            # Producer states start at phase 1 (``make_pipeline_state(Producer)``)
            # so the first ``producer_acquire`` on a freshly-initialised empty
            # barrier completes immediately.
            load_phase = Int32(1)
            logit_w_index = Int32(0)
            logit_w_phase = Int32(1)

            while pid_m < num_m_tiles:
                cute.arch.sync_threads()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                tXgX_m = tXgX[(None, pid_m, None, 0)]
                load_k = Int32(0)
                load_t = Int32(0)

                # ---- prologue: S complete TMA stages before any WGMMA ------
                if warp == 0:
                    prologue_stages = cutlass.min(Int32(self.stages), Int32(num_k_tiles) * Int32(num_pair_tiles))
                    for _ in range(prologue_stages):
                        pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                        mainloop.producer_acquire(pstate)
                        bar = mainloop.producer_get_barrier(pstate)
                        cute.copy(
                            tma_x,
                            tXgX_m[(None, load_k)],
                            tXsX[(None, load_index)],
                            tma_bar_ptr=bar,
                            mcast_mask=0,
                        )
                        tWgW_n = tWgW[(None, load_t * self.cluster_n + cluster_rank, None, 0)]
                        cute.copy(
                            tma_w,
                            tWgW_n[(None, load_k)],
                            tWsW[(None, load_index)],
                            tma_bar_ptr=bar,
                        )
                        mainloop.producer_commit(pstate)
                        pstate.advance()
                        load_index = pstate.index
                        load_phase = pstate.phase
                        load_k += 1
                        if load_k == num_k_tiles:
                            load_k = Int32(0)
                            load_t += 1

                for _ in range(num_pair_tiles):
                    accum.fill(0.0)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                    # ---- k = 0: no release yet, nothing to refill ----------
                    read_state = pipeline.PipelineState(self.stages, Int32(0), read_index, read_phase)
                    mainloop.consumer_wait(read_state)
                    cute.nvgpu.warpgroup.fence()
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        coord = (None, None, k_block, read_index)
                        cute.gemm(tiled_mma, accum, tCrX[coord], tCrW[coord], accum)
                    cute.nvgpu.warpgroup.commit_group()
                    read_state.advance()
                    read_index = read_state.index
                    read_phase = read_state.phase

                    # ---- steady state --------------------------------------
                    for _ in range(1, num_k_tiles):
                        read_state = pipeline.PipelineState(self.stages, Int32(0), read_index, read_phase)
                        mainloop.consumer_wait(read_state)
                        cute.nvgpu.warpgroup.fence()
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            coord = (None, None, k_block, read_index)
                            cute.gemm(tiled_mma, accum, tCrX[coord], tCrW[coord], accum)
                        cute.nvgpu.warpgroup.commit_group()
                        # Retire only the previous group: the group just issued
                        # stays IN FLIGHT across the release + TMA refill below.
                        cute.nvgpu.warpgroup.wait_group(1)
                        rstate = pipeline.PipelineState(self.stages, Int32(0), rel_index, rel_phase)
                        mainloop.consumer_release(rstate)
                        rstate.advance()
                        rel_index = rstate.index
                        rel_phase = rstate.phase
                        if warp == 0:
                            if load_t < num_pair_tiles:
                                pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                                mainloop.producer_acquire(pstate)
                                bar = mainloop.producer_get_barrier(pstate)
                                cute.copy(
                                    tma_x,
                                    tXgX_m[(None, load_k)],
                                    tXsX[(None, load_index)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=0,
                                )
                                tWgW_n = tWgW[(None, load_t * self.cluster_n + cluster_rank, None, 0)]
                                cute.copy(
                                    tma_w,
                                    tWgW_n[(None, load_k)],
                                    tWsW[(None, load_index)],
                                    tma_bar_ptr=bar,
                                )
                                mainloop.producer_commit(pstate)
                                pstate.advance()
                                load_index = pstate.index
                                load_phase = pstate.phase
                                load_k += 1
                                if load_k == num_k_tiles:
                                    load_k = Int32(0)
                                    load_t += 1
                        read_state.advance()
                        read_index = read_state.index
                        read_phase = read_state.phase

                    # ---- tile tail: the only WGMMA drain -------------------
                    cute.nvgpu.warpgroup.wait_group(0)
                    rstate = pipeline.PipelineState(self.stages, Int32(0), rel_index, rel_phase)
                    mainloop.consumer_release(rstate)
                    rstate.advance()
                    rel_index = rstate.index
                    rel_phase = rstate.phase
                    if warp == 0:
                        if load_t < num_pair_tiles:
                            pstate = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                            mainloop.producer_acquire(pstate)
                            bar = mainloop.producer_get_barrier(pstate)
                            cute.copy(
                                tma_x,
                                tXgX_m[(None, load_k)],
                                tXsX[(None, load_index)],
                                tma_bar_ptr=bar,
                                mcast_mask=0,
                            )
                            tWgW_n = tWgW[(None, load_t * self.cluster_n + cluster_rank, None, 0)]
                            cute.copy(
                                tma_w,
                                tWgW_n[(None, load_k)],
                                tWsW[(None, load_index)],
                                tma_bar_ptr=bar,
                            )
                            mainloop.producer_commit(pstate)
                            pstate.advance()
                            load_index = pstate.index
                            load_phase = pstate.phase
                            load_k += 1
                            if load_k == num_k_tiles:
                                load_k = Int32(0)
                                load_t += 1

                    # ---- stage the FP32 accumulator as 16-bit SMEM -----
                    wstate = pipeline.PipelineState(1, Int32(0), logit_w_index, logit_w_phase)
                    logits_pipeline.producer_acquire(wstate)
                    for i in cutlass.range_constexpr(n_accum):
                        row = row_base_in_tile + ((i % 4) // 2) * 8
                        col = col_base_in_tile + (i // 4) * 8 + (i % 2)
                        sLogits[row, col] = self.logit_dtype(accum[i])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.fence_acq_rel_cta()
                    self.mma_barrier.arrive_and_wait()
                    if tid == 0:
                        logits_pipeline.producer_commit(wstate)
                    wstate.advance()
                    logit_w_index = wstate.index
                    logit_w_phase = wstate.phase

                cute.arch.sync_threads()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                pid_m += m_tile_stride

            if warp == 0:
                ptail = pipeline.PipelineState(self.stages, Int32(0), load_index, load_phase)
                mainloop.producer_tail(ptail)
        else:
            # ==================== epilogue warp group ======================
            cute.arch.setmaxregister_decrease(self.epi_registers)

            row = local_tid
            chunk = cute.make_rmem_tensor((self.chunk_n,), Float32)
            logit_r_index = Int32(0)
            logit_r_phase = Int32(0)

            while pid_m < num_m_tiles:
                if cluster_rank == 0:
                    cluster_max[row] = Float32(NEG_INF_F32)
                    cluster_sum[row] = Float32(0.0)
                    cluster_target[row] = Float32(0.0)
                    if cutlass.const_expr(self.return_entropy):
                        cluster_weighted[row] = Float32(0.0)
                cute.arch.sync_threads()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                run_max = Float32(NEG_INF_F32)
                run_sum = Float32(0.0)
                run_target = Float32(0.0)
                run_weighted = Float32(0.0)
                global_row = pid_m * self.tile_m + row
                tgt_tile = Int32(-1)
                tgt_col = Int32(-1)
                if global_row < target.shape[0]:
                    tcol = Int32(target[global_row])
                    if tcol >= 0:
                        tgt_tile = tcol // self.tile_n
                        tgt_col = tcol % self.tile_n

                for pair_tile in range(num_pair_tiles):
                    rstate = pipeline.PipelineState(1, Int32(0), logit_r_index, logit_r_phase)
                    logits_pipeline.consumer_wait(rstate)
                    cute.arch.fence_acq_rel_cta()

                    n_tile = pair_tile * self.cluster_n + cluster_rank
                    n_base = n_tile * self.tile_n
                    valid_cols = Int32(weight.shape[0]) - n_base

                    # Single pass: fold each chunk straight into the
                    # running online-softmax state.
                    for ci in cutlass.range_constexpr(self.tile_n // self.chunk_n):
                        c0 = ci * self.chunk_n
                        chunk_max = Float32(MASK_F32)
                        # One dynamic bound check per CHUNK instead of one
                        # per column: at the benchmark shape every chunk is
                        # fully in range, so the masked path is never taken
                        # and the hot loop is a pure LDS + max sequence.
                        if c0 + self.chunk_n <= valid_cols:
                            for j in cutlass.range_constexpr(self.chunk_n):
                                v = Float32(sLogits[row, c0 + j]) * inverse_temperature
                                chunk[j] = v
                                chunk_max = _fmax(chunk_max, v)
                        else:
                            for j in cutlass.range_constexpr(self.chunk_n):
                                v = Float32(sLogits[row, c0 + j]) * inverse_temperature
                                if c0 + j >= valid_cols:
                                    v = Float32(MASK_F32)
                                chunk[j] = v
                                chunk_max = _fmax(chunk_max, v)
                        new_max = _fmax(run_max, chunk_max)
                        acc = Float32(0.0)
                        weighted = Float32(0.0)
                        for j in cutlass.range_constexpr(self.chunk_n):
                            exp_value = cute.math.exp2(
                                (chunk[j] - new_max) * LOG2_E,
                                fastmath=self.fast_math,
                            )
                            acc += exp_value
                            if cutlass.const_expr(self.return_entropy):
                                weighted += exp_value * chunk[j]
                        old_scale = cute.math.exp2(
                            (run_max - new_max) * LOG2_E,
                            fastmath=self.fast_math,
                        )
                        run_sum = run_sum * old_scale + acc
                        if cutlass.const_expr(self.return_entropy):
                            run_weighted = run_weighted * old_scale + weighted
                        run_max = new_max

                    if tgt_tile == n_tile:
                        run_target += Float32(sLogits[row, tgt_col]) * inverse_temperature

                    self.epi_barrier.arrive_and_wait()
                    if local_tid == 0:
                        logits_pipeline.consumer_release(rstate)
                    rstate.advance()
                    logit_r_index = rstate.index
                    logit_r_phase = rstate.phase

                cute.arch.sync_threads()

                _dsmem_atomic_fmax(root_max_addr + row * 4, run_max)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                global_max = _dsmem_load_f32(root_max_addr + row * 4)
                scaled_sum = run_sum * cute.math.exp2(
                    (run_max - global_max) * LOG2_E,
                    fastmath=self.fast_math,
                )
                _dsmem_atomic_add_f32(root_sum_addr + row * 4, scaled_sum)
                _dsmem_atomic_add_f32(root_target_addr + row * 4, run_target)
                if cutlass.const_expr(self.return_entropy):
                    scaled_weighted = run_weighted * cute.math.exp2(
                        (run_max - global_max) * LOG2_E,
                        fastmath=self.fast_math,
                    )
                    _dsmem_atomic_add_f32(root_weighted_addr + row * 4, scaled_weighted)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                if cluster_rank == 0:
                    if global_row < target.shape[0]:
                        row_lse = cluster_max[row] + cute.math.log2(cluster_sum[row], fastmath=self.fast_math) * LN2
                        lse[global_row] = row_lse
                        if target[global_row] == ignore_index:
                            token_loss[global_row] = Float32(0.0)
                            if cutlass.const_expr(self.return_entropy):
                                entropy[global_row] = BFloat16(0.0)
                        else:
                            token_loss[global_row] = row_lse - cluster_target[row]
                            if cutlass.const_expr(self.return_entropy):
                                entropy[global_row] = BFloat16(row_lse - cluster_weighted[row] / cluster_sum[row])

                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                pid_m += m_tile_stride


def _validate(x, weight, target):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise RuntimeError("SM90 fused scaled cross entropy forward requires a Hopper (sm90) GPU")
    if x.device != weight.device or x.device != target.device:
        raise ValueError("input, weight, and target must be on the same CUDA device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("SM90 fused scaled cross entropy forward supports BF16 input and weight only")
    if x.ndim != 2 or weight.ndim != 2 or target.ndim != 1:
        raise ValueError("expected input[BT,H], weight[V,H], and target[BT]")
    if x.shape[0] != target.shape[0] or x.shape[1] != weight.shape[1]:
        raise ValueError("input, weight, and target shapes are incompatible")


def _pad_hidden(x, weight, tile_k):
    hidden_size = x.shape[1]
    padded = (hidden_size + tile_k - 1) // tile_k * tile_k
    if padded == hidden_size:
        return x.contiguous(), weight.contiguous(), hidden_size
    return (
        torch.nn.functional.pad(x, (0, padded - hidden_size)),
        torch.nn.functional.pad(weight, (0, padded - hidden_size)),
        hidden_size,
    )


def scaled_ce_forward(
    _input,
    weight,
    target,
    temperature=1.0,
    ignore_index=-100,
    return_entropy=False,
    config: ScaledCEForwardConfig = None,
):
    """M-outer (self-TMA) forward.  Returns per-token ``(nll[M], lse[M])``."""
    _validate(_input, weight, target)
    if config is None:
        config = ScaledCEForwardConfig()
    if config.return_entropy != return_entropy:
        config = ScaledCEForwardConfig(**{**config.__dict__, "return_entropy": return_entropy})
    if config.tile_m != 128:
        raise ValueError("tile_m must be 128 (one epilogue thread per row)")
    if config.tile_n % 8 != 0 or config.tile_n % config.chunk_n != 0:
        raise ValueError("tile_n must be a multiple of 8 and of chunk_n")
    if config.smem_bytes() > HOPPER_MAX_SMEM_BYTES:
        raise ValueError(
            f"config needs ~{config.smem_bytes()} B of shared memory, "
            f"which exceeds the {HOPPER_MAX_SMEM_BYTES} B Hopper opt-in limit"
        )
    if config.register_total() > 64512:
        raise ValueError(f"register budget {config.register_total()} exceeds the usable 64512-register budget")

    target = target.contiguous()
    x_padded, weight_padded, _ = _pad_hidden(_input, weight, config.tile_k)
    bt = x_padded.shape[0]

    lse = torch.empty(bt, device=_input.device, dtype=torch.float32)
    token_loss = torch.empty(bt, device=_input.device, dtype=torch.float32)
    entropy = torch.empty(bt if return_entropy else 1, device=_input.device, dtype=torch.bfloat16)
    x_cute = to_cute_tensor(x_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    weight_cute = to_cute_tensor(weight_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    target_cute = to_cute_tensor(target, assumed_align=8)
    lse_cute = to_cute_tensor(lse, assumed_align=4)
    token_loss_cute = to_cute_tensor(token_loss, assumed_align=4)
    entropy_cute = to_cute_tensor(entropy, assumed_align=2)
    stream = _cute_stream()
    inverse_temperature = Float32(1.0 / temperature)
    max_active_clusters = _max_active_clusters(config.cluster_n)

    key = ("scaled_ce_forward", config, max_active_clusters)
    compiled = _compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _ScaledCEForwardSM90(config),
            x_cute,
            weight_cute,
            target_cute,
            lse_cute,
            token_loss_cute,
            entropy_cute,
            inverse_temperature,
            Int32(ignore_index),
            max_active_clusters,
            stream,
        )
        _compile_cache[key] = compiled
    compiled(
        x_cute,
        weight_cute,
        target_cute,
        lse_cute,
        token_loss_cute,
        entropy_cute,
        inverse_temperature,
        Int32(ignore_index),
        stream,
    )

    return token_loss, entropy if return_entropy else None, lse
