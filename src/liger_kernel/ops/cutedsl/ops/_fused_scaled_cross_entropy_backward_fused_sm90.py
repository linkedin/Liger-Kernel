"""One-CUDA-kernel SM90 backward for fused scaled cross entropy.

The persistent device-side wave loop composes the accepted dZ, dX, and dW
designs while preserving their phase-specific scheduling:

* dZ uses cooperative N256 self-TMA WGMMA and the static M-fast scheduler;
* dX uses cooperative N256/K64/s3 with cluster-pair W multicast;
* dW uses transposed ``order=(1, 0, 2)`` TMA and BF16 stores/reduce-adds.

Structure: mechanical composition, not a redesign
-------------------------------------------------
The three accepted kernel bodies are *copied* into phase-local ``@cute.jit``
device functions, split per warp group exactly the way the accepted kernels
already split internally (the accepted dX kernel is literally
``if warp_group == 0: producer else: _mainloop(); _epilogue()``)::

    _dz_epilogue / _dz_mainloop     <- _ScaledCEDzSM90.kernel
    _dx_producer / _dx_consumer     <- _ScaledCEDxGemmSM90.kernel_coop
    _dw_producer / _dw_consumer     <- _ScaledCEDwGemmSM90.kernel

``kernel`` does the setup that each accepted kernel does in its prologue --
SMEM arena, TMA partitions, pipelines, MMA thread slices, WGMMA operand
fragments, the accumulator -- and then runs the loop that used to live on the
host in ``_backward_waved``::

    for wave in range(num_waves):
        dZ phase                    # persistent tile loop, unchanged
        grid barrier; registers -> 24 / 240
        dX phase                    # persistent unit loop, unchanged
        bar.sync                    # sA / sB are aliased between dX and dW
        dW phase                    # persistent tile loop, unchanged
        registers -> 88 / 208; grid barrier

Each phase's persistent loop, tile scheduler, pipeline depth, store staging and
register budget are the accepted ones.  The only edits inside a phase body are
(a) tile coordinates gain the wave offset, and (b) pipeline state arrives and
leaves as explicit ``(index, phase)`` pairs instead of ``PipelineState``
objects, because it has to survive across waves.

Keeping the setup *outside* the wave loop is load-bearing, not cosmetic: a
variant that put the warp-group branch inside each phase function -- and so
rebuilt the fragments and used one accumulator per phase, inside the loop --
spilled 25.5 M local-store sectors (against 22 K) and ran 6 % *slower* than the
three separate launches.  See "Registers" below.

Launch geometry
---------------
One launch, grid ``(2 * C, 1, 1)``, cluster ``(2, 1, 1)``, 384 threads/CTA.
``C`` obeys a hard launch invariant -- **the grid must never exceed the
physical SM count**, or it is not fully resident and the software global
barrier deadlocks::

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    C = min(get_max_active_clusters(2), sm_count // 2, required_work_clusters)
    grid_x = 2 * C                       # asserted <= sm_count

``required_work_clusters`` is the largest per-wave cluster demand of the three
phases (dZ's scheduler plan, dX's ``ceil(m_tiles/2) * ceil(Hp/256)`` units,
``ceil(dW tiles / 2)``), so small problems shrink the grid instead of spinning
idle CTAs through every barrier.  On an H200 / 132 SM part at the target shape
this gives ``C = 66``, i.e. 132 CTAs on 132 SMs.  The shared-memory union is
~219 KiB, which pins occupancy at exactly one CTA per SM, so the whole grid is
*provably* simultaneously resident.  The device side reads the barrier target
from the real ``gridDim.x``, never from a theoretical maximum.  Results are
bitwise invariant to the launched cluster count (verified from ``C = 1``, i.e.
2 CTAs, up to the clamped maximum), so shrinking the grid is free of semantic
risk.

The fixed cluster is interpreted differently by each phase:

=====  =========================================================
phase  interpretation of the physical ``(2, 1, 1)`` cluster
=====  =========================================================
dZ     placement only.  ``cluster_idx = bidx // 2`` selects the persistent
       scheduler slot, ``bidx % 2`` is the accepted ``cluster_n = 2`` rank that
       picks the adjacent n-tile.  No multicast, no cluster barrier.
dX     real ``cluster_m = 2`` with B (= ``W``) TMA multicast, exactly as
       production: ``pid = bidx // 2``, ``cluster_rank_m = bidx % 2``.
dW     ignored.  Both CTAs claim independent ``V x H`` output tiles from the
       flat ``cta_id = bidx`` / ``num_ctas = gridDim.x`` raster.
=====  =========================================================

Shared memory is a **phase-serial union**, not a sum
----------------------------------------------------
The three phase storages are three ``cute.struct`` types instantiated at the
*same* base pointer.  Phases are serialised by a CTA-wide ``bar.sync`` (and, at
the dZ/dX boundary, by the global barrier), so the live ranges never overlap::

    offset        dZ                dX                dW
    0             sX      48 KiB    sA      48 KiB    sA      48 KiB
    49152         sW      96 KiB    sB      96 KiB    sB      96 KiB
    147456        sLogits 66 KiB    sC      32 KiB    sD      64 KiB
    215040        sDZ      8 KiB    -                 -
    -----------------------------------------------------------------
    total            223232 B          180224 B          212992 B

The mbarriers are **not** aliased: a small 160 B struct in front of the union
holds all four pipelines' barrier arrays (dZ mainloop, dZ tile hand-off, dX
mainloop, dW mainloop).  They are initialised once, before the wave loop, with a
single ``pipeline_init_arrive`` / ``pipeline_init_wait`` pair; the per-pipeline
``PipelineState``s are carried across waves as plain ``Int32`` index/phase pairs
(every pipeline is exactly balanced per wave, so this is correct and it avoids
the aliased-reinitialisation hazard entirely).

Registers
---------
Each phase runs at dZ's accepted ``88 / 208`` budget (WG0 ``setmaxnreg.dec 88``,
WG1/WG2 ``setmaxnreg.inc 208``; ``2*128*208 + 128*88 = 64512``, the hard ceiling
-- see the dZ module docstring for why 65536 deadlocks).  Setting
``repartition_registers=True`` additionally swaps to dX/dW's accepted
``24 / 240`` after the first grid barrier of each wave and swaps back before the
second (both swaps decrease before they increase, with the required ``bar.sync``
in between); it is bit-exact but ~15 % slower because the 24-register producer
warp group spills, so it is **off by default** -- see "Measured".

There is a **single** accumulator object shared by all three phases, built once
before the wave loop.  All three tiled MMAs are ``(2, 1, 1)`` atom layouts over
a ``(64, 256)`` atom with a ``128 x 256`` tile, so ``partition_shape_C`` is
identical, and the phases are serial, so the live ranges are provably disjoint.
This is the single most important structural detail in the file: with one
``cute.make_rmem_tensor`` accumulator per phase, created inside the wave loop,
ptxas cannot prove the live ranges disjoint, spills all three, and the kernel
loses ~1.5 ms (see the note under "Structure").

Global barrier
--------------
A **single** ``int32`` HBM counter, zeroed by the host before the launch.  Each
CTA keeps a CTA-uniform ``target_count`` that it raises by the launch's actual
``gridDim.x`` at every barrier, so barrier ``k`` completes at
``(k + 1) * gridDim.x``::

    <issuer warps drain their TMA stores with cp.async.bulk.wait_group 0>
    target_count += gridDim.x
    fence.proxy.async.global          # async-proxy stores -> generic proxy
    bar.sync 0
    if tid == 0:
        atomicAdd(counter, 1)                     # .release.gpu
        while ld.acquire.gpu(counter) < target_count: pass
    bar.sync 0
    fence.proxy.async.global

Safety: if any CTA has not yet arrived at barrier ``k``, no CTA can be past
barrier ``k``, so the global count is at most ``(N - 1) * (k + 1) + k =
(k + 1) * N - 1 < (k + 1) * N``.  A CTA therefore cannot satisfy its target
early no matter how far ahead it runs.  Because the counter starts at zero for
every host invocation there is no reset, no cross-launch monotone base and no
overflow handling.  The cost is one extra tiny ``torch.zeros`` fill per call
(4 bytes); the CuTe kernel count is still exactly one.

Termination rests on residency: the whole grid is co-resident (NCU reports
``waves_per_multiprocessor = 1``), so every CTA does reach every barrier and
the spin always completes.

Two barriers per wave:

1. after dZ, so no CTA reads the BF16 dZ wave workspace before every CTA's
   ``cp.async.bulk.tensor`` stores into it have completed;
2. after dW, so no CTA overwrites the workspace on the next wave before every
   CTA's dX/dW TMA *loads* out of it have completed (they have: a consumer only
   passes ``consumer_wait`` once the load landed, and each phase's producer and
   consumer counts are exactly balanced).

Wave/ragged handling
--------------------
Every wave is processed at full width (``m_tiles_per_wave`` M128 tiles).  Rows
past ``M`` get ``row_scale = 0`` from the dZ epilogue, so the workspace rows are
zero, dX's stores for them are dropped by TMA out-of-bounds predication and dW's
K contributions are zero.  This removes all ragged-tail bookkeeping from the
device loop at the cost of at most one partly-idle wave.

Deviations from the accepted phases (deliberate, measured)
----------------------------------------------------------
* dX stores **BF16 directly** instead of staging FP32 and letting the host
  cast.  The staging layout is built the same way the accepted dX builds its
  FP32 one (``make_smem_layout_a(ROW_MAJOR, (tile_m, tile_n, store_tile_n))``)
  and keeps dX's accepted ``store_tile_n = 32`` / ``store_stages = 1``; only the
  element type differs, which is exactly what the accepted dZ already does for
  its own BF16 store.  RTNE either way, so the result is bit-identical to
  production's ``fp32_out.to(bfloat16)``, and it removes a full ``[M, H]`` FP32
  buffer plus a per-wave copy kernel -- which a one-launch kernel cannot have.
* dW's per-tile pipeline state is carried incrementally instead of being
  recomputed from ``it``; the recomputation is only valid inside a single
  launch's tile stream.
* ``producer_tail`` is dropped for dX/dW: producer commits and consumer waits
  are exactly balanced per wave and the phase-boundary ``bar.sync`` keeps the
  producer from running into the next phase's aliased SMEM.

Measured (H200 SXM, CuTe DSL 4.5.2, ``sm_90a``)
------------------------------------------------
``M = 4096``, ``H = 4096``, ``V = 131072``, ``m_tiles_per_wave = 8``
(4 waves, 13.194 TFLOP of GEMM), interleaved A/B against the accepted
host-orchestrated backward in one process, same inputs and same BF16 outputs
(that three-launch baseline has since been retired from the tree, so these
numbers are the last head-to-head measurement against it):

======================  ===========  =================  ===========
implementation          median (ms)  effective TFLOP/s  peak alloc
======================  ===========  =================  ===========
production, 3 CuTe      20.72        636.9              1328 MiB
launches per wave
this kernel, 1 launch   20.08        657.0              1312 MiB
======================  ===========  =================  ===========

ratio fused/production 0.969 median (0.973 min, 0.969 mean) -> **1.032x**;
repeat runs land between 1.02x and 1.04x.  ``grad_input`` and ``grad_weight``
are **bit-identical** to the accepted backward on every validated shape
(aligned, ragged tokens/hidden/vocab, ``ignore_index`` rows, zero-gradient
rows, single- and multi-wave BF16 dW accumulation).

Nsight Compute confirms the intended geometry and the residency argument
behind the software barrier::

    grid 132 CTAs   block 384   cluster (2, 1, 1)   waves/SM 1.0
    dynamic SMEM 224256 B   occupancy limit 1 block/SM (SMEM and registers)
    registers/thread 168     tensor pipe active 91.7 % of peak sustained
    stall barrier 7.04 / stall membar 0.00   duration 19.82 ms

against the three production phases' 81.7 % / 99.2 % / 82.9 % (about 87 %
work-weighted), and the torch profiler counts exactly **one** CuTe kernel per
backward call -- plus two tiny ``torch.zeros`` fills (the dW accumulator, which
production pays too, and the barrier counter) -- against production's
16 CuTe kernels + 1 fill.  Torch-profiler census: production 17 CUDA kernels
per call, fused 3.

The one cost the separate launches do not pay is register spilling, and it is
the reason ``repartition_registers`` defaults to **off**.  With the per-phase
repartition on, WG0 runs the dX/dW producer at 24 registers and spills: 83 M
local-load and 22 M local-store sectors.  Pinning ``88 / 208`` throughout
removes it almost entirely and is a large, reproducible win at the target
shape (back-to-back, 10 iterations each, same process pattern)::

    repartition_registers   median ms   vs production   local ld / st sectors
    True  (24/240, 88/208)  23.03       0.906x          82.9 M / 22.2 M
    False (88/208 fixed)    20.08       1.032x          11.1 M /  0.04 M

So the fused kernel keeps dZ's accepted ``88 / 208`` split for every phase
rather than restoring dX/dW's accepted ``24 / 240``; that is the one register
deviation from the accepted per-phase configs, and it is measurement-driven.
``repartition_registers=True`` reproduces the accepted per-phase budgets and
still validates bit-exact, but costs ~15 %.

The residual local traffic that survives ``repartition_registers=False`` is
mostly the *second* BF16 store fragment.  dX and dW never stage at the same
time, so ``d_frag`` aliases ``c_frag`` whenever their store tiles are the same
width -- but the accepted widths differ (dX 32, dW 64), so by default they do
not alias.  Forcing a match is bit-exact and cuts local traffic sharply, yet
buys no time (target shape, wave 8, interleaved)::

    dx/dw store width   frag shared   local ld / st        median ms   ratio
    32 / 64 (accepted)  no            11.17 M / 38016      20.004      1.0000
    64 / 64             yes            4.88 M / 25344      19.997      1.0003
    32 / 32             yes            7.23 M / 31680      20.445      0.9784

Sharing, not subtile count, is what dominates: the accepted split runs *fewer*
store-loop iterations than the 32/32 variant (12 vs 16) yet moves the most
local traffic.  Registers/thread stay at 168 in all three.  Because the effect
is entirely off the critical path -- the kernel is 92-97 % tensor-pipe active
and the local traffic hits L1 -- the accepted per-phase store widths are kept.

"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import BFloat16
from cutlass import Float16
from cutlass import Float32
from cutlass import Int32
from cutlass import pipeline
from cutlass.pipeline.helpers import pipeline_init_arrive
from cutlass.pipeline.helpers import pipeline_init_wait
from cutlass.utils import LayoutEnum
from cutlass.utils import hopper_helpers

from liger_kernel.ops.cutedsl.ops.utils import ensure_cuda_context
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor

THREADS_PER_CTA = 384
NUM_MMA_WARP_GROUPS = 2
WARPS_PER_WARP_GROUP = 4

# Named barriers.  0 is reserved for __syncthreads().
DZ_MMA_BARRIER_ID = 4  # WG1 + WG2 (256 threads)
DZ_EPI_BARRIER_ID = 5  # WG0 (128 threads)
DX_EPI_BARRIER_ID = 6  # WG1 + WG2 (256 threads)
DW_STORE_BARRIER_ID = 7  # WG1 + WG2 (256 threads)

LOG2_E = 1.4426950408889634

HOPPER_MAX_SMEM_BYTES = 227 * 1024
VOCAB_ALIGN = 8
HIDDEN_TILE = 64

_compile_cache = {}
_stream_cache = {}
_max_active_clusters_cache = {}
_sm_count_cache = {}


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    stream = _stream_cache.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _stream_cache[raw] = stream
    return stream


def _max_active_clusters(cluster_size):
    key = (torch.cuda.current_device(), cluster_size)
    value = _max_active_clusters_cache.get(key)
    if value is None:
        torch.cuda.current_stream()
        value = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _max_active_clusters_cache[key] = value
    return value


def _sm_count():
    key = torch.cuda.current_device()
    value = _sm_count_cache.get(key)
    if value is None:
        torch.cuda.current_stream()
        value = int(cutlass.utils.HardwareInfo().get_device_multiprocessor_count())
        _sm_count_cache[key] = value
    return value


@dataclass(frozen=True)
class FusedBackwardConfig:
    """Knobs for the one-kernel backward.  Defaults reproduce the accepted
    per-phase geometry as closely as a single launch allows."""

    m_tiles_per_wave: int = 8

    tile_m: int = 128
    tile_n: int = 256
    tile_k: int = 64
    stages: int = 3

    # ---- dZ (accepted) ----
    dz_group_size: int = 32
    dz_raster: str = "heuristic"  # "heuristic" | "m" | "n"
    dz_logit_dtype: str = "f16"  # "f16" | "bf16"
    dz_logit_pad: int = 8
    dz_store_tile_n: int = 32
    dz_store_stages: int = 1

    # ---- dX (accepted; only the C element type is BF16, not FP32) ----
    dx_raster_group: int = 8
    dx_store_tile_n: int = 32
    dx_store_stages: int = 1

    # ---- dW (accepted / FROZEN) ----
    dw_store_tile_n: int = 64
    dw_store_stages: int = 4
    dw_group_m: int = 1

    # ---- registers: each phase runs at its own accepted budget ----
    epi_registers: int = 88
    mma_registers: int = 208
    repartition_registers: bool = False
    prod_registers: int = 24
    gemm_registers: int = 240

    fast_math: bool = True
    with_entropy: bool = False

    @property
    def wave_rows(self):
        return self.m_tiles_per_wave * self.tile_m

    @property
    def logit_stride(self):
        return self.tile_n + self.dz_logit_pad

    def register_total(self):
        return 2 * 128 * self.mma_registers + 128 * self.epi_registers

    def register_total_gemm(self):
        return 2 * 128 * self.gemm_registers + 128 * self.prod_registers

    def validate(self):
        if self.tile_m != 128:
            raise ValueError("tile_m is fixed at 128 (one row per dZ epilogue thread)")
        if self.m_tiles_per_wave < 1:
            raise ValueError("m_tiles_per_wave must be >= 1")
        for name, v in (
            ("dz_store_tile_n", self.dz_store_tile_n),
            ("dx_store_tile_n", self.dx_store_tile_n),
            ("dw_store_tile_n", self.dw_store_tile_n),
        ):
            if v <= 0 or self.tile_n % v != 0:
                raise ValueError(f"{name} must be a positive divisor of tile_n")
        if self.register_total() > 64512:
            raise ValueError(f"register budget {self.register_total()} exceeds 64512")
        if self.repartition_registers and self.register_total_gemm() > 64512:
            raise ValueError(f"repartitioned register budget {self.register_total_gemm()} exceeds 64512")
        return self


@dataclass(frozen=True)
class _SchedulerPlan:
    """Host-resolved Quack ``TileScheduler.Params`` for the dZ phase."""

    ncluster_m: int
    ncluster_n: int
    along_m: bool
    group_size: int
    group_size_tail: int
    num_groups_regular: int

    @property
    def ncluster_slow(self):
        return self.ncluster_n if self.along_m else self.ncluster_m

    @property
    def num_clusters(self):
        return self.ncluster_m * self.ncluster_n


def _plan_schedule(num_m_tiles, num_n_tiles, cluster_n, group_size, raster):
    """Mirror of Quack's ``TileScheduler.Params.create`` for a dense GEMM."""
    ncluster_m = num_m_tiles
    ncluster_n = (num_n_tiles + cluster_n - 1) // cluster_n
    if raster == "m":
        along_m = True
    elif raster == "n":
        along_m = False
    else:

        def round_up(a, b):
            return (a + b - 1) // b * b

        along_m = round_up(ncluster_n, group_size) > round_up(ncluster_m, group_size)
    ncluster_fast = ncluster_m if along_m else ncluster_n
    gs = min(group_size, ncluster_fast)
    return _SchedulerPlan(
        ncluster_m=ncluster_m,
        ncluster_n=ncluster_n,
        along_m=along_m,
        group_size=gs,
        group_size_tail=ncluster_fast % gs,
        num_groups_regular=ncluster_fast // gs,
    )


class _FusedBackwardSM90:
    """dZ -> dX -> dW for every token wave, inside one persistent kernel."""

    CLUSTER_M = 2

    def __init__(self, config: FusedBackwardConfig, plan: _SchedulerPlan):
        config.validate()
        self.cfg = config
        self.plan = plan
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        self.tile_k = config.tile_k
        self.stages = config.stages
        self.m_tiles_per_wave = config.m_tiles_per_wave
        self.wave_rows = config.wave_rows
        self.fast_math = config.fast_math
        self.with_entropy = config.with_entropy

        self.logit_dtype = Float16 if config.dz_logit_dtype == "f16" else BFloat16
        self.logit_stride = config.logit_stride
        self.dz_store_tile_n = config.dz_store_tile_n
        self.dz_store_stages = config.dz_store_stages
        self.dz_subtiles = self.tile_n // self.dz_store_tile_n

        self.dx_raster_group = config.dx_raster_group
        self.dx_store_tile_n = config.dx_store_tile_n
        self.dx_store_stages = config.dx_store_stages
        self.dx_subtiles = self.tile_n // self.dx_store_tile_n

        self.dw_store_tile_n = config.dw_store_tile_n
        self.dw_store_stages = config.dw_store_stages
        self.dw_subtiles = self.tile_n // self.dw_store_tile_n
        self.dw_group_m = config.dw_group_m

        self.epi_registers = config.epi_registers
        self.mma_registers = config.mma_registers
        self.repartition = config.repartition_registers
        self.prod_registers = config.prod_registers
        self.gemm_registers = config.gemm_registers

        self.tile_shape = (self.tile_m, self.tile_n, self.tile_k)
        self.buffer_align = 1024

        self.dz_mma_bar = pipeline.NamedBarrier(barrier_id=DZ_MMA_BARRIER_ID, num_threads=NUM_MMA_WARP_GROUPS * 128)
        self.dz_epi_bar = pipeline.NamedBarrier(barrier_id=DZ_EPI_BARRIER_ID, num_threads=128)
        self.dx_epi_bar = pipeline.NamedBarrier(barrier_id=DX_EPI_BARRIER_ID, num_threads=NUM_MMA_WARP_GROUPS * 128)
        self.dw_store_bar = pipeline.NamedBarrier(barrier_id=DW_STORE_BARRIER_ID, num_threads=NUM_MMA_WARP_GROUPS * 128)

    # ------------------------------------------------------------------ #
    # schedulers
    # ------------------------------------------------------------------ #
    @cute.jit
    def _dz_decode_work(self, work_idx: Int32, num_clusters_in_group: Int32, rank: Int32):
        """Quack static persistent M-fast decode (dZ), verbatim."""
        plan = self.plan
        gs = cutlass.const_expr(plan.group_size)
        gs_tail = cutlass.const_expr(plan.group_size_tail if plan.group_size_tail > 0 else 1)
        group_id = work_idx // num_clusters_in_group
        id_in_group = work_idx - group_id * num_clusters_in_group
        cid_slow, cid_fast_in_group = Int32(0), Int32(0)
        if cutlass.const_expr(plan.group_size_tail == 0):
            cid_slow = id_in_group // gs
            cid_fast_in_group = id_in_group - cid_slow * gs
        else:
            if group_id < plan.num_groups_regular:
                cid_slow = id_in_group // gs
                cid_fast_in_group = id_in_group - cid_slow * gs
            else:
                cid_slow = id_in_group // gs_tail
                cid_fast_in_group = id_in_group - cid_slow * gs_tail
        if group_id % 2 == 1:  # serpentine
            cid_slow = plan.ncluster_slow - 1 - cid_slow
        cid_fast = group_id * gs + cid_fast_in_group
        if cutlass.const_expr(plan.along_m):
            cid_m, cid_n = cid_fast, cid_slow
        else:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n * self.CLUSTER_M + rank

    def _dx_tile_coord(self, unit, num_m_units, num_n_units):
        """Linear unit id -> ``(m_unit, n_unit)`` under dX's grouped raster."""
        if cutlass.const_expr(self.dx_raster_group <= 1):
            return unit // num_n_units, unit % num_n_units
        group_tiles = num_m_units * self.dx_raster_group
        group = unit // group_tiles
        first_n = group * self.dx_raster_group
        group_n = cutlass.min(Int32(self.dx_raster_group), num_n_units - first_n)
        within = unit - group * group_tiles
        return within // group_n, first_n + within % group_n

    def _dw_tile_coord(self, tile_id, num_m_tiles, num_n_tiles):
        """dW's N-fastest (``group_m == 1``) raster."""
        if cutlass.const_expr(self.dw_group_m <= 1):
            return tile_id // num_n_tiles, tile_id % num_n_tiles
        tiles_per_group = self.dw_group_m * num_n_tiles
        first_m = (tile_id // tiles_per_group) * self.dw_group_m
        gm = cutlass.max(cutlass.min(num_m_tiles - first_m, self.dw_group_m), 1)
        idx = tile_id % tiles_per_group
        return first_m + idx % gm, idx // gm

    @staticmethod
    def _sub_src(store_tile_n, tile_n, sub, j):
        """Sub-tile fragment slot ``j`` -> flat accumulator register index."""
        gs = store_tile_n // 8
        block = 4 * gs
        return (j % block) + block * sub + (tile_n // 2) * (j // block)

    # ------------------------------------------------------------------ #
    # global barrier
    # ------------------------------------------------------------------ #
    @cute.jit
    def _grid_barrier(self, bar_ptr, target: Int32, tid: Int32):
        """Counting barrier over every CTA of the persistent grid.

        ``bar_ptr`` points at a single ``int32`` counter, zeroed by the host
        before the launch.  The caller keeps a CTA-uniform ``target`` that it
        raises by ``gridDim.x`` at every barrier, so barrier ``k`` of the
        launch completes once the counter reaches ``(k + 1) * gridDim.x``.

        Safety: if any CTA has not yet arrived at barrier ``k``, no CTA can be
        past barrier ``k``, so the count is at most ``(N - 1) * (k + 1) + k =
        (k + 1) * N - 1``, below the target.  A CTA therefore cannot satisfy
        its target early no matter how far ahead it runs.

        The caller must already have drained (``cp.async.bulk.wait_group 0``)
        every TMA store whose result the other CTAs are about to read.
        """
        cute.arch.fence_proxy("async.global")
        cute.arch.sync_threads()
        if tid == 0:
            cute.arch.atomic_add(bar_ptr, Int32(1), sem="release", scope="gpu")
            seen = cute.arch.load(bar_ptr, Int32, sem="acquire", scope="gpu")
            while seen < target:
                seen = cute.arch.load(bar_ptr, Int32, sem="acquire", scope="gpu")
        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.global")

    # ------------------------------------------------------------------ #
    # host side
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tma_load(tensor, smem_layout_staged, smem_tile, multicast=1):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if multicast == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            cute.slice_(smem_layout_staged, (None, None, 0)),
            smem_tile,
            num_multicast=multicast,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,  # (M, Hp, 1) row-major BF16     -- dZ A / dW B source
        w: cute.Tensor,  # (Vp, Hp, 1) row-major BF16    -- dZ B
        wt: cute.Tensor,  # (Hp, Vp, 1) MN-major BF16    -- dX B (view of w)
        xt: cute.Tensor,  # (Hp, M, 1) MN-major BF16     -- dW B (view of x)
        dz: cute.Tensor,  # (Wr, Vp, 1) row-major BF16   -- dZ store / dX A
        dzt: cute.Tensor,  # (Vp, Wr, 1) MN-major BF16   -- dW A (view of dz)
        dx: cute.Tensor,  # (M, Hp, 1) row-major BF16    -- dX C
        dw: cute.Tensor,  # (Vp, Hp, 1) row-major BF16   -- dW D (reduce-add)
        target: cute.Tensor,
        lse: cute.Tensor,
        scale: cute.Tensor,
        entropy: cute.Tensor,
        entropy_scale: cute.Tensor,
        inverse_temperature: Float32,
        barrier: cute.Tensor,  # int32 [1]
        ignore_index: Int32,
        vocab_size: Int32,
        num_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        x_layout = LayoutEnum.from_tensor(x)
        w_layout = LayoutEnum.from_tensor(w)
        wt_layout = LayoutEnum.from_tensor(wt)
        xt_layout = LayoutEnum.from_tensor(xt)
        dz_layout = LayoutEnum.from_tensor(dz)
        dzt_layout = LayoutEnum.from_tensor(dzt)
        dw_layout = LayoutEnum.from_tensor(dw)

        atom_layout = (NUM_MMA_WARP_GROUPS, 1, 1)

        def _mma(a_layout, b_layout, n):
            return hopper_helpers.make_trivial_tiled_mma(
                BFloat16,
                BFloat16,
                a_layout.sm90_mma_major_mode(),
                b_layout.sm90_mma_major_mode(),
                Float32,
                atom_layout,
                (64, n),
            )

        mma_dz = _mma(x_layout, w_layout, self.tile_n)
        mma_dx = _mma(dz_layout, wt_layout, self.tile_n)
        mma_dw = _mma(dzt_layout, xt_layout, self.tile_n)
        smma_dx = _mma(dz_layout, wt_layout, self.dx_store_tile_n)
        smma_dw = _mma(dzt_layout, xt_layout, self.dw_store_tile_n)

        # ---- SMEM layouts ------------------------------------------------
        dz_a = hopper_helpers.make_smem_layout_a(x_layout, self.tile_shape, BFloat16, self.stages)
        dz_b = hopper_helpers.make_smem_layout_b(w_layout, self.tile_shape, BFloat16, self.stages)
        dz_st = hopper_helpers.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n, self.dz_store_tile_n),
            BFloat16,
            self.dz_store_stages,
        )
        dx_a = hopper_helpers.make_smem_layout_a(dz_layout, self.tile_shape, BFloat16, self.stages)
        dx_b = hopper_helpers.make_smem_layout_b(wt_layout, self.tile_shape, BFloat16, self.stages)
        dx_st = hopper_helpers.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n, self.dx_store_tile_n),
            BFloat16,
            self.dx_store_stages,
        )
        dw_a = hopper_helpers.make_smem_layout_a(dzt_layout, self.tile_shape, BFloat16, self.stages)
        dw_b = hopper_helpers.make_smem_layout_b(xt_layout, self.tile_shape, BFloat16, self.stages)
        dw_st = hopper_helpers.make_smem_layout_epi(
            BFloat16, dw_layout, (self.tile_m, self.dw_store_tile_n), self.dw_store_stages
        )

        # ---- TMA atoms ---------------------------------------------------
        tma_x, t_x = self._tma_load(x, dz_a, (self.tile_m, self.tile_k))
        tma_w, t_w = self._tma_load(w, dz_b, (self.tile_n, self.tile_k))
        tma_dzs, t_dzs = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            dz,
            cute.slice_(dz_st, (None, None, 0)),
            (self.tile_m, self.dz_store_tile_n),
        )
        tma_dza, t_dza = self._tma_load(dz, dx_a, (self.tile_m, self.tile_k))
        tma_wt, t_wt = self._tma_load(wt, dx_b, (self.tile_n, self.tile_k), multicast=self.CLUSTER_M)
        tma_dx, t_dx = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            dx,
            cute.slice_(dx_st, (None, None, 0)),
            (self.tile_m, self.dx_store_tile_n),
        )
        tma_dwa, t_dwa = self._tma_load(dzt, dw_a, (self.tile_m, self.tile_k))
        tma_xt, t_xt = self._tma_load(xt, dw_b, (self.tile_n, self.tile_k))
        tma_dw_store, t_dw_store = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            dw,
            cute.slice_(dw_st, (None, None, 0)),
            (self.tile_m, self.dw_store_tile_n),
        )
        tma_dw_add, t_dw_add = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD),
            dw,
            cute.slice_(dw_st, (None, None, 0)),
            (self.tile_m, self.dw_store_tile_n),
        )

        logit_dtype = self.logit_dtype

        @cute.struct
        class BarrierStorage:
            dz_pipe: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            dz_tile: cute.struct.MemRange[cutlass.Int64, 2]
            dx_pipe: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            dw_pipe: cute.struct.MemRange[cutlass.Int64, self.stages * 2]

        @cute.struct
        class DzStorage:
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dz_a)], self.buffer_align]
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dz_b)], self.buffer_align]
            sL: cute.struct.Align[cute.struct.MemRange[logit_dtype, self.tile_m * self.logit_stride], self.buffer_align]
            sZ: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dz_st)], self.buffer_align]

        @cute.struct
        class DxStorage:
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dx_a)], self.buffer_align]
            sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dx_b)], self.buffer_align]
            sC: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dx_st)], self.buffer_align]

        @cute.struct
        class DwStorage:
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dw_a)], self.buffer_align]
            sB: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dw_b)], self.buffer_align]
            sD: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(dw_st)], self.buffer_align]

        self.bar_storage = BarrierStorage
        self.dz_storage = DzStorage
        self.dx_storage = DxStorage
        self.dw_storage = DwStorage
        self.union_bytes = max(DzStorage.__sizeof__(), DxStorage.__sizeof__(), DwStorage.__sizeof__())
        if cutlass.const_expr(self.union_bytes + 1024 > HOPPER_MAX_SMEM_BYTES):
            raise ValueError(
                f"phase-serial SMEM union is {self.union_bytes + 1024} B, over the "
                f"{HOPPER_MAX_SMEM_BYTES} B Hopper opt-in limit"
            )

        self.kernel(
            tma_x,
            t_x,
            tma_w,
            t_w,
            tma_dzs,
            t_dzs,
            tma_dza,
            t_dza,
            tma_wt,
            t_wt,
            tma_dx,
            t_dx,
            tma_dwa,
            t_dwa,
            tma_xt,
            t_xt,
            tma_dw_store,
            t_dw_store,
            tma_dw_add,
            t_dw_add,
            target,
            lse,
            scale,
            entropy,
            entropy_scale,
            inverse_temperature,
            barrier,
            mma_dz,
            mma_dx,
            mma_dw,
            smma_dx,
            smma_dw,
            dz_a,
            dz_b,
            dz_st,
            dx_a,
            dx_b,
            dx_st,
            dw_a,
            dw_b,
            dw_st,
            ignore_index,
            vocab_size,
        ).launch(
            grid=(num_clusters * self.CLUSTER_M, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(self.CLUSTER_M, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _regs_for_gemm(self, is_wg0: cutlass.Constexpr):
        """dZ's accepted budget (88 / 208) -> dX and dW's (24 / 240)."""
        if cutlass.const_expr(self.repartition):
            if cutlass.const_expr(is_wg0):
                cute.arch.setmaxregister_decrease(self.prod_registers)
                cute.arch.sync_threads()
            else:
                cute.arch.sync_threads()
                cute.arch.setmaxregister_increase(self.gemm_registers)

    @cute.jit
    def _regs_for_dz(self, is_wg0: cutlass.Constexpr):
        """dX/dW's accepted budget (24 / 240) -> dZ's (88 / 208)."""
        if cutlass.const_expr(self.repartition):
            if cutlass.const_expr(is_wg0):
                cute.arch.sync_threads()
                cute.arch.setmaxregister_increase(self.epi_registers)
            else:
                cute.arch.setmaxregister_decrease(self.mma_registers)
                cute.arch.sync_threads()

    # ------------------------------------------------------------------ #
    # phase 1: dZ = softmax(X @ W^T) - onehot          (one token wave)
    #
    # ``_dz_epilogue`` and ``_dz_mainloop`` are the WG0 and WG1/WG2 halves of
    # ``_ScaledCEDzSM90.kernel``, copied across with the persistent
    # ``for tile in range(...)`` loop intact.  Only the tile *coordinates*
    # gain the wave offset, and the pipeline states arrive and leave as
    # explicit ``(index, phase)`` pairs instead of ``PipelineState`` objects.
    # ------------------------------------------------------------------ #
    @cute.jit
    def _dz_epilogue(
        self,
        wave: Int32,
        tid: Int32,
        local_warp: Int32,
        row_in_tile: Int32,
        cluster_idx: Int32,
        cta_rank: Int32,
        num_clusters: Int32,
        dz_tiles: Int32,
        dz_group: Int32,
        tma_dzs: cute.CopyAtom,
        tZgZ: cute.Tensor,
        tZsZ: cute.Tensor,
        sL: cute.Tensor,
        sZ: cute.Tensor,
        tile_pipe: pipeline.PipelineAsync,
        target: cute.Tensor,
        lse: cute.Tensor,
        scale: cute.Tensor,
        entropy: cute.Tensor,
        entropy_scale: cute.Tensor,
        inverse_temperature: Float32,
        vals: cute.Tensor,
        ignore_index: Int32,
        vocab_size: Int32,
        cr_i: Int32,
        cr_p: Int32,
    ):
        row_base = wave * Int32(self.wave_rows)
        chunk = cutlass.const_expr(self.dz_store_tile_n)

        work_idx = Int32(cluster_idx)
        for _ in range(dz_tiles):
            pid_m, pid_n = self._dz_decode_work(work_idx, dz_group, cta_rank)
            row = row_base + pid_m * self.tile_m + row_in_tile
            n_base = pid_n * self.tile_n

            row_scale = Float32(0.0)
            row_entropy = Float32(0.0)
            row_entropy_scale = Float32(0.0)
            row_lse = Float32(0.0)
            tgt_local = Int32(-1)
            if row < target.shape[0]:
                tcol = Int32(target[row])
                if tcol != ignore_index:
                    row_lse = lse[row]
                    row_scale = scale[row]
                    if cutlass.const_expr(self.with_entropy):
                        row_entropy = Float32(entropy[row])
                        row_entropy_scale = entropy_scale[row]
                    if tcol >= n_base and tcol < n_base + self.tile_n:
                        tgt_local = tcol - n_base

            rstate = pipeline.PipelineState(1, Int32(0), cr_i, cr_p)
            tile_pipe.consumer_wait(rstate)
            cute.arch.fence_acq_rel_cta()

            for sub in cutlass.range_constexpr(self.dz_subtiles):
                stage = sub % self.dz_store_stages
                c0 = sub * chunk
                for j in cutlass.range_constexpr(chunk):
                    vals[j] = Float32(sL[row_in_tile, c0 + j])
                for j in cutlass.range_constexpr(chunk):
                    v = Float32(0.0)
                    if n_base + c0 + j < vocab_size:
                        if cutlass.const_expr(self.with_entropy):
                            if row_scale != 0.0 or row_entropy_scale != 0.0:
                                scaled_logit = vals[j] * inverse_temperature
                                probability = cute.math.exp2(
                                    (scaled_logit - row_lse) * LOG2_E,
                                    fastmath=self.fast_math,
                                )
                                v = probability * (
                                    row_scale + (row_lse - row_entropy - scaled_logit) * row_entropy_scale
                                )
                        else:
                            if row_scale != 0.0:
                                v = (
                                    cute.math.exp2(
                                        (vals[j] * inverse_temperature - row_lse) * LOG2_E,
                                        fastmath=self.fast_math,
                                    )
                                    * row_scale
                                )
                    vals[j] = v
                if tgt_local >= c0 and tgt_local < c0 + chunk:
                    for j in cutlass.range_constexpr(chunk):
                        if tgt_local == c0 + j:
                            vals[j] = vals[j] - row_scale

                if local_warp == 0:
                    cute.arch.cp_async_bulk_wait_group(self.dz_store_stages - 1, read=True)
                self.dz_epi_bar.arrive_and_wait()
                for j in cutlass.range_constexpr(chunk):
                    sZ[row_in_tile, j, stage] = BFloat16(vals[j])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.dz_epi_bar.arrive_and_wait()
                if local_warp == 0:
                    cute.copy(
                        tma_dzs,
                        tZsZ[(None, stage)],
                        tZgZ[(None, pid_m, pid_n * self.dz_subtiles + sub, 0)],
                    )
                    cute.arch.cp_async_bulk_commit_group()

            self.dz_epi_bar.arrive_and_wait()
            if tid == 0:
                tile_pipe.consumer_release(rstate)
            rstate.advance()
            cr_i = rstate.index
            cr_p = rstate.phase
            work_idx += num_clusters

        # Drain the wave workspace stores before the caller's grid barrier.
        if local_warp == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        self.dz_epi_bar.arrive_and_wait()
        if local_warp == 0:
            cute.arch.cp_async_bulk_wait_group(0)

        return cr_i, cr_p

    @cute.jit
    def _dz_mainloop(
        self,
        wave: Int32,
        tid: Int32,
        warp: Int32,
        cluster_idx: Int32,
        cta_rank: Int32,
        num_clusters: Int32,
        dz_tiles: Int32,
        dz_group: Int32,
        num_k: Int32,
        num_k_blocks: cutlass.Constexpr,
        tma_x: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tma_w: cute.CopyAtom,
        tWgW: cute.Tensor,
        tWsW: cute.Tensor,
        sL: cute.Tensor,
        mma: cute.TiledMma,
        rA: cute.Tensor,
        rB: cute.Tensor,
        accum: cute.Tensor,
        row_base_in_tile: Int32,
        col_base_in_tile: Int32,
        pipe: pipeline.PipelineAsync,
        tile_pipe: pipeline.PipelineAsync,
        ld_i: Int32,
        ld_p: Int32,
        rd_i: Int32,
        rd_p: Int32,
        rl_i: Int32,
        rl_p: Int32,
        wr_i: Int32,
        wr_p: Int32,
    ):
        m_off = wave * Int32(self.m_tiles_per_wave)
        n_accum = cute.size(accum)

        load_i = Int32(0)
        load_k = Int32(0)
        load_pid_m, load_pid_n = self._dz_decode_work(Int32(cluster_idx), dz_group, cta_rank)

        if warp == 4:
            prologue = cutlass.min(Int32(self.stages), dz_tiles * Int32(num_k))
            for _ in range(prologue):
                pstate = pipeline.PipelineState(self.stages, Int32(0), ld_i, ld_p)
                pipe.producer_acquire(pstate)
                bar = pipe.producer_get_barrier(pstate)
                cute.copy(
                    tma_x,
                    tXgX[(None, m_off + load_pid_m, None, 0)][(None, load_k)],
                    tXsX[(None, ld_i)],
                    tma_bar_ptr=bar,
                    mcast_mask=0,
                )
                cute.copy(
                    tma_w,
                    tWgW[(None, load_pid_n, None, 0)][(None, load_k)],
                    tWsW[(None, ld_i)],
                    tma_bar_ptr=bar,
                )
                pipe.producer_commit(pstate)
                pstate.advance()
                ld_i = pstate.index
                ld_p = pstate.phase
                load_k += 1
                if load_k == num_k:
                    load_k = Int32(0)
                    load_i += 1
                    load_pid_m, load_pid_n = self._dz_decode_work(
                        Int32(cluster_idx) + load_i * num_clusters, dz_group, cta_rank
                    )

        for _ in range(dz_tiles):
            mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
            pipe.consumer_wait(rstate)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, rd_i)
                cute.gemm(mma, accum, rA[coord], rB[coord], accum)
                mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            rstate.advance()
            rd_i = rstate.index
            rd_p = rstate.phase

            for _ in range(1, num_k):
                rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
                pipe.consumer_wait(rstate)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    coord = (None, None, kb, rd_i)
                    cute.gemm(mma, accum, rA[coord], rB[coord], accum)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(1)
                lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
                pipe.consumer_release(lstate)
                lstate.advance()
                rl_i = lstate.index
                rl_p = lstate.phase
                if warp == 4:
                    if load_i < dz_tiles:
                        pstate = pipeline.PipelineState(self.stages, Int32(0), ld_i, ld_p)
                        pipe.producer_acquire(pstate)
                        bar = pipe.producer_get_barrier(pstate)
                        cute.copy(
                            tma_x,
                            tXgX[(None, m_off + load_pid_m, None, 0)][(None, load_k)],
                            tXsX[(None, ld_i)],
                            tma_bar_ptr=bar,
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_w,
                            tWgW[(None, load_pid_n, None, 0)][(None, load_k)],
                            tWsW[(None, ld_i)],
                            tma_bar_ptr=bar,
                        )
                        pipe.producer_commit(pstate)
                        pstate.advance()
                        ld_i = pstate.index
                        ld_p = pstate.phase
                        load_k += 1
                        if load_k == num_k:
                            load_k = Int32(0)
                            load_i += 1
                            load_pid_m, load_pid_n = self._dz_decode_work(
                                Int32(cluster_idx) + load_i * num_clusters, dz_group, cta_rank
                            )
                rstate.advance()
                rd_i = rstate.index
                rd_p = rstate.phase

            cute.nvgpu.warpgroup.wait_group(0)
            lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
            pipe.consumer_release(lstate)
            lstate.advance()
            rl_i = lstate.index
            rl_p = lstate.phase
            if warp == 4:
                if load_i < dz_tiles:
                    pstate = pipeline.PipelineState(self.stages, Int32(0), ld_i, ld_p)
                    pipe.producer_acquire(pstate)
                    bar = pipe.producer_get_barrier(pstate)
                    cute.copy(
                        tma_x,
                        tXgX[(None, m_off + load_pid_m, None, 0)][(None, load_k)],
                        tXsX[(None, ld_i)],
                        tma_bar_ptr=bar,
                        mcast_mask=0,
                    )
                    cute.copy(
                        tma_w,
                        tWgW[(None, load_pid_n, None, 0)][(None, load_k)],
                        tWsW[(None, ld_i)],
                        tma_bar_ptr=bar,
                    )
                    pipe.producer_commit(pstate)
                    pstate.advance()
                    ld_i = pstate.index
                    ld_p = pstate.phase
                    load_k += 1
                    if load_k == num_k:
                        load_k = Int32(0)
                        load_i += 1
                        load_pid_m, load_pid_n = self._dz_decode_work(
                            Int32(cluster_idx) + load_i * num_clusters, dz_group, cta_rank
                        )

            wstate = pipeline.PipelineState(1, Int32(0), wr_i, wr_p)
            tile_pipe.producer_acquire(wstate)
            for i in cutlass.range_constexpr(n_accum):
                r = row_base_in_tile + ((i % 4) // 2) * 8
                c = col_base_in_tile + (i // 4) * 8 + (i % 2)
                sL[r, c] = self.logit_dtype(accum[i])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.fence_acq_rel_cta()
            self.dz_mma_bar.arrive_and_wait()
            if tid == 128:
                tile_pipe.producer_commit(wstate)
            wstate.advance()
            wr_i = wstate.index
            wr_p = wstate.phase

        return ld_i, ld_p, rd_i, rd_p, rl_i, rl_p, wr_i, wr_p

    # ------------------------------------------------------------------ #
    # phase 2: dX = dZ @ W                              (one token wave)
    #
    # ``_dx_producer`` and ``_dx_consumer`` are the WG0 and WG1/WG2 halves of
    # ``_ScaledCEDxGemmSM90.kernel_coop`` (the latter being its ``_mainloop``
    # followed by its ``_epilogue``), persistent ``while unit <`` loop
    # included.  W is multicast over the cluster pair exactly as in the
    # accepted kernel.  Only the C store element type is BF16, not FP32.
    # ------------------------------------------------------------------ #
    @cute.jit
    def _dx_producer(
        self,
        warp: Int32,
        cluster_idx: Int32,
        cta_rank: Int32,
        num_clusters: Int32,
        b_mcast_mask: Int32,
        m_tiles: cutlass.Constexpr,
        m_units: cutlass.Constexpr,
        n_units: Int32,
        num_k: Int32,
        tma_a: cute.CopyAtom,
        tAgA: cute.Tensor,
        tAsA: cute.Tensor,
        tma_b: cute.CopyAtom,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
        pipe: pipeline.PipelineAsync,
        ld_i: Int32,
        ld_p: Int32,
    ):
        if warp == 0:
            units = m_units * n_units
            unit = Int32(cluster_idx)
            while unit < units:
                m_unit, n_unit = self._dx_tile_coord(unit, m_units, n_units)
                m_tile = cutlass.min(m_unit * self.CLUSTER_M + cta_rank, m_tiles - 1)
                tA_m = tAgA[(None, m_tile, None, 0)]
                tB_n = tBgB[(None, n_unit, None, 0)]
                for k in range(num_k):
                    pstate = pipeline.PipelineState(self.stages, Int32(0), ld_i, ld_p)
                    pipe.producer_acquire(pstate)
                    bar = pipe.producer_get_barrier(pstate)
                    cute.copy(
                        tma_a,
                        tA_m[(None, k)],
                        tAsA[(None, ld_i)],
                        tma_bar_ptr=bar,
                        mcast_mask=0,
                    )
                    cute.copy(
                        tma_b,
                        tB_n[(None, k)],
                        tBsB[(None, ld_i)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_mcast_mask,
                    )
                    pipe.producer_commit(pstate)
                    pstate.advance()
                    ld_i = pstate.index
                    ld_p = pstate.phase
                unit += num_clusters

        return ld_i, ld_p

    @cute.jit
    def _dx_consumer(
        self,
        wave: Int32,
        issuer: cutlass.Boolean,
        cluster_idx: Int32,
        cta_rank: Int32,
        num_clusters: Int32,
        m_tiles: cutlass.Constexpr,
        m_units: cutlass.Constexpr,
        n_units: Int32,
        num_k: Int32,
        num_k_blocks: cutlass.Constexpr,
        tma_c: cute.CopyAtom,
        tCgC: cute.Tensor,
        tCsC: cute.Tensor,
        tCsC_frag: cute.Tensor,
        mma: cute.TiledMma,
        rA: cute.Tensor,
        rB: cute.Tensor,
        accum: cute.Tensor,
        c_frag: cute.Tensor,
        pipe: pipeline.PipelineAsync,
        rd_i: Int32,
        rd_p: Int32,
        rl_i: Int32,
        rl_p: Int32,
    ):
        m_off = wave * Int32(self.m_tiles_per_wave)
        units = m_units * n_units
        c_size = cute.size(c_frag)

        unit = Int32(cluster_idx)
        while unit < units:
            m_unit, n_unit = self._dx_tile_coord(unit, m_units, n_units)
            m_raw = m_unit * self.CLUSTER_M + cta_rank
            store_enabled = m_raw < m_tiles
            m_tile = cutlass.min(m_raw, m_tiles - 1)
            accum.fill(0.0)
            mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
            pipe.consumer_wait(rstate)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, rd_i)
                cute.gemm(mma, accum, rA[coord], rB[coord], accum)
            cute.nvgpu.warpgroup.commit_group()
            rstate.advance()
            rd_i = rstate.index
            rd_p = rstate.phase

            for _ in range(1, num_k):
                rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
                pipe.consumer_wait(rstate)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    coord = (None, None, kb, rd_i)
                    cute.gemm(mma, accum, rA[coord], rB[coord], accum)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(1)
                lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
                pipe.consumer_release(lstate)
                lstate.advance()
                rl_i = lstate.index
                rl_p = lstate.phase
                rstate.advance()
                rd_i = rstate.index
                rd_p = rstate.phase

            cute.nvgpu.warpgroup.wait_group(0)
            lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
            pipe.consumer_release(lstate)
            lstate.advance()
            rl_i = lstate.index
            rl_p = lstate.phase

            for sub in cutlass.range_constexpr(self.dx_subtiles):
                stage = sub % self.dx_store_stages
                if cutlass.const_expr(sub >= self.dx_store_stages):
                    if issuer:
                        cute.arch.cp_async_bulk_wait_group(self.dx_store_stages - 1, read=True)
                    self.dx_epi_bar.arrive_and_wait()
                for j in cutlass.range_constexpr(c_size):
                    c_frag[j] = BFloat16(accum[self._sub_src(self.dx_store_tile_n, self.tile_n, sub, j)])
                cute.autovec_copy(c_frag, tCsC_frag[(None, None, None, stage)])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.dx_epi_bar.arrive_and_wait()
                if issuer:
                    if store_enabled:
                        cute.copy(
                            tma_c,
                            tCsC[(None, stage)],
                            tCgC[(None, m_off + m_tile, n_unit * self.dx_subtiles + sub, 0)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
            if issuer:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            self.dx_epi_bar.arrive_and_wait()
            unit += num_clusters

        return rd_i, rd_p, rl_i, rl_p

    # ------------------------------------------------------------------ #
    # phase 3: dW += dZ^T @ X                           (one token wave)
    #
    # ``_dw_producer`` and ``_dw_consumer`` are the WG0 and WG1/WG2 halves of
    # ``_ScaledCEDwGemmSM90.kernel``: the transposed ``order=(1, 0, 2)``
    # operands and the epilogue that TMA reduce-adds BF16 into the persistent
    # HBM accumulator.  The cluster is ignored, exactly as in the accepted
    # ``cluster_m = 1`` kernel -- the two CTAs claim independent output tiles.
    # ------------------------------------------------------------------ #
    @cute.jit
    def _dw_producer(
        self,
        wave: Int32,
        warp: Int32,
        bidx: Int32,
        num_ctas: Int32,
        iters: Int32,
        total: Int32,
        m_tiles: Int32,
        n_tiles: Int32,
        num_k: Int32,
        tma_a: cute.CopyAtom,
        tPgP: cute.Tensor,
        tPsP: cute.Tensor,
        tma_b: cute.CopyAtom,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        pipe: pipeline.PipelineAsync,
        ld_i: Int32,
        ld_p: Int32,
    ):
        k_off = wave * num_k
        if warp == 0:
            for it in cutlass.range(iters, unroll=1):
                tile_id = it * num_ctas + bidx
                if tile_id < total:
                    m_tile, n_tile = self._dw_tile_coord(tile_id, m_tiles, n_tiles)
                    for k in cutlass.range(num_k, unroll=1):
                        pstate = pipeline.PipelineState(self.stages, Int32(0), ld_i, ld_p)
                        pipe.producer_acquire(pstate)
                        bar = pipe.producer_get_barrier(pstate)
                        cute.copy(
                            tma_a,
                            tPgP[(None, m_tile, k, 0)],
                            tPsP[(None, ld_i)],
                            tma_bar_ptr=bar,
                            mcast_mask=0,
                        )
                        cute.copy(
                            tma_b,
                            tQgQ[(None, n_tile, k_off + k, 0)],
                            tQsQ[(None, ld_i)],
                            tma_bar_ptr=bar,
                            mcast_mask=0,
                        )
                        pipe.producer_commit(pstate)
                        pstate.advance()
                        ld_i = pstate.index
                        ld_p = pstate.phase

        return ld_i, ld_p

    @cute.jit
    def _dw_consumer(
        self,
        wave: Int32,
        issuer: cutlass.Boolean,
        bidx: Int32,
        num_ctas: Int32,
        iters: Int32,
        total: Int32,
        m_tiles: Int32,
        n_tiles: Int32,
        num_k: Int32,
        num_k_blocks: cutlass.Constexpr,
        tma_d_store: cute.CopyAtom,
        tDgDStore: cute.Tensor,
        tDsDStore: cute.Tensor,
        tma_d_add: cute.CopyAtom,
        tDgDAdd: cute.Tensor,
        tDsDAdd: cute.Tensor,
        tDsD_frag: cute.Tensor,
        mma: cute.TiledMma,
        rA: cute.Tensor,
        rB: cute.Tensor,
        accum: cute.Tensor,
        d_frag: cute.Tensor,
        pipe: pipeline.PipelineAsync,
        rd_i: Int32,
        rd_p: Int32,
        rl_i: Int32,
        rl_p: Int32,
    ):
        d_size = cute.size(d_frag)

        for it in cutlass.range(iters, unroll=1):
            tile_id = it * num_ctas + bidx
            if tile_id < total:
                m_tile, n_tile = self._dw_tile_coord(tile_id, m_tiles, n_tiles)
                accum.fill(0.0)
                mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
                pipe.consumer_wait(rstate)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    coord = (None, None, kb, rd_i)
                    cute.gemm(mma, accum, rA[coord], rB[coord], accum)
                cute.nvgpu.warpgroup.commit_group()
                rstate.advance()
                rd_i = rstate.index
                rd_p = rstate.phase

                for _ in cutlass.range(1, num_k, unroll=1):
                    rstate = pipeline.PipelineState(self.stages, Int32(0), rd_i, rd_p)
                    pipe.consumer_wait(rstate)
                    cute.nvgpu.warpgroup.fence()
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        coord = (None, None, kb, rd_i)
                        cute.gemm(mma, accum, rA[coord], rB[coord], accum)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                    lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
                    pipe.consumer_release(lstate)
                    lstate.advance()
                    rl_i = lstate.index
                    rl_p = lstate.phase
                    rstate.advance()
                    rd_i = rstate.index
                    rd_p = rstate.phase

                cute.nvgpu.warpgroup.wait_group(0)
                lstate = pipeline.PipelineState(self.stages, Int32(0), rl_i, rl_p)
                pipe.consumer_release(lstate)
                lstate.advance()
                rl_i = lstate.index
                rl_p = lstate.phase

                for sub in cutlass.range_constexpr(self.dw_subtiles):
                    stage = sub % self.dw_store_stages
                    if issuer:
                        cute.arch.cp_async_bulk_wait_group(self.dw_store_stages - 1, read=True)
                    self.dw_store_bar.arrive_and_wait()
                    for j in cutlass.range_constexpr(d_size):
                        d_frag[j] = BFloat16(accum[self._sub_src(self.dw_store_tile_n, self.tile_n, sub, j)])
                    cute.autovec_copy(d_frag, tDsD_frag[(None, None, None, stage)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.dw_store_bar.arrive_and_wait()
                    if issuer:
                        if wave == 0:
                            cute.copy(
                                tma_d_store,
                                tDsDStore[(None, stage)],
                                tDgDStore[(None, m_tile, n_tile * self.dw_subtiles + sub, 0)],
                            )
                        else:
                            cute.copy(
                                tma_d_add,
                                tDsDAdd[(None, stage)],
                                tDgDAdd[(None, m_tile, n_tile * self.dw_subtiles + sub, 0)],
                            )
                        cute.arch.cp_async_bulk_commit_group()

        # Drain every dX and dW store before the caller's grid barrier.  The
        # read-drain and named barrier are unconditional: they free the store
        # SMEM, which the other GEMM phase aliases.
        if issuer:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        self.dw_store_bar.arrive_and_wait()
        if issuer:
            cute.arch.cp_async_bulk_wait_group(0)

        return rd_i, rd_p, rl_i, rl_p

    # ------------------------------------------------------------------ #
    # the one kernel: setup, then the host wave loop moved on-device
    # ------------------------------------------------------------------ #
    @cute.kernel
    def kernel(
        self,
        tma_x: cute.CopyAtom,
        x: cute.Tensor,
        tma_w: cute.CopyAtom,
        w: cute.Tensor,
        tma_dzs: cute.CopyAtom,
        dzs: cute.Tensor,
        tma_dza: cute.CopyAtom,
        dza: cute.Tensor,
        tma_wt: cute.CopyAtom,
        wt: cute.Tensor,
        tma_dx: cute.CopyAtom,
        dx: cute.Tensor,
        tma_dwa: cute.CopyAtom,
        dwa: cute.Tensor,
        tma_xt: cute.CopyAtom,
        xt: cute.Tensor,
        tma_dw_store: cute.CopyAtom,
        dw_store: cute.Tensor,
        tma_dw_add: cute.CopyAtom,
        dw_add: cute.Tensor,
        target: cute.Tensor,
        lse: cute.Tensor,
        scale: cute.Tensor,
        entropy: cute.Tensor,
        entropy_scale: cute.Tensor,
        inverse_temperature: Float32,
        barrier: cute.Tensor,
        mma_dz: cute.TiledMma,
        mma_dx: cute.TiledMma,
        mma_dw: cute.TiledMma,
        smma_dx: cute.TiledMma,
        smma_dw: cute.TiledMma,
        dz_a: cute.ComposedLayout,
        dz_b: cute.ComposedLayout,
        dz_st: cute.ComposedLayout,
        dx_a: cute.ComposedLayout,
        dx_b: cute.ComposedLayout,
        dx_st: cute.ComposedLayout,
        dw_a: cute.ComposedLayout,
        dw_b: cute.ComposedLayout,
        dw_st: cute.ComposedLayout,
        ignore_index: Int32,
        vocab_size: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = cute.arch.make_warp_uniform(tid // 128)
        lane = tid % 32
        local_warp = (tid % 128) // 32
        local_tid = tid % 128
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        num_ctas = grid_x
        num_clusters = grid_x // self.CLUSTER_M
        cluster_idx = bidx // self.CLUSTER_M

        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dzs)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dza)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_wt)
        if warp == 4:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dx)
        if warp == 8:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dwa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_xt)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dw_store)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_dw_add)

        # ---- SMEM: disjoint mbarriers, then one phase-serial union -------
        smem = cutlass.utils.SmemAllocator()
        bars = smem.allocate(self.bar_storage)
        union_ptr = smem.allocate(self.union_bytes, self.buffer_align)
        dzmem = self.dz_storage(union_ptr)
        dxmem = self.dx_storage(union_ptr)
        dwmem = self.dw_storage(union_ptr)

        sX = dzmem.sX.get_tensor(dz_a.outer, swizzle=dz_a.inner)
        sW = dzmem.sW.get_tensor(dz_b.outer, swizzle=dz_b.inner)
        sL = dzmem.sL.get_tensor(cute.make_layout((self.tile_m, self.tile_n), stride=(self.logit_stride, 1)))
        sZ = dzmem.sZ.get_tensor(dz_st.outer, swizzle=dz_st.inner)
        sXA = dxmem.sA.get_tensor(dx_a.outer, swizzle=dx_a.inner)
        sXB = dxmem.sB.get_tensor(dx_b.outer, swizzle=dx_b.inner)
        sXC = dxmem.sC.get_tensor(dx_st.outer, swizzle=dx_st.inner)
        sWA = dwmem.sA.get_tensor(dw_a.outer, swizzle=dw_a.inner)
        sWB = dwmem.sB.get_tensor(dw_b.outer, swizzle=dw_b.inner)
        sWD = dwmem.sD.get_tensor(dw_st.outer, swizzle=dw_st.inner)

        # ---- global tiles ------------------------------------------------
        gX = cute.local_tile(x, cute.slice_(self.tile_shape, (None, 0, None)), (None, None, None))
        gW = cute.local_tile(w, cute.slice_(self.tile_shape, (0, None, None)), (None, None, None))
        gZS = cute.local_tile(dzs, (self.tile_m, self.dz_store_tile_n), (None, None, None))
        gZA = cute.local_tile(dza, cute.slice_(self.tile_shape, (None, 0, None)), (None, None, None))
        gWT = cute.local_tile(wt, cute.slice_(self.tile_shape, (0, None, None)), (None, None, None))
        gDX = cute.local_tile(dx, (self.tile_m, self.dx_store_tile_n), (None, None, None))
        gZT = cute.local_tile(dwa, cute.slice_(self.tile_shape, (None, 0, None)), (None, None, None))
        gXT = cute.local_tile(xt, cute.slice_(self.tile_shape, (0, None, None)), (None, None, None))
        gDW = cute.local_tile(dw_store, (self.tile_m, self.dw_store_tile_n), (None, None, None))
        gDWAdd = cute.local_tile(dw_add, (self.tile_m, self.dw_store_tile_n), (None, None, None))

        one = cute.make_layout(1)
        # dX's W multicast: mode 0 of the (2, 1, 1) cluster is M, so peers that
        # differ in m-tile share the n-tile and B is the multicast operand.
        cta_layout = cute.make_layout((self.CLUSTER_M, 1, 1))
        cluster_coord = cta_layout.get_flat_coord(cta_rank)
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout, (None, 0, 0)).shape)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout, cluster_coord, mode=0)
        b_mcast_rank = cluster_coord[0]

        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_x, 0, one, cute.group_modes(sX, 0, 2), cute.group_modes(gX, 0, 2)
        )
        tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
            tma_w, 0, one, cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2)
        )
        tZsZ, tZgZ = cute.nvgpu.cpasync.tma_partition(
            tma_dzs, 0, one, cute.group_modes(sZ, 0, 2), cute.group_modes(gZS, 0, 2)
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_dza, 0, one, cute.group_modes(sXA, 0, 2), cute.group_modes(gZA, 0, 2)
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_wt, b_mcast_rank, b_cta_layout, cute.group_modes(sXB, 0, 2), cute.group_modes(gWT, 0, 2)
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            tma_dx, 0, one, cute.group_modes(sXC, 0, 2), cute.group_modes(gDX, 0, 2)
        )
        tPsP, tPgP = cute.nvgpu.cpasync.tma_partition(
            tma_dwa, 0, one, cute.group_modes(sWA, 0, 2), cute.group_modes(gZT, 0, 2)
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_xt, 0, one, cute.group_modes(sWB, 0, 2), cute.group_modes(gXT, 0, 2)
        )
        tDsDStore, tDgDStore = cute.nvgpu.cpasync.tma_partition(
            tma_dw_store, 0, one, cute.group_modes(sWD, 0, 2), cute.group_modes(gDW, 0, 2)
        )
        tDsDAdd, tDgDAdd = cute.nvgpu.cpasync.tma_partition(
            tma_dw_add, 0, one, cute.group_modes(sWD, 0, 2), cute.group_modes(gDWAdd, 0, 2)
        )

        def _bytes(la, lb):
            return cute.size_in_bytes(BFloat16, cute.slice_(la, (None, None, 0))) + cute.size_in_bytes(
                BFloat16, cute.slice_(lb, (None, None, 0))
            )

        # ---- pipelines (mbarriers are disjoint per phase) -----------------
        dz_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=bars.dz_pipe.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP),
            tx_count=_bytes(dz_a, dz_b),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tid,
            defer_sync=True,
        )
        tile_pipe = pipeline.PipelineAsync.create(
            barrier_storage=bars.dz_tile.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            defer_sync=True,
        )
        dx_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=bars.dx_pipe.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                WARPS_PER_WARP_GROUP * NUM_MMA_WARP_GROUPS * self.CLUSTER_M,
            ),
            tx_count=_bytes(dx_a, dx_b),
            cta_layout_vmnk=cute.make_layout((1, self.CLUSTER_M, 1, 1)),
            tidx=tid,
            defer_sync=True,
        )
        dw_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=bars.dw_pipe.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP),
            tx_count=_bytes(dw_a, dw_b),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tid,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(self.CLUSTER_M, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(self.CLUSTER_M, 1))

        bar_ptr = barrier.iterator

        # ---- extents -----------------------------------------------------
        num_k_dz = cute.size(gX, mode=[3])  # Hp / 64
        num_k_dx = cute.size(gZA, mode=[3])  # Vp / 64
        num_k_dw = cute.size(gZT, mode=[3])  # wave_rows / 64
        num_k_blocks = cutlass.const_expr(self.tile_k // 16)

        dz_group = Int32(self.plan.group_size * self.plan.ncluster_slow)
        dz_total = Int32(self.plan.num_clusters)
        dz_tiles = (dz_total - Int32(cluster_idx) + num_clusters - 1) // num_clusters

        dx_m_tiles = cutlass.const_expr(self.m_tiles_per_wave)
        dx_m_units = cutlass.const_expr((self.m_tiles_per_wave + self.CLUSTER_M - 1) // self.CLUSTER_M)
        dx_n_units = cute.ceil_div(wt.shape[0], self.tile_n)

        dw_m_tiles = cute.ceil_div(dwa.shape[0], self.tile_m)
        dw_n_tiles = cute.ceil_div(xt.shape[0], self.tile_n)
        dw_total = dw_m_tiles * dw_n_tiles
        dw_iters = cute.ceil_div(dw_total, num_ctas)

        num_waves = cute.ceil_div(x.shape[0], self.wave_rows)

        if warp_group == 0:
            # =============================================================
            # WG0: dZ softmax epilogue, then the dX and dW TMA producers.
            # Everything loop-invariant is built once, before the wave loop,
            # exactly as it is in each accepted kernel's prologue.
            # =============================================================
            cute.arch.setmaxregister_decrease(self.epi_registers)

            vals = cute.make_rmem_tensor((self.dz_store_tile_n,), Float32)

            z_cr_i, z_cr_p = Int32(0), Int32(0)
            x_ld_i, x_ld_p = Int32(0), Int32(1)
            w_ld_i, w_ld_p = Int32(0), Int32(1)
            target_count = Int32(0)

            for wave in cutlass.range(num_waves, unroll=1):
                z_cr_i, z_cr_p = self._dz_epilogue(
                    wave,
                    tid,
                    local_warp,
                    local_tid,
                    cluster_idx,
                    cta_rank,
                    num_clusters,
                    dz_tiles,
                    dz_group,
                    tma_dzs,
                    tZgZ,
                    tZsZ,
                    sL,
                    sZ,
                    tile_pipe,
                    target,
                    lse,
                    scale,
                    entropy,
                    entropy_scale,
                    inverse_temperature,
                    vals,
                    ignore_index,
                    vocab_size,
                    z_cr_i,
                    z_cr_p,
                )

                # The wave workspace must be visible to every CTA before dX
                # reads it.
                target_count = target_count + num_ctas
                self._grid_barrier(bar_ptr, target_count, tid)
                self._regs_for_gemm(True)

                x_ld_i, x_ld_p = self._dx_producer(
                    warp,
                    cluster_idx,
                    cta_rank,
                    num_clusters,
                    b_mcast_mask,
                    dx_m_tiles,
                    dx_m_units,
                    dx_n_units,
                    num_k_dx,
                    tma_dza,
                    tAgA,
                    tAsA,
                    tma_wt,
                    tBgB,
                    tBsB,
                    dx_pipe,
                    x_ld_i,
                    x_ld_p,
                )
                cute.arch.sync_threads()
                w_ld_i, w_ld_p = self._dw_producer(
                    wave,
                    warp,
                    bidx,
                    num_ctas,
                    dw_iters,
                    dw_total,
                    dw_m_tiles,
                    dw_n_tiles,
                    num_k_dw,
                    tma_dwa,
                    tPgP,
                    tPsP,
                    tma_xt,
                    tQgQ,
                    tQsQ,
                    dw_pipe,
                    w_ld_i,
                    w_ld_p,
                )

                # The final wave cannot overwrite the workspace again, and
                # kernel completion already orders its output stores.
                if wave + 1 < num_waves:
                    self._regs_for_dz(True)
                    target_count = target_count + num_ctas
                    self._grid_barrier(bar_ptr, target_count, tid)
        else:
            # =============================================================
            # WG1 + WG2: the dZ mainloop, then the dX and dW consumers.
            # =============================================================
            cute.arch.setmaxregister_increase(self.mma_registers)

            mma_wg = warp_group - 1
            mma_tid = tid - 128
            wg_layout = cute.make_layout(NUM_MMA_WARP_GROUPS, stride=128)
            thr_dz = mma_dz.get_slice(wg_layout(Int32(mma_wg)))
            thr_dx = mma_dx.get_slice(mma_tid)
            thr_dw = mma_dw.get_slice(mma_tid)
            thr_sx = smma_dx.get_slice(mma_tid)
            thr_sw = smma_dw.get_slice(mma_tid)

            rZA = mma_dz.make_fragment_A(thr_dz.partition_A(sX))
            rZB = mma_dz.make_fragment_B(thr_dz.partition_B(sW))
            rXA = mma_dx.make_fragment_A(thr_dx.partition_A(sXA))
            rXB = mma_dx.make_fragment_B(thr_dx.partition_B(sXB))
            rWA = mma_dw.make_fragment_A(thr_dw.partition_A(sWA))
            rWB = mma_dw.make_fragment_B(thr_dw.partition_B(sWB))

            # ONE accumulator for all three phases: the three tiled MMAs have
            # the same atom layout over the same tile, so ``partition_shape_C``
            # is identical, and the phases are serial, so the live ranges are
            # provably disjoint.  Three separate accumulators cost ptxas a
            # 25 M-sector local-memory spill.
            accum = cute.make_rmem_tensor(thr_dz.partition_shape_C((self.tile_m, self.tile_n)), Float32)

            tCsC_frag = thr_sx.partition_C(sXC)
            tDsD_frag = thr_sw.partition_C(sWD)
            c_frag = cute.make_rmem_tensor(thr_sx.partition_shape_C((self.tile_m, self.dx_store_tile_n)), BFloat16)
            # dX and dW never stage at the same time; when their store tiles
            # are the same width the fragment is literally the same registers.
            if cutlass.const_expr(self.dw_store_tile_n == self.dx_store_tile_n):
                d_frag = c_frag
            else:
                d_frag = cute.make_rmem_tensor(thr_sw.partition_shape_C((self.tile_m, self.dw_store_tile_n)), BFloat16)

            row_base_in_tile = mma_wg * 64 + local_warp * 16 + lane // 4
            col_base_in_tile = (lane % 4) * 2
            issuer = warp == 4

            z_ld_i, z_ld_p = Int32(0), Int32(1)
            z_rd_i, z_rd_p = Int32(0), Int32(0)
            z_rl_i, z_rl_p = Int32(0), Int32(0)
            z_wr_i, z_wr_p = Int32(0), Int32(1)
            x_rd_i, x_rd_p = Int32(0), Int32(0)
            x_rl_i, x_rl_p = Int32(0), Int32(0)
            w_rd_i, w_rd_p = Int32(0), Int32(0)
            w_rl_i, w_rl_p = Int32(0), Int32(0)
            target_count = Int32(0)

            for wave in cutlass.range(num_waves, unroll=1):
                (
                    z_ld_i,
                    z_ld_p,
                    z_rd_i,
                    z_rd_p,
                    z_rl_i,
                    z_rl_p,
                    z_wr_i,
                    z_wr_p,
                ) = self._dz_mainloop(
                    wave,
                    tid,
                    warp,
                    cluster_idx,
                    cta_rank,
                    num_clusters,
                    dz_tiles,
                    dz_group,
                    num_k_dz,
                    num_k_blocks,
                    tma_x,
                    tXgX,
                    tXsX,
                    tma_w,
                    tWgW,
                    tWsW,
                    sL,
                    mma_dz,
                    rZA,
                    rZB,
                    accum,
                    row_base_in_tile,
                    col_base_in_tile,
                    dz_pipe,
                    tile_pipe,
                    z_ld_i,
                    z_ld_p,
                    z_rd_i,
                    z_rd_p,
                    z_rl_i,
                    z_rl_p,
                    z_wr_i,
                    z_wr_p,
                )

                target_count = target_count + num_ctas
                self._grid_barrier(bar_ptr, target_count, tid)
                self._regs_for_gemm(False)

                (x_rd_i, x_rd_p, x_rl_i, x_rl_p) = self._dx_consumer(
                    wave,
                    issuer,
                    cluster_idx,
                    cta_rank,
                    num_clusters,
                    dx_m_tiles,
                    dx_m_units,
                    dx_n_units,
                    num_k_dx,
                    num_k_blocks,
                    tma_dx,
                    tCgC,
                    tCsC,
                    tCsC_frag,
                    mma_dx,
                    rXA,
                    rXB,
                    accum,
                    c_frag,
                    dx_pipe,
                    x_rd_i,
                    x_rd_p,
                    x_rl_i,
                    x_rl_p,
                )
                cute.arch.sync_threads()
                (w_rd_i, w_rd_p, w_rl_i, w_rl_p) = self._dw_consumer(
                    wave,
                    issuer,
                    bidx,
                    num_ctas,
                    dw_iters,
                    dw_total,
                    dw_m_tiles,
                    dw_n_tiles,
                    num_k_dw,
                    num_k_blocks,
                    tma_dw_store,
                    tDgDStore,
                    tDsDStore,
                    tma_dw_add,
                    tDgDAdd,
                    tDsDAdd,
                    tDsD_frag,
                    mma_dw,
                    rWA,
                    rWB,
                    accum,
                    d_frag,
                    dw_pipe,
                    w_rd_i,
                    w_rd_p,
                    w_rl_i,
                    w_rl_p,
                )

                if wave + 1 < num_waves:
                    self._regs_for_dz(False)
                    target_count = target_count + num_ctas
                    self._grid_barrier(bar_ptr, target_count, tid)


# ---------------------------------------------------------------------------
# Host entry point
# ---------------------------------------------------------------------------


def _pad_rows(t, rows):
    if t.shape[0] == rows:
        return t
    return torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[0]))


def fused_backward_one_kernel(
    grad_output,
    x_padded,
    weight_padded,
    target,
    lse,
    entropy=None,
    entropy_grad=None,
    ignore_index=-100,
    hidden_size=None,
    input_dtype=torch.bfloat16,
    temperature=1.0,
    config: FusedBackwardConfig = None,
    max_clusters=None,
):
    """One-launch dZ+dX+dW backward.  Returns ``(grad_input, grad_weight)``.

    Mirrors :func:`fused_scaled_cross_entropy_backward`'s token-wave semantics
    exactly (BF16 dZ wave workspace, BF16 HBM dW accumulator with TMA
    reduce-add), but performs every wave inside a single persistent kernel.
    """
    if config is None:
        config = FusedBackwardConfig()
    config.validate()
    ensure_cuda_context()

    if torch.cuda.get_device_capability(x_padded.device) != (9, 0):
        raise RuntimeError("the fused one-kernel SM90 backward requires a Hopper GPU")
    if x_padded.dtype != torch.bfloat16 or weight_padded.dtype != torch.bfloat16:
        raise TypeError("input and weight must be BF16")

    device = x_padded.device
    tokens, hidden_padded = x_padded.shape
    if hidden_size is None:
        hidden_size = hidden_padded
    if hidden_padded % HIDDEN_TILE != 0:
        raise ValueError("x_padded's hidden dimension must already be padded to a multiple of 64")
    vocab_size = weight_padded.shape[0]
    vocab_padded = (vocab_size + VOCAB_ALIGN - 1) // VOCAB_ALIGN * VOCAB_ALIGN
    weight_for_dx = _pad_rows(weight_padded.contiguous(), vocab_padded)

    scale = grad_output.detach().to(torch.float32).contiguous()
    with_entropy = entropy_grad is not None
    if with_entropy:
        if entropy is None or entropy.shape != (tokens,) or entropy_grad.shape != (tokens,):
            raise ValueError("entropy and entropy_grad must both have shape [M]")
        entropy = entropy.contiguous()
        entropy_scale = entropy_grad.detach().to(torch.float32).contiguous()
    else:
        entropy = torch.empty(1, device=device, dtype=torch.bfloat16)
        entropy_scale = torch.empty(1, device=device, dtype=torch.float32)
    if config.with_entropy != with_entropy:
        config = FusedBackwardConfig(**{**config.__dict__, "with_entropy": with_entropy})
    target = target.contiguous()
    lse = lse.contiguous()

    wave_rows = min(config.wave_rows, ((tokens + config.tile_m - 1) // config.tile_m) * config.tile_m)
    wave_rows = max(wave_rows, config.tile_m)
    m_tiles_per_wave = wave_rows // config.tile_m
    if m_tiles_per_wave != config.m_tiles_per_wave:
        config = FusedBackwardConfig(**{**config.__dict__, "m_tiles_per_wave": m_tiles_per_wave})

    dz_ws = torch.empty((wave_rows, vocab_padded), device=device, dtype=torch.bfloat16)
    grad_input_padded = torch.empty((tokens, hidden_padded), device=device, dtype=torch.bfloat16)
    grad_weight_padded = torch.empty((vocab_padded, hidden_padded), device=device, dtype=torch.bfloat16)

    num_m_tiles = m_tiles_per_wave
    num_n_tiles = (vocab_size + config.tile_n - 1) // config.tile_n
    plan = _plan_schedule(
        num_m_tiles, num_n_tiles, _FusedBackwardSM90.CLUSTER_M, config.dz_group_size, config.dz_raster
    )
    clusters = _max_active_clusters(_FusedBackwardSM90.CLUSTER_M)

    # Hard launch invariant: never launch more CTAs than there are physical SMs,
    # otherwise the grid is not fully resident and the software global barrier
    # deadlocks.  Also never launch more clusters than any phase can keep busy.
    sm_count = _sm_count()
    cluster_m = _FusedBackwardSM90.CLUSTER_M
    dz_clusters = plan.num_clusters
    dx_clusters = ((m_tiles_per_wave + cluster_m - 1) // cluster_m) * (
        (hidden_padded + config.tile_n - 1) // config.tile_n
    )
    dw_ctas = ((vocab_padded + config.tile_m - 1) // config.tile_m) * (
        (hidden_padded + config.tile_n - 1) // config.tile_n
    )
    dw_clusters = (dw_ctas + cluster_m - 1) // cluster_m
    required_work_clusters = max(dz_clusters, dx_clusters, dw_clusters)

    clusters = min(int(clusters), sm_count // cluster_m, required_work_clusters)
    if max_clusters is not None:
        clusters = min(clusters, max_clusters)
    clusters = int(clusters)
    if clusters < 1:
        raise RuntimeError("the fused one-kernel backward needs at least one resident cluster")
    grid_x = clusters * cluster_m
    if grid_x > sm_count:
        raise RuntimeError(
            f"fused backward would launch {grid_x} CTAs on {sm_count} SMs; "
            "the persistent grid must fit so the global barrier cannot deadlock"
        )

    x_c = to_cute_tensor(x_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    w_c = to_cute_tensor(weight_for_dx.unsqueeze(-1), leading_dim=1, assumed_align=16)
    wt_c = to_cute_tensor(weight_for_dx.transpose(0, 1).unsqueeze(-1), leading_dim=0, assumed_align=16)
    xt_c = to_cute_tensor(x_padded.transpose(0, 1).unsqueeze(-1), leading_dim=0, assumed_align=16)
    dz_c = to_cute_tensor(dz_ws.unsqueeze(-1), leading_dim=1, assumed_align=16)
    dzt_c = to_cute_tensor(dz_ws.transpose(0, 1).unsqueeze(-1), leading_dim=0, assumed_align=16)
    dx_c = to_cute_tensor(grad_input_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    dw_c = to_cute_tensor(grad_weight_padded.unsqueeze(-1), leading_dim=1, assumed_align=16)
    tgt_c = to_cute_tensor(target, assumed_align=8)
    lse_c = to_cute_tensor(lse, assumed_align=4)
    scale_c = to_cute_tensor(scale, assumed_align=4)
    entropy_c = to_cute_tensor(entropy, assumed_align=2)
    entropy_scale_c = to_cute_tensor(entropy_scale, assumed_align=4)
    inverse_temperature = Float32(1.0 / temperature)

    # A single int32 counter, zeroed for this launch.  Each device-side barrier
    # raises a CTA-uniform target by ``gridDim.x``, so no reset, no cross-launch
    # monotone base and no overflow handling are needed.
    barrier = torch.zeros(1, device=device, dtype=torch.int32)
    bar_c = to_cute_tensor(barrier, assumed_align=4)

    stream = _cute_stream()
    key = ("fused_bwd_one_kernel", config, plan, clusters)
    compiled = _compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _FusedBackwardSM90(config, plan),
            x_c,
            w_c,
            wt_c,
            xt_c,
            dz_c,
            dzt_c,
            dx_c,
            dw_c,
            tgt_c,
            lse_c,
            scale_c,
            entropy_c,
            entropy_scale_c,
            inverse_temperature,
            bar_c,
            Int32(ignore_index),
            Int32(vocab_size),
            clusters,
            stream,
        )
        _compile_cache[key] = compiled
    compiled(
        x_c,
        w_c,
        wt_c,
        xt_c,
        dz_c,
        dzt_c,
        dx_c,
        dw_c,
        tgt_c,
        lse_c,
        scale_c,
        entropy_c,
        entropy_scale_c,
        inverse_temperature,
        bar_c,
        Int32(ignore_index),
        Int32(vocab_size),
        stream,
    )

    del dz_c, dzt_c, dz_ws
    grad_input = grad_input_padded
    if hidden_size != hidden_padded:
        grad_input = grad_input_padded[:, :hidden_size].contiguous()
    if input_dtype != torch.bfloat16:
        grad_input = grad_input.to(input_dtype)
    grad_weight = grad_weight_padded
    if (vocab_size, hidden_size) != tuple(grad_weight_padded.shape):
        grad_weight = grad_weight_padded[:vocab_size, :hidden_size].contiguous()
    return grad_input, grad_weight


__all__ = [
    "FusedBackwardConfig",
    "fused_backward_one_kernel",
]
