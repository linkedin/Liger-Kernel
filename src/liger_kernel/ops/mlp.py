"""
Fused SwiGLU MLP: forward/backward orchestration + autograd.Function.

Shapes
------
input      : [B, S, dim]
gate_weight: [hidden_dim, dim]
up_weight  : [hidden_dim, dim]
down_weight: [dim, hidden_dim]

1. The forward kernel fuses gate_proj + up_proj + SiLU + gating into a single
Triton kernel (down_proj stays a plain cuBLAS GEMM).
2. The backward kernels fuse the SiLU/gating gradient (dG, dU) into one kernel,
and dI = dG @ Wg + dU @ Wu into another, avoiding an extra HBM round trip for the
intermediate dA tensor.
3. G and U are cached from the forward pass instead of being recomputed in
the backward pass (recompute-vs-store trade-off).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor

from liger_kernel.ops.utils import device_context


@triton.jit
def _l2_swizzle(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    base_pid_m = group_id * GROUP_SIZE_M
    effective_group_size_m = tl.minimum(num_pid_m - base_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = pid_in_group % effective_group_size_m + base_pid_m
    pid_n = pid_in_group // effective_group_size_m
    return pid_m, pid_n


# =====================================================================
# Forward Training
# =====================================================================
def _host_descriptor_pre_hook_fwd(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    nargs["desc_input"].block_shape = [1, BLOCK_M, BLOCK_K]
    nargs["desc_Wg"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["desc_Wu"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["desc_A"].block_shape = [1, BLOCK_M, BLOCK_N]
    nargs["desc_G"].block_shape = [1, BLOCK_M, BLOCK_N]
    nargs["desc_U"].block_shape = [1, BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GS},
            num_warps=w,
            num_stages=s,
            pre_hook=_host_descriptor_pre_hook_fwd,
        )
        for BM in [64, 128]
        for BN in [128]
        for BK in [64, 128]
        for GS in [1, 8]  # GROUP_SIZE_M = 1 equal to no L2 swizzle.
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["dim", "hidden_dim", "bucket_M"],
)
# grid = [ceil(S/Block_M) * ceil(hidden_dim/Block_N), B]
@triton.jit
def _swiglu_kernel_forward(
    # ======= TensorDescriptor =======
    desc_input,
    desc_Wg,
    desc_Wu,
    desc_A,
    desc_G,
    desc_U,
    # ============ shape ============
    S,
    dim,
    hidden_dim,
    bucket_M,
    # =========== meta data ===========
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    gate_multiplier: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    # L2 swizzle
    pid_m, pid_n = _l2_swizzle(pid, S, hidden_dim, BLOCK_M, BLOCK_N, GROUP_SIZE_M)

    # Compute the starting positions for input, gate and up
    M_start = pid_m * BLOCK_M
    N_start = pid_n * BLOCK_N

    # Compute input @ gate^T and input @ up^T
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_K):
        input_block = desc_input.load([b, M_start, k])  # [1, BLOCK_M, BLOCK_K]
        input_block = tl.reshape(input_block, [BLOCK_M, BLOCK_K])  # [BLOCK_M, BLOCK_K]
        gate_block = desc_Wg.load([N_start, k])  # [BLOCK_N, BLOCK_K]
        acc_gate = tl.dot(input_block, tl.trans(gate_block), acc=acc_gate)
        up_block = desc_Wu.load([N_start, k])  # [BLOCK_N, BLOCK_K]
        acc_up = tl.dot(input_block, tl.trans(up_block), acc=acc_up)

    # Compute A_block
    # Compile-time branch elimination, no runtime overhead.
    if gate_multiplier == 1.0:
        A_block = acc_gate * tl.sigmoid(acc_gate) * acc_up  # [BLOCK_M, BLOCK_N]
    else:
        acc_gate = acc_gate * gate_multiplier
        A_block = acc_gate * tl.sigmoid(acc_gate) * acc_up  # [BLOCK_M, BLOCK_N]

    # Write back
    dtype = desc_input.dtype
    A_block = tl.reshape(A_block, [1, BLOCK_M, BLOCK_N])
    desc_A.store([b, M_start, N_start], A_block.to(dtype))
    G_block = tl.reshape(acc_gate, [1, BLOCK_M, BLOCK_N])
    desc_G.store([b, M_start, N_start], G_block.to(dtype))
    U_block = tl.reshape(acc_up, [1, BLOCK_M, BLOCK_N])
    desc_U.store([b, M_start, N_start], U_block.to(dtype))


# =====================================================================
# Forward Inference
# =====================================================================
""" No need to save GU during inference."""


def _host_descriptor_pre_hook_fwd_inference(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    nargs["desc_input"].block_shape = [1, BLOCK_M, BLOCK_K]
    nargs["desc_Wg"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["desc_Wu"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["desc_A"].block_shape = [1, BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GS},
            num_warps=w,
            num_stages=s,
            pre_hook=_host_descriptor_pre_hook_fwd_inference,
        )
        for BM in [32, 64]
        for BN in [128]
        for BK in [64, 128]
        for GS in [1, 4]  # GROUP_SIZE_M = 1 equal to no L2 swizzle.
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["dim", "hidden_dim", "bucket_M"],
)
# grid = [ceil(S/Block_M) * ceil(hidden_dim/Block_N), B]
@triton.jit
def _swiglu_kernel_forward_inference(
    # ======= TensorDescriptor =======
    desc_input,
    desc_Wg,
    desc_Wu,
    desc_A,
    # ============ shape ============
    S,
    dim,
    hidden_dim,
    bucket_M,
    # =========== meta data ===========
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    gate_multiplier: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    # L2 swizzle
    pid_m, pid_n = _l2_swizzle(pid, S, hidden_dim, BLOCK_M, BLOCK_N, GROUP_SIZE_M)

    # Compute the starting positions for input, gate and up
    M_start = pid_m * BLOCK_M
    N_start = pid_n * BLOCK_N

    # Compute input @ gate^T and input @ up^T
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_K):
        input_block = desc_input.load([b, M_start, k])  # [1, BLOCK_M, BLOCK_K]
        input_block = tl.reshape(input_block, [BLOCK_M, BLOCK_K])  # [BLOCK_M, BLOCK_K]
        gate_block = desc_Wg.load([N_start, k])  # [BLOCK_N, BLOCK_K]
        acc_gate = tl.dot(input_block, tl.trans(gate_block), acc=acc_gate)
        up_block = desc_Wu.load([N_start, k])  # [BLOCK_N, BLOCK_K]
        acc_up = tl.dot(input_block, tl.trans(up_block), acc=acc_up)

    # Compute A_block
    # Compile-time branch elimination, no runtime overhead.
    if gate_multiplier == 1.0:
        A_block = acc_gate * tl.sigmoid(acc_gate) * acc_up  # [BLOCK_M, BLOCK_N]
    else:
        acc_gate = acc_gate * gate_multiplier
        A_block = acc_gate * tl.sigmoid(acc_gate) * acc_up  # [BLOCK_M, BLOCK_N]

    # Write back
    dtype = desc_input.dtype
    A_block = tl.reshape(A_block, [1, BLOCK_M, BLOCK_N])
    desc_A.store([b, M_start, N_start], A_block.to(dtype))


# =====================================================================
# Backward dU and dG
# =====================================================================
"""
Note: 
- We don't use autotune here, 
- because autotune could break the in-place overwrite operation. 
- Instead, we directly use the optimal configuration for the forward kernel.
"""


# grid = [ceil(S/Block_M) * ceil(hidden_dim/Block_N), B]
@triton.jit
def _swiglu_kernel_backward_dGU(
    # ======= TensorDescriptor =======
    desc_dO,
    desc_Wd,
    desc_G,
    desc_U,
    desc_dG,
    desc_dU,
    # ============ shape ============
    S,
    dim,
    hidden_dim,
    # =========== meta data ===========
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    # L2 swizzle
    pid_m, pid_n = _l2_swizzle(pid, S, hidden_dim, BLOCK_M, BLOCK_N, GROUP_SIZE_M)

    # Compute the starting positions for input, gate and up
    M_start = pid_m * BLOCK_M
    N_start = pid_n * BLOCK_N

    acc_A = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_K):
        dO_block = desc_dO.load([b, M_start, k])  # [1, BLOCK_M, BLOCK_K]
        dO_block = tl.reshape(dO_block, [BLOCK_M, BLOCK_K])  # [BLOCK_M, BLOCK_K]
        down_block = desc_Wd.load([k, N_start])  # [BLOCK_K, BLOCK_N]
        acc_A = tl.dot(dO_block, down_block, acc=acc_A)  # [BLOCK_M, BLOCK_N]

    # Load G and U
    G_block = desc_G.load([b, M_start, N_start])  # [1, BLOCK_M, BLOCK_N]
    G_block = tl.reshape(G_block, [BLOCK_M, BLOCK_N])  # [BLOCK_M, BLOCK_N]
    U_block = desc_U.load([b, M_start, N_start])  # [1, BLOCK_M, BLOCK_N]
    U_block = tl.reshape(U_block, [BLOCK_M, BLOCK_N])  # [BLOCK_M, BLOCK_N]

    # Compute and save dU, dG
    # NOTE: on Falcon H1 path, the forward kernel already stores G pre-multiplied
    # by gate_multiplier, so do NOT multiply it again here.
    G_block = G_block.to(tl.float32)
    U_block = U_block.to(tl.float32)

    sigmoid_G = tl.sigmoid(G_block)
    silu_G = G_block * sigmoid_G
    dtype = desc_G.dtype

    if gate_multiplier == 1.0 and down_multiplier == 1.0:
        dU = acc_A * silu_G  # [BLOCK_M, BLOCK_N]
        dG = acc_A * U_block * (sigmoid_G + silu_G - sigmoid_G * silu_G)  # [BLOCK_M, BLOCK_N]
    else:
        dU = acc_A * down_multiplier * silu_G  # [BLOCK_M, BLOCK_N]
        dG = (
            acc_A * down_multiplier * U_block * (sigmoid_G + silu_G - sigmoid_G * silu_G) * gate_multiplier
        )  # [BLOCK_M, BLOCK_N]

    dU = tl.reshape(dU, [1, BLOCK_M, BLOCK_N])
    desc_dU.store([b, M_start, N_start], dU.to(dtype))
    dG = tl.reshape(dG, [1, BLOCK_M, BLOCK_N])
    desc_dG.store([b, M_start, N_start], dG.to(dtype))


# =====================================================================
# Backward dI
# =====================================================================
def _host_descriptor_pre_hook_bwd_dI(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    nargs["desc_dG"].block_shape = [1, BLOCK_M, BLOCK_K]
    nargs["desc_Wg"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["desc_dU"].block_shape = [1, BLOCK_M, BLOCK_K]
    nargs["desc_Wu"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["desc_dI"].block_shape = [1, BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GS},
            num_warps=w,
            num_stages=s,
            pre_hook=_host_descriptor_pre_hook_bwd_dI,
        )
        for BM in [64, 128]
        for BN in [128]
        for BK in [64, 128]
        for GS in [1, 8]  # GROUP_SIZE_M = 1 equal to no L2 swizzle.
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["dim", "hidden_dim", "bucket_M"],
)
# grid = [ceil(S/Block_M) * ceil(dim/Block_N), B]
@triton.jit
def _swiglu_kernel_backward_dI(
    # ======= TensorDescriptor =======
    desc_dG,
    desc_Wg,
    desc_dU,
    desc_Wu,
    desc_dI,
    # ============ shape ============
    S,
    dim,
    hidden_dim,
    bucket_M,
    # =========== meta data ===========
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    # L2 swizzle
    pid_m, pid_n = _l2_swizzle(pid, S, dim, BLOCK_M, BLOCK_N, GROUP_SIZE_M)

    # Compute the starting positions
    M_start = pid_m * BLOCK_M
    N_start = pid_n * BLOCK_N

    # Compute dI = dG @ gate_weight + dU @ up_weight
    # [B, S, hidden_dim] @ [hidden_dim, dim] + ... @ ... = [B, S, dim]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, hidden_dim, BLOCK_K):
        dG_block = desc_dG.load([b, M_start, k])  # [1, BLOCK_M, BLOCK_K]
        dG_block = tl.reshape(dG_block, [BLOCK_M, BLOCK_K])  # [BLOCK_M, BLOCK_K]
        gate_weight_block = desc_Wg.load([k, N_start])  # [BLOCK_K, BLOCK_N]
        acc = tl.dot(dG_block, gate_weight_block, acc=acc)

        dU_block = desc_dU.load([b, M_start, k])  # [1, BLOCK_M, BLOCK_K]
        dU_block = tl.reshape(dU_block, [BLOCK_M, BLOCK_K])  # [BLOCK_M, BLOCK_K]
        up_weight_block = desc_Wu.load([k, N_start])  # [BLOCK_K, BLOCK_N]
        acc = tl.dot(dU_block, up_weight_block, acc=acc)

    # Save dI
    dI = acc  # [BLOCK_M, BLOCK_N]
    dI = tl.reshape(dI, [1, BLOCK_M, BLOCK_N])
    desc_dI.store([b, M_start, N_start], dI.to(desc_dI.dtype))


def get_bucket_m(M, threshold=4096, bucket_size=128):
    """
    Returns the globally unique bucket index mapped to M = B * S.
    - [0, 256)        : Fine-grained bucketing with a granularity of 32, occupying buckets 0 to 7.
    - [256, threshold): Coarse-grained bucketing with a granularity of 128.
    - >= threshold    : All fall into the last fixed large bucket.
    """
    # The first 256 lengths are bucketed with a granularity of 32, consuming 256 // 32 = 8 buckets.
    offset = 256 // 32

    if M >= threshold:
        # All extra-long sequences are placed into the final bucket: 8 + 4096//128 = 40
        return offset + threshold // bucket_size
    elif M < 256:
        # Fine-grained range.
        return M // 32

    # Coarse-grained range, must add the previous offset to avoid index conflicts with the [0, 256) range.
    return offset + M // bucket_size


DTYPE_SUPPORT_LIST = (torch.float16, torch.bfloat16, torch.float32)


def _ensure_tma_compatible(t: torch.Tensor) -> torch.Tensor:
    """
    Check if tensor meets TMA descriptor requirements.
    If not, make it contiguous.
    1. Last dimension must be contiguous (stride[-1] == 1).
    2. All other dimensions must be 16-byte aligned.
    """
    strides = t.stride()
    elem_bytes = t.element_size()
    for s in strides[:-1]:
        if (s * elem_bytes) % 16 != 0:
            raise ValueError(
                f"Tensor does not meet TMA descriptor requirements:"
                f"last stride must be 1, and all stride*elem_size must be multiple of 16."
                f"Got strides={strides}, elem_size={elem_bytes}"
            )
    if strides[-1] != 1:
        t_contig = t.contiguous()
        # Must check again.
        new_strides = t_contig.stride()
        for s in new_strides[:-1]:
            if (s * elem_bytes) % 16 != 0:
                raise ValueError(
                    "Tensor after .contiguous() still not TMA-compatible: "
                    "last stride must be 1, and all stride*elem_size must be multiple of 16."
                )
        return t_contig
    return t


def _check_inputs(input, gate_weight, up_weight, down_weight):
    if input.dtype not in DTYPE_SUPPORT_LIST:
        raise TypeError(f"flash_swiglu_mlp only supports {DTYPE_SUPPORT_LIST}, got {input.dtype}.")

    dtypes = {t.dtype for t in [input, gate_weight, up_weight, down_weight]}
    if len(dtypes) != 1:
        raise TypeError(f"All tensors must share the same dtype, got {dtypes}.")

    input = _ensure_tma_compatible(input)
    gate_weight = _ensure_tma_compatible(gate_weight)
    up_weight = _ensure_tma_compatible(up_weight)
    down_weight = _ensure_tma_compatible(down_weight)

    return input, gate_weight, up_weight, down_weight


def gemm_swiglu(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_multiplier: float,
    store_preact: bool = True,
):
    """
    Fused gate/up projection + SwiGLU.
    store_preact=True (training): also stores G and U for backward, returns (AGU, fwd_best_config).
    store_preact=False (inference): returns (A, None).
    """
    with device_context(input.device):
        B, S, dim = input.shape
        hidden_dim = gate_weight.shape[0]
        bucket_M = get_bucket_m(B * S)

        if store_preact:
            # Optimization: merge 3 torch.empty calls into 1 to reduce allocator overhead.
            AGU = torch.empty(B, S, 3 * hidden_dim, device=input.device, dtype=input.dtype)
            # Slice out 3 independent 3D views.
            A = AGU[..., :hidden_dim]
            G = AGU[..., hidden_dim : 2 * hidden_dim]
            U = AGU[..., 2 * hidden_dim :]
        else:
            A = torch.empty((B, S, hidden_dim), device=input.device, dtype=input.dtype)

        # Dummy block
        dummy_block_3D = [1, 1, 1]
        dummy_block_2D = [1, 1]

        # Create TensorDescriptor
        desc_input = TensorDescriptor.from_tensor(tensor=input, block_shape=dummy_block_3D, padding="zero")
        desc_Wg = TensorDescriptor.from_tensor(tensor=gate_weight, block_shape=dummy_block_2D, padding="zero")
        desc_Wu = TensorDescriptor.from_tensor(tensor=up_weight, block_shape=dummy_block_2D, padding="zero")
        desc_A = TensorDescriptor.from_tensor(tensor=A, block_shape=dummy_block_3D, padding="zero")

        grid = lambda META: (
            triton.cdiv(S, META["BLOCK_M"]) * triton.cdiv(hidden_dim, META["BLOCK_N"]),
            B,
        )

        if store_preact:
            desc_G = TensorDescriptor.from_tensor(tensor=G, block_shape=dummy_block_3D, padding="zero")
            desc_U = TensorDescriptor.from_tensor(tensor=U, block_shape=dummy_block_3D, padding="zero")
            _swiglu_kernel_forward[grid](
                desc_input,
                desc_Wg,
                desc_Wu,
                desc_A,
                desc_G,
                desc_U,
                S,
                dim,
                hidden_dim,
                bucket_M,
                gate_multiplier=gate_multiplier,
            )
            return AGU, _swiglu_kernel_forward.best_config
        else:
            _swiglu_kernel_forward_inference[grid](
                desc_input,
                desc_Wg,
                desc_Wu,
                desc_A,
                S,
                dim,
                hidden_dim,
                bucket_M,
                gate_multiplier=gate_multiplier,
            )
            return A, None


def swiglu_backward_dGU(
    dO: torch.Tensor,
    down_weight: torch.Tensor,
    AGU: torch.Tensor,
    fwd_best_config,
    gate_multiplier: float,
    down_multiplier: float,
) -> torch.Tensor:
    with device_context(dO.device):
        B, S, dim = dO.shape
        hidden_dim = down_weight.shape[1]

        G = AGU[..., hidden_dim : 2 * hidden_dim]
        U = AGU[..., 2 * hidden_dim :]

        # Reuse the memory of GU to store dGU, in order to save memory.
        dG, dU = G, U

        # Use forward best config.
        BLOCK_M = fwd_best_config.kwargs["BLOCK_M"]
        BLOCK_N = fwd_best_config.kwargs["BLOCK_N"]
        BLOCK_K = fwd_best_config.kwargs["BLOCK_K"]
        GROUP_SIZE_M = fwd_best_config.kwargs["GROUP_SIZE_M"]

        desc_dO = TensorDescriptor.from_tensor(tensor=dO, block_shape=[1, BLOCK_M, BLOCK_K], padding="zero")
        desc_Wd = TensorDescriptor.from_tensor(tensor=down_weight, block_shape=[BLOCK_K, BLOCK_N], padding="zero")
        desc_G = TensorDescriptor.from_tensor(tensor=G, block_shape=[1, BLOCK_M, BLOCK_N], padding="zero")
        desc_U = TensorDescriptor.from_tensor(tensor=U, block_shape=[1, BLOCK_M, BLOCK_N], padding="zero")
        desc_dG = TensorDescriptor.from_tensor(tensor=dG, block_shape=[1, BLOCK_M, BLOCK_N], padding="zero")
        desc_dU = TensorDescriptor.from_tensor(tensor=dU, block_shape=[1, BLOCK_M, BLOCK_N], padding="zero")

        grid_dGU = lambda META: (
            triton.cdiv(S, META["BLOCK_M"]) * triton.cdiv(hidden_dim, META["BLOCK_N"]),
            B,
        )

        _swiglu_kernel_backward_dGU[grid_dGU](
            desc_dO,
            desc_Wd,
            desc_G,
            desc_U,
            desc_dG,
            desc_dU,
            S,
            dim,
            hidden_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            gate_multiplier=gate_multiplier,
            down_multiplier=down_multiplier,
            num_warps=fwd_best_config.num_warps,
            num_stages=fwd_best_config.num_stages,
        )


def swiglu_backward_dI(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    AGU: torch.Tensor,
) -> torch.Tensor:
    with device_context(AGU.device):
        B, S, _ = AGU.shape
        hidden_dim, dim = gate_weight.shape
        bucket_M = get_bucket_m(B * S)
        dI = torch.empty((B, S, dim), device=gate_weight.device, dtype=gate_weight.dtype)
        dG = AGU[..., hidden_dim : 2 * hidden_dim]
        dU = AGU[..., 2 * hidden_dim :]

        dummy_block_3D = [1, 1, 1]
        dummy_block_2D = [1, 1]
        desc_dG = TensorDescriptor.from_tensor(tensor=dG, block_shape=dummy_block_3D, padding="zero")
        desc_dU = TensorDescriptor.from_tensor(tensor=dU, block_shape=dummy_block_3D, padding="zero")
        desc_dI = TensorDescriptor.from_tensor(tensor=dI, block_shape=dummy_block_3D, padding="zero")
        desc_Wg = TensorDescriptor.from_tensor(tensor=gate_weight, block_shape=dummy_block_2D, padding="zero")
        desc_Wu = TensorDescriptor.from_tensor(tensor=up_weight, block_shape=dummy_block_2D, padding="zero")

        grid_dI = lambda META: (
            triton.cdiv(S, META["BLOCK_M"]) * triton.cdiv(dim, META["BLOCK_N"]),
            B,
        )

        _swiglu_kernel_backward_dI[grid_dI](
            desc_dG,
            desc_Wg,
            desc_dU,
            desc_Wu,
            desc_dI,
            S,
            dim,
            hidden_dim,
            bucket_M,
        )

        return dI


class LigerMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        gate_weight,
        up_weight,
        down_weight,
        gate_multiplier=1.0,
        down_multiplier=1.0,
    ):
        assert input.is_cuda and gate_weight.is_cuda and up_weight.is_cuda and down_weight.is_cuda
        input, gate_weight, up_weight, down_weight = _check_inputs(input, gate_weight, up_weight, down_weight)
        # Note:
        # PyTorch automatically disables global gradient computation during
        # the forward pass, so torch.is_grad_enabled() cannot be used.
        store_preact = any(ctx.needs_input_grad)
        out, fwd_best_config = gemm_swiglu(input, gate_weight, up_weight, gate_multiplier, store_preact)
        if store_preact:
            ctx.save_for_backward(input, gate_weight, up_weight, down_weight, out)
            ctx.fwd_best_config = fwd_best_config
            ctx.gate_multiplier = gate_multiplier
            ctx.down_multiplier = down_multiplier
            hidden_dim = gate_weight.shape[0]
            if down_multiplier == 1.0:
                return F.linear(out[..., :hidden_dim], down_weight)
            else:
                return F.linear(out[..., :hidden_dim], down_weight) * down_multiplier
        else:
            # Inference mode: no GU saving.
            if down_multiplier == 1.0:
                return F.linear(out, down_weight)
            else:
                return F.linear(out, down_weight) * down_multiplier

    @staticmethod
    def backward(ctx, dO):
        dO = _ensure_tma_compatible(dO)
        input, gate_weight, up_weight, down_weight, AGU = ctx.saved_tensors
        gate_multiplier, down_multiplier = ctx.gate_multiplier, ctx.down_multiplier
        hidden_dim, dim = gate_weight.shape

        # Compute dWd.
        A = AGU[..., :hidden_dim]
        if down_multiplier == 1.0:
            dWd = dO.reshape(-1, dim).T @ A.reshape(-1, hidden_dim)  # [dim, hidden_dim]
        else:
            dWd = dO.reshape(-1, dim).T @ A.reshape(-1, hidden_dim) * down_multiplier

        # Compute dU and dG by triton kernel.
        # The original G and U will be overwritten by dG and dU.
        swiglu_backward_dGU(dO, down_weight, AGU, ctx.fwd_best_config, gate_multiplier, down_multiplier)

        # Compute dWug by a single large GEMM.
        dGU = AGU[..., hidden_dim:]
        dWug = dGU.reshape(-1, 2 * hidden_dim).T @ input.reshape(-1, dim)  # [2*hidden_dim, dim]
        dWg, dWu = dWug[:hidden_dim], dWug[hidden_dim:]  # [hidden_dim, dim]

        # Compute dI by triton kernel.
        dI = swiglu_backward_dI(gate_weight, up_weight, AGU)

        return dI, dWg, dWu, dWd, None, None
