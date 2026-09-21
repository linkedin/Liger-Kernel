from typing import Optional

import torch
import triton

from packaging.version import Version

# Trigger declaration of the ``jsd_loss_and_grad`` op location so the
# dispatcher knows where to discover the Triton + cuTile impls. Without this
# import the first dispatch call would return "no impl registered".
import liger_kernel.functional  # noqa: F401

from liger_kernel.backends import dispatch
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2
_TORCH_VERSION = Version(torch.__version__.split("+")[0])
_ADDMM_SUPPORTS_OUT_DTYPE = _TORCH_VERSION >= Version("2.8.0")


def _max_capability(device=None) -> int:
    # Runtime dispatch-plane key (e.g. 100 = sm_100/B200, 103 = sm_103/B300).
    # Called per wrapper invocation (never memoized) so mock-cc suites key
    # separately per call, same idiom as capability.is_satisfied.
    #
    # ``device`` selects WHICH GPU's capability is read. It must be threaded
    # from the tensor whose chunk budget we are sizing (``student_input``) so a
    # heterogeneous process does not size a cuda:1 input by cuda:0's plane.
    # ``None`` preserves the legacy current-device behavior; a non-CUDA device
    # (e.g. a CPU ``student_input`` while CUDA happens to be visible) falls back
    # to the tight pre-Blackwell budget via the exception guard below.
    if torch.cuda.is_available():
        try:
            maj, minor = torch.cuda.get_device_capability(device)
        except (IndexError, RuntimeError, AssertionError, ValueError):  # pragma: no cover
            return 0
        return 10 * maj + minor
    return 0


# Chunk-memory constants: the chunked forward below sizes its BT chunks so
# the transient per-chunk logits buffers live in a budget proportional to
# BT x H. The original budget constant was 1 (budget == BT x H), which is a
# memory floor, not a performance target: every chunk pays ~11 kernel
# launches + host passes (2 proj GEMMs, 2 log_softmax, softmax, 2
# contiguous, the dispatched inner JSD, the dx broadcast-reduction chain,
# the input GEMM and the grad-weight addmm), so over-chunking turns the
# whole forward into a launch-bound small-M loop. Measured on B200 (sm_100,
# bf16, inner JSD pinned to the auto-selected nvidia-cutedsl): at llama
# shapes (V = 128256, H = 4096) the constant-1 formula forces 32 chunks; a
# constant of 8 lands in the flat optimum (B200 knee measured at 8x) and is
# 1.37x faster end-to-end (4.6x at BT = 1024, where constant 1 degenerates
# to 32-row chunks). Re-measured on B300 (sm_103, identical llama-shape
# sweep): the smaller constants (2, 4) regress 12-156% exactly as on B200,
# but the knee sits one multiple further out -- constant 16 beats constant 8
# by -0.8% (BT=8192), -5.2% (BT=4096), -19.9% (BT=1024) at V=128256 and
# -2.0% at V=69120/H=5120, wash at V=32000/H=4096. The plane guard selects
# 16 on sm_103+ and keeps the B200-tuned 8 elsewhere (same plane-selected-
# constant pattern as FARN 305238c). Memory stays bounded by O(BT x H) --
# the constant only widens the hidden-budget proportionality (<= 16 x BT x H
# on B300).
_CHUNK_MEM_CONST_B200 = 8
_CHUNK_MEM_CONST_B300 = 16


def fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels,
    jsd_beta,
    ignore_index,
    has_label,
    temperature,
    jsd_impl=None,
    jsd_mode=None,
    accum_dtype=None,
):
    device = student_input.device
    dtype = student_input.dtype

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the transient memory increase over BT x H is num_chunks x BT x H (each
    # chunk materializes a chunk_size x V activation for one iteration).
    # Chunking on BT bounds that increase; the chunk count is sized below by
    # the constant-widened budget (see _CHUNK_MEM_CONST_B200/_B300).
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = ceil(32000/(8*4096)) = 1,
    #   chunk_size = BT (single pass; the constant-1 budget forces 8 chunks of 2048).
    BT, H = student_input.shape
    V = student_weight.shape[0]

    # num_chunks = ceil(V / (chunk_mem_const * H)): the loop above says memory
    # increases by (num_chunks x BT x H); sizing the chunk count by the
    # constant-widened budget keeps that increase <= chunk_mem_const x BT x H.
    # Only Blackwell widens the budget (measured there): sm_103+ uses the B300
    # constant, sm_100 the B200 constant; every pre-Blackwell device (Hopper,
    # Ampere, CPU/no-CUDA where _max_capability()==0) keeps the original tight
    # budget (constant 1 == ceil(V / H)), so non-Blackwell memory is unchanged.
    # Size the chunk budget by the plane of the GPU that actually holds the
    # fused-linear input, NOT the process-current device — otherwise a cuda:1
    # (B200) input could be sized by a cuda:0 (B300) plane and get ~2x the
    # intended transient-memory budget in a heterogeneous process.
    _cc = _max_capability(student_input.device)
    chunk_mem_const = _CHUNK_MEM_CONST_B300 if _cc > 100 else (_CHUNK_MEM_CONST_B200 if _cc == 100 else 1)
    inc_factor = triton.cdiv(V, chunk_mem_const * H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # ceil(BT / inc_factor)
    chunk_size = min(chunk_size, BT)
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    if accum_dtype is None:
        grad_weight = torch.zeros_like(student_weight, device=device) if student_weight.requires_grad else None
    else:
        grad_weight = (
            torch.zeros_like(student_weight, dtype=accum_dtype, device=device) if student_weight.requires_grad else None
        )
    grad_input = torch.zeros_like(student_input)
    # Optimization #1: scalar loss accumulator instead of a full (BT, V) fp32
    # buffer. At BT=4096 V=128256 that buffer is 2.1 GB; allocate a
    # (chunk_size, V) fp32 buffer per iteration and fold each chunk's sum into
    # a scalar. Matches the memory pattern in TileGym's reference.
    total_loss = torch.zeros((), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        # chunk both inputs, shape: chunk_size x H
        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        # shape: chunk_size x V
        # For anything starting from logits to the final JSD loss, we do computation
        # in FP32 to avoid losing numerical stability.
        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(torch.float32)

        # log-softmax with temperature
        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        student_prob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
        teacher_prob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)

        # Optimization #3: pre-compute softmax(logits) BEFORE the JSD kernel
        # launch so CUDA can overlap it with the kernel's startup. We need
        # softmax(student_logits) in the dx reduction below; computing it
        # eagerly here unblocks the launch pipeline.
        softmax_s_chunk = torch.softmax(student_logits_chunk, dim=-1)

        # ensure _input and target are contiguous
        student_prob_chunk = student_prob_chunk.contiguous()
        teacher_prob_chunk = teacher_prob_chunk.contiguous()

        # Compute the per-chunk loss + dx through the dispatcher so cuTile is
        # picked up when the user sets ``impl="nvidia-cutile"``. The dispatched
        # primitive writes the per-row loss tile into a (chunk_n_rows, V) buffer
        # and returns ``dx`` — either an in-place overwrite of ``student_prob_chunk``
        # (Triton) or a freshly allocated tensor (cuTile). We rebind so the rest
        # of the function sees the gradient regardless of which impl ran.
        chunk_labels = shift_labels[start_idx:end_idx] if has_label else None
        jsd_args = (
            student_prob_chunk,
            teacher_prob_chunk,
            chunk_labels,
            float(jsd_beta),
            int(ignore_index),
            float(n_non_ignore),
        )
        loss_chunk, student_prob_chunk = dispatch(
            "jsd_loss_and_grad",
            *jsd_args,
            impl=jsd_impl,
            mode=jsd_mode,
        )
        # Accumulate this chunk's loss into the scalar; the (chunk_size, V)
        # ``loss_chunk`` tensor is freed at the end of this iteration.
        total_loss = total_loss + loss_chunk.sum()
        # ``student_prob_chunk`` now holds dx (per-element). Continue the
        # logits-side reduction using the pre-computed softmax.
        student_logits_chunk = (
            student_prob_chunk
            - softmax_s_chunk * student_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(student_prob_chunk.shape)
        ) / temperature
        # now we traverse back to grad w.r.t. input to `lm_head` and grad
        # w.r.t. `lm_head` which should be computed in original dtype
        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            if accum_dtype is None:
                # Optimization #2: addmm_ fuses the matmul + accumulate into a
                # single cuBLAS call (vs separate ``matmul`` + ``add_``). Only
                # safe when dtypes match — fall back to the two-op path otherwise.
                if grad_weight.dtype == student_logits_chunk.dtype == student_input_chunk.dtype:
                    grad_weight.addmm_(student_logits_chunk.t(), student_input_chunk)
                else:
                    grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)
            else:
                grad_logits_t = student_logits_chunk.t()
                if (
                    _ADDMM_SUPPORTS_OUT_DTYPE
                    and grad_weight.device.type == "cuda"
                    and torch.cuda.get_device_capability(grad_weight.device)[0] >= 8
                    and grad_weight.dtype == torch.float32
                    and grad_logits_t.dtype in (torch.float16, torch.bfloat16)
                ):
                    input_chunk = student_input_chunk
                    if input_chunk.dtype != grad_logits_t.dtype:
                        input_chunk = input_chunk.to(grad_logits_t.dtype)
                    torch.addmm(
                        grad_weight,
                        grad_logits_t,
                        input_chunk,
                        out_dtype=torch.float32,
                        out=grad_weight,
                    )
                else:
                    grad_weight.add_(torch.mm(grad_logits_t, student_input_chunk).float())

    grad_weight = (
        grad_weight.to(student_weight.dtype) if grad_weight is not None and accum_dtype is not None else grad_weight
    )
    return total_loss, grad_input, grad_weight


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):
    # Skip the elementwise scale when JSD is the final loss and its upstream
    # gradient is exactly 1.0.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return grad_input, grad_weight

    BT, H = grad_input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

    element_mul_kernel[(n_rows,)](
        grad_input,
        grad_input.stride(-2),
        grad_output,
        H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32 if not is_hip() else 16,
    )

    if grad_weight is not None:
        V, H = grad_weight.shape
        n_rows = V
        element_mul_kernel[(n_rows,)](
            grad_weight,
            grad_weight.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

    return grad_input, grad_weight


class LigerFusedLinearJSDFunction(torch.autograd.Function):
    """
    Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor. Since JSD is the last layer, we can
    compute the gradient at the forward pass.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        jsd_beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        accum_dtype: Optional[torch.dtype] = None,
        jsd_impl=None,
        jsd_mode=None,
    ):
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (teacher_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            jsd_beta,
            ignore_index,
            has_label,
            temperature,
            jsd_impl,
            jsd_mode,
            accum_dtype,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(grad_output, grad_input, grad_weight)
        return (grad_input, grad_weight, None, None, None, None, None, None, None, None, None)
