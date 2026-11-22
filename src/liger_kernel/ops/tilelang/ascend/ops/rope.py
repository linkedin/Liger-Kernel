"""Liger RoPE and autograd wrappers for the Ascend TileLang backend."""

import torch

from torch.autograd.function import once_differentiable

from liger_kernel.ops.tilelang.ascend.ops import _rope_kernels as kernels

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate(q, k, cos, sin, width):
    tensors = (q, k, cos, sin)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("q, k, cos and sin must be tensors")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have BNSD rank 4")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k must share batch, sequence and head dimension")
    if width > 512:
        raise ValueError("ascend-tilelang RoPE supports rotary_dim <= 512")
    if q.dtype not in _DTYPES or k.dtype not in _DTYPES:
        raise TypeError("q and k must use FP16, BF16 or FP32")
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("cos and sin must share rank 3 shape [1 or B, S, D]")
    if cos.shape[0] not in (1, q.shape[0]) or cos.shape[1:] != (q.shape[2], width):
        raise ValueError("cos and sin must have shape [1 or B, S, rotary_dim]")
    if cos.dtype not in _DTYPES or sin.dtype not in _DTYPES:
        raise TypeError("cos and sin must use FP16, BF16 or FP32")
    if any(t.device != q.device for t in tensors) or q.device.type != "npu":
        raise ValueError("all inputs must be on the same NPU device")


def _launch(x, cos, sin, backward=False):
    if x is None:
        return None
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if not x.numel():
        return out
    if not (x.is_contiguous() or x.transpose(1, 2).is_contiguous()):
        x = x.contiguous()
    B, H, S, D = x.shape
    with torch.npu.device(x.device):
        cores = torch.npu.get_device_properties(x.device).vector_core_num
        dtype = str(x.dtype).removeprefix("torch.")
        phase_dtype = str(cos.dtype).removeprefix("torch.")
        bsnd = not x.is_contiguous() and x.transpose(1, 2).is_contiguous()
        route = kernels.rope_launch_route(B, H, S, D, dtype, phase_dtype, cores, bsnd)
        if route == "dense":
            x = x.contiguous()
        strides = tuple(x.stride()) if not x.is_contiguous() else None
        kernel = kernels.rope_kernel(
            B,
            H,
            S,
            D,
            cos.shape[0],
            str(x.dtype).removeprefix("torch."),
            str(cos.dtype).removeprefix("torch."),
            backward,
            cores,
            strides,
        )
        kernel(x, cos, sin, out)
    return out


def _grouped_pair_shape_gate(q, k, cos):
    """Reject shapes that cannot use grouped-pair before querying NPU properties."""
    B, Hq, S, D = q.shape
    Hk = k.shape[1]
    group, rows = 4, 32
    dtype = str(q.dtype).removeprefix("torch.")
    phase_dtype = str(cos.dtype).removeprefix("torch.")
    item_bytes = 4 if dtype == "float32" else 2
    phase_bytes = 4 if phase_dtype == "float32" else 2
    half = D // 2
    return (
        min(B, Hq, Hk, S) > 0
        and S >= 2048
        and S % rows == 0
        and Hq % group == 0
        and Hk % group == 0
        and D <= 128
        and dtype in ("float16", "bfloat16", "float32")
        and half * item_bytes % 32 == 0
        and half * phase_bytes % 32 == 0
    )


def _launch_pair(q, k, cos, sin, backward=False):
    """Use an eligible pair builder; keep all other paths on single launches."""
    if q is None or k is None or q.dtype != k.dtype:
        return _launch(q, cos, sin, backward), _launch(k, cos, sin, backward)
    if not q.numel() or not k.numel():
        return _launch(q, cos, sin, backward), _launch(k, cos, sin, backward)
    B, Hq, S, D = q.shape
    Hk = k.shape[1]
    dtype = str(q.dtype).removeprefix("torch.")
    phase_dtype = str(cos.dtype).removeprefix("torch.")
    with torch.npu.device(q.device):
        cores = torch.npu.get_device_properties(q.device).vector_core_num
        if _grouped_pair_shape_gate(q, k, cos) and all(
            x.is_contiguous() or x.transpose(1, 2).is_contiguous() for x in (q, k)
        ):
            builders = (kernels.rope_grouped_pair_kernel,)
            if S >= 8192:
                builders = (kernels.rope_sequence_pair_kernel, *builders)
            for builder in builders:
                kernel = builder(
                    B,
                    Hq,
                    Hk,
                    S,
                    D,
                    cos.shape[0],
                    dtype,
                    phase_dtype,
                    backward,
                    cores,
                    tuple(q.stride()) if not q.is_contiguous() else None,
                    tuple(k.stride()) if not k.is_contiguous() else None,
                )
                if kernel is not None:
                    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
                    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
                    kernel(q, k, cos, sin, qo, ko)
                    return qo, ko
        builder = kernels.rope_pair_kernel
        kernel = builder(
            B,
            Hq,
            Hk,
            S,
            D,
            cos.shape[0],
            str(q.dtype).removeprefix("torch."),
            str(cos.dtype).removeprefix("torch."),
            backward,
            cores,
        )
        if kernel is None:
            return _launch(q, cos, sin, backward), _launch(k, cos, sin, backward)
        q, k = q.contiguous(), k.contiguous()
        qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        kernel(q, k, cos, sin, qo, ko)
    return qo, ko


def _prepare_phases(cos, sin, head_dim, rotary_dim):
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("cos and sin must share shape [S, D] or [1 or B, S, D]")
    width = min(cos.shape[-1], head_dim) if rotary_dim is None else int(rotary_dim)
    if width % 2 or not 0 < width <= head_dim:
        raise ValueError(f"rotary_dim must be even and in (0, head_dim={head_dim}]; got {width}")
    if cos.shape[-1] < width // 2:
        raise ValueError(f"cos and sin must provide at least {width // 2} entries per row")
    if cos.shape[-1] < width:
        # The kernels load a full-width tile, including for half-width GPT-OSS phases.
        cos = torch.cat((cos[..., : width // 2], cos[..., : width // 2]), dim=-1)
        sin = torch.cat((sin[..., : width // 2], sin[..., : width // 2]), dim=-1)
    if cos.dtype != sin.dtype:
        cos, sin = cos.float(), sin.float()
    return cos[..., :width].contiguous(), sin[..., :width].contiguous(), width


def _rotate_prefix(q, k, cos, sin, width, backward):
    partial = width < (q if q is not None else k).shape[-1]
    q_prefix = q[..., :width] if partial and q is not None else q
    k_prefix = k[..., :width] if partial and k is not None else k
    q_out, k_out = _launch_pair(q_prefix, k_prefix, cos, sin, backward)
    if partial:
        if q is not None:
            q_out = torch.cat((q_out, q[..., width:]), dim=-1)
        if k is not None:
            k_out = torch.cat((k_out, k[..., width:]), dim=-1)
    return q_out, k_out


def rope_forward(q, k, cos, sin, rotary_dim=None):
    cos, sin, width = _prepare_phases(cos, sin, q.shape[-1], rotary_dim)
    _validate(q, k, cos, sin, width)
    q_out, k_out = _rotate_prefix(q, k, cos, sin, width, False)
    return q_out, k_out, cos, sin


def rope_backward(dq, dk, cos, sin, rotary_dim=None):
    if dq is None and dk is None:
        return None, None
    ref = dq if dq is not None else dk
    cos, sin, width = _prepare_phases(cos, sin, ref.shape[-1], rotary_dim)
    return _rotate_prefix(dq, dk, cos, sin, width, True)


class LigerRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rotary_dim=None):
        ctx.set_materialize_grads(False)
        q_out, k_out, cos, sin = rope_forward(q, k, cos, sin, rotary_dim)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = rope_backward(dq, dk, cos, sin)
        return (dq_out, dk_out, None, None, None, None, None)[: len(ctx.needs_input_grad)]
