"""Non-TP Blackwell scaled CE with projection-fused row statistics.

Forward keeps logits in TMEM/registers. Backward recomputes logits into a
bounded dZ workspace and uses native CuTe DSL GEMMs for both input gradients.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack

from liger_kernel.ops.cutedsl.ops._sm100_gemm import _compile_guard
from liger_kernel.ops.cutedsl.ops._sm100_gemm import _current_stream
from liger_kernel.ops.cutedsl.ops._sm100_gemm import _driver_stream
from liger_kernel.ops.cutedsl.ops._sm100_gemm import _host
from liger_kernel.ops.cutedsl.ops._sm100_gemm import _max_active_clusters
from liger_kernel.ops.cutedsl.ops.utils import ensure_cuda_context
from liger_kernel.ops.cutedsl.ops.utils import to_cute_tensor
from liger_kernel.ops.fused_linear_scaled_cross_entropy import _scaled_ce_outputs
from liger_kernel.ops.fused_linear_scaled_cross_entropy import _validate_fallback_inputs
from liger_kernel.ops.fused_linear_scaled_cross_entropy import _validate_selected_token_ids
from liger_kernel.ops.fused_linear_scaled_cross_entropy import _validate_temperature
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd

_ALIGNMENT = 64
_TOKEN_ALIGNMENT = 256
_TILE_N = 256
_FRAGMENT_N = 32
_BACKWARD_WORKSPACE_BYTES = 256 * 1024 * 1024
_LOG2_E = 1.4426950408889634
_COMPILE_CACHE = {}


class _ForwardEpilogue:
    def __init__(self, return_entropy, num_selected):
        self.return_entropy = return_entropy
        self.num_selected = num_selected

    @cute.jit
    def __call__(self, accumulator, coordinates, registers, copy, args):
        target, partial, target_logits, selected_ids, selected_logits, inverse_temperature, vocab, ignore_index = args
        row = coordinates[0, 0, 0, 0][0]
        n_tile = coordinates[0, 0, 0, 0][1] // _TILE_N
        label = target[row]
        running_max = Float32(-float("inf"))
        running_sum = Float32(0.0)
        running_weighted = Float32(0.0)
        values = cute.make_rmem_tensor((cute.size(registers),), Float32)
        exponentials = cute.make_rmem_tensor((cute.size(registers),), Float32)
        weighted = cute.make_rmem_tensor((cute.size(registers),), Float32)

        for fragment in cutlass.range_constexpr(_TILE_N // _FRAGMENT_N):
            cute.copy(copy, accumulator[None, None, None, fragment], registers)
            coords = coordinates[None, None, None, fragment]
            for element in cutlass.range_constexpr(cute.size(registers)):
                col = coords[element][1]
                value = registers[element] * inverse_temperature
                values[element] = Float32(-float("inf"))
                if col < vocab:
                    values[element] = value
                    if col == label and label != ignore_index:
                        target_logits[row] = value
                    for k in cutlass.range_constexpr(self.num_selected):
                        if col == selected_ids[k]:
                            selected_logits[row, k] = value

            chunk_max = values.load().reduce(cute.ReductionOp.MAX, Float32(-float("inf")), 0)
            new_max = cute.arch.fmax(running_max, chunk_max)
            previous_scale = cute.math.exp2((running_max - new_max) * _LOG2_E, fastmath=True)
            for element in cutlass.range_constexpr(cute.size(registers)):
                probability = Float32(0.0)
                weighted_value = Float32(0.0)
                if coords[element][1] < vocab:
                    probability = cute.math.exp2((values[element] - new_max) * _LOG2_E, fastmath=True)
                    if const_expr(self.return_entropy):
                        weighted_value = probability * values[element]
                exponentials[element] = probability
                if const_expr(self.return_entropy):
                    weighted[element] = weighted_value
            running_sum = running_sum * previous_scale + exponentials.load().reduce(
                cute.ReductionOp.ADD, Float32(0.0), 0
            )
            if const_expr(self.return_entropy):
                running_weighted = running_weighted * previous_scale + weighted.load().reduce(
                    cute.ReductionOp.ADD, Float32(0.0), 0
                )
            running_max = new_max

        partial[n_tile, 0, row] = running_max
        partial[n_tile, 1, row] = running_sum
        if const_expr(self.return_entropy):
            partial[n_tile, 2, row] = running_weighted


class _BackwardEpilogue:
    def __init__(self, return_entropy, num_selected):
        self.return_entropy = return_entropy
        self.num_selected = num_selected

    @cute.jit
    def __call__(self, accumulator, coordinates, registers, copy, args):
        (
            target,
            lse,
            entropy,
            grad_nll,
            grad_entropy,
            selected_ids,
            grad_selected,
            dz,
            inverse_temperature,
            vocab,
            ignore_index,
        ) = args
        row = coordinates[0, 0, 0, 0][0]
        label = target[row]
        normalizer = lse[row]
        nll_scale = grad_nll[row]
        selected_sum = Float32(0.0)
        for k in cutlass.range_constexpr(self.num_selected):
            selected_sum += grad_selected[row, k]
        entropy_value = Float32(0.0)
        entropy_scale = Float32(0.0)
        if const_expr(self.return_entropy):
            entropy_value = entropy[row]
            entropy_scale = grad_entropy[row]

        for fragment in cutlass.range_constexpr(_TILE_N // _FRAGMENT_N):
            cute.copy(copy, accumulator[None, None, None, fragment], registers)
            coords = coordinates[None, None, None, fragment]
            for element in cutlass.range_constexpr(cute.size(registers)):
                col = coords[element][1]
                value = Float32(0.0)
                if col < vocab and label != ignore_index:
                    logit = registers[element] * inverse_temperature
                    probability = cute.math.exp2((logit - normalizer) * _LOG2_E, fastmath=True)
                    value = probability * (nll_scale - selected_sum)
                    if col == label:
                        value -= nll_scale
                    if const_expr(self.return_entropy):
                        value += probability * (normalizer - entropy_value - logit) * entropy_scale
                    for k in cutlass.range_constexpr(self.num_selected):
                        if col == selected_ids[k]:
                            value += grad_selected[row, k]
                    value *= inverse_temperature
                if col < dz.shape[1]:
                    dz[row, col] = value.to(dz.element_type)


class _GradientEpilogue:
    def __init__(self, accumulate):
        self.accumulate = accumulate

    @cute.jit
    def __call__(self, accumulator, coordinates, registers, copy, args):
        (output,) = args
        previous = cute.make_rmem_tensor((_FRAGMENT_N,), Float32)
        result = cute.make_rmem_tensor((_FRAGMENT_N,), output.element_type)
        for fragment in cutlass.range_constexpr(_TILE_N // _FRAGMENT_N):
            cute.copy(copy, accumulator[None, None, None, fragment], registers)
            coords = coordinates[None, None, None, fragment]
            row, col = coords[0][0], coords[0][1]
            if row < output.shape[0] and col < output.shape[1]:
                destination = cute.make_tensor(
                    output.iterator + cute.crd2idx((row, col), output.layout),
                    cute.make_layout((_FRAGMENT_N,)),
                )
                if const_expr(self.accumulate):
                    cute.autovec_copy(destination, previous)
                for element in cutlass.range_constexpr(_FRAGMENT_N):
                    value = registers[element]
                    if const_expr(self.accumulate):
                        value += previous[element]
                    result[element] = value.to(output.element_type)
                cute.autovec_copy(result, destination)


@cute.kernel
def _finalize_kernel(
    partial: cute.Tensor,
    target: cute.Tensor,
    target_logits: cute.Tensor,
    nll: cute.Tensor,
    lse: cute.Tensor,
    entropy: cute.Tensor,
    saved_entropy: cute.Tensor,
    selected: cute.Tensor,
    ignore_index: Int64,
    return_entropy: cutlass.Constexpr,
    num_selected: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    row = block * 128 + tid
    if row < target.shape[0]:
        maximum = Float32(-float("inf"))
        for split in range(partial.shape[0]):
            maximum = cute.arch.fmax(maximum, partial[split, 0, row])
        total = Float32(0.0)
        weighted = Float32(0.0)
        for split in range(partial.shape[0]):
            scale = cute.math.exp2((partial[split, 0, row] - maximum) * _LOG2_E, fastmath=True)
            total += partial[split, 1, row] * scale
            if const_expr(return_entropy):
                weighted += partial[split, 2, row] * scale
        normalizer = maximum + cute.math.log(total)
        lse[row] = normalizer
        valid = target[row] != ignore_index
        nll[row] = Float32(0.0)
        if valid:
            nll[row] = normalizer - target_logits[row]
        if const_expr(return_entropy):
            value = Float32(0.0)
            if valid:
                value = normalizer - weighted / total
            saved_entropy[row] = value
            entropy[row] = value.to(entropy.element_type)
        for k in cutlass.range_constexpr(num_selected):
            value = Float32(0.0)
            if valid:
                value = selected[row, k] - normalizer
            selected[row, k] = value


@cute.jit
def _finalize(
    partial: cute.Tensor,
    target: cute.Tensor,
    target_logits: cute.Tensor,
    nll: cute.Tensor,
    lse: cute.Tensor,
    entropy: cute.Tensor,
    saved_entropy: cute.Tensor,
    selected: cute.Tensor,
    ignore_index: Int64,
    return_entropy: cutlass.Constexpr,
    num_selected: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _finalize_kernel(
        partial,
        target,
        target_logits,
        nll,
        lse,
        entropy,
        saved_entropy,
        selected,
        ignore_index,
        return_entropy,
        num_selected,
    ).launch(grid=(cute.ceil_div(target.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream)


def _matrix(tensor):
    leading_dim = 1 if tensor.stride(1) == 1 else 0
    return (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, divisibility=_ALIGNMENT)
    )


def _arguments(args):
    return tuple(
        to_cute_tensor(arg, assumed_align=arg.element_size()) if isinstance(arg, torch.Tensor) else arg for arg in args
    )


def _gemm(a, b, epilogue, args, key):
    # The shared mainloop constructs an output descriptor, but this epilogue
    # never stores the matrix. Only its row-statistics/gradient stores execute.
    dummy = a.new_empty((a.shape[0], 8))
    m_a, m_b = _matrix(a), _matrix(b)
    m_dummy = to_cute_tensor(dummy, assumed_align=16)
    stream = _driver_stream(_current_stream(a.device))
    cute_args = (_matrix(args[0]),) if isinstance(epilogue, _GradientEpilogue) else _arguments(args)
    a_k_major, b_k_major = a.stride(1) == 1, b.stride(1) == 1
    cache_key = (a.device, a.dtype, a_k_major, b_k_major, *key)
    compiled = _COMPILE_CACHE.get(cache_key)
    if compiled is None:
        with _compile_guard():
            compiled = cute.compile(
                _host,
                m_a,
                m_b,
                m_dummy,
                6,
                epilogue,
                0,
                False,
                False,
                1,
                _max_active_clusters(a.device.index),
                stream,
                cute_args,
                a_k_major,
                b_k_major,
            )
        _COMPILE_CACHE[cache_key] = compiled
    compiled(m_a, m_b, m_dummy, stream, cute_args)


def _pad_matrix(tensor, rows, cols):
    if tensor.shape == (rows, cols):
        return tensor.contiguous()
    return torch.nn.functional.pad(tensor, (0, cols - tensor.shape[1], 0, rows - tensor.shape[0]))


def _align(size):
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _forward(x, weight, target, temperature, ignore_index, return_entropy, selected_ids):
    tokens, hidden = x.shape
    vocab = weight.shape[0]
    padded_tokens = (tokens + _TOKEN_ALIGNMENT - 1) // _TOKEN_ALIGNMENT * _TOKEN_ALIGNMENT
    padded_hidden, padded_vocab = _align(hidden), _align(vocab)
    x = _pad_matrix(x, padded_tokens, padded_hidden)
    weight = _pad_matrix(weight, padded_vocab, padded_hidden)
    target = torch.nn.functional.pad(target.contiguous(), (0, padded_tokens - tokens), value=ignore_index)
    num_selected = selected_ids.numel() if selected_ids is not None else 0
    selected_ids = selected_ids if selected_ids is not None else target.new_empty(0)
    partial = torch.empty(
        (padded_vocab + _TILE_N - 1) // _TILE_N,
        3 if return_entropy else 2,
        padded_tokens,
        device=x.device,
        dtype=torch.float32,
    )
    nll = torch.empty(padded_tokens, device=x.device, dtype=torch.float32)
    lse = torch.empty_like(nll)
    target_logits = torch.empty_like(nll)
    entropy = x.new_empty(padded_tokens if return_entropy else 0)
    saved_entropy = nll.new_empty(padded_tokens if return_entropy else 0)
    selected = nll.new_empty(padded_tokens, num_selected)
    _gemm(
        x,
        weight,
        _ForwardEpilogue(return_entropy, num_selected),
        (
            target,
            partial,
            target_logits,
            selected_ids,
            selected,
            Float32(1.0 / temperature),
            Int32(vocab),
            Int64(ignore_index),
        ),
        ("forward", return_entropy, num_selected),
    )
    args = _arguments((partial, target, target_logits, nll, lse, entropy, saved_entropy, selected))
    stream = _driver_stream(_current_stream(x.device))
    key = (x.device, "finalize", return_entropy, num_selected)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        with _compile_guard():
            compiled = cute.compile(_finalize, *args, Int64(ignore_index), return_entropy, num_selected, stream)
        _COMPILE_CACHE[key] = compiled
    compiled(*args, Int64(ignore_index), stream)
    return nll, entropy, selected, x, weight, target, lse, saved_entropy, selected_ids


def _row_gradient(gradient, rows, columns=None):
    if columns is None:
        shape = (rows,)
    else:
        shape = (rows, columns)
    if gradient is None:
        return None
    gradient = gradient.to(torch.float32).contiguous()
    if gradient.shape == shape:
        return gradient
    padding = (0, rows - gradient.shape[0]) if columns is None else (0, 0, 0, rows - gradient.shape[0])
    return torch.nn.functional.pad(gradient, padding)


class LigerFusedScaledCrossEntropySM100Function(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        x,
        weight,
        target,
        temperature=1.0,
        ignore_index=-100,
        m_tiles_per_cluster=1,
        return_entropy=False,
        selected_token_ids=None,
    ):
        _validate_temperature(temperature)
        _validate_fallback_inputs(x, weight, target, ignore_index)
        if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3)):
            raise RuntimeError("Native Blackwell scaled cross entropy requires an SM100 or SM103 CUDA GPU")
        if x.dtype != torch.bfloat16:
            raise TypeError("Native Blackwell scaled cross entropy requires BF16 input and weight")
        if isinstance(m_tiles_per_cluster, bool) or not isinstance(m_tiles_per_cluster, int):
            raise TypeError("m_tiles_per_cluster must be an int")
        if m_tiles_per_cluster < 1:
            raise ValueError("m_tiles_per_cluster must be >= 1")
        if not isinstance(return_entropy, bool):
            raise TypeError("return_entropy must be a bool")
        selected_token_ids = _validate_selected_token_ids(selected_token_ids, weight)
        ctx.set_materialize_grads(False)
        ctx.original_shape = (x.shape[0], x.shape[1], weight.shape[0])
        ctx.temperature = temperature
        ctx.ignore_index = ignore_index
        ctx.return_entropy = return_entropy
        with torch.cuda.device(x.device):
            ensure_cuda_context()
            nll, entropy, selected, *saved = _forward(
                x, weight, target, temperature, ignore_index, return_entropy, selected_token_ids
            )
        ctx.save_for_backward(*saved)
        tokens = ctx.original_shape[0]
        return _scaled_ce_outputs(
            nll[:tokens],
            entropy[:tokens] if return_entropy else None,
            selected[:tokens] if selected_token_ids is not None else None,
        )

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_nll, grad_entropy=None, grad_selected=None):
        x, weight, target, lse, entropy, selected_ids = ctx.saved_tensors
        if not ctx.return_entropy:
            grad_selected, grad_entropy = grad_entropy, None
        tokens, hidden, vocab = ctx.original_shape
        padded_tokens = x.shape[0]
        num_selected = selected_ids.numel()
        with torch.cuda.device(x.device):
            ensure_cuda_context()
            grad_nll = _row_gradient(grad_nll, padded_tokens)
            if grad_nll is None:
                grad_nll = torch.zeros_like(lse)
            grad_entropy = _row_gradient(grad_entropy, padded_tokens)
            if grad_entropy is None:
                grad_entropy = torch.zeros_like(lse) if ctx.return_entropy else lse.new_empty(0)
            grad_selected = _row_gradient(grad_selected, padded_tokens, num_selected)
            if grad_selected is None:
                grad_selected = lse.new_zeros(padded_tokens, num_selected)
            grad_input = torch.empty_like(x) if ctx.needs_input_grad[0] else None
            grad_weight = torch.empty_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
            rows_per_chunk = max(
                _TOKEN_ALIGNMENT,
                _BACKWARD_WORKSPACE_BYTES // (weight.shape[0] * x.element_size() * _TOKEN_ALIGNMENT) * _TOKEN_ALIGNMENT,
            )
            rows_per_chunk = min(padded_tokens, rows_per_chunk)
            dz_buffer = x.new_empty(rows_per_chunk, weight.shape[0])
            for start in range(0, padded_tokens, rows_per_chunk):
                end = min(start + rows_per_chunk, padded_tokens)
                dz = dz_buffer[: end - start]
                _gemm(
                    x[start:end],
                    weight,
                    _BackwardEpilogue(ctx.return_entropy, num_selected),
                    (
                        target[start:end],
                        lse[start:end],
                        entropy[start:end],
                        grad_nll[start:end],
                        grad_entropy[start:end],
                        selected_ids,
                        grad_selected[start:end],
                        dz,
                        Float32(1.0 / ctx.temperature),
                        Int32(vocab),
                        Int64(ctx.ignore_index),
                    ),
                    ("backward", ctx.return_entropy, num_selected),
                )
                if grad_input is not None:
                    _gemm(
                        dz,
                        weight.t(),
                        _GradientEpilogue(False),
                        (grad_input[start:end],),
                        ("dx", False),
                    )
                if grad_weight is not None:
                    _gemm(
                        dz.t(),
                        x[start:end].t(),
                        _GradientEpilogue(start != 0),
                        (grad_weight,),
                        ("dw", start != 0),
                    )
            if grad_input is not None:
                grad_input = grad_input[:tokens, :hidden]
            if grad_weight is not None:
                grad_weight = grad_weight[:vocab, :hidden].to(weight.dtype)
        return grad_input, grad_weight, None, None, None, None, None, None
