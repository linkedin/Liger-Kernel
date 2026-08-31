import torch
import triton
import triton.language as tl


@triton.jit
def _selective_log_softmax_forward_kernel(
    logits,
    target,
    logps,
    lse,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).cast(tl.int64)
    row_logits = logits + row * n_cols

    row_max = float("-inf")
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(row_logits + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(row_max, tl.max(values))
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(values - new_max))
        row_max = new_max

    row_lse = row_max + tl.log(row_sum)
    target_col = tl.load(target + row)
    selected = tl.load(row_logits + target_col).to(tl.float32)
    tl.store(logps + row, selected - row_lse)
    tl.store(lse + row, row_lse)


@triton.jit
def _selective_log_softmax_backward_kernel(
    grad_logps,
    logits,
    target,
    lse,
    grad_logits,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).cast(tl.int64)
    row_logits = logits + row * n_cols
    row_grad_logits = grad_logits + row * n_cols
    row_grad = tl.load(grad_logps + row).to(tl.float32)
    row_lse = tl.load(lse + row).to(tl.float32)
    target_col = tl.load(target + row)

    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(row_logits + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        probabilities = tl.exp(values - row_lse)
        gradients = (tl.where(cols == target_col, 1.0, 0.0) - probabilities) * row_grad
        tl.store(row_grad_logits + cols, gradients, mask=cols < n_cols)


def _launch_config(n_cols):
    block_size = min(triton.next_power_of_2(n_cols), 4096)
    return {"BLOCK_SIZE": block_size, "num_stages": 2, "num_warps": 8}


class LigerSelectiveLogSoftmaxFunction(torch.autograd.Function):
    """Target-token log-softmax with FP32 accumulation and no FP32 vocabulary tensor."""

    generate_vmap_rule = True

    @staticmethod
    def forward(logits, target):
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if not target.is_contiguous():
            target = target.contiguous()

        n_cols = logits.shape[-1]
        n_rows = logits.numel() // n_cols
        flat_target = target.reshape(-1)
        logps = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty_like(logps)
        _selective_log_softmax_forward_kernel[(n_rows,)](
            logits,
            flat_target,
            logps,
            lse,
            n_cols=n_cols,
            **_launch_config(n_cols),
        )
        return logps.reshape(target.shape), lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        logits, target = inputs
        _, lse = output
        ctx.save_for_backward(logits, target, lse)
        ctx.mark_non_differentiable(lse)

    @staticmethod
    def backward(ctx, grad_logps, _grad_lse):
        logits, target, lse = ctx.saved_tensors
        n_cols = logits.shape[-1]
        n_rows = logits.numel() // n_cols
        grad_logits = torch.empty_like(logits)
        _selective_log_softmax_backward_kernel[(n_rows,)](
            grad_logps.contiguous(),
            logits,
            target.reshape(-1),
            lse,
            grad_logits,
            n_cols=n_cols,
            **_launch_config(n_cols),
        )
        return grad_logits, None


def liger_selective_log_softmax(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return LigerSelectiveLogSoftmaxFunction.apply(logits, target)[0]
