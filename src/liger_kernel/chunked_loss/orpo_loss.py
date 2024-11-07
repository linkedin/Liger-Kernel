import torch
from triton import next_power_of_2
import torch.nn.functional as F
from liger_kernel.ops.utils import element_mul_kernel

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def odds_ratio_loss(chosen_logps, rejected_logps, beta=1.0):
    """
    Compute odds-ratio loss.
    Args:
        chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
        rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
        beta (float): Weight for the odds ratio loss.
    """
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )
    ratio = torch.log(F.sigmoid(log_odds))
    return beta * ratio.sum()


class LigerFusedLinearORPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, bias=None, ignore_index=-100, compiled=True, pre_compiled=None):
        """
        Fused linear forward function with ORPO (Odds-Ratio Preference Optimization).
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            bias (torch.Tensor, optional): Bias tensor. Shape: (hidden_size,).
            ignore_index (int): Index to ignore for loss computation.
            compiled (bool): Whether to use compiled mode for chunk accumulation.
        """
        CHUNK_SIZE = 1

        def _compute_orpo_loss(input_chunk, weight, target_chunk, bias=None):
            len_chosen_chunk = target_chunk.shape[0] // 2

            unnorm_logits = input_chunk @ weight.t()  # chunk_size x V
            if bias is not None:
                unnorm_logits = unnorm_logits + bias
            unnorm_logits = unnorm_logits.float()
            norm_logits = F.log_softmax(unnorm_logits, dim=-1)

            chosen_nll_loss = F.nll_loss(
                norm_logits[:len_chosen_chunk].view(-1, norm_logits.shape[-1]),
                target_chunk[:len_chosen_chunk].view(-1),
                reduction="sum",
                ignore_index=ignore_index
            )
            all_logps = norm_logits.gather(-1, target_chunk.unsqueeze(2)).squeeze(2)
            chosen_logps = all_logps[:len_chosen_chunk].mean(dim=1)
            rejected_logps = all_logps[len_chosen_chunk:].mean(dim=1)

            or_loss = odds_ratio_loss(chosen_logps, rejected_logps)

            chosen_nll_loss = chosen_nll_loss / (target[:target.shape[0]//2] != ignore_index).sum().item()
            or_loss = or_loss / _input.shape[0]

            loss = chosen_nll_loss + or_loss
            return loss

        def compute_orpo_loss(input_chunk, weight, target_chunk, bias=None):
            return _compute_orpo_loss(input_chunk, weight, target_chunk, bias)

        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=_input.device)

        chunks = max(1, _input.shape[0] // (2 * CHUNK_SIZE))

        def accumulate_chunk(input_chunk, target_chunk):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), chunk_loss = torch.func.grad_and_value(
                    compute_orpo_loss, argnums=(0, 1, 3)
                )(input_chunk, weight, target_chunk, bias)
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
                    compute_orpo_loss, argnums=(0, 1)
                )(input_chunk, weight, target_chunk)
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input


        len_chosen = target.shape[0] // 2
        _chosen_input_chunks = torch.chunk(_input[:len_chosen], chunks=chunks, dim=0)
        _chosen_target_chunks = torch.chunk(target[:len_chosen], chunks=chunks, dim=0)
        _rejected_input_chunks = torch.chunk(_input[len_chosen:], chunks=chunks, dim=0)
        _rejected_target_chunks = torch.chunk(target[len_chosen:], chunks=chunks, dim=0)

        for (
            chosen_input_chunk,
            rejected_input_chunk,
            chosen_target_chunk,
            rejected_target_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
        ):
            input_chunk = torch.cat([chosen_input_chunk, rejected_input_chunk], dim=0)
            target_chunk = torch.cat([chosen_target_chunk, rejected_target_chunk], dim=0)

            if pre_compiled:
                if pre_compiled == "original":
                    from liger_kernel.ops.experimental.orpo_loss_accumulate_original import call as accumulate_chunk_compiled
                elif pre_compiled == "modified":
                    from liger_kernel.ops.experimental.orpo_loss_accumulate_modified import call as accumulate_chunk_compiled

                grad_input = accumulate_chunk_compiled([bias, input_chunk, weight, target_chunk, target, grad_bias, grad_weight, loss_acc])[0]
            else:
                if compiled:
                    accumulate_chunk = torch.compile(accumulate_chunk)
                grad_input = accumulate_chunk(input_chunk, target_chunk)

            grad_chosen_inputs.append(grad_input[: chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(grad_input[chosen_target_chunk.shape[0] :])

        # combine grad_chosen_inputs and grad_rejected_inputs
        grad_inputs = grad_chosen_inputs + grad_rejected_inputs

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias,
        )
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
            # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
            # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
            BT, H = grad_input.view(-1, grad_input.shape[-1]).shape
            n_rows = BT
            BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(H))

            element_mul_kernel[(n_rows,)](
                grad_input,
                grad_input.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
            )

            # handle grad_weight
            if grad_weight is not None:
                V, H = grad_weight.shape
                n_rows = V

                element_mul_kernel[(n_rows,)](
                    grad_weight,
                    grad_weight.stride(-2),
                    grad_output,
                    H,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=32,
                )

            if grad_bias is not None:
                V = grad_bias.shape[0]
                n_rows = V

                element_mul_kernel[(n_rows,)](
                    grad_bias,
                    grad_bias.stride(-1),
                    grad_output,
                    1,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=32,
                )
        return grad_input, grad_weight, None, grad_bias, None, None, None


if __name__ == "__main__":
    # Define input tensors
    B, T, H, V = 32, 1024, 768, 128256
    scalar = 1.0
    dtype = torch.bfloat16
    bias = True
    device = "cuda"

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    x = _input.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B, T,), device=device, dtype=torch.long)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    bias = torch.randn(V, device=device, dtype=dtype) if bias else None

    y = LigerFusedLinearORPOFunction.apply(x, weight, target, bias, -100, True)
    y.backward()