import torch
import torch.nn as nn
import torch.nn.functional as F
from triton import next_power_of_2

from liger_kernel.ops.utils import element_mul_kernel

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# Setting limit as 65536 for better performance due to less register spilling
MAX_FUSED_SIZE = 65536 // 2


def dpo_loss(chosen_logps, rejected_logps, beta=0.1):
    """
    Compute DPO loss (Direct Preference Optimization).
    Args:
        chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
        rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
        beta (float): Temperature parameter for the DPO loss.
    """
    logits_diff = (chosen_logps - rejected_logps) / beta
    losses = -F.logsigmoid(logits_diff)
    return losses.sum()


class LigerFusedLinearDPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ignore_index=-100,
        beta=0.1,
        compiled=True,
    ):
        """
        Fused linear layer with DPO (Direct Preference Optimization) loss.
        Handles both the forward and backward pass of the final linear layer with DPO loss.

        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target (torch.Tensor): Target tensor. Shape: (batch_size, seq_len).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            ignore_index (int): Index to ignore for loss computation.
            beta (float): Temperature parameter for the DPO loss.
            compiled (bool): Whether to use torch compile for chunk accumulation.
        """
        # TODO: Tune CHUNK_SIZE to fully utilize the GPU
        CHUNK_SIZE = 1

        def _compute_dpo_loss(input_chunk, weight, target_chunk, bias=None):
            len_chosen_chunk = target_chunk.shape[0] // 2

            unnorm_logits = input_chunk @ weight.t()  # chunk_size x V
            if bias is not None:
                unnorm_logits = unnorm_logits + bias
            unnorm_logits = unnorm_logits.float()
            norm_logits = F.log_softmax(unnorm_logits, dim=-1)

            # Compute NLL loss for chosen responses
            chosen_nll_loss = F.nll_loss(
                norm_logits[:len_chosen_chunk].view(-1, norm_logits.shape[-1]),
                target_chunk[:len_chosen_chunk].view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )
            chosen_nll_loss = (
                chosen_nll_loss / (target[: target.shape[0] // 2] != ignore_index).sum()
            )

            # Compute log probabilities for both chosen and rejected
            loss_mask = target_chunk != ignore_index
            label_chunk = torch.where(loss_mask, target_chunk, 0)
            per_token_logps = norm_logits.gather(-1, label_chunk.unsqueeze(-1)).squeeze(
                -1
            )
            average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            chosen_logps = average_log_prob[:len_chosen_chunk]
            rejected_logps = average_log_prob[len_chosen_chunk:]

            # Compute DPO loss
            preference_loss = dpo_loss(chosen_logps, rejected_logps, beta=beta)
            preference_loss = preference_loss / (target.shape[0] // 2)

            # Total loss combines NLL and DPO loss
            loss = chosen_nll_loss + preference_loss
            return loss, (preference_loss, chosen_logps, rejected_logps)

        def compute_dpo_loss(input_chunk, weight, target_chunk, bias=None):
            return _compute_dpo_loss(input_chunk, weight, target_chunk, bias)

        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        loss_acc = torch.zeros((), device=_input.device)

        chunks = max(1, _input.shape[0] // (2 * CHUNK_SIZE))

        def accumulate_chunk(input_chunk, target_chunk):
            if bias is not None:
                (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), (
                    chunk_loss,
                    (chunk_dpo_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    compute_dpo_loss, argnums=(0, 1, 3), has_aux=True
                )(
                    input_chunk, weight, target_chunk, bias
                )
                grad_bias.add_(chunk_grad_bias)
            else:
                (chunk_grad_input, chunk_grad_weight), (
                    chunk_loss,
                    (chunk_dpo_loss, chunk_chosen_logps, chunk_rejected_logps),
                ) = torch.func.grad_and_value(
                    compute_dpo_loss, argnums=(0, 1), has_aux=True
                )(
                    input_chunk, weight, target_chunk
                )
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
            target_chunk = torch.cat(
                [chosen_target_chunk, rejected_target_chunk], dim=0
            )

            if compiled:
                accumulate_chunk = torch.compile(accumulate_chunk)
            grad_input = accumulate_chunk(input_chunk, target_chunk)

            grad_chosen_inputs.append(grad_input[: chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(grad_input[chosen_target_chunk.shape[0] :])

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


class HF_DPO_Loss:
    """
    Implementation of Direct Preference Optimization (DPO) loss,
    adapted from the Hugging Face implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
    """

    def __init__(self, ignore_index: int = -100, beta: float = 0.1):
        self.ignore_index = ignore_index
        self.beta = beta

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        loss_mask = labels != self.ignore_index
        labels = torch.where(labels == self.ignore_index, 0, labels)
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute DPO loss for a batch of policy log probabilities."""
        logits_diff = (policy_chosen_logps - policy_rejected_logps) / self.beta
        losses = -F.logsigmoid(logits_diff)
        return losses

    def concatenated_forward(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
    ):
        len_chosen = _input.shape[0] // 2

        outputs = _input @ weight.t()
        if bias is not None:
            outputs = outputs + bias
        all_logits = outputs.float()

        def cross_entropy_loss(logits, labels):
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        chosen_nll_loss = cross_entropy_loss(
            all_logits[:len_chosen], target[:len_chosen]
        )

        all_logps = self.get_batch_logps(
            all_logits,
            target,
            average_log_prob=True,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return chosen_logps, rejected_logps, chosen_nll_loss

    def get_batch_loss_metrics(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
    ):
        policy_chosen_logps, policy_rejected_logps, nll_loss = (
            self.concatenated_forward(_input, weight, target, bias)
        )

        dpo_losses = self.dpo_loss(policy_chosen_logps, policy_rejected_logps)
        loss = nll_loss + dpo_losses.mean()
        return loss
