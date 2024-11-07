import torch
from typing import Dict, List, Union, Literal, Tuple
import torch.nn.functional as F
import torch.nn as nn


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels = torch.where(labels == label_pad_token_id, 0, labels)

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def odds_ratio_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    beta: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the ORPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
        The `log(sigmoid(log_odds_chosen))` for logging purposes.
    """

    # Derived from Eqs. (4) and (7) from https://huggingface.co/papers/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
    )
    ratio = F.logsigmoid(log_odds)
    losses = beta * ratio

    return losses, torch.mean(ratio), torch.mean(log_odds)


def concatenated_forward(
    model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    concatenated_batch = concatenated_inputs(
        batch,
        is_encoder_decoder=is_encoder_decoder,
        label_pad_token_id=label_pad_token_id,
        padding_value=padding_value,
        device=accelerator.device,
    )
    len_chosen = batch["chosen_labels"].shape[0]

    outputs = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        use_cache=False,
    )
    all_logits = outputs.logits

    def cross_entropy_loss(logits, labels):
        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)
        return loss

    if is_encoder_decoder:
        labels = concatenated_batch["concatenated_labels"].clone()
    else:
        labels = concatenated_batch["concatenated_input_ids"].clone()
        attention_mask = concatenated_batch["concatenated_attention_mask"]
        labels = torch.where(attention_mask == 1, labels, label_pad_token_id)

    chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

    all_logps = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=True,
        is_encoder_decoder=is_encoder_decoder,
        label_pad_token_id=label_pad_token_id,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)


def get_batch_loss_metrics(
    model,
    batch: Dict[str, Union[List, torch.LongTensor]],
    train_eval: Literal["train", "eval"] = "train",
):
    """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""

    forward_output = concatenated_forward(model, batch)
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_nll_loss,
    ) = forward_output[:5]

    losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = odds_ratio_loss(
        policy_chosen_logps, policy_rejected_logps
    )
    # full ORPO loss
    loss = policy_nll_loss - losses.mean()

    return loss