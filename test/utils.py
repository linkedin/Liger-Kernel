import importlib
import json
import os
import random

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from tokenizers import AddedToken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from liger_kernel.utils import infer_device

device = infer_device()


def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if device == "cuda":
        # If you are using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == "xpu":
        # If you are using XPU
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5):
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    tensor1 (torch.Tensor): First tensor to compare.
    tensor2 (torch.Tensor): Second tensor to compare.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        raise AssertionError("Input tensors must have the same shape.")

    # Calculate the difference between the tensors
    diff = torch.abs(tensor1 - tensor2)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(tensor2)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(tensor1), torch.isnan(tensor2))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(torch.isposinf(tensor1), torch.isposinf(tensor2))
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(torch.isneginf(tensor1), torch.isneginf(tensor2))

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Check if all elements are close
    all_close = num_mismatched == 0

    # Raise AssertionError with detailed information if there are mismatches
    if not all_close and num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        print_count = min(max_print, num_mismatched)
        for index in mismatched_indices[:print_count]:
            i = tuple(index.tolist())
            mismatch_details.append(f"Mismatch at index {i}: tensor1[{i}] = {tensor1[i]}, tensor2[{i}] = {tensor2[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")

        raise AssertionError("\n".join(mismatch_details))


# Pre-tokenized dataset using Mistral-7B tokenizer used for convergence tests
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/tiny_shakespeare_tokenized")

UNTOKENIZED_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/tiny_shakespeare.txt")

FAKE_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/fake_configs")


@dataclass
class MiniModelConfig:
    liger_kernel_patch_func: callable
    liger_kernel_patch_revert_func: callable
    model_class: PreTrainedModel
    mini_model_config: PretrainedConfig


def simple_collate_fn(data: List[Dict[str, Any]]):
    """A basic collate function to use for DataLoader"""

    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in data])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in data])
    labels = input_ids.clone()

    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )


def multimodal_collate_fn(data: List[Dict[str, Any]]):
    """A collate function to use for DataLoader for multimodal models"""
    batch = {}
    keys = set(data[0].keys())

    input_ids = torch.cat([torch.tensor(item["input_ids"]) for item in data])
    keys.remove("input_ids")
    batch["input_ids"] = input_ids

    labels = input_ids.clone()
    batch["labels"] = labels

    # Collate all other keys, e.g. pixel_values, attention_mask, image_grid_thw, etc
    for key in keys:
        batch[key] = torch.cat([item[key] for item in data])

    return BatchEncoding(batch)


def load_tokenizer_config(config_path: str) -> dict:
    """Load and process tokenizer configuration from a JSON file."""
    with open(config_path) as reader:
        tokenizer_config = json.load(reader)
    tokenizer_config["added_tokens_decoder"] = {
        k: AddedToken(**v) for k, v in tokenizer_config["added_tokens_decoder"].items()
    }
    return tokenizer_config


def train_bpe_tokenizer(special_tokens: List[str], unk_token: str = "<|unk|>"):
    """
    Train a tokenizer using the BPE algorithm.

    Parameters:
    unk_token (str): The token to use for unknown tokens.
    special_tokens (List[str]): A list of special tokens to use.

    Returns:
    Tokenizer: The trained tokenizer.
    """
    # Add unk_token to special_tokens if not already present
    if unk_token not in special_tokens:
        special_tokens.append(unk_token)

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=special_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    file = [UNTOKENIZED_DATASET_PATH]
    tokenizer.train(file, trainer)

    return tokenizer


def supports_bfloat16():
    if device == "cuda":
        return torch.cuda.get_device_capability() >= (8, 0)  # Ampere and newer
    elif device == "xpu":
        return True
    else:
        return False


def revert_liger_kernel_to_granite(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Granite.
    """

    from transformers.models.granite import modeling_granite

    importlib.reload(modeling_granite)
    model_config.model_class = modeling_granite.GraniteForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_llama(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Llama.
    """

    from transformers.models.llama import modeling_llama

    importlib.reload(modeling_llama)
    model_config.model_class = modeling_llama.LlamaForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_mllama(model_config: MiniModelConfig, model_type: str = "causal_lm"):
    """
    Revert all Liger kernel patches applied to MLlama.
    """

    assert model_type in [
        "causal_lm",
        "conditional_generation",
    ], f'model_type must be "causal_lm" or "conditional_generation", Got: {model_type}'
    import torch.nn as nn

    from transformers.models.mllama import modeling_mllama

    importlib.reload(nn)
    importlib.reload(modeling_mllama)
    if model_type == "causal_lm":
        model_config.model_class = modeling_mllama.MllamaForCausalLM
    else:
        model_config.model_class = modeling_mllama.MllamaForConditionalGeneration

    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_mistral(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Mistral.
    """

    from transformers.models.mistral import modeling_mistral

    importlib.reload(modeling_mistral)
    model_config.model_class = modeling_mistral.MistralForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_mixtral(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Mixtral.
    """

    from transformers.models.mixtral import modeling_mixtral

    importlib.reload(modeling_mixtral)
    model_config.model_class = modeling_mixtral.MixtralForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_gemma(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Gemma.
    """

    from transformers.models.gemma import modeling_gemma

    importlib.reload(modeling_gemma)
    model_config.model_class = modeling_gemma.GemmaForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_gemma2(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Gemma2.
    """

    from transformers.models.gemma2 import modeling_gemma2

    importlib.reload(modeling_gemma2)
    model_config.model_class = modeling_gemma2.Gemma2ForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_Paligemma(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Gemma2.
    """

    from transformers.models.gemma2 import modeling_gemma2
    from transformers.models.paligemma import modeling_paligemma
    from transformers.models.siglip import modeling_siglip

    importlib.reload(modeling_siglip)
    importlib.reload(modeling_gemma2)
    importlib.reload(modeling_paligemma)
    model_config.model_class = modeling_paligemma.PaliGemmaForConditionalGeneration
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_qwen2(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Qwen2.
    """

    from transformers.models.qwen2 import modeling_qwen2

    importlib.reload(modeling_qwen2)
    model_config.model_class = modeling_qwen2.Qwen2ForCausalLM

    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_qwen2_vl(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Qwen2-VL.
    """
    from transformers.models.qwen2_vl import modeling_qwen2_vl

    importlib.reload(modeling_qwen2_vl)
    model_config.model_class = modeling_qwen2_vl.Qwen2VLForConditionalGeneration
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_qwen2_5_vl(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Qwen2.5-VL.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    importlib.reload(modeling_qwen2_5_vl)
    model_config.model_class = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_phi3(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Phi3.
    """

    from transformers.models.phi3 import modeling_phi3

    importlib.reload(modeling_phi3)
    model_config.model_class = modeling_phi3.Phi3ForCausalLM
    print("Liger kernel patches have been reverted.")


def revert_liger_kernel_to_olmo2(model_config: MiniModelConfig):
    """
    Revert all Liger kernel patches applied to Olmo2.
    """

    from transformers.models.olmo2 import modeling_olmo2

    importlib.reload(modeling_olmo2)
    model_config.model_class = modeling_olmo2.Olmo2ForCausalLM
    print("Liger kernel patches have been reverted.")


class HFAlignmentLoss:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        ignore_index: int = -100,
        use_ref_model: bool = False,
        unpaired: bool = False,
        compute_nll_loss: bool = True,
        **kwargs,
    ):
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.use_ref_model = use_ref_model
        self.unpaired = unpaired
        self.compute_nll_loss = compute_nll_loss

    @abstractmethod
    def alignment_loss(self):
        pass

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of ignore_index are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            is_encoder_decoder: Whether the model is an encoder-decoder model.
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != self.ignore_index

        # dummy token; we'll ignore the losses on these tokens later
        labels = torch.where(labels == self.ignore_index, 0, labels)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_ref_logps(
        self,
        ref_input: torch.FloatTensor,
        ref_weight: torch.FloatTensor,
        target: torch.LongTensor,
        ref_bias: torch.FloatTensor,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
    ):
        """Compute the log probabilities of the given labels under the given reference model."""

        with torch.no_grad():
            ref_logits = ref_input @ ref_weight.t()
            if ref_bias is not None:
                ref_logits = ref_logits + ref_bias
            ref_all_logps = self.get_batch_logps(ref_logits, target, average_log_prob=average_log_prob)

            if self.unpaired and preference_labels is not None:
                # Split based on preference labels
                return (
                    ref_all_logps[preference_labels],
                    ref_all_logps[~preference_labels],
                )
            else:
                # Original paired behavior - split in half
                return (
                    ref_all_logps[: ref_input.shape[0] // 2],
                    ref_all_logps[ref_input.shape[0] // 2 :],
                )

    def concatenated_forward(
        self,
        _input: torch.FloatTensor,
        weight: torch.FloatTensor,
        target: torch.LongTensor,
        bias: Optional[torch.FloatTensor] = None,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
        nll_target: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        len_chosen = _input.shape[0] // 2

        outputs = _input @ weight.t()
        if bias is not None:
            outputs = outputs + bias
        all_logits = outputs.float()

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = nll_target if nll_target is not None else target
        chosen_nll_loss = torch.tensor(0.0, device=all_logits.device)
        if self.compute_nll_loss:
            chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            target,
            average_log_prob=average_log_prob,
        )

        if self.unpaired and preference_labels is not None:
            # Split based on labels tensor
            chosen_logps = all_logps[preference_labels]
            rejected_logps = all_logps[~preference_labels]
            chosen_logits = all_logits[preference_labels]
            rejected_logits = all_logits[~preference_labels]
        else:
            # Original paired behavior - split in half
            len_chosen = _input.shape[0] // 2
            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]
            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    def get_batch_loss_metrics(
        self,
        weight: torch.FloatTensor,
        _input: torch.FloatTensor,
        target: torch.LongTensor,
        bias: torch.FloatTensor = None,
        ref_input: torch.FloatTensor = None,
        ref_weight: torch.FloatTensor = None,
        ref_bias: torch.FloatTensor = None,
        average_log_prob: bool = True,
        preference_labels: torch.Tensor = None,
        nll_target: torch.LongTensor = None,
        **loss_kwargs,
    ):
        """Compute the loss metrics for the given batch of inputs for train or test."""
        forward_output = self.concatenated_forward(
            _input, weight, target, bias, average_log_prob, preference_labels, nll_target
        )
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]

        if self.use_ref_model:
            ref_chosen_logps, ref_rejected_logps = self.get_ref_logps(
                ref_input,
                ref_weight,
                target,
                ref_bias,
                average_log_prob,
                preference_labels,
            )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps
        alignment_loss_outputs = self.alignment_loss(policy_chosen_logps, policy_rejected_logps, **loss_kwargs)
        if isinstance(alignment_loss_outputs, tuple):
            losses, *aggregated_aux_outputs = alignment_loss_outputs
        else:
            losses, aggregated_aux_outputs = alignment_loss_outputs, []

        loss = policy_nll_loss * self.alpha + losses.mean()

        if not self.unpaired:
            return_vars = (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits.detach().mean(),
                policy_rejected_logits.detach().mean(),
                policy_nll_loss,
            )
            return loss, (*return_vars, *aggregated_aux_outputs)
        else:
            return_vars = (
                policy_chosen_logps.detach().sum(),
                policy_rejected_logps.detach().sum(),
                policy_chosen_logits.detach().sum(),
                policy_rejected_logits.detach().sum(),
            )
            return loss, (*return_vars, *aggregated_aux_outputs)


class HFDistillationLoss:
    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1,
    ):
        self.weight_hard_loss = weight_hard_loss
        self.weight_soft_loss = weight_soft_loss
        self.ignore_index = ignore_index
        self.temperature = temperature

    @abstractmethod
    def distillation_loss(self, student_logits, teacher_logits, **loss_kwargs):
        """Abstract method for computing distillation loss."""
        pass

    def concatenated_forward(
        self,
        student_input: torch.FloatTensor,
        student_weight: torch.FloatTensor,
        teacher_input: torch.FloatTensor,
        teacher_weight: torch.FloatTensor,
        target: torch.LongTensor,
        student_bias: torch.FloatTensor = None,
        teacher_bias: torch.FloatTensor = None,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Compute forward pass for both student and teacher models."""

        student_batch_seq_len_size, student_hidden_size = student_input.shape
        student_input_reshaped = student_input.view(-1, student_hidden_size)
        teacher_batch_seq_len_size, teacher_hidden_size = teacher_input.shape
        teacher_input_reshaped = teacher_input.view(-1, teacher_hidden_size)

        student_outputs = student_input_reshaped @ student_weight.t()
        if student_bias is not None:
            student_outputs = student_outputs + student_bias

        with torch.no_grad():
            teacher_outputs = teacher_input_reshaped @ teacher_weight.t()
            if teacher_bias is not None:
                teacher_outputs = teacher_outputs + teacher_bias

        student_logits = student_outputs.view(student_batch_seq_len_size, -1).float()
        teacher_logits = teacher_outputs.view(teacher_batch_seq_len_size, -1).float()

        if torch.all(target == self.ignore_index):
            return torch.tensor(0.0)

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = target
        ce_loss = cross_entropy_loss(
            student_logits.view(-1, student_logits.shape[-1]),
            labels.view(-1),
        )

        return (
            student_logits,
            teacher_logits,
            ce_loss,
        )

    def get_batch_loss_metrics(
        self,
        student_input: torch.FloatTensor,
        student_weight: torch.FloatTensor,
        teacher_input: torch.FloatTensor,
        teacher_weight: torch.FloatTensor,
        target: torch.LongTensor,
        student_bias: torch.FloatTensor = None,
        teacher_bias: torch.FloatTensor = None,
        **loss_kwargs,
    ):
        """Compute the distillation loss metrics for the given batch."""
        forward_output = self.concatenated_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            target,
            student_bias,
            teacher_bias,
        )
        (
            student_logits,
            teacher_logits,
            hard_loss,
        ) = forward_output

        student_logits /= self.temperature
        teacher_logits /= self.temperature

        soft_loss = self.distillation_loss(student_logits, teacher_logits, **loss_kwargs)
        # full loss
        loss = self.weight_hard_loss * hard_loss + self.weight_soft_loss * soft_loss
        return loss
