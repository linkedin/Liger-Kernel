from functools import partial

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.model.llama import lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP, LigerSwiGLUMLP


def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused lienar cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.llama import modeling_llama

    if rope:
        modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_llama.LlamaRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_llama.LlamaMLP = LigerSwiGLUMLP
    if cross_entropy:
        modeling_llama.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_llama.LlamaForCausalLM.forward = lce_forward


def apply_liger_kernel_to_mistral(
    rope: bool = True,
    cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Mistral models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """

    from transformers.models.mistral import modeling_mistral

    if rope:
        modeling_mistral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mistral.MistralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_mistral.CrossEntropyLoss = LigerCrossEntropyLoss
    if swiglu:
        modeling_mistral.MistralMLP = LigerSwiGLUMLP


def apply_liger_kernel_to_mixtral(
    rope: bool = True,
    cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Mixtral models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """

    from transformers.models.mixtral import modeling_mixtral

    if rope:
        modeling_mixtral.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_mixtral.MistralRMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_mixtral.CrossEntropyLoss = LigerCrossEntropyLoss
    if swiglu:
        modeling_mixtral.MixtralBlockSparseTop2MLP = LigerBlockSparseTop2MLP


def apply_liger_kernel_to_gemma(
    rope: bool = True,
    cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma2 models
    to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
    """
    from transformers.models.gemma import modeling_gemma

    if rope:
        modeling_gemma.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L109
        modeling_gemma.GemmaRMSNorm = partial(LigerRMSNorm, offset=1.0, init_fn="zeros")
    if cross_entropy:
        modeling_gemma.CrossEntropyLoss = LigerCrossEntropyLoss
    if geglu:
        modeling_gemma.GemmaMLP = LigerGEGLUMLP


def apply_liger_kernel_to_qwen2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused lienar cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2 import modeling_qwen2

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    if cross_entropy:
        modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward
    if swiglu:
        modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
