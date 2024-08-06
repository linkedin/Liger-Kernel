from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.model.llama import lce_forward
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP, LigerSwiGLUMLP


# TODO: probably rename utils.py as hf_patcher.py to be more descriptive
def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)
    to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused lienar cross entropy loss. Default is False.
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
    to make GPU go burrr.

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
    to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
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
