import warnings

from liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityFunction
from liger_kernel.chunked_loss.cpo_loss import LigerFusedLinearCPOFunction
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction
from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction
from liger_kernel.chunked_loss.kto_loss import LigerFusedLinearKTOFunction
from liger_kernel.chunked_loss.orpo_loss import LigerFusedLinearORPOFunction
from liger_kernel.chunked_loss.simpo_loss import LigerFusedLinearSimPOFunction

liger_fused_linear_orpo = LigerFusedLinearORPOFunction.apply
liger_fused_linear_dpo = LigerFusedLinearDPOFunction.apply
liger_fused_linear_jsd = LigerFusedLinearJSDFunction.apply
liger_fused_linear_cosine = LigerFusedLinearCosineSimilarityFunction.apply
liger_fused_linear_cpo = LigerFusedLinearCPOFunction.apply
liger_fused_linear_simpo = LigerFusedLinearSimPOFunction.apply
liger_fused_linear_kto = LigerFusedLinearKTOFunction.apply


def liger_fused_linear_grpo(
    _input,
    weight,
    selected_token_ids,
    attention_mask,
    advantages,
    bias=None,
    ref_per_token_logps=None,
    old_per_token_logps=None,
    ref_input=None,
    ref_weight=None,
    ref_bias=None,
    beta=0.04,
    epsilon_low=0.2,
    epsilon_high=0.2,
    loss_type="dapo",
    max_completion_length=None,
    importance_sampling_level="token",
    sapo_temperature_pos=1.0,
    sapo_temperature_neg=1.05,
    temperature=1.0,
    compiled=True,
    use_ref_model=True,
    chunk_size=1,
    vllm_is_ratio=None,
    delta=None,
    use_bias_correction_kl=False,
    vespo_k_pos=2.0,
    vespo_lambda_pos=3.0,
    vespo_k_neg=3.0,
    vespo_lambda_neg=2.0,
    num_items_in_batch=None,
):
    if _input.device.type == "npu":
        if use_ref_model and ref_per_token_logps is None and ref_input is None:
            raise AssertionError("If use_ref_model is True, ref_per_token_logps or ref_input must be provided")
        if use_ref_model and ref_per_token_logps is not None and ref_input is not None:
            warnings.warn("Both ref_per_token_logps and ref_input are provided. Using ref_per_token_logps.")
        from liger_kernel.ops.backends._ascend.ops.fused_linear_grpo import fused_linear_grpo_forward

        loss, kl_mean, clip_ratio = fused_linear_grpo_forward(
            _input,
            weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input if use_ref_model else None,
            ref_weight=ref_weight if use_ref_model else None,
            ref_bias=ref_bias if use_ref_model else None,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
            temperature=temperature,
            vllm_is_ratio=vllm_is_ratio,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            num_items_in_batch=num_items_in_batch,
            vespo_k_pos=vespo_k_pos,
            vespo_lambda_pos=vespo_lambda_pos,
            vespo_k_neg=vespo_k_neg,
            vespo_lambda_neg=vespo_lambda_neg,
            chunk_size=chunk_size,
        )
        metrics = []
        if beta != 0.0 and kl_mean is not None:
            metrics.append(kl_mean)
        metrics.append(clip_ratio)
        return loss, metrics

    return LigerFusedLinearGRPOFunction.apply(
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias,
        ref_per_token_logps,
        old_per_token_logps,
        ref_input,
        ref_weight,
        ref_bias,
        beta,
        epsilon_low,
        epsilon_high,
        loss_type,
        max_completion_length,
        importance_sampling_level,
        sapo_temperature_pos,
        sapo_temperature_neg,
        temperature,
        compiled,
        use_ref_model,
        chunk_size,
        vllm_is_ratio,
        delta,
        use_bias_correction_kl,
        vespo_k_pos,
        vespo_lambda_pos,
        vespo_k_neg,
        vespo_lambda_neg,
        num_items_in_batch,
    )
