from types import MethodType

import torch
import torch.nn.functional as F

from liger_kernel.ops import LigerQwen4ExpGRWriteFunction
from liger_kernel.ops import LigerQwen4ExpHyperConnectionPreFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormFusedFunction
from liger_kernel.ops.qwen4_exp import LigerGroupRMSNormWrite4Function
from liger_kernel.ops.qwen4_exp import qwen4_exp_ngram_hash
from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerExperts
from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

_NATIVE_NGRAM_FORWARD_ATTR = "_liger_qwen4_exp_native_ngram_forward"
_NATIVE_GATED_RESIDUAL_FORWARD_ATTR = "_liger_qwen4_exp_native_gated_residual_forward"
_NATIVE_DECODER_FORWARD_ATTR = "_liger_qwen4_exp_native_decoder_forward"
_NATIVE_MLP_FORWARD_ATTR = "_liger_qwen4_exp_native_mlp_forward"
_NATIVE_EXPERTS_FORWARD_ATTR = "_liger_qwen4_exp_native_experts_forward"


def _call_saved_native_forward(self, attribute, *args, **kwargs):
    native_forward = self.__dict__.get(attribute)
    if native_forward is None:
        native_forward = getattr(type(self), attribute)
    if getattr(native_forward, "__self__", None) is not None:
        return native_forward(*args, **kwargs)
    return native_forward(self, *args, **kwargs)


def _is_nvidia_cuda_tensor(tensor):
    return tensor.device.type == "cuda" and not is_hip()


def _has_liger_swiglu_activation(config):
    return config.hidden_act in ("silu", "swish")


def _can_use_liger_experts(config):
    return _has_liger_swiglu_activation(config) and getattr(config, "_experts_implementation", None) in (
        None,
        "eager",
    )


def liger_qwen4_exp_mlp_forward(self, hidden_states):
    """Use Liger SwiGLU only for activations supported by its SiLU-mul kernel."""
    if not _has_liger_swiglu_activation(self.config):
        return _call_saved_native_forward(self, _NATIVE_MLP_FORWARD_ATTR, hidden_states)
    return LigerQwen3MoeSwiGLUMLP.forward(self, hidden_states)


def liger_qwen4_exp_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Preserve HF ExpertsInterface dispatch for every explicitly selected non-eager backend."""
    if not _can_use_liger_experts(self.config):
        return _call_saved_native_forward(
            self,
            _NATIVE_EXPERTS_FORWARD_ATTR,
            hidden_states,
            top_k_index,
            top_k_weights,
        )
    return LigerExperts.forward(self, hidden_states, top_k_index, top_k_weights)


def _can_use_liger_hyper_connection(module, runtime_tensor):
    if not _is_nvidia_cuda_tensor(runtime_tensor):
        return False
    runtime_device = runtime_tensor.device
    execution_device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
    if execution_device is not None and torch.device(execution_device) != runtime_device:
        return False
    return all(parameter.device == runtime_device for parameter in module.parameters()) and all(
        buffer.device == runtime_device for buffer in module.buffers()
    )


def _uses_liger_hyper_connection(module, runtime_tensor):
    current_forward = getattr(module.forward, "__func__", module.forward)
    # Hooks must observe native injection weights, not the decoder's internal raw logits.
    return (
        current_forward is liger_qwen4_exp_gated_residual_forward
        and not _has_module_hooks(module)
        and _can_use_liger_hyper_connection(module, runtime_tensor)
    )


def _has_module_hooks(module):
    return any(
        getattr(module, f"_{hook_type}_hooks", None)
        or getattr(torch.nn.modules.module, f"_global_{hook_type}_hooks", None)
        for hook_type in ("forward_pre", "forward", "backward_pre", "backward")
    )


def _can_use_write4_linear(module):
    if type(module) is not torch.nn.Linear or module.bias is not None:
        return False
    if "forward" in module.__dict__:
        return False
    if hasattr(module, "_hf_hook") or hasattr(module, "_old_forward"):
        return False
    if _has_module_hooks(module):
        return False
    return type(module.weight) is torch.nn.Parameter and module.weight.layout == torch.strided


def _can_fuse_qwen4_exp_rms_norm(module):
    """Direct-read only an unwrapped Liger norm whose module call has no hooks."""
    if not getattr(module, "_liger_rms_norm_patched", False):
        return False
    if getattr(module.forward, "__func__", module.forward) is not LigerRMSNorm.forward:
        return False
    if hasattr(module, "_hf_hook") or hasattr(module, "_old_forward"):
        return False
    if hasattr(module, "parametrizations") and len(module.parametrizations) != 0:
        return False
    if _has_module_hooks(module):
        return False
    weight = getattr(module, "weight", None)
    return type(weight) is torch.nn.Parameter and weight.layout == torch.strided


def liger_qwen4_exp_ngram_embedding_forward(self, input_ids, past_key_values):
    """Use Liger hashing on CUDA tensors and native HF behavior on other devices."""
    input_device = input_ids.device
    hash_metadata = (
        self.layer_multipliers,
        self.ngram_heads_vocab_sizes,
        self.ngram_heads_offsets,
    )
    if not _is_nvidia_cuda_tensor(input_ids) or any(buffer.device != input_device for buffer in hash_metadata):
        return _call_saved_native_forward(self, _NATIVE_NGRAM_FORWARD_ATTR, input_ids, past_key_values)

    input_ids = input_ids.long()
    if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
        previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
    else:
        previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)

    if previous_context.device != input_device:
        return _call_saved_native_forward(self, _NATIVE_NGRAM_FORWARD_ATTR, input_ids, past_key_values)

    if past_key_values is not None:
        input_ids_to_cache = input_ids
        if (
            not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
            and input_ids.shape[1] < self.context_len
        ):
            input_ids_to_cache = F.pad(
                input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
            )
        past_key_values.update_conv_state(
            input_ids_to_cache,
            self.layer_idx,
            state_idx=2,
            conv_kernel_size=self.context_len,
        )

    ngram_ids = qwen4_exp_ngram_hash(
        previous_context,
        input_ids,
        self.layer_multipliers,
        self.ngram_heads_vocab_sizes,
        self.ngram_heads_offsets,
        self.eos_token_id,
    )
    execution_device = self.ngram_embedding.weight.device if self.ngram_embedding.weight.device.type != "meta" else None
    return self.ngram_embedding(ngram_ids.to(execution_device)).to(input_device).flatten(-2)


def liger_qwen4_exp_gated_residual_forward(self, hyper_input, *, return_write_logits=False):
    """Run the Qwen4Exp feature-wise HyperConnection pre-mix.

    The grouped RMSNorm fusion reuses normalized activations across the
    HyperConnection consumers. Its backward combines consumer gradients
    before running the RMSNorm gradient kernel.
    """
    if not _can_use_liger_hyper_connection(self, hyper_input):
        if return_write_logits:
            raise RuntimeError(
                "return_write_logits=True requires the Qwen4Exp Liger HyperConnection path on the input's "
                "NVIDIA CUDA device. The decoder must fall back before requesting raw write logits."
            )
        return _call_saved_native_forward(self, _NATIVE_GATED_RESIDUAL_FORWARD_ATTR, hyper_input)

    expected_features = self.hc_count * self.hidden_size
    if hyper_input.shape[-1] != expected_features:
        raise ValueError(f"Expected {expected_features} hyper-connection features, got {hyper_input.shape[-1]}.")
    n_groups = self.hc_count
    if n_groups != 4:
        # The fused Qwen4 HyperConnection path is specialized for hc_count=4.
        # Preserve HF's exact operation and BF16 reduction order for every
        # other valid count instead of partially entering the fused path.
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = self.input_mix_weight_up(input_mix_weight)
        input_mix_weight = torch.sigmoid(input_mix_weight).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        if self.block_inject_weight is None:
            if return_write_logits:
                raise RuntimeError(
                    "return_write_logits=True requires block_inject_weight "
                    "(Qwen4Exp GatedResidual must use combine mode)."
                )
            return mixed_input
        write_logits = self.block_inject_weight(hyper_input_normed) / self.hc_count
        if return_write_logits:
            return mixed_input, hyper_input, write_logits
        return mixed_input, hyper_input, 2 * torch.sigmoid(write_logits)

    if not _can_fuse_qwen4_exp_rms_norm(self.hc_norm):
        hyper_input_normed = self.hc_norm(hyper_input)
        norm_for_down = norm_for_write = norm_for_pre = hyper_input_normed
    else:
        weight = self.hc_norm.weight
        eps = getattr(self.hc_norm, "eps", None)
        if eps is None:
            eps = getattr(self.hc_norm, "variance_epsilon", 1e-6)
        offset = getattr(self.hc_norm, "offset", 1.0)
        casting_mode = getattr(self.hc_norm, "casting_mode", "gemma")
        if (
            return_write_logits
            and hyper_input.dtype == torch.bfloat16
            and _can_use_write4_linear(self.block_inject_weight)
        ):
            norm_for_down, norm_for_pre, write_logits, hyper_input = LigerGroupRMSNormWrite4Function.apply(
                hyper_input,
                weight,
                self.block_inject_weight.weight,
                eps,
                offset,
                casting_mode,
                n_groups,
                True,
            )
            norm_for_write = None
        else:
            norm_for_down, norm_for_write, norm_for_pre = LigerGroupRMSNormFusedFunction.apply(
                hyper_input, weight, eps, offset, casting_mode, n_groups
            )

    input_mix_hidden = F.silu(self.input_mix_weight_down(norm_for_down) / self.hc_count)
    input_mix_logits = self.input_mix_weight_up(input_mix_hidden)
    mixed_input = LigerQwen4ExpHyperConnectionPreFunction.apply(
        input_mix_logits,
        norm_for_pre,
        self.hc_count,
    )
    if self.block_inject_weight is None:
        if return_write_logits:
            raise RuntimeError(
                "return_write_logits=True requires block_inject_weight (Qwen4Exp GatedResidual must use combine mode)."
            )
        return mixed_input
    if norm_for_write is not None:
        write_logits = self.block_inject_weight(norm_for_write) / self.hc_count
    if return_write_logits:
        return mixed_input, hyper_input, write_logits
    injection_weights = 2 * torch.sigmoid(write_logits)
    return mixed_input, hyper_input, injection_weights


def liger_qwen4_exp_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    conv_mask=None,
    past_key_values=None,
    ple_input_ids=None,
    **kwargs,
):
    """HF-equivalent decoder forward with specialized Qwen4 hyper-connection post-injection kernels."""
    if not (
        _uses_liger_hyper_connection(self.attn_hyper_connection, hidden_states)
        and _uses_liger_hyper_connection(self.mlp_hyper_connection, hidden_states)
    ):
        return _call_saved_native_forward(
            self,
            _NATIVE_DECODER_FORWARD_ATTR,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            past_key_values=past_key_values,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

    if self.ple is not None:
        hidden_states = hidden_states + self.ple(
            hidden_states,
            ple_input_ids,
            past_key_values,
            conv_mask=conv_mask,
        )

    hidden_states, hyper_input, write_logits = self.attn_hyper_connection(
        hidden_states,
        return_write_logits=True,
    )
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states,
            cache_params=past_key_values,
            attention_mask=conv_mask,
            **kwargs,
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    hidden_states = LigerQwen4ExpGRWriteFunction.apply(
        hidden_states,
        hyper_input,
        write_logits,
    )

    hidden_states, hyper_input, write_logits = self.mlp_hyper_connection(
        hidden_states,
        return_write_logits=True,
    )
    hidden_states = self.mlp(hidden_states)
    return LigerQwen4ExpGRWriteFunction.apply(
        hidden_states,
        hyper_input,
        write_logits,
    )


def _patch_module_forward(module, native_attribute, liger_forward):
    # Keep Accelerate's public pre/post-hook wrapper and replace only its inner callable.
    forward_attribute = "_old_forward" if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward") else "forward"
    current_callable = getattr(module, forward_attribute)
    current_forward = getattr(current_callable, "__func__", current_callable)
    if current_forward is not liger_forward:
        module.__dict__[native_attribute] = current_forward
    setattr(module, forward_attribute, MethodType(liger_forward, module))


def _patch_class_forward(module_class, native_attribute, liger_forward):
    if module_class.forward is not liger_forward:
        setattr(module_class, native_attribute, module_class.forward)
    module_class.forward = liger_forward


def patch_qwen4_exp_text_module_for_ngram(module, ngram_embedding_class):
    if isinstance(module, ngram_embedding_class):
        _patch_module_forward(module, _NATIVE_NGRAM_FORWARD_ATTR, liger_qwen4_exp_ngram_embedding_forward)


def patch_qwen4_exp_text_ngram_class(ngram_embedding_class):
    _patch_class_forward(ngram_embedding_class, _NATIVE_NGRAM_FORWARD_ATTR, liger_qwen4_exp_ngram_embedding_forward)


def patch_qwen4_exp_text_swiglu_classes(mlp_class, experts_class):
    for module_class, native_attribute, liger_forward in (
        (mlp_class, _NATIVE_MLP_FORWARD_ATTR, liger_qwen4_exp_mlp_forward),
        (experts_class, _NATIVE_EXPERTS_FORWARD_ATTR, liger_qwen4_exp_experts_forward),
    ):
        _patch_class_forward(module_class, native_attribute, liger_forward)


def patch_qwen4_exp_text_module_for_swiglu(module, mlp_class, experts_class):
    if isinstance(module, mlp_class):
        native_attribute = _NATIVE_MLP_FORWARD_ATTR
        liger_forward = liger_qwen4_exp_mlp_forward
    elif isinstance(module, experts_class):
        native_attribute = _NATIVE_EXPERTS_FORWARD_ATTR
        liger_forward = liger_qwen4_exp_experts_forward
    else:
        return

    _patch_module_forward(module, native_attribute, liger_forward)


def patch_qwen4_exp_text_hyper_connection_classes(gated_residual_class, decoder_layer_class):
    for module_class, native_attribute, liger_forward in (
        (
            gated_residual_class,
            _NATIVE_GATED_RESIDUAL_FORWARD_ATTR,
            liger_qwen4_exp_gated_residual_forward,
        ),
        (decoder_layer_class, _NATIVE_DECODER_FORWARD_ATTR, liger_qwen4_exp_decoder_layer_forward),
    ):
        _patch_class_forward(module_class, native_attribute, liger_forward)


def patch_qwen4_exp_text_module_for_hyper_connection(module, gated_residual_class, decoder_layer_class):
    if isinstance(module, gated_residual_class):
        native_attribute = _NATIVE_GATED_RESIDUAL_FORWARD_ATTR
        liger_forward = liger_qwen4_exp_gated_residual_forward
    elif isinstance(module, decoder_layer_class):
        native_attribute = _NATIVE_DECODER_FORWARD_ATTR
        liger_forward = liger_qwen4_exp_decoder_layer_forward
    else:
        return

    _patch_module_forward(module, native_attribute, liger_forward)
