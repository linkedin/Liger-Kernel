# High-Level APIs

## AutoModel

| **AutoModel Variant** | **API** |
|------------------------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

This API extends the implementation of the `AutoModelForCausalLM` within the `transformers` library from Hugging Face.

::: liger_kernel.transformers.AutoLigerKernelForCausalLM
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

!!! Example "Try it Out"
    You can experiment as shown in this example [here](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#1-use-autoligerkernelforcausallm).

---

## Patching

You can also use the Patching APIs to use the kernels for a specific model architecture.

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`      | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`      | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`     | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL    | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`   | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy   |
| Phi3 & Phi3.5 | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |

### Function Signatures

::: liger_kernel.transformers.apply_liger_kernel_to_llama
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_mllama
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_mistral
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_mixtral
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma2
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen2
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.transformers.apply_liger_kernel_to_phi3
    options:
      extra:
        show_docstring: true
        show_signature: true

---

## Megatron-LM

Liger also exposes a patch for the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
training framework, replacing Megatron's native RMSNorm and both vocab-parallel
cross-entropy paths (fused and unfused) with Liger's Triton kernels.

| **Framework** | **API**                                                | **Supported Operations** |
|---------------|--------------------------------------------------------|--------------------------|
| Megatron-LM   | `liger_kernel.megatron.apply_liger_kernel_to_megatron` | RMSNorm, CrossEntropyLoss |

**Scope**: Initial release supports `tensor_model_parallel_size=1` only for
cross-entropy. Vocab-parallel cross-entropy (TP>1) is follow-up work — with
TP>1, each rank holds a sharded `[N, V/tp]` logits slice and cross-entropy
requires cross-rank all-reduces that Liger's kernel does not perform. The
patch raises a `RuntimeError` at patch time or call time if TP>1 is detected.

**Usage**:

```python
from liger_kernel.megatron import apply_liger_kernel_to_megatron

# Call before Megatron's forward pass reaches compute_language_model_loss.
# Defaults match Megatron's native CE behavior; no CE-specific config needed.
apply_liger_kernel_to_megatron(rms_norm=True, cross_entropy=True)
```

Both the fused (`config.cross_entropy_loss_fusion=True`,
`cross_entropy_fusion_impl='native'`) and unfused
(`config.cross_entropy_loss_fusion=False`) CE paths are patched in a single
call, so Megatron picks up Liger regardless of which path your config selects.

For training setups that need explicit kernel configuration (custom
`ignore_index`, `label_smoothing`, etc.), instantiate
`LigerMegatronCrossEntropy` directly and wire it into your model — see
`examples/megatron/run_mode2_hand_spec.py`.

::: liger_kernel.megatron.apply_liger_kernel_to_megatron
    options:
      extra:
        show_docstring: true
        show_signature: true
