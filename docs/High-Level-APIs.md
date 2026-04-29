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
training framework, replacing Megatron's native
`fused_vocab_parallel_cross_entropy` with Liger's Triton cross-entropy kernel.

| **Framework** | **API**                                                | **Supported Operations** |
|---------------|--------------------------------------------------------|--------------------------|
| Megatron-LM   | `liger_kernel.megatron.apply_liger_kernel_to_megatron` | CrossEntropyLoss         |

**Scope**: Initial release supports `tensor_model_parallel_size=1` only.
Vocab-parallel cross-entropy (TP>1) is follow-up work — with TP>1, each rank
holds a sharded `[N, V/tp]` logits slice and cross-entropy requires cross-rank
all-reduces that Liger's kernel does not perform. The patch raises a
`RuntimeError` at patch time or call time if TP>1 is detected.

**Usage**:

```python
from liger_kernel.megatron import apply_liger_kernel_to_megatron

# Call before Megatron's forward pass reaches compute_language_model_loss.
# Match Megatron's config: pass the same ignore_index and label_smoothing
# values used by your training setup (Liger does not auto-detect them).
apply_liger_kernel_to_megatron(
    ignore_index=-100,
    label_smoothing=cfg.label_smoothing_factor,
)
```

Ensure Megatron's fused-CE code path is enabled in your training config (e.g.
`--cross-entropy-loss-fusion` in the Megatron-LM CLI) — if the unfused path is
selected, the patched symbol is never called.

::: liger_kernel.megatron.apply_liger_kernel_to_megatron
    options:
      extra:
        show_docstring: true
        show_signature: true
