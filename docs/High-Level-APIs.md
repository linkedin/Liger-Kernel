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
cross-entropy paths (fused and unfused) with Liger kernels. It can also fuse
the GPT output projection with cross-entropy.

| **Framework** | **API**                                                | **Supported Operations** |
|---------------|--------------------------------------------------------|--------------------------|
| Megatron-LM   | `liger_kernel.megatron.apply_liger_kernel_to_megatron` | RMSNorm, CrossEntropyLoss, fused output projection + CrossEntropyLoss |
| Megatron-LM   | `liger_kernel.megatron.LigerMegatronFusedLinearCrossEntropy` | Fused output projection + CrossEntropyLoss |

Both cross-entropy patches and FLCE support TP1 and TP>1. Automatic FLCE
patching requires Megatron-Core 0.18 or newer, BF16 or FP16, a native
`ColumnParallelLinear` output layer, and no sequence parallelism,
gradient-accumulation fusion, or gathered logits. Unsupported configurations
raise an explicit error.

**Usage**:

```python
from liger_kernel.megatron import apply_liger_kernel_to_megatron

apply_liger_kernel_to_megatron(
    rms_norm=True,
    cross_entropy=True,
    fused_linear_cross_entropy=True,
)
```

Both the fused (`config.cross_entropy_loss_fusion=True`,
`cross_entropy_fusion_impl='native'`) and unfused
(`config.cross_entropy_loss_fusion=False`) CE paths are patched in a single
call. When `fused_linear_cross_entropy=True`, labeled standard GPT forwards
bypass both paths and use FLCE directly. For custom integrations, use
`LigerMegatronFusedLinearCrossEntropy` with replicated hidden states, a
contiguous local vocabulary shard, and global target indices.

For training setups that need explicit kernel configuration (custom
`ignore_index`, `label_smoothing`, etc.), instantiate
`LigerMegatronCrossEntropy` directly and wire it into your model — see
`examples/megatron/run_mode2_hand_spec.py`.

::: liger_kernel.megatron.LigerMegatronFusedLinearCrossEntropy
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.megatron.liger_megatron_fused_linear_cross_entropy_output_processor
    options:
      extra:
        show_docstring: true
        show_signature: true

::: liger_kernel.megatron.apply_liger_kernel_to_megatron
    options:
      extra:
        show_docstring: true
        show_signature: true
