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
cross-entropy paths (fused and unfused) with Liger kernels. An additional
opt-in patch fuses the GPT output projection with vocab-parallel
cross-entropy, avoiding a separately returned logits tensor between the
projection and loss.

| **Framework** | **API**                                                | **Supported Operations** |
|---------------|--------------------------------------------------------|--------------------------|
| Megatron-LM   | `liger_kernel.megatron.apply_liger_kernel_to_megatron` | RMSNorm, CrossEntropyLoss, fused output projection + CrossEntropyLoss |
| Megatron-LM   | `liger_kernel.megatron.LigerMegatronFusedLinearCrossEntropy` | Fused output projection + CrossEntropyLoss |

**Scope**: Both cross-entropy patches and FLCE support TP1 and TP>1. The FLCE
patch requires Megatron-Core 0.18 or newer and a native
`ColumnParallelLinear` output layer in BF16 or FP16. It intentionally rejects
gathered logits, sequence parallelism, gradient-accumulation fusion, deferred
embedding wgrad, disabled output dgrad reduction, MTP, MuP output scaling, and
separately returned output bias. A caller-supplied `output_processor` takes
precedence and is left unchanged.

The patch installs FLCE through Megatron's `GPTModel._postprocess`
`output_processor` hook. For custom model integrations, the
`LigerMegatronFusedLinearCrossEntropy` module accepts replicated hidden
states, the calling rank's contiguous `[V_local, H]` output-weight shard, and
global target indices. The default implementation uses Triton local kernels
and NCCL. Set `LIGER_KERNEL_IMPL=cutile` or `cutedsl` before importing Liger
to select a CuTile local path or the SM100 CuTe DSL persistent projection.

**Usage**:

```python
from liger_kernel.megatron import apply_liger_kernel_to_megatron

# Call before constructing the GPT model. FLCE is separately opt-in.
apply_liger_kernel_to_megatron(
    rms_norm=True,
    cross_entropy=True,
    fused_linear_cross_entropy=True,
)

# Or wire FLCE into a custom vocab-sharded output layer directly.
from liger_kernel.megatron import LigerMegatronFusedLinearCrossEntropy

loss_fn = LigerMegatronFusedLinearCrossEntropy(ignore_index=-100)
loss = loss_fn(hidden, local_output_weight, global_targets, tp_group=tp_group)
```

Both the fused (`config.cross_entropy_loss_fusion=True`,
`cross_entropy_fusion_impl='native'`) and unfused
(`config.cross_entropy_loss_fusion=False`) CE paths are patched in a single
call. When `fused_linear_cross_entropy=True`, labeled standard GPT forwards
bypass both materialized-logit paths and use FLCE directly.

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
