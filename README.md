<a name="readme-top"></a>

# Liger Kernel: Efficient Triton Kernels for LLM Training


<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;" colspan="2">Stable</th>
        <th style="padding: 10px;" colspan="2">Nightly</th>
        <th style="padding: 10px;">Discord</th>
        <th style="padding: 10px;">Gurubase (experimental)</th>
    </tr>
    <tr>
        <td style="padding: 10px;">
            <a href="https://pepy.tech/project/liger-kernel">
                <img src="https://static.pepy.tech/badge/liger-kernel" alt="Downloads (Stable)">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://pypi.org/project/liger-kernel">
                <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/liger-kernel?color=green">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://pepy.tech/project/liger-kernel-nightly">
                <img src="https://static.pepy.tech/badge/liger-kernel-nightly" alt="Downloads (Nightly)">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://pypi.org/project/liger-kernel-nightly">
                <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/liger-kernel-nightly?color=green">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://discord.gg/gpumode">
                <img src="https://dcbadge.vercel.app/api/server/gpumode?style=flat" alt="Join Our Discord">
            </a>
        </td>
        <td style="padding: 10px;">
            <a href="https://gurubase.io/g/liger-kernel">
                <img src="https://img.shields.io/badge/Gurubase-Ask%20Liger%20Kernel%20Guru-006BFF" alt="Ask Liger Kernel Guru">
            </a>
        </td>
    </tr>
</table>



<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png">

[Installation](#installation) | [Getting Started](#getting-started) | [Examples](#examples) | [APIs](#apis) | [Cite our work](#cite-this-work)

<details>
  <summary>Latest News 🔥</summary>
    
  - [2024/10/21] We have released the tech report of Liger Kernel on Arxiv: https://arxiv.org/pdf/2410.10989 
  - [2024/9/6] We release v0.2.1 ([X post](https://x.com/liger_kernel/status/1832168197002510649)). 2500+ Stars, 10+ New Contributors, 50+ PRs, 50k Downloads in two weeks!
  - [2024/8/31] CUDA MODE talk, [Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns), [Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)
  - [2024/8/23] Official release: check out our [X post](https://x.com/hsu_byron/status/1827072737673982056)

</details>


**Liger Kernel** is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU **training throughput by 20%** and reduces **memory usage by 60%**. We have implemented **Hugging Face Compatible** `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, and more to come. The kernel works out of the box with [Flash Attention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed). We welcome contributions from the community to gather the best kernels for LLM training.

## Supercharge Your Model with Liger Kernel

![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

With one line of code, Liger Kernel can increase throughput by more than 20% and reduce memory usage by 60%, thereby enabling longer context lengths, larger batch sizes, and massive vocabularies.


| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face + Liger Kernel scales up to 16K.

## Examples

### Basic

| **Example**                                    | **Description**                                                                                   | **Lightning Studio** |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train LLaMA 3-8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP | TBA                  |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 | TBA                  |

### Advanced

| **Example**                                    | **Description**                                                                                   | **Lightning Studio** |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------------|
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP      | TBA                  |

## Key Features

- **Ease of use:** Simply patch your Hugging Face model with one line of code, or compose your own model using our Liger Kernel modules.
- **Time and memory efficient:** In the same spirit as Flash-Attn, but for layers like **RMSNorm**, **RoPE**, **SwiGLU**, and **CrossEntropy**! Increases multi-GPU training throughput by 20% and reduces memory usage by 60% with **kernel fusion**, **in-place replacement**, and **chunking** techniques.
- **Exact:** Computation is exact—no approximations! Both forward and backward passes are implemented with rigorous unit tests and undergo convergence testing against training runs without Liger Kernel to ensure accuracy.
- **Lightweight:** Liger Kernel has minimal dependencies, requiring only Torch and Triton—no extra libraries needed! Say goodbye to dependency headaches!
- **Multi-GPU supported:** Compatible with multi-GPU setups (PyTorch FSDP, DeepSpeed, DDP, etc.).
- **Trainer Framework Integration**: [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory), [SFTTrainer](https://github.com/huggingface/trl/releases/tag/v0.10.1), [Hugging Face Trainer](https://github.com/huggingface/transformers/pull/32860), [SWIFT](https://github.com/modelscope/ms-swift)

## Target Audiences

- **Researchers**: Looking to compose models using efficient and reliable kernels for frontier experiments.
- **ML Practitioners**: Focused on maximizing GPU training efficiency with optimal, high-performance kernels.
- **Curious Novices**: Eager to learn how to write reliable Triton kernels to enhance training efficiency.


## Installation

### Dependencies

- `torch >= 2.1.2`
- `triton >= 2.3.0`

### Optional Dependencies

- `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

> **Note:**
> Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

To install the stable version:

```bash
$ pip install liger-kernel
```

To install the nightly version:

```bash
$ pip install liger-kernel-nightly
```

To install from source:

```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .
# or if using transformers
pip install -e .[transformers]
```

## Getting Started

There are a couple of ways to apply Liger kernels, depending on the level of customization required.

### 1. Use AutoLigerKernelForCausalLM

Using the `AutoLigerKernelForCausalLM` is the simplest approach, as you don't have to import a model-specific patching API. If the model type is supported, the modeling code will be automatically patched using the default settings.

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# This AutoModel wrapper class automatically monkey-patches the
# model with the optimized Liger kernels if the model is supported.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Apply Model-Specific Patching APIs

Using the [patching APIs](#patching), you can swap Hugging Face models with optimized Liger Kernels.

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1a. Adding this line automatically monkey-patches the model with the optimized Liger kernels
apply_liger_kernel_to_llama()

# 1b. You could alternatively specify exactly which kernels are applied
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# 2. Instantiate patched model
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. Compose Your Own Model

You can take individual [kernels](#kernels) to compose your models.

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# fuses linear + cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```


## Structure

### Source Code

- `ops/`: Core Triton operations.
- `transformers/`: PyTorch `nn.Module` implementations built on Triton operations, compliant with the `transformers` API.

### Tests

- `transformers/`: Correctness tests for the Triton-based layers.
- `convergence/`: Patches Hugging Face models with all kernels, runs multiple iterations, and compares weights, logits, and loss layer-by-layer.

### Benchmark

- `benchmark/`: Execution time and memory benchmarks compared to Hugging Face layers.

## APIs

### AutoModel

| **AutoModel Variant** | **API** |
|-----------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |


### Patching

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss         |
| Qwen2 & Qwen2.5      | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |



### Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| LayerNorm                       | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| FusedLinearCrossEntropy         | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| FusedLinearJSD                  | `liger_kernel.transformers.LigerFusedLinearJSD`             |

- **RMSNorm**: [RMSNorm](https://arxiv.org/pdf/1910.07467), which normalizes activations using their root mean square, is implemented by fusing the normalization and scaling steps into a single Triton kernel, and achieves ~3X speedup with ~3X peak memory reduction.
- **LayerNorm**: [LayerNorm](https://arxiv.org/pdf/1607.06450), which centers and normalizes activations across the feature dimension, is implemented by fusing the centering, normalization and scaling steps into a single Triton kernel, and achieves ~2X speedup.
- **RoPE**: [Rotary Positional Embedding](https://arxiv.org/pdf/2104.09864) is implemented by fusing the query and key embedding rotary into a single kernel with inplace replacement, and achieves ~3X speedup with ~3X peak memory reduction.
- **SwiGLU**: [Swish Gated Linear Units](https://arxiv.org/pdf/2002.05202), given by
$$\text{SwiGLU}(x)=\text{Swish}_{\beta}(xW+b)\otimes(xV+c)$$
, is implemented by fusing the elementwise multiplication (denoted by $\otimes$) into a single kernel with inplace replacement, and achieves parity speed with ~1.5X peak memory reduction.
- **GeGLU**: [GELU Gated Linear Units](https://arxiv.org/pdf/2002.05202), given by
$$\text{GeGLU}(x)=\text{GELU}(xW+b)\otimes(xV+c)$$
, is implemented by fusing the elementwise multiplication into a single kernel with inplace replacement, and achieves parity speed with ~1.5X peak memory reduction. Note that the [tanh approximation form of GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) is used.
- **CrossEntropy**: [Cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is implemented by computing both the loss and gradient in the forward pass with inplace replacement of input to reduce the peak memory by avoiding simultaneous materialization of both input logits and gradient. It achieves >2X speedup and >4X memory reduction for common vocab sizes (e.g., 32K, 128K, etc.).
<!-- TODO: verify vocab sizes are accurate  -->
- **FusedLinearCrossEntropy**: Peak memory usage of cross entropy loss is further improved by fusing the model head with the CE loss and chunking the input for block-wise loss and gradient calculation, a technique inspired by [Efficient Cross Entropy](https://github.com/mgmalek/efficient_cross_entropy). It achieves >4X memory reduction for 128k vocab size. **This is highly effective for large batch size, large sequence length, and large vocabulary sizes.** Please refer to the [Medusa example](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa) for individual kernel usage.
- **KLDivergence**: [KL Divergence](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) is implemented by fusing the forward into a single triton kernel, with reduction done outside the kernel. It achieves ~1.5X speed and ~15% memory reduction for 128K vocab size.
- **JSD**: [Generalized JSD](https://arxiv.org/pdf/2306.13649) (Jensen-Shannon divergence), is implemented by computing both the loss and gradient in the forward pass. It achieves ~1.5X speed and ~54% memory reduction for 128k vocab size.
- **FusedLinearJSD**: Peak memory usage of JSD loss is further improved by fusing the model head with the model head with the JSD and chunking the input for block-wise loss and gradient calculation. It achieves ~85% memory reduction for 128k vocab size where batch size $\times$ sequence length is 8192.


### Experimental Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul`

- **Embedding**: [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) is implemented by fusing embedding lookup and output operations. It achieves a peak speedup of ~1.5x in the forward pass and an overall speedup of ~1.1x.
- **Matmul int2xint8**: is implemented by using the cache tiled matrix multiplication and by fusing the matmul with the unpacking process which achieves a considerable speed up and performs on par with @torch.compile
<!-- TODO: be more specific about batch size -->
> **Note:**
> Reported speedups and memory reductions are with respect to the LLaMA 3-8B Hugging Face layer implementations. All models use 4K hidden size and 4K sequence length and are evaluated based on memory usage and wall time for the forward+backward pass on a single NVIDIA A100 80G GPU using small batch sizes. Liger kernels exhibit more efficient scaling to larger batch sizes, detailed further in the [Benchmark](./benchmark) folder.

## Contributing

[CONTRIBUTING GUIDE](https://github.com/linkedin/Liger-Kernel/blob/main/CONTRIBUTING.md)

## Acknowledgement


### Design

- [@claire_yishan](https://twitter.com/claire_yishan) for the LOGO design
- [Wave Snippets](https://www.wavesnippets.com/) for generating the animated code snippets

### Code

We referenced or used the following projects:



| # | Project                                                                                      | Description                                                                             | Location                                                                                                                         | License                                                                              |
|---|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1 | [Unsloth](https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43)                              | `calculate_settings` to determine block size and warp; We reuse it for Norm and MLP     | [Liger Kernel Utils](https://github.com/linkedin/Liger-Kernel/blob/e249eee723978bf8610ff1ea2297d048a2417e20/src/liger_kernel/ops/utils.py#L23) | [Apache](https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/LICENSE) |
| 2 | [Unsloth](https://github.com/unslothai/unsloth/blob/976d11a10d54383aeb7a692c69e01151a20bfd72/unsloth/kernels/rms_layernorm.py#L48)                              | We modified and added dW calculation on top of Unsloth implementation                   | [Liger Kernel RMS Norm](https://github.com/linkedin/Liger-Kernel/blob/e249eee723978bf8610ff1ea2297d048a2417e20/src/liger_kernel/ops/rms_norm.py#L50)  | [Apache](https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/LICENSE) |
| 3 | [Triton tutorial](https://triton-lang.org/main/index.html)                                    | We modified on top of triton tutorials                                                  | [Liger Kernel RMS Norm](https://github.com/linkedin/Liger-Kernel/blob/e249eee723978bf8610ff1ea2297d048a2417e20/src/liger_kernel/ops/rms_norm.py#L50)  | [MIT](https://github.com/triton-lang/triton/blob/main/LICENSE)                                  |
| 4 | [tiny shakespeare dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare)         | We use tiny shakespeare dataset to conduct convergence test on mini model               | [Liger Kernel Convergence](https://github.com/linkedin/Liger-Kernel/tree/main/test/convergence)                                  | N/A                                                                                   |
| 5 | [Efficient Cross Entropy](https://github.com/mgmalek/efficient_cross_entropy)                 | We use the idea of gradient-in-forward and chunking                                    | [Liger Kernel Linear Cross Entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)          | [MIT](https://github.com/mgmalek/efficient_cross_entropy/blob/main/LICENSE)            |
| 6 | [Flash attn](https://github.com/Dao-AILab/flash-attention)                                    | We take many optimization ideas from the work, such as tiling and recomputation         |                                                                                                                                  | [BSD](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE)                  |
| 7 | [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)                                           | We reference the design of automodel                                                   | [Liger Kernel Auto Model](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/auto_model.py)        | [MIT](https://github.com/casper-hansen/AutoAWQ/blob/main/LICENSE)                      |
| 8 | [llm.c](https://github.com/karpathy/llm.c)                                                    | We reference the design of end-to-end testing                                          | [Liger Kernel Convergence Tests](https://github.com/linkedin/Liger-Kernel/tree/main/test/convergence)                            | [MIT](https://github.com/karpathy/llm.c/blob/master/LICENSE)                           |

Many thanks to the contributors to these projects for their invaluable work that helped make Liger possible.

## License

This project is licensed under the [BSD 2-CLAUSE](https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE) License (see `LICENSE` for details).
It also includes components from projects licensed under:

- Apache License 2.0 (see `LICENSE-APACHE-2.0` for details).
- MIT License (see `LICENSE-MIT-AutoAWQ` for details).
- MIT License (see `LICENSE-MIT-Efficient Cross Entropy` for details).
- MIT License (see `LICENSE-MIT-llmc` for details).
- MIT License (see `LICENSE-MIT-triton` for details).

## Contact

- For public discussion, join [our discord channel](https://discord.gg/vNBDpjhb)
- For formal collaboration, send an email to byhsu@linkedin.com

## Cite this work

Biblatex entry:
```bib
@article{hsu2024ligerkernelefficienttriton,
      title={Liger Kernel: Efficient Triton Kernels for LLM Training}, 
      author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen},
      year={2024},
      eprint={2410.10989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10989},
      journal={arXiv preprint arXiv:2410.10989},
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://star-history.com/#linkedin/Liger-Kernel&Date)

## Contributors

<a href="https://github.com/linkedin/Liger-Kernel/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=linkedin/Liger-Kernel"/>
</a>

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
