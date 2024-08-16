# Liger Kernel

[![Downloads](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel) [![PyPI version](https://badge.fury.io/py/liger-kernel.svg)](https://badge.fury.io/py/liger-kernel) [![PyPI version](https://badge.fury.io/py/liger-kernel-nightly.svg)](https://badge.fury.io/py/liger-kernel-nightly)


[Installation](#installation) | [Getting Started](#getting-started) | [Structure](#structure) | [APIs](#apis) | [Contributing](#contributing)

**Liger (Linkedin GPU Efficient Runtime) Kernel** is **Efficient Triton Kernels for LLM Training**. We have implemented **Hugging Face Compatible** `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, and more to come. It can effectively increase multi-GPU **training throughput by 20%** and reduces **memory usage by 60%**. The kernel works out of the box with [flash attention](https://github.com/Dao-AILab/flash-attention), PyTorch FSDP, and Microsoft DeepSpeed. We welcome contributions from the community to gather the best kernels for LLM training.



## Supercharge Your Model with Liger Kernel

![Banner](/docs/images/banner.GIF)

Gain +20% throughput and reduce memory usage by 60%. Achieve longer context lengths and larger batch sizes. It’s also useful if you want to scale up your model to multi-head training or large vocabulary sizes.

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](docs/images/e2e-tps.png) | ![Memory](docs/images/e2e-memory.png) |


> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s. 
> - Hugging Face models start to OOM at a 4K context length, whereas Liger Kernel scales up to 16K.  
> - **Fused Linear Cross Entropy Loss** is enabled to significantly reduce memory usage.

## Examples

### Basic

| **Example**                                    | **Description**                                                                                   | **Lightning Studio** |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | Train llama3 8B ~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s | TBA                  |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s  | TBA                  |

### Advanced

| **Example**                                    | **Description**                                                                                   | **Lightning Studio** |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------------|
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s            | TBA                  |

## Key Features

- **Ease of use:** Simply patch your Hugging Face model with one line of code, or compose your own model using our kernels.
- **Time- and memory-efficient:** In the same spirit as Flash-Attn, but for layers like **RMSNorm**, **RoPE**, **CrossEntropy**! Increases multi-GPU training throughput by 20% and reduces memory usage by 60% with **kernel fusion**, **in-place replacement**, and **chunking** techniques.
- **Exact:** Exact kernels—no approximations. Both forward and backward are implemented with rigorous unit and convergence testing to ensure accuracy.
- **Lightweight:** The kernels have minimal dependencies, requiring only Torch and Triton—no extra libraries needed! Say goodbye to dependency headaches!
- **Multi-GPU supported:** Compatible with multi-GPU setups (PyTorch FSDP and DeepSpeed).

## Target Audiences

- **Researchers**: Looking to compose models using efficient and reliable kernels for frontier experiments.
- **ML Practitioners**: Focused on maximizing GPU training efficiency with optimal, high-performance kernels.
- **Curious Novices**: Eager to learn how to write reliable Triton kernels to enhance training efficiency.


## Installation

### Dependencies

- `torch >= 2.1.2`
- `triton >= 2.3.0`
- `transformers >= 4.40.1`

To install the stable version:

```bash
$ pip install liger-kernel 
```

To install the nightly version:

```bash
$ pip install liger-kernel-nightly
```

## Getting Started

### 1. Patch Existing Hugging Face Models

Using [patching APIs](#patching), you can swap Hugging Face models with optimized Liger Kernels.

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

model = transformers.AutoModelForCausalLM.from_pretrained("<some llama model>")

# By adding this line, it automatically monkey patches the model with the optimized kernels
apply_liger_kernel_to_llama() 
```

### 2. Compose Your Own Model

You can take individual [kernels](#kernels) to compose your models.

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).to("cuda")

# LigerFusedLinearCrossEntropyLoss fuses linear and cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.empty(4, dtype=torch.long, device="cuda").random_(256)

loss = loss_fn(model.weight, input, target)
loss.backward()
```


## Structure

### Source Code

- `ops/`: Core Triton operations.
- `transformers/`: PyTorch `nn.Module` implementations built on Triton operations, compliant with the `transformers` API.

### Tests

- `transformers/`: Correctness tests for the Triton-based layers.
- `convergence/`: Patches Hugging Face models with all kernels, runs multiple iterations, and compares weights, logits, and loss layer by layer.

### Benchmark

- `benchmark/`: Execution time and memory benchmarks compared to Hugging Face layers.

## APIs

### Patching

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| LLaMA (2 & 3) | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss        |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss         |

### Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| FusedLinearCrossEntropy         | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|

- **RMSNorm**: RMSNorm, which normalizes tensor activations using their root mean square, is accelerated by fusing the normalization and scaling steps into a single triton kernel, achieved ~3X speedup with ~3X peak memory reduction. [RMSNorm Paper](https://arxiv.org/pdf/1910.07467)
- **RoPE**: Fused the operations of query and key embedding rotary into a single kernel with inplace replacement, achieved ~3X speedup with ~3X peak memory reduction. [RoPE Paper](https://arxiv.org/pdf/2104.09864)
- **SwiGLU**: Leveraging the fused triton kernel for the elementwise transformation in $$SwiGLU_{\beta=1}$$ ($$\sigma(A) \odot B$$) with inplace replacement, achieved parity speed with ~1.5X peak memory reduction. [SwiGLU Paper](https://arxiv.org/pdf/2002.05202)
- **GeGLU**: Leveraging the fused triton kernel for the elementwise transformation in GeGLU with [tanh approximation form of GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) and inplace replacement, achieved parity speed with ~1.5X peak memory reduction. [GeGLU paper](https://arxiv.org/pdf/2002.05202)
- **CrossEntropy**: Computes both loss and the gradient in the forward path with inplace replacement of input to reduce the peak memory (avoid the materialization of both input logits and gradient), achieved >2X speedup and >4X memory reduction for common vocab sizes. [PyTorch CrossEntropyLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- **FusedLinearCrossEntropy**: Further improves upon the basic Liger Cross Entropy kernel on reducing the peak memory usage by fusing the model last output head layer with the CE loss and chunking the input for block-wise loss and gradient calculation, inspired by [Efficient Cross Entropy](https://github.com/mgmalek/efficient_cross_entropy), achieved >4X memory reduction for 128k vocab size. **This is highly effective for large batch size, large sequence length, and large vocab size model** Please refer to the [Medusa example](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa) for individual kernel usage.


> * Reported speedups and memory reductions are compared with Llama3 8B Hugging Face layer implementations with 4k default hidden size and 4k sequence length for single forward and backward pass on single NVIDIA A100 80G GPU with small batch sizes. Liger kernels exhibits more efficient scaling to larger batch sizes of tokens. See [Benchmark](./benchmark) folder for details.

## Note on ML Compiler

### 1. Torch Compile

Since Liger Kernel is 100% Triton-based, it works seamlessly with Torch Compile. In the following example, Liger Kernel can further optimize the model on top of Torch Compile, reducing the memory by more than half.

| Configuration                  | Throughput (tokens/sec) | Memory Reserved (MB) |
|--------------------------------|----------------------------|-------------------------|
| Torch Compile                  | 3780                       | 66358                   |
| Torch Compile + Liger Kernel   | 3702                       | 31000                   |

> **Note:**  
> 1. **Fused Linear Cross Entropy Loss** is enabled.  
> 2. Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Seq Len = 4096, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> 3. Tested on torch `2.5.0.dev20240731+cu118`

### 2. Lightning Thunder

*WIP*

## Contributing

[CONTRIBUTING GUIDE](https://github.com/linkedin/Liger-Kernel/blob/main/CONTRIBUTING.md)

## Acknowledgement

- [flash-attn](https://github.com/Dao-AILab/flash-attention) and [Unsloth](https://github.com/unslothai/unsloth) for inspiration in Triton kernels for training
- [tiny shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) for convergence testing by andrej karpathy


## License

[BSD 2-CLAUSE](https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE)

## Cite this work

Biblatex entry:
```bib
@software{liger2024,
  title  = {Liger-Kernel: Efficient Triton Kernels for LLM Training},
  author = {Hsu, Pin-Lun and Dai, Yun and Kothapalli, Vignesh and Song, Qingquan and Tang, Shao and Zhu, Siyu},
  url    = {https://github.com/linkedin/Liger-Kernel},
  year   = {2024}
}
```
