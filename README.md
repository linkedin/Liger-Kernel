# 🦁 Liger Kernel

[![Downloads](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel) [![PyPI version](https://badge.fury.io/py/liger-kernel.svg)](https://badge.fury.io/py/liger-kernel) [![PyPI version](https://badge.fury.io/py/liger-kernel-nightly.svg)](https://badge.fury.io/py/liger-kernel-nightly)

**Liger (Linkedin GPU Efficient Runtime) Kernel** is a collection of Triton-native kernels designed specifically for LLM training. It aims to be **ease of use**, **exact**, **time- and memory-efficient**, and **lightweight**. We welcome contributions from the community to gather the best kernels for LLM training.



## 🌟 Overview

### 🔌 Supercharge Your Model with Liger Kernel

Gain +20% throughput and reduce memory usage by 60%. Achieve longer context lengths and larger batch sizes. It’s also useful if you want to scale up your model to multi-head training or large vocabulary sizes.

| ⚡ Speed Up                 | 💾 Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](docs/images/e2e-tps.png) | ![Memory](docs/images/e2e-memory.png) |

> **Note:**  
> 1. Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s. 
> 2. HuggingFace models start to OOM at a 4K context length, whereas Liger Kernel scales up to 16K.  
> 3. **Fused Linear Cross Entropy Loss** is enabled to significantly reduce memory usage.

### ✨ Patch HF model with one line or use individual kernels

| 🛠️ Patch Existing HF Model               | 🧩 Compose Your Own Model       |
|--------------------------|-------------------------|
| ![Patch](docs/images/patch.gif) | ![Compose](docs/images/compose.gif) |

### ✨ Key Features

- **🔧 Ease of use:** Simply patch your Hugging Face model with one line of code, or compose your own model using our kernels.
- **🚀 Time- and memory-efficient:** In the same spirit as Flash-Attn, but for layers like RMSNorm, RoPE, CrossEntropy, and more—not just attention! Increases multi-GPU training throughput by 20% and reduces memory usage by 60%.
- **✅ Exact:** Exact (fwd + bwd) kernels—no approximations. Each kernel undergoes rigorous unit and convergence testing to ensure accuracy.
- **🌱 Lightweight:** The kernels have minimal dependencies, requiring only Torch and Triton—no extra libraries needed! Say goodbye to dependency headaches!
- **💻 Multi-GPU supported:** Compatible with multi-GPU setups (PyTorch FSDP and DeepSpeed).

### 🎯 Target Audiences

- **Researchers**: Looking to compose models using efficient and reliable kernels for frontier experiments.
- **ML Practitioners**: Focused on maximizing GPU training efficiency with optimal, high-performance kernels.
- **Curious Novices**: Eager to learn how to write reliable Triton kernels to enhance training efficiency.


## 🔧 Installation

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

## 🚀 Getting Started

### 1. 🛠️ Patch Existing Hugging Face Models

```python
from liger_kernel.transformers import apply_liger_kernel_to_llama
from transformers import Trainer


model = transformers.AutoModelForCausalLM.from_pretrained("<some llama model>")

# By adding this line, it automatically monkey patches the model with the optimized kernels
apply_liger_kernel_to_llama() 
```



| **Example**       | **Description**                                                  |
|------------------------|------------------------------------------------------------------|
| **Hugging Face Trainer** | [TODO] Add example for using Hugging Face Trainer with Liger Kernel |
| **Lightning Trainer**   | [TODO] Add example for using Lightning Trainer with Liger Kernel  |


### 2. 🧩 Compose Your Own Model

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

| **Example**       | **Description**                                                  |
|------------------------|------------------------------------------------------------------|
| **Multi-head Trainer** | [TODO] Add example for medusa training |


## ⚙️ Note on ML Compiler

### 1. ⚡ Torch Compile

Since Liger Kernel is 100% Triton-based, it works seamlessly with Torch Compile. In the following example, Liger Kernel can be further optimized on top of Torch Compile, reducing the memory by more than half.

| Configuration                  | ⚡ Throughput (tokens/sec) | 💾 Memory Reserved (MB) |
|--------------------------------|----------------------------|-------------------------|
| Torch Compile                  | 3780                       | 66358                   |
| Torch Compile + Liger Kernel   | 3702                       | 31000                   |

> **Note:**  
> 1. **Fused Linear Cross Entropy Loss** is enabled.  
> 2. Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Seq Len = 4096, Data Type = bf16, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> 3. Tested on torch `2.5.0.dev20240731+cu118`

### 2. 🌩️ Lightning Thunder

*WIP*

## 📂 Structure

### Source Code

- `ops/`: Core Triton operations.
- `transformers/`: PyTorch `nn.Module` implementations built on Triton operations, compliant with the `transformers` API.

### Tests

- `transformers/`: Correctness tests for the Triton-based layers.
- `convergence/`: Patches Hugging Face models with all kernels, runs multiple iterations, and compares weights, logits, and loss layer by layer.

### Benchmark

- `benchmark/`: Execution time and memory benchmarks compared to Hugging Face layers.

## 🔧 APIs

### Patching

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| LLaMA (2 & 3) | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss        |


### 🧩 Kernels

| **Kernel**                | **API**                                                     | **Description** |
|---------------------------|-------------------------------------------------------------|-----------------|
| RMSNorm                    | `liger_kernel.transformers.LigerRMSNorm`                    | [RMSNorm Paper](https://arxiv.org/pdf/1910.07467) |
| RoPE                       | `liger_kernel.transformers.liger_rotary_pos_emb`            | [RoPE Paper](https://arxiv.org/pdf/2104.09864)    |
| SwiGLU                     | `liger_kernel.transformers.LigerSwiGLUMLP`                  | [SwiGLU Paper](https://arxiv.org/pdf/2002.05202)  |
| CrossEntropy               | `liger_kernel.transformers.LigerCrossEntropyLoss`           | [PyTorch CrossEntropyLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| FusedLinearCrossEntropy    | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`| Inspired by [Efficient Cross Entropy](https://github.com/mgmalek/efficient_cross_entropy), with additional optimizations |


## 🛣️ Roadmap

WIP

## 🤝 Contributing

[CONTRIBUTING GUIDE](https://github.com/linkedin/Liger-Kernel/blob/main/CONTRIBUTING.md)

## 📜 License

[BSD 2-CLAUSE](https://github.com/linkedin/Liger-Kernel/blob/main/LICENSE)

## Cite this work

Biblatex entry:
```bib
@software{liger2024,
  title  = {Liger-Kernel: Efficient Triton Kernels for LLM Training},
  author = {Hsu, Pin-Lun and Dai, Yun and Kothapalli, Vignesh and Song, Qingquan and Tang, Shao},
  url    = {https://github.com/linkedin/Liger-Kernel},
  year   = {2024}
}
```
