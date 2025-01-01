<a name="readme-top"></a>

# Liger Kernel: Efficient Triton Kernels for LLM Training


<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;" colspan="2">Stable</th>
        <th style="padding: 10px;" colspan="2">Nightly</th>
        <th style="padding: 10px;">Discord</th>
        <th style="padding: 10px;">Build</th>
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
            <div style="display: block;">
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
            <div style="display: block;">
                <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml">
                    <img src="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml/badge.svg?event=schedule" alt="Build">
                </a>
            </div>
        </td>
    </tr>
</table>



<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png">


**Liger Kernel** is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU **training throughput by 20%** and reduces **memory usage by 60%**. We have implemented **Hugging Face Compatible** `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, and more to come. The kernel works out of the box with [Flash Attention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed). We welcome contributions from the community to gather the best kernels for LLM training.

We've also added optimized Post-Training kernels that deliver **up to 80% memory savings** for alignment and distillation tasks. We support losses like DPO, CPO, ORPO, SimPO, JSD, and many more. Check out [how we optimize the memory](https://x.com/hsu_byron/status/1866577403918917655).

## Supercharge Your Model with Liger Kernel

With one line of code, Liger Kernel can increase throughput by more than 20% and reduce memory usage by 60%, thereby enabling longer context lengths, larger batch sizes, and massive vocabularies.


| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face + Liger Kernel scales up to 16K.

## Optimize Post Training with Liger Kernel

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

We provide optimized post training kernels like DPO, ORPO, SimPO, and more which can reduce memory usage by up to 80%. You can easily use them as python modules.

```python
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

#### Key Features

- **Ease of use:** Simply patch your Hugging Face model with one line of code, or compose your own model using our Liger Kernel modules.
- **Time and memory efficient:** In the same spirit as Flash-Attn, but for layers like **RMSNorm**, **RoPE**, **SwiGLU**, and **CrossEntropy**! Increases multi-GPU training throughput by 20% and reduces memory usage by 60% with **kernel fusion**, **in-place replacement**, and **chunking** techniques.
- **Exact:** Computation is exact—no approximations! Both forward and backward passes are implemented with rigorous unit tests and undergo convergence testing against training runs without Liger Kernel to ensure accuracy.
- **Lightweight:** Liger Kernel has minimal dependencies, requiring only Torch and Triton—no extra libraries needed! Say goodbye to dependency headaches!
- **Multi-GPU supported:** Compatible with multi-GPU setups (PyTorch FSDP, DeepSpeed, DDP, etc.).
- **Trainer Framework Integration**: [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory), [SFTTrainer](https://github.com/huggingface/trl/releases/tag/v0.10.1), [Hugging Face Trainer](https://github.com/huggingface/transformers/pull/32860), [SWIFT](https://github.com/modelscope/ms-swift)

### Installation

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

# Install Default Dependencies
# Setup.py will detect whether you are using AMD or NVIDIA
pip install -e .

# Setup Development Dependencies
pip install -e ".[dev]"
```

!!! Note " Dependencies " 

    #### CUDA

    - `torch >= 2.1.2`
    - `triton >= 2.3.0`

    #### ROCm

    - `torch >= 2.5.0` Install according to the instruction in Pytorch official webpage.
    - `triton >= 3.0.0` Install from pypi. (e.g. `pip install triton==3.0.0`)

!!!Tip "Optional Dependencies "

    - `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

!!! Note
     Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).


#### Sponsorship and Collaboration
 
- [AMD](https://www.amd.com/en.html): Providing AMD GPUs for our AMD CI.
- [Intel](https://www.intel.com/): Providing Intel GPUs for our Intel CI.
- [Modal](https://modal.com/): Free 3000 credits from GPU MODE IRL for our NVIDIA CI.
- [EmbeddedLLM](https://embeddedllm.com/): Making Liger Kernel run fast and stable on AMD. 
- [HuggingFace](https://huggingface.co/): Integrating Liger Kernel into Hugging Face Transformers and TRL.
- [Lightning AI](https://lightning.ai/): Integrating Liger Kernel into Lightning Thunder.
- [Axolotl](https://axolotl.ai/): Integrating Liger Kernel into Axolotl.
- [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Integrating Liger Kernel into Llama-Factory.


!!! Note " Contact " 

    - For issues, create a Github ticket in this repository .
    - For open discussion, join [our discord channel](https://discord.gg/gpumode) .
    - For formal collaboration, send an email to byhsu@linkedin.com .

### Cite this work

Bib Latex entry:
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

### Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://star-history.com/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>