<a name="readme-top"></a>

# Liger Kernel: 用于大语言模型训练的高效 Triton 算子库

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <th style="padding: 10px;" colspan="2">Stable</th>
        <th style="padding: 10px;" colspan="2">Nightly</th>
        <th style="padding: 10px;">Discord</th>
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
            <a href="https://discord.gg/X4MaxPgA">
                <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/X4MaxPgA?style=flat" alt="Join Our Discord">
            </a>
        </td>
    </tr>
</table>

<img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png">

[安装指南](#安装指南) | [快速上手](#快速上手) | [示例教程](#示例教程) | [高阶-api](#高阶-api) | [底层算子-api](#底层算子-api) | [引用本项目](#引用本项目)

<details>
  <summary>最新动态 🔥</summary>

  - [2025/12/19] 我们在 https://discord.gg/X4MaxPgA 开设了 liger kernel discord 交流频道；我们还将在 2026 年 1 月中旬举办 Liger Kernel × Triton 中国线下交流会（Meetup）。
  - [2025/03/06] 我们联合发布了关于 TorchTune × Liger 的博文——[峰值性能，极致显存优化：通过 torch.compile 与 Liger Kernel 优化 torchtune 的性能](https://pytorch.org/blog/peak-performance-minimized-memory/)。
  - [2024/12/11] 我们发布了 [v0.5.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.5.0)：对后训练损失函数（DPO、ORPO、CPO 等）实现高达 80% 的显存节约！
  - [2024/12/5] 我们发布了 LinkedIn 工程博客——[Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)。
  - [2024/11/6] 我们发布了 [v0.4.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0)：全面支持 AMD，发布技术报告，接入 Modal CI，支持 Llama-3.2-Vision！
  - [2024/10/21] 我们在 ArXiv 上发布了 Liger Kernel 技术报告：https://arxiv.org/pdf/2410.10989
  - [2024/9/6] 我们发布了 v0.2.1（详见 [X 帖子](https://x.com/liger_kernel/status/1832168197002510649)）。在两周内获得了 2500+ Star、10+ 位新贡献者、50+ 个 PR 以及 5 万次下载！
  - [2024/8/31] CUDA MODE 技术分享：[Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns)，[讲义 Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)。
  - [2024/8/23] 正式官宣发布：查看我们的 [X 帖子](https://x.com/hsu_byron/status/1827072737673982056)。

</details>

**Liger Kernel** 是专门为大语言模型（LLM）训练设计的高效 Triton 算子集合。它能有效将多 GPU **训练吞吐量提升 20%**，并将**显存占用降低 60%**。我们已实现了与 **Hugging Face 完全兼容** 的 `RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy` 等算子，后续还将持续扩充。本算子库开箱即用，支持与 [Flash Attention](https://github.com/Dao-AILab/flash-attention)、[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 以及 [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed) 无缝协同。我们热忱欢迎社区参与贡献，共同汇聚最优秀的 LLM 训练算子。

我们还新增了优化后的后训练（Post-Training）算子，可在对齐与蒸馏任务中**节省高达 80% 的显存**。我们支持包括 DPO、CPO、ORPO、SimPO、KTO、JSD、GRPO（含 `bnpo`、`dr_grpo`、`dapo`、`cispo`、`sapo`、`luspo` 和 `vespo` 变体）等多种损失函数。欢迎查看我们[如何优化显存占用](https://x.com/hsu_byron/status/1866577403918917655)。

您可以访问官方文档网站获取更多安装指南、使用示例以及 API 参考文档：https://linkedin.github.io/Liger-Kernel/

您可以阅读 Liger Kernel 技术报告：https://openreview.net/forum?id=36SjAIT42G

## 使用 Liger Kernel 为您的模型加速

![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

仅需一行代码，Liger Kernel 即可将吞吐量提升 20% 以上，并减少 60% 的显存占用，从而支持更长的上下文长度、更大的批次大小（Batch Size）以及超大规模词表。

| 吞吐量加速 (Speed Up)    | 显存占用降低 (Memory Reduction) |
|--------------------------|---------------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **说明：**
> - 基准测试条件：LLaMA 3-8B，Batch Size = 8，数据类型 = `bf16`，优化器 = AdamW，激活检查点（Gradient Checkpointing）= True，分布式策略 = 8 张 A100 上的 FSDP1。
> - 原生 Hugging Face 模型在上下文长度达到 4K 时即出现 OOM（显存溢出），而 Hugging Face + Liger Kernel 可轻松扩展至 16K。

## 使用 Liger Kernel 优化后训练

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

我们提供了优化后的后训练算子（如 DPO、ORPO、SimPO 等），可将显存占用降低高达 80%。您可以非常方便地将其作为 Python 模块直接调用。

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## 示例教程

| **应用场景**                                    | **详细描述**                                                                                      |
|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)      | 在 Alpaca 数据集上使用 4 张 A100 配备 FSDP 训练 LLaMA 3-8B，速度提升约 20%，显存减少 40% 以上     |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning)         | 在 MMLU 数据集上使用 8 张 A100 配备 DeepSpeed ZeRO3 训练 LLaMA3-8B，吞吐量提升 15%，显存减少 40% |
| [**Medusa 多头 LLM（重新训练阶段）**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa)        | 针对 5 个 LM head 减少 80% 显存占用，并在 8 张 A100 配备 FSDP 下提升 40% 吞吐量                 |
| [**视觉-语言模型 SFT 微调**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh)      | 在图文数据上使用 4 张 A100 配备 FSDP 微调 Qwen2-VL                                                |
| [**Liger ORPO 训练器**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py)      | 使用配备 FSDP 的 Liger ORPO 训练器对齐 Llama 3.2，显存减少 50%                                    |

## 核心特性

- **极简易用：** 仅需一行代码即可为 Hugging Face 模型打上补丁（monkey-patch），或使用我们的 Liger Kernel 模块灵活构建专属模型。
- **极致时空效率：** 秉承与 Flash-Attn 相同的设计哲学，但全面赋能 **RMSNorm**、**RoPE**、**SwiGLU** 以及 **CrossEntropy** 等层！通过**算子融合（Kernel Fusion）**、**就地替换（In-place Replacement）**与**分块计算（Chunking）**技术，将多 GPU 训练吞吐量提升 20%，显存占用减少 60%。
- **数值精确无损：** 所有计算均为精确值——绝无任何近似！前向与反向传播均通过了严格的单元测试，并与未采用 Liger Kernel 的训练任务进行全面收敛性对比测试，确保精度一致。
- **轻量依赖：** Liger Kernel 仅依赖 Torch 与 Triton，无任何其他多余外部依赖库，彻底告别繁琐的依赖冲突困扰！
- **完备的多 GPU 支持：** 原生兼容各类多卡分布式架构（PyTorch FSDP、DeepSpeed、DDP 等）。
- **主流训练框架原生集成：** [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)、[LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory)、[SFTTrainer](https://github.com/huggingface/trl/releases/tag/v0.10.1)、[Hugging Face Trainer](https://github.com/huggingface/transformers/pull/32860)、[SWIFT](https://github.com/modelscope/ms-swift)、[oumi](https://github.com/oumi-ai/oumi/tree/main)。

## 安装指南

### 核心依赖

#### CUDA

- `torch >= 2.1.2`
- `triton >= 2.3.1`

#### ROCm

- `torch >= 2.5.0` 根据 PyTorch 官方页面说明进行安装。
- `triton >= 3.0.0` 从 PyPI 安装。（例如：`pip install triton==3.0.0`）

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

#### 昇腾 Ascend NPU

- `torch == 2.9.0`
- `torch_npu == 2.9.0`
- `triton-ascend == 3.2.2` 从昇腾专属 PyPI 镜像源安装（默认 PyPI 源未收录）。
- `CANN == 9.1.0`

```bash
pip install -e ".[dev]" --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

### 可选依赖

- `transformers >= 5.0.0`：如果您计划使用 transformers 模型补丁 API，则必须安装。您所使用的具体模型将决定 transformers 的最低版本要求。
- `cuda-tile`：在 CUDA 上启用可选的 cuTile 后端时需要。当您的环境中已预装 CUDA Toolkit 13.1 或更高版本，或者已安装现有的 tileiras 编译器时使用。
- `cuda-tile[tileiras]`：在 Python 环境中直接安装 tileiras 编译器并启用可选 cuTile 后端时需要。
- `nvidia-cutlass-dsl >= 4.6.0`：在 CUDA 上启用可选的 CuTe DSL 后端时需要（随 NVIDIA CUTLASS 附带的仅限 CUDA 的 Python DSL，`import cutlass.cute`）。面向 Hopper (SM90) 与 Blackwell (SM100/SM110) 架构。
- `liger-cute-kernels`：面向 Hopper 与 Blackwell 的原生 CUTLASS + NVSHMEM 算子，通过 `lck` extra 安装完全匹配的 Liger 版本。

> **说明：**
> 我们的算子完全继承了 [Triton](https://github.com/triton-lang/triton) 所支持的完整硬件兼容性矩阵。

安装稳定版本：

```bash
$ pip install liger-kernel
```

同时安装匹配的原生 Liger 通信算子：

```bash
$ pip install "liger-kernel[lck]" "liger-cute-kernels[cu12]"
```

原生 LCK 正式版 wheel 构建依然基于 **CUDA 12.9**（CUDA Toolkit 12.9.1）。
可选的 `cu12` extra 会安装 `nvidia-nvshmem-cu12==3.6.5`；`cu13` 则针对 CUDA 13 源码构建安装 `nvidia-nvshmem-cu13==3.6.5`。纯净的 `liger-cute-kernels` 不会安装 NVSHMEM。请仅选择一个与原生 wheel 构建工具链匹配的 extra：extras 仅用于选择依赖，而非更换编译好的 wheel。CUDA 13 源码构建当前在 Hopper 架构上可正常运行，但联合发布版构建目前因 Blackwell MoE 反向算子中的混合 CTA-group 指令受阻。此前已发布的构件保持不变。默认的 `liger-kernel` 安装保持不变。有关源码构建与兼容性详情，请参阅[原生 wheel 构建指南](liger_cute_kernels/README.md#building-the-native-wheel)。

安装每夜构建版（Nightly）：

```bash
$ pip install liger-kernel-nightly
```

从源码安装：

```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel

# 安装默认依赖
# setup.py 会自动检测本地后端并选择默认依赖。
# 在 ROCm 平台上，请先从 PyTorch ROCm 镜像源安装 ROCm PyTorch。
pip install -e .

# 安装开发环境依赖
pip install -e ".[dev]"

# ROCm 源码安装步骤
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
# 然后选择其一安装：
pip install -e .
pip install -e ".[dev]"

# 安装 cuTile 依赖
pip install -e ".[cutile]"

# 或者安装带有可选 tileiras 编译器的 cuTile
pip install -e ".[cutile-tileiras]"

# 安装 CuTe DSL（NVIDIA CUTLASS Python DSL）依赖
pip install -e ".[cutedsl]"

# 安装原生 Liger 通信算子
pip install -e ".[lck]"
```

### 启用 cuTile 后端

cuTile 是一套可选的仅限 CUDA 的 DSL 实现。在安装了 `cutile` 或 `cutile-tileiras` extra 后，显式启用它：

```bash
LIGER_KERNEL_IMPL=cutile python your_script.py
```

`LIGER_KERNEL_IMPL` 用于选择注册到 Liger 中的可选实现（当前支持 `cutile` 与 `cutedsl`）。在不受支持的设备上或未安装所需依赖的情况下选择此类实现将会抛出错误。

### 启用 CuTe DSL 后端

CuTe DSL 是随 NVIDIA CUTLASS 附带的可选、仅限 CUDA 的 Python DSL（`import cutlass.cute`），面向 Hopper (SM90) 与 Blackwell (SM100/SM110) 架构。在安装了 `cutedsl` extra 后，显式启用它：

```bash
pip install "liger-kernel[cutedsl]"
LIGER_KERNEL_IMPL=cutedsl python your_script.py
```

目前提供了以下算子的原生 `cutlass.cute` 实现：

- **RMSNorm**
- **RoPE**
- **SwiGLU**
- **Cross entropy**
- **Fused linear cross entropy**（包含 SM90 专用变体）
- **Fused scaled cross entropy** (SM90)

未提供 CuTe DSL 算子的操作将无缝回退至默认的 Triton 算子。

`cutedsl` extra 还会引入 `apache-tvm-ffi`，这使得编译后的算子可以直接接收 PyTorch 张量，而无需在每次调用时逐一编组（marshalling）。这是可选的——未安装该库时每个算子均会回退至编组启动模式——但对于耗时极短的算子而言，编组开销往往占主导地位，因此强烈建议安装。

### 融合缩放交叉熵算子 (Fused Scaled Cross Entropy)

`LigerFusedLinearScaledCrossEntropyFunction` 是一项额外的逐词元（per-token）算子，并非旨在替代面向归约的 Triton `LigerFusedLinearCrossEntropyFunction`。它接收 `input[M, H]`、`weight[V, H]` 和 `target[M]`，应用 `logits / temperature`，并返回 FP32 负对数似然 `[M]` 以及可选的与输入 dtype 一致的可微词表熵 `[M]`。归约操作保留在 PyTorch 中进行，且目标值等于 `ignore_index` 的行输出与梯度均为零。

```python
from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction

nll, entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
    x, weight, target, 1.0, -100, 1, True
)  # [M], [M]
loss = nll.sum() / (target != -100).sum().clamp_min(1)
```

各实现共享此公开契约，但采用不同的调度方式：

- **cuTile** (`LIGER_KERNEL_IMPL=cutile`) 支持 CUDA 上类型匹配的浮点 `input` 和 `weight` 张量以及有限的正标量 `temperature`。便携的临时 logits 预算为 256 MiB。大型 FP16/BF16 Blackwell 工作负载（`M >= 4096`, `V >= 131072`）会自动使用 512 MiB 以提升 GEMM 利用率。可将 `LIGER_CUTILE_SCALED_CE_WORKSPACE_MB` 设置为其他正 MiB 值以进行特定负载调优；例如 1024 可帮助大型 NLL 叠加熵的工作负载，但并非在所有场景下都更快。反向传播复用同一个工作空间，并将 `dX` 直接写入其最终张量，同时就地累加 `dW`。接受 `m_tiles_per_cluster` 参数以维持 API 兼容性，但不会改变 cuTile 调度方式。
- **CuTe SM90** 在 Hopper 上针对 BF16 输入使用 `LigerFusedScaledCrossEntropySM90Function`。其单独的前向计算采用固定的 cluster-M2 N160 片段算子（fragment kernel），带有经性能评测的长序列形状 split-N 查表，且绝不将 logits 写入 HBM；接受 `m_tiles_per_cluster` 以保持 API 兼容性，但不会改变该调度方式。反向传播在一个包含可复用 1024-token `dZ` 工作空间的持久集群算子（persistent cluster kernel）中运行 `dZ`、`dX` 和 `dW`。
- **回退实现 (Fallback)**：当默认前端无法使用 SM90 算子时，采用适配自 Verl 融合 PPO 公式的 512-token 分块（chunked）PyTorch 实现。

这三种非 TP 实现均可额外返回针对少量共享词表 ID 集合的可微对数概率：

```python
selected_token_ids = torch.tensor([no_token_id, yes_token_id], device=x.device, dtype=torch.int64)
nll, entropy, selected_log_probs = LigerFusedLinearScaledCrossEntropyFunction.apply(
    x, weight, target, 1.0, -100, 1, True, selected_token_ids
)  # [M], [M], [M, 2]
```

`selected_token_ids[K]` 为可选参数，必须位于权重所在的设备上，并保留所请求的顺序与重复项。附加输出为 FP32 格式，并使用与 NLL 相同的完整词表归一化项和温度系数。目标值无需包含在选定集合中。被忽略的行返回零且不贡献梯度；调用方必须在应用其自身损失之前对这些行进行掩码处理。在禁用熵的情况下，返回值为 `(nll, selected_log_probs)`；省略 ID 则保持原有的返回类型。空 ID 张量将产生 `[M, 0]` 输出。

选定 logits 会在现有的投影计算中读取，其梯度将在现有的反向工作空间内与 NLL/熵合并。无需二次投影或单独的稠密 LM-head 梯度计算。仅需追加选定的输出和少量行元数据。`selected_log_probs.exp()` 可获得全词表概率；而 `selected_log_probs.log_softmax(-1)` 则在选定集合上进行归一化。这支持仅选定、仅 NLL、仅熵以及组合反向传播。归约操作、辅助损失加权与因果位置对齐仍由调用方负责。下文的独立 TP 前端不接收选定 ID。

对于张量并行（TP）进程组内切分为等大小连续范围的分类器权重，请使用 TP 前端。目标值仍为全局词表索引：

```python
from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyTPFunction

nll, entropy = LigerFusedLinearScaledCrossEntropyTPFunction.apply(
    x, local_weight, target, tp_group, 1.0, -100, 1, True
)
```

TP 前端从进程组秩（rank）派生 `vocab_start`。Hopper 上的 BF16 输入在其原生核心可用时使用可选的 `liger_cute_kernels` 实现；其他配置使用适配自 Verl 融合 PPO 公式的 Liger 分块张量并行回退实现。两种路径均返回全局正确的逐词元输出，并在 TP 组间对输入梯度进行求和。原生 CUTLASS + NVSHMEM 路径完全通过 NVSHMEM 进行通信；没有 Torch 通信器或 Torch C++ ABI 跨越 TVM-FFI 边界。调用在 `tp_group` 上仍保持集合通信语义，因此在每个成员上的执行顺序必须完全一致。

原生算子共享由 `LigerExpertParallelFusedMoe` 使用的进程级 NVSHMEM 引导（bootstrap），而 TP 和 EP 则使用独立的缓存团队与独立命名的工作空间。应用程序设置必须从通用父组（通常为 `WORLD`）初始化 NVSHMEM，并在调用任一算子前解析两个进程组。首个 FSLCE 调用将确定其工作空间容量，因此在捕获 CUDA Graph 之前，请使用预期最大的 token、hidden、本地词表大小以及 `tiles_per_reduce` 配置对其进行预热（warmup）。

基于 H100 BF16 在 `H=4096`, `V=131072` 下每个提供方 60 次交错采样的前向中位数。有效 TFLOPS 计入通用的投影计算工作量（`2*M*H*V`）：

| M | 熵 (Entropy) | CuTe SM90 | cuTile | Verl Torch 回退 |
|---:|:---:|---:|---:|---:|
| 2048 | 否 | **3.12 ms / 706 TFLOPS** | 3.19 ms / 690 TFLOPS | 11.36 ms / 194 TFLOPS |
| 2048 | 是 | **3.13 ms / 703 TFLOPS** | 3.25 ms / 676 TFLOPS | 11.37 ms / 193 TFLOPS |
| 4096 | 否 | **6.05 ms / 727 TFLOPS** | 6.28 ms / 701 TFLOPS | 22.62 ms / 194 TFLOPS |
| 4096 | 是 | **6.11 ms / 720 TFLOPS** | 6.39 ms / 688 TFLOPS | 22.68 ms / 194 TFLOPS |
| 8192 | 否 | **12.25 ms / 718 TFLOPS** | 12.41 ms / 709 TFLOPS | 45.35 ms / 194 TFLOPS |
| 8192 | 是 | **12.30 ms / 715 TFLOPS** | 12.70 ms / 693 TFLOPS | 45.59 ms / 193 TFLOPS |
| 16384 | 否 | **23.84 ms / 738 TFLOPS** | 25.11 ms / 700 TFLOPS | 90.81 ms / 194 TFLOPS |
| 16384 | 是 | **24.09 ms / 730 TFLOPS** | 26.10 ms / 674 TFLOPS | 90.97 ms / 193 TFLOPS |
| 32768 | 否 | **49.69 ms / 708 TFLOPS** | 54.31 ms / 648 TFLOPS | 180.01 ms / 195 TFLOPS |
| 32768 | 是 | **49.85 ms / 706 TFLOPS** | 55.69 ms / 632 TFLOPS | 179.85 ms / 196 TFLOPS |

反向传播中位数基于每个提供方 30 次交错采样。有效 TFLOPS 计入 `6*M*H*V`：

| M | 熵 (Entropy) | CuTe SM90 | cuTile | Verl Torch 回退 |
|---:|:---:|---:|---:|---:|
| 8192 | 否 | **37.93 ms / 696 TFLOPS** | 42.55 ms / 620 TFLOPS | 85.17 ms / 310 TFLOPS |
| 8192 | 是 | **37.89 ms / 696 TFLOPS** | 41.43 ms / 637 TFLOPS | 121.00 ms / 218 TFLOPS |
| 16384 | 否 | **79.42 ms / 665 TFLOPS** | 83.78 ms / 630 TFLOPS | 166.85 ms / 316 TFLOPS |
| 16384 | 是 | **78.20 ms / 675 TFLOPS** | 84.12 ms / 627 TFLOPS | 239.39 ms / 220 TFLOPS |
| 32768 | 否 | 163.34 ms / 646 TFLOPS | **163.21 ms / 647 TFLOPS** | 331.08 ms / 319 TFLOPS |
| 32768 | 是 | **160.64 ms / 657 TFLOPS** | 161.73 ms / 653 TFLOPS | 477.51 ms / 221 TFLOPS |

完整前向加反向有效 TFLOPS 计入 `8*M*H*V`：

| M | 熵 (Entropy) | CuTe SM90 | cuTile | Verl Torch 回退 |
|---:|:---:|---:|---:|---:|
| 8192 | 否 | **50.39 ms / 698 TFLOPS** | 57.15 ms / 616 TFLOPS | 128.76 ms / 273 TFLOPS |
| 8192 | 是 | **50.22 ms / 701 TFLOPS** | 56.86 ms / 619 TFLOPS | 165.63 ms / 212 TFLOPS |
| 16384 | 否 | **104.77 ms / 672 TFLOPS** | 112.03 ms / 628 TFLOPS | 254.91 ms / 276 TFLOPS |
| 16384 | 是 | **103.65 ms / 679 TFLOPS** | 111.21 ms / 633 TFLOPS | 328.37 ms / 214 TFLOPS |
| 32768 | 否 | **211.30 ms / 666 TFLOPS** | 221.34 ms / 636 TFLOPS | 508.33 ms / 277 TFLOPS |
| 32768 | 是 | **210.60 ms / 668 TFLOPS** | 221.39 ms / 636 TFLOPS | 655.89 ms / 215 TFLOPS |

相同配置下基于 B200 的测量结果（使用自动 cuTile 工作空间选择）：

| 实现方案 | 前向 (Forward) | 反向 (Backward) | 完整端到端 (Full) | 完整峰值显存 |
|---|---:|---:|---:|---:|
| cuTile | 2.97 ms | 8.35 ms | 11.36 ms | 2.64 GiB |
| Torch | 5.24 ms | 7.15 ms | 12.85 ms | 7.22 GiB |

## 快速上手

根据所需的自定义程度，有以下几种方式应用 Liger 算子。

### 1. 使用 AutoLigerKernelForCausalLM

使用 `AutoLigerKernelForCausalLM` 是最简单直接的方式，无需导入特定模型的补丁 API。只要模型类型受支持，模型代码将使用默认设置自动打上补丁。

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# 该 AutoModel 包装类会在模型类型受支持时，
# 自动使用优化后的 Liger 算子为模型打上 monkey-patch。
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. 应用特定模型的补丁 API

通过使用[补丁 API](#模型打补丁-patching)，您可以将 Hugging Face 原生模型中的组件无缝替换为优化后的 Liger 算子。

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1a. 添加此行代码会自动使用优化后的 Liger 算子为模型打上 monkey-patch
apply_liger_kernel_to_llama()

# 1b. 您也可以精确指定应用哪些算子
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# 2. 实例化打上补丁后的模型
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

### 3. 自由组装专属模型

您可以单独调用各个[算子](#模型基础算子)，灵活组装您的专属模型。

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# 将线性层与交叉熵损失层融合，并执行分块计算以大幅降低显存
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```

## 高阶 API

### AutoModel

| **AutoModel 变体** | **API** |
|-----------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

### 模型打补丁 (Patching)

| **模型**   | **API**                                                      | **支持的操作**                                                          |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Llama4 (文本与多模态)      | `liger_kernel.transformers.apply_liger_kernel_to_llama4`   | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Ministral   | `liger_kernel.transformers.apply_liger_kernel_to_ministral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Muse Glimmer | `liger_kernel.transformers.apply_liger_kernel_to_muse_glimmer` | LayerNorm, RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Nemotron    | `liger_kernel.transformers.apply_liger_kernel_to_nemotron` | ReLUSquared, CrossEntropyLoss, FusedLinearCrossEntropy                  |
| Pixtral     | `liger_kernel.transformers.apply_liger_kernel_to_pixtral`  | RoPE, RMSNorm, SwiGLU|
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (文本)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (多模态)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma4 (文本)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma4_text`   | RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma4 (多模态)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma4`   | RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Paligemma, Paligemma2, & Paligemma2 Mix      | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`   | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ      | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL, & QVQ       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2.5-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`    | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`    |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3.5      | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_5`    | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3.5 MoE (文本与多模态) | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_5_moe` | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Granite 3.0 & 3.1   | `liger_kernel.transformers.apply_liger_kernel_to_granite`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| OLMo2   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Olmo3   | `liger_kernel.transformers.apply_liger_kernel_to_olmo3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| DeepSeek-V2   | `liger_kernel.transformers.apply_liger_kernel_to_deepseek_v2`     | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| DeepSeek-V3   | `liger_kernel.transformers.apply_liger_kernel_to_deepseek_v3`     | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| DeepSeek-V4   | `liger_kernel.transformers.apply_liger_kernel_to_deepseek_v4`     | RMSNorm, CrossEntropyLoss, FusedLinearCrossEntropy |
| GPT-OSS   | `liger_kernel.transformers.apply_liger_kernel_to_gpt_oss`     | RoPE, RMSNorm, CrossEntropyLoss, FusedLinearCrossEntropy |
| InternVL3   | `liger_kernel.transformers.apply_liger_kernel_to_internvl`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| HunyuanV1   | `liger_kernel.transformers.apply_liger_kernel_to_hunyuan_v1_dense`    |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| HunyuanV1 MoE | `liger_kernel.transformers.apply_liger_kernel_to_hunyuan_v1_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |

## 底层算子 API

- `Fused Linear` 融合线性算子将线性层与损失函数深度融合，可减少高达 80% 的显存占用——是显存受限场景下的理想选择。
- 其他算子采用算子融合与就地替换技术实现显存与计算性能的极致优化。

### 模型基础算子

| **算子名称**                    | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| Modulated RMSNorm               | `liger_kernel.transformers.LigerModulatedRMSNorm`           |
| LayerNorm                       | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy       | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention           | `liger_kernel.transformers.LigerMultiTokenAttention`        |
| Softmax                         | `liger_kernel.transformers.LigerSoftmax`                    |
| Sparsemax                       | `liger_kernel.transformers.LigerSparsemax`                  |
| mHC (超连接 / Hyper-Connections)| `liger_kernel.transformers.LigerMHC`                        |

### 对齐算子 (Alignment Kernels)

| **算子名称**                    | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Fused Linear CPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss         | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |
| Fused Linear GRPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearGRPOLoss`    |

### 蒸馏算子 (Distillation Kernels)

| **算子名称**                    | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD                  | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| Fused Linear KL Divergence        | `liger_kernel.transformers.LigerFusedLinearKLDivLoss`       |
| TVD                             | `liger_kernel.transformers.LigerTVDLoss`                    |

### 实验性算子 (Experimental Kernels)

| **算子名称**                    | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul` |

## 贡献、致谢与开源许可证

- [贡献指南](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
- [致谢名单](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
- [开源许可证信息](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

## 赞助与合作伙伴

- [Glows.ai](https://platform.glows.ai/)：为我们的开源开发者赞助 NVIDIA GPU 资源。
- [AMD](https://www.amd.com/en.html)：为我们的 AMD CI 提供 AMD GPU 支持。
- [Intel](https://www.intel.com/)：为我们的 Intel CI 提供 Intel GPU 支持。
- [Modal](https://modal.com/)：来自 GPU MODE IRL 的 3000 免费额度，用于我们的 NVIDIA CI。
- [EmbeddedLLM](https://embeddedllm.com/)：助力 Liger Kernel 在 AMD 平台上高速稳定运行。
- [HuggingFace](https://huggingface.co/)：将 Liger Kernel 原生集成至 Hugging Face Transformers 与 TRL。
- [Lightning AI](https://lightning.ai/)：将 Liger Kernel 集成至 Lightning Thunder。
- [Axolotl](https://axolotl.ai/)：将 Liger Kernel 原生集成至 Axolotl。
- [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory)：将 Liger Kernel 原生集成至 Llama-Factory。

## CI 状态

<table>
    <tr>
        <th>平台</th>
        <th>构建状态</th>
    </tr>
    <tr>
        <td>NVIDIA GPU</td>
        <td>
            <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml">
                <img src="https://img.shields.io/github/actions/workflow/status/linkedin/Liger-Kernel/nvi-ci.yml?branch=main&event=push&label=" alt="NVIDIA GPU Build">
            </a>
        </td>
    </tr>
    <tr>
        <td>AMD GPU</td>
        <td>
            <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml">
                <img src="https://img.shields.io/github/actions/workflow/status/linkedin/Liger-Kernel/amd-ci.yml?branch=main&event=push&label=" alt="AMD GPU Build">
            </a>
        </td>
    </tr>
    <tr>
        <td>Intel GPU</td>
        <td>
            <a href="https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml">
                <img src="https://img.shields.io/github/actions/workflow/status/linkedin/Liger-Kernel/intel-ci.yml?branch=main&event=push&label=" alt="Intel GPU Build">
            </a>
        </td>
    </tr>
    <tr>
        <td>Ascend NPU</td>
        <td>
            <a href="https://github.com/Ascend/Ascend-CI/actions/workflows/liger_kernel.yml">
                <img src="https://img.shields.io/github/actions/workflow/status/Ascend/Ascend-CI/liger_kernel.yml?branch=main&label=" alt="Ascend NPU Build">
            </a>
        </td>
    </tr>
</table>

## 联系我们

- 如遇到 Issue 或 Bug，欢迎在本项目仓库中提交 GitHub Issue。
- 参与开放讨论，欢迎加入我们在 [GPUMode 上的 Discord 频道](https://discord.com/channels/1189498204333543425/1275130785933951039)。
- 寻求正式商业或学术合作，请发送邮件至 Vaibhav Jindal(vjindal@linkedin.com)、Arup De(arde@linkedin.com)、Yiming Yue(yyue@linkedin.com)、Michael (Chun-Mao) Lai(chulai@linkedin.com)。

## 引用本项目

BibLaTeX 引用条目：
```bib
@inproceedings{
hsu2025ligerkernel,
title={Liger-Kernel: Efficient Triton Kernels for {LLM} Training},
author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen and Zhipeng Wang},
booktitle={Championing Open-source DEvelopment in ML Workshop @ ICML25},
year={2025},
url={https://openreview.net/forum?id=36SjAIT42G}
}
```

## Star 趋势

[![Star History Chart](https://star-history.dera.page/svg?repos=linkedin/Liger-Kernel&type=Date)](https://star-history.dera.page/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ 返回顶部 ↑
    </a>
</p>

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月31日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
