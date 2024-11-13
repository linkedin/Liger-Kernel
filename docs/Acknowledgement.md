
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
