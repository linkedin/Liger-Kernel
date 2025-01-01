## Model Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| LayerNorm                       | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy         | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|

### RMS Norm

RMS Norm simplifies the LayerNorm operation by eliminating mean subtraction, which reduces computational complexity while retaining effectiveness. 

This kernel performs normalization by scaling input vectors to have a unit root mean square (RMS) value. This method allows for a ~7x speed improvement and a ~3x reduction in memory footprint compared to
implementations in PyTorch.

!!! Example "Try it out"
    You can experiment as shown in this example [here](https://colab.research.google.com/drive/1CQYhul7MVG5F0gmqTBbx1O1HgolPgF0M?usp=sharing).

### RoPE

RoPE (Rotary Position Embedding) enhances the positional encoding used in transformer models.

The implementation allows for effective handling of positional information without incurring significant computational overhead.

!!! Example "Try it out"
    You can experiment as shown in this example [here](https://colab.research.google.com/drive/1llnAdo0hc9FpxYRRnjih0l066NCp7Ylu?usp=sharing).

### SwiGLU 

### GeGLU 

### CrossEntropy

This kernel is optimized for calculating the loss function used in classification tasks. 

The  kernel achieves a ~3x execution speed increase and a ~5x reduction in memory usage for substantial vocabulary sizes compared to implementations in PyTorch.

!!! Example "Try it out"
    You can experiment as shown in this example [here](https://colab.research.google.com/drive/1WgaU_cmaxVzx8PcdKB5P9yHB6_WyGd4T?usp=sharing).

### Fused Linear CrossEntropy

This kernel combines linear transformations with cross-entropy loss calculations into a single operation.

!!! Example "Try it out"
    You can experiment as shown in this example [here](https://colab.research.google.com/drive/1Z2QtvaIiLm5MWOs7X6ZPS1MN3hcIJFbj?usp=sharing)

## Alignment Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Fused Linear CPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss         | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |

## Distillation Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD                  | `liger_kernel.transformers.LigerFusedLinearJSD`             |

## Experimental Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul` |