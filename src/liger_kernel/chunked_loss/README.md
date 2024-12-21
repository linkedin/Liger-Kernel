# Liger FlexChunkLoss: Alignment and Distillation loss 

Liger FlexChunkLoss offers a versatile interface, delivering up to 80% memory savings and a 10% throughput boost for post-training loss functions, including alignment (DPO, ORPO, CPO, KTO) and very soon, distillation. Its flexible design supports custom losses, ensuring efficiency gains across diverse use cases.

### User interface

FlexChunkLoss offers two flexible usage options:  

1. **Via `Liger[Custom Loss]Trainer`**  
   For example, by simply replacing the HuggingFace `ORPOTrainer` with `LigerORPOTrainer` in your code, you can leverage our optimized ORPO implementation and immediately benefit from improved performance.  

2. **Using `nn.Module` Implementations of Custom Loss Functions**  
   Explore the [LigerORPOTrainer implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/orpo_trainer.py) to see how the modular design integrates custom loss functions seamlessly.  

### What's under the hood?

We employ chunking and fused kernel optimizations to enhance performance. By fusing the final linear layer with loss computation and calculating backward gradients during the forward pass, we significantly reduce the need for storing intermediate activations. All operations are implemented in PyTorch, leveraging `torch.compile` to streamline kernel execution without relying on extensive low-level optimizations. Additionally, we minimize `torch.compile` recompilations to reduce overhead and ensure consistent performance gains.

### Extending to custom loss functions

We provide two base classes: `LigerFusedLinearPreferenceBase` for alignment use cases and `LigerFusedLinearDistillationBase` for distillation use cases. These base classes manage chunking, kernel fusions, and Torch compilation.

To implement a custom loss function, you need to create a subclass that defines the custom preference or distillation loss function, capable of processing a given input chunk. The base class will take care of the optimizations, handling most of the heavy lifting for you.

For a working example, refer to the [ORPO loss implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/orpo_loss.py).