from dataclasses import dataclass, field

import datasets
import torch
import transformers
import time
import functools

from liger_kernel.transformers import apply_liger_kernel_to_llama
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW, SGD
from transformers.models.llama import modeling_llama
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


# Initialize distributed environment
dist.init_process_group(backend="nccl")
device_id = dist.get_rank()
torch.cuda.set_device(device_id)

# # Initialize FSDP parameters
# fsdp_params = {
#     "cpu_offload": CPUOffload(offload_params=True),
#     "mixed_precision": MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
# }

class FixedLengthTokenDataset(Dataset):
    def __init__(self, vocab_size, num_sequences, fixed_length=128):
        self.vocab_size = vocab_size
        self.num_sequences = num_sequences
        self.fixed_length = fixed_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Randomly generate a sequence of tokens
        input_ids = torch.randint(0, self.vocab_size, (self.fixed_length,), dtype=torch.long)
        labels = input_ids.clone()  # For causal LM, labels are the input shifted by one
        attention_mask = torch.ones(self.fixed_length, dtype=torch.long)  # No masking, all tokens attended to

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

apply_liger_kernel_to_llama(
    cross_entropy=False,
    fused_linear_cross_entropy=True
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/shared/public/models/Meta-Llama-3-8B",
    padding_side="left", truncation_side="left"
)


# Parameters
vocab_size = 128256  # Example vocabulary size, can be adjusted
num_sequences = 400  # Number of sequences in the dataset
fixed_length = 2048  # Set the desired fixed length in tokens

# Create the dataset and dataloader
dataset = FixedLengthTokenDataset(vocab_size=vocab_size, num_sequences=num_sequences, fixed_length=fixed_length)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "/shared/public/models/Meta-Llama-3-8B",
    use_cache=False,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)


# model = torch.compile(model)
model.model.compile()

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    # Transformer layer class to wrap
    transformer_layer_cls=set([modeling_llama.LlamaDecoderLayer]),
)

# Wrap the model with FSDP
model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    device_id=dist.get_rank(),
    use_orig_params=True,
)


# act checkpointing
# https://sourcegraph.com/github.com/huggingface/accelerate/-/blob/src/accelerate/accelerator.py?L1493


apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
    auto_wrap_policy=wrap_policy,
)



# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop (single pass through the dataset)
model.train()
start_time = time.time()
total_tokens = 0

for step, batch in enumerate(dataloader):

    # Track step start time
    step_start_time = time.time()

    # Move batch to GPU
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    labels = batch['labels'].to('cuda')

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Calculate step time
    step_time = time.time() - step_start_time

    # Update total tokens processed
    total_tokens += input_ids.numel()

    # Print statistics every few steps
    if step % 10 == 0 and device_id == 0:  # Adjust print frequency if necessary
        elapsed_time = time.time() - step_start_time
        throughput = input_ids.numel() / elapsed_time
        mem_allocated = torch.cuda.max_memory_allocated('cuda') / (1024 * 1024)  # in MB
        mem_reserved = torch.cuda.max_memory_reserved('cuda') / (1024 * 1024)    # in MB

        print(f"[Rank {device_id}] Step {step}, Loss: {loss.item():.4f}, Step Time: {step_time:.4f}s, "
            f"Throughput: {throughput:.2f} tokens/s, "
            f"Memory Allocated: {mem_allocated:.2f} MB, "
            f"Memory Reserved: {mem_reserved:.2f} MB")

# Print total time and throughput
print("Single pass through the dataset completed.")

# Shutdown the process group
dist.destroy_process_group()

# [Rank 0] Step 20, Loss: 11.8414, Step Time: 8.2615s, Throughput: 3966.34 tokens/s, Memory Allocated: 57967.86 MB, Memory Reserved: 77476.00 MB