from dataclasses import dataclass, field

import datasets
import torch
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama
import time

apply_liger_kernel_to_llama()

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     "/shared/public/models/Meta-Llama-3-8B",
#     use_cache=False,
#     attn_implementation="sdpa",
#     torch_dtype=torch.bfloat16,
# ).to("cuda")

# # Assuming a max sequence length of 128 tokens
# sequence_length = 128
# batch_size = 8  # You can adjust this as needed

# # Dummy input tensor with random integers representing token IDs
# dummy_input = torch.randint(0, 50256, (batch_size, sequence_length)).to(torch.long)

# # Move the dummy input to the appropriate device (e.g., GPU)
# dummy_input = dummy_input.to('cuda' if torch.cuda.is_available() else 'cpu')

# model = torch.compile(model)

# output = model(input_ids = dummy_input, labels = dummy_input)
# loss = output["loss"]
# loss.backward()

# print(loss)

# # import pdb; pdb.set_trace()
# # print(model)

# # model = torch.compile(model)

# # print(model)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/shared/public/models/Meta-Llama-3-8B",
    padding_side="left", truncation_side="left"
)

from torch.utils.data import Dataset, DataLoader
import torch

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

# Parameters
vocab_size = 128256  # Example vocabulary size, can be adjusted
num_sequences = 400  # Number of sequences in the dataset
fixed_length = 2048  # Set the desired fixed length in tokens

# Create the dataset and dataloader
dataset = FixedLengthTokenDataset(vocab_size=vocab_size, num_sequences=num_sequences, fixed_length=fixed_length)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
import transformers

# Assuming the dataset and dataloader are already created as in the previous example

# Initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "/shared/public/models/Meta-Llama-3-8B",
    use_cache=False,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
).to("cuda")

model.gradient_checkpointing_enable()
model = torch.compile(model)

# Define optimizer
optimizer = SGD(model.parameters(), lr=5e-5)

# Training loop (single pass through the dataset)
model.train()
start_time = time.time()
total_tokens = 0

from profiler import trace_handler


# Start profiling
# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(
#         wait=0, warmup=10, active=2, repeat=1, skip_first=0
#     ),
#     on_trace_ready=trace_handler(
#         "./profiler", True, True
#     ),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
# ) as prof:

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
    if step % 10 == 0:  # Adjust print frequency if necessary
        elapsed_time = time.time() - step_start_time
        throughput = input_ids.numel() / elapsed_time
        mem_allocated = torch.cuda.max_memory_allocated('cuda') / (1024 * 1024)  # in MB
        mem_reserved = torch.cuda.max_memory_reserved('cuda') / (1024 * 1024)    # in MB

        print(f"Step {step}, Loss: {loss.item():.4f}, Step Time: {step_time:.4f}s, "
            f"Throughput: {throughput:.2f} tokens/s, "
            f"Memory Allocated: {mem_allocated:.2f} MB, "
            f"Memory Reserved: {mem_reserved:.2f} MB")

# Print total time and throughput
print("Single pass through the dataset completed.")