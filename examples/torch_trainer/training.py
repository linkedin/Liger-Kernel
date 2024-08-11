from dataclasses import dataclass, field

import datasets
import torch
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama
import time

apply_liger_kernel_to_llama(
    cross_entropy=False,
    fused_linear_cross_entropy=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "/shared/public/models/Meta-Llama-3-8B",
    use_cache=False,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Assuming a max sequence length of 128 tokens
sequence_length = 128
batch_size = 8  # You can adjust this as needed

# Dummy input tensor with random integers representing token IDs
dummy_input = torch.randint(0, 50256, (batch_size, sequence_length)).to(torch.long)

# Move the dummy input to the appropriate device (e.g., GPU)
dummy_input = dummy_input.to('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.compile(model)

output = model(input_ids = dummy_input, labels = dummy_input)
loss = output["loss"]
loss.backward()

print(loss)

# import pdb; pdb.set_trace()
# print(model)

# model = torch.compile(model)

# print(model)