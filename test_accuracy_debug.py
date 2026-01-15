"""Debug script to test if accuracy is returned from model."""

import torch

# Test the flow through loss_utils
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

print("Testing LigerForCausalLMLoss with return_accuracy=True...")

# Create dummy tensors
hidden_states = torch.randn(4, 512, device="cuda")  # batch*seq_len, hidden_size
lm_head_weight = torch.randn(32000, 512, device="cuda")  # vocab_size, hidden_size
labels = torch.randint(0, 32000, (4,), device="cuda")  # batch*seq_len

# Test without accuracy
result1 = LigerForCausalLMLoss(
    hidden_states=hidden_states, lm_head_weight=lm_head_weight, labels=labels, hidden_size=512, return_accuracy=False
)
print(f"Without accuracy - Result type: {type(result1)}")
print(f"Without accuracy - Is tuple: {isinstance(result1, tuple)}")

# Test with accuracy
result2 = LigerForCausalLMLoss(
    hidden_states=hidden_states, lm_head_weight=lm_head_weight, labels=labels, hidden_size=512, return_accuracy=True
)
print(f"\nWith accuracy - Result type: {type(result2)}")
print(f"With accuracy - Is tuple: {isinstance(result2, tuple)}")
if isinstance(result2, tuple):
    print(f"With accuracy - Tuple length: {len(result2)}")
    print(f"With accuracy - Loss: {result2[0]}")
    print(f"With accuracy - Accuracy: {result2[1]}")
else:
    print(f"ERROR: Expected tuple but got {type(result2)}")
