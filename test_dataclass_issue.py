"""
Demonstrating why we can't just add attributes to ModelOutput dataclass.

The problem: When accelerate serializes/copies the output object,
Python attributes are lost but dict items survive.
"""

import pickle

from copy import deepcopy

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast

# Create a standard ModelOutput
output = CausalLMOutputWithPast(
    loss=torch.tensor(1.5),
    logits=torch.randn(2, 10, 50000),
    past_key_values=None,
    hidden_states=None,
    attentions=None,
)

# Try adding accuracy as an attribute
output.accuracy = torch.tensor(0.75)
print("✅ Original output has accuracy attribute:", hasattr(output, "accuracy"))
print(f"   output.accuracy = {output.accuracy}")

# Now try dict item assignment
output["my_custom_metric"] = torch.tensor(0.85)
print("✅ Original output has dict item:", "my_custom_metric" in output)
print(f"   output['my_custom_metric'] = {output['my_custom_metric']}")

print("\n" + "=" * 60)
print("What happens when accelerate copies the object?")
print("=" * 60)

# Simulate what accelerate does - deepcopy or pickle
print("\n1. Using deepcopy (what accelerate often does):")
output_copy = deepcopy(output)
print(f"   Has .accuracy attribute? {hasattr(output_copy, 'accuracy')}")
print(f"   Has ['my_custom_metric']? {'my_custom_metric' in output_copy}")
if "my_custom_metric" in output_copy:
    print(f"   ✅ Dict item survived: {output_copy['my_custom_metric']}")

print("\n2. Using pickle + unpickle (what happens in distributed training):")
serialized = pickle.dumps(output)
output_unpickled = pickle.loads(serialized)
print(f"   Has .accuracy attribute? {hasattr(output_unpickled, 'accuracy')}")
print(f"   Has ['my_custom_metric']? {'my_custom_metric' in output_unpickled}")
if "my_custom_metric" in output_unpickled:
    print(f"   ✅ Dict item survived: {output_unpickled['my_custom_metric']}")

print("\n" + "=" * 60)
print("Why does this happen?")
print("=" * 60)
print(f"ModelOutput is based on: {CausalLMOutputWithPast.__bases__}")
print("It's an OrderedDict subclass!")
print("- Dict items are part of the OrderedDict state")
print("- Python attributes are NOT part of OrderedDict state")
print("- When serialized, only OrderedDict state is preserved")

print("\n" + "=" * 60)
print("SOLUTION: Use output['accuracy'] instead of output.accuracy")
print("=" * 60)
