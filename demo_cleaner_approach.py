"""
Demonstrating a cleaner approach: No model patches needed!

Instead of patching 21+ models, we can:
1. Store accuracy as a model attribute
2. TRL reads it from the model after forward pass
"""

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast


# Simulated model
class FakeModel:
    def __init__(self):
        self._liger_last_accuracy = None

    def forward(self, **kwargs):
        # Model's forward pass (unchanged!)
        # Just calls LigerForCausalLMLoss normally
        loss = self.compute_loss_with_liger(**kwargs)

        # Model doesn't need to know about accuracy!
        # It just returns the loss as usual
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        return output

    def compute_loss_with_liger(self, return_token_accuracy=False):
        # Simulated LigerForCausalLMLoss call
        loss = torch.tensor(1.5)

        if return_token_accuracy:
            accuracy = torch.tensor(0.75)
            # Store accuracy on the model itself!
            self._liger_last_accuracy = accuracy

        # Always return just the loss (no tuple!)
        return loss


# TRL's side - extract accuracy after forward
def trl_compute_loss(model, inputs):
    """TRL's compute_loss method"""
    # Call model forward
    outputs = model.forward(**inputs)

    # After forward, check if model has accuracy
    if hasattr(model, "_liger_last_accuracy") and model._liger_last_accuracy is not None:
        # Add it to the output dict
        outputs["accuracy"] = model._liger_last_accuracy
        # Clean up
        model._liger_last_accuracy = None

    return outputs


# Test it
if __name__ == "__main__":
    model = FakeModel()

    # Without accuracy
    result1 = trl_compute_loss(model, {"return_token_accuracy": False})
    print(f"Without accuracy: loss={result1['loss']}, accuracy={'accuracy' in result1}")

    # With accuracy
    result2 = trl_compute_loss(model, {"return_token_accuracy": True})
    print(f"With accuracy: loss={result2['loss']}, accuracy={result2.get('accuracy')}")

    print("\n✅ Zero model patches needed!")
    print("   - Models don't call unpack_liger_loss_with_accuracy")
    print("   - Models don't create output dict early")
    print("   - LigerForCausalLMLoss stores accuracy on model")
    print("   - TRL reads it after forward pass")
