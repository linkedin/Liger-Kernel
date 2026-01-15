"""
Demo: Using custom output class eliminates the need for unpack helper!
"""

import torch

from src.liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast


# Simulated model forward pass
def model_forward_with_custom_output(**kwargs):
    # Simulate LigerForCausalLMLoss returning (loss, accuracy) tuple
    return_token_accuracy = kwargs.get("return_token_accuracy", False)

    if return_token_accuracy:
        loss = torch.tensor(1.5)
        accuracy = torch.tensor(0.75)
        result = (loss, accuracy)
    else:
        loss = torch.tensor(1.5)
        result = loss

    # OLD WAY: Had to unpack and manually add to dict
    # if isinstance(result, tuple):
    #     loss, accuracy = result
    #     output["accuracy"] = accuracy
    # else:
    #     loss = result

    # NEW WAY: Just unpack directly into the dataclass!
    if isinstance(result, tuple):
        loss, accuracy = result
        output = LigerCausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            accuracy=accuracy,  # ✨ Built-in field!
        )
    else:
        output = LigerCausalLMOutputWithPast(
            loss=result,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            # accuracy is None by default
        )

    return output


# Test it
if __name__ == "__main__":
    print("Without accuracy:")
    output1 = model_forward_with_custom_output(return_token_accuracy=False)
    print(f"  loss={output1.loss}, accuracy={output1.accuracy}")
    print(f"  'accuracy' in output? {'accuracy' in output1}")

    print("\nWith accuracy:")
    output2 = model_forward_with_custom_output(return_token_accuracy=True)
    print(f"  loss={output2.loss}, accuracy={output2.accuracy}")
    print(f"  'accuracy' in output? {'accuracy' in output2}")

    print("\n✅ Benefits:")
    print("   - No unpack_liger_loss_with_accuracy helper needed")
    print("   - Accuracy is a proper dataclass field")
    print("   - Type hints work correctly")
    print("   - Still works with dict access: output['accuracy']")

    # Verify dict access still works
    print(f"\nDict access test: output2['accuracy'] = {output2['accuracy']}")
