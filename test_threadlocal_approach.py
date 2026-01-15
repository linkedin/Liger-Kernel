"""
Test: Can we use a cleaner approach with thread-local storage?

This avoids patching all 21+ model files.
"""

import threading

import torch

# Thread-local storage for accuracy
_liger_thread_local = threading.local()


def set_last_accuracy(accuracy: torch.Tensor):
    """Store accuracy in thread-local storage."""
    _liger_thread_local.accuracy = accuracy


def get_last_accuracy() -> torch.Tensor:
    """Retrieve accuracy from thread-local storage."""
    return getattr(_liger_thread_local, "accuracy", None)


def clear_last_accuracy():
    """Clear accuracy from thread-local storage."""
    if hasattr(_liger_thread_local, "accuracy"):
        del _liger_thread_local.accuracy


# Simulated Liger loss function
def LigerForCausalLMLoss_v2(return_accuracy=False):
    """Modified version that uses thread-local storage."""
    loss = torch.tensor(1.5)

    if return_accuracy:
        accuracy = torch.tensor(0.75)
        # Instead of returning tuple, store in thread-local
        set_last_accuracy(accuracy)

    # Always return just the loss
    return loss


# Simulated model forward (no changes needed!)
def model_forward_unchanged():
    """Model code stays exactly the same - no patches needed!"""
    # ... model computation ...
    loss = LigerForCausalLMLoss_v2(return_accuracy=True)

    # Model just returns loss as usual
    output = {"loss": loss, "logits": None}
    return output


# TRL's compute_loss can extract accuracy
def trl_compute_loss():
    """TRL side - extract accuracy from thread-local after forward."""
    output = model_forward_unchanged()

    # After forward, check if accuracy was computed
    accuracy = get_last_accuracy()
    if accuracy is not None:
        output["accuracy"] = accuracy
        clear_last_accuracy()  # Clean up

    return output


# Test it
if __name__ == "__main__":
    result = trl_compute_loss()
    print(f"Loss: {result['loss']}")
    print(f"Accuracy: {result.get('accuracy', 'N/A')}")
    print("✅ Thread-local approach works!")
    print(f"\\nAccuracy after clear: {get_last_accuracy()}")
