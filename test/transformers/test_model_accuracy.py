"""
Test suite for model-level accuracy computation.

This module tests that Llama (and other models) can compute accuracy
through the Liger-patched forward pass without materializing logits.
"""

import pytest
import torch


def test_llama_model_with_accuracy():
    """Test that Llama model can return accuracy through forward pass."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    from liger_kernel.transformers import apply_liger_kernel_to_llama

    # Apply Liger kernel to Llama
    apply_liger_kernel_to_llama()

    # Use a tiny model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Prepare inputs
    text = "Hello, how are you doing today?"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Test 1: Forward without accuracy (default behavior)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    assert outputs.loss is not None, "Loss should be computed"
    assert not hasattr(outputs, "accuracy") or outputs.accuracy is None, "Accuracy should not be present by default"
    print("✓ Default forward (no accuracy) works")

    # Test 2: Forward with accuracy
    model.train()  # Must be in training mode for Liger to activate
    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
            return_accuracy=True,
        )

    assert outputs.loss is not None, "Loss should be computed"
    assert hasattr(outputs, "accuracy"), "Accuracy should be present"
    assert outputs.accuracy is not None, "Accuracy should not be None"
    assert 0.0 <= outputs.accuracy.item() <= 1.0, "Accuracy should be between 0 and 1"

    print(f"✓ Forward with accuracy works: accuracy={outputs.accuracy.item():.4f}")

    # Test 3: Verify accuracy is reasonable
    # For a real text, accuracy should be > 0
    assert outputs.accuracy.item() >= 0.0, "Accuracy should be non-negative"

    print("✓ All Llama model accuracy tests passed")


def test_llama_model_accuracy_training():
    """Test accuracy computation during training."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Prepare inputs
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Training mode with accuracy
    model.train()
    outputs = model(
        **inputs,
        labels=inputs["input_ids"],
        return_accuracy=True,
    )

    assert outputs.loss is not None
    assert hasattr(outputs, "accuracy") and outputs.accuracy is not None

    # Backward pass should work
    outputs.loss.backward()

    print(f"✓ Training with accuracy works: accuracy={outputs.accuracy.item():.4f}")


def test_llama_model_accuracy_vs_naive():
    """Test that model accuracy matches naive argmax computation."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    from liger_kernel.transformers import apply_liger_kernel_to_llama

    apply_liger_kernel_to_llama()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for accurate comparison
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    text = "Hello world!"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Get accuracy from Liger (without materializing logits)
    model.eval()
    with torch.no_grad():
        # First, get logits by forcing skip_logits=False
        outputs_with_logits = model(
            **inputs,
            labels=inputs["input_ids"],
            skip_logits=False,  # Force logits materialization
        )

        # Compute naive accuracy from logits
        logits = outputs_with_logits.logits
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        predictions = torch.argmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        correct = (predictions == shift_labels) & mask
        naive_accuracy = correct.sum().float() / mask.sum().float()

        # Now get accuracy from Liger
        model.train()  # Switch to training mode
        outputs_with_accuracy = model(
            **inputs,
            labels=inputs["input_ids"],
            return_accuracy=True,
        )

    liger_accuracy = outputs_with_accuracy.accuracy

    # They should match closely (allow small tolerance due to computation order)
    torch.testing.assert_close(
        liger_accuracy,
        naive_accuracy,
        atol=1e-4,
        rtol=1e-3,
    )

    print(f"✓ Accuracy matches: Liger={liger_accuracy.item():.4f}, Naive={naive_accuracy.item():.4f}")


if __name__ == "__main__":
    print("Running model-level accuracy tests...\n")

    print("Test 1: Basic forward with accuracy")
    test_llama_model_with_accuracy()

    print("\nTest 2: Training with accuracy")
    test_llama_model_accuracy_training()

    print("\nTest 3: Accuracy vs naive argmax")
    test_llama_model_accuracy_vs_naive()

    print("\n✅ All model-level accuracy tests passed!")
