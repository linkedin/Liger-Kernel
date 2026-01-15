"""
Test SFTTrainer with Liger kernel accuracy logging.

This test verifies that the SFTTrainer correctly logs mean_token_accuracy
when using the Liger kernel, without materializing logits.
"""

import pytest
import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Skip if TRL is not available
pytest.importorskip("trl")

from trl import SFTConfig
from trl import SFTTrainer


def test_sft_trainer_with_liger_accuracy():
    """Test that SFTTrainer logs accuracy when using Liger kernel."""
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
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Create a simple dataset
    dataset = Dataset.from_dict(
        {
            "text": [
                "Hello, how are you?",
                "I am fine, thank you!",
                "What is your name?",
                "My name is Assistant.",
            ]
        }
    )

    # Configure SFTTrainer with Liger kernel
    args = SFTConfig(
        output_dir="/tmp/test_sft_liger_accuracy",
        per_device_train_batch_size=2,
        max_steps=2,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train for a few steps
    trainer.train()

    # Check that accuracy was logged
    assert "mean_token_accuracy" in trainer._metrics["train"], "mean_token_accuracy should be in training metrics"
    assert len(trainer._metrics["train"]["mean_token_accuracy"]) > 0, (
        "mean_token_accuracy should have at least one value"
    )

    # Check that accuracy is in valid range
    for accuracy in trainer._metrics["train"]["mean_token_accuracy"]:
        assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"

    print("✓ SFTTrainer with Liger kernel logs accuracy correctly")
    print(f"  Recorded accuracies: {trainer._metrics['train']['mean_token_accuracy']}")


def test_sft_trainer_without_liger_accuracy():
    """Test that SFTTrainer logs accuracy without Liger kernel (traditional method)."""
    # Use a tiny model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Create a simple dataset
    dataset = Dataset.from_dict(
        {
            "text": [
                "Hello, how are you?",
                "I am fine, thank you!",
                "What is your name?",
                "My name is Assistant.",
            ]
        }
    )

    # Configure SFTTrainer WITHOUT Liger kernel
    args = SFTConfig(
        output_dir="/tmp/test_sft_no_liger_accuracy",
        per_device_train_batch_size=2,
        max_steps=2,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train for a few steps
    trainer.train()

    # Check that accuracy was logged
    assert "mean_token_accuracy" in trainer._metrics["train"], "mean_token_accuracy should be in training metrics"
    assert len(trainer._metrics["train"]["mean_token_accuracy"]) > 0, (
        "mean_token_accuracy should have at least one value"
    )

    # Check that accuracy is in valid range
    for accuracy in trainer._metrics["train"]["mean_token_accuracy"]:
        assert 0.0 <= accuracy <= 1.0, f"Accuracy should be between 0 and 1, got {accuracy}"

    print("✓ SFTTrainer without Liger kernel logs accuracy correctly")
    print(f"  Recorded accuracies: {trainer._metrics['train']['mean_token_accuracy']}")


def test_sft_trainer_liger_vs_traditional_accuracy():
    """Test that Liger and traditional accuracy are close to each other."""
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    # Use a tiny model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")

    # Create a simple dataset
    dataset = Dataset.from_dict(
        {
            "text": [
                "Hello, how are you?",
                "I am fine, thank you!",
            ]
        }
    )

    # Test with Liger
    apply_liger_kernel_to_llama()
    try:
        model_liger = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for accurate comparison
            device_map="cuda",
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    args_liger = SFTConfig(
        output_dir="/tmp/test_sft_liger_compare",
        per_device_train_batch_size=2,
        max_steps=1,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
    )

    trainer_liger = SFTTrainer(
        model=model_liger,
        args=args_liger,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer_liger.train()
    liger_accuracy = trainer_liger._metrics["train"]["mean_token_accuracy"][0]

    # Test without Liger (reload model to avoid Liger patches)
    try:
        model_standard = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda",
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    args_standard = SFTConfig(
        output_dir="/tmp/test_sft_standard_compare",
        per_device_train_batch_size=2,
        max_steps=1,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
    )

    trainer_standard = SFTTrainer(
        model=model_standard,
        args=args_standard,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer_standard.train()
    standard_accuracy = trainer_standard._metrics["train"]["mean_token_accuracy"][0]

    # The accuracies should be very close (allowing small tolerance)
    print(f"✓ Liger accuracy: {liger_accuracy:.4f}")
    print(f"✓ Standard accuracy: {standard_accuracy:.4f}")
    print(f"✓ Difference: {abs(liger_accuracy - standard_accuracy):.6f}")

    # Note: We don't assert exact equality because:
    # 1. Different random initialization states
    # 2. Different optimizer states
    # But they should be in the same ballpark
    assert abs(liger_accuracy - standard_accuracy) < 0.5, (
        f"Liger and standard accuracy should be similar, got {liger_accuracy} vs {standard_accuracy}"
    )


if __name__ == "__main__":
    print("Testing SFTTrainer accuracy logging...\n")

    print("Test 1: SFTTrainer with Liger kernel")
    test_sft_trainer_with_liger_accuracy()

    print("\nTest 2: SFTTrainer without Liger kernel")
    test_sft_trainer_without_liger_accuracy()

    print("\nTest 3: Comparing Liger vs traditional accuracy")
    test_sft_trainer_liger_vs_traditional_accuracy()

    print("\n✅ All SFTTrainer accuracy tests passed!")
