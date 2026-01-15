#!/usr/bin/env python3
"""
Simple demo with visible accuracy results
"""

import sys

import torch

sys.path.insert(0, "/mnt/home/kashif/Liger-Kernel/src")

from liger_kernel.ops.cross_entropy_with_accuracy import LigerCrossEntropyFunctionWithAccuracy


def main():
    print("=" * 70)
    print("SIMPLE DEMO: Token Accuracy Calculation")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small example for clarity
    batch_size = 2
    seq_len = 8
    vocab_size = 10  # Small vocab so we get some correct predictions

    BT = batch_size * seq_len
    V = vocab_size

    print("\nConfiguration:")
    print(f"  Tokens: {BT}")
    print(f"  Vocab: {V}")

    # Create logits that favor certain predictions
    logits = torch.randn(BT, V, device=device, dtype=torch.float32)

    # Create targets (random)
    target = torch.randint(0, V, (BT,), device=device)

    # Boost logits for the correct class to get ~50% accuracy
    for i in range(BT):
        if i % 2 == 0:  # Make every other token correct
            logits[i, target[i]] += 5.0  # Strong signal

    # Now set requires_grad after modifications
    logits.requires_grad = True

    print(f"\nTargets: {target.tolist()}")
    predictions = torch.argmax(logits, dim=-1)
    print(f"Predictions: {predictions.tolist()}")

    # Calculate accuracy both ways
    print("\n" + "=" * 70)

    # Method 1: Naive
    correct = (predictions == target).sum().item()
    naive_acc = correct / BT
    print(f"Naive accuracy: {naive_acc:.4f} ({correct}/{BT} correct)")

    # Method 2: Liger
    logits_liger = logits.detach().clone().requires_grad_(True)

    loss, z_loss, accuracy = LigerCrossEntropyFunctionWithAccuracy.apply(
        logits_liger,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        True,  # return_accuracy
    )

    print(f"Liger accuracy: {accuracy.item():.4f}")

    # Show per-token breakdown
    print("\n" + "=" * 70)
    print("Per-token breakdown:")
    print("=" * 70)
    print(f"{'Token':<8} {'Target':<8} {'Pred (Naive)':<15} {'Correct?':<10}")
    print("-" * 70)

    for i in range(BT):
        is_correct = "✓" if predictions[i] == target[i] else "✗"
        print(f"{i:<8} {target[i].item():<8} {predictions[i].item():<15} {is_correct:<10}")

    # Memory comparison
    print("\n" + "=" * 70)
    print("Memory Comparison")
    print("=" * 70)

    logits_memory = logits.numel() * logits.element_size() / 1024  # KB
    accuracy_memory = BT * 4 / 1024  # float32

    print(f"Logits tensor: {logits_memory:.2f} KB ({BT} × {V})")
    print(f"Accuracy output: {accuracy_memory:.2f} KB ({BT} floats)")
    print("\nLiger approach avoids materializing predictions!")
    print(f"Savings: {logits_memory:.2f} KB")

    # Show what happens with large vocab
    print("\n" + "=" * 70)
    print("Scaling to Real Models")
    print("=" * 70)

    scenarios = [
        ("GPT-2", 4, 1024, 50257),
        ("Llama-2-7B", 4, 2048, 32000),
        ("Llama-3-8B", 4, 2048, 128000),
    ]

    for name, bs, seq, vocab in scenarios:
        bt = bs * seq
        logits_mb = bt * vocab * 4 / (1024**2)
        accuracy_mb = bt * 4 / (1024**2)
        savings = logits_mb - accuracy_mb

        print(f"\n{name}:")
        print(f"  Config: {bs} × {seq} tokens, vocab={vocab}")
        print(f"  Logits: {logits_mb:.1f} MB")
        print(f"  Accuracy: {accuracy_mb:.3f} MB")
        print(f"  Savings: {savings:.1f} MB ({(savings / logits_mb) * 100:.1f}%)")


if __name__ == "__main__":
    main()
