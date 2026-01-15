"""
Test suite for functional API with accuracy computation.

This module tests the functional API (liger_cross_entropy and liger_fused_linear_cross_entropy)
to ensure proper handling of return_accuracy parameter and flexible return signatures.
"""

import pytest
import torch

import liger_kernel.transformers.functional as F


def test_liger_cross_entropy_return_signatures():
    """Test that liger_cross_entropy returns correct signature based on flags."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Test 1: Default - only loss
    result = F.liger_cross_entropy(logits, targets)
    assert isinstance(result, torch.Tensor), "Should return single tensor"
    assert result.ndim == 0, "Should be scalar loss"

    # Test 2: return_z_loss=True - (loss, z_loss)
    result = F.liger_cross_entropy(logits, targets, return_z_loss=True)
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 2, "Should have 2 elements"
    loss, z_loss = result
    assert loss.ndim == 0 and z_loss.ndim == 0

    # Test 3: return_accuracy=True - (loss, accuracy)
    result = F.liger_cross_entropy(logits, targets, return_accuracy=True)
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 2, "Should have 2 elements"
    loss, accuracy = result
    assert loss.ndim == 0 and accuracy.ndim == 0

    # Test 4: Both flags - (loss, z_loss, accuracy)
    result = F.liger_cross_entropy(logits, targets, return_z_loss=True, return_accuracy=True)
    assert isinstance(result, tuple), "Should return tuple"
    assert len(result) == 3, "Should have 3 elements"
    loss, z_loss, accuracy = result
    assert loss.ndim == 0 and z_loss.ndim == 0 and accuracy.ndim == 0

    print("✓ All return signature tests passed")


def test_liger_cross_entropy_accuracy_correctness():
    """Test that functional API computes correct accuracy."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Compute with functional API
    loss, accuracy = F.liger_cross_entropy(logits, targets, return_accuracy=True)

    # Compute expected accuracy
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        expected_accuracy = (predictions == targets).float().mean()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print(f"✓ Accuracy correctness test passed: {accuracy.item():.4f}")


def test_liger_fused_linear_cross_entropy_return_signatures():
    """Test that liger_fused_linear_cross_entropy returns correct signature."""
    B, T, H, V = 2, 64, 512, 10000
    device = "cuda"
    torch.manual_seed(42)

    hidden = torch.randn(B * T, H, dtype=torch.float32, device=device)
    weight = torch.randn(V, H, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Test 1: Default - only loss
    result = F.liger_fused_linear_cross_entropy(hidden, weight, targets)
    assert isinstance(result, torch.Tensor), "Should return single tensor"
    assert result.ndim == 0, "Should be scalar loss"

    # Test 2: return_z_loss=True - (loss, z_loss)
    result = F.liger_fused_linear_cross_entropy(hidden, weight, targets, return_z_loss=True)
    assert isinstance(result, tuple) and len(result) == 2

    # Test 3: return_accuracy=True - (loss, accuracy)
    result = F.liger_fused_linear_cross_entropy(hidden, weight, targets, return_accuracy=True)
    assert isinstance(result, tuple) and len(result) == 2

    # Test 4: Both flags - (loss, z_loss, accuracy)
    result = F.liger_fused_linear_cross_entropy(hidden, weight, targets, return_z_loss=True, return_accuracy=True)
    assert isinstance(result, tuple) and len(result) == 3
    loss, z_loss, accuracy = result
    assert loss.ndim == 0 and z_loss.ndim == 0 and accuracy.ndim == 0

    print("✓ All fused return signature tests passed")


def test_liger_fused_linear_cross_entropy_accuracy_correctness():
    """Test that fused functional API computes correct accuracy."""
    B, T, H, V = 2, 64, 512, 10000
    device = "cuda"
    torch.manual_seed(42)

    hidden = torch.randn(B * T, H, dtype=torch.float32, device=device)
    weight = torch.randn(V, H, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Compute with functional API
    loss, accuracy = F.liger_fused_linear_cross_entropy(hidden, weight, targets, return_accuracy=True)

    # Compute expected accuracy
    with torch.no_grad():
        logits = hidden @ weight.t()
        predictions = torch.argmax(logits, dim=-1)
        expected_accuracy = (predictions == targets).float().mean()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print(f"✓ Fused accuracy correctness test passed: {accuracy.item():.4f}")


def test_functional_api_with_all_features():
    """Test functional API with multiple features enabled simultaneously."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Test with label_smoothing, softcap, and accuracy
    loss, accuracy = F.liger_cross_entropy(
        logits,
        targets,
        label_smoothing=0.1,
        softcap=30.0,
        return_accuracy=True,
    )

    # Verify accuracy is computed correctly with these features
    with torch.no_grad():
        capped_logits = 30.0 * torch.tanh(logits / 30.0)
        predictions = torch.argmax(capped_logits, dim=-1)
        expected_accuracy = (predictions == targets).float().mean()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print(f"✓ Multi-feature test passed: accuracy={accuracy.item():.4f}")


def test_functional_api_reduction_none():
    """Test functional API with reduction='none'."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Test with reduction='none'
    loss, accuracy = F.liger_cross_entropy(logits, targets, reduction="none", return_accuracy=True)

    # Loss and accuracy should both be per-token
    assert loss.shape == (B * T,), f"Expected shape {(B * T,)}, got {loss.shape}"
    assert accuracy.shape == (B * T,), f"Expected shape {(B * T,)}, got {accuracy.shape}"

    # Verify per-token accuracy
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        expected_accuracy = (predictions == targets).float()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print("✓ Reduction='none' test passed")


def test_functional_api_with_ignore_index():
    """Test functional API properly handles ignore_index with accuracy."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Set some targets to ignore_index
    targets[::5] = -100

    # Compute with functional API
    loss, accuracy = F.liger_cross_entropy(logits, targets, ignore_index=-100, return_accuracy=True)

    # Verify ignored indices don't affect accuracy
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        mask = targets != -100
        correct = (predictions == targets) & mask
        expected_accuracy = correct.sum().float() / mask.sum().float()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print(f"✓ Ignore index test passed: accuracy={accuracy.item():.4f}")


def test_functional_api_backward_pass():
    """Test that backward pass works correctly with accuracy enabled."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Compute loss and accuracy
    loss, accuracy = F.liger_cross_entropy(logits, targets, return_accuracy=True)

    # Backward pass should work
    loss.backward()

    # Gradients should be computed
    assert logits.grad is not None, "Gradients should be computed"
    assert not torch.isnan(logits.grad).any(), "Gradients should not be NaN"
    assert not torch.isinf(logits.grad).any(), "Gradients should not be Inf"

    print("✓ Backward pass test passed")


@pytest.mark.parametrize("return_z_loss", [False, True])
@pytest.mark.parametrize("return_accuracy", [False, True])
def test_functional_api_combinations(return_z_loss, return_accuracy):
    """Test all combinations of return flags."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    logits = torch.randn(B * T, V, dtype=torch.float32, device=device)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    result = F.liger_cross_entropy(
        logits,
        targets,
        return_z_loss=return_z_loss,
        return_accuracy=return_accuracy,
    )

    # Verify return type
    if not return_z_loss and not return_accuracy:
        assert isinstance(result, torch.Tensor), "Should return single tensor"
    else:
        assert isinstance(result, tuple), "Should return tuple"
        expected_len = sum([1, return_z_loss, return_accuracy])
        assert len(result) == expected_len, f"Expected {expected_len} elements"

    print(f"✓ Combination test passed: z_loss={return_z_loss}, accuracy={return_accuracy}")


if __name__ == "__main__":
    print("Running functional API accuracy tests...\n")

    test_liger_cross_entropy_return_signatures()
    test_liger_cross_entropy_accuracy_correctness()
    test_liger_fused_linear_cross_entropy_return_signatures()
    test_liger_fused_linear_cross_entropy_accuracy_correctness()
    test_functional_api_with_all_features()
    test_functional_api_reduction_none()
    test_functional_api_with_ignore_index()
    test_functional_api_backward_pass()

    print("\nTesting all combinations of return flags...")
    for return_z_loss in [False, True]:
        for return_accuracy in [False, True]:
            test_functional_api_combinations(return_z_loss, return_accuracy)

    print("\n✅ All functional API tests passed!")
