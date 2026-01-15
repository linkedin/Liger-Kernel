"""
Test suite for cross entropy with accuracy computation.

This module tests that the Liger cross entropy implementation can compute
token accuracy without materializing the full logits tensor, providing
massive memory savings for large vocabulary models.
"""

import pytest
import torch

from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction


def _test_cross_entropy_accuracy(
    B=8,
    T=128,
    V=32000,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
    lse_square_scale=0.0,
    weight=None,
    softcap=None,
    dtype=torch.float32,
    atol=1e-5,
    rtol=1e-4,
):
    """Helper function to test cross entropy accuracy computation."""
    device = "cuda"
    torch.manual_seed(42)

    # Create input and target
    _input = torch.randn(B * T, V, dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Add some ignore indices
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target[indices_to_assign] = ignore_index

    # Compute loss and accuracy with Liger
    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        False,  # return_z_loss
        True,  # return_accuracy
    )

    # Compute expected accuracy using argmax (naive approach)
    with torch.no_grad():
        logits = _input.clone()
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)

        predictions = torch.argmax(logits, dim=-1)
        mask = target != ignore_index
        correct = (predictions == target) & mask
        expected_accuracy = correct.sum().float() / mask.sum().float()

    # Check that accuracy matches
    torch.testing.assert_close(
        accuracy,
        expected_accuracy,
        atol=atol,
        rtol=rtol,
        msg=f"Accuracy mismatch: got {accuracy}, expected {expected_accuracy}",
    )

    print(f"✓ Accuracy test passed: {accuracy.item():.4f} == {expected_accuracy.item():.4f}")
    return accuracy, expected_accuracy


def test_cross_entropy_accuracy_basic():
    """Test basic accuracy computation."""
    _test_cross_entropy_accuracy(B=4, T=32, V=1000)


def test_cross_entropy_accuracy_large_vocab():
    """Test accuracy with large vocabulary (like Llama-3)."""
    _test_cross_entropy_accuracy(B=2, T=128, V=128256)


def test_cross_entropy_accuracy_with_label_smoothing():
    """Test accuracy with label smoothing."""
    # Accuracy should be computed on argmax, not smoothed predictions
    _test_cross_entropy_accuracy(B=4, T=32, V=1000, label_smoothing=0.1)


def test_cross_entropy_accuracy_with_softcap():
    """Test accuracy with softcapping."""
    _test_cross_entropy_accuracy(B=4, T=32, V=1000, softcap=30.0)


def test_cross_entropy_accuracy_reduction_none():
    """Test accuracy with reduction='none'."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    _input = torch.randn(B * T, V, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Set some ignore indices
    target[::10] = -100

    # Compute with Liger
    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "none",  # reduction
        None,  # softcap
        False,  # return_z_loss
        True,  # return_accuracy
    )

    # accuracy should be per-token
    assert accuracy.shape == (B * T,), f"Expected shape {(B * T,)}, got {accuracy.shape}"

    # Compute expected per-token accuracy
    with torch.no_grad():
        predictions = torch.argmax(_input, dim=-1)
        expected_accuracy = (predictions == target).float()
        # For ignored indices, accuracy should be 0
        expected_accuracy[target == -100] = 0.0

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print("✓ Reduction='none' test passed")


def test_cross_entropy_accuracy_all_correct():
    """Test when all predictions are correct."""
    B, T, V = 4, 32, 100
    device = "cuda"
    torch.manual_seed(42)

    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Create logits where the correct class has maximum value
    _input = torch.randn(B * T, V, dtype=torch.float32, device=device)
    for i in range(B * T):
        if target[i] != -100:
            _input[i, target[i]] = _input[i].max() + 10.0

    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        True,  # return_accuracy
    )

    # Accuracy should be 1.0
    torch.testing.assert_close(accuracy, torch.tensor(1.0, device=device), atol=1e-5, rtol=1e-4)
    print(f"✓ All correct test passed: accuracy = {accuracy.item():.4f}")


def test_cross_entropy_accuracy_all_wrong():
    """Test when all predictions are wrong."""
    B, T, V = 4, 32, 100
    device = "cuda"
    torch.manual_seed(42)

    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Create logits where a wrong class has maximum value
    _input = torch.randn(B * T, V, dtype=torch.float32, device=device)
    for i in range(B * T):
        if target[i] != -100:
            wrong_class = (target[i] + 1) % V
            _input[i, wrong_class] = _input[i].max() + 10.0
            _input[i, target[i]] = _input[i].min() - 10.0

    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        True,  # return_accuracy
    )

    # Accuracy should be 0.0
    torch.testing.assert_close(accuracy, torch.tensor(0.0, device=device), atol=1e-5, rtol=1e-4)
    print(f"✓ All wrong test passed: accuracy = {accuracy.item():.4f}")


def test_fused_linear_cross_entropy_accuracy():
    """Test fused linear cross entropy with accuracy."""
    B, T, H, V = 2, 64, 512, 10000
    device = "cuda"
    torch.manual_seed(42)

    # Create inputs
    _input = torch.randn(B * T, H, dtype=torch.float32, device=device, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Set some ignore indices
    target[::10] = -100

    # Compute with Liger fused linear CE
    loss, z_loss, accuracy = LigerFusedLinearCrossEntropyFunction.apply(
        _input.clone(),
        weight.clone(),
        target,
        None,  # bias
        None,  # ce_weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        None,  # accum_dtype
        False,  # use_token_scaling
        True,  # return_accuracy
    )

    # Compute expected accuracy
    with torch.no_grad():
        logits = _input @ weight.t()
        predictions = torch.argmax(logits, dim=-1)
        mask = target != -100
        correct = (predictions == target) & mask
        expected_accuracy = correct.sum().float() / mask.sum().float()

    torch.testing.assert_close(accuracy, expected_accuracy, atol=1e-5, rtol=1e-4)
    print(f"✓ Fused linear CE accuracy test passed: {accuracy.item():.4f} == {expected_accuracy.item():.4f}")


def test_accuracy_backward_compatibility():
    """Test that return_accuracy=False maintains backward compatibility."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    _input = torch.randn(B * T, V, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Call without return_accuracy (default False)
    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        False,  # return_accuracy (explicit)
    )

    # Accuracy should be None
    assert accuracy is None, f"Expected accuracy=None, got {accuracy}"
    print("✓ Backward compatibility test passed: accuracy is None when not requested")


def test_accuracy_memory_efficiency():
    """Test that accuracy computation doesn't materialize logits."""
    B, T, V = 4, 2048, 128256  # Llama-3 size
    device = "cuda"

    torch.manual_seed(42)
    _input = torch.randn(B * T, V, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)

    # Measure memory before
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.max_memory_allocated(device)

    # Compute with accuracy (should NOT materialize logits)
    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input,
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        "mean",  # reduction
        None,  # softcap
        False,  # return_z_loss
        True,  # return_accuracy
    )

    mem_after = torch.cuda.max_memory_allocated(device)
    mem_used = (mem_after - mem_before) / 1024**3  # Convert to GB

    # Expected memory for logits would be B * T * V * 4 bytes = ~4GB for this config
    # Accuracy should use only B * T * 4 bytes = ~32KB
    logits_size_gb = (B * T * V * 4) / 1024**3
    accuracy_size_gb = (B * T * 4) / 1024**3

    print("✓ Memory efficiency test:")
    print(f"  Logits would need: {logits_size_gb:.2f} GB")
    print(f"  Accuracy needs: {accuracy_size_gb:.4f} GB")
    print(f"  Actual memory used: {mem_used:.2f} GB")
    print(f"  Memory savings: {logits_size_gb - mem_used:.2f} GB")

    # Memory used should be much less than logits size
    assert mem_used < logits_size_gb * 0.1, (
        f"Memory usage {mem_used:.2f} GB is too high. Expected < {logits_size_gb * 0.1:.2f} GB (10% of logits size)"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_accuracy_dtypes_and_reductions(dtype, reduction):
    """Test accuracy with different dtypes and reductions."""
    B, T, V = 4, 32, 1000
    device = "cuda"
    torch.manual_seed(42)

    _input = torch.randn(B * T, V, dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)
    target[::10] = -100

    loss, z_loss, accuracy = LigerCrossEntropyFunction.apply(
        _input.clone(),
        target,
        None,  # weight
        -100,  # ignore_index
        0.0,  # lse_square_scale
        0.0,  # label_smoothing
        reduction,
        None,  # softcap
        False,  # return_z_loss
        True,  # return_accuracy
    )

    # Accuracy should always be float32
    assert accuracy.dtype == torch.float32, f"Accuracy should be float32, got {accuracy.dtype}"

    # Verify shape based on reduction
    if reduction == "none":
        assert accuracy.shape == (B * T,)
    else:
        assert accuracy.shape == ()

    print(f"✓ dtype={dtype}, reduction={reduction} test passed")


if __name__ == "__main__":
    print("Running cross entropy accuracy tests...\n")

    test_cross_entropy_accuracy_basic()
    test_cross_entropy_accuracy_large_vocab()
    test_cross_entropy_accuracy_with_label_smoothing()
    test_cross_entropy_accuracy_with_softcap()
    test_cross_entropy_accuracy_reduction_none()
    test_cross_entropy_accuracy_all_correct()
    test_cross_entropy_accuracy_all_wrong()
    test_fused_linear_cross_entropy_accuracy()
    test_accuracy_backward_compatibility()
    test_accuracy_memory_efficiency()

    print("\nTesting different dtypes and reductions...")
    for dtype in [torch.float32, torch.bfloat16]:
        for reduction in ["mean", "sum", "none"]:
            test_accuracy_dtypes_and_reductions(dtype, reduction)

    print("\n✅ All tests passed!")
