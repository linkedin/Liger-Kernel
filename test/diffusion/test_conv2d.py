"""
Full integration test for Triton Conv2D with all features:
- Forward pass
- Backward pass (input, weight, bias gradients)
- With and without bias
- Different configurations
"""

import torch
from liger_kernel.ops.conv2d import TritonConv2dFunction

def test_full_pipeline():
    """Test complete forward and backward pipeline"""
    torch.manual_seed(2024)
    
    print("=" * 70)
    print("FULL INTEGRATION TEST")
    print("=" * 70)
    
    # Configuration
    N, C_in, C_out = 2, 3, 4
    H, W = 8, 8
    K = 3
    padding = 1
    
    # Create inputs
    x = torch.randn(N, C_in, H, W, device='cuda', requires_grad=True)
    w = torch.randn(C_out, C_in, K, K, device='cuda', requires_grad=True)
    bias = torch.randn(C_out, device='cuda', requires_grad=True)
    
    print(f"\nConfiguration:")
    print(f"  Input: {x.shape}")
    print(f"  Weight: {w.shape}")
    print(f"  Bias: {bias.shape}")
    print(f"  Padding: {padding}")
    
    # === Test 1: With Bias ===
    print("\n" + "-" * 70)
    print("Test 1: Forward and Backward with Bias")
    print("-" * 70)
    
    # PyTorch reference
    y_ref = torch.nn.functional.conv2d(x, w, bias=bias, padding=padding)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    
    x_grad_ref = x.grad.clone()
    w_grad_ref = w.grad.clone()
    bias_grad_ref = bias.grad.clone()
    
    # Clear gradients
    x.grad = None
    w.grad = None
    bias.grad = None
    
    # Triton implementation
    y_triton = TritonConv2dFunction.apply(x, w, bias, padding)
    loss_triton = y_triton.sum()
    loss_triton.backward()
    
    # Compare
    forward_match = torch.allclose(y_triton, y_ref, atol=1e-5)
    x_grad_match = torch.allclose(x.grad, x_grad_ref, atol=1e-4, rtol=1e-3)
    w_grad_match = torch.allclose(w.grad, w_grad_ref, atol=1e-4, rtol=1e-3)
    bias_grad_match = torch.allclose(bias.grad, bias_grad_ref, atol=1e-4)
    
    print(f"Forward pass: {'✓ PASS' if forward_match else '✗ FAIL'}")
    print(f"  Max diff: {(y_triton - y_ref).abs().max().item():.6e}")
    print(f"Input gradient: {'✓ PASS' if x_grad_match else '✗ FAIL'}")
    print(f"  Max diff: {(x.grad - x_grad_ref).abs().max().item():.6e}")
    print(f"Weight gradient: {'✓ PASS' if w_grad_match else '✗ FAIL'}")
    print(f"  Max diff: {(w.grad - w_grad_ref).abs().max().item():.6e}")
    print(f"Bias gradient: {'✓ PASS' if bias_grad_match else '✗ FAIL'}")
    print(f"  Max diff: {(bias.grad - bias_grad_ref).abs().max().item():.6e}")
    
    test1_pass = forward_match and x_grad_match and w_grad_match and bias_grad_match
    
    # === Test 2: Without Bias ===
    print("\n" + "-" * 70)
    print("Test 2: Forward and Backward without Bias")
    print("-" * 70)
    
    # Reset gradients
    x.grad = None
    w.grad = None
    
    # PyTorch reference (no bias)
    y_ref_no_bias = torch.nn.functional.conv2d(x, w, bias=None, padding=padding)
    loss_ref_no_bias = y_ref_no_bias.sum()
    loss_ref_no_bias.backward()
    
    x_grad_ref_no_bias = x.grad.clone()
    w_grad_ref_no_bias = w.grad.clone()
    
    # Clear gradients
    x.grad = None
    w.grad = None
    
    # Triton implementation (no bias)
    y_triton_no_bias = TritonConv2dFunction.apply(x, w, None, padding)
    loss_triton_no_bias = y_triton_no_bias.sum()
    loss_triton_no_bias.backward()
    
    # Compare
    forward_match_no_bias = torch.allclose(y_triton_no_bias, y_ref_no_bias, atol=1e-5)
    x_grad_match_no_bias = torch.allclose(x.grad, x_grad_ref_no_bias, atol=1e-4, rtol=1e-3)
    w_grad_match_no_bias = torch.allclose(w.grad, w_grad_ref_no_bias, atol=1e-4, rtol=1e-3)
    
    print(f"Forward pass: {'✓ PASS' if forward_match_no_bias else '✗ FAIL'}")
    print(f"  Max diff: {(y_triton_no_bias - y_ref_no_bias).abs().max().item():.6e}")
    print(f"Input gradient: {'✓ PASS' if x_grad_match_no_bias else '✗ FAIL'}")
    print(f"  Max diff: {(x.grad - x_grad_ref_no_bias).abs().max().item():.6e}")
    print(f"Weight gradient: {'✓ PASS' if w_grad_match_no_bias else '✗ FAIL'}")
    print(f"  Max diff: {(w.grad - w_grad_ref_no_bias).abs().max().item():.6e}")
    
    test2_pass = forward_match_no_bias and x_grad_match_no_bias and w_grad_match_no_bias
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (with bias): {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Test 2 (without bias): {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(f"\nOverall: {'✓✓✓ ALL TESTS PASSED ✓✓✓' if test1_pass and test2_pass else '✗✗✗ SOME TESTS FAILED ✗✗✗'}")
    print("=" * 70)
    
    return test1_pass and test2_pass


if __name__ == "__main__":
    success = test_full_pipeline()
    exit(0 if success else 1)
