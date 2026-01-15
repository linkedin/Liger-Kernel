#!/usr/bin/env python3

import torch

from src.liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction

# Import both implementations
from src.liger_kernel.ops.dpo_loss import liger_compatible_dpo_loss

# Use the same device detection as the existing tests
try:
    from liger_kernel.utils import infer_device

    device = infer_device()
except ImportError:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def test_liger_compatible_dpo():
    """Test the new Liger-compatible Triton DPO implementation"""
    print(f"🚀 Testing Liger-Compatible Triton DPO on device: {device}")
    print("=" * 70)

    # Test parameters that match Liger's expected format
    batch_size = 4  # Must be even (chosen + rejected pairs)
    seq_len = 6
    hidden_size = 16
    vocab_size = 32
    beta = 0.1

    torch.manual_seed(42)

    # Create test data in Liger's expected format
    _input = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_size, device=device, requires_grad=True)
    bias = torch.randn(vocab_size, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Reference model data
    ref_input = torch.randn(batch_size, seq_len, hidden_size, device=device)
    ref_weight = torch.randn(vocab_size, hidden_size, device=device)
    ref_bias = torch.randn(vocab_size, device=device)

    print("📊 Test Configuration:")
    print(f"   Batch size: {batch_size} (chosen+rejected pairs)")
    print(f"   Sequence length: {seq_len}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Beta: {beta}")

    # Test all loss types
    for loss_type in ["sigmoid", "apo_zero", "apo_down"]:
        print(f"\n🧪 Testing loss_type: {loss_type}")
        print("-" * 40)

        # Test Liger reference implementation
        print("✅ Testing Liger DPO (reference)...")
        _input_liger = _input.clone().detach().requires_grad_(True)
        weight_liger = weight.clone().detach().requires_grad_(True)
        bias_liger = bias.clone().detach().requires_grad_(True)

        # Create dummy full_target for Liger (it uses shape[0] for normalization)
        target.clone()

        # Use Liger's full DPO function instead of trying to call chunk_forward directly
        # This ensures we get the exact same computation path as Liger

        # Flatten input for Liger format: [batch*seq, hidden]
        input_flat = _input_liger.view(-1, hidden_size)
        ref_input_flat = ref_input.view(-1, hidden_size)
        target_flat = target.view(-1)

        # Call Liger's full DPO function
        liger_result = LigerFusedLinearDPOFunction.apply(
            input_flat,
            weight_liger,
            target_flat,
            bias_liger,
            ref_input_flat,
            ref_weight,
            ref_bias,
            -100,  # ignore_index
            beta,
            False,  # compute_nll_loss
            True,  # compiled
            True,  # use_ref_model
            True,  # average_log_prob
            1,  # chunk_size
            loss_type,
        )

        # Extract results
        (
            liger_loss,
            (
                chosen_logps_liger,
                rejected_logps_liger,
                chosen_logits_mean,
                rejected_logits_mean,
                nll_loss,
                liger_chosen_rewards,
                liger_rejected_rewards,
            ),
        ) = liger_result

        # Results are already computed by Liger's apply function

        print(f"   Loss: {liger_loss.item():.6f}")
        print(f"   Chosen rewards: {liger_chosen_rewards.mean().item():.6f}")
        print(f"   Rejected rewards: {liger_rejected_rewards.mean().item():.6f}")

        # Test gradients
        liger_loss.backward()
        liger_grad_norm = _input_liger.grad.norm().item()
        print(f"   Grad norm: {liger_grad_norm:.6f}")

        # Test Triton implementation
        print("✅ Testing Triton DPO (Liger-compatible)...")
        _input_triton = _input.clone().detach().requires_grad_(True)
        weight_triton = weight.clone().detach().requires_grad_(True)
        bias_triton = bias.clone().detach().requires_grad_(True)

        try:
            triton_loss, triton_chosen_rewards, triton_rejected_rewards = liger_compatible_dpo_loss(
                _input=_input_triton,
                weight=weight_triton,
                target=target,
                bias=bias_triton,
                ref_input=ref_input,
                ref_weight=ref_weight,
                ref_bias=ref_bias,
                ignore_index=-100,
                beta=beta,
                loss_type=loss_type,
                use_ref_model=True,
                average_log_prob=True,
            )

            print(f"   Loss: {triton_loss.item():.6f}")
            print(f"   Chosen rewards: {triton_chosen_rewards.mean().item():.6f}")
            print(f"   Rejected rewards: {triton_rejected_rewards.mean().item():.6f}")

            # Test gradients
            triton_loss.backward()
            triton_grad_norm = _input_triton.grad.norm().item()
            print(f"   Grad norm: {triton_grad_norm:.6f}")

            # Compare results
            loss_diff = abs(liger_loss.item() - triton_loss.item())
            chosen_rewards_diff = (liger_chosen_rewards - triton_chosen_rewards).abs().max().item()
            rejected_rewards_diff = (liger_rejected_rewards - triton_rejected_rewards).abs().max().item()
            grad_diff = abs(liger_grad_norm - triton_grad_norm)

            print("\n📈 Comparison Results:")
            print(f"   Loss difference: {loss_diff:.8f}")
            print(f"   Chosen rewards max diff: {chosen_rewards_diff:.8f}")
            print(f"   Rejected rewards max diff: {rejected_rewards_diff:.8f}")
            print(f"   Grad norm difference: {grad_diff:.8f}")

            # Tolerance check
            tolerance = 1e-5
            if (
                loss_diff < tolerance
                and chosen_rewards_diff < tolerance
                and rejected_rewards_diff < tolerance
                and grad_diff < tolerance
            ):
                print(f"   ✅ {loss_type.upper()} PASSED - Perfect match!")
            else:
                print(f"   ❌ {loss_type.upper()} FAILED - Differences too large")

        except Exception as e:
            print(f"   ❌ Triton implementation failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    test_liger_compatible_dpo()
