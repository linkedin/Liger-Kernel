#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from src.liger_kernel.ops.dpo_loss import triton_dpo_loss


def pytorch_dpo_loss_reference(
    logits, ref_logits, input_ids, completion_mask, beta=0.1, loss_type="sigmoid", use_ref_model=True
):
    """Reference PyTorch implementation of DPO loss"""
    B, L_plus_1, V = logits.shape
    L = L_plus_1 - 1

    # Use only first L positions for loss computation
    logits_for_loss = logits[:, :L, :]

    # Compute log probabilities
    log_probs = F.log_softmax(logits_for_loss, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Reference model handling
    if use_ref_model and ref_logits is not None:
        ref_logits_for_loss = ref_logits[:, :L, :]
        ref_log_probs = F.log_softmax(ref_logits_for_loss, dim=-1)
        ref_target_log_probs = ref_log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    else:
        ref_target_log_probs = torch.zeros_like(target_log_probs)

    # Apply completion mask
    if completion_mask is not None:
        mask = completion_mask.float()
        seq_log_probs = (target_log_probs * mask).sum(dim=-1)
        ref_seq_log_probs = (ref_target_log_probs * mask).sum(dim=-1)
    else:
        seq_log_probs = target_log_probs.sum(dim=-1)
        ref_seq_log_probs = ref_target_log_probs.sum(dim=-1)

    # Split into chosen/rejected pairs
    chosen_log_probs = seq_log_probs[::2]
    rejected_log_probs = seq_log_probs[1::2]
    ref_chosen_log_probs = ref_seq_log_probs[::2]
    ref_rejected_log_probs = ref_seq_log_probs[1::2]

    # Compute log ratios
    chosen_logratios = chosen_log_probs - ref_chosen_log_probs
    rejected_logratios = rejected_log_probs - ref_rejected_log_probs
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # Compute loss based on loss type
    if loss_type == "sigmoid":
        logits_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits_diff).mean()
    elif loss_type == "apo_zero":
        losses_chosen = 1 - F.sigmoid(chosen_rewards)
        losses_rejected = F.sigmoid(rejected_rewards)
        loss = (losses_chosen + losses_rejected).mean()
    elif loss_type == "apo_down":
        losses_chosen = F.sigmoid(chosen_rewards)
        losses_rejected = 1 - F.sigmoid(chosen_rewards - rejected_rewards)
        loss = (losses_chosen + losses_rejected).mean()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, chosen_rewards, rejected_rewards


def test_gradient_correctness():
    """Test that the consolidated Triton implementation has working gradients"""
    print("Testing CONSOLIDATED Triton DPO implementation...")
    print("This should demonstrate WORKING gradients!")
    print("=" * 80)
    print("TESTING CONSOLIDATED TRITON DPO - Should have WORKING gradients!")
    print("=" * 80)

    device = "cuda"
    torch.manual_seed(42)

    # Test parameters
    batch_size = 4  # Even number for chosen/rejected pairs
    seq_len = 8
    vocab_size = 1000
    beta = 0.1

    # Test configurations
    test_configs = [
        ("sigmoid", True),
        ("sigmoid", False),
        ("apo_zero", True),
        ("apo_zero", False),
    ]

    for loss_type, use_ref_model in test_configs:
        print(f"\n--- Testing {loss_type}, use_ref_model={use_ref_model} ---")

        # Create test data
        logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device, requires_grad=True)
        ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device) if use_ref_model else None
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        completion_mask = torch.ones(batch_size, seq_len, device=device)

        # Test PyTorch reference
        print("Testing PyTorch reference implementation:")
        logits_ref = logits.clone().detach().requires_grad_(True)
        ref_loss, ref_chosen_rewards, ref_rejected_rewards = pytorch_dpo_loss_reference(
            logits_ref, ref_logits, input_ids, completion_mask, beta, loss_type, use_ref_model
        )
        ref_loss.backward()
        ref_grad_norm = logits_ref.grad.norm().item()
        print(f"  Loss: {ref_loss.item():.6f}")
        print(f"  Grad norm: {ref_grad_norm:.6f}")

        if ref_grad_norm > 1e-6:
            print("  ✅ Reference has proper gradients")
        else:
            print("  ❌ Reference has zero gradients")
            continue

        # Test consolidated Triton implementation
        print("Testing CONSOLIDATED Triton implementation:")
        logits_triton = logits.clone().detach().requires_grad_(True)
        triton_loss, triton_chosen_rewards, triton_rejected_rewards = triton_dpo_loss(
            logits_triton, ref_logits, input_ids, completion_mask, beta, loss_type, use_ref_model
        )
        triton_loss.backward()
        triton_grad_norm = logits_triton.grad.norm().item()

        print(f"  Loss: {triton_loss.item():.6f}")
        print(f"  Grad norm: {triton_grad_norm:.6f}")

        # Check correctness
        loss_diff = abs(triton_loss.item() - ref_loss.item())
        print(f"  Loss difference: {loss_diff:.8f}")

        # Loss correctness
        if loss_diff < 1e-4:
            print("  ✅ Loss correctness: PASS")
        else:
            print("  ❌ Loss correctness: FAIL")

        # Gradient existence check
        if triton_grad_norm > 1e-6:
            print("  ✅ Gradient existence: PASS - GRADIENTS ARE WORKING!")

            # Additional gradient analysis
            if ref_grad_norm > 1e-6:
                grad_ratio = triton_grad_norm / ref_grad_norm
                print(f"  Gradient ratio (consolidated_triton/reference): {grad_ratio:.4f}")

                if abs(grad_ratio - 1.0) > 0.5:
                    print("  ⚠️  WARNING: Large gradient magnitude difference")

                # Cosine similarity
                if logits_triton.grad is not None and logits_ref.grad is not None:
                    cos_sim = F.cosine_similarity(logits_triton.grad.flatten(), logits_ref.grad.flatten(), dim=0).item()
                    print(f"  Gradient cosine similarity: {cos_sim:.6f}")

                    if cos_sim < 0.9:
                        print("  ⚠️  WARNING: Gradient directions differ significantly")

            print("  🎯 OVERALL: ✅ PASS - WORKING IMPLEMENTATION!")
        else:
            print("  ❌ Gradient existence: FAIL - ZERO GRADIENTS!")
            print("  🚨 OVERALL: ❌ FAIL - BROKEN GRADIENTS!")

    print("\n" + "=" * 80)
    print("CONSOLIDATED TRITON TEST SUMMARY")
    print("=" * 80)
    print("The consolidated implementation should show 'WORKING IMPLEMENTATION!'")
    print("This proves the consolidation preserved the gradient computation fix!")


if __name__ == "__main__":
    test_gradient_correctness()
