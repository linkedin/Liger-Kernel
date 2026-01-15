#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from src.liger_kernel.ops.dpo_loss import fused_selective_log_softmax
from src.liger_kernel.ops.dpo_loss import triton_dpo_loss

# Use the same device detection as the existing tests
try:
    from liger_kernel.utils import infer_device

    device = infer_device()
except ImportError:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def debug_selective_log_softmax():
    """Debug the selective log softmax computation"""
    print("🔍 Debugging Selective Log Softmax")
    print("=" * 40)

    # Simple test case
    batch_size = 2
    seq_len = 4
    vocab_size = 8

    # Create simple test data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"Logits shape: {logits.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")

    # Test with PyTorch reference
    logits_for_loss = logits[:, :seq_len, :]
    log_probs_ref = F.log_softmax(logits_for_loss, dim=-1)
    target_log_probs_ref = log_probs_ref.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    seq_log_probs_ref = target_log_probs_ref.sum(dim=-1)

    print(f"PyTorch seq log probs: {seq_log_probs_ref}")

    # Test with Triton
    triton_log_probs = fused_selective_log_softmax(logits, input_ids)
    triton_seq_log_probs = triton_log_probs.sum(dim=-1)

    print(f"Triton seq log probs: {triton_seq_log_probs}")
    print(f"Difference: {(seq_log_probs_ref - triton_seq_log_probs).abs().max().item()}")


def debug_dpo_forward():
    """Debug the DPO forward computation with simple data"""
    print("\n🔍 Debugging DPO Forward")
    print("=" * 40)

    # Very simple test case - Liger format: [chosen, rejected]
    batch_size = 2  # 1 chosen + 1 rejected = 1 pair
    seq_len = 3
    vocab_size = 4
    beta = 0.1

    torch.manual_seed(42)

    # Create logits in Liger format: [chosen, rejected]
    logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device)
    ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    completion_mask = torch.ones(batch_size, seq_len, device=device)

    print(f"Batch size: {batch_size} (1 chosen + 1 rejected)")
    print(f"Input IDs: {input_ids}")
    print(f"Logits shape: {logits.shape}")

    # Test our implementation
    try:
        loss, chosen_rewards, rejected_rewards = triton_dpo_loss(
            logits=logits,
            ref_logits=ref_logits,
            input_ids=input_ids,
            completion_mask=completion_mask,
            beta=beta,
            loss_type="sigmoid",
            use_ref_model=True,
            temperature=1.0,
        )

        print("✅ Success!")
        print(f"Loss: {loss.item():.6f}")
        print(f"Chosen rewards: {chosen_rewards}")
        print(f"Rejected rewards: {rejected_rewards}")

        # Test gradients
        loss.backward()
        print("✅ Backward pass succeeded!")

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()


def debug_manual_dpo():
    """Debug DPO computation manually step by step"""
    print("\n🔍 Manual DPO Step-by-Step Debug")
    print("=" * 40)

    # Simple case
    batch_size = 2  # [chosen, rejected]
    seq_len = 2
    vocab_size = 3
    beta = 0.1

    torch.manual_seed(42)

    # Create simple logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"Logits:\n{logits}")
    print(f"Ref logits:\n{ref_logits}")
    print(f"Input IDs: {input_ids}")

    # Compute manually
    policy_log_probs = F.log_softmax(logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    policy_target_logps = policy_log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    ref_target_logps = ref_log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    print(f"Policy target log probs: {policy_target_logps}")
    print(f"Ref target log probs: {ref_target_logps}")

    # Sum over sequence
    policy_seq_logps = policy_target_logps.sum(dim=-1)
    ref_seq_logps = ref_target_logps.sum(dim=-1)

    print(f"Policy seq log probs: {policy_seq_logps}")
    print(f"Ref seq log probs: {ref_seq_logps}")

    # Split chosen/rejected (Liger format: [chosen, rejected])
    chosen_logps = policy_seq_logps[0:1]  # First half
    rejected_logps = policy_seq_logps[1:2]  # Second half
    ref_chosen_logps = ref_seq_logps[0:1]
    ref_rejected_logps = ref_seq_logps[1:2]

    print(f"Chosen logps: {chosen_logps}")
    print(f"Rejected logps: {rejected_logps}")
    print(f"Ref chosen logps: {ref_chosen_logps}")
    print(f"Ref rejected logps: {ref_rejected_logps}")

    # Compute log ratios
    chosen_log_ratios = chosen_logps - ref_chosen_logps
    rejected_log_ratios = rejected_logps - ref_rejected_logps

    print(f"Chosen log ratios: {chosen_log_ratios}")
    print(f"Rejected log ratios: {rejected_log_ratios}")

    # Compute rewards
    chosen_rewards = beta * chosen_log_ratios
    rejected_rewards = beta * rejected_log_ratios

    print(f"Chosen rewards (beta * log_ratios): {chosen_rewards}")
    print(f"Rejected rewards (beta * log_ratios): {rejected_rewards}")

    # Compute DPO loss
    logits_diff = beta * (chosen_log_ratios - rejected_log_ratios)
    loss = -F.logsigmoid(logits_diff).mean()

    print(f"Logits diff: {logits_diff}")
    print(f"DPO loss: {loss}")


if __name__ == "__main__":
    print(f"🚀 Running Debug Tests on device: {device}")

    debug_selective_log_softmax()
    debug_manual_dpo()
    debug_dpo_forward()
