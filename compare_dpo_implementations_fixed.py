#!/usr/bin/env python3

import time

from contextlib import contextmanager

import torch

from src.liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss

# Import all three implementations
from src.liger_kernel.ops.dpo_loss import triton_dpo_loss
from src.liger_kernel.ops.dpo_loss_optimized import optimized_triton_dpo_loss


@contextmanager
def memory_tracker():
    """Context manager to track GPU memory usage"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    yield

    end_memory = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"  Memory allocated: {(end_memory - start_memory) / 1024**2:.2f} MB")
    print(f"  Memory reserved: {(end_reserved - start_reserved) / 1024**2:.2f} MB")
    print(f"  Peak memory: {peak_memory / 1024**2:.2f} MB")


def create_test_data(batch_size, seq_len, vocab_size, device):
    """Create test data for DPO loss comparison"""
    # Ensure even batch size for chosen/rejected pairs
    assert batch_size % 2 == 0

    # Create policy model data
    logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device, requires_grad=True)

    # Create reference model data
    ref_logits = torch.randn(batch_size, seq_len + 1, vocab_size, device=device)

    # Create target tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create completion mask (all ones for simplicity)
    completion_mask = torch.ones(batch_size, seq_len, device=device)

    # For Liger implementation, we need hidden states and weight matrices
    hidden_size = 512
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device, requires_grad=True)
    lm_head_weight = torch.randn(vocab_size, hidden_size, device=device, requires_grad=True)
    ref_hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device)
    ref_weight = torch.randn(vocab_size, hidden_size, device=device)

    # Create labels for Liger (flattened input_ids, but keep valid tokens only)
    labels = input_ids.clone()
    labels = labels.view(-1)  # Flatten to match hidden_states

    return {
        "logits": logits,
        "ref_logits": ref_logits,
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "hidden_states": hidden_states,
        "lm_head_weight": lm_head_weight,
        "ref_hidden_states": ref_hidden_states,
        "ref_weight": ref_weight,
        "labels": labels,
    }


def test_triton_original(data, beta, loss_type, use_ref_model):
    """Test original Triton DPO implementation"""
    print(f"=== Original Triton DPO (loss_type={loss_type}, ref_model={use_ref_model}) ===")

    # Clone logits to avoid shared gradient issues
    test_logits = data["logits"].clone().detach().requires_grad_(True)

    with memory_tracker():
        start_time = time.time()

        try:
            loss, chosen_rewards, rejected_rewards = triton_dpo_loss(
                logits=test_logits,
                ref_logits=data["ref_logits"] if use_ref_model else None,
                input_ids=data["input_ids"],
                completion_mask=data["completion_mask"],
                beta=beta,
                loss_type=loss_type,
                use_ref_model=use_ref_model,
                temperature=1.0,
            )

            # Test backward pass
            loss.backward()

            end_time = time.time()

            grad_norm = test_logits.grad.norm().item() if test_logits.grad is not None else 0.0

        except Exception as e:
            print(f"  Error: {e}")
            return None

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm: {grad_norm:.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "gradients": test_logits.grad.detach().cpu() if test_logits.grad is not None else None,
        "time": end_time - start_time,
        "grad_norm": grad_norm,
    }

    return results


def test_triton_optimized(data, beta, loss_type, use_ref_model, test_inplace=False):
    """Test optimized Triton DPO implementation"""
    inplace_str = " (inplace)" if test_inplace else ""
    print(f"=== Optimized Triton DPO{inplace_str} (loss_type={loss_type}, ref_model={use_ref_model}) ===")

    # Clone logits to avoid shared gradient issues
    test_logits = data["logits"].clone().detach().requires_grad_(True)

    with memory_tracker():
        start_time = time.time()

        try:
            loss, chosen_rewards, rejected_rewards = optimized_triton_dpo_loss(
                logits=test_logits,
                ref_logits=data["ref_logits"] if use_ref_model else None,
                input_ids=data["input_ids"],
                completion_mask=data["completion_mask"],
                beta=beta,
                loss_type=loss_type,
                use_ref_model=use_ref_model,
                temperature=1.0,
                inplace=test_inplace,
            )

            # Test backward pass
            loss.backward()

            end_time = time.time()

            grad_norm = test_logits.grad.norm().item() if test_logits.grad is not None else 0.0

        except Exception as e:
            print(f"  Error: {e}")
            return None

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm: {grad_norm:.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "gradients": test_logits.grad.detach().cpu() if test_logits.grad is not None else None,
        "time": end_time - start_time,
        "grad_norm": grad_norm,
    }

    return results


def test_liger_implementation(data, beta, loss_type, use_ref_model):
    """Test Liger DPO implementation with proper data formatting"""
    print(f"=== Liger DPO (loss_type={loss_type}, ref_model={use_ref_model}) ===")

    # Clone tensors to avoid shared gradient issues
    test_hidden = data["hidden_states"].clone().detach().requires_grad_(True)
    test_weight = data["lm_head_weight"].clone().detach().requires_grad_(True)

    # Create Liger loss function
    liger_loss_fn = LigerFusedLinearDPOLoss(
        ignore_index=-100,
        beta=beta,
        use_ref_model=use_ref_model,
        loss_type=loss_type,
    )

    with memory_tracker():
        start_time = time.time()

        try:
            # Liger expects different input format
            loss_output = liger_loss_fn(
                lin_weight=test_weight,
                _input=test_hidden,
                target=data["labels"],
                ref_input=data["ref_hidden_states"] if use_ref_model else None,
                ref_weight=data["ref_weight"] if use_ref_model else None,
            )

            (
                loss,
                (
                    chosen_logps,
                    rejected_logps,
                    chosen_logits_mean,
                    rejected_logits_mean,
                    nll_loss,
                    chosen_rewards,
                    rejected_rewards,
                ),
            ) = loss_output

            # Test backward pass
            loss.backward()

            end_time = time.time()

            hidden_grad_norm = test_hidden.grad.norm().item() if test_hidden.grad is not None else 0.0
            weight_grad_norm = test_weight.grad.norm().item() if test_weight.grad is not None else 0.0

        except Exception as e:
            print(f"  Error: {e}")
            return None

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm (hidden): {hidden_grad_norm:.6f}")
    print(f"  Grad norm (weight): {weight_grad_norm:.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "time": end_time - start_time,
        "grad_norm": hidden_grad_norm + weight_grad_norm,
    }

    return results


def compare_results(original_results, optimized_results, liger_results=None, tolerance=1e-3):
    """Compare results between implementations"""
    print("\n=== COMPARISON RESULTS ===")

    if original_results is None:
        print("Original Triton failed")
        return {}

    if optimized_results is None:
        print("Optimized Triton failed")
        return {}

    # Compare losses
    loss_diff_opt = abs(original_results["loss"] - optimized_results["loss"])

    print("Loss comparison:")
    print(f"  Original: {original_results['loss']:.6f}")
    print(f"  Optimized: {optimized_results['loss']:.6f} (diff: {loss_diff_opt:.6f})")
    if liger_results:
        loss_diff_liger = abs(original_results["loss"] - liger_results["loss"])
        print(f"  Liger: {liger_results['loss']:.6f} (diff: {loss_diff_liger:.6f})")

    # Compare rewards
    chosen_diff_opt = torch.abs(original_results["chosen_rewards"] - optimized_results["chosen_rewards"]).mean()
    rejected_diff_opt = torch.abs(original_results["rejected_rewards"] - optimized_results["rejected_rewards"]).mean()

    print("\nReward comparison (Original vs Optimized):")
    print(f"  Chosen rewards diff: {chosen_diff_opt:.6f}")
    print(f"  Rejected rewards diff: {rejected_diff_opt:.6f}")

    # Compare performance
    print("\nPerformance comparison:")
    print(f"  Original time: {original_results['time'] * 1000:.2f} ms")
    print(f"  Optimized time: {optimized_results['time'] * 1000:.2f} ms")
    if liger_results:
        print(f"  Liger time: {liger_results['time'] * 1000:.2f} ms")

    # Compare gradient norms (important!)
    print("\nGradient comparison:")
    print(f"  Original grad norm: {original_results['grad_norm']:.6f}")
    print(f"  Optimized grad norm: {optimized_results['grad_norm']:.6f}")
    if liger_results:
        print(f"  Liger grad norm: {liger_results['grad_norm']:.6f}")

    # Check if original implementation has gradient bug
    if original_results["grad_norm"] < 1e-6:
        print("  ⚠️  WARNING: Original implementation has broken gradients!")

    # Check correctness
    loss_correct_opt = loss_diff_opt < tolerance
    rewards_correct = chosen_diff_opt < tolerance and rejected_diff_opt < tolerance

    print(f"\nCorrectness check (tolerance={tolerance}):")
    print(f"  Optimized loss correct: {loss_correct_opt}")
    print(f"  Optimized rewards correct: {rewards_correct}")

    return {
        "loss_correct_opt": loss_correct_opt,
        "rewards_correct": rewards_correct,
    }


def run_focused_test():
    """Run focused comparison test on key scenarios"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = "cuda"

    # Focus on one configuration first
    config = {"batch_size": 8, "seq_len": 256, "vocab_size": 16000}
    loss_types = ["sigmoid", "apo_zero"]
    ref_model_settings = [False, True]
    beta = 0.1

    print(f"Testing with {config}")

    # Create test data once
    data = create_test_data(device=device, **config)

    for loss_type in loss_types:
        for use_ref_model in ref_model_settings:
            print(f"\n{'-' * 60}")
            print(f"Loss type: {loss_type}, Use ref model: {use_ref_model}")
            print(f"{'-' * 60}")

            # Test implementations
            original_results = test_triton_original(data, beta, loss_type, use_ref_model)
            optimized_results = test_triton_optimized(data, beta, loss_type, use_ref_model)
            optimized_inplace_results = test_triton_optimized(data, beta, loss_type, use_ref_model, test_inplace=True)

            # Only test Liger for sigmoid (since others might have compatibility issues)
            liger_results = None
            if loss_type == "sigmoid":
                liger_results = test_liger_implementation(data, beta, loss_type, use_ref_model)

            # Compare results
            compare_results(original_results, optimized_results, liger_results)

            # Test inplace optimization
            if optimized_inplace_results:
                print(
                    f"\nInplace optimization speedup: {optimized_results['time'] / optimized_inplace_results['time']:.2f}x"
                )


if __name__ == "__main__":
    run_focused_test()
