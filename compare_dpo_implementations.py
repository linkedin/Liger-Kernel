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

    # Create labels for Liger (flattened input_ids with padding)
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

    with memory_tracker():
        start_time = time.time()

        loss, chosen_rewards, rejected_rewards = triton_dpo_loss(
            logits=data["logits"],
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

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm: {data['logits'].grad.norm().item():.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "gradients": data["logits"].grad.detach().cpu(),
        "time": end_time - start_time,
    }

    # Clear gradients
    data["logits"].grad = None

    return results


def test_triton_optimized(data, beta, loss_type, use_ref_model):
    """Test optimized Triton DPO implementation"""
    print(f"=== Optimized Triton DPO (loss_type={loss_type}, ref_model={use_ref_model}) ===")

    with memory_tracker():
        start_time = time.time()

        loss, chosen_rewards, rejected_rewards = optimized_triton_dpo_loss(
            logits=data["logits"],
            ref_logits=data["ref_logits"] if use_ref_model else None,
            input_ids=data["input_ids"],
            completion_mask=data["completion_mask"],
            beta=beta,
            loss_type=loss_type,
            use_ref_model=use_ref_model,
            temperature=1.0,
            inplace=False,  # Test with inplace=False first
        )

        # Test backward pass
        loss.backward()

        end_time = time.time()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm: {data['logits'].grad.norm().item():.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "gradients": data["logits"].grad.detach().cpu(),
        "time": end_time - start_time,
    }

    # Clear gradients
    data["logits"].grad = None

    return results


def test_liger_implementation(data, beta, loss_type, use_ref_model):
    """Test Liger DPO implementation"""
    print(f"=== Liger DPO (loss_type={loss_type}, ref_model={use_ref_model}) ===")

    # Create Liger loss function
    liger_loss_fn = LigerFusedLinearDPOLoss(
        ignore_index=-100,
        beta=beta,
        use_ref_model=use_ref_model,
        loss_type=loss_type,
    )

    with memory_tracker():
        start_time = time.time()

        # Liger expects different input format
        loss_output = liger_loss_fn(
            lin_weight=data["lm_head_weight"],
            _input=data["hidden_states"],
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

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Chosen rewards mean: {chosen_rewards.mean().item():.6f}")
    print(f"  Rejected rewards mean: {rejected_rewards.mean().item():.6f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Grad norm (hidden): {data['hidden_states'].grad.norm().item():.6f}")
    print(f"  Grad norm (weight): {data['lm_head_weight'].grad.norm().item():.6f}")

    # Store results for comparison
    results = {
        "loss": loss.item(),
        "chosen_rewards": chosen_rewards.detach().cpu(),
        "rejected_rewards": rejected_rewards.detach().cpu(),
        "time": end_time - start_time,
    }

    # Clear gradients
    data["hidden_states"].grad = None
    data["lm_head_weight"].grad = None

    return results


def compare_results(original_results, optimized_results, liger_results, tolerance=1e-3):
    """Compare results between implementations"""
    print("\n=== COMPARISON RESULTS ===")

    # Compare losses
    loss_diff_opt = abs(original_results["loss"] - optimized_results["loss"])
    loss_diff_liger = abs(original_results["loss"] - liger_results["loss"])

    print("Loss comparison:")
    print(f"  Original: {original_results['loss']:.6f}")
    print(f"  Optimized: {optimized_results['loss']:.6f} (diff: {loss_diff_opt:.6f})")
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
    print(f"  Liger time: {liger_results['time'] * 1000:.2f} ms")
    print(f"  Speedup (Optimized): {original_results['time'] / optimized_results['time']:.2f}x")
    print(f"  Speedup (Liger): {original_results['time'] / liger_results['time']:.2f}x")

    # Check correctness
    loss_correct_opt = loss_diff_opt < tolerance
    loss_correct_liger = loss_diff_liger < tolerance
    rewards_correct = chosen_diff_opt < tolerance and rejected_diff_opt < tolerance

    print(f"\nCorrectness check (tolerance={tolerance}):")
    print(f"  Optimized loss correct: {loss_correct_opt}")
    print(f"  Liger loss correct: {loss_correct_liger}")
    print(f"  Optimized rewards correct: {rewards_correct}")

    return {
        "loss_correct_opt": loss_correct_opt,
        "loss_correct_liger": loss_correct_liger,
        "rewards_correct": rewards_correct,
        "speedup_opt": original_results["time"] / optimized_results["time"],
        "speedup_liger": original_results["time"] / liger_results["time"],
    }


def run_comprehensive_test():
    """Run comprehensive comparison test"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = "cuda"

    # Test parameters
    test_configs = [
        {"batch_size": 8, "seq_len": 256, "vocab_size": 16000},
        {"batch_size": 16, "seq_len": 512, "vocab_size": 32000},
    ]

    loss_types = ["sigmoid", "apo_zero", "apo_down"]
    ref_model_settings = [False, True]
    beta = 0.1

    all_results = []

    for config in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Testing with {config}")
        print(f"{'=' * 80}")

        # Create test data
        data = create_test_data(device=device, **config)

        for loss_type in loss_types:
            for use_ref_model in ref_model_settings:
                print(f"\n{'-' * 60}")
                print(f"Loss type: {loss_type}, Use ref model: {use_ref_model}")
                print(f"{'-' * 60}")

                try:
                    # Test all three implementations
                    original_results = test_triton_original(data, beta, loss_type, use_ref_model)
                    optimized_results = test_triton_optimized(data, beta, loss_type, use_ref_model)
                    liger_results = test_liger_implementation(data, beta, loss_type, use_ref_model)

                    # Compare results
                    comparison = compare_results(original_results, optimized_results, liger_results)

                    # Store summary
                    all_results.append(
                        {
                            "config": config,
                            "loss_type": loss_type,
                            "use_ref_model": use_ref_model,
                            "comparison": comparison,
                        }
                    )

                except Exception as e:
                    print(f"Error in test: {e}")
                    continue

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for result in all_results:
        config = result["config"]
        comparison = result["comparison"]
        print(
            f"Config: {config['batch_size']}x{config['seq_len']}x{config['vocab_size']}, "
            f"Loss: {result['loss_type']}, Ref: {result['use_ref_model']}"
        )
        print(f"  Optimized correct: {comparison['loss_correct_opt']}, Speedup: {comparison['speedup_opt']:.2f}x")
        print(f"  Liger correct: {comparison['loss_correct_liger']}, Speedup: {comparison['speedup_liger']:.2f}x")


if __name__ == "__main__":
    run_comprehensive_test()
