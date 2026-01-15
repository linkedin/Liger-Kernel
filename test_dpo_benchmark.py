#!/usr/bin/env python3
"""
Quick test script to benchmark Triton vs Liger DPO memory usage
"""

import time
import traceback

import torch


# Import both implementations
from src.liger_kernel.ops.dpo_loss import triton_dpo_loss
from transformers import AutoModelForCausalLM

from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss


def get_base_model(model):
    """Get the base model from different model architectures"""
    # Common attribute names for base models
    base_model_attrs = ["model", "transformer", "gpt_neox", "bert", "roberta"]

    for attr in base_model_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)

    # If none found, the model itself might be the base model
    return model


def benchmark_memory_usage():
    """Quick memory usage comparison"""
    device = "cuda"

    # Small model for testing
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.train()  # Ensure model is in training mode and parameters require grad

    print(f"Model type: {type(model)}")

    # Get base model for inspection
    if hasattr(model, "model"):
        base_model = model.model
    elif hasattr(model, "transformer"):
        base_model = model.transformer
    else:
        base_model = model

    print(f"Base model type: {type(base_model)}")

    device = next(model.parameters()).device
    print(f"Device: {device}")

    # Generate synthetic data
    batch_size = 4
    seq_len = 256

    # Create input that includes both prompt and completion
    input_ids = torch.randint(1, 1000, (batch_size, seq_len + 100), device=device)
    completion_ids = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.bool)

    # Get logits
    input_ids = torch.cat([torch.zeros(batch_size, 50, dtype=torch.int64, device=device), completion_ids], dim=1)

    print("\n" + "=" * 60)
    print("BENCHMARKING DPO IMPLEMENTATIONS")
    print("=" * 60)

    # Test Triton implementation
    print("\n🚀 Testing Triton DPO Implementation")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    try:
        # Don't use no_grad() since we need gradients for backward pass
        logits = model(input_ids).logits[:, -(seq_len + 1) :].contiguous()

        # Ensure all tensors are contiguous
        completion_ids_contiguous = completion_ids.contiguous()
        completion_mask_contiguous = completion_mask.contiguous()

        print(f"Logits shape: {logits.shape}")
        print(f"Input IDs shape: {completion_ids_contiguous.shape}")
        print(f"Completion mask shape: {completion_mask_contiguous.shape}")
        print(f"Logits contiguous: {logits.is_contiguous()}")
        print(f"Input IDs contiguous: {completion_ids_contiguous.is_contiguous()}")
        print(f"Completion mask contiguous: {completion_mask_contiguous.is_contiguous()}")
        print(f"Logits requires grad: {logits.requires_grad}")

        loss, chosen_rewards, rejected_rewards = triton_dpo_loss(
            logits=logits,
            ref_logits=None,
            input_ids=completion_ids_contiguous,
            completion_mask=completion_mask_contiguous,
            beta=0.1,
            loss_type="sigmoid",
            use_ref_model=False,
            temperature=1.0,
        )

        loss.backward()

        triton_time = time.time() - start_time
        triton_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"✅ Triton - Time: {triton_time:.3f}s, Memory: {triton_memory:.2f}GB")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Chosen rewards: {chosen_rewards.mean().item():.6f}")
        print(f"   Rejected rewards: {rejected_rewards.mean().item():.6f}")

    except Exception as e:
        print(f"❌ Triton implementation failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        triton_time = float("inf")
        triton_memory = float("inf")

    # Test Liger implementation
    print("\n🔥 Testing Liger DPO Implementation")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    try:
        # Create Liger DPO loss function
        liger_loss_fn = LigerFusedLinearDPOLoss(
            beta=0.1, loss_type="sigmoid", use_ref_model=False, compiled=True, chunk_size=1
        )

        # Create hidden states (simulate base model output)
        hidden_size = base_model.config.hidden_size
        model_dtype = next(model.parameters()).dtype

        # Don't use no_grad() since we need gradients for backward pass
        # Ensure dtype matches the model
        hidden_states = torch.randn(
            batch_size, seq_len + 50, hidden_size, device=device, dtype=model_dtype, requires_grad=True
        )
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_size, device=device, dtype=model_dtype, requires_grad=True
        )

        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Hidden states requires grad: {hidden_states.requires_grad}")
        print(f"Input tensor requires grad: {input_tensor.requires_grad}")
        print(f"Hidden states dtype: {hidden_states.dtype}")
        print(f"Model dtype: {model_dtype}")

        # Get model's output embeddings
        lm_head = model.get_output_embeddings()

        # Create target tensor (labels)
        target = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

        # Use Liger DPO loss
        loss_output = liger_loss_fn(
            lm_head.weight,
            hidden_states,
            target,
            bias=lm_head.bias if hasattr(lm_head, "bias") and lm_head.bias is not None else None,
            ref_input=None,  # No reference model
            ref_weight=None,
            ref_bias=None,
        )

        if isinstance(loss_output, tuple):
            loss = loss_output[0]
            print(f"   Additional outputs: {len(loss_output) - 1}")
        else:
            loss = loss_output

        loss.backward()

        liger_time = time.time() - start_time
        liger_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"✅ Liger - Time: {liger_time:.3f}s, Memory: {liger_memory:.2f}GB")
        print(f"   Loss: {loss.item():.6f}")

    except Exception as e:
        print(f"❌ Liger implementation failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        liger_time = float("inf")
        liger_memory = float("inf")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if triton_time != float("inf") and liger_time != float("inf"):
        speedup = liger_time / triton_time
        memory_ratio = liger_memory / triton_memory

        print(f"⚡ Speed: Triton is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Liger")
        print(f"💾 Memory: Triton uses {memory_ratio:.2f}x {'more' if memory_ratio > 1 else 'less'} memory than Liger")

        if speedup > 1.1:
            print("🏆 Triton wins on speed!")
        elif speedup < 0.9:
            print("🏆 Liger wins on speed!")
        else:
            print("🤝 Similar speed performance")

        if memory_ratio < 0.9:
            print("🏆 Triton wins on memory efficiency!")
        elif memory_ratio > 1.1:
            print("🏆 Liger wins on memory efficiency!")
        else:
            print("🤝 Similar memory usage")

    print("\n✨ Benchmark complete!")


if __name__ == "__main__":
    benchmark_memory_usage()
