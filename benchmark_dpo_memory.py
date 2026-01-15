import argparse
import json
import time

import torch
import tqdm

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_bs", type=int, default=512)  # Reduced for DPO (pairs)
    parser.add_argument("--mbs", type=int, default=8)  # Must be even for DPO pairs
    parser.add_argument("--mmbs", type=int, default=4)  # Must be even for DPO pairs
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    parser.add_argument("--loss", type=str, choices=["triton", "liger"], default="triton")
    parser.add_argument("--loss_type", type=str, choices=["sigmoid", "apo_zero", "apo_down"], default="sigmoid")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_ref_model", action="store_true", help="Use reference model")

    args = parser.parse_args()
    assert args.global_bs % args.mbs == 0 and args.mbs % args.mmbs == 0
    assert args.mbs % 2 == 0 and args.mmbs % 2 == 0, "Batch sizes must be even for DPO pairs"

    return args


def create_dpo_data(B, T, vocab_size, device):
    """Create DPO training data with chosen/rejected pairs"""
    # Input IDs for chosen and rejected sequences (interleaved)
    input_ids = torch.randint(0, vocab_size - 1, (B, T + 100), dtype=torch.int64, device=device)
    completion_ids = input_ids[:, -T:].contiguous()

    # Completion mask (all ones for simplicity)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)

    return completion_ids, completion_mask


if __name__ == "__main__":
    args = get_args()
    device = "cuda"

    # Set batch sizes (must be even for DPO pairs)
    B = args.mbs
    T = args.seq_len

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    lin_weight = model.lm_head.weight
    vocab_size = lin_weight.shape[0]

    # Get base model for hidden state extraction
    base_model = get_base_model(model)
    print(f"Model type: {type(model)}, Base model type: {type(base_model)}")

    # Create DPO data
    completion_ids, completion_mask = create_dpo_data(B, T, vocab_size, device)

    # Create reference logits if using reference model
    ref_logits = None
    if args.use_ref_model:
        with torch.no_grad():
            ref_input_ids = torch.cat([torch.zeros(B, 100, dtype=torch.int64, device=device), completion_ids], dim=1)
            ref_logits = model(ref_input_ids).logits[:, -(T + 1) :].detach()

    # Initialize loss functions
    if args.loss == "liger":
        liger_loss_fn = LigerFusedLinearDPOLoss(
            beta=args.beta, loss_type=args.loss_type, use_ref_model=args.use_ref_model, compiled=True, chunk_size=1
        )

    print(f"Starting benchmark: {args.loss} loss, {args.loss_type} type, beta={args.beta}")
    print(f"Global batch: {args.global_bs}, Micro batch: {args.mbs}, Micro-micro batch: {args.mmbs}")
    print(f"Sequence length: {args.seq_len}, Use ref model: {args.use_ref_model}")

    # Clear memory before benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pbar = tqdm.tqdm(total=args.global_bs)
    start_time = time.time()

    for mbs_idx in range(args.global_bs // args.mbs):
        for start in range(0, args.mbs, args.mmbs):
            end = start + args.mmbs

            if args.loss == "triton":
                # Triton DPO loss
                # Get logits for current micro-micro batch
                input_ids_batch = torch.cat(
                    [torch.zeros(args.mmbs, 100, dtype=torch.int64, device=device), completion_ids[start:end]], dim=1
                )
                logits = model(input_ids_batch, logits_to_keep=T + 1).logits[:, -(T + 1) :].contiguous()

                # Get reference logits for this batch
                ref_logits_batch = ref_logits[start:end] if ref_logits is not None else None
                if ref_logits_batch is not None:
                    ref_logits_batch = ref_logits_batch.contiguous()

                # Ensure input tensors are contiguous
                completion_ids_batch = completion_ids[start:end].contiguous()
                completion_mask_batch = completion_mask[start:end].contiguous()

                # Compute DPO loss using Triton
                loss, chosen_rewards, rejected_rewards = triton_dpo_loss(
                    logits=logits,
                    ref_logits=ref_logits_batch,
                    input_ids=completion_ids_batch,
                    completion_mask=completion_mask_batch,
                    beta=args.beta,
                    loss_type=args.loss_type,
                    use_ref_model=args.use_ref_model,
                    temperature=1.0,
                )

                loss.backward()
                del logits

            else:
                # Liger DPO loss
                # Get hidden states for current micro-micro batch
                input_ids_batch = torch.cat(
                    [torch.zeros(args.mmbs, 100, dtype=torch.int64, device=device), completion_ids[start:end]], dim=1
                )
                _input = base_model(input_ids=input_ids_batch, attention_mask=None).last_hidden_state[:, -(T + 1) : -1]

                # Get reference hidden states if using reference model
                ref_input = None
                ref_weight = None
                ref_bias = None
                if args.use_ref_model:
                    ref_input = _input.detach()  # Use same input for simplicity
                    ref_weight = lin_weight.detach()
                    ref_bias = (
                        model.lm_head.bias.detach()
                        if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None
                        else None
                    )

                # Compute DPO loss using Liger
                loss_output = liger_loss_fn(
                    lin_weight,
                    _input,
                    completion_ids[start:end],
                    bias=model.lm_head.bias if hasattr(model.lm_head, "bias") else None,
                    ref_input=ref_input,
                    ref_weight=ref_weight,
                    ref_bias=ref_bias,
                )

                loss = loss_output[0]
                loss.backward()

            pbar.update(args.mmbs)

    # Memory statistics
    memory_allocated = torch.cuda.memory_reserved()
    peak_memory = torch.cuda.max_memory_allocated()

    total_time = time.time() - start_time

    infos = {
        "global_bs": args.global_bs,
        "micro_bs": args.mbs,
        "micro_micro_bs": args.mmbs,
        "seq_len": args.seq_len,
        "loss_impl": args.loss,
        "loss_type": args.loss_type,
        "beta": args.beta,
        "use_ref_model": args.use_ref_model,
        "time(s)": round(total_time, 3),
        "samples/s": round(args.global_bs / total_time, 2),
        "memory_reserved(GB)": round(memory_allocated / 1024**3, 2),
        "peak_memory(GB)": round(peak_memory / 1024**3, 2),
        "model": args.model,
    }

    print("\nBenchmark Results:")
    print(json.dumps(infos, indent=2))

    # Save results
    with open("./dpo_benchmark_infos.jsonl", "a+") as f:
        f.write(json.dumps(infos, ensure_ascii=False) + "\n")

    print("\nResults saved to dpo_benchmark_infos.jsonl")
