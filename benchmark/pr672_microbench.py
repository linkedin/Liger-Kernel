#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import tqdm

from transformers import AutoModelForCausalLM

try:
    from triton_grpo_loss.core import triton_grpo_loss
except ModuleNotFoundError:
    from liger_kernel.transformers.grpo_loss import triton_grpo_loss

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_bs", type=int, default=1024)
    parser.add_argument("--mbs", type=int, default=16)
    parser.add_argument("--mmbs", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--model", type=str, default="/sharedata/mdy/models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--loss", type=str, choices=["my", "liger"], default="my")
    parser.add_argument("--loss-only", action="store_true", help="Isolate loss memory (no model forward).")
    parser.add_argument("--compiled", action="store_true", help="Use torch.compile in Liger loss.")
    parser.add_argument("--checkpoint-chunks", action="store_true", help="Checkpoint vocab chunks in Liger loss.")
    args = parser.parse_args()
    assert args.global_bs % args.mbs == 0 and args.mbs % args.mmbs == 0
    return args


if __name__ == "__main__":
    args = get_args()
    device = "cuda"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    if not os.path.exists(args.model):
        print(f"Using Hugging Face model id: {args.model}")

    # It should set B = args.mbs, then slice each part. But we just set B = args.mbs.
    B = args.mbs
    T = args.seq_len

    model = None
    lin_weight = None
    if not args.loss_only:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
        lin_weight = model.lm_head.weight
    else:
        lin_weight = torch.randn(50257, 2048, device=device, dtype=torch.bfloat16).requires_grad_(True)
    input_ids = torch.randint(0, 1000 - 1, (B, T + 100), dtype=torch.int64, device=device)
    completion_ids = input_ids[:, -T:].contiguous()
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)

    liger_loss_fn = LigerFusedLinearGRPOLoss(
        compiled=args.compiled,
        chunk_size=1,
        temperature=0.9,
        checkpoint_chunks=args.checkpoint_chunks,
    )

    pbar = tqdm.tqdm(total=args.global_bs)
    start_time = time.time()
    for _ in range(args.global_bs // args.mbs):
        total_token_this_mbs = completion_mask.sum()
        for start in range(0, args.mbs, args.mmbs):
            end = start + args.mmbs
            if args.loss == "my":
                if args.loss_only:
                    vocab = lin_weight.shape[0]
                    logits = torch.randn(args.mmbs, T + 1, vocab, device=device, dtype=torch.bfloat16).requires_grad_(
                        True
                    )
                else:
                    logits = model(input_ids[start:end], logits_to_keep=T + 1).logits
                torch.cuda.reset_peak_memory_stats()
                per_token_loss = triton_grpo_loss(
                    logits,
                    old_logp[start:end],
                    ref_logp[start:end],
                    completion_ids[start:end],
                    advantages[start:end],
                    completion_mask[start:end],
                )[0]
                loss = (per_token_loss * completion_mask[start:end]).sum() / total_token_this_mbs
                loss.backward()
                del logits
            else:
                if args.loss_only:
                    hidden = lin_weight.shape[1]
                    _input = torch.randn(args.mmbs, T, hidden, device=device, dtype=torch.bfloat16).requires_grad_(True)
                else:
                    _input = model.model(input_ids=input_ids[start:end], attention_mask=None).last_hidden_state[
                        :, -(T + 1) : -1
                    ]
                torch.cuda.reset_peak_memory_stats()
                # The loss is not right; it should use total_token_this_mbs to reduce the loss.
                loss, _ = liger_loss_fn(
                    _input,
                    lin_weight,
                    completion_ids[start:end],
                    completion_mask[start:end],
                    advantages[start:end],
                    ref_per_token_logps=ref_logp[start:end],
                    old_per_token_logps=old_logp[start:end],
                )
                loss.backward()
            pbar.update(args.mmbs)

    memory_allocated = torch.cuda.max_memory_allocated()

    total_time = time.time() - start_time
    infos = {
        "global_bs": args.global_bs,
        "micro_bs": args.mbs,
        "micro_micro_bs": args.mmbs,
        "seq_len": args.seq_len,
        "loss": args.loss,
        "time(s)": total_time,
        "sample/s": round(args.global_bs / total_time, 2),
        "memory(G)": round(memory_allocated / 1024**3, 2),
    }

    print(infos)
    with open("./infos.jsonl", "a+", encoding="utf-8") as fin:
        fin.write(json.dumps(infos, ensure_ascii=False) + "\n")
