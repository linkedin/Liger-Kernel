#!/usr/bin/env python3
"""Snapshot CUDA memory while allocating GRPO-shaped tensors."""

import argparse
import os
import sys

import torch

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _mb(value: float) -> float:
    return value / (1024**2)


def _snapshot(label: str) -> None:
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    allocated = _mb(stats["allocated_bytes.all.current"])
    peak = _mb(stats["allocated_bytes.all.peak"])
    reserved = _mb(stats["reserved_bytes.all.current"])
    peak_reserved = _mb(stats["reserved_bytes.all.peak"])
    print(
        f"{label:22s} alloc={allocated:10.2f}MB  "
        f"peak={peak:10.2f}MB  reserved={reserved:10.2f}MB  "
        f"peak_reserved={peak_reserved:10.2f}MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO memory snapshot.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (B).")
    parser.add_argument("--seq", type=int, default=1024, help="Sequence length (T).")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size (H).")
    parser.add_argument("--vocab", type=int, default=128256, help="Vocab size (V).")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bf16", help="Tensor dtype.")
    parser.add_argument("--full-logits", action="store_true", help="Materialize full logits + log_probs.")
    parser.add_argument("--run-grpo", action="store_true", help="Run Liger GRPO forward/backward.")
    parser.add_argument("--with-ref", action="store_true", help="Allocate reference inputs and weights.")
    parser.add_argument("--with-ref-logps", action="store_true", help="Allocate ref_per_token_logps instead.")
    parser.add_argument("--with-old-logps", action="store_true", help="Allocate old_logps tensor.")
    parser.add_argument(
        "--loss-type",
        choices=["grpo", "bnpo", "dr_grpo", "dapo"],
        default="dapo",
        help="Loss type for GRPO.",
    )
    parser.add_argument(
        "--importance-sampling-level",
        choices=["token", "sequence"],
        default="token",
        help="Importance sampling level.",
    )
    parser.add_argument("--chunk-size", type=int, default=1, help="Chunk size for GRPO loss.")
    parser.add_argument("--beta", type=float, default=0.04, help="KL beta.")
    parser.add_argument("--epsilon-low", type=float, default=0.2, help="Lower clip epsilon.")
    parser.add_argument("--epsilon-high", type=float, default=0.2, help="Upper clip epsilon.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Logit temperature.")
    parser.add_argument("--compiled", action="store_true", help="Use torch.compile in GRPO loss.")
    parser.add_argument("--summary", action="store_true", help="Print torch.cuda.memory_summary at end.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    if args.with_ref and args.with_ref_logps:
        raise SystemExit("Choose only one of --with-ref or --with-ref-logps.")

    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _snapshot("start")

    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
    x.requires_grad_(True)
    _snapshot("input")

    weight = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(args.vocab, device=device, dtype=dtype, requires_grad=True)
    _snapshot("weight+bias")

    selected_ids = torch.randint(0, args.vocab, (args.batch, args.seq), device=device, dtype=torch.long)
    attention_mask = torch.ones(args.batch, args.seq, device=device, dtype=dtype)
    advantages = torch.randn(args.batch, device=device, dtype=dtype)
    _snapshot("ids/mask/adv")

    if args.with_old_logps:
        old_logps = torch.randn(args.batch, args.seq, device=device, dtype=torch.float32)
        _snapshot("old_logps")
    else:
        old_logps = None

    ref_input = None
    ref_weight = None
    ref_bias = None
    ref_logps = None
    if args.with_ref:
        ref_input = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
        ref_weight = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype)
        ref_bias = torch.randn(args.vocab, device=device, dtype=dtype)
        _snapshot("ref_input/weight")
    if args.with_ref_logps:
        ref_logps = torch.randn(args.batch, args.seq, device=device, dtype=torch.float32)
        _snapshot("ref_logps")

    if args.full_logits:
        logits = x @ weight.t()
        logits = logits + bias
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected_logps = log_probs.gather(dim=-1, index=selected_ids.unsqueeze(-1)).squeeze(-1)
        _snapshot("full_logits/logp")
        del logits, log_probs, selected_logps

    if args.run_grpo:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_root = os.path.join(repo_root, "src")
        sys.path.insert(0, src_root)
        sys.path.insert(0, repo_root)
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        use_ref_model = args.with_ref or args.with_ref_logps
        grpo_loss = LigerFusedLinearGRPOLoss(
            beta=args.beta,
            compiled=args.compiled,
            use_ref_model=use_ref_model,
            chunk_size=args.chunk_size,
            epsilon_low=args.epsilon_low,
            epsilon_high=args.epsilon_high,
            loss_type=args.loss_type,
            max_completion_length=args.seq if args.loss_type == "dr_grpo" else None,
            importance_sampling_level=args.importance_sampling_level,
            temperature=args.temperature,
        )

        num_items_in_batch = attention_mask.sum() if args.loss_type == "dapo" else None
        _snapshot("pre_grpo")
        torch.cuda.reset_peak_memory_stats()
        try:
            loss, _ = grpo_loss(
                x,
                weight,
                selected_ids,
                attention_mask,
                advantages,
                bias,
                ref_logps,
                old_logps,
                ref_input,
                ref_weight,
                ref_bias,
                num_items_in_batch,
            )
        except TypeError:
            loss, _ = grpo_loss(
                x,
                weight,
                selected_ids,
                attention_mask,
                advantages,
                bias,
                ref_logps,
                old_logps,
                ref_input,
                ref_weight,
                ref_bias,
            )
        _snapshot("grpo_forward")
        loss.backward()
        _snapshot("grpo_backward")

    if args.summary:
        print(torch.cuda.memory_summary())

    _snapshot("end")


if __name__ == "__main__":
    main()
