"""Head-to-head GRPO loss benchmark: torch chunked vs triton unchunked vs triton chunked.

Compares the three Liger GRPO loss implementations from the hidden-state
boundary (lm_head projection included, since the chunked variants fuse it):

  chunked_torch:  LigerFusedLinearGRPOLoss (fused linear, torch/cuBLAS chunking)
  triton:         triton_grpo_loss on materialized (B, L+1, V) logits
  chunked_triton: chunked_triton_grpo_loss (fused linear, Triton kernels)

Measures forward+backward wall time (CUDA events) and peak memory above the
resident inputs, per micro-batch. Config mirrors GRPO training on Qwen3.5-MoE:
dapo loss, sequence-level importance sampling, beta=0, temperature 1.0,
eps 0.2/0.2, hidden 2048, vocab 248320, bf16.

Run from the repo root:
  PYTHONPATH=src python benchmark/scripts/benchmark_chunked_grpo_loss_head_to_head.py
"""

import argparse

import torch

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.transformers.chunked_grpo_loss import chunked_triton_grpo_loss
from liger_kernel.transformers.grpo_loss import triton_grpo_loss

HIDDEN_SIZE = 2048
VOCAB_SIZE = 248320
LOSS_KWARGS = dict(
    temperature=1.0,
    beta=0.0,
    eps_low=0.2,
    eps_high=0.2,
    loss_type="dapo",
    importance_sampling_level="sequence",
)


def make_inputs(batch, seq_len, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(batch, seq_len + 1, HIDDEN_SIZE, device=device, generator=gen).to(torch.bfloat16).mul_(0.02)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device=device, generator=gen).to(torch.bfloat16).mul_(0.02)
    completion_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len), device=device, generator=gen)
    lengths = torch.randint(seq_len // 2, seq_len + 1, (batch,), device=device, generator=gen)
    mask = (torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    advantages = torch.randn(batch, device=device, dtype=torch.float32, generator=gen)
    return {
        "hidden": hidden,
        "weight": weight,
        "completion_ids": completion_ids,
        "mask": mask,
        "advantages": advantages,
        "num_items_in_batch": mask.sum(),
    }


def run_variant(variant, inputs, chunked_torch_module):
    hidden, weight = inputs["hidden"], inputs["weight"]
    common = dict(num_items_in_batch=inputs["num_items_in_batch"])
    if variant == "chunked_torch":
        loss, _ = chunked_torch_module(
            hidden[:, :-1, :],
            weight,
            inputs["completion_ids"],
            inputs["mask"],
            inputs["advantages"],
            **common,
        )
    elif variant == "triton":
        logits = hidden @ weight.t()
        loss, _ = triton_grpo_loss(
            logits,
            None,
            None,
            inputs["completion_ids"],
            inputs["advantages"],
            inputs["mask"],
            inplace=True,
            reduce=True,
            **LOSS_KWARGS,
            **common,
        )
    elif variant == "chunked_triton":
        loss, _ = chunked_triton_grpo_loss(
            hidden[:, :-1, :].contiguous(),
            weight,
            None,
            None,
            inputs["completion_ids"],
            inputs["advantages"],
            inputs["mask"],
            reduce=True,
            **LOSS_KWARGS,
            **common,
        )
    else:
        raise ValueError(variant)
    return loss


def bench(variant, inputs, chunked_torch_module, warmup, iters):
    hidden, weight = inputs["hidden"], inputs["weight"]

    def once():
        hidden.grad = None
        weight.grad = None
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = run_variant(variant, inputs, chunked_torch_module)
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - baseline) / 1024**3
        hidden.requires_grad_(False)
        weight.requires_grad_(False)
        return start.elapsed_time(end), peak

    try:
        for _ in range(warmup):
            once()
        times, peaks = zip(*[once() for _ in range(iters)])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"
    except RuntimeError as err:
        return f"FAIL: {err}"
    t = torch.tensor(times)
    return t.mean().item(), t.std().item(), max(peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 4096, 16384, 32768, 65535, 65536])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    chunked_torch_module = LigerFusedLinearGRPOLoss(
        beta=0.0,
        compiled=False,
        use_ref_model=False,
        epsilon_low=LOSS_KWARGS["eps_low"],
        epsilon_high=LOSS_KWARGS["eps_high"],
        loss_type=LOSS_KWARGS["loss_type"],
        importance_sampling_level=LOSS_KWARGS["importance_sampling_level"],
        temperature=LOSS_KWARGS["temperature"],
    )
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Hidden {HIDDEN_SIZE}, vocab {VOCAB_SIZE}, batch {args.batch_size}, bf16")
    print(f"Config: {LOSS_KWARGS}\n")

    # quick loss parity sanity check
    inputs = make_inputs(args.batch_size, 1024, device)
    losses = {
        v: run_variant(v, inputs, chunked_torch_module).item() for v in ["chunked_torch", "triton", "chunked_triton"]
    }
    print(f"Loss parity @1024: {losses}\n")
    del inputs

    variants = ["chunked_torch", "triton", "chunked_triton"]
    header = f"{'seq_len':>8} {'logits_GiB':>11}" + "".join(f" {v + '_ms':>22} {v + '_peak_GiB':>18}" for v in variants)
    print(header)
    print("-" * len(header))
    for seq_len in args.seq_lens:
        inputs = make_inputs(args.batch_size, seq_len, device)
        logits_gib = args.batch_size * (seq_len + 1) * VOCAB_SIZE * 2 / 1024**3
        row = f"{seq_len:>8} {logits_gib:>11.1f}"
        for variant in variants:
            result = bench(variant, inputs, chunked_torch_module, args.warmup, args.iters)
            if isinstance(result, str):
                label = result if len(result) < 20 else "LAUNCH FAIL"
                row += f" {label:>22} {'-':>18}"
            else:
                mean_ms, std_ms, peak = result
                row += f" {mean_ms:>14.1f} ±{std_ms:>5.1f} {peak:>18.2f}"
        print(row)
        del inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
