"""Collect CPU/NPU traces separately from the unmodified acceptance benchmark."""

import argparse

from pathlib import Path

import torch
import torch_npu

from liger_kernel.transformers.softmax import LigerSoftmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["torch", "liger"], required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    x = torch.randn(args.rows, args.cols, device="npu", dtype=getattr(torch, args.dtype), requires_grad=True)
    grad = torch.randn_like(x)
    op = LigerSoftmax() if args.provider == "liger" else torch.nn.Softmax(dim=-1)

    def step():
        x.grad = None
        op(x).backward(grad)

    # Exclude compilation and allocator initialization from the trace.
    for _ in range(5):
        step()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
    ) as profiler:
        for _ in range(11):
            step()
            profiler.step()
        torch.npu.synchronize()


if __name__ == "__main__":
    main()
