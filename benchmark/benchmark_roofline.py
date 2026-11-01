"""
Roofline-style memory bandwidth benchmark for Liger kernels.
Run each --kernel mode under Nsight Compute (see docs/Roofline-Benchmark.md).
"""

import argparse

import torch

from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers.rms_norm import LigerRMSNorm

TOKENS = 4096  # batch * seq
HIDDEN = 4096  # LLaMA 3-8B
INTERMEDIATE = 14336  # LLaMA 3-8B
DTYPE = torch.float16
WARMUP = 10
ITERS = 20


def run(kernel: str):
    dev = "cuda"
    if kernel == "copy":
        x = torch.randn(TOKENS, INTERMEDIATE, device=dev, dtype=DTYPE)
        out = torch.empty_like(x)
        fn = lambda: torch.mul(x, 1.0, out=out)
    elif kernel == "rmsnorm":
        x = torch.randn(TOKENS, HIDDEN, device=dev, dtype=DTYPE)
        m = LigerRMSNorm(HIDDEN, eps=1e-6).to(dev).to(DTYPE)
        fn = lambda: m(x)
    elif kernel == "swiglu":
        a = torch.randn(TOKENS, INTERMEDIATE, device=dev, dtype=DTYPE)
        b = torch.randn(TOKENS, INTERMEDIATE, device=dev, dtype=DTYPE)
        fn = lambda: LigerSiLUMulFunction.apply(a, b)
    else:
        raise ValueError(kernel)

    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=["copy", "rmsnorm", "swiglu"], required=True)
    run(p.parse_args().kernel)
