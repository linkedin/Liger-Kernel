import torch
import triton
import triton.testing
from liger_kernel.ops.swiglu import _swiglu_forward_kernel

@torch.no_grad()
def bench_one(n_rows, n_cols=12288, dtype=torch.bfloat16, block_size=16384, warps=16, stages=4, iters=200):
    a = torch.randn((n_rows, n_cols), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    stride = c.stride(0)

    def fn():
        _swiglu_forward_kernel[(n_rows,)](
            a, b, c, stride,
            n_cols=n_cols,
            BLOCK_SIZE=block_size,
            num_warps=warps,
            num_stages=stages,
        )

    # warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(fn, rep=iters)
    return ms

def sweep():
    n_cols = 12288
    rows_list = [30000, 100000]  # represent different (batch*seq) regimes
    configs = [
        ("default", 16384, 4, 2),
        ("default", 16384, 8, 2),
        ("default", 16384, 16, 2),
        ("default", 16384, 32, 2),
        ("default", 16384, 4, 3),
        ("default", 16384, 8, 3),
        ("default", 16384, 16, 3),
        ("default", 16384, 32, 3),
        ("default", 16384, 4, 4),
        ("default", 16384, 8, 4),
        ("default", 16384, 16, 4),
        ("default", 16384, 32, 4),
        ("default", 16384, 4, 5),
        ("default", 16384, 8, 5),
        ("default", 16384, 16, 5),
        ("default", 16384, 32, 5),
        ("default", 16384, 4, 6),
        ("default", 16384, 8, 6),
        ("default", 16384, 16, 6),
        ("default", 16384, 32, 6),
        # ("ncols-8w", 12288, 8),
        # ("ncols-16w", 12288, 16),
        # ("ncols-32w", 12288, 32),
        # ("big-32w", 32768, 32),
    ]
    for n_rows in rows_list:
        print("\nrows:", n_rows)
        best_ms = float('inf')
        best_config = None
        for name, bs, w, s in configs:
            ms = bench_one(n_rows, n_cols=n_cols, block_size=bs, warps=w, stages=s)
            # print(f"{name:10s}  BLOCK={bs:5d} warps={w:2d} stages={s:2d}  {ms:.4f} ms")
            if ms < best_ms:
                best_ms = ms
                best_config = (name, bs, w, s)
        if best_config is not None:
            name, bs, w, s = best_config
            print(f"BEST for rows={n_rows}: {name:10s}  BLOCK={bs:5d} warps={w:2d} stages={s:2d}  {best_ms:.4f} ms")

sweep()