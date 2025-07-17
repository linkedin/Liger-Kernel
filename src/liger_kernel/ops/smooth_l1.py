
import time
import torch
import triton
import triton.language as tl
from typing import Literal
import numpy as np

MAX_FUSED_SIZE = 65536 // 4

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_SUM = tl.constexpr(1)
_REDUCTION_MODE_MEAN = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}


@triton.jit
def l1_smooth_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    loss_mask_ptr,
    loss_mask_stride,
    loss_count_ptr,
    loss_count_stride,
    HAS_LOSS_MASK: tl.constexpr,
    n_cols,
    beta: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    X_ptr += row_id * X_stride
    Y_ptr += row_id * Y_stride
    loss_mask_ptr += row_id * loss_mask_stride
    loss_sum = 0.0
    loss_count = 0
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = tl.abs(x - y)
        loss = tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        if HAS_LOSS_MASK:
            loss_mask = tl.load(loss_mask_ptr + col_offsets, mask=mask, other=False)
            loss = tl.where(loss_mask, loss, 0.0)
            loss_count += tl.sum(loss_mask, axis=0)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + col_offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)
        if HAS_LOSS_MASK:
            tl.store(loss_count_ptr, loss_count)

def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


def smooth_l1_loss(X, Y, beta=1.0, reduction="none", loss_mask=None, eps=1e-5):
    n_rows, n_cols = X.shape
    # We don't need to allocate sum value
    # TODO: implement eps
    loss_shape = (n_rows, n_cols) if reduction == "none" else (n_rows,)
    loss = torch.zeros(loss_shape, device=X.device, dtype=torch.float32)
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = get_num_warps(BLOCK_SIZE)
    if loss_mask is not None:
        assert loss_mask.shape == (n_rows, n_cols), f"loss_mask must be of shape (BT, V). Got: {loss_mask.shape}"
        loss_mask = loss_mask.contiguous()
        has_loss_mask = True
        loss_count = torch.zeros(n_rows, device=X.device, dtype=torch.float32)
    else:
        loss_mask = torch.empty(0, device=X.device, dtype=torch.bool)
        has_loss_mask = False
        loss_count = None

    l1_smooth_kernel[n_rows,](
        X, X.stride(0), 
        Y, Y.stride(0),
        loss, loss.stride(0), 
        loss_mask, loss_mask.stride(0) if has_loss_mask else 0,
        loss_count, loss_count.stride(0) if has_loss_mask else 0,
        has_loss_mask,
        n_cols,
        beta,
        reduction=_str_to_reduction_mode[reduction],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return loss.sum()
    elif reduction == _REDUCTION_MODE_SUM.value:
        return loss.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return loss.sum()
    else:
        return loss


def benchmark_function(func, *args, **kwargs):
    """Benchmark a function with multiple iterations and return average time."""
    # Warmup
    for _ in range(5):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Actual timing
    times = []
    n_iterations = 50
    
    for _ in range(n_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times), result


def run_correctness_test(X, Y, beta=1.0, reduction="none"):
    """Verify that Triton kernel produces correct results."""
    # Reference implementation
    loss_fn = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    ref_loss = loss_fn(X, Y)
    
    # Triton implementation
    triton_loss = smooth_l1_loss(X, Y, beta=beta, reduction=reduction)
    
    # Compare results
    if reduction == "none":
        max_diff = torch.max(torch.abs(ref_loss - triton_loss)).item()
        rel_error = torch.max(torch.abs((ref_loss - triton_loss) / (ref_loss + 1e-8))).item()
    else:
        max_diff = torch.abs(ref_loss - triton_loss).item()
        rel_error = torch.abs((ref_loss - triton_loss) / (ref_loss + 1e-8)).item()
    
    return max_diff, rel_error

def forward_full(X, Y, beta=1.0, reduction="none", loss_mask=None, eps=1e-5):
    loss = torch.nn.SmoothL1Loss(reduction="none")(X, Y)
    loss_reg = torch.sum(torch.mean(loss_mask * loss, 1)) / (loss_mask.sum() + eps)
    return loss_reg


def benchmark_smooth_l1():
    """Run comprehensive benchmark comparing PyTorch vs Triton implementations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        # (BT, V)
        (1024, 2**17),
        (2048, 2**17),
        (4096, 2**17),
        (8192, 2**17),
        (16384, 2**17),
        # (32768, 2**17),
        # (65536, 2**17),
    ]
    
    # reduction_modes = ["none", "mean", "sum"]
    reduction_modes = ["none"]
    # beta_values = [0.1, 1.0, 2.0]
    
    print(f"{'Config':<15} {'Reduction':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 120)
    
    for batch_size, seq_len in test_configs:
        for reduction in reduction_modes:
            # Generate test data
            X = torch.randn(batch_size, seq_len, device=device, dtype=torch.float32, requires_grad=False)
            Y = torch.randn(batch_size, seq_len, device=device, dtype=torch.float32, requires_grad=False)
            loss_mask = torch.randint(0, 2, (batch_size, seq_len), device=device, dtype=torch.bool)
            # Correctness test
            # max_diff, rel_error = run_correctness_test(X, Y, reduction=reduction)
            # Benchmark Triton kernel
            triton_time, triton_std, _ = benchmark_function(
                smooth_l1_loss, X, Y, reduction="batchmean", loss_mask=loss_mask, eps=1e-5
            )
            # Benchmark PyTorch reference
            # loss_fn = forward_full(X, Y, reduction=reduction, loss_mask=loss_mask, eps=1e-5)
            pytorch_time, pytorch_std, _ = benchmark_function(forward_full, X, Y, loss_mask=loss_mask, eps=1e-5)
            
            # Calculate speedup
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            config_str = f"{batch_size}x{seq_len}"
            print(f"{config_str:<15} {reduction:<10} {pytorch_time*1000:<15.3f} {triton_time*1000:<15.3f} {speedup:<10.2f}")
    
    print("=" * 120)
    print("\nBenchmark completed!")
    print("Notes:")
    print("- Times are averaged over 50 iterations")
    print("- Max Diff: Maximum absolute difference between reference and triton")
    print("- Rel Error: Maximum relative error")
    print("- Speedup: PyTorch time / Triton time")


def profile_memory_usage():
    """Profile peak memory usage of both implementations."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return
    
    device = "cuda"
    batch_size, seq_len = 512, 16384
    
    print("\nPeak Memory Usage Profiling")
    print("=" * 50)
    
    X = torch.randn(batch_size, seq_len, device=device, dtype=torch.float32, requires_grad=False)
    Y = torch.randn(batch_size, seq_len, device=device, dtype=torch.float32, requires_grad=False)
    loss_mask = torch.randint(0, 2, (batch_size, seq_len), device=device, dtype=torch.bool)
    
    # Profile PyTorch reference peak memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    ref_loss = forward_full(X, Y, reduction="none", loss_mask=loss_mask, eps=1e-5)
    
    torch.cuda.synchronize()
    pytorch_peak_memory = torch.cuda.max_memory_allocated()
    
    # Clean up and profile Triton kernel peak memory
    del ref_loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    triton_loss = smooth_l1_loss(X, Y, reduction="batchmean", loss_mask=loss_mask, eps=1e-5)
    
    torch.cuda.synchronize()
    triton_peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"PyTorch Peak Memory: {pytorch_peak_memory / 1024**2:.2f} MB")
    print(f"Triton Peak Memory:  {triton_peak_memory / 1024**2:.2f} MB")
    print(f"Memory Ratio:        {pytorch_peak_memory / triton_peak_memory:.2f}x")


if __name__ == "__main__":
    benchmark_smooth_l1()
    profile_memory_usage()