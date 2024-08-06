import os

import torch
import triton
from utils import _test_memory, get_current_file_directory

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.linear_fused_cross_entropy import (
    LigerLinearFusedCrossEntropyLoss,
)


class TorchLinearFusedCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        logits = self.lin(x)
        logits = logits.float()
        return self.ce_loss(logits, y)


class LinearFusedLigerCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with liger cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        self.liger_ce_loss = LigerCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        logits = self.lin(x)
        logits = logits.float()
        return self.liger_ce_loss(logits, y)


#############################################################################
# Test the memory consumption of the linear fused cross entropy loss
#############################################################################


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["BT"],
            x_vals=[2**i for i in range(12, 16)],
            xlabel="B x T",
            line_arg="provider",
            line_vals=["liger", "huggingface", "huggingface_ligerce"],
            line_names=["Liger", "Hugging Face", "Hugging Face+LigerCE"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
                ("red", "solid"),
            ],
            ylabel="GPU memory usage (MB)",
            plot_name="linear-fused-cross-entropy-memory-benchmark",
            args={"H": 4096, "V": 128256, "dtype": torch.bfloat16},
        )
    ]
)
def bench_memory_cross_entropy(BT, H, V, provider, dtype, device="cuda"):
    print(
        f"Running benchmark with BT={BT}, H={H}, V={V}, dtype={dtype} provider={provider}"
    )
    torch_lf_ce = TorchLinearFusedCE(H=H, V=V, dtype=dtype).to(device)
    liger_lf_ce = (
        LigerLinearFusedCrossEntropyLoss(in_features=H, num_classes=V)
        .type(dtype)
        .to(device)
    )
    lf_liger_ce = LinearFusedLigerCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_lf_ce(_input, target)
        elif provider == "huggingface":
            return torch_lf_ce(_input, target)
        elif provider == "huggingface_ligerce":
            return lf_liger_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem = _test_memory(full, _iter=10)
    return mem / 2**20


def benchmark_memory_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "linear_fused_cross_entropy_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_memory_cross_entropy.run(save_path=output_dir, print_data=True)


#############################################################################
# Test the speed of the linear fused cross entropy loss
#############################################################################


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["BT"],
            x_vals=[2**i for i in range(12, 16)],
            xlabel="B x T",
            line_arg="provider",
            line_vals=["huggingface", "huggingface_ligerce"],
            line_names=["Hugging Face", "Hugging Face+LigerCE"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
                ("red", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="linear-fused-cross-entropy-fwd-speed-benchmark",
            args={"H": 4096, "V": 128256, "mode": "forward", "dtype": torch.bfloat16},
        ),
        triton.testing.Benchmark(
            x_names=["BT"],
            x_vals=[2**i for i in range(12, 16)],
            xlabel="B x T",
            line_arg="provider",
            line_vals=["liger", "huggingface", "huggingface_ligerce"],
            line_names=["Liger", "Hugging Face", "Hugging Face+LigerCE"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
                ("red", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="linear-fused-cross-entropy-full-speed-benchmark",
            args={"H": 4096, "V": 128256, "mode": "full", "dtype": torch.bfloat16},
        ),
    ]
)
def bench_speed_cross_entropy(BT, H, V, provider, mode, dtype, device="cuda"):
    print(
        f"Running benchmark with BT={BT}, H={H}, V={V}, provider={provider} mode={mode} dtype={dtype}"
    )
    torch_lf_ce = TorchLinearFusedCE(H=H, V=V, dtype=dtype).to(device)
    liger_lf_ce = (
        LigerLinearFusedCrossEntropyLoss(in_features=H, num_classes=V)
        .type(dtype)
        .to(device)
    )
    lf_liger_ce = LinearFusedLigerCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_lf_ce(_input, target)
        elif provider == "huggingface":
            return torch_lf_ce(_input, target)
        elif provider == "huggingface_ligerce":
            return lf_liger_ce(_input, target)

    quantiles = [0.5, 0.2, 0.8]

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
    elif mode == "backward":
        y = fwd()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[_input],
            rep=100,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles, rep=100)
    return ms, min_ms, max_ms


def benchmark_speed_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "linear_fused_cross_entropy_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_speed_cross_entropy.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_memory_cross_entropy_wrapper()
    benchmark_speed_cross_entropy_wrapper()
