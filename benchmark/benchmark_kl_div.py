import os
import triton
import torch
import torch.nn as nn
from liger_kernel.transformers.kl_div import LigerKLDIVLoss

from utils import (
    QUANTILES,
    _print_memory_banner,
    _print_speed_banner,
    _test_memory,
    get_current_file_directory,
)

S, E = 12, 18

@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(S, E)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "torch"],
            line_names=[
                "Liger",
                "Torch",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="time (ms)",
			plot_name="kl-div-fwd-speed-benchmark",
            args={"B": 8, "T": 512, "mode": "forward", "dtype": torch.bfloat16},
        ),
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(S, E)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "torch"],
            line_names=["Liger", "Torch"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="time (ms)",
            plot_name="kl-div-full-speed-benchmark",
            args={"B": 8, "T": 512, "mode": "full", "dtype": torch.bfloat16},
        ),
    ]
)
def bench_speed_kldiv(B, T, V, provider, mode, dtype, device="cuda"):
	torch_kl_div = nn.KLDivLoss(reduction="batchmean")
	liger_kl_div = LigerKLDIVLoss(reduction="batchmean")

	_input = torch.randn(B * T, V, requires_grad=True, device="cuda").log_softmax(dim=-1)
	target = torch.randn(B * T, V, device="cuda").softmax(dim=-1)

	def fwd():
		if provider == "liger":
			return liger_kl_div(_input, target)
		else:
			return torch_kl_div(_input, target)

	if mode == "forward":
		ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100)
	elif mode == "backward":
		y = fwd()

		ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[_input],
            rep=100,
        )
	elif mode == "full":

		def full():
			y = fwd()
			y.backward(retain_graph=True)

		ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100)
	return ms, min_ms, max_ms

def benchmark_speed_swiglu_wrapper():
    _print_speed_banner()

    curr_dir = get_current_file_directory()
    dir_name = "kl_div_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_kldiv.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_swiglu_wrapper()
